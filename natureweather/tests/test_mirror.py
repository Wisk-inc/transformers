# Copyright 2026 Nathan. Apache-2.0.
"""
The Hub mirror: what a hosted notebook's wiped disk does not take with it.

The Hub itself is faked, so these run offline and fast; what they pin is the contract -- pull only
what is missing and matches, push only what is not already there, never let a failure stop a run,
and keep the live watch alive until the run ends.
"""

from __future__ import annotations

import json
import textwrap
import time
import types
from pathlib import Path

import pytest
import torch
from naturev1 import HubMirror, follow, launch, stop
from naturev1.checkpoint import CheckpointManager, TrainingState


class FakeHub:
    """A dict standing in for a repository: path -> bytes."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.uploads: list[str] = []

    def install(self, monkeypatch):
        import huggingface_hub

        hub = self

        class Api:
            def __init__(self, token=None):
                pass

            def get_paths_info(self, repo_id, paths, repo_type=None):
                return [types.SimpleNamespace(path=p, size=len(hub.files[p]), lfs=None) for p in paths if p in hub.files]

            def create_repo(self, *args, **kwargs):
                pass

            def upload_file(self, path_or_fileobj, path_in_repo, repo_id, **kwargs):
                hub.files[path_in_repo] = Path(path_or_fileobj).read_bytes()
                hub.uploads.append(path_in_repo)

        def download(repo_id, filename, repo_type=None, local_dir=None, token=None, **kwargs):
            target = Path(local_dir) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(hub.files[filename])
            return str(target)

        monkeypatch.setattr(huggingface_hub, "HfApi", Api)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
        return self


def test_pull_only_when_missing_and_push_only_when_different(tmp_path, monkeypatch):
    hub = FakeHub().install(monkeypatch)
    mirror = HubMirror("someone/model", token="hf_fake")
    local = tmp_path / "batch.json"
    assert not mirror.pull(local, "staging/batch.json"), "nothing on the Hub yet"
    local.write_text('{"batch": 8}')
    assert mirror.push(local, "staging/batch.json", background=False)
    assert hub.uploads == ["staging/batch.json"]
    assert not mirror.push(local, "staging/batch.json", background=False), "same bytes: not pushed again"
    local.unlink()
    assert mirror.pull(local, "staging/batch.json") and json.loads(local.read_text()) == {"batch": 8}
    assert not mirror.pull(local, "staging/batch.json"), "already here"


def test_a_staging_for_other_years_is_not_pulled(tmp_path, monkeypatch):
    hub = FakeHub().install(monkeypatch)
    hub.files["staging/era5.json"] = json.dumps({"variables": ["a", "b"], "indices": [1, 2, 3]}).encode()
    hub.files["staging/era5.npy"] = b"array"
    mirror = HubMirror("someone/model", token="hf_fake")
    assert not mirror.pull_staging(tmp_path / "era5.npy", "staging/era5", ["a", "b"], [10, 11])
    assert not (tmp_path / "era5.json").exists(), "a sidecar without its array is not left behind"
    assert not mirror.pull_staging(tmp_path / "era5.npy", "staging/era5", ["a", "c"], [1, 2])
    assert mirror.pull_staging(tmp_path / "era5.npy", "staging/era5", ["a", "b"], [2, 3, 4])
    assert (tmp_path / "era5.npy").read_bytes() == b"array" and (tmp_path / "era5.json").exists()


def test_push_staging_sends_the_array_before_its_sidecar(tmp_path, monkeypatch):
    hub = FakeHub().install(monkeypatch)
    (tmp_path / "era5.npy").write_bytes(b"x" * 100)
    (tmp_path / "era5.json").write_text('{"variables": ["a"], "indices": [1]}')
    mirror = HubMirror("someone/model", token="hf_fake")
    assert mirror.push_staging(tmp_path / "era5.npy", "staging/era5")
    mirror.wait(30)
    assert hub.uploads == ["staging/era5.npy", "staging/era5.json"]


def test_without_a_token_nothing_is_pushed_and_it_is_said_once(tmp_path, monkeypatch, capsys):
    FakeHub().install(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    mirror = HubMirror("someone/model")
    (tmp_path / "f").write_text("x")
    assert not mirror.push(tmp_path / "f", "f") and not mirror.push(tmp_path / "f", "f")
    assert capsys.readouterr().out.count("no HF_TOKEN") == 1
    assert "pull-only" in mirror.describe()
    assert HubMirror(None).describe() == "mirror: off (no HF_REPO)"


def test_a_failing_hub_never_stops_the_run(tmp_path, monkeypatch, capsys):
    import huggingface_hub

    class Broken:
        def __init__(self, token=None):
            pass

        def get_paths_info(self, *a, **k):
            raise ConnectionError("no network")

    monkeypatch.setattr(huggingface_hub, "HfApi", Broken)
    mirror = HubMirror("someone/model", token="hf_fake")
    (tmp_path / "f").write_text("x")
    assert not mirror.pull(tmp_path / "g", "g")
    mirror.push(tmp_path / "f", "f", background=False)
    assert "skipped" in capsys.readouterr().out


def test_push_now_waits_for_the_upload(tmp_path, monkeypatch):
    hub = FakeHub().install(monkeypatch)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "hf_fake")
    manager = CheckpointManager(tmp_path / "stage1", repo_id="someone/model", push_every_seconds=3600.0,
                                backup_dir=str(tmp_path / "backup"))
    torch.manual_seed(0)
    manager.save(torch.nn.Linear(2, 2), None, state=TrainingState(step=5), force=True)
    assert manager.push_now(TrainingState(step=5))
    assert "checkpoints/stage1/latest.pt" in hub.files


def test_the_watch_runs_until_the_run_ends_and_proves_life(tmp_path, monkeypatch, capsys):
    import importlib

    launch_module = importlib.import_module("naturev1.launch")
    monkeypatch.setattr(launch_module, "ALIVE_EVERY", 3.0)
    monkeypatch.setattr(launch_module, "STATUS_EVERY", 4.0)
    fake = tmp_path / "fake_pipeline.py"
    fake.write_text(textwrap.dedent('''
        import time
        print("working", flush=True)
        time.sleep(12)
        print("done", flush=True)
    '''))
    home = tmp_path / "data"
    launch(directory=home, script=fake, supervise=False)
    try:
        began = time.time()
        follow(directory=home, wait=30, watch=None)
        shown = capsys.readouterr().out
        assert time.time() - began < 40
        assert "watching until the run finishes" in shown
        assert "[alive]" in shown and "session up" in shown, "quiet stretches get a proof-of-life line"
        assert "[status]" in shown
        assert "done" in shown and "the run has ended" in shown
    finally:
        stop(directory=home, wait=10)


def test_the_trainer_log_line_carries_pace_and_eta(tmp_path):
    from naturev1 import NatureConfig, NatureV1, Trainer, TrainSettings

    from ihelix import fibonacci_sphere

    model = NatureV1(NatureConfig(analysis_channels=4, hidden_size=16, num_layers=1, num_heads=2,
                                  num_kv_heads=1, head_dim=8, intermediate_size=32, latent_points=32,
                                  history_frames=2, lead_times_hours=(6,)),
                     fibonacci_sphere(32, num_neighbours=8, cluster_size=8))
    grid = fibonacci_sphere(40, num_neighbours=8, cluster_size=8)

    def batches():
        for _ in range(4):
            yield {"analysis": torch.randn(1, 2, 40, 4), "analysis_grid": grid, "output_grid": grid,
                   "calendar": torch.zeros(1, 2, 6), "field_target": torch.zeros(1, 40, 1, 7),
                   "field_mask": torch.ones(1, 40, 1, 7)}

    import io
    from contextlib import redirect_stdout

    out = io.StringIO()
    with redirect_stdout(out):
        Trainer(model, TrainSettings(max_steps=4, precision="fp32", warmup_steps=1, log_every=2,
                                     checkpoint_dir=str(tmp_path)), device="cpu").fit(batches(), epochs=1)
    text = out.getvalue()
    assert "s/step" in text and "h left" in text and "epoch 1/1" in text and "/4" in text


@pytest.mark.parametrize("name", ["HubMirror", "session_started"])
def test_the_mirror_is_exported(name):
    import naturev1

    assert hasattr(naturev1, name)
