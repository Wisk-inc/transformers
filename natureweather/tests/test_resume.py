# Copyright 2026 Nathan. Apache-2.0.
"""
A run that stops must be able to continue -- from whatever state the stop left on disk.

A kill can land in the middle of any write. Each test here leaves behind what such a kill leaves -- a
cut-off file, a stray temporary, a deleted run folder, a full disk -- and checks that the next start
continues instead of crashing, because a supervised run that crashes on every start gives up.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from naturev1 import progress
from naturev1.checkpoint import CheckpointManager, TrainingState
from naturev1.era5 import Normalizer
from naturev1.safeio import read_json, remove_stale_temporaries


def _net():
    torch.manual_seed(0)
    return torch.nn.Linear(4, 3)


def _manager(tmp_path, **kwargs):
    return CheckpointManager(tmp_path / "run" / "stage1", backup_dir=str(tmp_path / "backup"), **kwargs)


def test_every_snapshot_has_a_backup_copy_that_restores_a_lost_run(tmp_path):
    net = _net()
    manager = _manager(tmp_path)
    manager.save(net, None, state=TrainingState(step=7, epoch=1), force=True)
    assert (tmp_path / "backup" / "stage1" / "latest.pt").exists()

    # The run folder is gone -- deleted to free space, a wiped disk -- but the backup is not.
    import shutil

    shutil.rmtree(tmp_path / "run")
    assert read_json(tmp_path / "backup" / "stage1" / "state.json")["step"] == 7
    fresh = torch.nn.Linear(4, 3)
    state = _manager(tmp_path).load(fresh)
    assert state is not None and state.step == 7
    assert torch.equal(fresh.weight, net.weight)


def test_a_damaged_latest_falls_back_to_a_snapshot_then_the_backup(tmp_path):
    net = _net()
    manager = _manager(tmp_path)
    manager.save(net, None, state=TrainingState(step=3), force=True)
    directory = tmp_path / "run" / "stage1"
    (directory / "latest.pt").write_bytes(b"cut off mid-write")
    for snapshot in directory.glob("step_*.pt"):
        snapshot.write_bytes(b"also damaged")
    state = _manager(tmp_path).load(torch.nn.Linear(4, 3))
    assert state is not None and state.step == 3, "the backup copy is the last resort, and it works"


def test_half_written_saves_from_killed_runs_are_cleaned_up(tmp_path):
    directory = tmp_path / "run" / "stage1"
    directory.mkdir(parents=True)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    stale = directory / f".latest.pt.{dead.pid}.tmp"
    stale.write_bytes(b"x" * 1000)
    live = directory / f".latest.pt.{os.getpid()}.tmp"
    live.write_bytes(b"in use")
    _manager(tmp_path)
    assert not stale.exists(), "a dead process's temporary is removed"
    assert live.exists(), "a live process's temporary is left alone"
    assert remove_stale_temporaries(directory) == 0


def test_a_full_disk_does_not_stop_training(tmp_path, monkeypatch, capsys):
    net = _net()
    manager = _manager(tmp_path)
    manager.save(net, None, state=TrainingState(step=1), force=True)

    def full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("naturev1.checkpoint.atomic_torch_save", full)
    assert manager.save(net, None, state=TrainingState(step=2), force=True) is None
    assert "could not save" in capsys.readouterr().out
    state = _manager(tmp_path).load(torch.nn.Linear(4, 3))
    assert state.step == 1, "the last good checkpoint is intact"


def test_damaged_statistics_are_recomputed_not_a_crash(tmp_path):
    xr = pytest.importorskip("xarray")
    times = np.array([np.datetime64("2010-01-01T00") + np.timedelta64(6 * i, "h") for i in range(16)])
    store = xr.Dataset({"2m_temperature": (("time", "longitude", "latitude"),
                                           np.random.default_rng(0).normal(280, 5, (16, 6, 5)).astype(np.float32))},
                       coords={"time": times, "longitude": np.arange(6) * 60.0, "latitude": np.linspace(-90, 90, 5)})
    cache = tmp_path / "stats.json"
    cache.write_text('{"variables": ["2m_temperature"], "mean": [28')        # cut off mid-write
    normalizer = Normalizer.fit(store, ["2m_temperature"], cache=cache)
    assert np.isfinite(normalizer.mean).all()
    assert read_json(cache)["variables"] == ["2m_temperature"], "rewritten whole"


def test_a_damaged_storm_feature_bank_is_rebuilt(tmp_path):
    from naturev1 import ERA5Window, pair_tracks_with_reanalysis, storm_feature_bank
    from test_fixes_080 import WIDTH, _grid, _model, _store, _storm

    store = _store()
    variables = ("2m_temperature", "surface_pressure", "toa_incident_solar_radiation_6hr", "land_sea_mask",
                 "temperature", "geopotential")
    tracks = [_storm("AL012005", 24, 12, 15.0)]
    starts, targets, _ = pair_tracks_with_reanalysis(tracks, store.time.values.astype("datetime64[s]")
                                                     .astype(np.int64), (1, 2), history=2)
    base = ERA5Window(store, indices=starts[:1], history=2, lead_steps=1, variables=variables, levels=[500],
                      channels=WIDTH, augment=False)
    model, grid = _model(), _grid()
    model.config.lead_times_hours = (6, 12)
    cache = tmp_path / "bank.pt"
    cache.write_bytes(b"cut off")
    (tmp_path / "bank.pt.partial").write_bytes(b"cut off too")
    bank = storm_feature_bank(model, store, base, starts, targets, grid, cache=str(cache), precision="fp32",
                              progress=False)
    assert bank["features"].shape[0] == len(starts)
    assert torch.load(cache, weights_only=False)["features"].shape[0] == len(starts), "rewritten whole"


def test_progress_says_where_each_stage_continues_from(tmp_path, monkeypatch, capsys):
    home = tmp_path / "data"
    monkeypatch.setenv("NATUREV1_BACKUP_DIR", str(tmp_path / "backup"))
    manager = CheckpointManager(home / "naturev1_ckpt" / "stage1")
    manager.save(_net(), None, state=TrainingState(step=1234, epoch=2), force=True)
    progress(home)
    shown = capsys.readouterr().out
    assert "stage one, pretraining" in shown and "step 1,234, epoch 2" in shown and "backup copy" in shown
    assert "stage 1b, rollout" in shown and "not started" in shown


def test_a_hub_push_is_one_line_in_the_log_not_a_progress_bar(tmp_path, monkeypatch, capsys):
    """The upload library's progress bars used to fill the training log; one line says what happened."""
    import huggingface_hub

    calls = {}

    class FakeApi:
        def create_repo(self, repo, **kwargs):
            calls["repo"] = repo

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, **kwargs):
            calls["path"] = path_in_repo

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_fake")
    manager = CheckpointManager(tmp_path / "stage1", repo_id="someone/model", push_every_seconds=0.0,
                                backup_dir=str(tmp_path / "backup"))
    manager.save(_net(), None, state=TrainingState(step=42), force=True)
    assert manager._upload is not None
    manager._upload.join(timeout=30)
    shown = capsys.readouterr().out
    assert "pushed stage1 step 42 to someone/model" in shown and "GB in" in shown
    assert calls["path"] == "checkpoints/stage1/latest.pt"
    assert "Processing Files" not in shown and "%|" not in shown
