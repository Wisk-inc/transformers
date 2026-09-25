# Copyright 2026 Nathan. Apache-2.0.
"""
A multi-day run meets every failure eventually. These pin the ones that were found the hard way.

* Re-running re-downloaded a finished 45 GB staging: the plan was measured against free disk, which the
  staging itself had used up, so the second run asked for fewer years and wiped the first.
* Changing the number of years threw away everything already on disk.
* One allocation too many near a full card ended the run.
* Re-running a finished stage one trained its last epoch again.
* A Hub mirror without a token retried, and complained, every minute.
"""

from __future__ import annotations

import json
import os
import types

import numpy as np
import pytest
import torch
from naturev1 import CachedERA5, NatureConfig, NatureV1, Trainer, TrainSettings, era5_loader
from naturev1 import era5 as era5_module
from naturev1.checkpoint import CheckpointManager
from naturev1.era5 import Normalizer, materialise
from naturev1.fallback import resolve_staging

from ihelix import fibonacci_sphere


xr = pytest.importorskip("xarray")
LAT, LON = np.linspace(-90, 90, 7), np.arange(12) * 30.0


def _store(steps=48):
    times = np.array([np.datetime64("2010-01-01T00") + np.timedelta64(6 * i, "h") for i in range(steps)])
    values = np.arange(steps, dtype=np.float32)[:, None, None] + np.zeros((steps, len(LON), len(LAT)), np.float32)
    return xr.Dataset({"2m_temperature": (("time", "longitude", "latitude"), 280 + values)},
                      coords={"time": times, "longitude": LON, "latitude": LAT})


NORMALIZER = Normalizer(["2m_temperature"], np.array([280.0], np.float32), np.array([1.0], np.float32),
                        np.zeros(1, np.float32))


def _stage(store, indices, path, **kwargs):
    return materialise(store, np.asarray(indices), path, variables=("2m_temperature",), normalizer=NORMALIZER,
                       chunk=4, workers=2, progress=False, **kwargs)


def _staged_steps(path):
    values = np.load(path)
    return [int(round(float(v))) for v in values[:, 0, 0, 0]]


@pytest.fixture()
def reads(monkeypatch):
    """Every step fetched from the network, in order of fetching."""
    seen = []
    original = era5_module.read_block

    def counting(dataset, variables, selector, levels=None):
        seen.extend(int(i) for i in np.atleast_1d(selector))
        return original(dataset, variables, selector, levels)

    monkeypatch.setattr(era5_module, "read_block", counting)
    return seen


# ------------------------------------------------------------------------------------------- staging --

def test_the_plan_does_not_shrink_because_of_its_own_staging(tmp_path, monkeypatch):
    """The reported bug, exactly: 6 years planned, staged, and then 3 planned on the re-run."""
    staged = tmp_path / "era5.npy"
    per_year = 29040 * 89 * 2 * 1460
    free = {"bytes": int(80e9)}
    monkeypatch.setattr(os, "statvfs", lambda p: types.SimpleNamespace(f_bavail=free["bytes"], f_frsize=1))

    first, _, _ = resolve_staging(6, 89, 29040, str(tmp_path), staged=(staged,))
    staged.write_bytes(b"")
    os.truncate(staged, 6 * per_year)                        # the staging now occupies the disk
    free["bytes"] -= 6 * per_year
    again, _, _ = resolve_staging(6, 89, 29040, str(tmp_path), staged=(staged,))
    blind, _, _ = resolve_staging(6, 89, 29040, str(tmp_path))
    assert first == again == 6
    assert blind < 6, "without counting the staged file, the plan shrinks -- which caused the re-download"


def test_a_finished_staging_is_never_fetched_twice(tmp_path, reads):
    store, path = _store(), tmp_path / "era5.npy"
    _stage(store, range(8, 24), path)
    fetched = len(reads)
    stamp = path.stat().st_mtime_ns
    _stage(store, range(8, 24), path)
    assert len(reads) == fetched and path.stat().st_mtime_ns == stamp


def test_asking_for_fewer_years_keeps_the_staging_and_trains_on_what_was_asked(tmp_path, reads):
    store, path = _store(), tmp_path / "era5.npy"
    _stage(store, range(0, 32), path)
    fetched, stamp = len(reads), path.stat().st_mtime_ns
    _stage(store, range(16, 32), path)                       # fewer years
    assert len(reads) == fetched and path.stat().st_mtime_ns == stamp, "nothing re-downloaded or rewritten"

    subset = CachedERA5(path, history=2, lead_steps=1, augment=False, restrict_to=np.arange(16, 32))
    starts = [json.loads(path.with_suffix(".json").read_text())["indices"][i] for i in subset.indices]
    assert starts and min(starts) >= 16, "the run gets the years it asked for, not all that are staged"


def test_asking_for_more_years_downloads_only_the_new_ones(tmp_path, reads):
    store, path = _store(), tmp_path / "era5.npy"
    _stage(store, range(16, 32), path)
    reads.clear()
    _stage(store, range(0, 32), path)                        # more years
    assert sorted(reads) == list(range(0, 16)), "only the steps not already on disk come from the network"
    assert _staged_steps(path) == list(range(0, 32)), "and every step lands in its right place"
    assert not list(tmp_path.glob("*.previous*")), "the old staging is cleaned up once the new one is complete"


def test_growth_without_room_keeps_what_is_staged(tmp_path, reads, monkeypatch):
    store, path = _store(), tmp_path / "era5.npy"
    _stage(store, range(16, 32), path)
    reads.clear()
    monkeypatch.setattr(era5_module, "_free_bytes", lambda directory: 10)
    _stage(store, range(0, 32), path)
    assert reads == [] and _staged_steps(path) == list(range(16, 32)), "nothing deleted for lack of space"


def test_an_interrupted_download_resumes_only_into_the_same_steps(tmp_path, monkeypatch):
    store, path = _store(), tmp_path / "era5.npy"
    original = era5_module.read_block
    calls = {"n": 0}

    def dies(dataset, variables, selector, levels=None):
        calls["n"] += 1
        if calls["n"] > 1:
            raise KeyboardInterrupt                           # the cell is stopped part-way
        return original(dataset, variables, selector, levels)

    monkeypatch.setattr(era5_module, "read_block", dies)
    with pytest.raises(KeyboardInterrupt):
        materialise(store, np.arange(0, 16), path, variables=("2m_temperature",), normalizer=NORMALIZER,
                    chunk=4, workers=1, progress=False)
    monkeypatch.setattr(era5_module, "read_block", original)
    _stage(store, range(16, 32), path)                       # same length, different years
    assert _staged_steps(path) == list(range(16, 32)), "no step from the other request survives"


# ------------------------------------------------------------------------------------- training --

def _model():
    config = NatureConfig(analysis_channels=4, hidden_size=16, num_layers=1, num_heads=2, num_kv_heads=1,
                          head_dim=8, intermediate_size=32, latent_points=32, history_frames=2,
                          lead_times_hours=(6,))
    return NatureV1(config, fibonacci_sphere(32, num_neighbours=8, cluster_size=8))


def _batches(count=6, size=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    grid = fibonacci_sphere(40, num_neighbours=8, cluster_size=8)
    return [{"analysis": torch.randn(size, 2, 40, 4, generator=generator), "analysis_grid": grid,
             "output_grid": grid, "calendar": torch.zeros(size, 2, 6),
             "field_target": torch.randn(size, 40, 1, 7, generator=generator),
             "field_mask": torch.ones(size, 40, 1, 7)} for _ in range(count)]


def test_out_of_memory_halves_the_pieces_and_keeps_the_gradient(tmp_path, monkeypatch):
    torch.manual_seed(0)
    model = _model()
    trainer = Trainer(model, TrainSettings(max_steps=10, checkpoint_dir=str(tmp_path / "a"), precision="fp32",
                                           warmup_steps=1, ema_decay=0.0), device="cpu")
    batch = _batches(1)[0]
    trainer._forward_backward(batch)
    whole = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)

    original = model.forward

    def small_card(*args, **kwargs):
        if kwargs["analysis"].shape[0] > 2:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", small_card)
    trainer._forward_backward(batch)
    assert trainer.micro_batch == 2
    for name, grad in whole.items():
        assert torch.allclose(model.get_parameter(name).grad, grad, atol=1e-5), name


def test_rollout_recovers_from_out_of_memory(staged_rollout, monkeypatch):
    model, stepper, batch = staged_rollout
    from naturev1.state import _rollout_step
    from naturev1.train import batch_size, free_after_oom, slice_batch

    original = model.forward

    def small_card(*args, **kwargs):
        if kwargs["analysis"].shape[0] > 1:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", small_card)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    loss, parts, micro = _rollout_step(model, batch, stepper, 2, None, None, 1, "fp32", None, optimizer, None,
                                       batch_size, slice_batch, free_after_oom)
    assert micro == 1 and torch.isfinite(loss) and parts["state_horizon"] == 2.0


@pytest.fixture()
def staged_rollout(tmp_path):
    from naturev1 import StateStepper
    from naturev1.corpora import latlon_to_coords
    from naturev1.era5 import equiangular_weights

    from ihelix import FieldGrid, Geometry

    lat, lon = np.linspace(-90, 90, 7), np.arange(12) * 30.0
    variables = ["2m_temperature", "surface_pressure", "temperature@500"]
    values = np.random.default_rng(0).normal(size=(20, 7, 12, 3)).astype(np.float16)
    path = tmp_path / "roll.npy"
    np.save(path, values)
    path.with_suffix(".json").write_text(json.dumps({
        "variables": variables, "time": (1.6e9 + np.arange(20) * 21600).astype(np.int64).tolist(),
        "statistics": Normalizer(variables, np.zeros(3, np.float32), np.ones(3, np.float32),
                                 np.zeros(3, np.float32)).to_dict()}))
    grid = FieldGrid.source(latlon_to_coords(lat, lon), Geometry.globe(),
                            weights=torch.tensor(equiangular_weights(lat, 12)))
    dataset = CachedERA5(path, history=2, lead_steps=1, channels=4, augment=False, state_steps=2)
    stepper = StateStepper.build(dataset, 4, grid, samples=4)
    config = NatureConfig(analysis_channels=4, hidden_size=16, num_layers=1, num_heads=2, num_kv_heads=1,
                          head_dim=8, intermediate_size=32, latent_points=32, history_frames=2,
                          lead_times_hours=(6,), state_channels=stepper.roles.num_prognostic)
    model = NatureV1(config, fibonacci_sphere(32, num_neighbours=8, cluster_size=8))
    batch = next(iter(era5_loader(dataset, batch_size=2, num_workers=0, analysis_grid=grid, output_grid=grid)))
    return model, stepper, batch


def test_rerunning_a_finished_stage_trains_nothing(tmp_path):
    """The last epoch was never marked done, so every re-run trained it again -- about two hours."""
    settings = TrainSettings(max_steps=1000, checkpoint_dir=str(tmp_path), precision="fp32", warmup_steps=1,
                             ema_decay=0.0, log_every=1000)
    first = Trainer(_model(), settings, device="cpu")
    state = first.fit(_batches(3), epochs=2)
    assert state.step == 6 and state.epoch == 2

    again = Trainer(_model(), settings, device="cpu")
    again.resume()
    assert again.fit(_batches(3), epochs=2).step == 6, "a finished stage must not train another epoch"


def test_a_hub_mirror_without_a_token_waits_and_says_so_once(tmp_path, monkeypatch, capsys):
    import huggingface_hub

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    manager = CheckpointManager(tmp_path / "stage1", repo_id="someone/some-model", push_every_seconds=0.0)
    model = _model()
    for _ in range(3):
        manager.save(model, None, force=True)
    printed = capsys.readouterr().out
    assert printed.count("no Hugging Face token") == 1
    assert manager._upload is None, "nothing is uploaded, and nothing blocks training"
