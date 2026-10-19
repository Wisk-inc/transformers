# Copyright 2026 Nathan. Apache-2.0.
"""
The 0.25-degree stage (0.18): targets staged from a WeatherBench-2-shaped store (north to south, as the real one), the
path trained with checkpoints, stopped and resumed, scored against bilinear upsampling, and loaded back for decoding --
all on a small constructed world whose fine-scale truth is the coarse field plus a mountain the coarse grid cannot see.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from naturev1.era5 import Normalizer
from naturev1.highres import HIGHRES_FIELDS, bilinear_points, bilinear_upsample
from naturev1.model import NatureConfig, NatureV1

from ihelix import fibonacci_sphere


FIELDS = list(HIGHRES_FIELDS)
VARIABLES = FIELDS + ["surface_pressure", "land_sea_mask"]
TINY = {"analysis_channels": 12, "state_channels": 8, "hidden_size": 32, "num_layers": 2, "num_heads": 2,
        "num_kv_heads": 1, "head_dim": 16, "intermediate_size": 64, "latent_points": 96, "latent_neighbours": 12,
        "latent_cluster": 16, "landmark_dim": 16, "index_branching": 2, "index_beam_width": 2, "index_topk": 2,
        "recurrent_heads": 2, "recurrent_head_dim": 16, "history_frames": 2, "min_radius_km": 600.0,
        "max_radius_km": 2000.0, "encode_radius_km": 900.0, "lead_times_hours": (6, 12, 24)}
MOUNTAIN_GAIN = 0.5                     # normalised units per km of orography: what only the fine grid can know


def _mountain(lat, lon):
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    return 3000.0 * np.exp(-((la - 20.0) ** 2 + (((lo - 280.0 + 180) % 360) - 180) ** 2) / (2 * 15.0 ** 2))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    import xarray as xr

    directory = tmp_path_factory.mktemp("highres")
    steps, rows, cols = 40, 19, 36
    start = np.datetime64("2010-12-27T00", "s")
    times = start + np.arange(steps) * np.timedelta64(6, "h")
    rng = np.random.default_rng(0)
    values = rng.normal(0, 0.5, (steps, rows, cols, len(VARIABLES))).astype(np.float16)
    means = np.array([0.0, 0.0, 101300.0, 288.0, 1.0, 101000.0, 0.3], np.float32)
    stds = np.array([6.0, 6.0, 900.0, 12.0, 0.2, 900.0, 0.4], np.float32)
    normalizer = Normalizer(VARIABLES, means, stds, np.zeros(len(VARIABLES), np.float32))
    lat, lon = np.linspace(-90, 90, rows), np.arange(cols) * 10.0
    era5 = directory / "era5.npy"
    np.save(era5, values)
    era5.with_suffix(".json").write_text(json.dumps({
        "variables": VARIABLES, "shape": list(values.shape), "statistics": normalizer.to_dict(), "latitude": lat.tolist(),
        "longitude": lon.tolist(), "time": times.astype(np.int64).tolist()}))

    # The fine grid: 5 degrees, stored north to south like WeatherBench 2's 0.25-degree ERA5.
    lat_hr, lon_hr = np.linspace(-90, 90, 37), np.arange(72) * 5.0
    orography = _mountain(lat_hr, lon_hr)
    coarse = torch.tensor(values[..., :len(FIELDS)].astype(np.float64))
    fine = bilinear_upsample(coarse, lat, lon, lat_hr, lon_hr).numpy() + MOUNTAIN_GAIN * orography[None, :, :, None] / 1000
    raw = {name: normalizer.decode(fine[..., c], name)[:, ::-1] for c, name in enumerate(FIELDS)}
    store = xr.Dataset(
        {**{name: (("time", "latitude", "longitude"), raw[name].astype(np.float32)) for name in FIELDS},
         "geopotential_at_surface": (("latitude", "longitude"), (orography * 9.80665)[::-1].astype(np.float32)),
         "land_sea_mask": (("latitude", "longitude"), (orography > 100)[::-1].astype(np.float32))},
        coords={"time": times.astype("datetime64[ns]"), "latitude": lat_hr[::-1], "longitude": lon_hr})
    torch.manual_seed(0)
    model = NatureV1(NatureConfig(**TINY), fibonacci_sphere(96, num_neighbours=12, cluster_size=16))
    torch.save({"model": model.state_dict(), "config": model.config.to_dict()}, directory / "base.pt")
    return {"dir": directory, "era5": era5, "store": store, "orography": orography, "lat_hr": lat_hr, "lon_hr": lon_hr}


def test_point_interpolation_matches_the_gridded_one():
    lat, lon = np.linspace(-90, 90, 19), np.arange(36) * 10.0
    field = torch.randn(2, 19, 36, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    lat_hr, lon_hr = np.linspace(-90, 90, 37), np.arange(72) * 5.0
    full = bilinear_upsample(field, lat, lon, lat_hr, lon_hr).reshape(2, -1, 3)
    la, lo = np.meshgrid(lat_hr, lon_hr, indexing="ij")
    idx = np.array([0, 5, 71, 72, 1000, 2663])
    points = bilinear_points(field, lat, lon, la.ravel()[idx], lo.ravel()[idx])
    assert torch.allclose(points, full[:, idx], atol=1e-12)
    north_first = bilinear_points(field.flip(1), lat[::-1].copy(), lon, la.ravel()[idx], lo.ravel()[idx])
    assert torch.allclose(north_first, points, atol=1e-5), "either latitude order (the pole row is clipped by 1e-6)"


def _run(world, tmp, *extra, steps=3, capsys=None):
    from naturev1.highres import main

    main(["--checkpoint", str(world["dir"] / "base.pt"), "--era5", str(world["era5"]), "--data", str(tmp / "hr"),
          "--output", str(tmp / "runs"), "--train-years", "2010", "--val-years", "2011", "--every-hours", "6",
          "--val-every-hours", "6", "--hours", "1", "--points", "600", "--workers", "0", "--width", "32",
          "--max-steps", str(steps), "--checkpoint-minutes", "0", "--eval-minutes", "1000", "--stage-workers", "2",
          *extra])


def test_the_stage_stages_trains_stops_resumes_and_scores(world, tmp_path, monkeypatch, capsys):
    from naturev1 import highres

    monkeypatch.setattr(highres, "open_highres_store", lambda store=None: world["store"])
    _run(world, tmp_path, steps=3)
    out = capsys.readouterr().out
    targets = np.load(tmp_path / "hr" / "highres_targets.npy", mmap_mode="r")
    assert targets.shape == (40, 37, 72, len(FIELDS)) and targets.dtype == np.float16
    static = np.load(tmp_path / "hr" / "highres_static.npz")
    assert static["latitude"][0] == -90.0 and np.allclose(static["orography_m"], world["orography"], atol=0.5), \
        "the north-to-south store is turned south to north"
    latest = torch.load(tmp_path / "runs" / "highres" / "latest.pt", map_location="cpu", weights_only=False)
    assert latest["state"]["step"] == 3
    assert {k.split(".")[0] for k in latest["model"]} == {"decoder", "decode_norm", "head"}, "the path alone, not the backbone"
    assert "[highres] final held-out" in out and "[highres] finished:" in out

    # Stopped after 3 steps; started again (data already on disk) it continues from step 3 instead of starting over.
    _run(world, tmp_path, "--train-only", steps=6)
    out = capsys.readouterr().out
    assert "resumed at step 3" in out
    final = torch.load(tmp_path / "runs" / "highres" / "highres.pt", map_location="cpu", weights_only=False)
    assert final["steps"] == 6 and final["fields"] == FIELDS
    report = final["report"]
    assert set(report["fields"]) == set(FIELDS) and report["times"] == 20
    assert 0 <= report["better_fields"] <= len(FIELDS)

    # The backbone is untouched; the path decodes the whole fine grid.
    from naturev1.highres import decode_highres, load_backbone, load_highres

    backbone = load_backbone(str(world["dir"] / "base.pt")).eval()
    original = torch.load(world["dir"] / "base.pt", map_location="cpu", weights_only=False)["model"]
    assert all(torch.equal(v, original[k]) for k, v in backbone.state_dict().items() if k.startswith("decoder."))
    path = load_highres(str(tmp_path / "runs" / "highres" / "highres.pt"), backbone)
    current = torch.randn(1, 96, 32)
    mean, log_var = decode_highres(backbone, path, current, torch.zeros(1, 19, 36, len(FIELDS)), rows_per_chunk=10)
    assert mean.shape == (1, 37, 72, len(FIELDS)) and torch.isfinite(mean).all()


def test_the_path_learns_what_bilinear_upsampling_cannot(world, tmp_path, monkeypatch):
    """The fine truth is the coarse field plus a mountain: the head reads orography, bilinear upsampling cannot."""
    from naturev1 import highres
    from naturev1.highres import HighResDecoder, HighResPairs, HighResStep, load_backbone, static_features
    from naturev1.sattrain import staged_grid

    monkeypatch.setattr(highres, "open_highres_store", lambda store=None: world["store"])
    meta = json.loads(world["era5"].with_suffix(".json").read_text())
    times = highres.pair_times(meta, [2010], 6)
    targets, static_file = highres.stage_highres(tmp_path, times, meta["statistics"], workers=2)
    z = np.load(static_file)
    backbone = load_backbone(str(world["dir"] / "base.pt")).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    torch.manual_seed(0)
    path = HighResDecoder(backbone, len(FIELDS), 32, z["latitude"], z["longitude"],
                          static_features(z["orography_m"], z["land"], z["latitude"]))
    data = HighResPairs(world["era5"], targets, 12, 2, times)
    step = HighResStep(backbone, path, backbone._aligned(staged_grid(meta)), data.coarse_latitude, data.coarse_longitude)
    untrained = data[1]
    _, parts = step(untrained["analysis"][None], untrained["calendar"][None], untrained["coarse"][None],
                    untrained["target"][None], torch.arange(37 * 72))
    assert float(parts["mse"]) == pytest.approx(float(parts["mse_bilinear"]), rel=1e-5), "starts as bilinear upsampling"
    assert float(parts["mse_bilinear"]) > 0.01, "the mountain is real"
    optimiser = torch.optim.Adam(path.parameters(), lr=3e-3)
    for k in range(300):
        item = data[k % len(data)]
        loss, parts = step(item["analysis"][None], item["calendar"][None], item["coarse"][None], item["target"][None],
                           torch.arange(37 * 72))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    _, parts = step(untrained["analysis"][None], untrained["calendar"][None], untrained["coarse"][None],
                    untrained["target"][None], torch.arange(37 * 72))
    assert float(parts["mse"]) < 0.5 * float(parts["mse_bilinear"]), (float(parts["mse"]), float(parts["mse_bilinear"]))


def test_the_plan_fills_a_b300_and_shrinks_for_small_cards():
    from naturev1.hardware import describe_gpu_plan, gpu_plan

    b300, h100, small = (gpu_plan(288, 1, "NVIDIA B300 SXM6 AC", 64), gpu_plan(80, 1, "NVIDIA H100 80GB HBM3", 64),
                         gpu_plan(24, 1, "NVIDIA GeForce RTX 4090", 64))
    assert b300["tier"] == "B300" and b300["highres_points"] == 721 * 1440, "a B300 decodes the whole globe every step"
    assert b300["distill_batch"] * 30.0 <= 0.9 * 288 and b300["storm_batch"] * 16.0 <= 0.9 * 288, "aims at 90%, not past it"
    assert b300["distill_batch"] >= 4 * h100["distill_batch"] and b300["storm_batch"] >= 4 * h100["storm_batch"]
    assert b300["swarm_particles"] > h100["swarm_particles"] and b300["satnet_batch"] % 8 == 0
    assert b300["storm_lr"] > h100["storm_lr"], "a bigger batch takes a (square-root) bigger step"
    assert small["distill_batch"] == 1 and small["distill_horizon"] == 4 and small["storm_batch"] == 1
    assert h100["distill_batch"] == 2 and h100["storm_batch"] == 4, "the recorded defaults on an 80 GB card"
    assert "B300" in describe_gpu_plan(b300)


def test_a_stop_request_saves_and_exits_and_the_next_start_resumes(world, tmp_path, monkeypatch, capsys):
    """The runbook's stop: the job sees its stop file, saves a checkpoint, exits without the end-of-run scoring (so it
    is never mistaken for a finished run), and the next start continues from that checkpoint."""
    from naturev1 import highres

    monkeypatch.setattr(highres, "open_highres_store", lambda store=None: world["store"])
    stop = tmp_path / "highres.stop"
    stop.write_text("stop")
    monkeypatch.setenv("NATUREV1_STOP_FILE", str(stop))
    with pytest.raises(SystemExit) as stopped:
        _run(world, tmp_path, steps=5)
    assert stopped.value.code == 3
    out = capsys.readouterr().out
    assert "stopped on request at step 1" in out and "final held-out" not in out
    assert torch.load(tmp_path / "runs" / "highres" / "latest.pt", weights_only=False)["state"]["step"] == 1
    assert not (tmp_path / "runs" / "highres" / "highres.pt").exists()
    stop.unlink()
    _run(world, tmp_path, "--train-only", steps=3)
    out = capsys.readouterr().out
    assert "resumed at step 1" in out and "[highres] finished:" in out


def test_a_small_disk_keeps_only_the_newest_weight_snapshots_here(world, tmp_path, monkeypatch):
    from naturev1 import highres

    monkeypatch.setattr(highres, "open_highres_store", lambda store=None: world["store"])
    monkeypatch.setenv("NATUREV1_KEEP_HISTORY", "1")
    _run(world, tmp_path, steps=3)
    snapshots = sorted((tmp_path / "runs" / "highres" / "history").glob("step_*.safetensors"))
    assert [p.name for p in snapshots] == ["step_00000003.safetensors"]
    assert (tmp_path / "runs" / "highres" / "latest.pt").exists()
