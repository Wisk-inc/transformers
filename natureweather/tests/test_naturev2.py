# Copyright 2026 Nathan. Apache-2.0.
"""
NatureV2 (0.17): ocean heat content and storm size, the infrared structure of the storm patches, and the in-model
satellite encoder fed in the storm stage. Off by default: an earlier configuration builds exactly the earlier model;
switched on, every new input starts as an exact no-op, learns, reaches RI 2.0, and travels through the banks.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from ihelix import fibonacci_sphere

from naturev1.model import OPTIONAL_STATE, TRACKED_HEADS, NatureConfig, NatureV1
from naturev1.stormsat import PATCH_DEGREES, STORM_IR_FEATURES, encode_bt, storm_ir_features
from naturev1.thermo import STORM_OCEAN_FEATURES, storm_ocean

LEADS = (6, 12, 24)
TINY = dict(analysis_channels=12, state_channels=8, hidden_size=32, num_layers=3, num_heads=2, num_kv_heads=1,
            head_dim=16, intermediate_size=64, latent_points=96, latent_neighbours=12, latent_cluster=16,
            landmark_dim=16, index_branching=2, index_beam_width=2, index_topk=2, recurrent_heads=2,
            recurrent_head_dim=16, history_frames=2, min_radius_km=600.0, max_radius_km=2000.0,
            encode_radius_km=900.0, lead_times_hours=LEADS)


def _model(**switches) -> NatureV1:
    torch.manual_seed(0)
    return NatureV1(NatureConfig(**{**TINY, **switches}), fibonacci_sphere(96, num_neighbours=12, cluster_size=16))


def _hurricane(pixels: int = 256, eye: bool = True) -> np.ndarray:
    """A mature storm in band 13: warm eye, cold dense overcast to 200 km, warmer beyond (kelvin)."""
    offsets = (np.arange(pixels) - (pixels - 1) / 2.0) * PATCH_DEGREES * 111.2
    radius = np.sqrt(offsets[:, None] ** 2 + offsets[None, :] ** 2)
    return np.where(radius < 20, 285.0 if eye else 195.0, np.where(radius < 200, 195.0, 260.0))


# ----------------------------------------------------------------------------------------- features --

def test_ocean_features_read_the_atlas_at_the_storm_and_its_size():
    centres = torch.tensor([[17.0, -78.0], [40.0, -40.0], [float("nan"), float("nan")]])
    env = torch.zeros(3, 8)
    env[:, 7] = 1.0
    env[:, 6] = (30.5 - 26.5) / 3.0                                       # 30.5 C under each core
    radii = torch.full((3, 3, 4), float("nan"))
    radii[0, 0] = torch.tensor([150.0, 120.0, 60.0, 90.0])
    when = int(np.datetime64("2025-10-27T00", "s").astype(np.int64))
    out = storm_ocean(centres, torch.full((3,), when), env, radii)
    assert out.shape == (3, len(STORM_OCEAN_FEATURES))
    caribbean = dict(zip(STORM_OCEAN_FEATURES, out[0].tolist()))
    assert 0.6 < caribbean["d26"] < 1.2 and 0.5 < caribbean["tchp"] < 1.2, "deep warm water south of Jamaica"
    assert caribbean["sst_anomaly"] > 0.3, "30.5 C is warmer than the late-October climatology"
    assert caribbean["gale_radius"] == pytest.approx(1.05) and caribbean["gale_observed"] == 1.0
    cold = dict(zip(STORM_OCEAN_FEATURES, out[1].tolist()))
    assert cold["d26"] == 0.0 and cold["observed"] == 1.0, "no 26 C water at 40 N in October"
    assert cold["sst_anomaly"] == 0.0, "an anomaly of ~10 K is not the sea: dropped"
    assert out[2].abs().sum() == 0.0, "no centre, nothing known"
    assert storm_ocean(centres[:1], when).shape == (1, len(STORM_OCEAN_FEATURES)), "environment and radii optional"


def test_infrared_structure_sees_the_eye_the_cold_ring_and_the_burst():
    image, earlier = _hurricane(), np.where(_hurricane() < 250, 230.0, 260.0)
    codes = torch.from_numpy(np.stack([encode_bt(image), encode_bt(earlier)]))[None]
    out = dict(zip(STORM_IR_FEATURES, storm_ir_features(codes, torch.tensor([[True, True]]), torch.tensor([0.0]))[0].tolist()))
    assert out["present"] == 1.0 and out["pc50"] == pytest.approx(1.0) and out["pc30"] == pytest.approx(1.0)
    assert out["eye_warm"] > 1.5, "a warm eye inside a cold ring"
    assert out["min_bt"] == pytest.approx((195.0 - 273.15 + 50.0) / 30.0, abs=0.02)
    assert out["d_pc50_6h"] == pytest.approx(1.0) and out["d_min_bt_6h"] < -3.0, "tops cooled 35 K in six hours"
    alone = storm_ir_features(codes, torch.tensor([[True, False]]))[0]
    assert alone[STORM_IR_FEATURES.index("d_pc50_6h")] == 0.0, "no trend without the earlier image"
    assert storm_ir_features(codes, torch.tensor([[False, False]])).abs().sum() == 0.0


# --------------------------------------------------------------------------------------------- model --

def test_switched_off_it_is_the_v1_model_and_old_checkpoints_load():
    v1, v2 = _model(ri2=True), _model(ri2=True, storm_ocean=True, storm_ir=True)
    extra = set(v2.state_dict()) - set(v1.state_dict())
    assert extra == {"storm_ocean_embed.weight", "storm_ir_embed.weight"}
    assert TRACKED_HEADS[-2:] == ("storm_ocean", "storm_ir"), "new counts only ever appended"
    assert {"storm_ocean_embed.", "storm_ir_embed."} <= set(OPTIONAL_STATE)
    # A V1 checkpoint (no new embeds, a shorter record) loads into V2; the trunk of RI 2.0 is the one shape that differs.
    state = {k: v for k, v in v1.state_dict().items() if not k.startswith("ri2_head.")}
    state["head_steps"] = state["head_steps"][:-2]
    fresh = _model(ri2=True, storm_ocean=True, storm_ir=True)
    fresh.load_state_dict({**state, **{k: v for k, v in fresh.state_dict().items() if k.startswith("ri2_head.")}})
    assert fresh.trained_heads()["storm_ocean"] == 0 and fresh.trained_heads()["storm_ir"] == 0
    for key in ("blocks.0.attn.q_proj.weight", "track_head.parameters_head.weight"):
        if key in state:
            assert torch.equal(fresh.state_dict()[key], state[key])


def test_every_new_input_is_an_exact_no_op_until_trained():
    model = _model(ri2=True, storm_ocean=True, storm_ir=True).eval()
    pooled = torch.randn(4, 32)
    centre = torch.tensor([[15.0, -50.0], [20.0, -60.0], [-15.0, 150.0], [25.0, 130.0]])
    codes = torch.from_numpy(np.stack([np.stack([encode_bt(_hurricane()), encode_bt(_hurricane(eye=False))])] * 4))
    available = torch.ones(4, 2, dtype=torch.bool)
    ocean = torch.rand(4, len(STORM_OCEAN_FEATURES))
    with torch.no_grad():
        plain = model.storm_heads(pooled, centre)
        fed = model.storm_heads(pooled, centre, storm_ocean=ocean)
    for key in ("intensity_mean", "ri2_logits", "displacement", "landfall_logit"):
        assert torch.equal(plain[key], fed[key]), key
    with torch.no_grad():
        model.storm_satellite.project.weight.zero_()                   # the encoder itself starts silent too
        a = model.storm_heads(pooled, centre, storm_satellite=(codes, available), storm_ocean=ocean)
    assert torch.equal(a["ri2_logits"], plain["ri2_logits"]) and torch.equal(a["intensity_mean"], plain["intensity_mean"])


def test_the_new_inputs_learn_and_are_counted():
    model = _model(ri2=True, storm_ocean=True, storm_ir=True)
    model.freeze_backbone()
    heads = [p for p in model.storm_head_parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(heads, lr=0.5)
    pooled = torch.randn(6, 32)
    centre = torch.tensor([[15.0, -50.0]] * 3 + [[22.0, 130.0]] * 3)
    images = [_hurricane(), _hurricane(eye=False)]
    codes = torch.from_numpy(np.stack([np.stack([encode_bt(images[i % 2]), encode_bt(images[(i + 1) % 2])])
                                       for i in range(6)]))
    available = torch.ones(6, 2, dtype=torch.bool)
    ocean = torch.rand(6, len(STORM_OCEAN_FEATURES))
    before_ocean = model.storm_ocean_embed.weight.clone()
    before_ir = model.storm_ir_embed.weight.clone()
    before_trunk = model.ri2_head.trunk[0].weight[:, -model.ri2_head.extra:].clone()
    for _ in range(3):
        out = model.storm_heads(pooled, centre, storm_satellite=(codes, available), storm_ocean=ocean)
        loss = out["intensity_mean"][..., 0].pow(2).mean() + out["ri2_logits"].pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.mark_trained({"intensity": 1.0, "ri2": 1.0}, {"storm_center": centre, "storm_satellite": (codes, available),
                                                             "storm_ocean": ocean})
    assert not torch.equal(model.storm_ocean_embed.weight, before_ocean)
    assert not torch.equal(model.storm_ir_embed.weight, before_ir)
    assert not torch.equal(model.ri2_head.trunk[0].weight[:, -model.ri2_head.extra:], before_trunk)
    trained = model.trained_heads()
    assert trained["storm_ocean"] == 3 and trained["storm_ir"] == 3 and trained["storm_satellite"] == 3


def test_a_southern_storm_is_mirrored_into_the_poleward_frame():
    model = _model(ri2=True, storm_ir=True, hemisphere_frame=True).eval()
    torch.manual_seed(3)
    with torch.no_grad():
        model.storm_satellite.project.weight.normal_(0, 0.05)
        model.storm_ir_embed.weight.normal_(0, 0.05)
    # An image that is not symmetric north-south: a cold band only north of the centre.
    image = _hurricane(eye=False).copy()
    image[:128] = np.minimum(image[:128], 200.0)
    image[128:] = np.maximum(image[128:], 255.0)
    north_codes = torch.from_numpy(np.stack([encode_bt(image), encode_bt(image)]))[None]
    south_codes = north_codes.flip(-2)
    available = torch.ones(1, 2, dtype=torch.bool)
    pooled = torch.randn(1, 32)
    with torch.no_grad():
        north = model.storm_heads(pooled, torch.tensor([[15.0, -50.0]]), storm_satellite=(north_codes, available))
        south = model.storm_heads(pooled, torch.tensor([[-15.0, -50.0]]), storm_satellite=(south_codes, available))
        unmirrored = model.storm_heads(pooled, torch.tensor([[-15.0, -50.0]]), storm_satellite=(north_codes, available))
    assert torch.allclose(north["intensity_mean"], south["intensity_mean"], atol=1e-5), "the same storm, the other hemisphere"
    assert not torch.allclose(unmirrored["intensity_mean"], south["intensity_mean"], atol=1e-5)


# ---------------------------------------------------------------------------------- the storm stage --

class FakePatches:
    """Staged storm patches (2, 320, 320) for every table row: a hurricane at the row's own position."""

    def __init__(self, rows) -> None:
        self.rows = set(int(r) for r in rows)
        image = _hurricane(320)
        self.image = np.stack([encode_bt(image), encode_bt(image - 10.0)])

    def __contains__(self, row) -> bool:
        return int(row) in self.rows

    def patch(self, row):
        return self.image if int(row) in self.rows else None


@pytest.fixture(scope="module")
def stage(tmp_path_factory):
    from test_stage_b import VARIABLES, era5, storm

    from naturev1.era5 import Normalizer, lead_offsets
    from naturev1.sattrain import staged_grid
    from naturev1.state import StateStepper, channel_roles
    from naturev1.stormtrain import storm_times

    directory = tmp_path_factory.mktemp("v2")
    path, times = era5(directory)
    meta = json.loads(path.with_suffix(".json").read_text())
    grid = staged_grid(meta)
    model = _model(ri2=True, storm_ocean=True, storm_ir=True)
    stepper = StateStepper(channel_roles(VARIABLES, 12), torch.full((12,), 0.5), Normalizer.from_dict(meta["statistics"]),
                           grid, VARIABLES.index("surface_pressure"))
    tracks = [storm("NORTH", 15.0, -50.0, 0.4), storm("SOUTH", -14.0, 150.0, -0.4)]
    items = storm_times(tracks, times, lead_offsets(LEADS), history=2)
    sid = np.array([t.storm_id for t in tracks for _ in t.time])
    stamps = np.concatenate([t.time.astype("datetime64[s]").astype(np.int64) for t in tracks])
    table = {"sid": sid, "time": stamps, "lat": np.concatenate([t.latitude for t in tracks]).astype(np.float64),
             "lon": np.concatenate([t.longitude for t in tracks]).astype(np.float64)}
    # Every fix but the first of each storm has a patch: the first fixes test a missing six-hour-old image.
    patches = FakePatches([i for i in range(len(sid)) if i not in (0, len(tracks[0].time))])
    return path, grid, model, stepper, items, directory, table, patches, VARIABLES


def test_storms_carry_their_infrared_now_and_six_hours_earlier(stage):
    path, _, model, _, items, _, table, patches, _ = stage
    from naturev1.stormtrain import StormTimeDataset

    dataset = StormTimeDataset(path, items, 12, table=table, patches=patches, train=False)
    first = dataset[0]                                                   # both storms at their first fix: no patch staged
    assert first["satellite/ir"].shape[1:] == (2, 256, 256) and first["satellite/ir"].dtype == torch.uint8
    assert not first["satellite/available"].any() and first["satellite/ir"].sum() == 0
    later = dataset[2]
    count = int(later["storm_mask"].sum())
    assert later["satellite/available"][:count].all(), "a patch now and six hours earlier for every storm"
    assert (later["satellite/ir"][:count, 0] > 0).float().mean() > 0.9
    assert "storm/current_wind_radii_nmi" in later and later["storm/current_wind_radii_nmi"].shape[1:] == (3, 4)


def test_the_storm_stage_trains_the_new_inputs_and_the_banks_keep_them(stage):
    path, grid, model, stepper, items, directory, table, patches, variables = stage
    from naturev1.engine import EngineSettings
    from naturev1.stormtrain import finalise_storms, memmap_feature_bank, train_storms

    settings = EngineSettings(run="stage-b-v2", output=str(directory / "runs"), hours=1.0, reserve_minutes=0.0,
                              batch_size=2, workers=0, max_steps=3, warmup_steps=1, checkpoint_minutes=1e9,
                              eval_minutes=1e9, learning_rate=1e-3)
    ocean_before = model.storm_ocean_embed.weight.clone()
    ir_before = model.storm_ir_embed.weight.clone()
    half = len(items) // 2
    report = train_storms(model, settings, path, stepper, grid, variables, items[:half], items[half:], tail=1,
                          retrieval_share=0.0, tail_lr_scale=1.0, table=table, patches=patches)
    assert np.isfinite(report["loss"])
    trained = model.trained_heads()
    assert trained["storm_ocean"] == 3 and trained["storm_ir"] == 3 and trained["storm_satellite"] == 3
    assert not torch.equal(model.storm_ocean_embed.weight, ocean_before)
    assert not torch.equal(model.storm_ir_embed.weight, ir_before)

    bank = memmap_feature_bank(model, path, items, stepper, grid, variables, rollout=False, precision="fp32",
                               table=table, patches=patches)
    count = sum(len(g) for _, g in items)
    assert bank["targets"]["storm_ocean"].shape == (count, len(STORM_OCEAN_FEATURES))
    assert bank["satellite"]["ir"].shape == (count, 2, 256, 256) and bank["satellite"]["available"].shape == (count, 2)
    assert bank["targets"]["storm_ocean"][:, -1].sum() > 0, "the synthetic storms sit over mapped ocean"
    report = finalise_storms(model, bank, bank)
    assert "TEST SEASONS" in report


def test_patches_only_staging_writes_the_storms_and_no_frames(tmp_path):
    from test_satellite_v2 import START, STEPS, track, write_gridsat

    from naturev1.ibtracs import storm_label_table
    from naturev1.satdata import GlobalFrames, StormPatches
    from naturev1.satstage import stage_month

    raw = tmp_path / "raw"
    raw.mkdir()
    for k in range(STEPS):
        stamp = (START + np.timedelta64(3 * k, "h")).astype(object)
        write_gridsat(raw / f"GRIDSAT-B1.{stamp:%Y.%m.%d.%H}.v02r01.nc", k)
    table = storm_label_table([track("north", "2005222N15310"), track("south", "2005222S15150")])
    root = tmp_path / "staged"
    summary = stage_month(root, 2005, 8, table, resolution=2.5, raw_dir=raw, keep_raw=True, download=False,
                          global_frames=False)
    assert summary["frames"] is False and summary["storm_patches_ok"] == 2 * STEPS
    assert not (root / "global" / "2005-08.npy").exists() and (root / "global" / "2005-08.json").exists()
    assert len(GlobalFrames(root)) == 0, "a patches-only month offers no frames"
    patches = StormPatches(root)
    first = next(iter(patches.index))
    assert patches.patch(first).shape == (2, 320, 320)
    again = stage_month(root, 2005, 8, table, resolution=2.5, raw_dir=tmp_path / "nothing", download=False,
                        global_frames=False)
    assert again["frames"] is False, "a staged month is not redone"


def _era5_across_new_year(directory):
    """A synthetic staged ERA5 memmap from 27 December 2010 to 6 January 2011, so seasons 2010 and 2011 both exist."""
    from test_stage_b import VARIABLES

    from naturev1.era5 import Normalizer

    steps, rows, cols = 40, 19, 36
    start = np.datetime64("2010-12-27T00", "s")
    rng = np.random.default_rng(0)
    values = rng.normal(0, 0.3, (steps, rows, cols, len(VARIABLES))).astype(np.float16)
    path = directory / "era5.npy"
    np.save(path, values)
    means = np.array([290, 101300, 101000, 0, 0, 0, 0, 0.005, 1e6, 0.3], np.float32)
    stds = np.array([10, 800, 800, 8, 8, 15, 15, 0.003, 1e6, 0.4], np.float32)
    stats = Normalizer(VARIABLES, means, stds, np.zeros(len(VARIABLES), np.float32)).to_dict()
    times = (start + np.arange(steps) * np.timedelta64(6, "h")).astype(np.int64)
    path.with_suffix(".json").write_text(json.dumps({
        "variables": VARIABLES, "shape": [steps, rows, cols, len(VARIABLES)], "statistics": stats,
        "latitude": np.linspace(-90, 90, rows).tolist(), "longitude": (np.arange(cols) * 10.0).tolist(),
        "time": times.tolist()}))
    return path, start


def test_the_storm_stage_command_runs_a_naturev2_base_end_to_end(tmp_path, monkeypatch):
    """``python -m naturev1.stormtrain`` as the NatureV2 cell launches it: a V2 base, patches into the heads, no
    retrieval, and the release decision measured against a separate reference model."""
    from naturev1 import ibtracs, sattrain
    from naturev1.besttrack import Track
    from naturev1.stormtrain import main

    path, start = _era5_across_new_year(tmp_path)

    def track(sid, lat0, lon0, dlat):
        n = 24
        times = start + np.timedelta64(36, "h") + np.arange(n) * np.timedelta64(6, "h")
        wind = np.linspace(40, 110, n).astype(np.float32)
        return Track(storm_id=sid, name=sid, time=times, latitude=(lat0 + dlat * np.arange(n)).astype(np.float32),
                     longitude=(lon0 - 0.5 * np.arange(n)).astype(np.float32), max_wind_kt=wind,
                     min_pressure_hpa=(1010 - 0.8 * wind).astype(np.float32), status=np.array(["HU"] * n),
                     rmw_nmi=np.full(n, 20.0, np.float32), wind_radii_nmi=np.full((n, 3, 4), 60.0, np.float32),
                     record=np.array([""] * n))

    tracks = [track("NORTH", 15.0, -50.0, 0.3), track("SOUTH", -14.0, 150.0, -0.3)]
    monkeypatch.setattr(ibtracs, "parse_ibtracs", lambda *a, **k: tracks)
    monkeypatch.setattr(sattrain, "VALIDATION_SEASONS", (2011,))
    monkeypatch.setattr(sattrain, "TEST_SEASONS", (2011,))

    sid = np.array([t.storm_id for t in tracks for _ in t.time])
    stamps = np.concatenate([t.time.astype("datetime64[s]").astype(np.int64) for t in tracks])
    np.savez(tmp_path / "labels.npz", sid=sid, time=stamps,
             lat=np.concatenate([t.latitude for t in tracks]).astype(np.float64),
             lon=np.concatenate([t.longitude for t in tracks]).astype(np.float64))
    storms = tmp_path / "sat" / "storms"
    storms.mkdir(parents=True)
    image = _hurricane(320)
    np.save(storms / "2010-12.npy", np.stack([np.stack([encode_bt(image), encode_bt(image - 10.0)])] * len(sid)))
    (storms / "2010-12.json").write_text(json.dumps({"rows": list(range(len(sid))), "ok": [True] * len(sid)}))

    base = _model(storm_ocean=True, storm_ir=True)
    torch.save({"model": base.state_dict(), "config": base.config.to_dict()}, tmp_path / "base_v2.pt")
    reference = _model(ri2=False)
    torch.save({"model": reference.state_dict(), "config": reference.config.to_dict()}, tmp_path / "v1_release.pt")
    torch.save(torch.full((12,), 0.5), tmp_path / "tendency.pt")

    output = tmp_path / "runs"
    main(["--base", str(tmp_path / "base_v2.pt"), "--era5", str(path), "--tendency", str(tmp_path / "tendency.pt"),
          "--ibtracs", "unused.csv", "--table", str(tmp_path / "labels.npz"), "--root", str(tmp_path / "sat"),
          "--output", str(output), "--hours", "1", "--max-steps", "2", "--batch-size", "2", "--workers", "0",
          "--tail", "1", "--rotations", "1", "--ensemble-members", "1", "--no-rollout-members",
          "--patches-in-heads", "--compare-to", str(tmp_path / "v1_release.pt")])

    release = torch.load(output / "stage-b" / "latest.pt", map_location="cpu", weights_only=False)
    config = release["config"]
    assert config["storm_ocean"] and config["storm_ir"] and config["ri2"] and config["hemisphere_frame"]
    assert config["satnet_embed"] == 512 and config["retrieval_layers"] == 0, "no trained retrieval, none built"
    steps = dict(zip(TRACKED_HEADS, release["model"]["head_steps"].tolist()))
    assert steps["storm_ocean"] == 2 and steps["storm_ir"] == 2 and steps["storm_satellite"] == 2 and steps["ri2"] == 2
    report = (output / "stage-b" / "report.txt").read_text()
    assert "AGAINST THE MODEL IT STARTED FROM" in report
    comparison = json.loads((output / "comparison.json").read_text())
    assert len(comparison["new_global_track_km"]) == len(LEADS)
