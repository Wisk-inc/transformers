# Copyright 2026 Nathan. Apache-2.0.
"""
Stage B on synthetic data: the tail of the backbone and the heads train together on every storm at a
time, the frozen blocks stay exactly where they were, and the banks carry both rollout members into a
consensus, calibration and bias correction that a test-season scorecard reports.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from ihelix import fibonacci_sphere

from naturev1.besttrack import Track
from naturev1.engine import EngineSettings
from naturev1.era5 import Normalizer, lead_offsets
from naturev1.model import NatureConfig, NatureV1
from naturev1.sattrain import staged_grid
from naturev1.state import StateStepper, channel_roles
from naturev1.stormtrain import finalise_storms, memmap_feature_bank, storm_times, train_storms

VARIABLES = ["2m_temperature", "mean_sea_level_pressure", "surface_pressure", "u_component_of_wind@850",
             "v_component_of_wind@850", "u_component_of_wind@200", "v_component_of_wind@200",
             "specific_humidity@700", "toa_incident_solar_radiation_6hr", "land_sea_mask"]
START = np.datetime64("2010-08-01T00", "s")
STEPS = 40
LEADS = (6, 12, 24)


def era5(directory):
    rows, cols = 19, 36
    rng = np.random.default_rng(0)
    values = rng.normal(0, 0.3, (STEPS, rows, cols, len(VARIABLES))).astype(np.float32)
    means = np.array([290, 101300, 101000, 0, 0, 0, 0, 0.005, 1e6, 0.3], np.float32)
    stds = np.array([10, 800, 800, 8, 8, 15, 15, 0.003, 1e6, 0.4], np.float32)
    path = directory / "era5.npy"
    np.save(path, values.astype(np.float16))
    stats = Normalizer(VARIABLES, means, stds, np.zeros(len(VARIABLES), np.float32)).to_dict()
    times = (START + np.arange(STEPS) * np.timedelta64(6, "h")).astype(np.int64)
    path.with_suffix(".json").write_text(json.dumps({
        "variables": VARIABLES, "shape": [STEPS, rows, cols, len(VARIABLES)], "statistics": stats,
        "latitude": np.linspace(-90, 90, rows).tolist(), "longitude": (np.arange(cols) * 10.0).tolist(),
        "time": times.tolist()}))
    return path, times


def storm(sid, lat0, lon0, dlat, year_offset=0):
    n = 24
    times = (START + np.timedelta64(6 * 6, "h") + np.arange(n) * np.timedelta64(6, "h"))
    wind = np.linspace(40, 110, n).astype(np.float32)
    return Track(storm_id=sid, name=sid, time=times, latitude=(lat0 + dlat * np.arange(n)).astype(np.float32),
                 longitude=(lon0 - 0.5 * np.arange(n)).astype(np.float32), max_wind_kt=wind,
                 min_pressure_hpa=(1010 - 0.8 * wind).astype(np.float32), status=np.array(["HU"] * n),
                 rmw_nmi=np.full(n, 20.0, np.float32), wind_radii_nmi=np.full((n, 3, 4), 60.0, np.float32),
                 record=np.array([""] * n))


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    directory = tmp_path_factory.mktemp("b")
    path, times = era5(directory)
    meta = json.loads(path.with_suffix(".json").read_text())
    grid = staged_grid(meta)
    torch.manual_seed(0)
    config = NatureConfig(analysis_channels=12, state_channels=8, hidden_size=32, num_layers=3, num_heads=2,
                          num_kv_heads=1, head_dim=16, intermediate_size=64, latent_points=96, latent_neighbours=12,
                          latent_cluster=16, landmark_dim=16, index_branching=2, index_beam_width=2, index_topk=2,
                          recurrent_heads=2, recurrent_head_dim=16, history_frames=2, min_radius_km=600.0,
                          max_radius_km=2000.0, encode_radius_km=900.0, lead_times_hours=LEADS, ri2=True)
    model = NatureV1(config, fibonacci_sphere(96, num_neighbours=12, cluster_size=16))
    roles = channel_roles(VARIABLES, 12)
    stepper = StateStepper(roles, torch.full((12,), 0.5), Normalizer.from_dict(meta["statistics"]), grid,
                           VARIABLES.index("surface_pressure"))
    tracks = [storm("NORTH", 15.0, -50.0, 0.4), storm("SOUTH", -14.0, 150.0, -0.4)]
    items = storm_times(tracks, times, lead_offsets(LEADS), history=2)
    return path, grid, model, stepper, items, directory


def test_items_group_every_storm_at_a_time(setup):
    _, _, _, _, items, _ = setup
    assert items and all(len(group) == 2 for _, group in items), "both storms share every initial time"


def test_tail_and_heads_learn_and_the_frozen_blocks_do_not_move(setup, capsys):
    path, grid, model, stepper, items, directory = setup
    frozen = {k: v.clone() for k, v in model.blocks[0].state_dict().items()}
    tail = {k: v.clone() for k, v in model.blocks[-1].state_dict().items()}
    heads = model.track_head.parameters_head.weight.clone()
    settings = EngineSettings(run="stage-b-test", output=str(directory / "runs"), hours=1.0, reserve_minutes=0.0,
                              batch_size=2, workers=0, max_steps=3, warmup_steps=1, checkpoint_minutes=1e9,
                              eval_minutes=1e9, learning_rate=1e-3)
    half = len(items) // 2
    report = train_storms(model, settings, path, stepper, grid, VARIABLES, items[:half], items[half:], tail=1,
                          retrieval_share=0.0, tail_lr_scale=1.0)
    assert np.isfinite(report["loss"]) and "state" in report and "anchor" in report
    for key, value in model.blocks[0].state_dict().items():
        assert torch.equal(value, frozen[key]), f"frozen block changed: {key}"
    assert any(not torch.equal(v, tail[k]) for k, v in model.blocks[-1].state_dict().items()), "the tail trained"
    assert not torch.equal(model.track_head.parameters_head.weight, heads)
    trained = model.trained_heads()
    assert trained["backbone_tail"] == 3 and trained["ri2"] == 3 and trained["track"] > 0
    assert model.config.hemisphere_frame
    assert "intensity weights by class" in capsys.readouterr().out


def test_banks_carry_both_rollout_members_and_finalise_reports(setup):
    path, grid, model, stepper, items, _ = setup
    bank = memmap_feature_bank(model, path, items, stepper, grid, VARIABLES, rollout=True, precision="fp32",
                               rotations=3)
    count = sum(len(g) for _, g in items)
    assert bank["features"].shape == (count, 32)
    assert bank["features_tta"].shape == (count, 2, 32) and bank["features_far_tta"].shape == (count, 2, 32)
    assert not torch.allclose(bank["features_tta"][:, 0], bank["features"]), "a rotated member is a different pass"
    assert bank["targets"]["rollout_steering"].shape == (count, len(LEADS), 2)
    assert bank["targets"]["rollout_vortex"].shape == (count, len(LEADS), 2)
    report = finalise_storms(model, bank, bank)
    assert "TEST SEASONS" in report and "roll-steer" in report
    assert "rotation ensemble on held-out seasons" in report and model.config.storm_rotations in (1, 2, 3)
    from naturev1.model import TRACK_MEMBERS
    from naturev1.thermo import STORM_PHYSICS_FEATURES

    assert model.track_consensus.shape[-1] == len(TRACK_MEMBERS) and torch.isfinite(model.track_consensus).all()
    assert bank["targets"]["storm_physics"].shape == (count, len(STORM_PHYSICS_FEATURES))
    assert "INTENSITY (MAE kt)" in report and "predicted consensus error" in report
    assert int(model.statistics.fitted[:4].min()) == 1, "regression, logistic, climatology and consensus fitted"
    assert (bank["targets"]["basin"] == -1).all(), "synthetic tracks carry no basin"

    from naturev1.stormtrain import compare_models, describe_comparison

    torch.manual_seed(1)
    old = NatureV1(model.config, fibonacci_sphere(96, num_neighbours=12, cluster_size=16))
    old.config = NatureConfig.from_dict({**model.config.to_dict(), "hemisphere_frame": False, "ri2": False})
    old_bank = memmap_feature_bank(old, path, items, stepper, grid, VARIABLES, rollout=False, precision="fp32")
    result = compare_models(model, bank, old, old_bank)
    assert len(result["new_global_track_km"]) == len(LEADS) and 0.0 <= result["better_share_global"] <= 1.0
    assert "better than the starting model" in describe_comparison(result)
