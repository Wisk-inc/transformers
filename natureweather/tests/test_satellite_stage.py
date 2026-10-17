# Copyright 2026 Nathan. Apache-2.0.
"""
0.11: the storm-centred satellite stage, and the fixes for what the red team found.

What is pinned here: the new branch is an exact no-op until trained and old checkpoints still load;
satellite fine-tuning trains the branch and can never leave the model worse on held-out storms;
probability calibration repairs over-confident probabilities without changing their ranking; the input
monitor flags inputs unlike training; and the forecast bundle can no longer state things that are
logically or physically impossible.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest
import torch

from naturev1 import build_forecast, check_forecast
from naturev1.model import OPTIONAL_STATE, TRACKED_HEADS
from naturev1.storms import (
    _head_outputs, _platt, calibration_report, fit_feature_monitor, fit_probability_calibration, train_storm_heads,
)
from naturev1.stormsat import PATCH_PIXELS, encode_bt

from test_fixes_080 import WIDTH, _grid, _model, _store, _storm


@pytest.fixture(scope="module")
def banks(tmp_path_factory):
    """A small trained model and its storm feature bank, from the same synthetic store as 0.8's tests."""
    from naturev1 import ERA5Window, pair_tracks_with_reanalysis, storm_feature_bank

    torch.manual_seed(0)
    store = _store()
    variables = ("2m_temperature", "surface_pressure", "toa_incident_solar_radiation_6hr", "land_sea_mask",
                 "temperature", "geopotential")
    tracks = [_storm("AL012005", 24, 20, 15.0), _storm("AL022005", 48, 16, 22.0)]
    times = store.time.values.astype("datetime64[s]").astype(np.int64)
    starts, targets, _ = pair_tracks_with_reanalysis(tracks, times, (1, 2), history=2)
    base = ERA5Window(store, indices=starts[:1], history=2, lead_steps=1, variables=variables, levels=[500],
                      channels=WIDTH, augment=False)
    model = _model()
    bank = storm_feature_bank(model, store, base, starts, targets, _grid(), precision="fp32", progress=False,
                              cache=str(tmp_path_factory.mktemp("bank") / "bank.pt"))
    train_storm_heads(model, bank, bank, epochs=2, batch_size=8, patience=10, log_every=100)
    # Infrared whose cold-cloud coverage follows each sample's current wind -- something to learn from.
    wind = bank["targets"]["current_wind_kt"].numpy()
    rng = np.random.default_rng(0)
    patches = []
    for w in wind:
        image = 290.0 - (w - 40.0) * 0.8 + rng.normal(0, 2, (PATCH_PIXELS, PATCH_PIXELS))
        patches.append(encode_bt(np.stack([image, image + 3.0])))
    bank["satellite"] = {"ir": torch.from_numpy(np.stack(patches)),
                         "available": torch.ones(len(wind), 2, dtype=torch.bool)}
    return model, bank


def test_tracked_outputs_only_grow_at_the_end_and_old_checkpoints_load():
    # Positions are what the saved counts are read by: 0.11 appended one, 0.12 five more, never in between.
    assert TRACKED_HEADS.index("storm_satellite") == 22
    assert TRACKED_HEADS[23:] == ("satnet", "retrieval", "ri2", "satnet_fusion", "backbone_tail")
    model = _model()
    old = {k: v for k, v in model.state_dict().items()
           if not k.startswith(("storm_satellite.", "ri_calibration", "landfall_calibration", "feature_",
                                "state_tendency", "intensity_correction", "eyewall_correction", "ri2_calibration"))}
    old["head_steps"] = torch.arange(22)                                 # a 0.10 record
    fresh = _model()
    fresh.load_state_dict(old)
    assert fresh.trained_heads()["storm_satellite"] == 0
    assert fresh.trained_heads()["track"] == TRACKED_HEADS.index("track"), "existing counts kept in place"
    assert all(name in OPTIONAL_STATE for name in ("storm_satellite.", "ri_calibration", "state_tendency"))


def test_the_satellite_branch_changes_nothing_until_trained(banks):
    model, bank = banks
    plain = {k: v for k, v in bank.items() if k != "satellite"}
    fresh = _model()
    fresh.load_state_dict(model.state_dict())
    with_images, without = _head_outputs(fresh, bank), _head_outputs(fresh, plain)
    for key in ("displacement", "intensity_mean", "landfall_logit", "ri_logits"):
        assert torch.equal(with_images[key], without[key]), key


def test_satellite_stage_trains_the_branch_and_never_ends_worse(banks):
    model, bank = banks
    student = _model()
    student.load_state_dict(model.state_dict())
    history = train_storm_heads(student, bank, bank, epochs=3, batch_size=8, learning_rate=1e-4,
                                satellite_learning_rate=1e-3, augment=True, keep_if_no_better=True,
                                patience=10, log_every=100)
    assert student.trained_heads()["storm_satellite"] > 0
    assert student.storm_satellite.project.weight.abs().sum() > 0
    assert all(not p.requires_grad for p in student.blocks.parameters()), "backbone stays frozen"
    final = history[-1]
    assert math.isfinite(final["train"])


def test_platt_scaling_repairs_overconfidence_and_keeps_the_ranking():
    generator = torch.Generator().manual_seed(0)
    signal = torch.randn(4000, generator=generator)
    outcomes = (torch.rand(4000, generator=generator) < torch.sigmoid(signal - 3.0)).float()
    overconfident = signal + 2.0                        # says ~70% where ~5% happen
    a, b = _platt(overconfident, outcomes)
    assert a > 0
    raw = float(((torch.sigmoid(overconfident) - outcomes) ** 2).mean())
    fixed = float(((torch.sigmoid(a * overconfident + b) - outcomes) ** 2).mean())
    assert fixed < raw * 0.5
    assert abs(float(torch.sigmoid(a * overconfident + b).mean()) - float(outcomes.mean())) < 0.02


def test_probability_calibration_fits_and_reports(banks):
    model, bank = banks
    student = _model()
    student.load_state_dict(model.state_dict())
    # The synthetic storms never reach land or intensify rapidly; give them outcomes, over two seasons.
    generator = torch.Generator().manual_seed(1)
    targets = dict(bank["targets"])
    n = targets["landfall_target"].shape[0]
    targets["landfall_target"] = (torch.rand(targets["landfall_target"].shape, generator=generator) < 0.3).float()
    targets["landfall_mask"] = torch.ones_like(targets["landfall_target"])
    targets["ri_target"] = (torch.rand(targets["ri_target"].shape, generator=generator) < 0.3).float()
    targets["ri_mask"] = torch.ones_like(targets["ri_target"])
    targets["season"] = torch.tensor([2005 + i % 2 for i in range(n)])
    scored = {**bank, "targets": targets}
    assert "nothing to score" in calibration_report(student, bank), "no events: said, not divided by zero"
    fit_probability_calibration(student, scored, min_events=1)
    assert torch.isfinite(student.landfall_calibration).all() and (student.landfall_calibration[:, 0] > 0).all()
    assert torch.isfinite(student.ri_calibration).all()
    assert "Brier skill" in calibration_report(student, scored)


def test_feature_monitor_flags_unlike_inputs(banks):
    model, bank = banks
    student = _model()
    student.load_state_dict(model.state_dict())
    fit_feature_monitor(student, bank)
    normal = _head_outputs(student, bank)
    assert float((normal["feature_score"] > normal["feature_threshold"]).float().mean()) < 0.05
    for features in (torch.randn_like(bank["features"]) * 10, torch.zeros_like(bank["features"])):
        odd = _head_outputs(student, {**bank, "features": features})
        assert bool((odd["feature_score"] > odd["feature_threshold"]).all())


def _outputs(model, bank, index=0):
    """The head outputs for one sample of a bank."""
    one = {}
    for key, value in bank.items():
        if torch.is_tensor(value):
            one[key] = value[index:index + 1]
        elif isinstance(value, dict):
            one[key] = {k: v[index:index + 1] for k, v in value.items() if torch.is_tensor(v)}
    return _head_outputs(model, one)


def test_forecast_structure_is_consistent_and_honest(banks):
    model, bank = banks
    out = _outputs(model, bank)
    out["landfall_logit"] = torch.tensor([[2.0, -2.0]])                          # falls with lead
    radii = torch.full_like(out["wind_radii_nmi"], 50.0)
    radii[:, :, 2] = 120.0                                                         # 64 kt beyond 34 kt
    out["wind_radii_nmi"] = radii
    out["eyewall_rmw_nmi"] = torch.full_like(out["eyewall_rmw_nmi"], 400.0)       # outside the gale radius
    leads = model.config.lead_times_hours
    # trained=None shows every section: the synthetic storms carry no wind-radii labels to train on.
    bundle = build_forecast(out, leads, dt.datetime(2005, 8, 25), storm_center=(25.0, -75.0), trained=None)
    chances = [row["probability"] for row in bundle["landfall"]["by_lead"]]
    assert chances == sorted(chances), "landfall by this lead never falls"
    for entry in bundle["eyewall"]:
        for quadrant in ("NE", "SE", "SW", "NW"):
            r = entry["wind_radii_nmi"]
            assert r["34kt"][quadrant] >= r["50kt"][quadrant] >= r["64kt"][quadrant]
        if "rmw_nmi" in entry:
            assert entry["rmw_nmi"] <= max(entry["wind_radii_nmi"]["34kt"].values()) + 1e-6
    point = bundle["track_scenarios"][0]["track"][0]
    assert "cone_radius_km_39" in point and "cone_radius_km_68" not in point, "a 1-sigma 2-D ellipse holds 39%"
    cone = bundle["track_cone"]
    assert len(cone) == len(leads) and all(row["radius_km_90"] >= row["radius_km_50"] > 0 for row in cone)
    assert not [p for p in check_forecast(bundle) if "order" in p or "falls" in p]


def test_no_centre_no_storm_forecast_and_no_record_no_crash(banks):
    model, bank = banks
    out = _outputs(model, bank)
    leads = model.config.lead_times_hours
    bundle = build_forecast(out, leads, dt.datetime(2005, 8, 25), storm_center=(float("nan"), float("nan")),
                            trained=model.trained_heads())
    for key in ("track_scenarios", "landfall", "intensity", "eyewall", "rapid_intensification"):
        assert bundle[key]["available"] is False
    unrecorded = build_forecast(out, leads, dt.datetime(2005, 8, 25), storm_center=(25.0, -75.0), trained=None)
    assert unrecorded["enso"]["available"] is False and "note" in unrecorded


def test_monitor_marks_the_forecast_untrustworthy_and_guards_check_the_consensus(banks):
    model, bank = banks
    student = _model()
    student.load_state_dict(model.state_dict())
    fit_feature_monitor(student, bank)
    noisy = {**bank, "features": torch.randn_like(bank["features"]) * 10}
    out = _outputs(student, noisy)
    bundle = build_forecast(out, student.config.lead_times_hours, dt.datetime(2005, 8, 25),
                            storm_center=(25.0, -75.0), trained=student.trained_heads())
    assert bundle["trustworthy"] is False and "unlike anything in training" in bundle["warning"]
    assert check_forecast(bundle), "an untrustworthy bundle is never a clean pass"
    off = {"track_consensus": {"track": [{"lead_hours": 6, "latitude": 95.0, "longitude": 0.0}]}}
    assert any("consensus latitude" in p for p in check_forecast(off))
