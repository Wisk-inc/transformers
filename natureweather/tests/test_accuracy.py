# Copyright 2026 Nathan. Apache-2.0.
"""
0.12 accuracy add-ons: the test-time rotation ensemble, intensity-class rebalancing, the weight average, the
iterated satellite centre fix and the gridded wind-speed probabilities.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ihelix import FieldGrid, Geometry, fibonacci_sphere

from naturev1.corpora import latlon_to_coords
from naturev1.model import (
    ROTATION_STEP_DEG, NatureConfig, NatureV1, calendar_features, merge_rotations, rotation_angles, turned_centre,
)

LEADS = (6, 12, 24, 48)


def _tiny_model(**over):
    torch.manual_seed(0)
    config = NatureConfig(analysis_channels=8, state_channels=3, hidden_size=32, num_layers=2, num_heads=2, num_kv_heads=1,
                          head_dim=16, intermediate_size=64, latent_points=96, latent_neighbours=12, latent_cluster=16,
                          landmark_dim=16, index_branching=2, index_beam_width=2, index_topk=2, recurrent_heads=2,
                          recurrent_head_dim=16, history_frames=2, min_radius_km=600.0, max_radius_km=2000.0,
                          encode_radius_km=900.0, lead_times_hours=LEADS, **over)
    model = NatureV1(config, fibonacci_sphere(96, num_neighbours=12, cluster_size=16)).eval()
    with torch.no_grad():
        # Untrained heads are built to ignore their input; give them something to say.
        for parameter in model.storm_head_parameters():
            parameter.normal_(0, 0.05)
    return model


def _atmosphere(batch=2):
    """A smooth, longitude-varying global analysis on a 5 x 7.5 degree grid, two frames."""
    lats, lons = np.linspace(-87.5, 87.5, 36), np.arange(0, 360, 7.5)
    grid = FieldGrid.source(latlon_to_coords(lats, lons), Geometry.globe())
    lat = torch.tensor(np.repeat(lats, len(lons)), dtype=torch.float32)
    lon = torch.tensor(np.tile(lons, len(lats)), dtype=torch.float32)
    field = torch.stack([torch.cos(torch.deg2rad(lat) * k) * torch.sin(torch.deg2rad(lon) * (k % 3 + 1))
                         for k in range(8)], -1)
    analysis = field.view(1, 1, -1, 8).repeat(batch, 2, 1, 1)
    calendar = calendar_features(torch.tensor([[1.5e9, 1.5e9 + 21600]] * batch, dtype=torch.float64))
    return analysis, grid, calendar


# ------------------------------------------------------------------------------ rotation ensemble --

def test_rotation_angles_never_repeat_and_a_smaller_ensemble_is_a_prefix_of_a_larger_one():
    assert rotation_angles(1) == [] and len(rotation_angles(6)) == 5
    assert rotation_angles(6)[:2] == rotation_angles(3), "a bank built with more rotations serves a model using fewer"
    angles = rotation_angles(12)
    assert len({round(a, 6) for a in angles}) == len(angles)
    for a in angles:
        assert min(a % 1.5, 1.5 - a % 1.5) > 1e-3, "never on the analysis grid's own spacing"
    assert ROTATION_STEP_DEG == pytest.approx(137.50776, abs=1e-5)
    turned = turned_centre(torch.tensor([[20.0, 179.0], [-10.0, -175.0]]), 10.0)
    assert torch.allclose(turned, torch.tensor([[20.0, -171.0], [-10.0, -165.0]]))


def test_merging_rotations_gives_the_mixture_not_an_average_of_variances():
    a = {"intensity_mean": torch.tensor([[10.0]]), "intensity_log_var": torch.tensor([[math.log(4.0)]]),
         "displacement": torch.tensor([[[1.0, 0.0]]]), "log_scale": torch.tensor([[[0.0, 0.0]]]),
         "ri_logits": torch.tensor([2.0]), "landfall_time_log_var": torch.tensor([math.log(1.0)]),
         "count": torch.tensor([3])}
    b = {"intensity_mean": torch.tensor([[14.0]]), "intensity_log_var": torch.tensor([[math.log(2.0)]]),
         "displacement": torch.tensor([[[3.0, 0.0]]]), "log_scale": torch.tensor([[[0.0, 0.0]]]),
         "ri_logits": torch.tensor([0.0]), "landfall_time_log_var": torch.tensor([math.log(3.0)]),
         "count": torch.tensor([3])}
    merged = merge_rotations([a, b])
    assert float(merged["intensity_mean"]) == pytest.approx(12.0)
    # Law of total variance: mean of variances (3) plus variance of means (4).
    assert float(merged["intensity_log_var"].exp()) == pytest.approx(7.0, rel=1e-5)
    # Track scale: sqrt(mean of scale^2 (1) + spread of displacements (1)) east, 1 north.
    assert merged["log_scale"].exp()[0, 0].tolist() == pytest.approx([math.sqrt(2.0), 1.0], rel=1e-5)
    assert float(merged["ri_logits"]) == pytest.approx(1.0), "logits pool in logit space"
    assert float(merged["landfall_time_log_var"].exp()) == pytest.approx(2.0, rel=1e-5)
    assert merged["count"].dtype == torch.long and int(merged["count"]) == 3
    same = merge_rotations([a, a])
    for key in a:
        assert torch.allclose(same[key].float(), a[key].float(), atol=1e-6), key


def test_rotated_pools_see_the_same_storm_through_different_arithmetic():
    model = _tiny_model()
    analysis, grid, calendar = _atmosphere()
    centre = torch.tensor([[20.0, -60.0], [-15.0, 120.0]])
    with torch.no_grad():
        latent = model.encode(None, None, analysis, grid, calendar)
        current = model.norm(model._run_blocks(latent, model.latent_grid))[:, -1]
        base = model.storm_pool(current, centre)
        elsewhere = model.storm_pool(current, torch.tensor([[50.0, 10.0], [-40.0, 0.0]]))
        for degrees in rotation_angles(4):
            near, far = model.rotated_pools({"analysis": (analysis, grid)}, calendar, centre, degrees)
            change = float((near - base).norm() / base.norm())
            assert 1e-4 < change < 0.6, (degrees, change)
            assert change < 0.5 * float((elsewhere - base).norm() / base.norm())
            assert far.shape == near.shape and torch.isfinite(far).all()
        # One atmosphere, several storms: one backbone pass serves them all.
        single = model.rotated_pools({"analysis": (analysis[:1], grid)}, calendar[:1], centre, 30.0)[0]
        both = model.rotated_pools({"analysis": (analysis, grid)}, calendar, centre, 30.0)[0]
        assert torch.allclose(single, both, atol=1e-5)


def test_the_forecast_and_the_bank_compute_the_same_rotation_ensemble():
    model = _tiny_model()
    analysis, grid, calendar = _atmosphere()
    centre, state = torch.tensor([[20.0, -60.0], [-15.0, 120.0]]), torch.randn(2, 6)
    with torch.no_grad():
        single = model(analysis=analysis, analysis_grid=grid, calendar=calendar, storm_center=centre, storm_state=state)
        model.config.storm_rotations = 3
        ensemble = model(analysis=analysis, analysis_grid=grid, calendar=calendar, storm_center=centre,
                         storm_state=state)
        latent = model.encode(None, None, analysis, grid, calendar)
        current = model.norm(model._run_blocks(latent, model.latent_grid))[:, -1]
        members = [model.storm_heads(model.storm_pool(current, centre), centre, state, None, None,
                                     model.storm_pool(current, centre, 2.0 * model.config.storm_pool_radius_km))]
        for degrees in rotation_angles(3):
            near, far = model.rotated_pools({"analysis": (analysis, grid)}, calendar, centre, degrees)
            members.append(model.storm_heads(near, centre, state, None, None, far))
        by_hand = merge_rotations(members)
        model.train()
        training = model(analysis=analysis, analysis_grid=grid, calendar=calendar, storm_center=centre,
                         storm_state=state)
    for key in ("intensity_mean", "intensity_log_var", "displacement", "log_scale", "ri_logits"):
        assert torch.allclose(ensemble[key], by_hand[key], atol=1e-5), key
    assert not torch.allclose(ensemble["intensity_mean"], single["intensity_mean"]), "the ensemble changes the answer"
    assert (ensemble["intensity_log_var"].exp() >= 0.0).all()
    # Training never pays for the ensemble: a training pass is one pass.
    assert torch.allclose(training["intensity_mean"], single["intensity_mean"], atol=1e-5)


def _rotation_bank(model, n=64, rotations=3, seed=0):
    from naturev1.model import STORM_ENVIRONMENT_FEATURES

    g = torch.Generator().manual_seed(seed)
    hidden = model.config.hidden_size
    features = torch.randn(n, hidden, generator=g)
    far = torch.randn(n, hidden, generator=g)
    targets = {"storm_center": torch.stack([10 + 20 * torch.rand(n, generator=g), -80 + 40 * torch.rand(n, generator=g)], -1),
               "storm_state": torch.randn(n, 6, generator=g),
               "storm_environment": torch.zeros(n, len(STORM_ENVIRONMENT_FEATURES)),
               "track_valid": torch.ones(n, len(LEADS)), "intensity_mask": torch.ones(n, len(LEADS), 2)}
    bank = {"features": features, "features_far": far, "targets": targets,
            "features_tta": features.unsqueeze(1) + 0.5 * torch.randn(n, rotations - 1, hidden, generator=g),
            "features_far_tta": far.unsqueeze(1) + 0.5 * torch.randn(n, rotations - 1, hidden, generator=g)}
    return bank


def _truth_from(model, bank, rotations):
    from naturev1.storms import _head_outputs

    model.config.storm_rotations = rotations
    out = _head_outputs(model, bank)
    size = out["displacement"].shape[0]
    probabilities = out["mode_logits"].softmax(-1)
    bank["targets"]["track_target"] = (out["displacement"] * probabilities.view(size, -1, 1, 1)).sum(1)
    bank["targets"]["intensity_target"] = out["intensity_mean"].clone()


def test_held_out_seasons_decide_whether_the_rotation_ensemble_is_used():
    from naturev1.stormtrain import select_rotations

    model = _tiny_model()
    bank = _rotation_bank(model)
    _truth_from(model, bank, rotations=3)                   # the world agrees with the three-member ensemble
    report = select_rotations(model, bank)
    assert model.config.storm_rotations == 3, report
    assert "-> 3 members" in report

    _truth_from(model, bank, rotations=1)                   # ... and now with a single pass
    report = select_rotations(model, bank)
    assert model.config.storm_rotations == 1 and "single pass" in report, report

    del bank["features_tta"], bank["features_far_tta"]
    model.config.storm_rotations = 4
    assert "not in the bank" in select_rotations(model, bank) and model.config.storm_rotations == 1


def test_a_bank_with_fewer_rotations_than_the_model_uses_what_it_has():
    from naturev1.storms import _head_outputs

    model = _tiny_model()
    bank = _rotation_bank(model, rotations=2)
    model.config.storm_rotations = 2
    two = _head_outputs(model, bank)
    model.config.storm_rotations = 5
    five = _head_outputs(model, bank)
    assert torch.allclose(two["intensity_mean"], five["intensity_mean"])


# ------------------------------------------------------------------------------- class balancing --

def _items(winds):
    fixes = [SimpleNamespace(track=SimpleNamespace(max_wind_kt=np.array([w], dtype=float)), point=0) for w in winds]
    return [(k, fixes[k:k + 4]) for k in range(0, len(fixes), 4)]


def test_class_weights_lift_the_rare_strong_storms_and_keep_the_loss_its_size():
    from naturev1.stormtrain import CLASS_EDGES_KT, class_weights

    winds = [35.0] * 800 + [80.0] * 150 + [120.0] * 50 + [float("nan")] * 4
    weights = class_weights(_items(winds))
    assert weights[0] < weights[1] < weights[2]
    counts = torch.tensor([800.0, 150.0, 50.0])
    assert float((weights * counts).sum() / counts.sum()) == pytest.approx(1.0, rel=1e-5)
    assert float(weights[2] / weights[0]) == pytest.approx(4.0, rel=1e-5), "square root of a 16x imbalance"
    capped = class_weights(_items([35.0] * 9900 + [80.0] * 99 + [120.0] * 1))
    assert float(capped[2] / capped[0]) == pytest.approx(4.0, rel=1e-5), "capped at four times the commonest"
    # The training step classifies a wind exactly as the count did, edges included.
    for wind in (30.0, 63.0, 64.0, 95.0, 96.0, 150.0):
        step = int(torch.bucketize(torch.tensor(wind), torch.tensor(CLASS_EDGES_KT), right=True))
        assert step == int(np.searchsorted(CLASS_EDGES_KT, wind)), wind


def test_sample_weights_weigh_the_intensity_likelihood():
    from naturev1.losses import total_loss

    outputs = {"intensity_mean": torch.zeros(2, 3, 2), "intensity_log_var": torch.zeros(2, 3, 2)}
    batch = {"intensity_target": torch.stack([torch.zeros(3, 2), torch.full((3, 2), 3.0)]),
             "intensity_mask": torch.ones(2, 3, 2)}
    plain = total_loss(outputs, batch, ())[1]["intensity"]
    batch["sample_weight"] = torch.tensor([1.0, 1.0])
    assert total_loss(outputs, batch, ())[1]["intensity"] == pytest.approx(plain)
    batch["sample_weight"] = torch.tensor([1e-6, 1.0])
    heavy = total_loss(outputs, batch, ())[1]["intensity"]
    assert heavy == pytest.approx(0.5 * (9.0 + math.log(2 * math.pi)), rel=1e-4), "only the second storm counts"


# --------------------------------------------------------------------------------- weight average --

class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.layer(x)


def test_the_weight_average_is_used_for_evaluation_and_survives_a_resume(tmp_path):
    from naturev1.engine import Engine, EngineSettings

    settings = EngineSettings(run="ema-test", output=str(tmp_path), hours=1.0, reserve_minutes=0.0, workers=0,
                              warmup_steps=1, learning_rate=0.05, ema_decay=0.9, precision="fp32")
    torch.manual_seed(0)
    toy = _Toy()
    engine = Engine(settings, toy, toy)
    x, y = torch.randn(32, 4), torch.randn(32, 1)
    for _ in range(25):
        engine.backward_step(((engine.model(x) - y) ** 2).mean())
    live = [p.detach().clone() for p in toy.parameters()]
    assert not all(torch.equal(p, e) for p, e in zip(live, engine.ema_params)), "the average lags the weights"
    with engine.averaged():
        for p, e in zip(toy.parameters(), engine.ema_params):
            assert torch.equal(p, e)
    for p, saved in zip(toy.parameters(), live):
        assert torch.equal(p, saved), "the live weights come back after evaluation"
    engine.checkpoint()

    again = _Toy()
    resumed = Engine(settings, again, again)
    assert resumed.resume() and resumed.step == 25
    for e, f in zip(engine.ema_params, resumed.ema_params):
        assert torch.equal(e, f), "the average resumes where it was, not from the live weights"


# ------------------------------------------------------------------------------ satellite centre --

class _Pointing(torch.nn.Module):
    """A stand-in storm network that sees 80% of the way from where the loop was cut to the true centre."""

    def __init__(self, truth, cuts):
        super().__init__()
        self.truth, self.cuts = truth, cuts
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, codes, available, latitude, context=None):
        lat, lon = self.cuts[-1]
        north = (self.truth[0] - lat) * 111.195
        east = (self.truth[1] - lon) * 111.195 * math.cos(math.radians(lat))
        return {"centre_offset_km": 0.8 * torch.tensor([[north, east]]),
                "centre_log_var": torch.full((1, 2), math.log(15.0 ** 2))}


def test_the_centre_fix_iterates_to_the_storm():
    from naturev1.satnet import refine_centre

    cuts = []

    def fetch(lat, lon):
        cuts.append((lat, lon))
        return torch.zeros(1, 4, 2, 8, 8, dtype=torch.uint8), torch.ones(1, 4, dtype=torch.bool)

    truth = (18.0, -179.0)
    network = _Pointing(truth, cuts)
    found = refine_centre(network, fetch, (16.5, 179.5), passes=8, tolerance_km=2.0)
    lat, lon = found["centre"]
    assert abs(lat - truth[0]) < 0.05 and abs(((lon - truth[1] + 180) % 360) - 180) < 0.05, found["centre"]
    assert -180.0 <= lon < 180.0, "across the dateline and back into range"
    assert 2 < found["passes"] <= 8 and len(cuts) == found["passes"], "each pass reads a loop cut at the better centre"
    assert found["uncertainty_km"] == pytest.approx(15.0 * math.sqrt(2.0), rel=1e-5)
    once = refine_centre(_Pointing(truth, cuts), fetch, (16.5, 179.5), passes=1)
    assert once["passes"] == 1 and abs(once["centre"][0] - truth[0]) > abs(lat - truth[0])


# ------------------------------------------------------------------------------ wind probabilities --

def _westward(leads, wind_kt=100.0):
    count = len(leads)
    displacement = torch.zeros(1, 1, count, 2)
    for i, hours in enumerate(leads):
        displacement[0, 0, i, 1] = -hours / 6
    return {"mode_logits": torch.zeros(1, 1), "displacement": displacement,
            "log_scale": torch.full((1, 1, count, 2), -6.0), "correlation": torch.zeros(1, 1, count),
            "intensity_mean": torch.zeros(1, count, 2).index_fill_(2, torch.tensor([0]), wind_kt * 0.514444),
            "intensity_log_var": torch.full((1, count, 2), -8.0)}


def test_the_wind_probability_map_follows_the_track_and_wraps_the_dateline():
    from naturev1.windfield import wind_probability_map

    hazard = wind_probability_map(_westward(LEADS), 0, (20.0, -178.0), LEADS, half_width_deg=6.0, step_deg=1.0,
                                  samples=20, trained={})
    lats, lons, p = hazard["latitude"], hazard["longitude"], hazard["probability"]
    assert p.shape == (13, 13, len(LEADS), 3) and hazard["arrival_hours"].shape == (13, 13, 2)
    assert float(lons.min()) >= -180.0 and float(lons.max()) < 180.0 and float(lons.max()) > 170.0
    row = int((lats - 20.0).abs().argmin())
    column = lambda lon: int((((lons - lon + 180.0) % 360.0) - 180.0).abs().argmin())
    ahead = column(178.0)                       # four degrees west, across the dateline: the +24 h position
    assert float(p[row, ahead, -1, 2]) == 1.0, "hurricane-force on the track"
    assert float(p[row, ahead, 1, 2]) == 0.0 and float(p[row, ahead, 2, 2]) == 1.0, "arrives by +24 h, not by +12 h"
    assert float(p[row, column(-173.0), -1, 2]) == 0.0, "nothing east, where the storm never goes"
    assert float(p[0, ahead, -1, 2]) == 0.0, "nor six degrees to the south"
    assert (p.diff(dim=2) >= 0).all(), "cumulative with lead"
