# Copyright 2026 Nathan. Apache-2.0.
"""
The hurricane physics: steering, shear, the residual track head, and the multi-model consensus.

Each check is against an answer worked out by hand -- a storm in a uniform westerly must go east by
exactly u*t, shear between two levels must be their difference, a consensus must find the member that
is exactly right -- so a sign error or a wrong unit cannot hide behind a loss curve.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from naturev1 import (
    STORM_ENVIRONMENT_FEATURES,
    TRACK_MEMBERS,
    NatureConfig,
    NatureV1,
    advect,
    consensus_track,
    describe_environment,
    fit_track_consensus,
    steering_track,
    storm_environment,
)
from naturev1.corpora import latlon_to_coords
from naturev1.era5 import Normalizer, equiangular_weights
from naturev1.losses import track_mixture_nll
from naturev1.steering import METRES_PER_DEGREE

from ihelix import FieldGrid, Geometry, fibonacci_sphere


LAT, LON = np.linspace(-90.0, 90.0, 37), np.arange(72) * 5.0          # a 5-degree globe
VARIABLES = ["u_component_of_wind@850", "v_component_of_wind@850", "u_component_of_wind@500",
             "v_component_of_wind@500", "u_component_of_wind@200", "v_component_of_wind@200",
             "specific_humidity@700", "2m_temperature"]
LEADS = (6, 12, 24)


def _grid():
    return FieldGrid.source(latlon_to_coords(LAT, LON), Geometry.globe(),
                            weights=torch.tensor(equiangular_weights(LAT, len(LON))))


def _normalizer():
    # Identity for the winds, so a physical wind of 10 m/s is a normalized 10; the two others carry an
    # offset and a scale, so decoding is exercised too.
    mean = np.array([0, 0, 0, 0, 0, 0, 0.004, 300.0], np.float32)
    std = np.array([1, 1, 1, 1, 1, 1, 0.002, 5.0], np.float32)
    return Normalizer(VARIABLES, mean, std, np.zeros(len(VARIABLES), np.float32))


def _state(u=(0.0, 0.0, 0.0), v=(0.0, 0.0, 0.0), q=0.004, t2m=300.0, batch=1):
    points = len(LAT) * len(LON)
    normalizer = _normalizer()
    state = torch.zeros(batch, points, len(VARIABLES))
    for level, (uu, vv) in enumerate(zip(u, v)):
        state[..., 2 * level] = uu
        state[..., 2 * level + 1] = vv
    state[..., 6] = float((q - normalizer.mean[6]) / normalizer.std[6])
    state[..., 7] = float((t2m - normalizer.mean[7]) / normalizer.std[7])
    return state, normalizer


def test_a_uniform_westerly_carries_the_storm_east_by_u_times_t():
    state, normalizer = _state(u=(10.0, 10.0, 10.0))
    centre = torch.tensor([[20.0, -60.0]])
    track = steering_track(state, VARIABLES, normalizer, _grid(), centre, LEADS, beta_drift_ms=None)
    for lead, hours in enumerate(LEADS):
        expected = 10.0 * hours * 3600 / (METRES_PER_DEGREE * math.cos(math.radians(20.0)))
        assert abs(float(track[0, lead, 1]) - expected) < 0.01 * expected, hours
        assert abs(float(track[0, lead, 0])) < 1e-6


def test_environment_reads_steering_shear_moisture_and_ocean_in_their_units():
    state, normalizer = _state(u=(5.0, 10.0, 20.0), v=(0.0, 0.0, 0.0), q=0.010, t2m=302.65)
    features = storm_environment(state, VARIABLES, normalizer, _grid(), torch.tensor([[15.0, -50.0]]))[0]
    named = dict(zip(STORM_ENVIRONMENT_FEATURES, features.tolist()))
    # Thickness-weighted mean of 5 (850), 10 (500), 20 (200) m/s: weights 175, 325, 150 of 650.
    assert abs(named["steering_u"] * 10 - (5 * 175 + 10 * 325 + 20 * 150) / 650) < 1e-3
    assert abs(named["shear_u"] * 10 - 15.0) < 1e-3 and abs(named["shear_speed"] * 10 - 15.0) < 1e-3
    assert abs(named["mid_level_humidity"] * 5 - 10.0) < 1e-3                     # g/kg
    assert abs(named["ocean_temperature"] * 3 - 3.0) < 1e-3                        # 3 K above 26.5 C
    assert named["observed"] == 1.0
    assert "shear 29 kt" in describe_environment(features)


def test_without_upper_winds_the_physics_stands_down_instead_of_inventing():
    state, normalizer = _state()
    variables = ["specific_humidity@700", "2m_temperature"]
    cut = state[..., 6:]
    norm = Normalizer(variables, normalizer.mean[6:], normalizer.std[6:], np.zeros(2, np.float32))
    centre = torch.tensor([[20.0, -60.0]])
    features = storm_environment(cut, variables, norm, _grid(), centre)[0]
    assert features[-1] == 0.0 and float(features[:5].abs().sum()) == 0.0
    track = steering_track(cut, variables, norm, _grid(), centre, LEADS)
    assert float(track.abs().sum()) == 0.0, "no wind: the storm is left where it is"


def test_beta_drift_is_poleward_in_both_hemispheres():
    state, normalizer = _state()
    centres = torch.tensor([[20.0, -60.0], [-20.0, 60.0]])
    track = steering_track(state.expand(2, -1, -1), VARIABLES, normalizer, _grid(), centres, (24,))
    assert float(track[0, 0, 0]) > 0 and float(track[1, 0, 0]) < 0
    assert float(track[0, 0, 1]) < 0 and float(track[1, 0, 1]) < 0                 # westward in both


def test_advection_wraps_the_dateline():
    moved = advect(torch.tensor([[10.0, 179.9]]), torch.tensor([10.0]), torch.tensor([0.0]), 6.0)
    expected = 179.9 + 10.0 * 6 * 3600 / (METRES_PER_DEGREE * math.cos(math.radians(10.0))) - 360.0
    assert abs(float(moved[0, 1]) - expected) < 0.01 and float(moved[0, 1]) < 0


def _model():
    config = NatureConfig(analysis_channels=8, hidden_size=32, num_layers=1, num_heads=2, num_kv_heads=1,
                          head_dim=16, intermediate_size=64, latent_points=48, history_frames=2,
                          lead_times_hours=LEADS, track_modes=2)
    torch.manual_seed(0)
    return NatureV1(config, fibonacci_sphere(48, num_neighbours=8, cluster_size=12))


def test_the_track_head_is_a_residual_on_the_prior():
    model = _model().eval()
    pooled, centre = torch.randn(3, 32), torch.tensor([[20.0, -60.0]] * 3)
    prior = torch.randn(3, len(LEADS), 2)
    plain = model.storm_heads(pooled, centre)
    with_prior = model.storm_heads(pooled, centre, track_prior=prior)
    assert torch.allclose(with_prior["displacement"] - prior.unsqueeze(1), plain["displacement"], atol=1e-6)
    assert torch.equal(with_prior["track_prior"], prior)
    assert not model.uses_track_prior()
    model.mark_trained({"track": 1.0}, {"track_prior": prior})
    assert model.uses_track_prior()


def test_a_checkpoint_from_before_the_physics_loads_and_behaves_the_same():
    old = _model().eval()
    state = {k: v for k, v in old.state_dict().items()
             if not k.startswith(("storm_environment_embed.", "storm_far_pool.", "track_consensus"))}
    new = _model().eval()
    new.load_state_dict(state)
    pooled, far = torch.randn(2, 32), torch.randn(2, 32)
    centre, storm_state = torch.tensor([[20.0, -60.0]] * 2), torch.randn(2, 6)
    environment = torch.randn(2, len(STORM_ENVIRONMENT_FEATURES))
    before = old.storm_heads(pooled, centre, storm_state)
    after = new.storm_heads(pooled, centre, storm_state, environment, None, far)
    for key in ("displacement", "intensity_mean", "ri_logits", "landfall_logit"):
        assert torch.allclose(before[key], after[key], atol=1e-6), key
    assert float(new.track_consensus.abs().sum()) == 0.0, "no consensus until one is fitted"


def test_the_consensus_finds_the_member_that_is_right():
    torch.manual_seed(1)
    count, leads = 200, len(LEADS)
    truth = torch.randn(count, leads, 2)
    members = torch.stack([truth + torch.randn(count, leads, 2) * 0.5, truth + torch.randn(count, leads, 2) * 0.3,
                           truth.clone(), truth + torch.randn(count, leads, 2)], 2)
    valid = torch.ones(count, leads, dtype=torch.bool)
    latitude = torch.full((count,), 25.0)
    weights = fit_track_consensus(members, truth, valid, latitude)
    assert weights.shape == (leads, len(TRACK_MEMBERS))
    assert torch.allclose(weights.sum(-1), torch.ones(leads), atol=1e-4) and (weights >= 0).all()
    assert (weights[:, 2] > 0.97).all(), weights
    blended = consensus_track(members, weights)
    assert float((blended - truth).abs().mean()) < 0.02


def test_samples_with_nothing_to_verify_do_not_dilute_the_track_loss():
    logits, disp = torch.zeros(2, 2), torch.zeros(2, 2, 3, 2)
    scale, rho = torch.zeros(2, 2, 3, 2), torch.zeros(2, 2, 3)
    target = torch.ones(2, 3, 2)
    valid = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    both = track_mixture_nll(logits, disp, scale, rho, target, valid)
    one = track_mixture_nll(logits[:1], disp[:1], scale[:1], rho[:1], target[:1], valid[:1])
    assert torch.allclose(both, one)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_bank_carries_the_physics_and_the_scorecard_shows_every_member(tmp_path):
    from naturev1 import (
        ERA5Window,
        pair_tracks_with_reanalysis,
        storm_feature_bank,
        storm_scorecard,
        train_storm_heads,
    )
    from test_fixes_080 import WIDTH, _store, _storm
    from test_fixes_080 import _grid as _coarse_grid
    from test_fixes_080 import _model as _bank_model

    store = _store()
    variables = ("2m_temperature", "surface_pressure", "toa_incident_solar_radiation_6hr", "land_sea_mask",
                 "temperature", "geopotential")
    tracks = [_storm("AL012005", 24, 20, 15.0), _storm("AL022005", 48, 16, 22.0)]
    times = store.time.values.astype("datetime64[s]").astype(np.int64)
    starts, targets, _ = pair_tracks_with_reanalysis(tracks, times, (1, 2), history=2)
    base = ERA5Window(store, indices=starts[:1], history=2, lead_steps=1, variables=variables, levels=[500],
                      channels=WIDTH, augment=False)
    model, grid = _bank_model(), _coarse_grid()
    model.config.lead_times_hours = (6, 12)
    bank = storm_feature_bank(model, store, base, starts, targets, grid, cache=str(tmp_path / "bank.pt"),
                              precision="fp32", progress=False)
    assert bank["features_far"].shape == bank["features"].shape
    assert bank["targets"]["storm_environment"].shape == (len(starts), len(STORM_ENVIRONMENT_FEATURES))
    assert bank["targets"]["track_prior"].shape == (len(starts), 2, 2)
    assert (bank["targets"]["storm_environment"][:, -1] == 0).all(), "this store has no winds: physics stands down"
    assert set(bank["targets"]["season"].tolist()) == {2005}

    train_storm_heads(model, bank, bank, epochs=3, batch_size=8, patience=10, log_every=100)
    assert model.uses_track_prior()
    assert torch.allclose(model.track_consensus.sum(-1), torch.ones(2), atol=1e-4), "weights fitted and saved"
    card = storm_scorecard(model, bank, (6, 12))
    assert "steering" in card and "consensus" in card and "one season" in card
