# Copyright 2026 Nathan. Apache-2.0.
"""naturev1.cyclone.swarm: the particle-filter sandbox, checked on constructed storms where the answer is known."""

from __future__ import annotations

import numpy as np
import torch

from naturev1.cyclone import geometry as geo
from naturev1.cyclone.data import LEADS
from naturev1.cyclone.members import build_context
from naturev1.cyclone.swarm import SPEED_CAP_MS, Swarm, SwarmFields, SwarmPriors, classify_tracks

from test_cyclone import _case, _vortex_env


def _swarm(case, env, particles=4000, seed=0, **priors):
    ctx = build_context(case, env)
    return Swarm.from_context(case, ctx, particles, priors=SwarmPriors(**priors) if priors else None, seed=seed)


def _error_km(track, case, k):
    return float(geo.great_circle_km(track[k, 0], track[k, 1], case.truth["lat"][k], case.truth["lon"][k]))


def test_the_fixes_teach_the_swarm_how_the_storm_is_really_moving():
    # The environment steers at (-5, 0) m/s, but the storm has been moving at (-18, 4) km/h = (-5.0, 1.1) m/s with
    # beta drift on top of that: the physics alone is wrong, and the fixes are the only way to know by how much.
    case, env = _case(motion_kmh=(-18.0, 4.0)), _vortex_env(u=-5.0, v=0.0)
    filtered = _swarm(case, env)
    filtered.assimilate_history(case)
    blind = _swarm(case, env)                       # launched at the latest fix with nothing learned
    blind.hours = 0.0
    k24, k72 = LEADS.index(24), LEADS.index(72)
    learned = filtered.forecast(LEADS, case.lat, case.lon).mean_track()
    prior = blind.forecast(LEADS, case.lat, case.lon).mean_track()
    assert _error_km(learned, case, k24) < 0.7 * _error_km(prior, case, k24)
    assert _error_km(learned, case, k72) < _error_km(prior, case, k72)
    assert len(filtered.history) == 4 and all(0.0 < h["ess"] <= 1.0 for h in filtered.history)


def test_resampling_keeps_the_swarm_normalised_and_diverse():
    case, env = _case(), _vortex_env()
    sw = _swarm(case, env, particles=2000)
    sw.advance_to(-12.0)
    ess = sw.update(float(sw.lat[0]), float(sw.lon[0]), sigma_km=5.0)       # a fix only a few worlds agree with
    assert ess < 0.5
    w = torch.exp(sw.logw)
    assert abs(float(w.sum()) - 1.0) < 1e-9 and torch.allclose(w, w[0].expand_as(w))
    assert float(sw.params["depth"].std()) > 0.0 and float(sw.params["static"].std()) > 0.0, "jittered, not cloned"


def test_guardrails_cap_speed_and_keep_perturbations_non_divergent():
    case, env = _case(), _vortex_env()
    sw = _swarm(case, env, particles=500)
    sw.params["static"] = sw.params["static"] * 200.0                       # absurd steering errors
    u, v = sw.velocity(sw.lat, sw.lon, 24.0)
    assert float(torch.hypot(u, v).max()) <= SPEED_CAP_MS + 1e-6
    f = sw.fields
    k = f.n_modes
    mode_u = f.grid[0, 12:12 + k].double().numpy()
    mode_v = f.grid[0, 12 + k:12 + 2 * k].double().numpy()
    cos = np.cos(np.radians(f.lat))[None, :, None]
    div = (np.gradient(mode_u, axis=2) + np.gradient(mode_v * cos, axis=1)) / cos         # spherical, per grid step
    vort = (np.gradient(mode_v, axis=2) - np.gradient(mode_u * cos, axis=1)) / cos
    assert np.abs(div).mean() < 0.02 * np.abs(vort).mean(), "streamfunction modes carry no divergence"


def test_streamed_swarm_agrees_with_one_batch_and_stays_bounded():
    case, env = _case(), _vortex_env()
    sw = _swarm(case, env, particles=3000, seed=1)
    sw.assimilate_history(case)
    one = sw.forecast(LEADS, case.lat, case.lon, particles=6000)
    streamed = sw.stream(6000, LEADS, case.lat, case.lon, batch=1500)
    k72 = LEADS.index(72)
    gap = geo.great_circle_km(*one.mean_track()[k72], *streamed["mean_track"][k72])
    assert gap < 60.0 and streamed["particles"] == 6000
    assert np.all((streamed["strike_probability"] >= 0) & (streamed["strike_probability"] <= 1 + 1e-9))
    assert abs(streamed["cone_km"][k72] - one.cone_km()[k72]) < 0.25 * one.cone_km()[k72] + 60.0
    assert abs(sum(streamed["scenarios"].values()) - 1.0) < 1e-9


def test_track_classification():
    hours = np.arange(0.0, 121.0, 3.0)
    n = hours.size
    still = np.stack([np.full(n, 25.0), np.full(n, -75.0)], -1)
    west = np.stack([np.full(n, 15.0), -50.0 - 0.15 * hours], -1)
    lat_r = 20.0 + 0.12 * hours
    lon_r = np.where(hours < 60, -60.0 - 0.15 * hours, -69.0 + 0.2 * (hours - 60))
    recurve = np.stack([lat_r, lon_r], -1)
    ang = np.radians(hours * 3.0)
    loop = np.stack([20.0 + 1.5 * np.sin(ang), -60.0 - 1.5 * (1 - np.cos(ang)) - 0.01 * hours], -1)
    labels = classify_tracks(np.stack([still, west, recurve, loop]), hours)
    assert list(labels) == ["stall", "steady", "recurve", "loop"]


def test_strike_probability_marks_the_corridor():
    case, env = _case(), _vortex_env()
    sw = _swarm(case, env, particles=400, seed=2, static_ms=0.01, growth_ms_per_sqrt_h=0.0, residual_ms=0.01,
                launch_sigma_km=0.1, layer_sd=0.0, beta_log_sd=0.0, beta_angle_sd_deg=0.0)      # one world, 400 times
    sw.hours = 0.0
    fc = sw.forecast(LEADS, case.lat, case.lon)
    mean = fc.mean_track()
    glat = np.arange(5.0, 35.0, 0.5)
    glon = np.arange(-100.0, -40.0, 0.5)
    p = fc.strike_probability(glat, glon, radius_km=111.0, until_h=72)
    i = int(np.argmin(np.abs(glat - mean[LEADS.index(48), 0])))
    j = int(np.argmin(np.abs(glon - mean[LEADS.index(48), 1])))
    assert p[i, j] > 0.9, "on the track"
    assert p[0, 0] == 0.0 and p.max() <= 1.0 + 1e-9, "far away and bounded"


def test_fields_round_trip_box_longitudes():
    case, env = _case(lon=178.0), _vortex_env(lon0=178.0)
    ctx = build_context(case, env)
    f = SwarmFields.from_context(ctx)
    assert f.lon[0] <= f.to_box(-179.0) <= f.lon[-1], "a point east of the dateline lands inside the box"


def test_physics_forecast_carries_a_swarm_centred_on_the_consensus():
    from naturev1.cyclone.forecast import physics_forecast
    from naturev1.cyclone.synthetic import scenario_inputs, scenario_library

    case, env = scenario_inputs(scenario_library()["dorian_stall"], LEADS)
    out = physics_forecast(case, env, intensity=False, swarm=3000)
    sw = out["swarm"]
    centred = sw["forecast"].mean_track()
    ok = np.isfinite(out["track"][:, 0])
    gap = geo.great_circle_km(centred[ok, 0], centred[ok, 1], out["track"][ok, 0], out["track"][ok, 1])
    assert np.all(gap < 5.0), "the swarm is re-centred on the consensus track"
    assert abs(sum(v["probability"] for v in sw["scenarios"].values()) - 1.0) < 1e-9
    assert sw["strike_probability"].max() <= 1.0 + 1e-9 and sw["strike_probability"].max() > 0.5
    assert np.all(np.diff(sw["cone_km"][1:]) > 0), "the cone widens with lead"
