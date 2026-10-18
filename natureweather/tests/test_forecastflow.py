# Copyright 2026 Nathan. Apache-2.0.
"""Forecast-flow steering (the turns) and the intensity trend / rapid-intensification pieces, on constructed storms."""

from __future__ import annotations

import numpy as np

from naturev1.cyclone import geometry as geo
from naturev1.cyclone.data import HISTORY_HOURS, LEADS
from naturev1.cyclone.forecastflow import ForecastFlow, fflow_member
from naturev1.cyclone.intensity import fit_ri, recent_change, ri_features, ri_probability, trend_intensity
from naturev1.cyclone.members import build_context
from naturev1.swarm.core import Swarm, SwarmFields

from test_cyclone import _case, _vortex_env

HOURS = np.arange(0, 121, 12)


def _flow(env, turn_hour=None, before=(-5.0, 0.0), after=(0.0, 6.0), layers=None):
    """Uniform layer winds that switch from ``before`` to ``after`` at ``turn_hour`` (all layers equal unless given)."""
    shape = (env.lat.size, env.lon.size)
    data = np.zeros((HOURS.size, 6) + shape, np.float32)
    for k, h in enumerate(HOURS):
        for j in range(3):
            u, v = layers[j] if layers is not None else (after if turn_hour is not None and h >= turn_hour else before)
            data[k, 2 * j], data[k, 2 * j + 1] = u, v
    return ForecastFlow(env.lat, env.lon, HOURS, data, "test")


def test_layers_blend_by_steering_depth_and_interpolate_in_time():
    env = _vortex_env()
    flow = _flow(env, layers=[(-2.0, 0.0), (-6.0, 1.0), (-10.0, 2.0)])
    lat, lon = np.array([20.0]), np.array([-60.0])
    assert np.allclose(flow.velocity(env, lat, lon, 30.0, depth=0.0), ([-2.0], [0.0]))
    assert np.allclose(flow.velocity(env, lat, lon, 30.0, depth=0.5), ([-6.0], [1.0]))
    assert np.allclose(flow.velocity(env, lat, lon, 30.0, depth=1.0), ([-10.0], [2.0]))
    assert np.allclose(flow.velocity(env, lat, lon, 30.0, depth=0.25), ([-4.0], [0.5]))
    turning = _flow(env, turn_hour=60)
    u, v = turning.field(54.0)                                     # halfway between the 48 h and 60 h fields
    assert np.allclose(u, -2.5) and np.allclose(v, 3.0)
    assert np.allclose(turning.field(500.0)[1], 6.0), "beyond the last hour: held, not extrapolated"


def test_the_member_turns_when_the_forecast_flow_turns():
    # A storm going west under an easterly that the model forecasts to give way to a southerly after 48 hours (a
    # trough digging in): steered by the analysis it keeps going west; steered by the forecast it turns north.
    case = _case(motion_kmh=(-18.0, 0.0))
    env = _vortex_env(u=-5.0, v=0.0)
    ctx = build_context(case, env)
    straight = fflow_member(ctx, _flow(env), LEADS)
    turning = fflow_member(ctx, _flow(env, turn_hour=48), LEADS)
    k36, k48, k120 = LEADS.index(36), LEADS.index(48), LEADS.index(120)
    # the same until the flow starts to change (between the 36 h and 48 h forecast fields)
    assert geo.great_circle_km(*straight[k36], *turning[k36]) < 30.0
    north = geo.local_displacement_km(*straight[k120], *turning[k120])[1]
    assert north > 800.0, f"turned only {north:.0f} km north"
    west_after = geo.local_displacement_km(*turning[k48], *turning[k120])[0]
    assert west_after > -700.0, "and stopped going west"


def test_the_swarm_worlds_follow_the_forecast_flow():
    case = _case(motion_kmh=(-18.0, 0.0))
    env = _vortex_env(u=-5.0, v=0.0)
    ctx = build_context(case, env)
    k120 = LEADS.index(120)
    tracks = {}
    for name, flow in (("straight", _flow(env)), ("turning", _flow(env, turn_hour=48))):
        fields = SwarmFields.from_context(ctx, forecast=flow)
        sw = Swarm.from_context(case, ctx, 3000, seed=0, fields=fields)
        sw.assimilate_history(case)
        tracks[name] = sw.forecast(LEADS, case.lat, case.lon).mean_track()
    north = geo.local_displacement_km(*tracks["straight"][k120], *tracks["turning"][k120])[1]
    assert north > 600.0, f"the swarm turned only {north:.0f} km north"


def _along(pi=150.0, land=None, shear=5.0, rh=0.7, sst=302.5):
    hours = np.arange(0.0, 121.0, 1.0)
    six = np.arange(0.0, 121.0, 6.0)
    return {"hours": hours, "six_hours": six, "pi_kt": np.full(six.size, pi),
            "land": np.zeros(hours.size) if land is None else land, "height_m": np.zeros(hours.size),
            "shear_ms": np.full(six.size, shear), "rh_mid": np.full(six.size, rh), "sst_eff_k": np.full(hours.size, sst)}


def _intensifying(rise_kt=40.0, wind=110.0):
    """A storm at ``wind`` kt that has gained ``rise_kt`` over the last 24 hours, steadily."""
    case = _case(wind=wind)
    case.history["wind"] = np.array([wind + rise_kt * h / 24.0 for h in HISTORY_HOURS])
    return case


def test_the_trend_member_follows_rapid_intensification_up_to_the_potential_intensity():
    case = _intensifying(rise_kt=40.0, wind=110.0)
    assert recent_change(case, 24) == 40.0
    path = trend_intensity(case, _along(pi=150.0), LEADS, tau_h=24.0)
    k24, k120 = LEADS.index(24), LEADS.index(120)
    assert path[k24] > 130.0, "keeps intensifying"
    assert np.all(path <= 150.0 + 1e-9), "never above the potential intensity"
    landfall = np.where(np.arange(0.0, 121.0) >= 24.0, 1.0, 0.0)
    ashore = trend_intensity(case, _along(pi=150.0, land=landfall), LEADS, tau_h=24.0)
    assert ashore[k120] < 0.6 * path[k120], "and decays over land"


def test_rapid_intensification_probability_ranks_the_favourable_storm_higher():
    rng = np.random.default_rng(0)
    cases, alongs = [], []
    for _ in range(400):                         # synthetic truth: RI happens in low shear over warm water
        shear, sst = rng.uniform(2, 20), rng.uniform(299, 304)
        favourable = shear < 8 and sst > 302
        case = _intensifying(rise_kt=rng.uniform(-5, 25), wind=rng.uniform(40, 100))
        k24 = LEADS.index(24)
        case.truth["wind"] = np.full(len(LEADS), case.wind_kt) + (40.0 if favourable and rng.random() < 0.8 else 5.0) * (np.arange(len(LEADS)) >= k24)
        cases.append(case)
        alongs.append(_along(shear=shear, sst=sst))
    table = fit_ri(cases, alongs)
    good = ri_probability(cases[0], _along(shear=4.0, sst=303.5), table)
    bad = ri_probability(cases[0], _along(shear=18.0, sst=299.5), table)
    assert 0.0 <= bad < good <= 1.0 and good > 0.5 > bad
    assert ri_features(cases[0], _along()).shape == (len(table["features"]),)


def test_gfs_forecast_flow_tracks_the_model_storm_and_keeps_the_environment(monkeypatch, tmp_path):
    # Synthetic GFS forecasts (no network): a uniform easterly-plus-southerly flow at every level and a pressure low
    # moving north-west 1 degree every 12 hours. The reader must follow GFS's low and hand back the flow.
    from naturev1.cyclone.data import PREVIOUS_LEVELS
    from naturev1.swarm import analysis

    lat2, lon2 = np.meshgrid(analysis.LAT_AXIS, analysis.LON_AXIS, indexing="ij")

    def fake(self, when, lead):
        clat, clon = 20.0 + lead / 12.0, (300.0 - lead / 12.0)
        r2 = ((lat2 - clat) ** 2 + ((lon2 - clon) * np.cos(np.radians(clat))) ** 2)
        shape = (len(PREVIOUS_LEVELS),) + lat2.shape
        return {"u": np.full(shape, -4.0), "v": np.full(shape, 2.0), "msl": 101000.0 - 3000.0 * np.exp(-r2 / 4.0)}

    monkeypatch.setattr(analysis.GFSAnalysis, "forecast_fields", fake)
    case = _case(lat=20.0, lon=-60.0, motion_kmh=(-9.0, 9.0))
    flow, track = analysis.GFSAnalysis(tmp_path).forecast_flow("2025-10-27T00", case)
    assert flow.data.shape[:2] == (11, 6)
    for k, h in enumerate(LEADS):
        assert geo.great_circle_km(*track[k], 20.0 + h / 12.0, -60.0 - h / 12.0) < 120.0, f"lost GFS's storm at {h} h"
    u, v = flow.velocity(_vortex_env(), np.array([25.0]), np.array([-65.0]), 60.0, depth=0.5)
    assert abs(u[0] + 4.0) < 0.3 and abs(v[0] - 2.0) < 0.3, "the environment survives the vortex removal"


def test_physics_forecast_uses_the_forecast_flow_consensus_and_leans_on_the_model_late():
    from naturev1.cyclone.forecast import physics_forecast

    case = _case(motion_kmh=(-18.0, 0.0))
    env = _vortex_env(u=-5.0, v=0.0)
    flow = _flow(env, turn_hour=48)
    model = np.stack(geo.move(case.lat, case.lon, -18.0 * np.minimum(np.asarray(LEADS, float), 48.0),
                              20.0 * np.maximum(np.asarray(LEADS, float) - 48.0, 0.0)), -1)     # west, then north
    plain = physics_forecast(case, env, intensity=False)
    steered = physics_forecast(case, env, forecast_flow=flow, model_track=model, intensity=False)
    assert plain["consensus_table"] == "CONSENSUS_BASIC" and steered["consensus_table"] == "CONSENSUS_BASIC_FF"
    k120 = LEADS.index(120)
    assert geo.great_circle_km(*steered["track"][k120], *model[k120]) < 0.3 * geo.great_circle_km(*plain["track"][k120], *model[k120])
    assert np.isfinite(steered["members"]["fflow"]).all()


def test_model_deepening_feeds_the_intensity_member_and_the_ri_probability():
    from naturev1.cyclone.intensity import (RI_FEATURES, category_probabilities, model_deficit_change,
                                            model_trend_intensity, ri_probability)

    hours = np.array([0, 6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120], float)
    deep = {"hours": hours, "msl_min_hpa": 1000.0 - 1.0 * hours, "msl_env_hpa": np.full(hours.size, 1012.0)}   # 24 hPa by 24 h
    flat = {"hours": hours, "msl_min_hpa": np.full(hours.size, 1000.0), "msl_env_hpa": np.full(hours.size, 1012.0)}
    dp = model_deficit_change({"a": {"extras": deep}, "b": flat}, LEADS)
    k24 = LEADS.index(24)
    assert abs(dp[k24] + 12.0) < 1e-9, "the mean of a 24 hPa deepening and none"
    assert model_deficit_change(None, LEADS) is None
    case = _intensifying(rise_kt=10.0, wind=80.0)
    table = {"leads": list(LEADS), "kt_per_hpa": [1.5] * len(LEADS)}
    v = model_trend_intensity(case, _along(pi=150.0), LEADS, dp, table)
    assert abs(v[k24] - (80.0 + 1.5 * 12.0)) < 1e-6 and np.all(v <= 150.0 + 1e-9)
    weights = [0.0] * len(RI_FEATURES) + [2.0, 0.0, 0.0]
    weights[0] = -2.0
    ri = {"weights": weights, "base_rate": 0.1}
    with_model = ri_probability(case, _along(), ri, model_dp=dp)
    without = ri_probability(case, _along(), ri)
    assert without < 0.2 and with_model > without, "a deepening model raises P(RI); no model counts as none"
    spread = {"quantiles": {"normal": [[-10.0, 0.0, 10.0]] * len(LEADS), "rapid": [[0.0, 10.0, 20.0]] * len(LEADS)}, "thresholds_kt": [64, 96]}
    normal = category_probabilities(np.full(len(LEADS), 90.0), "normal", spread)["probability"]
    rapid = category_probabilities(np.full(len(LEADS), 90.0), "rapid", spread)["probability"]
    assert normal.shape == (len(LEADS), 2) and abs(normal[0, 1] - 1 / 3) < 1e-9 and abs(rapid[0, 1] - 2 / 3) < 1e-9
