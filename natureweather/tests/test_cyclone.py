# Copyright 2026 Nathan. Apache-2.0.
"""
naturev1.cyclone: every module checked on constructed atmospheres and storms where the right answer is known.
All offline and fast; the hindcast numbers themselves live in the benchmark, not here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from naturev1.cyclone import geometry as geo
from naturev1.cyclone.data import HISTORY_HOURS, LEADS, Case
from naturev1.cyclone.fields import Environment, synthetic_environment


def _case(lat=20.0, lon=-60.0, motion_kmh=(-18.0, 4.0), wind=80.0, basin="NA", status="HU", hours=HISTORY_HOURS,
          truth=None, time="2019-09-01T12"):
    hist = {"lat": [], "lon": [], "wind": [], "pressure": []}
    for h in hours:
        la, lo = geo.move(lat, lon, motion_kmh[0] * h, motion_kmh[1] * h)
        hist["lat"].append(float(la))
        hist["lon"].append(float(lo))
        hist["wind"].append(wind)
        hist["pressure"].append(970.0)
    if truth is None:
        tl, to = geo.move(lat, lon, motion_kmh[0] * np.asarray(LEADS, float), motion_kmh[1] * np.asarray(LEADS, float))
        truth = {"lat": tl, "lon": to, "wind": np.full(len(LEADS), wind), "pressure": np.full(len(LEADS), np.nan),
                 "valid": np.ones(len(LEADS), bool)}
    return Case("TEST01", "TEST", basin, np.datetime64(time, "s"), lat, lon, wind, 970.0, status,
                {k: np.asarray(v, float) for k, v in hist.items()}, truth=truth)


def _vortex_env(lat0=20.0, lon0=-60.0, u=-5.0, v=0.0, vmax=15.0, rmax_km=150.0, centre=None, alpha=0.55):
    """Uniform flow plus a modified-Rankine vortex at every level (cyclonic per hemisphere)."""
    from naturev1.cyclone.bogus import rankine_profile

    clat, clon = centre or (lat0, lon0)
    sign = 1.0 if clat >= 0 else -1.0

    def uf(la, lo, p):
        x, y = geo.to_local(clat, clon, la, lo)
        r = np.maximum(np.hypot(x, y), 1.0)
        vt = sign * rankine_profile(r, vmax, rmax_km, alpha)
        return u - vt * y / r

    def vf(la, lo, p):
        x, y = geo.to_local(clat, clon, la, lo)
        r = np.maximum(np.hypot(x, y), 1.0)
        vt = sign * rankine_profile(r, vmax, rmax_km, alpha)
        return v + vt * x / r

    return synthetic_environment(lat0, lon0, u=uf, v=vf)


# ------------------------------------------------------------------------------------------ geometry --

def test_great_circle_moves_are_exact_and_reversible():
    lat, lon = geo.destination(25.0, 179.0, 90.0, 500.0)
    assert -180.0 <= float(lon) < 180.0 and float(lon) < 0, "crossed the dateline cleanly"
    assert geo.great_circle_km(25.0, 179.0, lat, lon) == pytest.approx(500.0, rel=1e-9)
    e, n = geo.local_displacement_km(25.0, 179.0, lat, lon)
    assert math.hypot(e, n) == pytest.approx(500.0, rel=1e-9) and e > 0
    la, lo = geo.from_local(10.0, 20.0, 300.0, -400.0)
    x, y = geo.to_local(10.0, 20.0, la, lo)
    assert (float(x), float(y)) == pytest.approx((300.0, -400.0), abs=1e-6)


def test_along_and_cross_track_follow_the_nhc_convention():
    # Storm moves due north 500 km; forecast 100 km further north and 50 km east (to the right).
    tl, to = geo.destination(20.0, -60.0, 0.0, 500.0)
    fl, fo = geo.move(20.0, -60.0, 50.0, 600.0)
    along, cross = geo.along_cross_track(20.0, -60.0, tl, to, fl, fo)
    assert float(along) == pytest.approx(100.0, abs=3.0) and float(cross) == pytest.approx(50.0, abs=3.0)


def test_track_interpolation_follows_great_circles():
    lat, lon = geo.interpolate_track([0, 6], [20.0, 21.0], [-60.0, -62.0], [3.0])
    mid = geo.destination(20.0, -60.0, geo.bearing_deg(20, -60, 21, -62), 0.5 * geo.great_circle_km(20, -60, 21, -62))
    assert (float(lat[0]), float(lon[0])) == pytest.approx((float(mid[0]), float(mid[1])), abs=1e-9)


# -------------------------------------------------------------------------------------------- fields --

def test_spherical_vorticity_and_layer_means():
    env = synthetic_environment(20.0, -60.0, u=lambda la, lo, p: np.full_like(la, 10.0) * (p / 1000.0), v=0.0)
    # Pressure-weighted 850-200 mean of u = 10 p/1000: (10/1000) * mean p over [200, 850] = 5.25.
    assert float(env.layer_mean("u", 200, 850)[20, 30]) == pytest.approx(5.25, rel=1e-3)
    assert float(env.level("u", 775)[20, 30]) == pytest.approx(7.75, abs=0.05), "log-pressure interpolation"
    rot = _vortex_env(u=0.0, vmax=20.0)
    zeta = rot.vorticity(rot.fields["u"][2], rot.fields["v"][2])
    i, j = np.unravel_index(np.argmax(zeta), zeta.shape)
    assert abs(rot.lat[i] - 20.0) <= 1.5 and float(zeta.max()) > 1e-4, "cyclonic maximum at the centre"
    assert float(rot.sample(rot.fields["u"][0], 20.0, -60.0 + 360.0)) == pytest.approx(float(rot.sample(rot.fields["u"][0], 20.0, -60.0)))


# ---------------------------------------------------------------------------------- vortex & steering --

def test_vortex_removal_recovers_the_environment_around_the_storm():
    from naturev1.cyclone.steering import disc_mean
    from naturev1.cyclone.vortex import remove_vortex

    # A Rankine vortex (irrotational outside its core): all of its vorticity is inside the removal radius.
    env = _vortex_env(u=-5.0, v=1.0, vmax=25.0, rmax_km=150.0, alpha=1.0)
    split = remove_vortex(env, 20.0, -60.0)
    clean = split.environment
    for lat, lon in ((20.0, -60.0), (20.0, -57.0), (22.0, -61.0)):
        raw = np.array([disc_mean(env, env.fields[c][2], lat, lon, 300.0)[0] for c in ("u", "v")]) - (-5.0, 1.0)
        left = np.array([disc_mean(clean, clean.fields[c][2], lat, lon, 300.0)[0] for c in ("u", "v")]) - (-5.0, 1.0)
        # At the centre the storm cancels anyway; off-centre, at least three quarters of its wind is removed.
        assert np.hypot(*left) <= max(0.25 * np.hypot(*raw), 0.2), (lat, lon, raw, left)
    profile = split.tangential_profile(2)
    assert np.nanmax(profile) > 5.0, "the removed vortex is the cyclonic circulation"


def test_steering_layers_and_optimal_layer():
    from naturev1.cyclone.steering import SteeringFlow, intensity_layer, optimal_layer

    env = synthetic_environment(20.0, -60.0, u=lambda la, lo, p: np.where(p >= 500, -8.0, 4.0) + 0 * la, v=0.0)
    deep = SteeringFlow(env, 200, 850).velocity(20.0, -60.0)[0][0]
    shallow = SteeringFlow(env, 700, 850).velocity(20.0, -60.0)[0][0]
    assert shallow == pytest.approx(-8.0, abs=1e-6) and -8.0 < deep < 0.0
    best = optimal_layer(env, 20.0, -60.0, (-8.0, 0.0))
    assert best["top"] >= 500 and best["mismatch_ms"] < 0.5
    assert intensity_layer(30.0)[0] > intensity_layer(140.0)[0], "strong storms are steered by a deeper layer"


def test_beta_drift_is_poleward_and_westward_in_both_hemispheres_and_grows_with_size():
    from naturev1.cyclone.beta import BetaDrift

    for lat, poleward in ((20.0, 1.0), (-20.0, -1.0)):
        env = synthetic_environment(lat, 100.0)
        small, big = BetaDrift(env, 2.0), BetaDrift(env, 10.0)
        u, v = small.velocity(lat, 100.0)
        U, V = big.velocity(lat, 100.0)
        u, v, U, V = float(u[0]), float(v[0]), float(U[0]), float(V[0])
        assert u < 0 and v * poleward > 0, "west and poleward"
        assert math.hypot(U, V) > 1.8 * math.hypot(u, v), "outer strength sets the speed"


def test_trajectory_integration_and_decaying_residual():
    from naturev1.cyclone.trajectory import DecayingResidual, combined, integrate

    flow = lambda la, lo, h: (np.full(np.shape(la), -5.0), np.zeros(np.shape(la)))
    track = integrate(20.0, -60.0, flow, [24, 48])
    assert geo.great_circle_km(20.0, -60.0, track[0, 0], track[0, 1]) == pytest.approx(432.0, rel=1e-3)
    res = DecayingResidual(0.0, 2.0, 12.0)
    moved = integrate(20.0, -60.0, combined(res), [120])
    expected = 2.0 * 12.0 * 3600.0 * (1 - math.exp(-10.0)) / 1000.0
    assert geo.great_circle_km(20.0, -60.0, moved[0, 0], moved[0, 1]) == pytest.approx(expected, rel=2e-2)
    members = integrate(np.array([10.0, 30.0]), np.array([0.0, 0.0]), flow, [12])
    assert members.shape == (2, 1, 2)


def test_barotropic_vortex_drifts_with_beta_and_the_flow():
    from naturev1.cyclone.barotropic import BarotropicModel, PoissonSolver, dst1

    a = np.random.default_rng(0).standard_normal((6, 5))
    assert np.allclose(dst1(dst1(a, 0), 0) * 2 / 7, a)
    n, h = 21, 1000.0
    x = np.arange(n) * h
    X, Y = np.meshgrid(x, x)
    psi = np.sin(np.pi * X / x[-1]) * np.sin(np.pi * Y / x[-1]) + 1e-3 * X
    lap = np.zeros_like(psi)
    lap[1:-1, 1:-1] = (psi[1:-1, 2:] + psi[1:-1, :-2] + psi[2:, 1:-1] + psi[:-2, 1:-1] - 4 * psi[1:-1, 1:-1]) / h ** 2
    assert np.abs(PoissonSolver(n, h).solve(lap, psi) - psi).max() < 1e-9
    for lat, poleward in ((20.0, 1.0), (-20.0, -1.0)):
        out = BarotropicModel(synthetic_environment(lat, -60.0), lat, -60.0, 6.0, cells=49).run([48])["track"]
        e, nth = geo.local_displacement_km(lat, -60.0, out[0, 0], out[0, 1])
        assert e < -50 and nth * poleward > 150, (lat, e, nth)


def test_bogus_profile_matches_the_outer_strength():
    from naturev1.cyclone.bogus import alpha_for_size, rankine_profile, vmax_for_outer_strength

    vm = vmax_for_outer_strength(6.0, 150.0)
    radii = np.array([300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
    assert rankine_profile(radii, vm, 150.0).mean() == pytest.approx(6.0, rel=1e-6)
    assert alpha_for_size(250.0) < alpha_for_size(60.0)


# -------------------------------------------------------------------------------- Dorian-type regimes --

def test_a_col_is_found_and_flagged_as_a_stall():
    from naturev1.cyclone.stall import critical_points, stall_diagnostics, timing_spread
    from naturev1.cyclone.steering import SteeringFlow

    # A saddle at the storm: u = a x, v = -a y (hyperbolic flow), zero at (20 N, 60 W).
    def uf(la, lo, p):
        x, _ = geo.to_local(20.0, -60.0, la, lo)
        return 4.0 * x / 1000.0

    def vf(la, lo, p):
        _, y = geo.to_local(20.0, -60.0, la, lo)
        return -4.0 * y / 1000.0

    env = synthetic_environment(20.0, -60.0, u=uf, v=vf)
    flow = SteeringFlow(env, 400, 850)
    points = critical_points(env, flow.u, flow.v, 20.0, -60.0)
    saddle = [p for p in points if p["kind"] == "saddle"]
    assert saddle and saddle[0]["distance_km"] < 120.0
    track = np.tile([[20.0, -60.0]], (len(LEADS), 1))
    diag = stall_diagnostics(flow, track, LEADS, 20.0, -60.0)
    assert diag["stall_probability"] > 0.6 and diag["min_speed_ms"] < 1.0
    uniform = lambda la, lo, h: (np.full(np.shape(la), -5.0), np.zeros(np.shape(la)))
    spread = timing_spread(20.0, -60.0, uniform, [48])
    assert spread[0] == pytest.approx(2 * 0.45 * 48 * 3.6, rel=0.02), "1 mph each way for 48 h"


def test_ridge_axis_strength_and_tendency():
    from naturev1.cyclone.ridge import ridge_diagnostics
    from naturev1.cyclone.synthetic import Blob, Scenario, scenario_environment

    sc = Scenario("ridge", 20.0, -60.0, blobs=[Blob(8e6, 0.0, 900.0, 900.0, d_amplitude=-1e5)])
    env = scenario_environment(sc, 0.0)
    env.previous = {"z500": scenario_environment(sc, -12.0).fields["z"][list(env.levels).index(500)]}
    d = ridge_diagnostics(env, 20.0, -60.0)
    assert 24.0 < d["ridge_axis_lat"] < 32.0, d
    assert d["ridge_strength_m"] > 10.0 and d["ridge_tendency_m"] < 0.0, "a weakening high"


def test_vorticity_budget_motion_follows_the_flow():
    from naturev1.cyclone.vorticity_advection import budget_motion

    env = _vortex_env(u=-6.0, v=2.0, vmax=20.0, rmax_km=250.0)
    m = budget_motion(env, 20.0, -60.0)
    assert m is not None and (m["u"], m["v"]) == pytest.approx((-6.0, 2.0), abs=2.5)


def test_fine_layer_shear_and_tilt():
    from naturev1.cyclone.shear import shear_diagnostics, vortex_tilt

    env = synthetic_environment(20.0, -60.0, u=lambda la, lo, p: (850.0 - p) / 650.0 * 10.0 + 0 * la, v=0.0)
    d = shear_diagnostics(env, 20.0, -60.0, motion_ms=(-5.0, 0.0))
    assert d["shear_850_200"] == pytest.approx(10.0, abs=0.05) and d["shear_850_500"] == pytest.approx(10 * 350 / 650, abs=0.05)
    assert d["generalised_shear"] > 2.0 and abs(d["shear_vs_motion_deg"]) > 170.0, "shear against the motion"
    tilted = _vortex_env(u=0.0, vmax=20.0, centre=(20.0, -60.0))
    upper = _vortex_env(u=0.0, vmax=20.0, centre=(21.5, -58.5))
    tilted.fields["u"][5], tilted.fields["v"][5] = upper.fields["u"][5], upper.fields["v"][5]      # 500 hPa
    t = vortex_tilt(tilted, 20.0, -60.0)
    assert t["tilt_km"] > 60.0 and 0.0 < t["tilt_dir"] < 90.0, "mid-level centre to the north-east"


def test_reformation_targets_the_downshear_convection():
    from naturev1.cyclone.reformation import candidate_centres, reformation_outlook

    env = _vortex_env(u=-3.0, vmax=12.0)
    lat_g, lon_g = np.meshgrid(env.lat, env.lon, indexing="ij")
    x, y = geo.to_local(20.0, -60.0, lat_g, lon_g)
    env.fields["msl"] = 101000.0 - 600.0 * np.exp(-((x ** 2 + y ** 2) / 2 / 150.0 ** 2)) - 500.0 * np.exp(-(((x - 360) ** 2 + y ** 2) / 2 / 120.0 ** 2))
    env.fields["tcwv"] = 45.0 + 15.0 * np.exp(-(((x - 250) ** 2 + y ** 2) / 2 / 120.0 ** 2))
    cands = candidate_centres(env, 20.0, -60.0)
    assert len([c for c in cands if c["kind"] == "pressure"]) >= 2, "competing centres"
    out = reformation_outlook(env, env, 20.0, -60.0, 35.0)
    assert out["target_xy_km"][0] > 50.0 and 0.0 <= out["jump_probability"] <= 1.0


def test_damped_anomaly_flow_goes_from_analysis_to_climatology():
    from naturev1.cyclone.climatology import DampedAnomalyFlow

    class Flow:
        def __init__(self, u):
            self.u = u

        def velocity(self, la, lo):
            return np.full(np.shape(np.atleast_1d(la)), self.u), np.zeros(np.shape(np.atleast_1d(la)))

    f = DampedAnomalyFlow(Flow(-8.0), Flow(-2.0), 48.0)
    assert float(f(20.0, -60.0, 0.0)[0][0]) == pytest.approx(-8.0)
    assert float(f(20.0, -60.0, 48.0)[0][0]) == pytest.approx(-2.0 - 6.0 / math.e)
    assert float(f(20.0, -60.0, 1e4)[0][0]) == pytest.approx(-2.0, abs=1e-6)


def test_ensemble_spreads_with_lead_and_its_mean_is_on_the_flow():
    from naturev1.cyclone.beta import BetaDrift
    from naturev1.cyclone.ensemble import FlowPerturbations, physics_ensemble
    from naturev1.cyclone.steering import SteeringFlow

    env = synthetic_environment(20.0, -60.0, u=-5.0)
    pert = FlowPerturbations(20.0, -60.0, env, members=8)
    la, lo = np.full(8, 21.0), np.full(8, -59.0)
    u0, v0 = pert.velocity(la, lo, 0.0)
    u1, v1 = pert.velocity(la, lo, 60.0)
    assert np.allclose(u1, 2 * u0) and np.allclose(v1, 2 * v0), "errors double after 60 h"
    case = _case(motion_kmh=(-18.0, 0.0))
    flows = {"med": SteeringFlow(env, 400, 850), "dlm": SteeringFlow(env, 200, 850)}
    ens = physics_ensemble(case, env, flows, BetaDrift(env, 0.5), (-5.0, 0.0), LEADS, members=12)
    assert ens["members"].shape == (12, len(LEADS), 2)
    assert ens["spread_km"][-1] > ens["spread_km"][1] > 0


def test_regimes_classify_the_obvious_cases():
    from naturev1.cyclone.regimes import classify

    case = _case(lat=20.0, wind=100.0)
    assert classify(case, {"stall_probability": 0.9}, {})["regime"] == "stall"
    assert classify(_case(lat=35.0), {"stall_probability": 0.1}, {})["regime"] == "high"
    assert classify(case, {"stall_probability": 0.1}, {"ridge_gap_deg": 1.0})["regime"] == "recurve"
    assert classify(_case(wind=30.0), None, None)["regime"] == "weak"
    assert classify(case, None, {"ridge_gap_deg": 15.0})["regime"] == "straight"


# --------------------------------------------------------------------------------------- intensity --

def test_cold_wake_is_worst_under_a_slow_strong_storm():
    from naturev1.cyclone.ocean import effective_sst, wake_cooling_k

    assert wake_cooling_k(140, 1.0, 302.0) > wake_cooling_k(140, 6.0, 302.0) > wake_cooling_k(60, 6.0, 302.0)
    hours = np.arange(0, 49, 1.0)
    stalled = effective_sst(np.full(49, 302.0), np.full(49, 150.0), np.full(49, 0.5), hours)
    moving = effective_sst(np.full(49, 302.0), np.full(49, 150.0), np.full(49, 6.0), hours)
    assert stalled[-1] < moving[-1] - 1.0 and stalled[-1] < stalled[0], "Dorian sat on its own wake"


def test_land_and_terrain_decay():
    from naturev1.cyclone.land import apply_land_decay

    hours = np.arange(0, 25, 1.0)
    water = np.full(25, 120.0)
    land = (hours >= 6).astype(float)
    flat = apply_land_decay(hours, water, land, np.zeros(25))
    hills = apply_land_decay(hours, water, land, np.full(25, 1500.0))
    assert flat[5] == 120.0 and flat[-1] < 60.0 and hills[-1] < flat[-1]


def test_shifor_learns_a_climatology_and_lgem_approaches_its_ceiling():
    from naturev1.cyclone.intensity import LGEM_BETA, LGEM_KAPPA, LGEM_N, fit_shifor, lgem_intensity, shifor_intensity

    rng = np.random.default_rng(0)
    cases = []
    for k in range(300):
        wind = float(rng.uniform(35, 120))
        future = np.minimum(wind + 0.1 * (130 - wind) * np.asarray(LEADS, float) / 24.0, 160)
        truth = {"lat": np.zeros(len(LEADS)), "lon": np.zeros(len(LEADS)), "wind": future,
                 "pressure": np.full(len(LEADS), np.nan), "valid": np.ones(len(LEADS), bool)}
        cases.append(_case(lat=float(rng.uniform(12, 28)), wind=wind, truth=truth))
    table = fit_shifor(cases, LEADS)
    pred = np.stack([shifor_intensity(c, LEADS, table) for c in cases])
    truth = np.stack([c.truth["wind"] for c in cases])
    assert np.mean(np.abs(pred - truth)) < 0.3 * np.mean(np.abs(truth - np.stack([np.full(len(LEADS), c.wind_kt) for c in cases])))
    hours = np.arange(0, 121, 1.0)
    along = {"hours": hours, "six_hours": hours[::6], "pi_kt": np.full(21, 140.0), "shear_ms": np.full(21, 3.0),
             "rh_mid": np.full(21, 0.7), "sst_eff_k": np.full(121, 302.0), "land": np.zeros(121), "height_m": np.zeros(121)}
    v = lgem_intensity(_case(wind=50.0), along, LEADS, LGEM_KAPPA, LGEM_N, LGEM_BETA)     # the integrator, not the fit
    assert v[-1] > 100.0 and v[-1] <= 141.0 and np.all(np.diff(v) >= -1e-6), "logistic growth to the ceiling"


def test_roci_of_a_constructed_low():
    from naturev1.cyclone.structure import gale_radius_estimate, outer_closed_isobar

    env = synthetic_environment(20.0, -60.0)
    lat_g, lon_g = np.meshgrid(env.lat, env.lon, indexing="ij")
    r = geo.great_circle_km(20.0, -60.0, lat_g, lon_g)
    env.fields["msl"] = 101200.0 - 4000.0 * np.exp(-0.5 * (r / 300.0) ** 2)
    out = outer_closed_isobar(env, 20.0, -60.0)
    assert 500.0 <= out["roci_km"] <= 1100.0 and out["centre_hpa"] < 975.0
    assert gale_radius_estimate(120.0, 25.0, out["roci_km"]) > gale_radius_estimate(40.0, 15.0, 300.0)


def test_eddy_flux_convergence_vanishes_for_a_symmetric_vortex():
    from naturev1.cyclone.outflow import eddy_flux_convergence

    env = _vortex_env(u=0.0, vmax=20.0)
    assert abs(eddy_flux_convergence(env, 20.0, -60.0)) < 1.0


# ------------------------------------------------------------------------------ tracking & genesis --

def test_the_field_tracker_follows_a_moving_low_below_the_grid_spacing():
    from naturev1.cyclone.field_tracker import track_in_fields

    lat_axis = np.linspace(-90, 90, 121)
    lon_axis = np.arange(240) * 1.5
    lat_g, lon_g = np.meshgrid(lat_axis, lon_axis, indexing="ij")
    hours = np.array([6.0, 12.0, 18.0])
    msl = []
    truth = []
    for h in hours:
        la, lo = geo.move(20.3, -59.7, -20.0 * h, 5.0 * h)
        truth.append((float(la), float(lo)))
        r = geo.great_circle_km(la, lo, lat_g, lon_g)
        msl.append(101000.0 - 3000.0 * np.exp(-0.5 * (r / 250.0) ** 2))
    track = track_in_fields((20.3, -59.7), hours, np.stack(msl), lat_axis, lon_axis, first_motion_kmh=(-20.0, 5.0))
    errors = [geo.great_circle_km(a[0], a[1], b[0], b[1]) for a, b in zip(track, truth)]
    assert max(errors) < 40.0, errors


def test_pouch_is_closed_in_the_wave_frame_and_open_in_the_earth_frame():
    from naturev1.cyclone.pouch import pouch_closedness

    lat = np.linspace(0, 45, 31)
    lon = np.arange(240) * 1.5
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    x, y = geo.to_local(12.0, 300.0, lat_g, lon_g)
    r = np.maximum(np.hypot(x, y), 1.0)
    vt = 6.0 * (r / 400.0) * np.exp(1 - r / 400.0)
    fields = {}
    for lev in ("850", "600"):
        fields[f"u{lev}"] = -7.0 - vt * y / r
        fields[f"v{lev}"] = vt * x / r
    env = Environment(lat=lat, lon=lon, levels=(850,), fields=fields)
    closed = pouch_closedness(env, 12.0, 300.0, phase_ms=(-7.0, 0.0))
    open_ = pouch_closedness(env, 12.0, 300.0, phase_ms=(0.0, 0.0))
    assert closed["closedness"] > 0.7 and closed["share_cyclonic"] > open_["share_cyclonic"]


def test_genesis_logistic_separates_and_detects():
    from naturev1.cyclone.genesis import Candidate, design_matrix, fit_logistic, genesis_probability, verify_genesis

    rng = np.random.default_rng(1)
    cands = []
    for k in range(600):
        good = k < 60
        f = {"vort850": rng.normal(5 if good else 3, 1), "rh600": rng.normal(0.8 if good else 0.55, 0.08),
             "shear": rng.normal(0.6 if good else 1.5, 0.3), "pouch": rng.uniform(0.5, 1) if good else rng.uniform(0, 0.5)}
        cands.append(Candidate(np.datetime64("2020-08-01T00") + np.timedelta64(12 * k, "h"), 15.0, -40.0, f, label=float(good)))
    x, y = design_matrix(cands), np.array([c.label for c in cands])
    w = fit_logistic(x, y, l2=1e-3)
    p = genesis_probability(x, w)
    events = [(c.time + np.timedelta64(24, "h"), c.lat, c.lon) for c in cands if c.label]
    r = verify_genesis(cands, p, events, threshold=0.5)
    assert r["brier_skill"] > 0.6 and r["detection_rate"] > 0.9


def test_synthetic_dorian_stalls_and_implanted_vortices_are_balanced():
    from naturev1.cyclone.synthetic import implant_vortex, scenario_inputs, scenario_library, stall_hours, truth_track

    lib = scenario_library()
    hours = np.arange(0, 121, 6.0)
    assert stall_hours(truth_track(lib["dorian_stall"], hours), hours) >= 36.0
    assert stall_hours(truth_track(lib["straight_runner"], hours), hours) == 0.0
    case, env = scenario_inputs(lib["dorian_stall"], LEADS)
    assert case.truth["valid"].all() and "z500" in env.previous
    storm = implant_vortex(env, 26.5, -76.0, 60.0)
    centre = (int(np.argmin(np.abs(env.lat - 26.5))), int(np.argmin(np.abs(((env.lon + 180) % 360 - 180) + 76.0))))
    k850, k300 = list(env.levels).index(850), list(env.levels).index(300)
    assert storm.fields["z"][k850][centre] < env.fields["z"][k850][centre] - 500.0, "a low"
    assert storm.fields["t"][k300][centre] > env.fields["t"][k300][centre], "with a warm core"


# ------------------------------------------------------------------------ consensus, verification --

def test_consensus_recovers_the_right_blend():
    from naturev1.cyclone.consensus import apply_consensus, fit_consensus, project_simplex

    assert project_simplex(np.array([0.5, 0.8, -0.2])).sum() == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    cases, f = [], []
    for k in range(400):
        c = _case(lat=float(rng.uniform(12, 30)), motion_kmh=(float(rng.normal(-15, 5)), float(rng.normal(5, 5))))
        tl, to = c.truth["lat"], c.truth["lon"]
        good = np.stack([tl + rng.normal(0, 0.05, len(LEADS)), to + rng.normal(0, 0.05, len(LEADS))], -1)
        bad = np.stack([tl + rng.normal(0, 1.0, len(LEADS)), to + rng.normal(0, 1.0, len(LEADS))], -1)
        cases.append(c)
        f.append(np.stack([bad, good], 1))
    stack = np.stack(f)
    w = fit_consensus(cases, stack, reference=1)
    assert (w["cross"][:, 1] > 0.9).all()
    blended = apply_consensus(cases, stack, w)
    assert np.nanmean(geo.great_circle_km(blended[..., 0], blended[..., 1], stack[:, :, 1, 0], stack[:, :, 1, 1])) < 25.0


def test_verification_is_homogeneous_and_bootstrapped_by_storm():
    from naturev1.cyclone.verify import block_bootstrap, compare

    groups = np.repeat(np.arange(10), 5)
    stats = block_bootstrap(groups, lambda idx: float(len(np.unique(groups[idx]))), samples=50)
    assert stats.max() <= 10
    cases = [_case() for _ in range(40)]
    for i, c in enumerate(cases):
        c.storm_id = f"S{i // 4}"
    base = np.full((40, len(LEADS)), 100.0) + np.random.default_rng(0).normal(0, 5, (40, len(LEADS)))
    better = base * 0.7
    better[0, 0] = np.nan
    res = compare(cases, {"b": base, "m": better}, "b", LEADS, samples=200)
    assert res["count"][0] == 39 and res["members"]["m"]["beats"][2]


def test_baselines_and_kalman_on_steady_motion():
    from naturev1.cyclone.baselines import xtrp_great_circle, xtrp_latlon
    from naturev1.cyclone.kalman import estimate_motion, kalman_track

    case = _case(lat=15.0, motion_kmh=(-20.0, 0.0))
    for f in (xtrp_latlon(case, LEADS, 12), xtrp_great_circle(case, LEADS, 12), kalman_track(case, LEADS)):
        err = geo.great_circle_km(f[:, 0], f[:, 1], case.truth["lat"], case.truth["lon"])
        assert err[2] < 25.0, err
    est = estimate_motion(case)
    assert est.velocity_kmh == pytest.approx([-20.0, 0.0], abs=1.0)


def test_cliper_and_analogs_learn_from_history():
    from naturev1.cyclone.analogs import AnalogDatabase
    from naturev1.cyclone.cliper import cliper_track, fit_cliper

    rng = np.random.default_rng(0)
    hist = [_case(lat=float(rng.uniform(12, 25)), lon=float(rng.uniform(-80, -40)),
                  motion_kmh=(float(rng.normal(-18, 3)), float(rng.normal(3, 2)))) for _ in range(500)]
    table = fit_cliper(hist, LEADS)
    case = _case(lat=18.0, motion_kmh=(-18.0, 3.0))
    err = geo.great_circle_km(*cliper_track(case, LEADS, table)[-1], case.truth["lat"][-1], case.truth["lon"][-1])
    assert err < 120.0
    db = AnalogDatabase.from_cases(hist, LEADS)
    f = db.forecast(case)
    assert geo.great_circle_km(f[-1, 0], f[-1, 1], case.truth["lat"][-1], case.truth["lon"][-1]) < 150.0


def test_shipped_coefficients_are_consistent_with_the_member_suite():
    from naturev1.cyclone import coefficients
    from naturev1.cyclone.members import DEFAULT_MEMBERS

    cons = coefficients.CONSENSUS
    assert tuple(cons["members"]) == DEFAULT_MEMBERS
    assert np.asarray(cons["along"]).shape == (len(LEADS), len(DEFAULT_MEMBERS))
    assert np.allclose(np.asarray(cons["cross"]).sum(-1), 1.0, atol=1e-3)
    assert set(coefficients.CLIPER) >= {"leads", "NA", "EP", "WP", "SH"}
    assert "held_out" in coefficients.PROVENANCE


def test_members_run_on_a_synthetic_dorian():
    from naturev1.cyclone.members import run_members
    from naturev1.cyclone.synthetic import scenario_inputs, scenario_library

    case, env = scenario_inputs(scenario_library()["dorian_stall"], LEADS)
    out = run_members(case, env, None, None, names=("xtrp6", "cliper", "budget_mix", "evo12", "baro", "baro_res12"))
    for name in ("xtrp6", "cliper", "budget_mix", "evo12", "baro", "baro_res12"):
        assert out[name].shape == (len(LEADS), 2) and np.isfinite(out[name][:4]).all(), name


def test_case_selection_rules():
    from naturev1.besttrack import Track
    from naturev1.cyclone.data import cases_from_tracks

    times = np.arange(np.datetime64("2019-08-30T00"), np.datetime64("2019-09-03T00"), np.timedelta64(6, "h")).astype("datetime64[s]")
    n = len(times)
    status = np.array(["TD"] + ["HU"] * (n - 3) + ["EX", "EX"])
    track = Track("X", "X", times, np.linspace(20, 28, n).astype(np.float32), np.linspace(-60, -75, n).astype(np.float32),
                  np.full(n, 100.0, np.float32), np.full(n, 950.0, np.float32), status, np.full(n, np.nan, np.float32),
                  np.full((n, 3, 4), np.nan, np.float32), np.array([""] * n), basin="NA")
    cases = cases_from_tracks([track], [2019])
    assert all(str(c.time)[11:13] in ("00", "12") for c in cases)
    assert str(cases[0].time) != str(times[0]), "the first fix has no history to extrapolate"
    last = cases[-1]
    assert not last.truth["valid"][-1], "verified only while still tropical"
