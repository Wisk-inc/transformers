# Copyright 2026 Nathan. Apache-2.0.
"""Landfall timing and parish numbers, barometer centre fixes, point-wind nudging, the two-model blend weight."""

from __future__ import annotations

import numpy as np
import torch

from naturev1.cyclone import geometry as geo
from naturev1.cyclone.consensus import blend_with_model
from naturev1.cyclone.data import LEADS
from naturev1.swarm.core import SwarmForecast
from naturev1.swarm.geography import synthetic_island
from naturev1.swarm.impact import ImpactSettings, paint
from naturev1.swarm.landfall import format_landfall, landfall_summary
from naturev1.swarm.observations import centre_from_pressures, fix_from_pressures, holland_pressure, nudge_wind_map

from test_cyclone import _case


def _island_run(n=200, seed=0):
    """Worlds heading west over the synthetic island, spread north-south; an impact painting; a result dict."""
    region = synthetic_island()
    lat_c, lon_c = float(region.lat.mean()), float(region.lon.mean())
    case = _case(lat=lat_c, lon=lon_c + 4.0, motion_kmh=(-20.0, 0.0), wind=100.0)
    hours = np.arange(0.0, 48.1, 3.0)
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0.0, 0.25, n)
    track = np.stack([np.repeat((lat_c + offsets)[:, None], hours.size, 1), np.repeat(case.lon, n)[:, None] - (20.0 * hours / (111.2 * np.cos(np.radians(lat_c))))[None, :]], -1)
    positions = track[:, [list(hours).index(h) if h in hours else -1 for h in (6, 12, 24, 36, 48, 48, 48, 48)]]
    fc = SwarmForecast(positions=positions, weights=np.full(n, 1.0 / n), leads=list(LEADS), track=track, track_hours=hours)
    leads = np.asarray(LEADS, float)
    result = {"leads": list(LEADS), "track": np.stack(geo.move(case.lat, case.lon, -20.0 * leads, 0.0 * leads), -1),
              "intensity": np.array([100, 105, 110, 100, 90, 80, 70, 60], float), "intensity_regime": "normal",
              "swarm": {"forecast": fc}}
    impact = paint(region, track, hours, np.concatenate([[case.wind_kt], result["intensity"]]), np.concatenate([[0.0], leads]),
                   settings=ImpactSettings(cell_stride=2))
    return region, case, result, impact


def test_landfall_summary_times_the_crossing_and_ranks_the_units():
    region, case, result, impact = _island_run()
    s = landfall_summary(region, result, case, impact, where=None, analysis_time="2025-10-27T00")
    assert s["p_landfall"] > 0.9
    # the island's coast is ~4 degrees west of the start at 20 km/h: a little under a day
    assert 12.0 < s["hours"]["median"] < 30.0 and s["hours"]["p10"] <= s["hours"]["median"] <= s["hours"]["p90"]
    assert s["times"]["median"].startswith("2025-10-2") and s["times"]["median"].endswith("UTC")
    assert s["consensus"]["time"] and abs(s["consensus"]["hour"] - s["hours"]["median"]) < 6.0
    assert s["peak"]["intensity_kt"] == 110.0 and s["peak"]["hour"] == 24.0
    assert s["units"] and all(k in s["units"][0] for k in ("peak_gust_kt", "peak_kt_p90", "peak_hour", "hu_arrival_h"))
    u = s["units"][0]
    assert u["peak_gust_kt"] > u["peak_kt"] >= 64.0 and u["peak_kt_p90"] >= u["peak_kt"] - 1e-6
    assert np.isfinite(u["hu_arrival_h"]) and u["hu_arrival_h"] >= u["arrival_h"] - 1e-6 and u["peak_hour"] >= u["arrival_h"] - 1e-6
    text = format_landfall(s)
    assert "centre comes ashore" in text and "peak intensity: 110 kt" in text
    assert "mean_peak_kt" in impact.cell_maps and np.nanmax(impact.cell_maps["mean_peak_kt"]) >= 64.0


def test_barometers_place_the_centre_and_read_its_pressure():
    rng = np.random.default_rng(1)
    true = dict(lat=18.1, lon=-77.6, pc=935.0, penv=1010.0, rmw=30.0)
    lat = true["lat"] + rng.uniform(-1.5, 1.5, 300)
    lon = true["lon"] + rng.uniform(-1.5, 1.5, 300)
    r = geo.great_circle_km(lat, lon, true["lat"], true["lon"])
    p = holland_pressure(r, true["pc"], true["penv"], true["rmw"]) + rng.normal(0.0, 1.0, 300)       # phones: ~1 hPa
    fit = centre_from_pressures(lat, lon, p, sigma_hpa=1.0)
    assert geo.great_circle_km(fit["lat"], fit["lon"], true["lat"], true["lon"]) < 12.0
    assert abs(fit["pc_hpa"] - true["pc"]) < 6.0 and fit["sigma_km"] < 30.0 and 95.0 < fit["wind_kt"] < 130.0   # Atkinson-Holliday: 108 kt at 935 hPa
    fix, _ = fix_from_pressures("2025-10-28T15:00", lat, lon, p)
    assert fix.source == "BARO" and fix.sigma_km == fit["sigma_km"]


def test_point_winds_correct_the_map_locally():
    glat, glon = np.arange(17.5, 18.81, 0.05), np.arange(-78.5, -76.99, 0.05)
    grid = np.full((glat.size, glon.size), 80.0)
    out, innovation = nudge_wind_map(grid, glat, glon, [18.1], [-77.7], [120.0], length_km=25.0, obs_error_kt=5.0, background_error_kt=15.0)
    i, j = np.argmin(np.abs(glat - 18.1)), np.argmin(np.abs(glon + 77.7))
    assert innovation[0] == 40.0 and out[i, j] > 110.0, "the map moves most of the way to the observation"
    far_i, far_j = np.argmin(np.abs(glat - 17.6)), np.argmin(np.abs(glon + 78.4))
    assert abs(out[far_i, far_j] - 80.0) < 0.5, "and not at all far away"


def test_two_or_more_models_get_the_heavier_blend_weight():
    case = _case()
    leads = np.asarray(LEADS, float)
    physics = np.stack(geo.move(case.lat, case.lon, -20.0 * leads, 0.0 * leads), -1)
    model = np.stack(geo.move(case.lat, case.lon, -20.0 * leads, 10.0 * leads), -1)
    spec = {"leads": list(LEADS), "alpha": [0.5] * 8, "alpha_multi": [0.9] * 8, "offset_tau_h": None}
    one = blend_with_model(case, physics, model, spec, LEADS, n_models=1)
    many = blend_with_model(case, physics, model, spec, LEADS, n_models=3)
    k = LEADS.index(48)
    n1 = geo.local_displacement_km(case.lat, case.lon, *one[k])[1]
    n3 = geo.local_displacement_km(case.lat, case.lon, *many[k])[1]
    assert abs(n1 - 0.5 * 480.0) < 2.0 and abs(n3 - 0.9 * 480.0) < 2.0
