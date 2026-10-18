# Copyright 2026 Nathan. Apache-2.0.
"""naturev1.swarm: the satellite path (fixes), 4D worldlines, the region raster and parish impacts -- offline."""

from __future__ import annotations

import numpy as np

from naturev1.cyclone import geometry as geo
from naturev1.cyclone.members import build_context
from naturev1.swarm.core import Swarm
from naturev1.swarm.fixes import FIX_SIGMA_KM, Fix, assimilate, merge, parse_bdeck, parse_fdeck, window
from naturev1.swarm.geography import _rasterise, synthetic_island
from naturev1.swarm.impact import ImpactSettings, paint

from test_cyclone import _case, _vortex_env

FDECK = """\
AL, 13, 202510190000,  10, DVTS,         CI,  , 1250N,  5530W,      , 3,  25, 2,     , 2, DVRK,    ,     ,
AL, 13, 202510211727,  50, AIRC,           ,  , 1376N,  7041W,      , 1,    ,  ,     ,  ,     ,    ,     ,
AL, 13, 202510211735,  50, AIRC,           ,  ,      ,       ,      , 3,    ,  ,     ,  ,     ,    ,     ,
AL, 13, 202510211740,  30, MICR,          I,  , 1380N,  7045W,      , 2,  90, 2,     ,  ,     ,    ,     ,
AL, 13, 202510211745,  30, AMSR,          C,  , 1381N,  7046W,      , 2,    ,  ,     ,  ,     ,    ,     ,
"""
BDECK = """\
AL, 13, 2025102812,   , BEST,   0, 176N,  779W, 155,  899, HU,  34, NEQ,  170,  130,   60,   80,
AL, 13, 2025102812,   , BEST,   0, 176N,  779W, 155,  899, HU,  50, NEQ,   50,   45,   30,   45,
AL, 13, 2025102818,   , BEST,   0, 182N,  780W, 145,  910, HU,  34, NEQ,  170,  130,   60,   80,
"""


def test_fdeck_keeps_centre_fixes_with_their_own_uncertainty():
    fixes = parse_fdeck(FDECK)
    assert [f.source for f in fixes] == ["DVTS", "AIRC", "AMSR"], "intensity-only and position-less rows dropped"
    dvts, airc, amsr = fixes
    assert abs(dvts.lat - 12.5) < 1e-9 and abs(dvts.lon + 55.3) < 1e-9 and dvts.wind_kt == 25.0
    assert airc.sigma_km == FIX_SIGMA_KM["AIRC"] and dvts.sigma_km > 2 * FIX_SIGMA_KM["DVTS"], "confidence 3 widens"
    merged = merge(fixes[1:], minutes=30.0)
    assert len(merged) == 1 and merged[0].sigma_km < airc.sigma_km and "AIRC" in merged[0].source
    assert len(window(fixes, "2025-10-21T00:00", "2025-10-22T00:00")) == 2


def test_bdeck_folds_wind_radii_rows():
    best = parse_bdeck(BDECK)
    assert len(best) == 2 and best[0].wind_kt == 155.0 and abs(best[1].lat - 18.2) < 1e-9 and best[1].lon == -78.0


def test_sub_hourly_fixes_drive_a_4d_swarm():
    case, env = _case(motion_kmh=(-18.0, 4.0), wind=60.0), _vortex_env()
    sw = Swarm.from_context(case, build_context(case, env), 2000, seed=3)
    sw.step_hours, sw.record = 0.25, 50
    fixes = []
    for minutes in range(-24 * 60 + 20, 1, 20):                         # a fix every 20 minutes
        la, lo = geo.move(case.lat, case.lon, -18.0 * minutes / 60.0, 4.0 * minutes / 60.0)
        fixes.append(Fix(case.time + np.timedelta64(minutes, "m"), float(la), float(lo), 15.0, "AIRC",
                         60.0 + 40.0 * (minutes + 24 * 60) / (24 * 60)))   # strengthening to 100 kt
    history = assimilate(sw, fixes, case.time)
    assert len(history) == len(fixes) and abs(sw.hours) < 1e-6
    lines, hours = sw.worldline_array()
    assert lines.shape[0] == 50 and lines.shape[2] == 3 and np.all(np.diff(hours) <= 0.25 + 1e-9)
    assert float(sw.params["depth"].mean()) > 0.55, "a strengthening storm (60 -> 100 kt) reaches deeper steering"
    assert np.nanmean(lines[:, -1, 2]) > np.nanmean(lines[:, 0, 2])
    step = geo.great_circle_km(lines[:, 1:, 0], lines[:, 1:, 1], lines[:, :-1, 0], lines[:, :-1, 1])
    assert np.nanmax(step) < 40.0, "worldlines are continuous: no jumps inside a 15-minute micro-segment"
    w = np.exp(sw.logw.numpy())
    lat, lon = sw.lat.numpy(), sw.lon.numpy()
    d = geo.great_circle_km(lat, lon, case.lat, case.lon)
    assert float((w * d).sum()) < 30.0, "the filtered worlds sit on the latest fix"


def test_rasteriser_fills_polygons_and_respects_holes():
    lat = np.arange(0, 10.01, 0.1)
    lon = np.arange(0, 10.01, 0.1)
    square = np.array([[2, 2], [8, 2], [8, 8], [2, 8], [2, 2]], float)
    hole = np.array([[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]], float)
    mask = _rasterise([(square, [hole])], lat, lon)
    assert abs(mask.sum() - 3200) < 150 and not mask[50, 50] and mask[30, 30] and not mask[90, 90]


def test_parish_impacts_follow_the_track_and_the_mountains():
    region = synthetic_island(peak_m=2000.0)                             # four parishes west to east, a ridge
    hours = np.arange(0.0, 49.0, 1.0)
    # From the south, crossing the western half of the island northward at 6 m/s, 130 kt over water.
    lat = 16.0 + hours * 6.0 * 3.6 / 111.2
    track = np.stack([lat, np.full_like(lat, -77.9)], -1)[None].repeat(64, 0)
    track[:, :, 1] += np.linspace(-0.05, 0.05, 64)[:, None]
    result = paint(region, track, hours, np.full(hours.size, 130.0), hours,
                   settings=ImpactSettings(intensity_sd0=0.05, intensity_sd_per_h=0.0, size_log_sd=0.1))
    by = {u["name"]: u for u in result.units}
    assert by["Parish 1"]["p64"] > 0.9 and by["Parish 2"]["p64"] > by["Parish 4"].get("p64", 0.0)
    assert by["Parish 4"]["p64"] < 0.5, "the far end of the island is spared hurricane-force wind"
    rain = result.cell_maps["mean_rain_mm"]
    i_ridge = int(np.argmin(np.abs(region.lat - 18.1)))
    j = int(np.argmin(np.abs(region.lon - (-77.9))))
    south_slope = rain[i_ridge - 8, j]                                    # windward side, flow from the south
    assert np.isfinite(south_slope) and south_slope > 50.0
    assert np.nanmax(result.cell_maps["arrival_h"]) > np.nanmin(result.cell_maps["arrival_h"])
    assert "Parish 1" in result.table()


def test_impact_keeps_land_decay_and_respects_a_best_track():
    region = synthetic_island()
    hours = np.arange(0.0, 25.0, 1.0)
    track = np.stack([np.full(hours.size, 18.1), -79.5 + hours * 0.1], -1)[None]     # straight along the island
    decayed = paint(region, track, hours, np.full(hours.size, 120.0), hours,
                    settings=ImpactSettings(intensity_sd0=0.0, intensity_sd_per_h=0.0, size_log_sd=0.0))
    kept = paint(region, track, hours, np.full(hours.size, 120.0), hours,
                 settings=ImpactSettings(intensity_sd0=0.0, intensity_sd_per_h=0.0, size_log_sd=0.0, land_decay=False))
    peak = lambda r: max(u["peak_kt"] for u in r.units if u["name"] == "Parish 4")
    assert peak(decayed) < peak(kept), "a storm running the length of the island weakens over it"


def test_gfs_coarsening_is_conservative():
    from naturev1.swarm.analysis import LAT_AXIS, _coarsen

    lat_q = np.arange(90.0, -90.0 - 1e-9, -0.25)
    field = np.repeat(lat_q[:, None], 1440, 1)                           # value = latitude
    out = _coarsen(field)
    assert out.shape == (121, 240)
    assert np.allclose(out[10:-10, 0], LAT_AXIS[10:-10], atol=0.05) and np.allclose(_coarsen(np.ones((721, 1440))), 1.0)


def test_case_from_bdeck_builds_history_and_truth():
    from naturev1.swarm.analysis import case_from_bdeck

    t0 = np.datetime64("2025-10-26T00:00")
    best = [Fix(t0 + np.timedelta64(6 * k, "h"), 15.0 + 0.2 * k, -75.0 - 0.3 * k, 20.0, "BEST", 100.0 + k)
            for k in range(-8, 25)]
    case = case_from_bdeck(best, "2025-10-28T00:00", "AL132025", "TEST")
    assert abs(case.lat - (15.0 + 0.2 * 8)) < 1e-9 and case.history["lat"].shape == (9,)
    assert np.isfinite(case.history["lat"]).all() and case.truth["valid"][:-1].all() and not case.truth["valid"][-1]
