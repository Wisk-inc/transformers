# Copyright 2026 Nathan. Apache-2.0.
"""The peak-intensity probability, the rain a parish gets if hit, the rain multiplier, the previous-cycle error feedback."""

from __future__ import annotations

import numpy as np

from naturev1.cyclone import coefficients, geometry as geo
from naturev1.cyclone.consensus import apply_track_feedback
from naturev1.cyclone.data import LEADS
from naturev1.cyclone.intensity import PEAK_FEATURES, peak_features, peak_probability
from naturev1.swarm.impact import ImpactSettings, paint
from naturev1.swarm.landfall import format_landfall, landfall_summary

from test_cyclone import _case
from test_landfall import _island_run


def _along(hours=121, pi=150.0, tchp=90.0, shear=5.0, rh=0.70, land=0.0):
    six = np.arange(0.0, hours, 6.0)
    return {"hours": np.arange(float(hours)), "six_hours": six, "pi_kt": np.full(six.size, pi), "pi_warm_kt": np.full(six.size, pi),
            "tchp_kj_cm2": np.full(six.size, tchp), "shear_ms": np.full(six.size, shear), "rh_mid": np.full(six.size, rh),
            "land": np.full(hours, land), "height_m": np.zeros(hours)}


def test_peak_probability_is_calibrated_to_the_table_and_monotonic():
    case = _case(wind=70.0, lat=16.0)
    table = coefficients.PEAK
    assert list(table["thresholds_kt"]) == [64.0, 96.0, 113.0, 137.0] and len(table["mean"]) == len(PEAK_FEATURES) - 1
    x = peak_features(case, _along(), np.full(len(LEADS), 90.0), 0.2, np.array([np.nan, np.nan, -12.0] + [np.nan] * 5))
    assert x.shape == (len(PEAK_FEATURES),) and x[0] == 1.0 and x[1] == 70.0 and x[11] == 1.0, "intercept, analysis wind, has_dp"
    quiet = peak_probability(case, _along(pi=100.0, tchp=20.0, shear=15.0, rh=0.40), np.linspace(70, 55, len(LEADS)), 0.02)
    explosive = peak_probability(case, _along(pi=165.0, tchp=120.0, shear=3.0, rh=0.75), np.linspace(70, 125, len(LEADS)), 0.6,
                                 np.array([np.nan, np.nan, -25.0, -35.0, -40.0] + [np.nan] * 3))
    assert quiet["probability"][0] == 1.0 and explosive["probability"][0] == 1.0, "already a hurricane"
    for j in (1, 2, 3):
        assert 0.0 <= quiet["probability"][j] < explosive["probability"][j] <= 1.0
    assert explosive["probability"][2] > 0.5 and quiet["probability"][3] < 0.05
    assert peak_probability(case, _along(), np.full(len(LEADS), 90.0), 0.2, table={}) == {}


def test_rain_if_hit_is_not_diluted_by_the_misses_and_the_multiplier_scales_it():
    region, case, result, impact = _island_run(n=150)
    u = impact.units[0]
    assert "rain_mm_if_hit" in u and "rain_mm_p90" in u and "rain_max_mm_if_hit" in u
    assert u["rain_max_mm_if_hit"] >= u["rain_mm_if_hit"] - 1e-6 and u["rain_mm_p90"] >= 0.0
    # worlds that miss the island by a lot: the swarm-mean rain drops, the rain given a hit does not
    fc = result["swarm"]["forecast"]
    hours, track = np.asarray(fc.track_hours, float), fc.track
    far = track.copy()
    far[: far.shape[0] // 2, :, 0] += 6.0                       # half the worlds pass 6 degrees north
    wet = paint(region, far, hours, np.concatenate([[case.wind_kt], result["intensity"]]), np.concatenate([[0.0], np.asarray(LEADS, float)]),
                settings=ImpactSettings(cell_stride=2))
    v = next(x for x in wet.units if x["unit"] == u["unit"])
    assert v["rain_mm"] < 0.7 * u["rain_mm"] and v["rain_mm_if_hit"] > 0.9 * u["rain_mm_if_hit"]
    kt, hh = np.concatenate([[case.wind_kt], result["intensity"]]), np.concatenate([[0.0], np.asarray(LEADS, float)])
    one = paint(region, track, hours, kt, hh, settings=ImpactSettings(cell_stride=2, rain_scale=1.0))
    doubled = paint(region, track, hours, kt, hh, settings=ImpactSettings(cell_stride=2, rain_scale=2.0))
    a = next(x for x in one.units if x["unit"] == u["unit"]); d = next(x for x in doubled.units if x["unit"] == u["unit"])
    assert abs(d["rain_mm"] - 2.0 * a["rain_mm"]) < 1e-3 * max(a["rain_mm"], 1.0) and abs(d["peak_kt"] - a["peak_kt"]) < 1e-6
    assert 1.0 <= ImpactSettings().rain_scale <= 2.5, "the shipped multiplier comes from the radar calibration"


def test_previous_cycle_error_feeds_back_into_the_track():
    case = _case()
    leads = np.asarray(LEADS, float)
    track = np.stack(geo.move(case.lat, case.lon, -20.0 * leads, 0.0 * leads), -1)
    # the previous cycle, 12 h ago, forecast the storm 60 km north of where it is now analysed
    prev_time = np.datetime64(case.time, "m") - np.timedelta64(12, "h")
    prev = np.stack(geo.move(case.lat, case.lon, -20.0 * (leads - 12.0), np.full(leads.size, 60.0)), -1)
    table = {"leads": list(LEADS), "max_lead_h": 48, "beta": {"12": [0.0, 0.2, 0.5, 0.5, 0.5, 0, 0, 0], "24": [0.0] * 8}}
    out = apply_track_feedback(case, track, prev, prev_time, LEADS, table)
    k24, k72 = LEADS.index(24), LEADS.index(72)
    n24 = geo.local_displacement_km(case.lat, case.lon, *out[k24])[1]
    assert abs(n24 + 0.5 * 60.0) < 2.0, "half the 60 km northward error is taken out at 24 h"
    assert np.allclose(out[k72], track[k72]), "nothing beyond 48 h"
    assert np.allclose(apply_track_feedback(case, track, prev, prev_time - np.timedelta64(60, "h"), LEADS, table), track), "too old"
    assert apply_track_feedback(case, track, None, None, LEADS, table) is track


def test_landfall_text_carries_the_peak_probabilities_and_rain_if_hit():
    region, case, result, impact = _island_run()
    result["peak_probability"] = {"thresholds_kt": [64.0, 96.0, 113.0, 137.0], "probability": [1.0, 0.62, 0.31, 0.08]}
    s = landfall_summary(region, result, case, impact, where=None, analysis_time="2025-10-27T00")
    text = format_landfall(s)
    assert "chance the storm peaks at or above" in text and "category 4 (113 kt) 31%" in text
    assert "rain if hit" in text and "max mm" in text
