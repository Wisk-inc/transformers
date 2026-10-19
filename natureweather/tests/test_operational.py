# Copyright 2026 Nathan. Apache-2.0.
"""The operations layer: infrared diagnostics from a GOES scene, the cycle record, and the previous cycle found again."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np

from naturev1.cyclone import geometry as geo
from naturev1.satellite import SatelliteScene
from naturev1.swarm.bulletin import history, load_previous, write_bulletin
from naturev1.swarm.infrared import ir_metrics

from test_landfall import _island_run


def _scene(lat0=18.0, lon0=-77.0, eye=True, asym=0.0, n=120):
    lat = lat0 + np.linspace(-3, 3, n)[:, None] * np.ones((1, n))
    lon = lon0 + np.ones((n, 1)) * np.linspace(-3, 3, n)[None, :]
    d = geo.great_circle_km(lat, lon, lat0, lon0)
    ang = np.arctan2((lon - lon0) * np.cos(np.radians(lat0)), lat - lat0)
    canopy = np.clip(1.0 - (d - 120.0) / 120.0, 0.0, 1.0) * (1.0 + asym * np.cos(ang))      # a cold canopy out to ~240 km
    bt = 285.0 - 90.0 * np.clip(canopy, 0.0, 1.0)
    if eye:
        bt = np.where(d < 20.0, 270.0, bt)
    bt = np.where(d > 400.0, 290.0, bt)
    return SatelliteScene(latitude=lat.ravel(), longitude=lon.ravel(), values=bt.ravel()[:, None], area_km2=np.full(lat.size, 16.0),
                          channels=("C13",), timestamp=dt.datetime(2025, 10, 27, 0, 0), satellite="GOES-19")


def test_infrared_metrics_read_a_cold_ring_with_an_eye():
    m = ir_metrics(_scene(), 18.0, -77.0)
    assert m["min_bt_c"] < -70.0 and 0.5 < m["pc50"] <= 1.0 and m["pc30"] >= m["pc50"]
    assert m["eye_warm_k"] > 15.0 and "eye visible" in m["text"]
    assert m["ring_asymmetry_k"] < 3.0, "a symmetric ring"
    lop = ir_metrics(_scene(asym=0.6), 18.0, -77.0)
    assert lop["ring_asymmetry_k"] > m["ring_asymmetry_k"] + 5.0, "a lopsided ring reads as asymmetric"
    earlier = _scene(eye=False)
    earlier.values[:, 0] += 15.0                                           # warmer tops three hours ago
    trend = ir_metrics(_scene(), 18.0, -77.0, previous=earlier, previous_hours=3.0)
    assert trend["d_min_bt_per_h"] < 0.0 and "pts/h" in trend["text"]
    empty = ir_metrics(_scene(), 40.0, -40.0)
    assert "no usable scene" in empty["text"]


def test_bulletin_round_trip_and_previous_cycle(tmp_path):
    region, case, result, impact = _island_run()
    result["peak_probability"] = {"thresholds_kt": [64.0, 96.0, 113.0, 137.0], "probability": [1.0, 0.5, 0.2, 0.05]}
    result["ri_probability"] = 0.12
    run = {"case": case, "result": result, "impact": impact, "analysis_time": np.datetime64("2025-10-27T00:00"), "models": {}, "r34_km": 150.0}
    from naturev1.swarm.landfall import landfall_summary

    summary = landfall_summary(region, result, case, impact, where=None, analysis_time="2025-10-27T00")
    out = write_bulletin(run, summary, tmp_path / "runs", "al999999")
    rec = json.loads(out["json"].read_text())
    assert rec["storm_id"] == "al999999" and rec["forecast"]["peak_probability"]["probability"][1] == 0.5
    assert rec["landfall"]["p_landfall"] > 0.9 and rec["parishes"] and "rain_mm_if_hit" in rec["parishes"][0]
    assert out["csv"].exists() and "peak_kt_if_hit" in out["csv"].read_text().splitlines()[0]
    assert len(history(tmp_path / "runs", "AL999999")) == 1
    prev = load_previous(tmp_path / "runs", "al999999", "2025-10-27T12:00")
    assert prev is not None and prev["track"].shape == np.asarray(result["track"]).shape and np.isfinite(prev["track"]).all()
    assert str(prev["time"]).startswith("2025-10-27T00")
    assert load_previous(tmp_path / "runs", "al999999", "2025-10-26T12:00") is None, "nothing earlier than the only record"
    assert load_previous(tmp_path / "runs", "al999999", "2025-10-30T00:00") is None, "too old to feed back"


def test_rain_shape_puts_more_rain_right_of_the_heading_and_widens_the_tail():
    from naturev1.cyclone.data import LEADS
    from naturev1.swarm.impact import ImpactSettings, paint

    region, case, result, impact = _island_run(n=60)
    fc = result["swarm"]["forecast"]
    hours, track = np.asarray(fc.track_hours, float), fc.track
    kt, hh = np.concatenate([[case.wind_kt], result["intensity"]]), np.concatenate([[0.0], np.asarray(LEADS, float)])
    sym = paint(region, track, hours, kt, hh, settings=ImpactSettings(cell_stride=2, rain_scale=1.0, rain_tail=1.0, rain_asymmetry=0.0))
    asym = paint(region, track, hours, kt, hh, settings=ImpactSettings(cell_stride=2, rain_scale=1.0, rain_tail=1.0, rain_asymmetry=0.6))
    wide = paint(region, track, hours, kt, hh, settings=ImpactSettings(cell_stride=2, rain_scale=1.0, rain_tail=2.0, rain_asymmetry=0.0))
    lat_c = float(np.nanmean(track[:, :, 0]))
    north = region.lat[:, None] > lat_c + 0.4
    south = region.lat[:, None] < lat_c - 0.4
    r_sym, r_asym, r_wide = (np.nan_to_num(x.cell_maps["mean_rain_mm"]) for x in (sym, asym, wide))
    # the worlds head west: the right-hand side of the heading is north, and the asymmetry moves rain there
    ratio_sym = r_sym[north & region.land].mean() / max(r_sym[south & region.land].mean(), 1e-6)
    ratio_asym = r_asym[north & region.land].mean() / max(r_asym[south & region.land].mean(), 1e-6)
    assert ratio_asym > ratio_sym * 1.15, (ratio_sym, ratio_asym)
    # a wider tail: more rain far from the track relative to the core
    dist = np.broadcast_to(np.abs(region.lat[:, None] - lat_c), region.land.shape)
    lo, hi = np.percentile(dist[region.land], [30, 75])
    core = (dist < lo) & region.land
    far = (dist > hi) & region.land
    assert far.sum() > 20 and core.sum() > 20
    # a wider tail adds rain beyond the radius of maximum rain everywhere, and never favours the core
    assert r_wide.sum() > 1.1 * r_sym.sum() and r_wide[far].mean() / r_wide[core].mean() >= r_sym[far].mean() / r_sym[core].mean()


def test_peak_and_ri_follow_the_callers_leads():
    from naturev1.cyclone.intensity import peak_probability, ri_probability
    from test_cyclone import _case
    from test_peak import _along

    case = _case(wind=70.0, lat=16.0)
    leads = (12, 24, 48, 72)
    dp = np.array([np.nan, -20.0, -30.0, np.nan])
    p = peak_probability(case, _along(), np.array([80.0, 95.0, 110.0, 100.0]), 0.3, dp, leads=leads)
    assert len(p["probability"]) == 4 and all(0.0 <= x <= 1.0 for x in p["probability"])
    q = ri_probability(case, _along(), model_dp=dp, leads=leads)
    assert 0.0 <= q <= 1.0
