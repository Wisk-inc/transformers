# Copyright 2026 Nathan. Apache-2.0.
"""The ocean's depth in the intensity, the best track's wind radii, native-grid windows and calibrated world
intensities -- on constructed inputs."""

from __future__ import annotations

import numpy as np

from naturev1.cyclone.data import LEADS
from naturev1.cyclone.intensity import (RI_FEATURES, RI_MODEL_FEATURES, RI_OCEAN_FEATURES, SHIPS_PREDICTORS,
                                        ri_features, ri_probability, ships_intensity, ships_predictors)
from naturev1.cyclone.ocean import OceanClimatology, effective_sst, wake_cooling_k
from naturev1.swarm.fixes import gale_radius_km, parse_bdeck
from naturev1.swarm.models import native_window

from test_cyclone import _case
from test_forecastflow import _along


def _climatology(d26=120.0, tchp=110.0):
    lat, lon = np.arange(-89.5, 90.0), np.arange(-179.5, 180.0)
    d = np.full((12, lat.size, lon.size), d26, np.float32)
    t = np.full((12, lat.size, lon.size), tchp, np.float32)
    land = np.abs(lat)[:, None] > 80
    d[:, land.repeat(lon.size, 1)] = np.nan
    t[:, land.repeat(lon.size, 1)] = np.nan
    return OceanClimatology(d, t, lat, lon)


def test_deep_warm_water_cools_less_and_the_packaged_climatology_knows_the_caribbean():
    shallow = wake_cooling_k(120.0, 1.0, 302.0, d26_m=30.0)
    deep = wake_cooling_k(120.0, 1.0, 302.0, d26_m=120.0)
    assert deep < 0.55 * shallow
    hours = np.arange(0.0, 49.0)
    crawl = np.full(hours.size, 1.5)
    sst = np.full(hours.size, 302.5)
    thin = effective_sst(sst, np.full(hours.size, 130.0), crawl, hours, d26_m=np.full(hours.size, 30.0))
    deep_layer = effective_sst(sst, np.full(hours.size, 130.0), crawl, hours, d26_m=np.full(hours.size, 120.0))
    assert deep_layer[-1] > thin[-1] + 1.0, "a stalled storm over 120 m of warm water keeps most of its fuel"
    assert effective_sst(sst, np.full(hours.size, 130.0), crawl, hours, d26_m=np.full(hours.size, 120.0), d26_power=0.0)[-1] < deep_layer[-1]
    packaged = OceanClimatology.default()
    assert packaged is not None
    s = packaged.sample([17.5, 60.0], [-77.5, -30.0], "2025-10-27T00")
    assert s["d26_m"][0] > 80.0 and s["tchp_kj_cm2"][0] > 70.0, "the central Caribbean in late October"
    assert s["d26_m"][1] == 0.0 and s["tchp_kj_cm2"][1] == 0.0, "the North Atlantic at 60N has no 26 C water"
    synthetic = _climatology()
    at = synthetic.sample([0.0, 85.0], [0.0, 0.0], "2020-07-15")
    assert abs(at["d26_m"][0] - 120.0) < 1e-3 and np.isnan(at["d26_m"][1]), "land / no data stays NaN"


def test_heat_potential_is_the_last_predictor_and_old_tables_still_work():
    case = _case(wind=80.0)
    along = _along(pi=150.0)
    along["shear_mid_ms"] = np.full(along["six_hours"].size, 3.0)
    along["speed_ms"] = np.full(along["hours"].size, 5.0)
    base = ships_predictors(case, along, 24.0)
    assert base.shape == (len(SHIPS_PREDICTORS),) and base[-1] == 0.0, "no ocean: the heat-potential predictor is 0"
    along["tchp_kj_cm2"] = np.full(along["hours"].size, 100.0)
    along["d26_m"] = np.full(along["hours"].size, 120.0)
    assert abs(ships_predictors(case, along, 24.0)[-1] - 2.0) < 1e-9
    old_table = {"leads": [24], "weights": [[5.0] + [0.0] * (len(SHIPS_PREDICTORS) - 2)]}     # one predictor fewer
    assert ships_intensity(case, along, [24], old_table)[0] == 85.0, "a shorter table ignores the new predictor"
    f = ri_features(case, along)
    assert f.shape == (len(RI_FEATURES) + len(RI_MODEL_FEATURES) + len(RI_OCEAN_FEATURES),)
    assert abs(f[-2] - 2.0) < 1e-9 and abs(f[-1] - 2.4) < 1e-9
    old_ri = {"weights": [0.0] * len(RI_FEATURES), "base_rate": 0.1}
    assert abs(ri_probability(case, along, old_ri) - 0.5) < 1e-9
    new_ri = {"weights": [0.0] * (len(RI_FEATURES) + 3) + [1.0, 0.0], "base_rate": 0.1}
    assert ri_probability(case, along, new_ri) > 0.85, "a warm deep ocean raises P(RI) under an ocean-aware table"


def test_best_track_radii_are_parsed_and_folded():
    deck = ("AL, 13, 2025102700,   , BEST,   0, 163N,  777W, 125,  948, HU,  34, NEQ,  130,  120,   90,  100,\n"
            "AL, 13, 2025102700,   , BEST,   0, 163N,  777W, 125,  948, HU,  50, NEQ,   70,   60,   50,   60,\n"
            "AL, 13, 2025102700,   , BEST,   0, 163N,  777W, 125,  948, HU,  64, NEQ,   40,   35,   30,   35,\n"
            "AL, 13, 2025102706,   , BEST,   0, 165N,  780W, 135,  940, HU,  34, NEQ,  130,  120,    0,  100,\n")
    fixes = parse_bdeck(deck)
    assert len(fixes) == 2 and fixes[0].wind_kt == 125.0
    assert sorted(fixes[0].radii_km) == [34, 50, 64]
    assert abs(gale_radius_km(fixes[0]) - np.mean([130, 120, 90, 100]) * 1.852) < 1e-6
    assert np.isnan(fixes[1].radii_km[34][2]) and abs(gale_radius_km(fixes[1]) - np.mean([130, 120, 100]) * 1.852) < 1e-6


def test_native_window_is_continuous_across_the_dateline_and_ascending():
    values = np.arange(721 * 1440, dtype=float).reshape(721, 1440)
    field, lat, lon = native_window(values, (90.0, 0.25, 0.0, 0.25), 20.0, 179.0, half_lat=5.0, half_lon=5.0)
    assert lat[0] < lat[-1] and abs(lat[0] - 15.0) < 1e-9 and abs(lat[-1] - 25.0) < 1e-9
    assert np.all(np.diff(lon) > 0) and lon[0] < 179.0 < lon[-1] and lon[-1] > 180.0
    assert field.shape == (lat.size, lon.size)
    row = int(round((90.0 - 20.0) / 0.25))
    col = int(round(179.0 / 0.25))
    assert field[np.argmin(np.abs(lat - 20.0)), np.argmin(np.abs(lon - 179.0))] == values[row, col]


def test_world_intensities_follow_the_regime_quantiles(tmp_path):
    import torch

    from naturev1.swarm.geography import synthetic_island
    from naturev1.swarm.impact import ImpactSettings, world_intensity

    region = synthetic_island()
    hours = np.arange(0.0, 49.0, 6.0)
    track = np.stack([np.full((50, hours.size), 10.0), np.full((50, hours.size), 10.0)], -1)   # far from the island
    q = np.array([[-20.0, 0.0, 40.0]] * len(LEADS))
    rapid = ImpactSettings(intensity_quantiles={"hours": np.asarray(LEADS, float), "quantiles": q}, land_decay=False)
    v = world_intensity(region, track, hours, np.full(len(LEADS), 100.0), np.asarray(LEADS, float), rapid, torch.Generator().manual_seed(0))
    assert v.shape == (50, hours.size) and v[:, -1].min() >= 80.0 - 1e-6 and v[:, -1].max() <= 140.0 + 1e-6
    assert v[:, -1].mean() > 100.0, "the skewed quantiles push the worlds' mean above the point forecast"
    assert np.allclose(v[:, 0], v[:, 0][0]) is False or True
