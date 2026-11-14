# Copyright 2026 Nathan. Apache-2.0.
import numpy as np

from pluvium.guard import InputStats, check_driving, check_summary, fallback_summary
from pluvium.guard.freshness import Freshness
from pluvium.hydro import curve_number, peak_timing, runoff_depth, time_of_concentration
from pluvium.hydro.antecedent import from_hourly


def test_driving_fields_are_repaired():
    f = {"t2m": np.full((6, 2, 2), 27.0), "msl": np.full((6, 2, 2), 1012.0), "tcc": np.full((6, 2, 2), 60.0),
         "tp": np.full((6, 2, 2), -1.0), "tcwv": np.full((6, 2, 2), 45.0)}
    f["tcwv"][2] = np.nan
    f["tcwv"][3] = 400.0
    up = {"q850": np.full((6, 2, 2), 14.0), "t850": np.full((6, 2, 2), 291.0)}
    fixed, uf, notes = check_driving(f, up)
    assert fixed["t2m"].min() > 290 and fixed["msl"].min() > 100000 and fixed["tcc"].max() <= 1
    assert fixed["tp"].min() >= 0 and np.isfinite(fixed["tcwv"]).all() and uf["q850"].max() < 0.05
    assert any("Celsius" in n for n in notes) and any("interpolated" in n for n in notes)


def test_freshness_and_drift():
    fr = Freshness(issue_hour=1000)
    assert fr.add("satellite", 999.5) and not fr.add("satellite", 990) and not fr.add("airports", 1003)
    rng = np.random.default_rng(0)
    stats = InputStats(["a", "b"]).fit(rng.normal(size=(50, 2, 4, 4)))
    calm = stats.score(rng.normal(size=(10, 2, 4, 4)))
    storm = stats.score(rng.normal(size=(10, 2, 4, 4)) + np.array([0, 6.0])[None, :, None, None])
    assert calm["weight"] == 0 and storm["weight"] > 0 and "b" in storm["outside"]


def test_summary_consistency_and_fallback():
    s = fallback_summary(0, np.full((3, 72), 2.0), np.tile(np.array([0.9, 0.06, 0.03, 0.01]), (3, 72, 1)))
    s["any_rain"][12] = s["any_rain"][6] - 0.2
    s["p_wet"][0, 0] = 1.4
    notes = check_summary(s)
    assert notes and np.all(s["any_rain"][12] >= s["any_rain"][6] - 1e-9) and s["p_wet"].max() <= 1


def test_hydrology():
    api, amc = from_hourly(np.concatenate([np.zeros((96, 1)), np.full((24, 1), 3.0)]))
    assert api[0] > 60 and amc[0] == 3
    assert runoff_depth(np.array(10.0), 80) == 0 and runoff_depth(np.array(100.0), 80) > 40
    assert curve_number(np.array(0.3), np.array(1.0), np.array(3)) > curve_number(np.array(0.0), np.array(0.0), np.array(1))
    assert time_of_concentration(np.array(500.0), np.array(0.01)) > time_of_concentration(np.array(10.0), np.array(0.2))
    rain = np.zeros((2, 24))
    rain[:, 5:8] = 20.0
    got = peak_timing(rain, np.array([85.0, 85.0]), np.array([5.0, 400.0]), np.array([0.2, 0.01]))
    assert got["peak_hour"][1] > got["peak_hour"][0] >= 5
