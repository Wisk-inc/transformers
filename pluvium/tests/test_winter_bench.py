# Copyright 2026 Nathan. Apache-2.0.
"""The second truth: box means of a fine grid, and winter scored against two truths that disagree."""

import json
from pathlib import Path

import numpy as np

from pluvium.benchmarks import winter_bench
from pluvium.blend import THRESHOLDS, to_logits
from pluvium.sources.mrms_daily import BOX, MRMSDaily, box_means
from pluvium.util import hour_of

K = len(THRESHOLDS)


def test_box_means_average_the_covered_part_of_each_box():
    lat = 30.005 + 0.01 * np.arange(200)                                       # 30 to 32 N at 0.01 degrees
    lon = -100.995 + 0.01 * np.arange(300)                                     # 101 to 98 W
    values = np.tile(np.arange(300, dtype=np.float32)[None], (200, 1))         # rain rises eastwards
    out = box_means(values, lat, lon, np.array([31.0, 31.0]), np.array([-100.0, -99.0]), half=0.125)
    assert abs(out[0] - values[:, 87:113].mean()) < 0.6 and abs(out[1] - out[0] - 100.0) < 1e-3
    holes = values.copy()
    holes[:, :98] = np.nan                                                     # no radar in the west
    out = box_means(holes, lat, lon, np.array([31.0, 31.0, 31.0]), np.array([-100.5, -100.0, -99.0]), half=0.125)
    assert np.isnan(out[0]) and np.isfinite(out[1]) and abs(out[2] - 199.5) < 0.6    # over half covered: kept
    assert out[1] > values[:, 87:113].mean()                                   # only the covered, wetter half
    outside = box_means(values, lat, lon, np.array([40.0]), np.array([-99.0]), half=0.125)
    assert np.isnan(outside[0])
    cells = MRMSDaily("unused", np.array([31.0, 19.0, 54.95]), np.array([-99.0, -99.0, -99.0]), "x")
    assert cells.covered().tolist() == [True, False, False] and BOX[0] == 20.0


class FakeStore:
    """One winter month of daily blocks at lead 2: a sharp component, a blunt one, and two truths."""

    def __init__(self, rng, days=24, n=120):
        self.folder = Path("great_plains")
        self.names = ["sharp", "blunt"]
        self.lat, self.lon = np.linspace(36.0, 44.0, n), np.linspace(-102.0, -94.0, n)
        self.run = hour_of("2025-12-02T00") + 24 * np.arange(days)
        self.lead = np.full(days, 2)
        snow = rng.gamma(0.5, 6.0, (days, n))                                  # what really fell
        self.radar = snow.astype(np.float32)
        self.y = np.where(rng.random((days, n)) < 0.5, 0.0, snow).astype(np.float32)   # the satellite misses half
        sig = lambda noise: np.clip(1 / (1 + np.exp(-(np.log1p(snow * rng.lognormal(0, noise, snow.shape))[..., None]
                                                      - np.log1p(THRESHOLDS)) * 3.0)), 1e-4, 1 - 1e-4)
        self.p = {"sharp": sig(0.2), "blunt": sig(1.5)}

    def quarters(self):
        return [0]

    def quarter(self, q):
        z = np.stack([to_logits(self.p[m]) for m in self.names], 2).astype(np.float16)
        present = np.ones((self.run.size, 2), bool)
        present[:3, 1] = False
        return {"test": np.arange(self.run.size), "test_z": z, "test_present": present}


class FakeRadar:
    def __init__(self, store):
        self.store = store
        self.asked = []

    def covered(self):
        return np.ones(self.store.lat.size, bool)

    def day(self, day_hour, download=True):
        self.asked.append(int(day_hour))
        k = int((day_hour - 24 - self.store.run[0]) // 24)                     # lead 2: valid a day after the run
        return self.store.radar[k] if 0 <= k < self.store.run.size and k != 5 else None


def test_winter_is_scored_against_both_truths_on_the_same_rows():
    rng = np.random.default_rng(0)
    store = FakeStore(rng)
    radar = FakeRadar(store)
    keep = np.clip(0.5 * store.p["sharp"] + 0.5 * store.p["blunt"], 1e-4, 1).astype(np.float16)
    res = winter_bench.run(store, np.arange(store.run.size), keep, radar, "DJF", min_cells=50)
    assert res["days"] == 23 and set(res["groups"]) == {"days 1-3"}            # one day had no radar file
    g = res["groups"]["days 1-3"]
    assert g["rows"] == 23 * 120 and g["skill"]["blunt"]["rows"] < g["rows"]   # the blunt one missed three runs
    assert g["wet_share"]["mrms"]["1"] > 1.5 * g["wet_share"]["cmorph"]["1"]   # the satellite sees half of it
    assert 0.5 < g["truths_agree"]["1"] < 0.9
    for m in ("stack", "sharp", "blunt"):
        assert g["skill"][m]["mrms"]["1"] > g["skill"][m]["cmorph"]["1"]       # everything looks better against radar
    assert g["skill"]["sharp"]["mrms"]["1"] > g["skill"]["stack"]["mrms"]["1"] > g["skill"]["blunt"]["mrms"]["1"]
    assert winter_bench.run(store, np.arange(store.run.size), keep, radar, "JJA")["groups"] == {}
    text = winter_bench.markdown(json.loads(json.dumps(res)), {"sharp": "Sharp", "blunt": "Blunt"})
    assert "Great Plains, DJF: 23 days" in text and "Pluvium Blend" in text and "**" in text and "truths agree" in text
    assert winter_bench.month_of(hour_of("2026-02-28T00")) == 2 and np.isnan(winter_bench.skill(np.zeros(4), np.zeros(4)))
