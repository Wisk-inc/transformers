# Copyright 2026 Nathan. Apache-2.0.
"""The scenario benchmark on a synthetic archive: coupled scenarios keep the ensemble's structure and the calibrated
amounts, uncoupled ones keep only the amounts."""

import numpy as np

from pluvium.benchmarks.scenario_bench import KINDS, LEAD_GROUPS, Accumulator, markdown, members_at, run
from pluvium.blend import THRESHOLDS
from pluvium.scenarios.scores import box_groups, nearby_pairs
from pluvium.util import hour_of


def band_field(rng, members, h, w, shift=0.0):
    """Members with a rain band across a grid at slightly different places: coherent in space."""
    rows = np.arange(h)[None, :, None]
    centre = rng.uniform(0.3 * h, 0.7 * h, members)[:, None, None] + shift
    return (40.0 * np.exp(-0.5 * ((rows - centre) / 2.5) ** 2) * np.ones((1, 1, w))).astype(np.float32)


class FakeStore:
    def __init__(self, rng, blocks=12, h=14, w=12):
        lat, lon = np.meshgrid(np.linspace(10.0, 13.25, h), np.linspace(-80.0, -77.25, w), indexing="ij")
        self.lat, self.lon = lat.ravel(), lon.ravel()
        self.run = hour_of("2025-04-02T00") + 48 * (np.arange(blocks) // 3)
        self.lead = 1 + np.arange(blocks) % 3
        self.grid = (np.linspace(10.0, 13.25, h), np.linspace(-80.0, -77.25, w))
        self.fields = {int(r): band_field(rng, 20, h, w) for r in np.unique(self.run)}
        self.y = np.stack([band_field(rng, 1, h, w)[0].ravel() for _ in range(blocks)])   # the truth: one more band
        from pathlib import Path

        self.folder = Path("synthetic")


class FakeReader:
    """Cumulative rain per member: the same daily band on each of three days."""

    def __init__(self, store):
        self.store = store

    def load(self, region, system, run):
        day = self.store.fields.get(int(run))
        if day is None:
            return None
        values = np.stack([day * (k + 1) for k in range(3)], 1)           # (M, S, H, W) cumulative
        return {"values": values, "lat": self.store.grid[0], "lon": self.store.grid[1]}


def climate_chances(rng, n, h, w, samples=600):
    """Calibrated chances per place: where a band may lie is unknown, so every row of the grid has a modest chance."""
    fields = band_field(rng, samples, h, w).reshape(samples, -1)
    return np.stack([(fields >= t).mean(0) for t in THRESHOLDS], 1).astype(np.float16)


def test_members_are_read_at_places_for_a_lead_day():
    rng = np.random.default_rng(0)
    store = FakeStore(rng)
    z = FakeReader(store).load("x", "ens", int(store.run[0]))
    m1, m3 = members_at(z, 1, store.lat, store.lon), members_at(z, 3, store.lat, store.lon)
    assert m1.shape == (20, store.lat.size) and np.allclose(m1, m3, atol=1e-4)         # daily, not cumulative
    assert members_at(z, 4, store.lat, store.lon) is None                              # the run stops at day 3
    z["values"][:17] = np.nan
    assert members_at(z, 1, store.lat, store.lon) is None                              # too few whole members


def test_coupled_scenarios_keep_structure_and_amounts():
    rng = np.random.default_rng(1)
    store = FakeStore(rng)
    blocks = np.arange(store.run.size)
    p = np.stack([climate_chances(rng, store.lat.size, 14, 12) for _ in blocks])
    res = run(store, blocks, p, FakeReader(store), "x", every=1)
    row = res["groups"]["days 1-3"]
    assert set(row) == set(KINDS) and row["coupled"]["days"] == 12 and res["areas"] >= 1
    assert abs(row["coupled"]["crps"] - row["uncoupled"]["crps"]) < 0.08 * row["coupled"]["crps"]   # the same amounts
    assert row["coupled"]["areal_crps"] < 0.8 * row["uncoupled"]["areal_crps"]         # but the right totals
    assert row["coupled"]["variogram"] < row["uncoupled"]["variogram"]
    assert row["coupled"]["energy"] < row["uncoupled"]["energy"]
    assert 0.7 < row["coupled"]["sharpness"] < 1.3                                     # as varied as the weather
    assert res["groups"]["days 4-7"]["coupled"] == {}
    text = markdown([res])
    assert "Synthetic" in text and "days 11-14" in text and text.count("|") > 40 and list(LEAD_GROUPS)[0] in text


def test_accumulator_ignores_days_without_truth():
    rng = np.random.default_rng(2)
    lat, lon = np.repeat(np.linspace(0, 3, 8), 8), np.tile(np.linspace(0, 3, 8), 8)
    pairs, groups = nearby_pairs(lat, lon, 150.0), box_groups(lat, lon, 2.0)
    acc = Accumulator()
    acc.add(rng.gamma(1.0, 5.0, (10, 64)), np.full(64, np.nan), pairs, groups)
    assert acc.n == 0 and acc.result() == {}
    acc.add(rng.gamma(1.0, 5.0, (10, 64)), rng.gamma(1.0, 5.0, 64), pairs, groups)
    out = acc.result()
    assert out["days"] == 1 and out["crps"] > 0 and out["sharpness"] > 0
