# Copyright 2026 Nathan. Apache-2.0.
"""The tail's own scores: threshold-weighted CRPS, warning tiers, and the tail benchmark on a small store."""

import numpy as np

from pluvium.benchmarks import tail_bench
from pluvium.blend import THRESHOLDS
from pluvium.pool.cv import crps_from_chances
from pluvium.tails import alerts
from pluvium.tails.twcrps import tail_skill, tail_widths, tw_crps
from pluvium.util import hour_of

K = len(THRESHOLDS)


def chances(signal, sharp=4.0):
    return np.clip(1 / (1 + np.exp(-(np.log1p(signal)[..., None] - np.log1p(THRESHOLDS)) * sharp)), 1e-5, 1 - 1e-5)


def test_tail_crps_is_the_crps_above_a_threshold():
    rng = np.random.default_rng(0)
    y = rng.gamma(0.5, 12.0, 3000)
    p = chances(y * rng.lognormal(0, 0.5, y.size))
    np.testing.assert_allclose(tw_crps(p, y, 0.0), crps_from_chances(p, y), rtol=1e-9)      # threshold 0: the CRPS
    assert np.all(tw_crps(p, y, 25.0) <= tw_crps(p, y, 0.0) + 1e-12)
    w = tail_widths(25.0)
    assert w[THRESHOLDS <= 25.0].sum() == 0.0 and abs(w.sum() - (150.0 - 25.0)) < 1e-9
    assert abs(tail_widths(30.0).sum() - 120.0) < 1e-9 and tail_widths(30.0)[THRESHOLDS == 35.0][0] == 5.0
    perfect = (y[:, None] >= THRESHOLDS[None]).astype(float)
    assert tw_crps(perfect, y, 25.0).max() == 0.0
    y_nan = y.copy()
    y_nan[:5] = np.nan
    assert np.isnan(tw_crps(p, y_nan, 25.0)[:5]).all()
    dry = y < 5.0                                                                           # dry days cost nothing here
    low = chances(np.full(dry.sum(), 0.5))
    assert tw_crps(low, y[dry], 25.0).mean() < 1e-3 < crps_from_chances(low, y[dry]).mean()
    signal = rng.gamma(0.5, 12.0, 3000)                                                     # a forecast that knows
    rain = signal * rng.lognormal(0, 0.3, signal.size)
    climate = np.tile((rain[:, None] >= THRESHOLDS[None]).mean(0), (rain.size, 1))          # against the base rates
    assert tail_skill(chances(signal, 4.0), climate, rain, 25.0) > 0.3
    assert abs(tail_skill(climate, climate, rain, 25.0)) < 1e-12


def test_tiers_and_what_they_meant():
    rng = np.random.default_rng(1)
    signal = rng.gamma(0.4, 20.0, 20_000)
    y = signal * rng.lognormal(0, 0.5, signal.size)
    p = chances(signal, 2.5)
    levels = alerts.tier(p, 50.0)
    assert set(np.unique(levels)) <= {0, 1, 2, 3} and (levels == 3).sum() > 0
    v = alerts.verify(levels, y, 50.0)
    assert v["watch"]["issued"] >= v["warning"]["issued"] >= v["extreme"]["issued"] > 0
    assert v["watch"]["hit_rate"] < v["warning"]["hit_rate"] < v["extreme"]["hit_rate"]     # higher tiers mean more
    assert v["watch"]["caught"] > v["warning"]["caught"] > v["extreme"]["caught"]           # and catch less
    assert v["extreme"]["hit_rate"] > 5 * v["base_rate"] and v["events"] == int((y >= 50).sum())
    none = alerts.verify(np.zeros(100, np.int8), np.zeros(100), 50.0)
    assert none["watch"]["issued"] == 0 and np.isnan(none["watch"]["hit_rate"]) and np.isnan(none["watch"]["caught"])
    assert alerts.wording(0) == "" and "likely" in alerts.wording(2) and "50 mm" in alerts.wording(3)
    assert "lower bound" in alerts.wording(0, True) and alerts.wording(3, True).count(";") == 1
    rel = alerts.reliability(p, y, 50.0)
    assert rel[0]["observed"] < rel[-1]["observed"] and sum(r["n"] for r in rel) == y.size


class FakeStore:
    def __init__(self, rng, blocks=24, n=200):
        from pathlib import Path

        self.folder = Path("testland")
        self.run = hour_of("2025-10-20T00") + 48 * (np.arange(blocks) // 6)
        self.lead = np.tile([1, 2, 3, 5, 9, 12], blocks // 6)
        self.signal = rng.gamma(0.4, 25.0, (blocks, n))
        self.y = (self.signal * rng.lognormal(0, 0.4, (blocks, n))).astype(np.float32)
        self.y[0, :10] = np.nan
        self.raw_mean = (self.signal * 1.3).astype(np.float32)
        self._raw = chances(self.signal * rng.lognormal(0, 1.2, (blocks, n)), 8.0).astype(np.float32)   # sharp, noisy

    def raw_p(self, block):
        return self._raw[int(block)]


def test_tail_bench_scores_blend_against_the_raw_consensus():
    rng = np.random.default_rng(2)
    store = FakeStore(rng)
    blocks = np.arange(24)
    p = chances(store.signal, 3.0).astype(np.float16)
    case = str(np.datetime64(int(store.run[0]), "h"))[:10]
    res = tail_bench.run(store, blocks, p, case_days=(case,))
    assert set(res["groups"]) == {"days 1-3", "days 4-7", "days 8-10", "days 11-14"}
    g = res["groups"]["days 1-3"]
    assert g["rows"] == 12 * 200 - 10 and g["skill_tw25"] > 0.1 and g["skill_bs50"] > 0.1    # calibrated beats raw
    assert g["alerts"]["extreme"]["hit_rate"] > g["alerts"]["watch"]["hit_rate"]
    assert abs(sum(r["n"] for r in g["reliability"]) - g["rows"]) == 0
    assert len(res["cases"]) == 1 and res["cases"][0]["lead"] == 1 and 0 <= res["cases"][0]["blend_p50"] <= 1
    text = tail_bench.markdown([res])
    assert "Testland" in text and "Extreme" in text
    table = tail_bench.case_table(res, case)
    assert table[0].startswith("Observed on") and table[-1].startswith("| 1 |")
    assert tail_bench.case_table(res, "1999-01-01") == []
