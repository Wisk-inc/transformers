# Copyright 2026 Nathan. Apache-2.0.
"""The statistics of the operational benchmark and the baseline readers' parsing, on synthetic data."""

import json

import numpy as np

from pluvium.benchmarks.operational import block_bootstrap, isotonic, issue_times, logistic_apply, logistic_fit, roc_area
from pluvium.sources.baselines import _ecmwf_rows, daily
from pluvium.util import hour_of


def test_isotonic_is_monotone_and_calibrated():
    rng = np.random.default_rng(0)
    x = rng.random(50_000)
    truth = 0.1 + 0.8 * x ** 2
    y = (rng.random(x.size) < truth).astype(float)
    kx, ky = isotonic(x, y)
    assert np.all(np.diff(kx) > 0) and np.all(np.diff(ky) >= -1e-12)
    grid = np.linspace(0.05, 0.95, 10)
    assert np.abs(np.interp(grid, kx, ky) - (0.1 + 0.8 * grid ** 2)).max() < 0.06


def test_isotonic_handles_ties_and_constant_predictors():
    x = np.r_[np.zeros(900), np.ones(100)]
    y = np.r_[np.zeros(850), np.ones(50), np.ones(80), np.zeros(20)]
    kx, ky = isotonic(x, y)
    assert abs(np.interp(0.0, kx, ky) - 50 / 900) < 1e-9
    assert abs(np.interp(1.0, kx, ky) - 0.8) < 1e-9
    kx, ky = isotonic(np.full(10, 0.3), np.r_[np.ones(3), np.zeros(7)])
    assert abs(np.interp(0.3, kx, ky) - 0.3) < 1e-9


def test_logistic_recovers_coefficients():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40_000, 2))
    beta = np.array([1.5, -0.7, 0.3])
    p = 1 / (1 + np.exp(-(X @ beta[:2] + beta[2])))
    y = (rng.random(p.size) < p).astype(float)
    got = logistic_fit(X, y, l2=0.0)
    assert np.abs(got - beta).max() < 0.06
    assert np.abs(logistic_apply(got, X) - p).mean() < 0.02


def test_roc_area_matches_pairs():
    rng = np.random.default_rng(2)
    p = rng.integers(0, 5, 300) / 4.0                 # many ties
    y = rng.random(300) < 0.3
    pos, neg = p[y], p[~y]
    brute = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (pos.size * neg.size)
    assert abs(roc_area(p, y) - brute) < 1e-12


def test_block_bootstrap():
    rng = np.random.default_rng(3)
    d = rng.normal(-0.01, 0.002, 61)
    mean, lo, hi, share = block_bootstrap(d, resamples=500)
    assert abs(mean - d.mean()) < 1e-12 and lo < mean < hi and share == 1.0


def test_daily_and_issue_times():
    cum = np.array([[[1.0, 0.0], [3.0, 2.0], [3.0, 7.0]]])        # (members, 24/48/72 h, points)
    np.testing.assert_allclose(daily(cum), [[[1.0, 0.0], [2.0, 2.0], [0.0, 5.0]]])
    t = issue_times(("2025-05-01T06", "2025-05-04T00"))
    assert t.tolist() == [hour_of("2025-05-02T00"), hour_of("2025-05-03T00")]


def test_ecmwf_index_members_sorted():
    rows = [{"param": "tp", "number": "21", "_offset": 100, "_length": 10, "type": "pf"},
            {"param": "2t", "number": "1", "_offset": 0, "_length": 5, "type": "pf"},
            {"param": "tp", "_offset": 5, "_length": 20, "type": "cf"},
            {"param": "tp", "number": "4", "_offset": 200, "_length": 30, "type": "pf"}]
    text = "\n".join(json.dumps(r) for r in rows)
    assert _ecmwf_rows(text) == [(0, 5, 24), (4, 200, 229), (21, 100, 109)]
