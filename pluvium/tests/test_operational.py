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


def test_evaluate_end_to_end_on_synthetic_systems(tmp_path):
    """A skilful system beats a noisy one, blends at least match their parts, and every score is finite."""
    from pluvium.benchmarks.operational import PLUVIUM, Operational, OperationalConfig, markdown

    rng = np.random.default_rng(4)
    R, P = 50, 60
    inits = hour_of("2025-04-01T00") + 24 * np.arange(R)
    signal = rng.gamma(0.6, 8.0, size=(R, 3, P)).astype(np.float32)                 # the "true" daily rain
    obs = signal * rng.lognormal(0, 0.3, size=signal.shape).astype(np.float32)
    cfg = OperationalConfig(root=str(tmp_path), geography="", checkpoint="", systems=("ens", "gfs"),
                            test_period=("2025-05-01T00", "2025-05-20T00"), window_days=20, lag_days=3)
    op = Operational.__new__(Operational)
    op.cfg, op.root = cfg, tmp_path
    op.land = np.arange(P)
    op.lat, op.lon = np.linspace(17.8, 18.3, P), np.linspace(-78, -76.5, P)
    op.eval_inits, op.test_inits = inits, inits[(inits >= hour_of("2025-05-01T00")) & (inits < hour_of("2025-05-20T00"))]
    op.train_inits = inits[:1]
    op.thr = (1.0, 10.0)

    def system(noise, members):
        m = signal[:, None] * rng.lognormal(0, noise, size=(R, members, 3, P))
        point = np.stack([(m >= t).mean(1) for t in op.thr], 2).astype(np.float32)
        return {"point": point, "nbhd": point.copy(), "amount": m.mean(1).astype(np.float32),
                "members": m.astype(np.float32), "inits": inits}

    fake = {PLUVIUM: system(0.3, 40), "ens": system(0.8, 20), "gfs": system(1.2, 1)}
    op.predictors = lambda name: fake.get(name)
    op.truth = lambda _: obs
    op.reference = lambda sel: np.broadcast_to(np.stack([(obs >= t).mean((0, 2)) for t in op.thr], 1)[None, :, :, None],
                                               (len(sel), 3, 2, P)).astype(np.float32)
    res = op.evaluate(resamples=200)
    s = res["systems"]
    assert len(res["test_runs"]) == 19
    for d in ("1", "2", "3"):
        assert s[PLUVIUM]["days"][d]["1"]["bs"] < s["gfs"]["days"][d]["1"]["bs"]
        assert np.isfinite(s["blend_all"]["days"][d]["10"]["bs"])
    assert res["versus"]["gfs"]["day1_1mm"]["delta_bs"] < 0
    text = markdown(res)
    assert "Pluvium" in text and "ECMWF ENS" in text


def test_calibrators_recover_a_logistic_truth():
    from pluvium.benchmarks.operational import calibrator

    rng = np.random.default_rng(5)
    p = rng.beta(0.6, 2.0, 30_000)                                   # a forecaster's chances
    truth = 1 / (1 + np.exp(-(0.4 + 0.8 * np.log(np.clip(p, 0.01, 0.99) / (1 - np.clip(p, 0.01, 0.99))))))
    y = (rng.random(p.size) < truth).astype(float)
    for method in ("logistic", "isotonic"):
        fit = calibrator(p, y, "point", method)
        assert np.abs(fit(p) - truth).mean() < 0.03, method
    amount = rng.gamma(0.5, 6.0, 30_000)
    y = (rng.random(amount.size) < 1 - np.exp(-amount / 5)).astype(float)
    fit = calibrator(amount, y, "amount", "logistic")
    assert fit(np.array([0.0]))[0] < fit(np.array([20.0]))[0]
