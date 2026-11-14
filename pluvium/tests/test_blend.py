# Copyright 2026 Nathan. Apache-2.0.
"""Pluvium Blend on synthetic forecasts: monotone chances, calibration, quantiles and the features of a system."""

import numpy as np

from pluvium.blend import THRESHOLDS, assemble, exceed, feature_names, fit, predict, quantiles, system_features


def _world(n, seed):
    rng = np.random.default_rng(seed)
    signal = rng.gamma(0.5, 6.0, n).astype(np.float32)                         # the rain the weather "means"
    truth = np.where(rng.random(n) < 0.6, signal * rng.lognormal(0, 0.5, n), 0.0).astype(np.float32)
    good = signal[None] * rng.lognormal(0, 0.3, (20, n))                        # an informative ensemble
    poor = signal[None] * rng.lognormal(0, 1.5, (1, n)) + rng.gamma(0.3, 3.0, (1, n))   # a noisy single model
    return truth, good.astype(np.float32), poor.astype(np.float32)


def _rows(good, poor, n):
    feats = {"ens": system_features(good), "gfs": system_features(poor)}
    clim = {"clim_p1": np.full(n, 0.4), "clim_p10": np.full(n, 0.1), "clim_p25": np.full(n, 0.03), "clim_mean": np.full(n, 1.0)}
    return assemble(feats, ["ens", "gfs"], {"ens"}, clim, 1.0, n)


def test_blend_is_monotone_calibrated_and_uses_the_better_system():
    y, good, poor = _world(60_000, 0)
    x = _rows(good, poor, y.size)
    assert x.shape[1] == len(feature_names(["ens", "gfs"], {"ens"}))
    net = fit(x, y, epochs=8)
    yt, gt, pt = _world(20_000, 1)
    p = predict(net, _rows(gt, pt, yt.size))
    assert np.all(np.diff(p, axis=1) <= 1e-6)                                    # chances fall with the threshold
    for t in (1.0, 10.0):
        pt_ = exceed(p, t)
        assert abs(pt_.mean() - (yt >= t).mean()) < 0.02                         # calibrated in the mean
    clim_bs = np.mean((np.full(yt.size, (y >= 1).mean()) - (yt >= 1)) ** 2)
    assert np.mean((exceed(p, 1.0) - (yt >= 1)) ** 2) < 0.8 * clim_bs            # skilful


def test_quantiles_and_exceed():
    p = np.array([[0.6, 0.55, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.01, 0.005, 0.001]], np.float32)
    q = quantiles(p)[0]
    assert np.all(np.diff(q) >= 0) and q[0] == 0.0 and q[-1] >= 99.0
    assert abs((q >= 1.0).mean() - 0.5) < 0.02                                  # P(rain >= 1 mm) = 0.5
    assert 0.15 <= exceed(p, 12.0)[0] <= 0.2                                     # between the 10 and 15 mm chances
    assert THRESHOLDS[0] == 0.2


def test_system_features_ignore_missing_members():
    m = np.array([[0.0, 5.0], [12.0, np.nan], [np.nan, np.nan]], np.float32)
    f = system_features(m)
    np.testing.assert_allclose(f["p10"], [0.5, 0.0])
    np.testing.assert_allclose(f["mean"], np.log1p([6.0, 5.0]), rtol=1e-6)
