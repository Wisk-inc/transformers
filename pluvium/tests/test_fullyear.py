# Copyright 2026 Nathan. Apache-2.0.
"""The full-year benchmark's scores, on synthetic numbers with known answers."""

import numpy as np

from pluvium.benchmarks.fullyear import acc_per_run, best_cut, crps_quantiles, csi, daily_members, sharpness
from pluvium.verify import crps_samples


def test_daily_members_from_cumulative():
    cum = np.array([[[[1.0]], [[4.0]], [[4.0]]]])                    # (members, steps, H, W)
    z = {"values": cum}
    assert daily_members(z, 1)[0, 0, 0] == 1.0 and daily_members(z, 2)[0, 0, 0] == 3.0
    assert daily_members(z, 3)[0, 0, 0] == 0.0 and daily_members(z, 4) is None


def test_csi_and_best_cut():
    y = np.array([True, True, False, False, True])
    p = np.array([0.9, 0.6, 0.4, 0.1, 0.3])
    assert csi(p, y, 0.5) == 2 / 3
    assert best_cut(p, y) <= 0.3                                      # catching the third event pays


def test_acc_and_sharpness():
    rng = np.random.default_rng(0)
    ob = rng.gamma(1.0, 5.0, 400)
    clim = np.full(400, 5.0)
    blocks = np.repeat([1, 2], 200)
    assert acc_per_run(ob, ob, clim, blocks) > 0.999
    assert acc_per_run(rng.permutation(ob), ob, clim, blocks) < 0.2
    assert abs(sharpness(ob, ob, blocks) - 1.0) < 1e-9
    assert sharpness(0.5 * ob + 0.5 * ob.mean(), ob, blocks) < 0.6         # a blurred field is less sharp


def test_crps_of_quantiles_matches_samples():
    rng = np.random.default_rng(1)
    draws = rng.gamma(0.8, 4.0, (2000, 50))
    levels = (np.arange(100) + 0.5) / 100
    q = np.quantile(draws, levels, axis=0).T                           # (50, 100)
    y = rng.gamma(0.8, 4.0, 50)
    np.testing.assert_allclose(crps_quantiles(q, y), crps_samples(draws, y), rtol=0.03, atol=0.02)
