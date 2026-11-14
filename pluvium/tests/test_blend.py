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


def test_a_bag_of_nets_stays_monotone_and_saves(tmp_path):
    import torch

    from pluvium.blend import BlendBag, BlendNet

    y, good, poor = _world(8_000, 2)
    x = _rows(good, poor, y.size)
    bag = fit(x, y, min_steps=60, members=3)
    assert isinstance(bag, BlendBag) and len(bag.nets) == 3 and bag.mu.numel() == x.shape[1]
    p = predict(bag, x[:500])
    assert np.all(np.diff(p, axis=1) <= 1e-6)
    singles = np.stack([predict(net, x[:500]) for net in bag.nets])
    np.testing.assert_allclose(p, singles.mean(0), atol=1e-6)                    # the bag averages its nets
    torch.save(bag.state_dict(), tmp_path / "bag.pt")
    again = BlendBag([BlendNet(x.shape[1]) for _ in range(3)])
    again.load_state_dict(torch.load(tmp_path / "bag.pt"))
    again.eval()
    np.testing.assert_allclose(predict(again, x[:500]), p, atol=1e-6)


def test_a_saved_blend_model_loads_and_pools_its_components(tmp_path):
    import torch

    from pluvium.blend import BlendNet, StackPool, feature_names
    from pluvium.blendlive import BlendModel

    systems = ["ens", "gfs"]
    nets = {"ens": BlendNet(len(feature_names(["ens"], {"ens", "gefs"}))),
            "gfs": BlendNet(len(feature_names(["gfs"], {"ens", "gefs"}))),
            "all": BlendNet(len(feature_names(systems, {"ens", "gefs"})))}
    for net in nets.values():
        net.eval()                                                              # as fit() leaves them
    pool = StackPool(3)
    clim = {m: {"clim_p1": np.full((2, 2), 0.3, np.float32), "clim_p10": np.full((2, 2), 0.1, np.float32),
                "clim_p25": np.full((2, 2), 0.02, np.float32), "clim_mean": np.full((2, 2), 1.0, np.float32)}
            for m in range(1, 13)}
    model = BlendModel(nets, pool, systems, {"name": "t", "target_box": [0, 1, 0, 1]}, clim,
                       (np.array([0.0, 0.25]), np.array([0.0, 0.25])), {"components": ["ens", "gfs", "all"]})
    model.save(tmp_path / "blend")
    again = BlendModel.load(tmp_path / "blend")
    assert list(again.components()) == ["ens", "gfs", "all"]
    y, good, poor = _world(50, 4)
    per = {"ens": system_features(good), "gfs": system_features(poor)}
    c = again._clim_at(np.zeros(50), np.zeros(50), 6)
    p = again.chances(per, c, 2.0, 50)
    np.testing.assert_allclose(p, model.chances(per, c, 2.0, 50), atol=1e-6)
    assert p.shape == (50, len(THRESHOLDS)) and np.all(np.diff(p, axis=1) <= 1e-7)
    assert np.all(np.isfinite(again.chances({"gfs": per["gfs"]}, c, 9.0, 50)))      # a system missing
    legacy = tmp_path / "old"                                                        # saved before the pool
    model.save(legacy)
    torch.save({"state": nets["all"].state_dict(), "n_in": nets["all"].mu.numel()}, legacy / "blend.pt")
    old = BlendModel.load(legacy)
    assert old.pool is None and old.chances(per, c, 2.0, 50).shape == (50, len(THRESHOLDS))
