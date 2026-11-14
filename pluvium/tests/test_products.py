# Copyright 2026 Nathan. Apache-2.0.
import numpy as np
import torch

from pluvium.calibrate import Calibrator, pava
from pluvium.flood import accumulate, flow_network
from pluvium.online import OnlineCorrector
from pluvium.products import summarize


def fake_out(p=3, t=24, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"trans": torch.randn(1, t, p, k, k, generator=g), "init": torch.randn(1, p, k, generator=g),
            "amt_mu": torch.randn(1, t, p, k - 1, generator=g) * 0.3 + torch.tensor([-0.3, 1.5, 2.7]),
            "amt_sig": torch.full((1, t, p, k - 1), 0.5), "thunder": torch.randn(1, t, p, generator=g) - 3,
            "kind": torch.randn(1, t, p, 3, generator=g), "cause": torch.randn(1, 10, p, 6, generator=g)}


def test_summary_is_consistent_with_its_samples():
    out = fake_out()
    s = summarize(out, 0, samples=4000, keep_samples=True)
    assert np.allclose(s["classes"].sum(-1), 1, atol=1e-5)
    assert np.all(s["any_rain"][24] >= s["p_wet"][:, :24].max(-1) - 1e-6)
    # the sampled paths' mean total agrees with the chain's expected amounts
    assert np.allclose(s["samples24"].mean(0), s["expected_mm"][:, :24].sum(-1), rtol=0.12, atol=0.3)


def test_mixture_of_identical_components_is_the_component():
    out = fake_out(seed=3)
    a, b = summarize(out, 0, samples=50), summarize([out, out], 0, samples=50)
    assert np.allclose(a["p_wet"], b["p_wet"], atol=1e-6)
    assert np.allclose(a["any_rain"][24], b["any_rain"][24], atol=1e-6)


def test_isotonic_calibration_improves_overconfident_forecasts():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0, 1, 20000)
    y = (rng.random(20000) < truth).astype(float)
    said = truth**0.5
    xs, ys = pava(said, y)
    assert np.all(np.diff(ys) >= -1e-12)
    assert np.mean((np.interp(said, xs, ys) - y) ** 2) < np.mean((said - y) ** 2)
    cal = Calibrator.fit(np.tile(said[:, None], (1, 6)), np.tile(y[:, None], (1, 6)), np.ones((20000, 6), bool))
    assert cal.to_json()


def test_online_corrector_learns_a_wet_bias():
    rng = np.random.default_rng(0)
    c = OnlineCorrector(lr=0.2)
    for _ in range(40):
        p = np.full((5, 24), 0.6)
        obs = (rng.random((5, 24)) < 0.2).astype(float)
        c.update(p, obs, ["JM/St. Ann"] * 5, np.full(5, -77.0), 0)
    fixed = c.apply_hourly(np.full((2, 24), 0.6), ["JM/St. Ann", "JM/Portland"], np.full(2, -77.0), 0)
    assert fixed[0].mean() < 0.45                    # corrected towards the observed 20%
    assert fixed[1].mean() < 0.6                     # Portland borrows Jamaica's correction


def test_flow_accumulation_on_a_valley():
    y, x = np.mgrid[0:20, 0:20]
    elev = np.abs(x - 10) * 10.0 + y * 2.0 + 1.0                   # a valley draining south to the sea
    land = y > 0
    elev[~land] = 0.0
    receiver, levels = flow_network(elev, land)
    acc = accumulate(np.where(land, 1.0, 0.0).ravel(), receiver, levels).reshape(20, 20)
    assert acc[1, 10] > 0.5 * land.sum()                           # most of the land drains through the valley mouth
