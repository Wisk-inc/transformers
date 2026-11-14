# Copyright 2026 Nathan. Apache-2.0.
import numpy as np
import torch

from pluvium.engine import LeadShrink, PointRefiner, RegimeExperts, TemporalMixer, knn_on_sphere
from pluvium.prob import ConformalIntervals, GPDTail, SpatialCopula, fit_gpd, roc_auc, sedi
from pluvium.prob.copula import area_probability
from pluvium.prob.scores import brier_decomposition


def test_engine_blocks_keep_shape_and_start_as_identity():
    torch.manual_seed(0)
    h = torch.randn(2, 5, 7, 16)
    idx, km = knn_on_sphere(np.linspace(17, 18, 7), np.linspace(-77, -76, 7), 4)
    r = PointRefiner(16, 2, 4)
    assert torch.allclose(r(h, torch.from_numpy(idx), torch.from_numpy(km)), h + r.layers[0]["ffn"](r.layers[0]["ffn_norm"](h)))
    t = TemporalMixer(16, 2)
    assert torch.allclose(t(h), h)                                   # gate starts at zero
    e = RegimeExperts(16, 3)
    assert torch.allclose(e(h), h) and float(e.balance.detach()) >= -1e-6     # experts start silent
    prior = torch.tensor([[0.9, 0.1], [0.4, 0.6]]).log()
    s = LeadShrink(prior)
    trans = torch.zeros(1, 72, 3, 2, 2)
    out = s(trans)
    assert torch.allclose(out[:, 0], trans[:, 0]) and (out[:, -1] - trans[:, -1]).abs().max() > 0


def test_copula_correlation_falls_with_distance():
    rng = np.random.default_rng(0)
    lat = np.array([18.0, 18.05, 18.9])
    lon = np.array([-77.0, -77.0, -77.0])
    z = SpatialCopula(lat, lon).normals(400, 30.0, rng)
    c = np.corrcoef(z.T)
    assert c[0, 1] > 0.6 and c[0, 2] < c[0, 1]
    wet = np.stack([np.ones(4, bool), np.zeros(4, bool)], 1)
    assert area_probability(wet, np.array(["a", "a"]))["a"] == 1.0


def test_gpd_fit_recovers_parameters_and_return_periods():
    rng = np.random.default_rng(1)
    xi, sigma = 0.15, 8.0
    u = rng.random(20000)
    x = sigma / xi * ((1 - u) ** -xi - 1)
    xi_hat, sigma_hat = fit_gpd(x)
    assert abs(xi_hat - xi) < 0.05 and abs(sigma_hat - sigma) / sigma < 0.08
    tail = GPDTail(20.0, xi, sigma, rate_per_year=5.0)
    assert tail.return_period_years(tail.level(10.0)) == np.float64(10.0) or abs(tail.return_period_years(tail.level(10.0)) - 10) < 1e-6


def test_conformal_intervals_reach_their_coverage():
    rng = np.random.default_rng(2)
    y = rng.gamma(2.0, 5.0, 4000)
    lo, hi = np.full(4000, 4.0), np.full(4000, 12.0)                 # too narrow
    c = ConformalIntervals(alpha=0.2)
    c.fit(lo[:2000], hi[:2000], y[:2000], 24)
    a, b = c.apply(lo[2000:], hi[2000:], 24)
    after = ((y[2000:] >= a) & (y[2000:] <= b)).mean()
    assert c.coverage_before[24] < 0.6 and after >= 0.75 and after > c.coverage_before[24] + 0.2


def test_scores():
    rng = np.random.default_rng(3)
    y = rng.random(5000) < 0.2
    p = np.clip(y * 0.6 + rng.random(5000) * 0.4, 0, 1)
    assert roc_auc(p, y) > 0.8
    d = brier_decomposition(p, y.astype(float))
    assert abs(d["brier"] - (d["reliability"] - d["resolution"] + d["uncertainty"])) < 0.02
    assert sedi(p > 0.5, y) > 0.5
