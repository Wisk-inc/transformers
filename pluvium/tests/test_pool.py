# Copyright 2026 Nathan. Apache-2.0.
"""The pools: the gated batch pool, its gates, the online aggregator and month-wise cross-validation."""

import numpy as np

from pluvium.blend import THRESHOLDS, StackPool, fit_stack, predict_stack, to_logits
from pluvium.pool import (
    OnlineAggregator,
    build_gates,
    fit_pool,
    gate_names,
    leave_month_out,
    predict_pool,
    run_online,
    summarise,
)
from pluvium.pool.cv import brier_skill, calendar_month, crps_from_chances, month_of
from pluvium.pool.gates import regime_gates, season_gates
from pluvium.util import hour_of


def world(n, seed, winter=None):
    """Truth and three components: sharp, noisy, and one that is good only in winter (when ``winter`` marks rows)."""
    rng = np.random.default_rng(seed)
    signal = rng.gamma(0.5, 6.0, n)
    y = (signal * rng.lognormal(0, 0.2, n)).astype(np.float32)          # what falls follows the signal closely

    def comp(noise):
        f = signal * rng.lognormal(0, noise, n)
        return np.clip(1 / (1 + np.exp(-(np.log1p(f)[:, None] - np.log1p(THRESHOLDS[None])) * 2.5)), 1e-4, 1 - 1e-4)

    winter = np.zeros(n, bool) if winter is None else winter
    third = np.where(winter[:, None], comp(0.15), comp(2.5))
    z = np.stack([to_logits(comp(0.4)), to_logits(comp(1.5)), to_logits(third)], 1)
    return z, y


def test_gated_pool_with_three_bands_matches_the_first_pool():
    z, y = world(20_000, 0)
    present = np.ones((y.size, 3))
    lead = np.full(y.size, 3.0)
    old = predict_stack(fit_stack(z, present, lead, y), z, present, lead)
    new = predict_pool(fit_pool(z, present, lead, y, bands=StackPool.BANDS, smooth=0.0, iters=150), z, present, lead)
    assert np.abs(old - new).max() < 0.02
    assert np.all(np.diff(new, axis=1) <= 1e-7)


def test_season_gate_learns_a_component_that_is_good_only_in_winter():
    n = 40_000
    rng = np.random.default_rng(1)
    valid = hour_of("2024-01-01T00") + 24 * rng.integers(0, 365, n)
    winter = np.isin(calendar_month(valid), (12, 1, 2))
    z, y = world(n, 2, winter)
    present, lead = np.ones((n, 3)), np.full(n, 2.0)
    half = np.arange(n) < n // 2
    plain = fit_pool(z[half], present[half], lead[half], y[half], iters=120)
    gates = build_gates("season", z, present, valid)
    gated = fit_pool(z[half], present[half], lead[half], y[half], gates[half], iters=120)
    test = ~half & winter
    p_plain = predict_pool(plain, z[test], present[test], lead[test])
    p_gated = predict_pool(gated, z[test], present[test], lead[test], gates[test])
    k = int(np.searchsorted(THRESHOLDS, 1.0))
    bs = lambda p: np.mean((p[:, k] - (y[test] >= 1.0)) ** 2)            # noqa: E731
    assert bs(p_gated) < 0.8 * bs(p_plain)                               # winter is forecast clearly better
    import torch

    with torch.no_grad():
        w = gated.weights(torch.tensor([2 / 14.0, 2 / 14.0]),
                          torch.from_numpy(season_gates(np.array([hour_of("2024-01-15T00"), hour_of("2024-07-15T00")]))))
    assert float(w[0, k, 2]) > 10 * float(w[1, k, 2])                    # the third component counts in January
    shy = fit_pool(z[half], present[half], lead[half], y[half], gates[half], iters=120, ridge=1.0)
    assert bs(predict_pool(shy, z[test], present[test], lead[test], gates[test])) > bs(p_gated)   # too much ridge


def test_gates_have_the_documented_shapes():
    z, _ = world(100, 3)
    present = np.ones((100, 3), bool)
    valid = np.full(100, hour_of("2025-03-21T00"))
    assert build_gates("none", z, present, valid) is None
    assert build_gates("season", z, present, valid).shape == (100, 2)
    assert build_gates("both", z, present, valid).shape == (100, 5) and len(gate_names("both")) == 5
    r = regime_gates(z, present)
    assert r.shape == (100, 3) and np.isfinite(r).all()
    year = season_gates(np.array([hour_of("2025-01-01T00"), hour_of("2025-07-02T00")]))
    assert year[0, 0] > 0.99 and year[1, 0] < -0.99                      # opposite phases half a year apart


def test_online_aggregator_follows_the_expert_that_becomes_best():
    n_days, per_day = 120, 300
    agg = OnlineAggregator(3, eta=2.0, share=0.02)
    k1 = int(np.searchsorted(THRESHOLDS, 1.0))
    mix_loss, best_loss = [], []
    for day in range(n_days):
        z, y = world(per_day, 100 + day, winter=np.full(per_day, day >= 60))   # the third expert is good from day 60
        p = 1 / (1 + np.exp(-z))
        present, lead = np.ones((per_day, 3), bool), np.full(per_day, 2)
        mix = agg.combine(p, present, lead)
        truth = (y >= 1.0).astype(float)
        if day >= 90:
            mix_loss.append(np.mean((mix[:, k1] - truth) ** 2))
            best_loss.append(np.mean((p[:, 2, k1] - truth) ** 2))
        if day == 59:
            before = agg.weights()[0, 0].copy()
        agg.update(p, present, lead, y)
    after = agg.weights()[0, 0]
    assert before[0] > 0.6 and after[2] > 0.6                           # the weight moved to the newly best expert
    assert np.mean(mix_loss) < 1.15 * np.mean(best_loss)                # and the mixture is close to it
    assert np.all(np.diff(mix, axis=1) <= 1e-7)
    assert agg.weights().shape == (4, 3, 3) and np.allclose(agg.weights().sum(-1), 1.0)
    state = agg.state()
    other = OnlineAggregator(3)
    other.load_state(state)
    np.testing.assert_allclose(other.weights(), agg.weights())


def test_online_aggregation_is_causal_and_absent_experts_are_skipped():
    z, y = world(600, 5)
    p = 1 / (1 + np.exp(-z))
    present = np.ones((600, 3), bool)
    present[:, 1] = False                                                # the second expert never shows up
    issued = np.repeat(np.arange(6) * 24, 100)
    known = issued + 48
    agg = OnlineAggregator(3)
    out = run_online(agg, p, present, np.full(600, 1), y, issued, known)
    assert out.shape == (600, len(THRESHOLDS)) and agg.updates == 4      # days 0-3 were known by the last issue
    first = OnlineAggregator(3).combine(p[:100], present[:100], np.full(100, 1))
    np.testing.assert_allclose(out[:100], first, atol=1e-6)              # the first forecasts saw no truth
    assert 0.2 < agg.weights()[0, 0, 1] < 0.5                             # an absent expert is neither punished nor rewarded


def test_leave_month_out_and_scores():
    n = 12_000
    rng = np.random.default_rng(6)
    valid = hour_of("2024-04-01T00") + 24 * rng.integers(0, 365, n)
    z, y = world(n, 7)
    present, lead = np.ones((n, 3)), rng.integers(1, 15, n).astype(np.float32)
    months = month_of(valid)
    assert np.unique(months).size == 12
    calls = []

    def fit(train):
        calls.append(int(train.sum()))
        return fit_pool(z[train], present[train], lead[train], y[train], iters=30)

    p = leave_month_out(fit, lambda pool, test: predict_pool(pool, z[test], present[test], lead[test]), months, n)
    assert len(calls) == 12 and max(calls) < n
    s = summarise(p, y, lead, valid)
    assert s["all"]["1"] > 0.2 and "days 1-3" in s and "DJF days 1-3" in s
    perfect = (y[:, None] >= THRESHOLDS[None]).astype(np.float32)
    assert brier_skill(perfect, y)["10"] > 0.999 and crps_from_chances(perfect, y).max() < 1e-9
    worse = crps_from_chances(np.full((n, len(THRESHOLDS)), 0.5, np.float32), y).mean()
    assert worse > crps_from_chances(p, y).mean() > 0
