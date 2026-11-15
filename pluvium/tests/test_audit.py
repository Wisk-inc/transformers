# Copyright 2026 Nathan. Apache-2.0.
"""The hand-over audit: kept where the raw systems are the better guide beyond the training range, switched off where
they overcall, and the trust the raw systems earned at the top of the training forecasts."""

import numpy as np

from pluvium.blend import THRESHOLDS
from pluvium.tails.audit import Audit, audit_handover, weighted_brier
from pluvium.tails.override import blend_raw, tail_trust
from pluvium.tails.support import Support


def chances(amount: np.ndarray, sharp: float = 6.0) -> np.ndarray:
    """Chances of each threshold for a forecast amount: a steep curve around it."""
    return (1 / (1 + np.exp(-(np.log1p(amount)[:, None] - np.log1p(THRESHOLDS)[None]) * sharp))).astype(np.float32)


def year(overcall: float, shrink: bool, n: int = 60_000, seed: int = 0) -> tuple:
    """A training period: ordinary days and a few storms. The raw systems forecast ``overcall`` times the rain; the
    calibrated forecast is right, except that with ``shrink`` it never goes beyond 30 mm."""
    rng = np.random.default_rng(seed)
    day = rng.integers(0, 300, n)
    truth = rng.gamma(0.5, 8.0, n)
    truth[day < 6] *= 6.0                                                 # six storm days
    seen = truth * rng.lognormal(0, 0.25, n)
    calibrated = chances(np.minimum(seen, 30.0) if shrink else seen, 3.0)
    raw_mean = (seen * overcall).astype(np.float32)
    return calibrated, chances(raw_mean), raw_mean, truth.astype(np.float32), day


def test_audit_keeps_the_handover_where_calibration_shrinks_storms():
    p, p_raw, raw_mean, y, day = year(overcall=1.0, shrink=True)
    support = Support.from_rows(raw_mean, None, 0.99)
    a = audit_handover(p, p_raw, raw_mean, y, support, None, day)
    assert isinstance(a, Audit) and a.keep and a.handed < a.without
    assert a.rows > 50 and 0.0 < a.share < 0.02 and a.days >= 3
    assert a.chance_of_harm < 0.1                                         # however the storm days are resampled
    assert a.brier["100"][1] < a.brier["100"][0]
    assert "kept" in a.line() and a.to_json()["keep"] is True


def test_audit_switches_the_handover_off_where_the_raw_systems_overcall():
    p, p_raw, raw_mean, y, day = year(overcall=3.0, shrink=False)
    support = Support.from_rows(raw_mean, None, 0.99)
    a = audit_handover(p, p_raw, raw_mean, y, support, None, day)
    assert not a.keep and a.handed > a.without and a.chance_of_harm > 0.9
    assert "switched off" in a.line()
    # the verdict is about rows beyond the edge only: everywhere else the two forecasts are the same forecast
    inside = raw_mean <= support.edge_mm
    assert weighted_brier(p[inside], y[inside], 50.0, np.ones(inside.sum())) > 0


def test_audit_without_evidence_keeps_the_handover_and_honours_weights():
    p, p_raw, raw_mean, y, day = year(overcall=1.0, shrink=True, n=5_000)
    far = Support(edge_mm=1e6)                                            # nothing in training lies beyond it
    a = audit_handover(p, p_raw, raw_mean, y, far, None, day)
    assert a.keep and a.rows == 0 and a.days == 0 and a.handed == a.without and a.chance_of_harm == 0.0
    y[:10] = np.nan                                                       # rows without truth are left out
    w = np.ones(y.size)
    w[::2] = 3.0
    support = Support.from_rows(raw_mean, w, 0.99)
    b = audit_handover(p, p_raw, raw_mean, y, support, w, day)
    assert np.isfinite(b.without) and np.isfinite(b.handed) and b.rows > 0


def test_tail_trust_discounts_systems_that_overcall_and_the_guard_only_raises():
    honest = year(overcall=1.0, shrink=False)
    loud = year(overcall=3.0, shrink=False)
    k50 = int(np.searchsorted(THRESHOLDS, 50.0))
    t_honest = tail_trust(honest[1], honest[3], honest[2])
    t_loud = tail_trust(loud[1], loud[3], loud[2])
    assert t_honest.shape == THRESHOLDS.shape and np.all((t_honest >= 0) & (t_honest <= 1))
    assert t_honest[k50] > 0.65 and t_loud[k50] < 0.6 * t_honest[k50]     # even honest systems overshoot at the very top
    assert np.all(np.diff(t_loud[10:]) <= 1e-6)                          # the rarer the amount, the less they earned
    assert np.all(tail_trust(honest[1][:50], honest[3][:50], honest[2][:50]) == 1.0)   # too few rows to judge
    p, p_raw = loud[0][:200], loud[1][:200]
    g = np.full(200, 0.5, np.float32)
    guarded = blend_raw(p, p_raw, g, t_loud, only_raise=True)
    plain = blend_raw(p, p_raw, g)
    assert np.all(guarded >= p - 1e-6) and np.all(np.diff(guarded, axis=1) <= 1e-6)
    assert guarded[:, k50].mean() <= plain[:, k50].mean() + 1e-6          # the discount takes the overcall back out
