# Copyright 2026 Nathan. Apache-2.0.
"""
The hand-over: when a forecast leaves the range calibration was trained on, the raw consensus takes it back.

    final = (1 - g) * calibrated + g * raw consensus,      g = clip(support score, 0, 1) * agreement

``g`` is the share of trust handed to the raw systems. Inside the training range the support score is negative and
``g`` is exactly zero, so every ordinary day, and every score computed on ordinary days, is untouched by
construction. Beyond the edge ``g`` rises linearly in the log of the amount; with the default width the raw consensus
is the forecast at twice the edge.

``agreement`` guards against one wild system: it is the share of the systems that themselves call for heavy rain
(here, raw amount at least ``heavy`` mm), scaled so that half of them agreeing is enough. One model drawing a bullseye
where the others are dry does not take the forecast away from the calibration.

The mixture is taken threshold by threshold, in probability. Both parts are decreasing in the threshold, so the result
is too, and the expected amount is the same mixture of the two expected amounts.

**The guarded hand-over.** The raw systems are not equally trustworthy everywhere. Over mountains they call for
heavy rain far more often than it is observed, and the calibration is right to discount them. So the raw consensus
can be discounted before it is handed to: :func:`tail_trust` measures, on the wettest 1% of the training forecasts,
how often each amount was observed relative to how often the raw consensus called for it, and the hand-over then
goes to ``trust * raw``. With ``only_raise`` the hand-over can lift a chance and never lower it: its purpose is the
storm the calibration has shrunk.
"""

from __future__ import annotations

import numpy as np

from ..blend import THRESHOLDS
from .support import Support, weighted_quantile


def handover(support: Support, raw_mean: np.ndarray, agree_share: np.ndarray | None = None,
             cap: float = 0.95) -> np.ndarray:
    """The share of trust handed to the raw consensus per row, in ``[0, cap]``: zero inside the training range."""
    g = np.clip(support.score(raw_mean), 0.0, 1.0)
    if agree_share is not None:
        g = g * np.clip(2.0 * np.asarray(agree_share, np.float64), 0.0, 1.0)
    return np.clip(g, 0.0, cap).astype(np.float32)


def blend_raw(p_calibrated: np.ndarray, p_raw: np.ndarray, g: np.ndarray, trust: np.ndarray | None = None,
              only_raise: bool = False) -> np.ndarray:
    """The final chances ``(N, K)``: a mixture of the calibrated and raw chances with weight ``g`` (N,) on the raw.

    ``trust`` (K,) discounts the raw chances first (:func:`tail_trust`). With ``only_raise`` a chance is never
    lowered by the hand-over."""
    g = np.asarray(g, np.float32)[:, None]
    target = p_raw if trust is None else p_raw * np.asarray(trust, np.float32)[None]
    if only_raise:
        target = np.maximum(target, p_calibrated)
    out = (1.0 - g) * p_calibrated + g * target
    return np.minimum.accumulate(out, axis=1)


def tail_trust(p_raw: np.ndarray, y: np.ndarray, raw_mean: np.ndarray, weights: np.ndarray | None = None,
               top: float = 0.01, pseudo: float = 20.0) -> np.ndarray:
    """How far the raw consensus has earned trust at the top of the training forecasts, per threshold, in ``[0, 1]``.

    On the rows whose raw consensus amount is in the wettest ``top`` share, it is the number of times each amount
    was observed over the number of times the raw consensus called for it. A threshold with few calls leans on the
    threshold below it (``pseudo`` calls' worth), so the rarest amounts do not get a trust of their own from a
    handful of rows. 1 means the raw consensus was right or too cautious there; 0.4 means it called for the amount
    two and a half times as often as it fell."""
    ok = np.isfinite(y) & np.isfinite(raw_mean)
    w = np.ones(y.shape[0]) if weights is None else np.asarray(weights, np.float64)
    if ok.sum() < 100:
        return np.ones(p_raw.shape[1], np.float32)
    edge = weighted_quantile(raw_mean[ok], 1.0 - top, w[ok])
    rows = ok & (raw_mean >= edge)
    if rows.sum() < 50:
        return np.ones(p_raw.shape[1], np.float32)
    wr = w[rows][:, None]
    observed = (wr * (y[rows][:, None] >= THRESHOLDS[None])).sum(0)
    called = (wr * p_raw[rows].astype(np.float64)).sum(0)
    scale = wr.sum() / rows.sum()                                 # pseudo calls in the same units as the weights
    out, below = np.ones(p_raw.shape[1]), 1.0
    for k in range(p_raw.shape[1]):
        out[k] = np.clip((observed[k] + pseudo * scale * below) / (called[k] + pseudo * scale), 0.0, 1.0)
        below = out[k]
    return out.astype(np.float32)


def agree_share_from(p_raw: np.ndarray, heavy_index: int) -> np.ndarray:
    """The share of the systems that call for heavy rain, read off the consensus chances: the consensus chance of the
    ``heavy_index`` threshold itself. Each system votes with the share of its members that reach the threshold, so
    this is exactly the share of systems calling for it when the systems are single runs or unanimous ensembles."""
    return np.clip(p_raw[:, heavy_index], 0.0, 1.0)


__all__ = ["agree_share_from", "blend_raw", "handover", "tail_trust"]
