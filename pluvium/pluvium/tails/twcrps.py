# Copyright 2026 Nathan. Apache-2.0.
"""
Scoring the tail on its own: the threshold-weighted CRPS.

The CRPS adds up the Brier scores of every amount, and on most days most of that sum comes from small amounts. A
forecast can be excellent by CRPS and poor where it matters most. Brier scores at 50 or 100 mm look at one amount
each. The threshold-weighted CRPS (Gneiting and Ranjan, 2011) is the proper score in between: the same sum, but only
over amounts above a threshold, so it scores the whole upper tail and nothing else.

With chances given at fixed thresholds the integral is a sum, with the chance taken as constant between thresholds,
exactly as in :func:`pluvium.pool.cv.crps_from_chances` (which is this score with the threshold at zero).
"""

from __future__ import annotations

import numpy as np

from ..blend import THRESHOLDS


def tail_widths(threshold: float, thresholds: np.ndarray = THRESHOLDS) -> np.ndarray:
    """The width (mm) each threshold stands for in the integral above ``threshold``: zero for thresholds at or below
    it, and the part of the first interval above it for the next one."""
    edges = np.concatenate([[0.0], np.asarray(thresholds, np.float64)])
    lo, hi = np.maximum(edges[:-1], threshold), edges[1:]
    return np.clip(hi - lo, 0.0, None)


def tw_crps(p: np.ndarray, y: np.ndarray, threshold: float = 25.0, thresholds: np.ndarray = THRESHOLDS) -> np.ndarray:
    """Threshold-weighted CRPS per row (mm) for chances ``p`` (N, K) of exceeding ``thresholds`` and rain ``y`` (N,):
    the Brier scores of every amount above ``threshold``, integrated. NaN where ``y`` is unknown."""
    width = tail_widths(threshold, thresholds)
    truth = (np.asarray(y)[:, None] >= np.asarray(thresholds)[None]).astype(np.float64)
    out = ((np.asarray(p, np.float64) - truth) ** 2 * width[None]).sum(1)
    return np.where(np.isfinite(y), out, np.nan)


def tail_skill(p: np.ndarray, reference: np.ndarray, y: np.ndarray, threshold: float = 25.0) -> float:
    """Skill of ``p`` against ``reference`` chances in the tail: 1 - twCRPS / twCRPS of the reference."""
    ok = np.isfinite(y)
    mine, ref = tw_crps(p[ok], y[ok], threshold).mean(), tw_crps(reference[ok], y[ok], threshold).mean()
    return float(1.0 - mine / ref) if ref > 0 else float("nan")


__all__ = ["tail_skill", "tail_widths", "tw_crps"]
