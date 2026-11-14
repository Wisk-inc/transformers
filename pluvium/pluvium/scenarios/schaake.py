# Copyright 2026 Nathan. Apache-2.0.
"""
The Schaake shuffle: spatial structure from past observed days, when no ensemble is at hand.

Ensemble copula coupling (:mod:`.ecc`) needs a physical ensemble for its template. Beyond an ensemble's reach, or in a
deployment that reads only single-run systems, the template can come from history instead (Clark et al. 2004): pick
``M`` past days from the same time of year, rank the observed rain across those days at every place, and hand out the
calibrated quantiles in that order. Each scenario then has the spatial pattern of a real past day.

The patterns are climatological: they know nothing of today's weather. Where an ensemble exists, coupling to it is
better; this is the fallback, and the reference a flow-dependent method should beat.
"""

from __future__ import annotations

import numpy as np

from ..blend import quantile_at
from .ecc import ranks


def pick_days(day_hours: np.ndarray, target_hour: int, count: int, window_days: int = 45,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """Indices of ``count`` past days within ``window_days`` of the target's day of year (any year, before the
    target). Fewer are returned when history is short."""
    rng = rng or np.random.default_rng(0)
    day_hours = np.asarray(day_hours, np.int64)
    doy = (day_hours // 24) % 365.2425
    target = (target_hour // 24) % 365.2425
    gap = np.abs(doy - target)
    gap = np.minimum(gap, 365.2425 - gap)
    ok = np.flatnonzero((gap <= window_days) & (day_hours < target_hour))
    if ok.size <= count:
        return ok
    return np.sort(rng.choice(ok, count, replace=False))


def schaake(p: np.ndarray, history: np.ndarray, tiebreak: np.ndarray | None = None,
            rng: np.random.Generator | None = None) -> np.ndarray:
    """Calibrated scenarios ``(M, N)`` from chances ``p`` (N, K) and ``history`` (M, N): observed rain on M past days
    at the same places."""
    rng = rng or np.random.default_rng(0)
    r = ranks(history, tiebreak, rng)
    return quantile_at(p, ((r + 0.5) / history.shape[0]).T).T


__all__ = ["pick_days", "schaake"]
