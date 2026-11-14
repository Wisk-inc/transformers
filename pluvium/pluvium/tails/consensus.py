# Copyright 2026 Nathan. Apache-2.0.
"""
The raw consensus: what the systems say before anyone calibrates them.

Calibration is learned from the past, and the past holds few record storms. The raw systems are physics, or AI trained
on forty years of reanalysis, and they can forecast an event unlike anything in a one-year training set. Their plain
agreement is therefore kept alongside the calibrated forecast, as a distribution in the same form: the chance of
reaching each threshold.

Every system has one vote, however many members it runs: a system's chance of a threshold is the share of its members
that reach it (0 or 1 for a single run), and the consensus is the mean over the systems present. Ordinary days make
this forecast badly calibrated, which is why it is not the forecast. On the day the calibrated forecast has left the
range it was trained on, it is the best evidence there is (:mod:`.support`, :mod:`.override`).
"""

from __future__ import annotations

import numpy as np

from ..blend import THRESHOLDS


def system_chances(members: np.ndarray, thresholds=THRESHOLDS) -> np.ndarray:
    """One system's chance of each threshold at each place: ``(N, K)`` from members ``(M, N)`` (NaN members ignored)."""
    m = np.asarray(members, np.float32)
    ok = np.isfinite(m)
    n = np.maximum(ok.sum(0), 1)
    return np.stack([((m >= t) & ok).sum(0) / n for t in thresholds], 1).astype(np.float32)


def raw_consensus(raw: dict, n: int, thresholds=THRESHOLDS) -> tuple[np.ndarray, np.ndarray]:
    """``(chances (n, K), mean amount (n,))`` from ``{system: members (M, n)}``; zeros when no system is present."""
    if not raw:
        return np.zeros((n, len(thresholds)), np.float32), np.zeros(n, np.float32)
    chances = np.mean([system_chances(m, thresholds) for m in raw.values()], 0)
    means = np.mean([np.nanmean(np.maximum(np.asarray(m, np.float32), 0.0), 0) for m in raw.values()], 0)
    return chances.astype(np.float32), np.nan_to_num(means).astype(np.float32)


def agreement(raw: dict, threshold: float, share: float = 0.5) -> np.ndarray:
    """How many systems call for ``threshold`` mm at each place (a system calls for it when at least ``share`` of its
    members reach it)."""
    if not raw:
        return np.zeros(0, np.int64)
    calls = [np.nanmean(np.asarray(m, np.float32) >= threshold, 0) >= share for m in raw.values()]
    return np.sum(calls, 0).astype(np.int64)


__all__ = ["agreement", "raw_consensus", "system_chances"]
