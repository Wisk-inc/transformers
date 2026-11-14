# Copyright 2026 Nathan. Apache-2.0.
"""
One sharp map from many scenarios: the probability-matched mean and the medoid.

A mean of scenarios places rain well and gets its amounts wrong: peaks are flattened and light rain is spread too
wide. Two standard ways to publish a single map that still looks like weather:

* the **probability-matched mean** (Ebert 2001) keeps the mean's pattern and restores the scenarios' amounts. The
  places are ranked by the mean; the pooled scenario values are ranked too; the wettest place gets the amount at the
  matching rank of the pool, and so on down. The map has the mean's placement and a realistic distribution of
  amounts;
* the **medoid** is the scenario closest to all the others: a single, physically coherent member that best stands for
  the set. Unlike any mean it is a field the atmosphere could actually produce.
"""

from __future__ import annotations

import numpy as np


def probability_matched_mean(scenarios: np.ndarray) -> np.ndarray:
    """``(N,)`` from scenarios ``(M, N)``: the mean's spatial ranking with the pooled scenarios' amounts."""
    s = np.asarray(scenarios, np.float64)
    m, n = s.shape
    mean = np.nanmean(s, 0)
    pooled = np.sort(np.nan_to_num(s).ravel())[::-1]                 # descending, M * N values
    amounts = pooled.reshape(n, m).mean(1)                           # every M-th block's mean: N amounts, descending
    order = np.argsort(-mean, kind="stable")
    out = np.empty(n)
    out[order] = amounts
    return out.astype(np.float32)


def medoid(scenarios: np.ndarray, power: float = 0.5) -> int:
    """The index of the scenario with the smallest mean distance to the others. Distances are taken on
    ``amount ** power`` (0.5 by default), so a few extreme places do not decide alone."""
    s = np.nan_to_num(np.asarray(scenarios, np.float64)) ** power
    sq = (s * s).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * s @ s.T, 0.0)
    return int(np.argmin(np.sqrt(d2).mean(1)))


__all__ = ["medoid", "probability_matched_mean"]
