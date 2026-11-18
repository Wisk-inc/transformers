# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Dynamic time warping between two feature sequences, with an optional Sakoe-Chiba band."""

from __future__ import annotations

import numpy as np


def cost_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance between every frame of ``a (N, D)`` and of ``b (M, D)``."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    sq = (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T
    return np.sqrt(np.maximum(sq, 0.0))


def dtw(cost: np.ndarray, band: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """The cheapest monotonic path through ``cost (N, M)``: ``(rows, cols, total cost)``. ``band`` (a share of the
    longer length) keeps the path near the diagonal, which stops a degenerate path from hiding a timing error."""
    cost = np.asarray(cost, np.float64)
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.zeros(0, int), np.zeros(0, int), float("nan")
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    width = None if band is None else max(int(band * max(n, m)), abs(n - m) + 1)
    for i in range(1, n + 1):
        lo, hi = 1, m
        if width is not None:
            centre = int(round(i * m / n))
            lo, hi = max(1, centre - width), min(m, centre + width)
        row = cost[i - 1, lo - 1 : hi]
        prev = np.minimum(acc[i - 1, lo - 1 : hi], acc[i - 1, lo : hi + 1])
        cur = np.empty(hi - lo + 1)
        left = acc[i, lo - 1]
        for k in range(hi - lo + 1):                    # the left neighbour depends on the cell just filled
            left = row[k] + min(prev[k], left)
            cur[k] = left
        acc[i, lo : hi + 1] = cur
    rows, cols = [n - 1], [m - 1]
    i, j = n, m
    while i > 1 or j > 1:
        steps = [(acc[i - 1, j - 1], i - 1, j - 1), (acc[i - 1, j], i - 1, j), (acc[i, j - 1], i, j - 1)]
        _, i, j = min(steps, key=lambda s: s[0])
        rows.append(i - 1)
        cols.append(j - 1)
    return np.array(rows[::-1]), np.array(cols[::-1]), float(acc[n, m])


__all__ = ["cost_matrix", "dtw"]
