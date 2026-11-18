# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Whether a difference between two systems is real: bootstrap confidence intervals of a mean, and paired tests on
the items both systems were measured on (the sign test, exact; the Wilcoxon signed-rank test, normal approximation
with ties and zeros handled). Benchmarks of a few dozen items are noisy; a ranking without these is a guess.
"""

from __future__ import annotations

import math
import random


def finite(xs) -> list[float]:
    return [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]


def mean_ci(xs, level: float = 0.95, resamples: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """``(mean, low, high)`` with a percentile bootstrap interval; NaNs when nothing is finite."""
    xs = finite(xs)
    if not xs:
        return math.nan, math.nan, math.nan
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, m, m
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(resamples))
    lo = means[int((1 - level) / 2 * resamples)]
    hi = means[min(resamples - 1, int((1 + level) / 2 * resamples))]
    return m, lo, hi


def sign_test(diffs) -> float:
    """Two-sided exact p-value that positive and negative differences are equally likely (zeros dropped)."""
    d = [x for x in finite(diffs) if x != 0]
    n, k = len(d), sum(x > 0 for x in d)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def wilcoxon(diffs) -> float:
    """Two-sided p-value of the Wilcoxon signed-rank test (normal approximation, average ranks for ties)."""
    d = [x for x in finite(diffs) if x != 0]
    n = len(d)
    if n < 6:
        return sign_test(d)
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    w = sum(r for r, x in zip(ranks, d, strict=True) if x > 0)
    mean = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mean) / sd if sd > 0 else 0.0
    return min(1.0, math.erfc(abs(z) / math.sqrt(2)))


def paired(a: dict, b: dict) -> list[float]:
    """Differences ``a - b`` over the item ids both dicts hold, in a fixed order."""
    keys = sorted(set(a) & set(b))
    return [a[k] - b[k] for k in keys]


def effect_size(diffs) -> float:
    """Mean difference over its standard deviation (paired Cohen's d)."""
    d = finite(diffs)
    if len(d) < 2:
        return math.nan
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    return m / sd if sd > 0 else math.nan


__all__ = ["effect_size", "finite", "mean_ci", "paired", "sign_test", "wilcoxon"]
