# Copyright 2026 Nathan. Apache-2.0.
"""
The generalised Pareto distribution: the shape of rain beyond a high threshold.

Whatever the distribution of daily rain, the amounts by which it exceeds a high enough threshold follow, to a good
approximation, a generalised Pareto distribution (Pickands, 1975):

    P(excess > x) = (1 + shape * x / scale) ** (-1 / shape)        (exp(-x / scale) when shape is 0)

``shape`` says how heavy the tail is: 0 is an exponential tail, positive values (0.05 to 0.25 for daily rain in most
climates) mean the largest days are larger than an exponential would allow. It is the one tool for saying something
about amounts rarer than the record holds, and it is honest only a few times beyond the record's length.

The fit is by probability-weighted moments (Hosking and Wallis, 1987): closed-form, stable for the small samples
a single place gives, and nearly as efficient as maximum likelihood for shapes in the range rain has.
"""

from __future__ import annotations

import numpy as np

SHAPE_RANGE = (-0.3, 0.5)


def fit_pwm(excess: np.ndarray, shape_range: tuple = SHAPE_RANGE) -> tuple:
    """``(shape, scale)`` of a generalised Pareto distribution from excesses over a threshold (all positive)."""
    x = np.sort(np.asarray(excess, np.float64))
    x = x[np.isfinite(x) & (x > 0)]
    n = x.size
    if n < 2:
        return 0.0, float(x.mean()) if n else float("nan")
    p = (np.arange(1, n + 1) - 0.35) / n
    a0, a1 = x.mean(), float(np.mean((1.0 - p) * x))
    if a0 - 2.0 * a1 <= 1e-12:                                       # heavier than the method can describe
        shape = shape_range[1]
    else:
        shape = float(np.clip(2.0 - a0 / (a0 - 2.0 * a1), *shape_range))
    return shape, float(a0 * (1.0 - shape))                         # the scale that keeps the mean excess


def survival(x: np.ndarray, shape: float, scale: np.ndarray) -> np.ndarray:
    """P(excess > x). ``x`` and ``scale`` broadcast."""
    x = np.maximum(np.asarray(x, np.float64), 0.0)
    scale = np.maximum(np.asarray(scale, np.float64), 1e-9)
    if abs(shape) < 1e-6:
        return np.exp(-x / scale)
    return np.maximum(1.0 + shape * x / scale, 0.0) ** (-1.0 / shape)   # zero beyond the end of a bounded tail


def quantile(p: np.ndarray, shape: float, scale: np.ndarray) -> np.ndarray:
    """The excess not exceeded with probability ``p``."""
    p = np.clip(np.asarray(p, np.float64), 0.0, 1.0 - 1e-12)
    scale = np.asarray(scale, np.float64)
    if abs(shape) < 1e-6:
        return -scale * np.log1p(-p)
    return scale / shape * ((1.0 - p) ** (-shape) - 1.0)


def mean_excess(shape: float, scale: np.ndarray) -> np.ndarray:
    """The mean of the distribution (finite for shapes below 1)."""
    return np.asarray(scale, np.float64) / (1.0 - shape)


__all__ = ["SHAPE_RANGE", "fit_pwm", "mean_excess", "quantile", "survival"]
