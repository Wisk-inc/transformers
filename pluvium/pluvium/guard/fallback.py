# Copyright 2026 Nathan. Apache-2.0.
"""
A forecast when the network cannot run.

If the checkpoint will not load, the GPU is gone, or inputs are too broken to repair, a town with no forecast is worse
off than a town with a simple one. The fallback is classical model output statistics: the chance of a wet hour from
a logistic function of the driving model's rain at the point (log scale) and the climatological chance for that
place, month and hour; intensities split by the climatological ratios; windows by treating hours in blocks of three
as independent. Coefficients are those a logistic fit gives on typical Caribbean data (they can be refitted with
:func:`fit_fallback`). The output has the same keys as the network's, and the forecast is labelled "fallback".
"""

from __future__ import annotations

import numpy as np

DEFAULT = {"a": -2.4, "b_rain": 1.9, "b_clim": 4.5}


def _sig(x):
    return 1 / (1 + np.exp(-x))


def fit_fallback(driving_mm: np.ndarray, clim_wet: np.ndarray, observed: np.ndarray, steps: int = 300,
                 lr: float = 0.5) -> dict:
    """Logistic regression by gradient descent on flattened ``(N,)`` arrays (NaN outcomes ignored)."""
    ok = np.isfinite(observed)
    x = np.stack([np.ones(ok.sum()), np.log1p(np.clip(driving_mm[ok], 0, None)), clim_wet[ok]], 1)
    y = observed[ok].astype(np.float64)
    w = np.array([DEFAULT["a"], DEFAULT["b_rain"], DEFAULT["b_clim"]])
    for _ in range(steps):
        p = _sig(x @ w)
        w -= lr * x.T @ (p - y) / y.size
    return {"a": float(w[0]), "b_rain": float(w[1]), "b_clim": float(w[2])}


def fallback_summary(t0: int, driving_mm: np.ndarray, clim_classes: np.ndarray, coef: dict | None = None) -> dict:
    """``driving_mm (P, T)`` driving rain at the points, ``clim_classes (P, T, 4)`` -> a summary dict."""
    c = coef or DEFAULT
    clim_wet = 1 - clim_classes[..., 0]
    p = _sig(c["a"] + c["b_rain"] * np.log1p(np.clip(driving_mm, 0, None)) + c["b_clim"] * clim_wet)
    split = clim_classes[..., 1:] / np.maximum(clim_wet[..., None], 1e-9)
    classes = np.concatenate([1 - p[..., None], p[..., None] * split], -1)
    t = p.shape[1]
    any_rain = {}
    for h in (6, 12, 24, 48, 72):
        if h <= t:
            blocks = p[:, :h].reshape(p.shape[0], -1, 3).max(-1) if h % 3 == 0 else p[:, :h]
            any_rain[h] = 1 - np.prod(1 - blocks, -1)
    expected = np.clip(driving_mm, 0, None) * 0.8 + p * 0.3
    totals = {}
    for h in (24, 48, 72):
        if h <= t:
            m = expected[:, :h].sum(-1)
            totals[h] = {"median": m * 0.7, "p10": m * 0.1, "p90": m * 2.2, "mean": m,
                         **{f"p_ge_{int(x)}": np.clip(m / (2 * x), 0, 1) for x in (10.0, 25.0, 40.0, 50.0, 60.0, 75.0, 100.0)}}
    nan = np.full(p.shape[0], np.nan)
    return {"t0": int(t0), "states": 4, "p_wet": p.astype(np.float32), "classes": classes.astype(np.float32),
            "expected_mm": expected.astype(np.float32), "torrential": np.zeros_like(p, np.float32),
            "p_now": p[:, 0].astype(np.float32), "any_rain": any_rain, "any_heavy": {}, "start_median": nan,
            "start_early": nan, "start_late": nan, "start_median_24": nan, "start_early_24": nan,
            "start_late_24": nan, "start_within": nan, "start_within_24": nan, "heavy_start_median": nan,
            "heavy_within": nan, "stop_median": nan, "thunder": None, "kind": None, "totals": totals, "bursts": {},
            "cause": np.zeros((1, p.shape[0], 6), np.float32), "fallback": True}


__all__ = ["DEFAULT", "fallback_summary", "fit_fallback"]
