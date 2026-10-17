# Copyright 2026 Nathan. Apache-2.0.
"""
Many teachers, one target: a skill-weighted mixture with an honest variance.

The idea: teachers disagree, and their disagreement is information. Averaging them blindly teaches NatureV1 the
blur of five models; picking one teaches it that model's biases. So the target at each lead, channel and grid
point is a **mixture**:

* the mean is a weighted average of the teachers, with weights per teacher, channel and lead set by how well
  each teacher verified against ERA5 on a calibration period (:func:`skill_weights`, inverse mean-squared error)
  -- GraphCast may be trusted most for 500 hPa height at day 5, HRES for humidity, the ensemble mean of FGN for
  10 m wind at day 7;
* the variance is the mixture's: the members' own ensemble variance (FGN and GenCast archive their spread) plus
  the weighted spread of the teacher means about the mixture mean -- where the teachers disagree, the target is
  soft and NatureV1 is not punished for hedging; where they agree, it is sharp;
* channels a teacher does not provide (Aurora's three levels, FuXi's two) simply drop out of the mixture for
  that channel, and a channel no teacher provides is masked (or falls back to the ERA5 truth when it exists).

Everything is in NatureV1's normalised channel space, so the variance is directly comparable with the model's
predicted ``log_var``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def skill_weights(errors: dict[str, np.ndarray], floor: float = 1e-6) -> dict[str, np.ndarray]:
    """
    Inverse-MSE weights per teacher from calibration errors.

    Args:
        errors: ``{teacher: (S, C) mean squared error against ERA5}`` in normalised units (NaN where the teacher
            does not provide the channel).

    Returns:
        ``{teacher: (S, C)}`` weights summing to one over the teachers that provide each (lead, channel).
    """
    names = list(errors)
    stack = np.stack([np.asarray(errors[n], np.float64) for n in names])
    inv = np.where(np.isfinite(stack), 1.0 / np.maximum(stack, floor), 0.0)
    total = inv.sum(0, keepdims=True)
    w = np.where(total > 0, inv / np.where(total > 0, total, 1.0), 0.0)
    return {n: w[i] for i, n in enumerate(names)}


def combine(means: dict[str, np.ndarray], variances: dict[str, np.ndarray] | None = None,
            weights: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The mixture target.

    Args:
        means: ``{teacher: (S, P, C)}`` aligned, normalised forecasts (NaN where not provided).
        variances: ``{teacher: (S, P, C)}`` ensemble variances in normalised units, where a teacher has them.
        weights: ``{teacher: (S, C)}`` from :func:`skill_weights`; equal weights when omitted.

    Returns:
        ``(mean, variance, covered)``: ``(S, P, C)``, ``(S, P, C)`` and a ``(S, P, C)`` bool of where any teacher
        contributed.
    """
    names = list(means)
    stack = np.stack([np.asarray(means[n], np.float64) for n in names])            # (T, S, P, C)
    ok = np.isfinite(stack)
    if weights is None:
        w = np.ones((len(names),) + stack.shape[1:2] + stack.shape[3:])
    else:
        w = np.stack([np.asarray(weights[n], np.float64) for n in names])          # (T, S, C)
    w = np.broadcast_to(w[:, :, None, :], stack.shape) * ok
    total = w.sum(0)
    covered = total > 0
    norm = np.where(covered, total, 1.0)
    mean = (np.where(ok, stack, 0.0) * w).sum(0) / norm
    spread = (w * np.where(ok, (stack - mean[None]) ** 2, 0.0)).sum(0) / norm
    within = np.zeros_like(mean)
    if variances:
        for i, n in enumerate(names):
            if n in variances:
                v = np.asarray(variances[n], np.float64)
                within = within + np.where(np.isfinite(v), v, 0.0) * w[i] / norm
    variance = spread + within
    return np.where(covered, mean, np.nan), np.where(covered, variance, np.nan), covered


def save_weights(weights: dict[str, np.ndarray], path: str | Path) -> None:
    Path(path).write_text(json.dumps({n: np.asarray(w).tolist() for n, w in weights.items()}))


def load_weights(path: str | Path) -> dict[str, np.ndarray]:
    return {n: np.asarray(w, np.float64) for n, w in json.loads(Path(path).read_text()).items()}


__all__ = ["combine", "load_weights", "save_weights", "skill_weights"]
