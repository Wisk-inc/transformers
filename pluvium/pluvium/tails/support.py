# Copyright 2026 Nathan. Apache-2.0.
"""
Support: has the calibration ever seen a forecast like this one?

A calibration is a map learned on the forecasts of its training period. Inside the range of those forecasts it is the
best thing known. Outside it, the map is an extrapolation of a small network, and what it extrapolates is the lesson of
ordinary days: big forecasts usually verify smaller. A record storm is exactly where that lesson is wrong.

The support score says how far a forecast lies beyond the training forecasts. It is measured on the raw consensus
amount, in log(1 + mm), against a high quantile of the same quantity over the training rows:

    score = (log1p(amount) - log1p(edge)) / width

It is zero at the edge of the support, negative inside and positive outside. ``edge`` is, by default, the 99.9th
percentile of the raw consensus amount on training rows (one row in a thousand lies beyond it, by construction);
``width`` sets how fast trust is handed over (:mod:`.override`): the score reaches one, and the hand-over is
complete, at ``exp(width)`` times the edge (twice the edge with the default, log 2). Both are chosen by the
withheld-extremes experiment (:mod:`.withheld`), not on test data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def weighted_quantile(values: np.ndarray, q: float, weights: np.ndarray | None = None) -> float:
    """The ``q`` quantile of ``values`` with row weights (rows kept with unequal chances carry the inverse chance)."""
    values = np.asarray(values, np.float64)
    ok = np.isfinite(values)
    values = values[ok]
    if values.size == 0:
        return float("nan")
    w = np.ones(values.size) if weights is None else np.asarray(weights, np.float64)[ok]
    order = np.argsort(values)
    cum = np.cumsum(w[order])
    return float(values[order][min(int(np.searchsorted(cum, q * cum[-1])), values.size - 1)])


@dataclass
class Support:
    """The edge of the training forecasts (mm of raw consensus rain) and the width of the hand-over."""

    edge_mm: float
    width: float = 0.6931

    @classmethod
    def from_rows(cls, raw_mean: np.ndarray, weights: np.ndarray | None = None, quantile: float = 0.999,
                  width: float = 0.6931, floor_mm: float = 20.0) -> Support:
        """From the raw consensus amounts of the training rows. ``floor_mm`` keeps the edge sensible in a dry place
        or a short training period, where the 99.9th percentile may be a few millimetres."""
        edge = weighted_quantile(raw_mean, quantile, weights)
        return cls(max(float(edge) if np.isfinite(edge) else floor_mm, floor_mm), width)

    def score(self, raw_mean: np.ndarray) -> np.ndarray:
        """How far outside the training forecasts each row lies (positive: outside)."""
        return (np.log1p(np.maximum(raw_mean, 0.0)) - np.log1p(self.edge_mm)) / self.width

    def outside(self, raw_mean: np.ndarray) -> np.ndarray:
        return np.asarray(raw_mean) > self.edge_mm


__all__ = ["Support", "weighted_quantile"]
