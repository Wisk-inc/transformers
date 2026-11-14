# Copyright 2026 Nathan. Apache-2.0.
"""
The fractions skill score: is the rain in the right neighbourhood, and at what scale does the forecast become useful?

A sharp forecast of a storm 40 km from where it fell scores worse, place by place, than a forecast of no storm at
all: it is wrong where it rains and wrong where it does not. The fractions skill score (Roberts and Lean 2008)
compares, at each place, the share of the neighbourhood that passes a threshold in the forecast and in the observed
field. It rises from near zero at the grid scale to its ceiling as the neighbourhood grows; the scale at which it
passes ``0.5 + base rate / 2`` is the smallest scale at which the forecast has useful skill.

Used here on full grids: single runs, scenario fields and the probability-matched mean against the daily truth.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def fractions(field: np.ndarray, threshold: float, size: int) -> np.ndarray:
    """The share of each place's ``size`` x ``size`` neighbourhood that reaches ``threshold`` (NaN counts as dry)."""
    hit = (np.nan_to_num(field, nan=-1.0) >= threshold).astype(np.float64)
    return uniform_filter(hit, size=size, mode="constant") if size > 1 else hit


def fss_parts(forecast: np.ndarray, observed: np.ndarray, threshold: float, size: int,
              valid: np.ndarray | None = None) -> tuple[float, float]:
    """The two sums the score is made of, so that it can be accumulated over many days before dividing:
    ``(sum of squared differences of fractions, sum of squared fractions)``."""
    f = fractions(forecast, threshold, size)
    o = fractions(observed, threshold, size)
    ok = np.isfinite(observed) if valid is None else valid
    return float(((f - o) ** 2)[ok].sum()), float((f ** 2 + o ** 2)[ok].sum())


def fss(forecast: np.ndarray, observed: np.ndarray, threshold: float, size: int) -> float:
    """The fractions skill score of one field (1 is perfect, 0 no overlap at that scale)."""
    num, den = fss_parts(forecast, observed, threshold, size)
    return 1.0 - num / den if den > 0 else float("nan")


class FSSAccumulator:
    """The score over many days, for several thresholds and neighbourhood sizes."""

    def __init__(self, thresholds=(1.0, 10.0, 25.0), sizes=(1, 3, 5, 9)) -> None:
        self.thresholds, self.sizes = tuple(thresholds), tuple(sizes)
        self.num = np.zeros((len(self.thresholds), len(self.sizes)))
        self.den = np.zeros_like(self.num)
        self.base = np.zeros(len(self.thresholds))
        self.count = 0

    def add(self, forecast: np.ndarray, observed: np.ndarray) -> None:
        ok = np.isfinite(observed)
        if not ok.any():
            return
        for a, t in enumerate(self.thresholds):
            for b, s in enumerate(self.sizes):
                num, den = fss_parts(forecast, observed, t, s, ok)
                self.num[a, b] += num
                self.den[a, b] += den
            self.base[a] += float((observed[ok] >= t).mean())
        self.count += 1

    def scores(self) -> np.ndarray:
        """``(thresholds, sizes)`` scores."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.den > 0, 1.0 - self.num / self.den, np.nan)

    def useful_scale(self, step_km: float) -> dict:
        """``{threshold: km}``: the smallest neighbourhood width at which the score passes ``0.5 + base rate / 2``
        (interpolated between sizes; NaN when it never does)."""
        out = {}
        s = self.scores()
        for a, t in enumerate(self.thresholds):
            target = 0.5 + 0.5 * self.base[a] / max(self.count, 1)
            row = s[a]
            above = np.flatnonzero(row >= target)
            if above.size == 0:
                out[t] = float("nan")
            elif above[0] == 0:
                out[t] = self.sizes[0] * step_km
            else:
                k = above[0]
                w = (target - row[k - 1]) / max(row[k] - row[k - 1], 1e-9)
                out[t] = float((self.sizes[k - 1] + w * (self.sizes[k] - self.sizes[k - 1])) * step_km)
        return out


__all__ = ["FSSAccumulator", "fractions", "fss", "fss_parts"]
