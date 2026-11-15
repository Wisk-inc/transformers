# Copyright 2026 Nathan. Apache-2.0.
"""
The hand-over audit: do the raw systems know better here, when the forecast leaves the training range?

The record-storm hand-over (:mod:`.override`) rests on a premise: beyond anything the calibration was trained on, the
raw systems are the better guide. That is true where the calibration shrinks a storm it has never seen. It is false
where the raw systems call for heavy rain far more often than it falls, as they do over mountains against a satellite
truth: there the calibration is right to discount them, and the hand-over swaps a good forecast for a worse one.

Which case holds is a fact about a place, and it can be measured before a single forecast is issued. Every training
row has an out-of-fold forecast: calibrated chances from networks that never saw that row's days. The audit applies
the hand-over to those forecasts and compares the Brier score for heavy rain with and without it, over the whole
training period:

    harm = Brier(with hand-over) - Brier(without),      at ``amount`` mm, over all training rows

The hand-over is kept unless it did harm. It costs nothing to run, because the out-of-fold forecasts are the ones the
pool is fitted on, and it is repeated at every refit, so a place can earn the hand-over, or lose it, as data arrives.

The staged-storm experiment (:mod:`.withheld`) asks the same question of the few largest storms, with those storms
taken out of training altogether. The audit asks it of every day the training period held.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..blend import THRESHOLDS, exceed
from .override import agree_share_from, blend_raw, handover
from .support import Support

HEAVY_INDEX = int(np.searchsorted(THRESHOLDS, 25.0))


@dataclass
class Audit:
    """What the hand-over did to the training period's out-of-fold forecasts."""

    keep: bool                      # the verdict: use the hand-over here
    amount: float                   # the amount the verdict is about, mm
    without: float                  # Brier score at ``amount`` over all rows, calibrated
    handed: float                   # the same with the hand-over
    share: float                    # share of rows the hand-over touched (weights undone)
    rows: int                       # rows it touched in the sample
    days: int                       # distinct days among them
    edge_mm: float                  # the edge of the training forecasts
    chance_of_harm: float           # share of resamples of those days in which the hand-over raised the score
    brier: dict                     # {amount: [without, handed]} for other heavy amounts

    def to_json(self) -> dict:
        return asdict(self)

    def line(self) -> str:
        change = 100.0 * (self.handed - self.without) / max(self.without, 1e-12)
        return (f"hand-over audit: {self.rows:,} training rows on {self.days} days beyond {self.edge_mm:.0f} mm; Brier at "
                f"{self.amount:g} mm {self.without:.5f} -> {self.handed:.5f} ({change:+.2f}%); "
                + ("kept" if self.keep else "switched off"))


def weighted_brier(p: np.ndarray, y: np.ndarray, t: float, w: np.ndarray) -> float:
    return float(np.sum(w * (exceed(p, t) - (y >= t)) ** 2) / np.sum(w))


def audit_handover(p_calibrated: np.ndarray, p_raw: np.ndarray, raw_mean: np.ndarray, y: np.ndarray, support: Support,
                   weights: np.ndarray | None = None, day: np.ndarray | None = None, amount: float = 50.0,
                   others=(25.0, 100.0), resamples: int = 500, seed: int = 0) -> Audit:
    """Audit the hand-over on out-of-fold forecasts.

    ``p_calibrated`` and ``p_raw`` are the calibrated and raw consensus chances ``(N, K)`` of the training rows,
    ``raw_mean`` their raw consensus amounts, ``y`` the rain that fell, ``weights`` the rows' weights (rows kept with
    unequal chances carry the inverse chance) and ``day`` anything that is equal for rows of one day (for the
    resampling). The verdict is to keep the hand-over unless it raised the Brier score at ``amount`` mm."""
    ok = np.isfinite(y) & np.isfinite(raw_mean)
    w = (np.ones(y.shape[0]) if weights is None else np.asarray(weights, np.float64)) * ok
    g = handover(support, np.where(ok, raw_mean, 0.0), agree_share_from(p_raw, HEAVY_INDEX))
    touched = ok & (g > 1e-4)
    p_handed = p_calibrated.copy()
    if touched.any():
        p_handed[touched] = blend_raw(p_calibrated[touched], p_raw[touched].astype(np.float32), g[touched])
    yy = np.where(ok, y, 0.0)
    without, handed = weighted_brier(p_calibrated, yy, amount, w), weighted_brier(p_handed, yy, amount, w)
    brier = {f"{t:g}": [weighted_brier(p_calibrated, yy, t, w), weighted_brier(p_handed, yy, t, w)] for t in others}
    days, chance = 0, 0.0
    if touched.any():
        o = yy[touched] >= amount
        diff = w[touched] * ((exceed(p_handed[touched], amount) - o) ** 2 - (exceed(p_calibrated[touched], amount) - o) ** 2)
        label = np.arange(int(touched.sum())) if day is None else np.asarray(day)[touched]
        _, index = np.unique(label, return_inverse=True)
        per_day = np.bincount(index, diff)                          # each day's part of the change in the score
        days = int(per_day.size)
        rng = np.random.default_rng(seed)
        draws = per_day[rng.integers(0, days, (resamples, days))].sum(1)
        chance = float(np.mean(draws > 0))
    return Audit(keep=bool(handed <= without), amount=float(amount), without=without, handed=handed,
                 share=float(w[touched].sum() / max(w.sum(), 1e-12)), rows=int(touched.sum()), days=days,
                 edge_mm=float(support.edge_mm), chance_of_harm=chance, brier=brier)


__all__ = ["HEAVY_INDEX", "Audit", "audit_handover", "weighted_brier"]
