# Copyright 2026 Nathan. Apache-2.0.
"""
Showing the network the rare events.

Each candidate issue time gets a weight from what happened in its 72-hour window, averaged over the target cells:
the share of heavy hours (at least 7.6 mm/h) relative to its mean over all windows, and the share of wet hours. Issue
times are drawn in proportion to ``1 + a * heavy / mean_heavy + b * wet / mean_wet``. Because that changes the
distribution the loss is averaged over, each drawn example carries an importance weight ``(mean_w / w) ** lam``:
``lam = 1`` is exactly unbiased (only variance is reduced), ``lam = 0`` is plain oversampling; the default 0.5 keeps
most of the emphasis on rare events while the loss stays close to the true objective.
"""

from __future__ import annotations

import numpy as np

from ..util import log


class RareEventSampler:
    def __init__(self, starts: np.ndarray, weights: np.ndarray, temper: float = 0.5) -> None:
        self.starts = np.asarray(starts)
        w = np.maximum(np.asarray(weights, np.float64), 1e-6)
        self.p = w / w.sum()
        self.importance = (w.mean() / w) ** temper

    @classmethod
    def from_truth(cls, starts: np.ndarray, read, hours_ahead: int = 72, heavy_weight: float = 2.0,
                   wet_weight: float = 1.0, temper: float = 0.5, chunk: int = 2000) -> RareEventSampler:
        """``read(hours) -> (len(hours), N)`` truth (mm/h, NaN missing) over the cells used for weighting."""
        starts = np.asarray(starts)
        hours = np.arange(starts.min(), starts.max() + hours_ahead)
        heavy = np.zeros(hours.size)
        wet = np.zeros(hours.size)
        for a in range(0, hours.size, chunk):
            x = read(hours[a: a + chunk])
            ok = np.isfinite(x)
            n = np.maximum(ok.sum(1), 1)
            heavy[a: a + chunk] = (np.nan_to_num(x) >= 7.6).sum(1) / n
            wet[a: a + chunk] = (np.nan_to_num(x) >= 0.2).sum(1) / n
        ch, cw = np.concatenate([[0], np.cumsum(heavy)]), np.concatenate([[0], np.cumsum(wet)])
        i0 = starts - hours[0]
        win_heavy = (ch[i0 + hours_ahead] - ch[i0]) / hours_ahead
        win_wet = (cw[i0 + hours_ahead] - cw[i0]) / hours_ahead
        w = 1 + heavy_weight * win_heavy / max(win_heavy.mean(), 1e-9) + wet_weight * win_wet / max(win_wet.mean(), 1e-9)
        sampler = cls(starts, w, temper)
        top = np.argsort(-w)[: max(1, len(w) // 20)]
        log(f"curriculum: {len(starts):,} issue times; the top 5% by rare-event weight are drawn "
            f"{sampler.p[top].sum() / (len(top) / len(w)):.1f}x as often as uniform sampling would")
        return sampler

    def draw(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        """``n`` issue times and their importance weights."""
        k = rng.choice(self.starts.size, n, p=self.p)
        return self.starts[k], self.importance[k]


__all__ = ["RareEventSampler"]
