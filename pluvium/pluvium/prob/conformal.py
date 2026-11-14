# Copyright 2026 Nathan. Apache-2.0.
"""
Prediction intervals that keep their promise.

The 10th-90th percentile of the sampled totals is an 80% interval only if the model's spread is right. Conformalised
quantile regression (Romano, Patterson and Candes 2019) makes it right on average: on held-out forecasts, the score
``E = max(low - y, y - high)`` measures how far each outcome fell outside its interval; the interval is then widened
(or narrowed) by the ``(1 - alpha)(1 + 1/n)`` quantile of those scores. With exchangeable data the corrected interval
covers the truth with probability at least ``1 - alpha``. Fitted separately for each total window (24/48/72 h).
"""

from __future__ import annotations

import json

import numpy as np


class ConformalIntervals:
    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self.margin: dict[int, float] = {}
        self.coverage_before: dict[int, float] = {}

    def fit(self, low: np.ndarray, high: np.ndarray, observed: np.ndarray, window: int) -> float:
        ok = np.isfinite(observed)
        lo, hi, y = low[ok], high[ok], observed[ok]
        if y.size < 20:
            return 0.0
        scores = np.maximum(lo - y, y - hi)
        level = min(1.0, (1 - self.alpha) * (1 + 1.0 / y.size))
        self.margin[int(window)] = float(np.quantile(scores, level))
        self.coverage_before[int(window)] = float(((y >= lo) & (y <= hi)).mean())
        return self.margin[int(window)]

    def apply(self, low: np.ndarray, high: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
        m = self.margin.get(int(window), 0.0)
        return np.maximum(low - m, 0.0), high + m

    def to_json(self) -> str:
        return json.dumps({"alpha": self.alpha, "margin": self.margin, "coverage_before": self.coverage_before})

    @classmethod
    def from_json(cls, text: str) -> ConformalIntervals:
        d = json.loads(text)
        c = cls(d.get("alpha", 0.2))
        c.margin = {int(k): v for k, v in d.get("margin", {}).items()}
        c.coverage_before = {int(k): v for k, v in d.get("coverage_before", {}).items()}
        return c


__all__ = ["ConformalIntervals"]
