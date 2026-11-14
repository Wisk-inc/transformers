# Copyright 2026 Nathan. Apache-2.0.
"""
Situations unlike anything in training.

A network trained on 2016-2021 has never seen some situations: a stronger hurricane, a record-warm sea, a driving
model upgraded with a different moisture bias. In such inputs its confident probabilities are least trustworthy.
At training time the distribution of every coarse input channel is summarised (mean, standard deviation, 0.5% and
99.5% quantiles); live, the share of each channel's values outside its training range and the mean absolute z-score
give an out-of-distribution score. The forecast is then pulled towards climatology with a weight that grows with the
score (no change at all for ordinary inputs), and the forecast notes name the channels responsible.
"""

from __future__ import annotations

import json

import numpy as np


class InputStats:
    def __init__(self, names: list[str]) -> None:
        self.names = list(names)
        self.mean = self.std = self.low = self.high = None

    def fit(self, samples: np.ndarray) -> InputStats:
        """``samples (N, C, ...)`` of coarse channels from training examples."""
        x = np.moveaxis(np.asarray(samples, np.float64), 1, 0).reshape(len(self.names), -1)
        self.mean, self.std = np.nanmean(x, 1), np.nanstd(x, 1) + 1e-6
        self.low, self.high = np.nanquantile(x, 0.005, 1), np.nanquantile(x, 0.995, 1)
        return self

    def score(self, channels: np.ndarray) -> dict:
        """``channels (F, C, H, W)`` live -> ``{"score", "outside": {name: share}, "weight"}``."""
        x = np.moveaxis(np.asarray(channels, np.float64), 1, 0).reshape(len(self.names), -1)
        outside = ((x < self.low[:, None]) | (x > self.high[:, None])).mean(1)
        z = np.nanmean(np.abs(x - self.mean[:, None]) / self.std[:, None], 1)
        score = float(np.nanmax(outside) * 10 + max(0.0, float(np.nanmax(z)) - 2.0))
        weight = float(np.clip((score - 1.0) / 9.0, 0, 0.5))           # 0 for ordinary inputs, at most 0.5
        flagged = {self.names[k]: float(outside[k]) for k in np.argsort(-outside)[:3] if outside[k] > 0.05}
        return {"score": score, "outside": flagged, "weight": weight}

    @staticmethod
    def shrink(p: np.ndarray, climatology: np.ndarray, weight: float) -> np.ndarray:
        return (1 - weight) * p + weight * climatology

    def to_json(self) -> str:
        return json.dumps({"names": self.names, "mean": self.mean.tolist(), "std": self.std.tolist(),
                           "low": self.low.tolist(), "high": self.high.tolist()})

    @classmethod
    def from_json(cls, text: str) -> InputStats:
        d = json.loads(text)
        s = cls(d["names"])
        s.mean, s.std, s.low, s.high = (np.array(d[k]) for k in ("mean", "std", "low", "high"))
        return s


__all__ = ["InputStats"]
