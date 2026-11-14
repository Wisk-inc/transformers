# Copyright 2026 Nathan. Apache-2.0.
"""
What the pool may look at besides the forecasts themselves: the season and the weather regime.

Gates are a handful of numbers per forecast row. They change the pool's weights and offsets linearly (before a
softplus), with a ridge penalty, so a gate that carries nothing is ignored.

* **Season**: the first annual harmonic of the day forecast. Two numbers describe any smooth annual cycle, in either
  hemisphere, with no list of months to overfit.
* **Regime**: read off the components' own forecasts, so no extra data are needed and nothing can leak:
  how wet the day looks (the mean log-odds of 1 mm), how much the components disagree about it (their spread), and how
  heavy it looks (the mean log-odds of 25 mm). Widespread steady rain, scattered showers and a cyclone landfall sit in
  different corners of that space, and the systems' relative merits differ between them.
"""

from __future__ import annotations

import numpy as np

from ..blend import THRESHOLDS

SEASON = ("cos_year", "sin_year")
REGIME = ("wet", "disagree", "heavy")
_I1 = int(np.searchsorted(THRESHOLDS, 1.0))
_I25 = int(np.searchsorted(THRESHOLDS, 25.0))


def season_gates(valid_hour: np.ndarray) -> np.ndarray:
    """``(N, 2)``: cosine and sine of the time of year of the day forecast (hours since the epoch)."""
    day = np.asarray(valid_hour, np.float64) / 24.0
    phase = 2.0 * np.pi * (day % 365.2425) / 365.2425
    return np.stack([np.cos(phase), np.sin(phase)], 1).astype(np.float32)


def regime_gates(z: np.ndarray, present: np.ndarray) -> np.ndarray:
    """``(N, 3)`` from the components' log-odds ``z`` (N, C, K) and presence ``(N, C)``: wetness, disagreement and
    heaviness, each scaled to be of order one."""
    w = present.astype(np.float32)
    n = np.maximum(w.sum(1), 1.0)
    z1 = z[:, :, _I1].astype(np.float32)
    z25 = z[:, :, _I25].astype(np.float32)
    wet = (z1 * w).sum(1) / n
    dis = np.sqrt(np.maximum(((z1 - wet[:, None]) ** 2 * w).sum(1) / n, 0.0))
    heavy = (z25 * w).sum(1) / n
    return np.stack([wet / 4.0, dis / 2.0, (heavy + 4.0) / 4.0], 1).astype(np.float32)


def build_gates(kind: str, z: np.ndarray, present: np.ndarray, valid_hour: np.ndarray) -> np.ndarray | None:
    """Gates for a design: ``"none"``, ``"season"``, ``"regime"`` or ``"both"``."""
    if kind == "none":
        return None
    parts = []
    if kind in ("season", "both"):
        parts.append(season_gates(valid_hour))
    if kind in ("regime", "both"):
        parts.append(regime_gates(z, present))
    if not parts:
        raise ValueError(f"unknown gates {kind!r}")
    return np.concatenate(parts, 1)


def gate_names(kind: str) -> tuple:
    return {"none": (), "season": SEASON, "regime": REGIME, "both": SEASON + REGIME}[kind]


__all__ = ["REGIME", "SEASON", "build_gates", "gate_names", "regime_gates", "season_gates"]
