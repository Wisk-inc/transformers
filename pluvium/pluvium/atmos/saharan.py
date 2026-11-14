# Copyright 2026 Nathan. Apache-2.0.
"""
The Saharan air layer (SAL): why a moist-looking summer forecast stays dry.

From June to September, outbreaks of hot, dry, dusty air from the Sahara cross the Atlantic between about 850 and
500 hPa (Dunion and Velden 2004). Over the Caribbean they cap the trade-wind cumulus, dry the mid-levels and come
with a mid-level easterly jet; showers that the low-level moisture would otherwise allow do not happen. Coarse
driving models carry the dryness but a network sees it only as one humidity among many. The index here combines the
three signatures (mid-level dryness, warmth of the 850-700 hPa layer relative to normal, and the easterly jet at 700
hPa), so "SAL overhead" is one clear input.
"""

from __future__ import annotations

import numpy as np

from ..util import calendar
from .thermo import saturation_vapour_pressure, vapour_pressure_from_q

SEASON = (5, 6, 7, 8, 9, 10)
NORMAL_T700 = 283.5          # K, a typical Caribbean summer 700 hPa temperature


def saharan_index(upper: dict, hours: np.ndarray) -> np.ndarray:
    """``(F, H, W)`` in [0, 1]: 0 no SAL, 1 a strong outbreak overhead (zero outside May-October)."""
    rh700 = vapour_pressure_from_q(upper["q700"], 700.0) / saturation_vapour_pressure(upper["t700"])
    rh500 = vapour_pressure_from_q(upper["q500"], 500.0) / saturation_vapour_pressure(upper["t500"])
    dry = np.clip((0.55 - 0.5 * (rh700 + rh500)) / 0.25, 0, 1)
    warm = np.clip((upper["t700"] - NORMAL_T700) / 2.0, 0, 1)
    jet = np.clip((-upper["u700"] - 8.0) / 6.0, 0, 1)
    month = calendar(np.asarray(hours).astype(np.int64))["month"]
    season = np.isin(month, SEASON).astype(np.float64)[:, None, None]
    return (dry * (0.5 + 0.5 * warm) * (0.7 + 0.3 * jet) * season).astype(np.float32)


__all__ = ["saharan_index"]
