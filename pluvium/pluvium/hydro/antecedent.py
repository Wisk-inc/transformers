# Copyright 2026 Nathan. Apache-2.0.
"""
How wet the ground is before the storm.

A dry hillside soaks up the first 30-50 mm of a storm; after a week of showers it sheds almost everything. The
antecedent precipitation index (Kohler and Linsley 1951) carries that memory: ``API_d = k API_(d-1) + P_d`` with a
daily decay ``k`` near 0.85 in the humid tropics. The SCS method's antecedent moisture condition (I dry, II normal,
III wet) follows from the last five days' rain (thresholds for the growing season: under 36 mm, 36-53 mm, over
53 mm).
"""

from __future__ import annotations

import numpy as np


def antecedent_index(daily_mm: np.ndarray, k: float = 0.85) -> np.ndarray:
    """``daily_mm (D, ...)`` oldest first -> API at the end of the last day ``(...)``."""
    api = np.zeros(np.shape(daily_mm)[1:])
    for d in range(np.shape(daily_mm)[0]):
        api = k * api + np.nan_to_num(np.asarray(daily_mm)[d])
    return api


def moisture_condition(five_day_mm: np.ndarray) -> np.ndarray:
    """SCS antecedent moisture condition: 1 dry, 2 normal, 3 wet."""
    x = np.asarray(five_day_mm, np.float64)
    return np.where(x < 36.0, 1, np.where(x <= 53.0, 2, 3))


def from_hourly(hourly_mm: np.ndarray, k: float = 0.85) -> tuple[np.ndarray, np.ndarray]:
    """``hourly_mm (H, ...)`` (oldest first, a multiple of 24 hours) -> ``(API, AMC)``."""
    h = np.asarray(hourly_mm, np.float64)
    days = h.shape[0] // 24
    daily = np.nan_to_num(h[: days * 24]).reshape((days, 24) + h.shape[1:]).sum(1)
    return antecedent_index(daily, k), moisture_condition(daily[-5:].sum(0))


__all__ = ["antecedent_index", "from_hourly", "moisture_condition"]
