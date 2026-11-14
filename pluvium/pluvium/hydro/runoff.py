# Copyright 2026 Nathan. Apache-2.0.
"""
How much rain becomes runoff: the SCS curve-number method (USDA Soil Conservation Service, TR-55).

``Q = (P - Ia)^2 / (P - Ia + S)`` for rain ``P`` above the initial abstraction ``Ia = 0.2 S``, with the potential
retention ``S = 25400 / CN - 254`` (mm). The curve number ``CN`` (0-100) describes the land: it is not mapped for the
Caribbean at town scale, so it is estimated from what is known -- steep slopes shed more (+), built-up places shed
more (+, from the density of populated places around), and it is adjusted to the antecedent moisture condition with
the standard conversions (Chow, Maidment and Mays 1988): dry ``CN_I = 4.2 CN / (10 - 0.058 CN)``, wet
``CN_III = 23 CN / (10 + 0.13 CN)``.
"""

from __future__ import annotations

import numpy as np


def curve_number(slope: np.ndarray, urban: np.ndarray, condition: np.ndarray | None = None, base: float = 72.0) -> np.ndarray:
    """``slope`` (m/m), ``urban`` (0-1 share built-up), ``condition`` (1/2/3) -> CN."""
    cn = base + 12.0 * np.clip(np.asarray(slope) / 0.3, 0, 1) + 14.0 * np.clip(np.asarray(urban), 0, 1)
    cn = np.clip(cn, 40, 98)
    if condition is not None:
        c = np.asarray(condition)
        cn = np.where(c == 1, 4.2 * cn / (10 - 0.058 * cn), np.where(c == 3, 23 * cn / (10 + 0.13 * cn), cn))
    return np.clip(cn, 30, 99)


def runoff_depth(rain_mm: np.ndarray, cn: np.ndarray) -> np.ndarray:
    """Runoff (mm) from cumulative rain (mm) for curve numbers ``cn`` (broadcast)."""
    s = 25400.0 / np.asarray(cn, np.float64) - 254.0
    ia = 0.2 * s
    p = np.asarray(rain_mm, np.float64)
    return np.where(p > ia, (p - ia) ** 2 / np.maximum(p - ia + s, 1e-9), 0.0)


def urban_share(town_lat: np.ndarray, town_lon: np.ndarray, population: np.ndarray, radius_km: float = 3.0) -> np.ndarray:
    """A built-up share from the population of places within ``radius_km`` (saturating at 30,000 people)."""
    from scipy.spatial import cKDTree

    la, lo = np.radians(town_lat), np.radians(town_lon)
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    groups = cKDTree(xyz).query_ball_point(xyz, r=radius_km / 6371.0)
    pop = np.array([np.asarray(population)[g].sum() for g in groups], np.float64)
    return np.clip(pop / 30000.0, 0, 1)


__all__ = ["curve_number", "runoff_depth", "urban_share"]
