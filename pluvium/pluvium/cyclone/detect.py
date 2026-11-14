# Copyright 2026 Nathan. Apache-2.0.
"""
Finding a storm's centre in a pressure field.

The centre is a local minimum of sea-level pressure that is closed: pressure rises on every side. Three things make
the difference between a tracker that works and one that jumps to the wrong low:

* candidates are every closed local minimum near the first guess, and the nearest one is taken, not the deepest: a
  mid-latitude trough or a second storm nearby is often deeper;
* "closed" is measured: the mean pressure on a ring 300 to 500 km out, minus the centre. A real cyclone stands at
  least a few hPa below its surroundings; a ripple in a monsoon trough does not;
* the centre is refined below the grid spacing with a parabola through the neighbours, so a quarter-degree grid
  gives positions good to a few kilometres rather than 28.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter, uniform_filter

from .geodesy import distance_km


def window(lat: np.ndarray, lon: np.ndarray, clat: float, clon: float, radius_km: float) -> tuple[slice, slice]:
    """Index slices of the part of a regular grid within ``radius_km`` of a point (a box, slightly generous)."""
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(np.cos(np.radians(clat)), 0.2))
    i0, i1 = np.searchsorted(lat, [clat - dlat, clat + dlat])
    j0, j1 = np.searchsorted(lon, [clon - dlon, clon + dlon])
    return slice(max(i0 - 1, 0), min(i1 + 1, lat.size)), slice(max(j0 - 1, 0), min(j1 + 1, lon.size))


def refine(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, i: int, j: int) -> tuple[float, float, float]:
    """Sub-grid position and value of the minimum at ``(i, j)``: a parabola through each axis's three points."""
    def vertex(a, b, c):                                   # offset in grid steps of the parabola's minimum, and its drop
        den = a - 2 * b + c
        if not np.isfinite(den) or den <= 1e-9:
            return 0.0, 0.0
        off = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
        return off, 0.25 * (a - c) * off
    di = dj = drop_i = drop_j = 0.0
    if 0 < i < lat.size - 1:
        di, drop_i = vertex(field[i - 1, j], field[i, j], field[i + 1, j])
    if 0 < j < lon.size - 1:
        dj, drop_j = vertex(field[i, j - 1], field[i, j], field[i, j + 1])
    step_i = lat[1] - lat[0]
    step_j = lon[1] - lon[0]
    return float(lat[i] + di * step_i), float(lon[j] + dj * step_j), float(field[i, j] - drop_i - drop_j)


def depth(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, clat: float, clon: float, centre: float,
          ring_km: tuple = (300.0, 500.0)) -> float:
    """How far the centre stands below its surroundings: mean pressure on a ring minus the centre pressure (hPa)."""
    si, sj = window(lat, lon, clat, clon, ring_km[1])
    sub = field[si, sj]
    d = distance_km(clat, clon, lat[si][:, None], lon[sj][None, :])
    ring = (d >= ring_km[0]) & (d <= ring_km[1]) & np.isfinite(sub)
    if ring.sum() < 8:
        return float("nan")
    return float(sub[ring].mean() - centre)


def candidates(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, clat: float, clon: float, radius_km: float,
               min_depth: float = 1.5, smooth: int = 3) -> list[dict]:
    """Closed local minima within ``radius_km`` of a point, nearest first: ``{lat, lon, pressure, depth, distance}``."""
    si, sj = window(lat, lon, clat, clon, radius_km)
    sub = field[si, sj]
    if sub.size == 0 or not np.isfinite(sub).any():
        return []
    filled = np.where(np.isfinite(sub), sub, np.nanmax(sub))
    smoothed = uniform_filter(filled, size=smooth, mode="nearest") if smooth > 1 else filled
    is_min = smoothed == minimum_filter(smoothed, size=5, mode="nearest")
    out = []
    for i, j in zip(*np.nonzero(is_min)):
        gi, gj = si.start + i, sj.start + j
        if gi in (0, lat.size - 1) or gj in (0, lon.size - 1):
            continue
        # the raw field's minimum next to the smoothed one (smoothing shifts the centre by up to a grid step)
        a, b = max(gi - 1, 0), max(gj - 1, 0)
        patch = field[a:gi + 2, b:gj + 2]
        if not np.isfinite(patch).any():
            continue
        pi, pj = np.unravel_index(np.nanargmin(patch), patch.shape)
        gi, gj = a + pi, b + pj
        plat, plon, p = refine(field, lat, lon, gi, gj)
        dist = float(distance_km(clat, clon, plat, plon))
        if dist > radius_km:
            continue
        dep = depth(field, lat, lon, plat, plon, p)
        if not np.isfinite(dep) or dep < min_depth:
            continue
        out.append({"lat": plat, "lon": plon, "pressure": p, "depth": dep, "distance": dist})
    out.sort(key=lambda c: c["distance"])
    return out


def find_centre(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, clat: float, clon: float, radius_km: float = 400.0,
                min_depth: float = 1.5) -> dict | None:
    """The storm centre nearest a first guess, or ``None`` when no closed low is within reach."""
    found = candidates(field, lat, lon, clat, clon, radius_km, min_depth)
    return found[0] if found else None


__all__ = ["candidates", "depth", "find_centre", "refine", "window"]
