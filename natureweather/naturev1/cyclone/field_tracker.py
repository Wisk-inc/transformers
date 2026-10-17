# Copyright 2026 Nathan. Apache-2.0.
"""
Following a storm through a gridded forecast: a GFDL-style tracker for any model's output.

The idea: comparing NatureV1's physics with GraphCast, Pangu, HRES, Aurora or FGN on hurricanes needs their
tracks, and none of them outputs a track -- they output fields. The tracker used by operational centres
(Marchok's GFDL vortex tracker) finds the storm in each forecast step from several fields and averages the
fixes; this one does the same with what every model in WeatherBench 2 provides:

* **sea-level pressure minimum** (every model has it), refined below the grid spacing by fitting a quadratic
  surface to the 3x3 neighbourhood of the lowest grid value -- at 1.5 degrees the refinement matters (a grid
  point can be 80 km from the true minimum);
* **850 hPa relative vorticity maximum** (cyclonic sign per hemisphere) and **850/700 hPa geopotential
  minima** when the model provides those fields;
* each step searches only within a radius of the position the previous motion predicts (first guess
  = previous position + previous displacement), so a nearby low is not grabbed;
* the fix is the mean of the parameter fixes that agree (within 250 km of their median);
* the track stops when the storm is lost: no closed low (the minimum is not at least ``depth_hpa`` below the
  mean pressure on a ring 300-500 km out) or the fixes disagree.

Works on any ``(time, lat, lon)`` field arrays on a regular grid (latitude ascending, longitude 0..360).
"""

from __future__ import annotations

import numpy as np

from .geometry import great_circle_km, move, local_displacement_km


def _refine(field: np.ndarray, i: int, j: int) -> tuple[float, float]:
    """Sub-grid offset (di, dj) of an extremum from a quadratic fit to its 3x3 neighbourhood (wrapped in j)."""
    h, w = field.shape
    if i <= 0 or i >= h - 1:
        return 0.0, 0.0
    c = field[i, j]
    n, s = field[i + 1, j], field[i - 1, j]
    e, wv = field[i, (j + 1) % w], field[i, (j - 1) % w]
    di = 0.5 * (s - n) / (s - 2 * c + n) if (s - 2 * c + n) != 0 else 0.0
    dj = 0.5 * (wv - e) / (wv - 2 * c + e) if (wv - 2 * c + e) != 0 else 0.0
    return float(np.clip(di, -0.5, 0.5)), float(np.clip(dj, -0.5, 0.5))


def find_extremum(field: np.ndarray, lat_axis: np.ndarray, lon_axis: np.ndarray, guess: tuple[float, float],
                  radius_km: float, minimum: bool = True) -> tuple[float, float, float] | None:
    """``(lat, lon, value)`` of the field's extremum within ``radius_km`` of ``guess``, sub-grid refined."""
    lat_g, lon_g = np.meshgrid(lat_axis, lon_axis, indexing="ij")
    d = great_circle_km(guess[0], guess[1], lat_g, lon_g)
    inside = d < radius_km
    if not inside.any():
        return None
    data = field if minimum else -field
    masked = np.where(inside, data, np.inf)
    i, j = np.unravel_index(int(np.argmin(masked)), masked.shape)
    di, dj = _refine(data, i, j)
    step_lat = lat_axis[1] - lat_axis[0]
    step_lon = lon_axis[1] - lon_axis[0]
    lat = float(lat_axis[i] + di * step_lat)
    lon = float(((lon_axis[j] + dj * step_lon + 180.0) % 360.0) - 180.0)
    return lat, lon, float(field[i, j])


def closed_low_depth(msl: np.ndarray, lat_axis, lon_axis, lat: float, lon: float, centre_value: float,
                     ring=(300.0, 500.0)) -> float:
    """Ring-mean pressure minus the centre value (Pa) -- how closed the low is."""
    lat_g, lon_g = np.meshgrid(lat_axis, lon_axis, indexing="ij")
    d = great_circle_km(lat, lon, lat_g, lon_g)
    band = (d >= ring[0]) & (d <= ring[1])
    return float(msl[band].mean() - centre_value) if band.any() else 0.0


def track_in_fields(start: tuple[float, float], hours, msl: np.ndarray, lat_axis, lon_axis,
                    vorticity850: np.ndarray | None = None, z850: np.ndarray | None = None,
                    z700: np.ndarray | None = None, first_motion_kmh: tuple[float, float] | None = None,
                    search_km: float = 350.0, depth_pa: float = 100.0, agree_km: float = 250.0) -> np.ndarray:
    """
    Track a storm from ``start`` through forecast fields at ``hours`` (``msl`` is ``(T, H, W)`` Pa).

    Returns ``(T, 2)`` lat/lon, NaN after the storm is lost. ``first_motion_kmh`` (east, north) seeds the first
    guess; later guesses extrapolate the tracked motion.
    """
    hours = np.asarray(hours, np.float64)
    out = np.full((len(hours), 2), np.nan)
    position = (float(start[0]), float(start[1]))
    motion = first_motion_kmh or (0.0, 0.0)
    last_hours = 0.0
    hemisphere = 1.0 if start[0] >= 0 else -1.0
    for k, h in enumerate(hours):
        if not np.isfinite(msl[k]).any():
            continue                                         # a lead this model does not provide (12-hourly output)
        dt = h - last_hours
        guess = move(position[0], position[1], motion[0] * dt, motion[1] * dt) if dt > 0 else position
        guess = (float(guess[0]), float(guess[1]))
        fixes = []
        found = find_extremum(msl[k], lat_axis, lon_axis, guess, search_km, minimum=True)
        if found is None:
            break
        depth = closed_low_depth(msl[k], lat_axis, lon_axis, found[0], found[1], found[2])
        if depth < depth_pa:
            break
        fixes.append(found[:2])
        if vorticity850 is not None:
            f = find_extremum(hemisphere * vorticity850[k], lat_axis, lon_axis, guess, search_km, minimum=False)
            if f is not None:
                fixes.append(f[:2])
        for z in (z850, z700):
            if z is not None:
                f = find_extremum(z[k], lat_axis, lon_axis, guess, search_km, minimum=True)
                if f is not None:
                    fixes.append(f[:2])
        fixes = np.asarray(fixes)
        median = np.median(fixes, axis=0)
        near = great_circle_km(median[0], median[1], fixes[:, 0], fixes[:, 1]) < agree_km
        use = fixes[near] if near.any() else fixes[:1]
        # Average positions in the local frame of the pressure fix (safe across the dateline).
        east, north = local_displacement_km(fixes[0, 0], fixes[0, 1], use[:, 0], use[:, 1])
        lat, lon = move(fixes[0, 0], fixes[0, 1], east.mean(), north.mean())
        new = (float(lat), float(lon))
        if dt > 0:
            e, n = local_displacement_km(position[0], position[1], new[0], new[1])
            motion = (float(e) / dt, float(n) / dt)
        position, last_hours = new, h
        out[k] = new
    return out


__all__ = ["closed_low_depth", "find_extremum", "track_in_fields"]
