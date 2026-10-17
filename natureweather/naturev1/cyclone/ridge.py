# Copyright 2026 Nathan. Apache-2.0.
"""
The subtropical ridge and the troughs that break it: where, how strong, and how it is changing.

The idea: whether a storm keeps going west, recurves or stalls is decided by the shape of the subtropical high
north of it (south of it in the southern hemisphere) and by the mid-latitude troughs that erode it. A
forecaster reads that off the 500 hPa chart; this module reads it off the analysis, as numbers:

* **ridge axis latitude** in the storm's longitude band -- where the environmental 500 hPa zonal wind turns
  from easterly to westerly going poleward (a storm reaching it recurves);
* **ridge strength** -- the highest 500 hPa height on the poleward side, above the storm's own, and its
  **12-hour tendency** from the previous analysis (Dorian's Bermuda High weakened and slid east: a negative
  tendency on the ridge in the storm's band is that signal);
* **ridge weakness** -- how much lower the ridge is in the storm's band than 15 degrees either side: a break
  the storm can find;
* **upstream trough** -- the strongest cyclonic 500 hPa vorticity poleward of the ridge and up to 30 degrees
  west of the storm: its amplitude, distance and how far equatorward it reaches (Dorian's trough "failed to dip
  far enough" -- that is this number).

Everything is in the poleward frame (southern-hemisphere storms mirrored), so the numbers mean the same in
both hemispheres. :mod:`naturev1.cyclone.regimes` turns them into a regime.
"""

from __future__ import annotations

import numpy as np

from .fields import Environment
from .geometry import great_circle_km, hemisphere


def _band(env: Environment, lon0: float, half_width: float) -> np.ndarray:
    offset = ((env.lon - lon0 + 180.0) % 360.0) - 180.0
    return np.abs(offset) <= half_width


def ridge_diagnostics(env: Environment, lat0: float, lon0: float, band_deg: float = 7.5) -> dict:
    """Ridge and trough numbers for a storm at (lat0, lon0); see the module docstring."""
    sign = float(hemisphere(lat0))
    poleward = sign * (env.lat - lat0)                                      # >0 poleward of the storm
    z = env.level("z", 500) / 9.80665                                       # m
    u = env.level("u", 500)
    band = _band(env, lon0, band_deg)
    rows = np.argsort(poleward)
    zonal = u[:, band].mean(1)
    height = z[:, band].mean(1)
    storm_row = int(np.argmin(np.abs(env.lat - lat0)))
    # Ridge axis: first easterly-to-westerly change going poleward from the storm (in the poleward frame,
    # westerly is u > 0 in both hemispheres).
    axis = np.nan
    ordered = [r for r in rows if poleward[r] >= -2.0]
    for a, b in zip(ordered[:-1], ordered[1:]):
        if zonal[a] <= 0.0 < zonal[b]:
            frac = -zonal[a] / (zonal[b] - zonal[a])
            axis = float(env.lat[a] + frac * (env.lat[b] - env.lat[a]))
            break
    pole_rows = [r for r in rows if 0.0 <= poleward[r] <= 25.0]
    peak_row = pole_rows[int(np.argmax(height[pole_rows]))] if pole_rows else storm_row
    strength = float(height[peak_row] - height[storm_row])
    tendency = np.nan
    if "z500" in env.previous:
        previous = env.previous["z500"] / 9.80665
        tendency = float(previous[:, band].mean(1)[peak_row] * -1.0 + height[peak_row])
    # Weakness: ridge-line height here versus 15 degrees either side.
    sides = []
    for shift in (-15.0, 15.0):
        side = _band(env, lon0 + shift, band_deg)
        if side.any():
            sides.append(float(z[pole_rows][:, side].mean(1).max()) if pole_rows else np.nan)
    weakness = float(np.nanmean(sides) - height[peak_row]) if sides else np.nan
    # Upstream trough: strongest cyclonic vorticity poleward of the ridge axis, 0-30 degrees upstream (west).
    zeta = sign * env.vorticity(env.level("u", 500), env.level("v", 500))
    offset = ((env.lon - lon0 + 180.0) % 360.0) - 180.0
    ridge_lat = axis if np.isfinite(axis) else lat0 + sign * 10.0
    upstream = (offset >= -30.0) & (offset <= 5.0)
    region = (sign * (env.lat - ridge_lat) >= -3.0)[:, None] & upstream[None, :]
    trough = {"amplitude": np.nan, "distance_km": np.nan, "equatorward_lat": np.nan}
    if region.any():
        candidate = np.where(region, zeta, -np.inf)
        i, j = np.unravel_index(int(np.argmax(candidate)), candidate.shape)
        trough["amplitude"] = float(candidate[i, j])
        trough["distance_km"] = float(great_circle_km(lat0, lon0, env.lat[i], env.lon[j]))
        strong = region & (zeta > 0.5 * candidate[i, j])
        if strong.any():
            reach = np.abs(env.lat[np.nonzero(strong)[0]] - lat0).min()
            trough["equatorward_lat"] = float(reach)
    return {"ridge_axis_lat": axis, "ridge_gap_deg": float(sign * (axis - lat0)) if np.isfinite(axis) else np.nan,
            "ridge_strength_m": strength, "ridge_tendency_m": tendency, "ridge_weakness_m": weakness,
            "trough_amplitude": trough["amplitude"], "trough_distance_km": trough["distance_km"],
            "trough_reach_deg": trough["equatorward_lat"]}


__all__ = ["ridge_diagnostics"]
