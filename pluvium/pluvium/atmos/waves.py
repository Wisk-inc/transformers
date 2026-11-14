# Copyright 2026 Nathan. Apache-2.0.
"""
Tropical (easterly) waves: the main source of Caribbean rain from June to November.

An easterly wave is a trough in the 700 hPa flow moving west at 6-9 m/s. Ahead of (west of) the trough the wind backs
to north-easterly and the air sinks; at and behind the axis it veers to south-easterly and showers and thunderstorms
break out. The rain arrives with the axis, so its *timing* is the wave's *position and speed*.

Detection: in easterlies, the axis is where the meridional wind changes from northerly to southerly going east
(``dv/dx > 0`` with ``v`` near zero) and the relative vorticity is cyclonic. Each frame's axes are found, smoothed,
and the hours until an axis comes within 150 km of every point are computed from the frames themselves (the driving
model's own propagation, the same way front arrival times are made).
"""

from __future__ import annotations

import numpy as np

from ..features import _grad, arrival_hours, distance_km


def trough_strength(u700: np.ndarray, v700: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """``(F, H, W)`` trough-axis strength (``dv/dx`` per 100 km, in m/s) where the flow is easterly and v small."""
    vx, _ = _grad(v700, lat, lon)
    _, uy = _grad(u700, lat, lon)
    vort = vx - uy
    axis = (vx > 0.4) & (np.abs(v700) < 3.0) & (u700 < -2.0) & (vort > 0)
    return np.where(axis, np.clip(vx, 0, 5), 0.0).astype(np.float32)


def wave_channels(upper: dict, lat: np.ndarray, lon: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """``(F, 2, H, W)``: trough strength, and hours until a trough axis is within 150 km (divided by 24; 3 = none)."""
    strength = trough_strength(upper["u700"], upper["v700"], lat, lon)
    near = distance_km(strength > 0.5, lat, lon) < 150.0
    eta = arrival_hours(near, np.asarray(hours, np.float64))
    return np.stack([strength / 2.0, np.clip(eta / 24.0, 0, 3)], 1).astype(np.float32)


__all__ = ["trough_strength", "wave_channels"]
