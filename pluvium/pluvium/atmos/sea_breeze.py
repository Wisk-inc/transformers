# Copyright 2026 Nathan. Apache-2.0.
"""
The sea-breeze front: where island afternoon storms are born.

On a sunny day the land heats, and by late morning cooler sea air pushes inland as a shallow gravity current. Its
leading edge -- the sea-breeze front -- lifts the moist air ahead of it; on islands the fronts from opposite coasts
meet inland in the afternoon, and that is where and when the storms go up. Gravity-current theory (Simpson 1994)
gives the front's speed, ``c = k sqrt(g' h)`` with ``g' = g dtheta / theta``, ``h`` about 500 m and ``k`` about 0.6;
the background wind adds its onshore component (an offshore wind stronger than ``c`` holds the front at the coast).

For every point and frame this module gives the hours since the front passed it (negative: still to come), from
the point's distance to the coast, the direction to the sea, the local wind, the time of day and a land-sea
temperature contrast that grows with the sunshine since sunrise. The feature the network sees is a pulse that peaks
as the front passes: the convergence line's moment.
"""

from __future__ import annotations

import numpy as np

from ..features import local_solar_hour

G, H_M, K = 9.81, 500.0, 0.62


def sea_breeze_arrival(coast_km: np.ndarray, sea_x: np.ndarray, sea_y: np.ndarray, u10: np.ndarray, v10: np.ndarray,
                       lon: np.ndarray, hours: np.ndarray, land: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ``(pulse (F, N), hours since passage (F, N))``. ``coast_km`` (N,) distance inland (positive over land),
    ``sea_x``/``sea_y`` (N,) unit vector towards the nearest sea, ``u10``/``v10`` (F, N), ``hours`` (F,) frame
    hours (UTC since 1970), ``land`` (N,) bool.
    """
    lst = local_solar_hour(lon[None, :], np.asarray(hours, np.float64)[:, None])        # (F, N)
    since_sunrise = np.clip(lst - 6.0, 0, 12)
    contrast = 4.0 * np.clip(np.sin(np.pi * np.clip(lst - 6.0, 0, 12) / 12.0), 0, 1)     # K, peaks mid-afternoon
    c = K * np.sqrt(G * np.maximum(contrast, 0.1) / 300.0 * H_M)                          # m/s
    onshore = -(u10 * sea_x[None] + v10 * sea_y[None])                                   # wind from the sea
    speed = np.maximum(c + onshore, 0.0)
    onset = 3.0                                                                           # hours after sunrise
    travelled_km = np.where(since_sunrise > onset, speed * 3.6 * (since_sunrise - onset), 0.0)
    dist = np.maximum(np.asarray(coast_km, np.float64), 0.0)[None]
    with np.errstate(divide="ignore", invalid="ignore"):
        passed_h = np.where(speed > 0.3, (travelled_km - dist) / np.maximum(speed * 3.6, 1e-6), -12.0)
    held = c + onshore < 0.3                                                             # pinned at the coast
    passed_h = np.where(held | ~land[None] | (lst < 6) | (lst > 19), -12.0, np.clip(passed_h, -12, 12))
    pulse = np.exp(-0.5 * (passed_h / 1.5) ** 2) * (contrast > 1.0)
    return pulse.astype(np.float32), passed_h.astype(np.float32)


__all__ = ["sea_breeze_arrival"]
