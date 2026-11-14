# Copyright 2026 Nathan. Apache-2.0.
"""
Wind meeting terrain, at every town and every hour.

The orographic layer (:mod:`pluvium.physics`) solves the linear theory on 2 km tiles, but only where tiles exist and
only for the tile-mean wind. Two simpler, local quantities complete it:

* **forced ascent** ``w = V . grad(h)`` -- the vertical velocity the low-level wind is forced into by the slope, at
  two scales (terrain smoothed to about 5 km and 20 km): the small scale lights up individual ridges, the large one
  whole mountain ranges;
* **the Froude number** ``Fr = U / (N h)`` -- whether the air goes over the mountain (``Fr > 1``: rain on the
  windward slope and spill-over) or is blocked and flows around it (``Fr < 1``: rain shifts upwind of the
  barrier, and the lee is very dry). ``N`` is a moist buoyancy frequency of 0.006 per second and ``h`` the local
  relief.
"""

from __future__ import annotations

import numpy as np

N_MOIST = 0.006
_CACHE: dict = {}


def _gradients(terrain, km: float) -> tuple[np.ndarray, np.ndarray]:
    """d(h)/dx and d(h)/dy (m per m) of the terrain smoothed at ``km``, cached per terrain."""
    from scipy import ndimage

    key = (id(terrain), km)
    if key not in _CACHE:
        lat = terrain.lat
        dy_km = abs(lat[1] - lat[0]) * 111.195
        dx_km = abs(terrain.lon[1] - terrain.lon[0]) * 111.195 * np.cos(np.radians(lat.mean()))
        h = ndimage.gaussian_filter(np.clip(terrain.elevation, 0, None).astype(np.float64),
                                    (km / dy_km, km / dx_km))
        gy, gx = np.gradient(h, dy_km * 1000.0 * np.sign(lat[1] - lat[0]), dx_km * 1000.0)
        _CACHE[key] = (gx.astype(np.float32), gy.astype(np.float32))
    return _CACHE[key]


def upslope_features(terrain, lat: np.ndarray, lon: np.ndarray, u: np.ndarray, v: np.ndarray,
                     relief_m: np.ndarray) -> np.ndarray:
    """``(F, N, 3)``: forced ascent at 5 km and 20 km (cm/s, signed, scaled) and the Froude number (scaled)."""
    out = []
    for km in (5.0, 20.0):
        gx, gy = _gradients(terrain, km)
        sx = terrain.sample(gx, lat, lon)
        sy = terrain.sample(gy, lat, lon)
        out.append(np.clip((u * sx[None] + v * sy[None]) * 100.0 / 5.0, -5, 5))          # cm/s / 5
    speed = np.hypot(u, v)
    froude = speed / (N_MOIST * np.maximum(np.asarray(relief_m, np.float64), 50.0))[None]
    out.append(np.clip(froude, 0, 6) / 3.0)
    return np.stack(out, -1).astype(np.float32)


__all__ = ["upslope_features"]
