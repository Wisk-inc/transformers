# Copyright 2026 Nathan. Apache-2.0.
"""
Beta drift that depends on the storm and on the air around it, not a fixed 1 m/s north-west.

The idea: a vortex on a sphere drifts because its own winds carry planetary vorticity around it, spinning up
a pair of "beta gyres" whose flow pushes the vortex poleward and westward (Holland 1983; Chan and Williams
1987; Fiorino and Elsberry 1989). Two facts from that theory are used here, where a fixed vector ignores both:

* **Speed scales with the vortex's outer strength** -- the tangential wind 300-800 km out, not the eyewall --
  and with the planetary vorticity gradient. A large, strong-outer-wind typhoon drifts at 3-4 m/s; a small,
  compact hurricane at well under 1. The outer strength is measured from the vortex that
  :func:`naturev1.cyclone.vortex.remove_vortex` took out of the analysis; the gradient is beta at the storm's
  latitude.
* **Direction follows the gradient of absolute vorticity, not just of f** -- the drift is up the gradient of
  ``sign(f) * (f + zeta_environment)``, turned towards the west by the gyre angle. Where the environment's own
  vorticity gradient is strong (the flank of a monsoon trough, the edge of a ridge), the drift turns with it.

The scale constants are physical defaults checked on the 2016/2017/2019 seasons only.
"""

from __future__ import annotations

import numpy as np

from .fields import Environment
from .geometry import beta as beta_at, hemisphere, rotate
from .steering import disc_smooth

#: Drift speed for a storm at 20 degrees with an outer tangential wind of 6 m/s, m/s.
DRIFT_MS = 1.6
#: Angle from the up-gradient direction towards the west, degrees.
GYRE_ANGLE_DEG = 40.0
#: Outer-strength radii, km, and the clip on the measured strength, m/s.
OUTER_RADII_KM = (300.0, 400.0, 500.0, 600.0, 700.0, 800.0)
OUTER_CLIP_MS = (1.0, 15.0)


def outer_strength(split, level_hpa: int = 850) -> float:
    """Mean cyclonic tangential wind of the removed vortex between 300 and 800 km at ``level_hpa``, m/s."""
    levels = list(split.environment.levels)
    k = levels.index(level_hpa) if level_hpa in levels else int(np.argmin([abs(p - level_hpa) for p in levels]))
    profile = split.tangential_profile(k, OUTER_RADII_KM)
    value = float(np.nanmean(profile)) if np.isfinite(profile).any() else 4.0
    return float(np.clip(value, *OUTER_CLIP_MS))


class BetaDrift:
    """
    Drift velocity anywhere in the box for a vortex of a given outer strength.

    ``eta_gradient`` is the smoothed gradient of the environment's deep-layer absolute vorticity, so the
    drift direction responds to the environment as well as to the planet.
    """

    def __init__(self, env: Environment, outer_ms: float, top: float = 300.0, bottom: float = 850.0,
                 smoothing_km: float = 800.0, drift_ms: float | None = None, angle_deg: float | None = None) -> None:
        self.env, self.outer_ms = env, outer_ms
        self.drift_ms = DRIFT_MS if drift_ms is None else drift_ms
        self.angle_deg = GYRE_ANGLE_DEG if angle_deg is None else angle_deg
        u = env.layer_mean("u", top, bottom)
        v = env.layer_mean("v", top, bottom)
        zeta = env.vorticity(u, v)
        self.smoothing_km = smoothing_km
        self.gx = disc_smooth(env, env.ddx(zeta), smoothing_km)
        self.gy = disc_smooth(env, env.ddy(zeta), smoothing_km)

    def velocity(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        lat = np.atleast_1d(np.asarray(lat, np.float64))
        lon = np.atleast_1d(np.asarray(lon, np.float64))
        b = beta_at(lat)
        gx = self.env.sample(self.gx, lat, lon)
        gy = self.env.sample(self.gy, lat, lon) + b
        sign = hemisphere(lat)
        gx, gy = sign * gx, sign * gy
        size = np.hypot(gx, gy)
        speed = self.drift_ms * np.sqrt(np.clip(size / beta_at(20.0), 0.05, 4.0)) * np.sqrt(self.outer_ms / 6.0)
        ex, ey = gx / np.maximum(size, 1e-15), gy / np.maximum(size, 1e-15)
        dx, dy = rotate(ex, ey, sign * self.angle_deg)
        return speed * dx, speed * dy


__all__ = ["BetaDrift", "DRIFT_MS", "GYRE_ANGLE_DEG", "outer_strength"]
