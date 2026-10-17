# Copyright 2026 Nathan. Apache-2.0.
"""
Take the storm out of the analysis before asking what steers it.

The idea (Galarneau and Davis 2013's local inversion, used here as a filter): the wind in an analysis near
a storm is mostly the storm's own circulation. Averaged symmetrically around the exact centre it cancels,
but nothing in a real calculation is symmetric -- the centre falls between grid points, the averaging disc
is clipped by the grid, a slow storm's future positions sit inside its own initial vortex, and a second
storm nearby is someone else's vortex. So the vortex is removed explicitly:

1. Inside a radius ``R0`` of the centre, the vortex's vorticity and divergence are what exceeds the
   environment's (the mean over a ring just outside ``R0``).
2. Their winds are recovered everywhere by inverting them on the storm's tangent plane -- Biot-Savart for
   the rotational part, its divergent counterpart for the inflow and outflow -- with the self-interaction
   of each source cell regularised by half a grid length (which leaves ~15-20% of the vortex's wind on its
   nearest neighbours: measured, and the setting every reported benchmark number was produced with).
3. Subtracting those winds from the analysis at every level leaves the environment: the flow the storm would
   feel if it were a point.

The returned :class:`VortexSplit` keeps both halves, because the vortex half is information too: its outer
tangential wind (300-800 km) sets the beta drift (:mod:`naturev1.cyclone.beta`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fields import Environment

#: Radius inside which vorticity and divergence count as the storm's, km.
VORTEX_RADIUS_KM = 600.0
#: Width of the ring that defines the environment's own vorticity, km.
RING_KM = 300.0


@dataclass
class VortexSplit:
    environment: Environment          # the analysis with the vortex's winds removed (u, v replaced)
    vortex_u: np.ndarray              # (L, H, W) the removed winds
    vortex_v: np.ndarray
    centre: tuple[float, float]

    def tangential_profile(self, level_index: int, radii_km=(100, 200, 300, 400, 500, 600, 700, 800)) -> np.ndarray:
        """Azimuthal-mean tangential wind (m/s, cyclonic positive) of the removed vortex at those radii."""
        env = self.environment
        x, y = env.local_coordinates(*self.centre)
        r = np.hypot(x, y)
        sign = 1.0 if self.centre[0] >= 0 else -1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            vt = sign * (x * self.vortex_v[level_index] - y * self.vortex_u[level_index]) / r
        out = []
        width = 0.5 * env.spacing_deg * 111.0 + 50.0
        for radius in radii_km:
            ring = np.abs(r - radius) < width
            out.append(float(np.nanmean(vt[ring])) if ring.any() else np.nan)
        return np.asarray(out)


def _invert(source_x, source_y, zeta, delta, area, target_x, target_y, soften_km):
    """Winds (m/s) at targets induced by vorticity/divergence (s^-1) at sources, tangent-plane Green's functions."""
    dx = (target_x[:, None] - source_x[None, :]) * 1000.0
    dy = (target_y[:, None] - source_y[None, :]) * 1000.0
    r2 = dx * dx + dy * dy + (soften_km * 1000.0) ** 2
    weight = area[None, :] / (2.0 * np.pi * r2)
    u = (-(dy * zeta[None, :]) + dx * delta[None, :]) * weight
    v = ((dx * zeta[None, :]) + dy * delta[None, :]) * weight
    return u.sum(1), v.sum(1)


def remove_vortex(env: Environment, lat0: float, lon0: float, radius_km: float = VORTEX_RADIUS_KM,
                  ring_km: float = RING_KM, include_divergent: bool = True) -> VortexSplit:
    """
    Split the analysis winds at every level into vortex and environment around (lat0, lon0).

    Returns a :class:`VortexSplit` whose ``environment`` is a copy of ``env`` with ``u`` and ``v`` replaced by
    the environmental winds (other fields untouched).
    """
    x, y = env.local_coordinates(lat0, lon0)
    r = np.hypot(x, y)
    inside = r < radius_km
    ring = (r >= radius_km) & (r < radius_km + ring_km)
    area = env.cell_area_m2()
    sx, sy, sa = x[inside], y[inside], area[inside]
    tx, ty = x.ravel(), y.ravel()
    soften = 0.5 * env.spacing_deg * 111.0
    u_all, v_all = env.fields["u"], env.fields["v"]
    vu = np.zeros_like(u_all)
    vv = np.zeros_like(v_all)
    for k in range(u_all.shape[0]):
        zeta = env.vorticity(u_all[k], v_all[k])
        delta = env.divergence(u_all[k], v_all[k]) if include_divergent else np.zeros_like(zeta)
        z_src = zeta[inside] - (zeta[ring].mean() if ring.any() else 0.0)
        d_src = delta[inside] - (delta[ring].mean() if ring.any() else 0.0)
        iu, iv = _invert(sx, sy, z_src, d_src, sa, tx, ty, soften)
        vu[k] = iu.reshape(u_all.shape[1:])
        vv[k] = iv.reshape(u_all.shape[1:])
    fields = dict(env.fields)
    fields["u"] = u_all - vu
    fields["v"] = v_all - vv
    cleaned = Environment(lat=env.lat, lon=env.lon, levels=env.levels, fields=fields, previous=env.previous,
                          time=env.time)
    cleaned.previous_levels = env.previous_levels
    return VortexSplit(cleaned, vu, vv, (float(lat0), float(lon0)))


__all__ = ["RING_KM", "VORTEX_RADIUS_KM", "VortexSplit", "remove_vortex"]
