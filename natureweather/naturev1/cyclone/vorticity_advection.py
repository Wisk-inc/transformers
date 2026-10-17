# Copyright 2026 Nathan. Apache-2.0.
"""
Where the vorticity budget says the storm is going: motion as the propagation of the vortex's own vorticity.

The idea (Holland 1983; Chan 1984; Wu and Wang 2000): a tropical cyclone moves toward the region of maximum
increase of its relative vorticity. That turns motion into a budget that can be evaluated on the analysis at
the initial time, with no best-track history at all:

1. the advective tendency of absolute vorticity, ``T = -V . grad(zeta + f)``, from the layer-mean *total* wind
   (storm included -- its asymmetries are the point: the flow across the centre, the beta gyres the analysis
   already holds);
2. if the vortex translates at ``C`` without changing shape, ``d zeta / d t = -C . grad(zeta)``;
3. so ``C`` is the least-squares fit of ``-C . grad(zeta)`` to ``T`` over the core (area-weighted, 2x2 normal
   equations), a vector the storm's own vorticity structure says it is moving with.

The fit is only trusted when the vortex is resolved -- when the core's vorticity gradient carries enough
signal (the normal matrix is well conditioned) -- otherwise None. The motion estimate is independent of the
best track, which is exactly what makes it useful in a consensus with extrapolation: when a centre has jumped
(a reformation) or the best-track history is short, the budget does not inherit the error.
"""

from __future__ import annotations

import numpy as np

from .fields import Environment

#: Layer of the wind and vorticity (top, bottom) hPa, core radius km and the speed clip m/s.
LAYER = (300.0, 850.0)
CORE_KM = 450.0
MAX_SPEED_MS = 20.0


def budget_motion(env: Environment, lat0: float, lon0: float, layer=LAYER, core_km: float = CORE_KM) -> dict | None:
    """``{"u", "v", "condition", "explained"}`` m/s of the vorticity-budget motion, or None when unresolved."""
    u = env.layer_mean("u", *layer)
    v = env.layer_mean("v", *layer)
    zeta = env.vorticity(u, v)
    q = zeta + env.coriolis()
    tendency = -(u * env.ddx(q) + v * env.ddy(q))
    gx, gy = env.ddx(zeta), env.ddy(zeta)
    r = env.distance_km(lat0, lon0)
    w = np.where(r < core_km, np.exp(-0.5 * (r / (0.6 * core_km)) ** 2), 0.0) * env.cell_area_m2()
    if w.sum() <= 0:
        return None
    a = np.array([[np.sum(w * gx * gx), np.sum(w * gx * gy)], [np.sum(w * gx * gy), np.sum(w * gy * gy)]])
    b = -np.array([np.sum(w * gx * tendency), np.sum(w * gy * tendency)])
    eig = np.linalg.eigvalsh(a)
    if eig[0] <= 0 or eig[1] / max(eig[0], 1e-40) > 50.0:
        return None
    c = np.linalg.solve(a, b)
    speed = float(np.hypot(*c))
    if not np.isfinite(speed):
        return None
    if speed > MAX_SPEED_MS:
        c = c * MAX_SPEED_MS / speed
    residual = tendency + c[0] * gx + c[1] * gy
    explained = 1.0 - np.sum(w * residual ** 2) / max(np.sum(w * tendency ** 2), 1e-40)
    return {"u": float(c[0]), "v": float(c[1]), "condition": float(eig[1] / eig[0]), "explained": float(explained)}


__all__ = ["CORE_KM", "LAYER", "budget_motion"]
