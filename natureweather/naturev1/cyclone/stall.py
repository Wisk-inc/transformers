# Copyright 2026 Nathan. Apache-2.0.
"""
Steering collapse: finding the stagnation point a storm can park in, before it parks there.

The idea, from Dorian over the Bahamas in 2019: the Bermuda High weakened and drifted east while the trough to
the north stayed too far north, and between them the deep-layer flow fell to nothing. Models saw the
slowdown and missed the *stall* -- a storm that keeps its current motion (extrapolation, or a learned
correction trained on storms that keep moving) cannot stop. Physically, the storm was heading into a **col**:
a saddle point of the steering flow, where the flows from two opposing pressure systems cancel. Saddles are
topology, and topology can be computed:

* :func:`critical_points` finds every zero of the (vortex-removed, disc-averaged) steering field in a region
  -- cells where both components change sign -- refines each with Newton's method on the bilinear field, and
  classifies it by the flow's Jacobian: ``det < 0`` is a saddle (col), ``det > 0`` a centre or node (the
  middle of a high or a low).
* :func:`stall_diagnostics` measures what matters for a forecast: the weakest steering the storm meets along
  its forecast path and when, the nearest col to that path, and a stall probability built from them. Weak
  flow *at a col* is the trap; weak flow in the middle of a broad high is not -- there the storm still drifts
  with beta.
* :func:`timing_spread` quantifies the Dorian timing problem: during a stall, a 1 mph (0.45 m/s) speed error
  becomes 48 miles in 48 hours. It returns how far apart along-track trajectories with the steering speeded
  up and slowed down by a given amount end up.

:mod:`naturev1.cyclone.regimes` uses these to change what the consensus trusts: in the stall regime the
current motion is not persisted (it decays within hours) and extrapolation's weight falls.
"""

from __future__ import annotations

import numpy as np

from .fields import Environment
from .geometry import great_circle_km

#: Steering speed (m/s) below which flow counts as collapsed, and the col distance scale (km).
COLLAPSE_MS = 2.5
COL_SCALE_KM = 400.0


def critical_points(env: Environment, u: np.ndarray, v: np.ndarray, lat0: float, lon0: float,
                    radius_km: float = 2000.0) -> list[dict]:
    """
    Zeros of a gridded flow ``(u, v)`` within ``radius_km`` of (lat0, lon0).

    Returns dicts ``{"lat", "lon", "kind": "saddle"|"centre", "det", "distance_km"}``; the Jacobian is in
    s^-2 (from the spherical derivatives).
    """
    dudx, dudy = env.ddx(u), env.ddy(u)
    dvdx, dvdy = env.ddx(v), env.ddy(v)
    out = []
    h, w = u.shape
    for i in range(h - 1):
        for j in range(w - 1):
            cu = u[i:i + 2, j:j + 2]
            cv = v[i:i + 2, j:j + 2]
            if cu.min() > 0 or cu.max() < 0 or cv.min() > 0 or cv.max() < 0:
                continue
            # Newton on the bilinear interpolant inside the cell.
            a, b = 0.5, 0.5
            for _ in range(20):
                fu = (1 - a) * (1 - b) * cu[0, 0] + (1 - a) * b * cu[0, 1] + a * (1 - b) * cu[1, 0] + a * b * cu[1, 1]
                fv = (1 - a) * (1 - b) * cv[0, 0] + (1 - a) * b * cv[0, 1] + a * (1 - b) * cv[1, 0] + a * b * cv[1, 1]
                ju_a = -(1 - b) * cu[0, 0] - b * cu[0, 1] + (1 - b) * cu[1, 0] + b * cu[1, 1]
                ju_b = -(1 - a) * cu[0, 0] + (1 - a) * cu[0, 1] - a * cu[1, 0] + a * cu[1, 1]
                jv_a = -(1 - b) * cv[0, 0] - b * cv[0, 1] + (1 - b) * cv[1, 0] + b * cv[1, 1]
                jv_b = -(1 - a) * cv[0, 0] + (1 - a) * cv[0, 1] - a * cv[1, 0] + a * cv[1, 1]
                det = ju_a * jv_b - ju_b * jv_a
                if abs(det) < 1e-12:
                    break
                da = (fu * jv_b - fv * ju_b) / det
                db = (ju_a * fv - jv_a * fu) / det
                a, b = a - da, b - db
                if abs(da) + abs(db) < 1e-6:
                    break
            if not (-0.05 <= a <= 1.05 and -0.05 <= b <= 1.05):
                continue
            lat = env.lat[i] + a * (env.lat[1] - env.lat[0])
            lon = env.lon[j] + b * (env.lon[1] - env.lon[0])
            distance = float(great_circle_km(lat0, lon0, lat, lon))
            if distance > radius_km:
                continue
            ii, jj = min(i + int(round(a)), h - 1), min(j + int(round(b)), w - 1)
            jac = dudx[ii, jj] * dvdy[ii, jj] - dudy[ii, jj] * dvdx[ii, jj]
            out.append({"lat": float(lat), "lon": float(((lon + 180.0) % 360.0) - 180.0),
                        "kind": "saddle" if jac < 0 else "centre", "det": float(jac), "distance_km": distance})
    # Merge duplicates found from neighbouring cells.
    merged = []
    for point in sorted(out, key=lambda p: p["distance_km"]):
        if all(great_circle_km(point["lat"], point["lon"], m["lat"], m["lon"]) > 150.0 for m in merged):
            merged.append(point)
    return merged


def stall_diagnostics(flow, track: np.ndarray, leads, lat0: float, lon0: float) -> dict:
    """
    How close a forecast path comes to steering collapse.

    Args:
        flow: a :class:`naturev1.cyclone.steering.SteeringFlow` (its ``u``, ``v`` are disc-averaged fields).
        track: ``(L, 2)`` the forecast positions at ``leads``.

    Returns ``{"start_speed_ms", "min_speed_ms", "min_speed_hour", "col_distance_km", "col_hour",
    "stall_probability", "cols": [...]}``.
    """
    env = flow.env
    lats = np.concatenate([[lat0], track[:, 0]])
    lons = np.concatenate([[lon0], track[:, 1]])
    hours = np.concatenate([[0.0], np.asarray(leads, np.float64)])
    good = np.isfinite(lats)
    u, v = flow.velocity(lats[good], lons[good])
    speed = np.hypot(u, v)
    k = int(np.argmin(speed))
    cols = [p for p in critical_points(env, flow.u, flow.v, lat0, lon0, 2500.0) if p["kind"] == "saddle"]
    col_distance, col_hour = np.inf, np.nan
    for p in cols:
        d = great_circle_km(lats[good], lons[good], p["lat"], p["lon"])
        j = int(np.argmin(d))
        if d[j] < col_distance:
            col_distance, col_hour = float(d[j]), float(hours[good][j])
    weak = 1.0 / (1.0 + np.exp((speed[k] - COLLAPSE_MS) / 0.75))
    near = np.exp(-col_distance / COL_SCALE_KM) if np.isfinite(col_distance) else 0.0
    probability = float(np.clip(weak * (0.35 + 0.65 * near), 0.0, 1.0))
    return {"start_speed_ms": float(speed[0]), "min_speed_ms": float(speed[k]), "min_speed_hour": float(hours[good][k]),
            "col_distance_km": col_distance, "col_hour": col_hour, "stall_probability": probability, "cols": cols}


def timing_spread(lat0: float, lon0: float, velocity, leads, delta_ms: float = 0.45) -> np.ndarray:
    """
    ``(L,)`` km between trajectories whose steering is sped up and slowed down by ``delta_ms`` along the motion
    -- the along-track uncertainty one mph of speed error buys (Dorian: 48 miles in 48 hours).
    """
    from .trajectory import integrate

    def nudged(sign):
        def f(lat, lon, hours):
            u, v = velocity(lat, lon, hours)
            speed = np.maximum(np.hypot(u, v), 1e-6)
            return u + sign * delta_ms * u / speed, v + sign * delta_ms * v / speed
        return f

    fast = integrate(lat0, lon0, nudged(+1.0), leads)
    slow = integrate(lat0, lon0, nudged(-1.0), leads)
    return great_circle_km(fast[:, 0], fast[:, 1], slow[:, 0], slow[:, 1])


__all__ = ["COLLAPSE_MS", "COL_SCALE_KM", "critical_points", "stall_diagnostics", "timing_spread"]
