# Copyright 2026 Nathan. Apache-2.0.
"""
Vertical wind shear in fine layers, and the tilt it puts in the vortex.

The idea: "850-200 hPa shear" is one number for a profile that can hide everything -- a storm under 25 kt of
850-200 shear concentrated above 300 hPa is barely touched; the same 25 kt between 850 and 500 hPa tears its
low-level centre from its convection (Dorian's early life: a small, sheared, lopsided storm with competing
centres). So the shear here is a profile:

* the environmental wind (vortex removed) averaged over the 200-800 km annulus SHIPS uses, at every level;
* shear vectors for seven layer pairs, from 925-700 to 300-200, so where in the column the shear sits is known;
* **generalised shear** (Knaff et al. 2005): the pressure-weighted RMS departure of the wind profile from its
  deep-layer mean -- a single number that sees curvature a two-level difference cannot;
* the shear direction relative to the storm's motion (shear from behind the storm hurts less than shear
  against it -- the storm-relative shear);
* **vortex tilt** from the analysis itself: the 850 and 500 hPa vortex centres (vorticity centroids within
  300 km), their displacement, and its direction relative to the shear (a tilt downshear-left is the
  precession a resilient vortex shows; downshear-right is one coming apart).

:func:`tilt_drift` turns the tilt into a small motion term: a tilted vortex's low-level centre is pulled toward
realignment under the mid-level circulation (Wu and Emanuel 1993's "tilt-induced motion"), here as the tilt
vector closing over ``tau_h`` hours.
"""

from __future__ import annotations

import numpy as np

from .fields import Environment
from .geometry import hemisphere

#: Layer pairs (bottom, top) hPa.
PAIRS = ((850, 200), (850, 500), (700, 400), (500, 200), (925, 700), (300, 200), (850, 300))
ANNULUS_KM = (200.0, 800.0)


def annulus_profile(env: Environment, lat0: float, lon0: float, annulus=ANNULUS_KM) -> dict:
    """``{level: (u, v)}`` area-mean environmental wind in the annulus around the storm, m/s."""
    r = env.distance_km(lat0, lon0)
    ring = (r >= annulus[0]) & (r <= annulus[1])
    area = env.cell_area_m2() * ring
    total = area.sum()
    out = {}
    for k, p in enumerate(env.levels):
        out[int(p)] = (float((env.fields["u"][k] * area).sum() / total), float((env.fields["v"][k] * area).sum() / total))
    return out


def shear_diagnostics(env_clean: Environment, lat0: float, lon0: float, motion_ms=None) -> dict:
    """All shear numbers (m/s and degrees) for the storm; ``env_clean`` is the vortex-removed analysis."""
    profile = annulus_profile(env_clean, lat0, lon0)
    out = {}
    for bottom, top in PAIRS:
        du = profile[top][0] - profile[bottom][0]
        dv = profile[top][1] - profile[bottom][1]
        out[f"shear_{bottom}_{top}"] = float(np.hypot(du, dv))
        out[f"shear_{bottom}_{top}_dir"] = float(np.degrees(np.arctan2(du, dv)) % 360.0)   # direction it blows toward
    levels = [p for p in sorted(profile, reverse=True) if 200 <= p <= 850]
    weights = np.gradient(np.asarray(levels, np.float64)) * -1.0
    weights = np.abs(weights) / np.abs(weights).sum()
    u = np.array([profile[p][0] for p in levels])
    v = np.array([profile[p][1] for p in levels])
    ub, vb = (weights * u).sum(), (weights * v).sum()
    out["generalised_shear"] = float(np.sqrt((weights * ((u - ub) ** 2 + (v - vb) ** 2)).sum()))
    if motion_ms is not None and np.hypot(*motion_ms) > 0.5:
        du = profile[200][0] - profile[850][0]
        dv = profile[200][1] - profile[850][1]
        relative = np.degrees(np.arctan2(du, dv) - np.arctan2(motion_ms[0], motion_ms[1]))
        out["shear_vs_motion_deg"] = float((relative + 180.0) % 360.0 - 180.0)
        out["storm_relative_shear"] = float(np.hypot(du - 0.0, dv - 0.0))
    return out


def _centroid(env: Environment, field: np.ndarray, lat0: float, lon0: float, radius_km: float, sign: float):
    x, y = env.local_coordinates(lat0, lon0)
    r = np.hypot(x, y)
    w = np.where(r < radius_km, np.maximum(sign * field, 0.0), 0.0)
    if w.sum() <= 0:
        return None
    return float((w * x).sum() / w.sum()), float((w * y).sum() / w.sum())


def vortex_tilt(env: Environment, lat0: float, lon0: float, radius_km: float = 300.0) -> dict | None:
    """850 and 500 hPa vortex centroids (km from the best-track centre) and the tilt between them."""
    sign = float(hemisphere(lat0))
    centres = {}
    for p in (850, 500):
        zeta = env.vorticity(env.level("u", p), env.level("v", p))
        c = _centroid(env, zeta, lat0, lon0, radius_km, sign)
        if c is None:
            return None
        centres[p] = c
    tx = centres[500][0] - centres[850][0]
    ty = centres[500][1] - centres[850][1]
    return {"low_xy_km": centres[850], "mid_xy_km": centres[500], "tilt_x_km": tx, "tilt_y_km": ty,
            "tilt_km": float(np.hypot(tx, ty)), "tilt_dir": float(np.degrees(np.arctan2(tx, ty)) % 360.0)}


def tilt_drift(tilt: dict | None, tau_h: float = 36.0, fraction: float = 0.5):
    """A velocity callable closing ``fraction`` of the tilt over ``tau_h`` hours, decaying as it closes."""
    from .trajectory import DecayingResidual

    if tilt is None:
        return None
    seconds = tau_h * 3600.0
    return DecayingResidual(fraction * tilt["tilt_x_km"] * 1000.0 / seconds, fraction * tilt["tilt_y_km"] * 1000.0 / seconds, tau_h)


__all__ = ["ANNULUS_KM", "PAIRS", "annulus_profile", "shear_diagnostics", "tilt_drift", "vortex_tilt"]
