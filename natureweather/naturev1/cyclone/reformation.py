# Copyright 2026 Nathan. Apache-2.0.
"""
Competing centres and the jump to a new one.

The idea, from Dorian's birth in the Caribbean: forecasters tracked three competing circulation centres; the
storm then re-formed around a convective cluster to the east, skipped Hispaniola's mountains and exploded.
A steering model treats the storm as a fixed point carried by the wind and cannot represent a centre that
"teleports" 30-50 km. But a reformation is not random -- it happens where the evidence already is:

* **candidate centres**: every local sea-level-pressure minimum and every local cyclonic 850 hPa vorticity
  maximum within 400 km of the best-track centre (more than one well-separated candidate = competing centres);
* **where the convection and the mid-level vortex are**: the 500 hPa vorticity centroid and the column-water
  maximum (deep moisture is where the convection lives) within 300 km;
* under shear, the low-level centre re-forms *downshear*, under the mid-level circulation and the convection
  (the well-documented pattern in sheared, weak storms), and weak storms re-form far more often than strong
  ones.

:func:`reformation_outlook` returns the candidates, the expected jump vector (probability times the offset to
the convective/mid-level centre) and the probability, from shear, intensity and the offset's alignment with
the shear. :func:`reformation_drift` turns the expected jump into a motion term spent over the first ``tau_h``
hours -- so the forecast leans toward where the centre is likely to re-form rather than ignoring it.
"""

from __future__ import annotations

import numpy as np

from .fields import Environment
from .geometry import hemisphere

SEARCH_KM = 400.0


def _local_extrema(field: np.ndarray, mask: np.ndarray, maximum: bool) -> list[tuple[int, int]]:
    out = []
    data = field if maximum else -field
    h, w = data.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if not mask[i, j]:
                continue
            centre = data[i, j]
            block = data[i - 1:i + 2, j - 1:j + 2]
            if centre >= block.max() and centre > np.median(block):
                out.append((i, j))
    return out


def candidate_centres(env: Environment, lat0: float, lon0: float, radius_km: float = SEARCH_KM) -> list[dict]:
    """Local pressure minima and cyclonic 850 hPa vorticity maxima near the storm (km offsets and strength)."""
    sign = float(hemisphere(lat0))
    x, y = env.local_coordinates(lat0, lon0)
    mask = np.hypot(x, y) < radius_km
    zeta = sign * env.vorticity(env.level("u", 850), env.level("v", 850))
    out = []
    for i, j in _local_extrema(env.fields["msl"], mask, maximum=False):
        out.append({"kind": "pressure", "x_km": float(x[i, j]), "y_km": float(y[i, j]), "value": float(env.fields["msl"][i, j])})
    for i, j in _local_extrema(zeta, mask, maximum=True):
        if zeta[i, j] > 0:
            out.append({"kind": "vorticity", "x_km": float(x[i, j]), "y_km": float(y[i, j]), "value": float(zeta[i, j])})
    return out


def reformation_outlook(env: Environment, env_clean: Environment, lat0: float, lon0: float, wind_kt: float) -> dict:
    """Candidates, the convective/mid-level target, the jump probability and the expected jump (km)."""
    from .shear import shear_diagnostics, vortex_tilt

    sign = float(hemisphere(lat0))
    x, y = env.local_coordinates(lat0, lon0)
    r = np.hypot(x, y)
    near = r < 300.0
    candidates = candidate_centres(env, lat0, lon0)
    tilt = vortex_tilt(env, lat0, lon0)
    water = np.where(near, env.fields["tcwv"], -np.inf)
    i, j = np.unravel_index(int(np.argmax(water)), water.shape)
    moist = (float(x[i, j]), float(y[i, j]))
    mid = tilt["mid_xy_km"] if tilt else moist
    target = (0.5 * (mid[0] + moist[0]), 0.5 * (mid[1] + moist[1]))
    shear = shear_diagnostics(env_clean, lat0, lon0)
    magnitude = shear["shear_850_500"] + 0.5 * shear["shear_850_200"]
    direction = np.radians(shear["shear_850_200_dir"])
    unit = np.array([np.sin(direction), np.cos(direction)])
    offset = np.asarray(target)
    distance = float(np.hypot(*offset))
    alignment = float(offset @ unit / max(distance, 1e-6)) if distance > 0 else 0.0
    competing = len({(c["kind"], round(c["x_km"] / 150), round(c["y_km"] / 150)) for c in candidates if c["kind"] == "pressure"})
    weak = 1.0 / (1.0 + np.exp(((wind_kt if np.isfinite(wind_kt) else 40.0) - 45.0) / 8.0))
    sheared = 1.0 / (1.0 + np.exp(-(magnitude - 10.0) / 3.0))
    logit = -2.0 + 2.0 * weak + 1.5 * sheared + 0.8 * max(alignment, 0.0) + 0.5 * min(competing - 1, 2) + distance / 300.0
    probability = float(1.0 / (1.0 + np.exp(-logit)))
    if distance < 40.0:
        probability *= distance / 40.0
    return {"candidates": candidates, "competing_centres": competing, "target_xy_km": target,
            "jump_probability": probability, "expected_jump_km": (probability * target[0], probability * target[1]),
            "shear_alignment": alignment, "sign": sign}


def reformation_drift(outlook: dict, tau_h: float = 12.0, share: float = 1.0):
    """A decaying velocity that moves the storm by ``share`` of the expected jump over the first ``tau_h`` hours."""
    from .trajectory import DecayingResidual

    jx, jy = outlook["expected_jump_km"]
    seconds = tau_h * 3600.0
    return DecayingResidual(share * jx * 1000.0 / seconds, share * jy * 1000.0 / seconds, tau_h)


__all__ = ["candidate_centres", "reformation_drift", "reformation_outlook"]
