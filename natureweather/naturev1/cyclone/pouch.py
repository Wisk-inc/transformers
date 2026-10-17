# Copyright 2026 Nathan. Apache-2.0.
"""
The marsupial pouch: a closed circulation in the frame that moves with the wave.

The idea (Dunkerton, Montgomery and Wang 2009): most Atlantic and many Pacific storms form inside tropical
waves, and the wave matters because of what it looks like *in its own moving frame*. In the Earth-relative
frame a wave's circulation is usually open -- air blows straight through it. In the frame moving west with the
wave, a critical layer forms where the flow speed equals the wave's phase speed, and the streamlines close
into a "pouch" that protects the moist, rotating core from dry air, letting convection organise long enough
for a vortex to spin up. A disturbance with a closed pouch is a disturbance that can become a storm; one
without, usually cannot.

:func:`pouch_closedness` evaluates the circulation of the wave-relative 850/700 hPa flow on rings around a
disturbance: the share of the ring where the relative tangential wind is cyclonic, the mean relative tangential
wind, and the relative circulation -- 1 for a perfectly closed, strongly cyclonic pouch, 0 for none. The phase
speed comes from the disturbance's own motion (its candidate track) or a climatological -7 m/s westward.
"""

from __future__ import annotations

import numpy as np

from .geometry import destination, hemisphere

DEFAULT_PHASE_MS = (-7.0, 0.0)


def pouch_closedness(env, lat0: float, lon0: float, phase_ms=DEFAULT_PHASE_MS, levels=("850", "600"),
                     radii_km=(250.0, 400.0, 550.0), azimuths: int = 24) -> dict:
    """Closedness of the wave-relative circulation around (lat0, lon0) in an analysis band."""
    sign = float(hemisphere(lat0))
    shares, speeds = [], []
    for level in levels:
        u = env.fields[f"u{level}"] - phase_ms[0]
        v = env.fields[f"v{level}"] - phase_ms[1]
        for radius in radii_km:
            bearing = np.arange(azimuths) * 360.0 / azimuths
            plat, plon = destination(lat0, lon0, bearing, radius)
            uu, vv = env.sample(u, plat, plon), env.sample(v, plat, plon)
            b = np.radians(bearing)
            # Tangential unit vector for counter-clockwise rotation at a point bearing b from the centre.
            te, tn = -np.cos(b), np.sin(b)
            vt = sign * (uu * te + vv * tn)
            shares.append(float(np.mean(vt > 0)))
            speeds.append(float(np.mean(vt)))
    share = float(np.mean(shares))
    speed = float(np.mean(speeds))
    score = share * float(np.clip(speed / 5.0, 0.0, 1.0))
    return {"share_cyclonic": share, "mean_tangential_ms": speed, "closedness": score}


def phase_from_history(candidate) -> tuple[float, float]:
    """Phase speed (m/s) of a genesis candidate from its last 12-24 hours of positions, or the default."""
    from .geometry import velocity_from_positions

    if not candidate.history:
        return DEFAULT_PHASE_MS
    earlier = candidate.history[-2] if len(candidate.history) >= 2 else candidate.history[-1]
    hours = (candidate.time - earlier.time) / np.timedelta64(1, "h")
    if hours <= 0:
        return DEFAULT_PHASE_MS
    u, v = velocity_from_positions(earlier.lat, earlier.lon, candidate.lat, candidate.lon, hours)
    return float(np.clip(u, -15, 5)), float(np.clip(v, -5, 5))


__all__ = ["DEFAULT_PHASE_MS", "phase_from_history", "pouch_closedness"]
