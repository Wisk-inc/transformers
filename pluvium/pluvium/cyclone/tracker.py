# Copyright 2026 Nathan. Apache-2.0.
"""
Following one storm through one forecast.

The tracker starts from the position and motion known in real time and steps through the forecast. At each step it
predicts where the storm should be (last position plus last motion), looks for the closed low nearest that guess, and
updates the motion. A step with no closed low is bridged by dead reckoning once; a second miss in a row ends the track,
because the model has dissipated the storm or merged it into a front.

The same tracker runs on every member of every system, on whatever grid the system publishes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detect import find_centre
from .geodesy import offset_km, shift


@dataclass
class TrackerConfig:
    #: search radius at the first step, km (a model's initial vortex can sit well away from the analysed position)
    first_radius_km: float = 450.0
    #: search radius at later steps, km, plus ``radius_per_hour`` for every hour since the last fix
    radius_km: float = 250.0
    radius_per_hour: float = 12.0
    #: the least depth of a closed low, hPa (ring mean minus centre)
    min_depth: float = 1.5
    #: steps that may be bridged without a fix before the track ends
    bridge: int = 1
    #: how much of the newly measured motion replaces the old (1 = no memory)
    motion_gain: float = 0.7
    #: storms do not move faster than this, km/h (a fix implying more is rejected)
    max_speed_kmh: float = 110.0


def track_member(msl: np.ndarray, lat: np.ndarray, lon: np.ndarray, steps, start: dict,
                 cfg: TrackerConfig | None = None) -> np.ndarray:
    """One member's track: ``(steps, 3)`` of latitude, longitude and central pressure (hPa); NaN where there is no fix.

    ``msl`` is ``(steps, H, W)`` in hPa on a regular grid (``lat`` and ``lon`` ascending); ``steps`` are hours from the
    start; ``start`` has ``lat``, ``lon`` and the motion ``east_kmh``, ``north_kmh`` (see
    :func:`pluvium.cyclone.storms.first_guess`)."""
    cfg = cfg or TrackerConfig()
    out = np.full((len(steps), 3), np.nan)
    plat, plon = float(start["lat"]), float(start["lon"])
    ve, vn = float(start.get("east_kmh", 0.0)), float(start.get("north_kmh", 0.0))
    last_time, misses, fixes = 0.0, 0, 0
    for k, step in enumerate(steps):
        dt = float(step) - last_time
        glat, glon = shift(plat, plon, ve * dt, vn * dt)
        glat, glon = float(glat), float(glon)
        if not (lat[0] <= glat <= lat[-1] and lon[0] <= glon <= lon[-1]):
            break                                                    # the storm has left the basin's box
        radius = cfg.first_radius_km if fixes == 0 else cfg.radius_km + cfg.radius_per_hour * dt
        found = find_centre(msl[k], lat, lon, glat, glon, radius, cfg.min_depth)
        if found is not None:
            east, north = offset_km(plat, plon, found["lat"], found["lon"])
            if fixes > 0 and np.hypot(east, north) / max(dt, 1e-6) > cfg.max_speed_kmh:
                found = None                                         # a jump to another low
        if found is None:
            misses += 1
            if misses > cfg.bridge:
                break
            plat, plon, last_time = glat, glon, float(step)          # dead reckoning; no fix recorded
            continue
        misses = 0
        east, north = offset_km(plat, plon, found["lat"], found["lon"])
        gain = cfg.motion_gain if fixes > 0 else 0.5                 # the first fix also absorbs the initial offset
        ve = (1 - gain) * ve + gain * float(east) / dt
        vn = (1 - gain) * vn + gain * float(north) / dt
        plat, plon, last_time = found["lat"], found["lon"], float(step)
        out[k] = (plat, plon, found["pressure"])
        fixes += 1
    return out


def track_members(msl: np.ndarray, lat: np.ndarray, lon: np.ndarray, steps, start: dict,
                  cfg: TrackerConfig | None = None) -> np.ndarray:
    """Every member's track: ``(members, steps, 3)`` from ``msl`` of shape ``(members, steps, H, W)``. Members with
    missing fields from some step on are tracked up to that step."""
    out = np.full((msl.shape[0], len(steps), 3), np.nan)
    for m in range(msl.shape[0]):
        have = np.isfinite(msl[m]).reshape(len(steps), -1).any(1)
        last = int(np.argmin(have)) if not have.all() else len(steps)
        if last == 0:
            continue
        out[m, :last] = track_member(msl[m, :last], lat, lon, list(steps)[:last], start, cfg)
    return out


__all__ = ["TrackerConfig", "track_member", "track_members"]
