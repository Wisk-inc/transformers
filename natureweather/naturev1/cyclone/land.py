# Copyright 2026 Nathan. Apache-2.0.
"""
Land and mountains along a forecast track: when the storm meets them, and what they do to it.

The idea: a storm's intensity forecast is decided as much by *when it reaches land* as by anything over the
ocean, and the track forecast says when. So land is read along the forecast track, hour by hour, from the
analysis's land-sea mask and surface height:

* **landfall time** -- the first hour the centre's land fraction passes one half;
* **inland decay** -- Kaplan and DeMaria (1995): over land the wind decays exponentially towards a background
  ``V_b`` with rate ``alpha``, starting from the landfall intensity times a reduction factor for the coastal
  roughness jump;
* **terrain** -- mountains disrupt the circulation far faster than flat land (Hispaniola, Taiwan, Luzon): the
  decay rate grows with the surface height under the centre (``alpha * (1 + h / TERRAIN_SCALE_M)``), the effect
  that tore up storms on Hispaniola's 3,000 m peaks and that Dorian avoided by re-forming east of it.

:func:`land_along_track` samples the fields; :func:`apply_land_decay` turns an over-water intensity forecast
into one that knows where the coast and the mountains are.
"""

from __future__ import annotations

import numpy as np

from .fields import Environment
from .geometry import interpolate_track

BACKGROUND_KT = 26.7
DECAY_PER_H = 0.095
COAST_FACTOR = 0.9
TERRAIN_SCALE_M = 1000.0


def land_along_track(env: Environment, lat0: float, lon0: float, track: np.ndarray, leads, step_h: float = 1.0) -> dict:
    """Hourly land fraction and surface height (m) along the forecast track (great-circle interpolated)."""
    hours = np.concatenate([[0.0], np.asarray(leads, np.float64)])
    lats = np.concatenate([[lat0], track[:, 0]])
    lons = np.concatenate([[lon0], track[:, 1]])
    good = np.isfinite(lats)
    hourly = np.arange(0.0, hours[good][-1] + 1e-9, step_h)
    la, lo = interpolate_track(hours[good], lats[good], lons[good], hourly)
    land = np.clip(env.sample(env.fields["lsm"], la, lo), 0.0, 1.0)
    height = np.maximum(env.sample(env.fields["zs"], la, lo) / 9.80665, 0.0)
    landfall = hourly[np.argmax(land > 0.5)] if (land > 0.5).any() else np.nan
    return {"hours": hourly, "lat": la, "lon": lo, "land": land, "height_m": height, "landfall_hour": float(landfall)}


def land_weight(land: np.ndarray) -> np.ndarray:
    """How much of the inland decay applies: none below 40% land, all of it above 90%. A 1.5-degree cell over the
    Bahamas is partly land; a storm over it is still over water almost everywhere that matters."""
    return np.clip((np.asarray(land, np.float64) - 0.4) / 0.5, 0.0, 1.0)


def apply_land_decay(hours: np.ndarray, water_kt: np.ndarray, land: np.ndarray, height_m: np.ndarray) -> np.ndarray:
    """
    Intensity (kt) along ``hours`` given an over-water forecast ``water_kt``: over land the wind decays
    Kaplan-DeMaria style from where it came ashore -- at a rate scaled by how much of the cell is land, faster over
    high terrain; back over water it resumes from the decayed value and follows the over-water forecast's *changes*
    (it does not jump back to it).
    """
    water = np.asarray(water_kt, np.float64)
    out = water.copy()
    weight = land_weight(land)
    current = water[0]
    ashore = False
    for k in range(1, len(hours)):
        dt = hours[k] - hours[k - 1]
        change = water[k] - water[k - 1]
        if weight[k] > 0.0:
            if not ashore:
                current = BACKGROUND_KT + (1.0 - (1.0 - COAST_FACTOR) * weight[k]) * (current - BACKGROUND_KT)
                ashore = True
            rate = DECAY_PER_H * weight[k] * (1.0 + height_m[k] / TERRAIN_SCALE_M)
            current = BACKGROUND_KT + (current - BACKGROUND_KT) * np.exp(-rate * dt)
            current = min(current + (1.0 - weight[k]) * max(change, 0.0), water[k])
        else:
            ashore = False
            current = min(current + change, water[k])
        out[k] = max(current, 15.0)
    return out


__all__ = ["BACKGROUND_KT", "DECAY_PER_H", "apply_land_decay", "land_along_track", "land_weight"]
