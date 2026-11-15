# Copyright 2026 Nathan. Apache-2.0.
"""
How close will it come? Closest approach to a place, and the chance of the centre passing nearby.

The question people on an island ask is not where the centre will be at 72 hours but whether it will pass them. Two
answers are given here:

* :func:`closest_approach` for one track: the smallest distance to a place and when, with the track filled in between
  its 12-hourly positions;
* :func:`passage_chance` for a cloud of member tracks (:mod:`.cloud`): the weight of the members that pass within a
  radius of the place before a horizon. With a dressed cloud this is a calibrated chance.

A member that loses the storm stops there: a storm that has dissipated does not pass anything.
"""

from __future__ import annotations

import numpy as np

from .cloud import Cloud
from .geodesy import distance_km


def fill(leads: np.ndarray, lat: np.ndarray, lon: np.ndarray, start: tuple | None = None, every: int = 3) -> tuple:
    """Positions every ``every`` hours along a track (or many: ``lat``/``lon`` may be ``(M, L)``), linearly between
    its leads. ``start`` is the position at hour 0. Returns ``(hours, lat, lon)``; NaN after a track ends."""
    lat, lon = np.atleast_2d(np.asarray(lat, np.float64)), np.atleast_2d(np.asarray(lon, np.float64))
    leads = np.asarray(leads, np.float64)
    if start is not None:
        leads = np.concatenate([[0.0], leads])
        lat = np.concatenate([np.full((lat.shape[0], 1), start[0]), lat], 1)
        lon = np.concatenate([np.full((lon.shape[0], 1), start[1]), lon], 1)
    hours = np.arange(leads[0], leads[-1] + 1e-9, every)
    k = np.clip(np.searchsorted(leads, hours, side="right") - 1, 0, leads.size - 2)
    f = (hours - leads[k]) / (leads[k + 1] - leads[k])
    dlon = (lon[:, k + 1] - lon[:, k] + 180.0) % 360.0 - 180.0                   # the short way round
    out_lat = lat[:, k] + f[None] * (lat[:, k + 1] - lat[:, k])
    out_lon = lon[:, k] + f[None] * dlon
    exact = f == 0.0                                                             # at a lead, the next one may be missing
    out_lat[:, exact], out_lon[:, exact] = lat[:, k[exact]], lon[:, k[exact]]
    return hours, out_lat, out_lon


def closest_approach(points: dict, lat: float, lon: float, start: tuple | None = None, horizon: int | None = None) -> tuple:
    """``(distance km, hour)`` of a track's closest approach to a place (``(nan, nan)`` for an empty track)."""
    leads = sorted(k for k, p in points.items() if np.isfinite(p[0]) and (horizon is None or k <= horizon))
    if not leads or (len(leads) < 2 and start is None):
        return float("nan"), float("nan")
    hours, tlat, tlon = fill(np.array(leads), np.array([points[k][0] for k in leads]),
                             np.array([points[k][1] for k in leads]), start)
    d = distance_km(tlat[0], tlon[0], lat, lon)
    k = int(np.nanargmin(d))
    return float(d[k]), float(hours[k])


def member_approach(cloud: Cloud, lat: float, lon: float, start: tuple | None = None,
                    horizon: int | None = None) -> np.ndarray:
    """Every member's closest distance (km) to a place before ``horizon`` (inf for a member with no fix at all)."""
    use = cloud.leads <= (horizon if horizon is not None else cloud.leads.max())
    if cloud.members == 0 or use.sum() < (1 if start is not None else 2):
        return np.full(cloud.members, np.inf)
    _, tlat, tlon = fill(cloud.leads[use], cloud.lat[:, use], cloud.lon[:, use], start)
    d = distance_km(tlat, tlon, lat, lon)
    return np.where(np.isfinite(d).any(1), np.nanmin(np.where(np.isfinite(d), d, np.inf), 1), np.inf)


def passage_chance(cloud: Cloud, lat: float, lon: float, radius_km: float = 150.0, start: tuple | None = None,
                   horizon: int | None = None) -> float:
    """The chance that the centre passes within ``radius_km`` of a place before ``horizon`` hours."""
    if cloud.members == 0:
        return float("nan")
    return float((cloud.member_weight() * (member_approach(cloud, lat, lon, start, horizon) <= radius_km)).sum())


def observed_passage(track, init: int, lat: float, lon: float, radius_km: float = 150.0, horizon: int = 120) -> bool | None:
    """Whether the best track passed within ``radius_km`` of a place in the ``horizon`` hours after ``init``
    (``None`` when the best track does not cover the start)."""
    hours = [int(h) for h in track.hours if init <= int(h) <= init + horizon]
    if len(hours) < 2:
        return None
    pos = [track.position(h) for h in hours]
    _, tlat, tlon = fill(np.array(hours, np.float64), np.array([p[0] for p in pos]), np.array([p[1] for p in pos]))
    return bool(np.nanmin(distance_km(tlat[0], tlon[0], lat, lon)) <= radius_km)


__all__ = ["closest_approach", "fill", "member_approach", "observed_passage", "passage_chance"]
