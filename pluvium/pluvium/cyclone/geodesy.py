# Copyright 2026 Nathan. Apache-2.0.
"""
Distances, bearings and displacements on the sphere, for storm tracks.

Track errors are great-circle distances. A consensus is a mean of positions, taken as a mean of unit vectors so that it
is right at any longitude. Errors are split along and across the observed heading, because models are slow or fast
(along) far more persistently than they are left or right (cross), and the two are corrected separately.
"""

from __future__ import annotations

import numpy as np

EARTH_KM = 6371.0088
KM_PER_NM = 1.852


def distance_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance (haversine), km. Arguments broadcast."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(y, x)) % 360.0


def destination(lat, lon, bearing, km) -> tuple[np.ndarray, np.ndarray]:
    """The point ``km`` away from ``(lat, lon)`` along ``bearing`` (degrees clockwise from north)."""
    p, b, d = np.radians(lat), np.radians(bearing), np.asarray(km) / EARTH_KM
    p2 = np.arcsin(np.clip(np.sin(p) * np.cos(d) + np.cos(p) * np.sin(d) * np.cos(b), -1.0, 1.0))
    l2 = np.radians(lon) + np.arctan2(np.sin(b) * np.sin(d) * np.cos(p), np.cos(d) - np.sin(p) * np.sin(p2))
    return np.degrees(p2), (np.degrees(l2) + 180.0) % 360.0 - 180.0


def offset_km(lat0, lon0, lat1, lon1) -> tuple[np.ndarray, np.ndarray]:
    """Displacement from point 0 to point 1 as ``(east km, north km)`` (distance and bearing, resolved)."""
    d = distance_km(lat0, lon0, lat1, lon1)
    b = np.radians(bearing_deg(lat0, lon0, lat1, lon1))
    return d * np.sin(b), d * np.cos(b)


def shift(lat, lon, east_km, north_km) -> tuple[np.ndarray, np.ndarray]:
    """Move ``(lat, lon)`` by a displacement given as east and north km (the inverse of :func:`offset_km`)."""
    d = np.hypot(east_km, north_km)
    b = np.degrees(np.arctan2(east_km, north_km))
    return destination(lat, lon, b, d)


def mean_position(lat, lon, weights=None, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean position as the mean of unit vectors; NaN positions are ignored (all NaN gives NaN)."""
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    ok = np.isfinite(lat) & np.isfinite(lon)
    w = np.ones_like(lat) if weights is None else np.broadcast_to(np.asarray(weights, np.float64), lat.shape)
    w = np.where(ok, w, 0.0)
    p, l_ = np.radians(np.where(ok, lat, 0.0)), np.radians(np.where(ok, lon, 0.0))
    x = (w * np.cos(p) * np.cos(l_)).sum(axis)
    y = (w * np.cos(p) * np.sin(l_)).sum(axis)
    z = (w * np.sin(p)).sum(axis)
    total = w.sum(axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        mlat = np.degrees(np.arctan2(z, np.hypot(x, y)))
        mlon = np.degrees(np.arctan2(y, x))
    return np.where(total > 0, mlat, np.nan), np.where(total > 0, mlon, np.nan)


def along_cross(lat_f, lon_f, lat_o, lon_o, heading_deg) -> tuple[np.ndarray, np.ndarray]:
    """The forecast's error against the observed position, split by the observed heading: ``(along, cross)`` km,
    positive when the forecast is ahead of the storm and to the right of its track."""
    east, north = offset_km(lat_o, lon_o, lat_f, lon_f)
    h = np.radians(heading_deg)
    along = east * np.sin(h) + north * np.cos(h)
    cross = east * np.cos(h) - north * np.sin(h)
    return along, cross


def from_along_cross(lat_o, lon_o, along, cross, heading_deg) -> tuple[np.ndarray, np.ndarray]:
    """The position ``along`` km ahead and ``cross`` km to the right of ``(lat_o, lon_o)`` for a heading."""
    h = np.radians(heading_deg)
    east = along * np.sin(h) + cross * np.cos(h)
    north = along * np.cos(h) - cross * np.sin(h)
    return shift(lat_o, lon_o, east, north)


__all__ = ["EARTH_KM", "KM_PER_NM", "along_cross", "bearing_deg", "destination", "distance_km", "from_along_cross",
           "mean_position", "offset_km", "shift"]
