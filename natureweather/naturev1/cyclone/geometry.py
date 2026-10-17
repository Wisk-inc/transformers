# Copyright 2026 Nathan. Apache-2.0.
"""
Spherical geometry for storm tracks: every move, distance and frame on the sphere, never on a lat-lon sheet.

The idea: a track model that steps a storm east by ``u * dt / (R cos(lat))`` degrees is right in the
deep tropics and quietly wrong everywhere a storm actually recurves -- the error grows with latitude and
blows up near the poles, and a longitude step taken in degrees across the dateline is a 360-degree jump.
Here every displacement is a great-circle move from a bearing and a distance, every local calculation
happens in an azimuthal-equidistant frame centred on the storm (distances from the centre are exact, and
the frame turns with the storm), and verification splits errors into along- and cross-track parts the way
the National Hurricane Center does.

All functions take and return numpy arrays (broadcasting), degrees and kilometres.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088
KM_PER_DEGREE = np.pi * EARTH_RADIUS_KM / 180.0
OMEGA = 7.2921159e-5                      # Earth's rotation, s^-1


def wrap_longitude(lon):
    """Longitude back into [-180, 180)."""
    return (np.asarray(lon, np.float64) + 180.0) % 360.0 - 180.0


def great_circle_km(lat0, lon0, lat1, lon1):
    """Haversine distance in kilometres."""
    p0, p1 = np.radians(lat0), np.radians(lat1)
    dp = p1 - p0
    dl = np.radians(np.asarray(lon1, np.float64) - np.asarray(lon0, np.float64))
    h = np.sin(dp / 2) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(dl / 2) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def bearing_deg(lat0, lon0, lat1, lon1):
    """Initial great-circle bearing from point 0 to point 1, degrees clockwise from north, in [0, 360)."""
    p0, p1 = np.radians(lat0), np.radians(lat1)
    dl = np.radians(np.asarray(lon1, np.float64) - np.asarray(lon0, np.float64))
    x = np.sin(dl) * np.cos(p1)
    y = np.cos(p0) * np.sin(p1) - np.sin(p0) * np.cos(p1) * np.cos(dl)
    return np.degrees(np.arctan2(x, y)) % 360.0


def destination(lat, lon, bearing, distance_km):
    """The point ``distance_km`` along the great circle leaving (lat, lon) on ``bearing``; (lat, lon)."""
    p, l0 = np.radians(lat), np.radians(lon)
    b = np.radians(bearing)
    d = np.asarray(distance_km, np.float64) / EARTH_RADIUS_KM
    p1 = np.arcsin(np.clip(np.sin(p) * np.cos(d) + np.cos(p) * np.sin(d) * np.cos(b), -1.0, 1.0))
    l1 = l0 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(p), np.cos(d) - np.sin(p) * np.sin(p1))
    return np.degrees(p1), wrap_longitude(np.degrees(l1))


def move(lat, lon, east_km, north_km):
    """Move by a local (east, north) displacement in km, as one great-circle step."""
    east_km, north_km = np.asarray(east_km, np.float64), np.asarray(north_km, np.float64)
    distance = np.hypot(east_km, north_km)
    bearing = np.degrees(np.arctan2(east_km, north_km))
    return destination(lat, lon, bearing, distance)


def local_displacement_km(lat0, lon0, lat1, lon1):
    """(east, north) km from point 0 to point 1 in point 0's azimuthal-equidistant frame (exact distance)."""
    distance = great_circle_km(lat0, lon0, lat1, lon1)
    b = np.radians(bearing_deg(lat0, lon0, lat1, lon1))
    return distance * np.sin(b), distance * np.cos(b)


def to_local(lat0, lon0, lat, lon):
    """Azimuthal-equidistant (x east, y north) km of points around a centre -- alias of the displacement."""
    return local_displacement_km(lat0, lon0, lat, lon)


def from_local(lat0, lon0, x_km, y_km):
    """Inverse of :func:`to_local`."""
    return move(lat0, lon0, x_km, y_km)


def velocity_from_positions(lat0, lon0, lat1, lon1, hours):
    """Mean (east, north) motion in m/s that takes a storm from point 0 to point 1 in ``hours``."""
    east, north = local_displacement_km(lat0, lon0, lat1, lon1)
    scale = 1000.0 / (np.asarray(hours, np.float64) * 3600.0)
    return east * scale, north * scale


def coriolis(lat):
    """f = 2 Omega sin(lat), s^-1."""
    return 2.0 * OMEGA * np.sin(np.radians(lat))


def beta(lat):
    """df/dy = 2 Omega cos(lat) / R, m^-1 s^-1."""
    return 2.0 * OMEGA * np.cos(np.radians(lat)) / (EARTH_RADIUS_KM * 1000.0)


def hemisphere(lat):
    """+1 north (and on the equator), -1 south."""
    return np.where(np.asarray(lat) < 0.0, -1.0, 1.0)


def rotate(east, north, degrees):
    """Rotate a vector counter-clockwise (mathematical sense) by ``degrees``."""
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    return c * east - s * north, s * east + c * north


def along_cross_track(lat_init, lon_init, lat_truth, lon_truth, lat_fcst, lon_fcst):
    """
    NHC-style decomposition of a forecast error, km: along-track (positive = forecast ahead of the storm)
    and cross-track (positive = forecast to the right of the storm's actual motion).

    The reference direction is the actual motion from the initial position to the verifying position; for
    a storm that did not move (under 10 km) the decomposition is undefined and NaN is returned.
    """
    te, tn = local_displacement_km(lat_init, lon_init, lat_truth, lon_truth)
    fe, fn = local_displacement_km(lat_init, lon_init, lat_fcst, lon_fcst)
    length = np.hypot(te, tn)
    with np.errstate(invalid="ignore", divide="ignore"):
        ue, un = te / length, tn / length
    de, dn = fe - te, fn - tn
    along = de * ue + dn * un
    cross = de * un - dn * ue                 # right of motion is positive
    bad = length < 10.0
    return np.where(bad, np.nan, along), np.where(bad, np.nan, cross)


def interpolate_track(hours, lat, lon, at_hours):
    """
    Positions at ``at_hours`` along a track given at ``hours``, interpolating along great circles in each
    segment (exact for a constant-bearing segment, unlike linear interpolation of latitude and longitude).
    """
    hours = np.asarray(hours, np.float64)
    at = np.atleast_1d(np.asarray(at_hours, np.float64))
    out_lat, out_lon = np.full(at.shape, np.nan), np.full(at.shape, np.nan)
    for k, h in enumerate(at):
        if h < hours[0] - 1e-9 or h > hours[-1] + 1e-9:
            continue
        j = int(np.clip(np.searchsorted(hours, h, side="right") - 1, 0, len(hours) - 2))
        span = hours[j + 1] - hours[j]
        frac = 0.0 if span <= 0 else (h - hours[j]) / span
        distance = great_circle_km(lat[j], lon[j], lat[j + 1], lon[j + 1])
        b = bearing_deg(lat[j], lon[j], lat[j + 1], lon[j + 1])
        out_lat[k], out_lon[k] = destination(lat[j], lon[j], b, frac * distance)
    return out_lat, out_lon


__all__ = ["EARTH_RADIUS_KM", "KM_PER_DEGREE", "OMEGA", "along_cross_track", "bearing_deg", "beta", "coriolis",
           "destination", "from_local", "great_circle_km", "hemisphere", "interpolate_track",
           "local_displacement_km", "move", "rotate", "to_local", "velocity_from_positions", "wrap_longitude"]
