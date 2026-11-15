# Copyright 2026 Nathan. Apache-2.0.
"""
The interpolator: yesterday evening's run as this morning's guidance.

Global models finish hours after their start time. When a forecaster writes the 12 UTC advisory, the newest complete
ECMWF run started at 00 UTC. The standard remedy, used for every "early" aid in the National Hurricane Center's decks
(the codes ending in ``I``), is to take the older run's track, read it 12 hours further along, and slide the whole
track so that its position now matches where the storm is observed to be.

That makes a late forecast comparable with the official forecast, which is issued with only early guidance in hand.
"""

from __future__ import annotations

import numpy as np

from .geodesy import bearing_deg, destination, distance_km, offset_km, shift


def extend(points: dict, hours: int = 12) -> dict:
    """The track carried ``hours`` past its last lead at its own final motion (speed and heading of the last leg).

    A run that ends at 120 hours gives early guidance only to 108. The last 12 hours are filled this way, so the
    early forecast reaches the official forecast's longest lead. Nothing is added when the last leg is missing."""
    leads = sorted(k for k, p in points.items() if np.isfinite(p[0]))
    if len(leads) < 2:
        return dict(points)
    a, b = points[leads[-2]], points[leads[-1]]
    span = leads[-1] - leads[-2]
    if span <= 0 or span > 2 * hours:
        return dict(points)
    km = float(distance_km(a[0], a[1], b[0], b[1])) * hours / span
    lat, lon = destination(b[0], b[1], float(bearing_deg(a[0], a[1], b[0], b[1])), km) if km > 0 else (b[0], b[1])
    out = dict(points)
    out[leads[-1] + hours] = (float(lat), float(lon))
    return out


def interpolate(points: dict, observed_now: tuple, age_hours: int = 12, taper_hours: float | None = None,
                extend_hours: int = 0) -> dict:
    """The early version of a forecast: ``{lead from now: (lat, lon)}``.

    ``points`` is the older run's track ``{lead: (lat, lon)}``; ``observed_now`` is the storm's position ``age_hours``
    after that run started. The offset between the two is applied to every later position. With ``taper_hours`` the
    offset fades linearly to zero by that lead (some centres do this for long leads); by default it is kept in full,
    as in the NHC interpolator for track. With ``extend_hours`` the older run is first carried that much further at
    its final motion (:func:`extend`), which restores the leads lost to the run's age."""
    if age_hours not in points or not np.isfinite(points[age_hours][0]):
        return {}
    if extend_hours:
        points = extend(points, extend_hours)
    east, north = offset_km(points[age_hours][0], points[age_hours][1], observed_now[0], observed_now[1])
    out = {0: (float(observed_now[0]), float(observed_now[1]))}
    for lead, (lat, lon) in points.items():
        new = lead - age_hours
        if new <= 0 or not np.isfinite(lat):
            continue
        keep = 1.0 if taper_hours is None else max(0.0, 1.0 - new / taper_hours)
        nlat, nlon = shift(lat, lon, float(east) * keep, float(north) * keep)
        out[int(new)] = (float(nlat), float(nlon))
    return out


__all__ = ["extend", "interpolate"]
