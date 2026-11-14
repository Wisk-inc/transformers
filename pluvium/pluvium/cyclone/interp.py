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

from .geodesy import offset_km, shift


def interpolate(points: dict, observed_now: tuple, age_hours: int = 12, taper_hours: float | None = None) -> dict:
    """The early version of a forecast: ``{lead from now: (lat, lon)}``.

    ``points`` is the older run's track ``{lead: (lat, lon)}``; ``observed_now`` is the storm's position ``age_hours``
    after that run started. The offset between the two is applied to every later position. With ``taper_hours`` the
    offset fades linearly to zero by that lead (some centres do this for long leads); by default it is kept in full,
    as in the NHC interpolator for track."""
    if age_hours not in points or not np.isfinite(points[age_hours][0]):
        return {}
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


__all__ = ["interpolate"]
