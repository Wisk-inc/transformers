# Copyright 2026 Nathan. Apache-2.0.
"""
The baselines a track model has to beat, defined exactly -- and all of them, so "beats extrapolation" is not
won against a strawman.

* ``xtrp12``: the classic: the last 12 hours of motion continued in a straight line in latitude and
  longitude (NHC's XTRP; also what :func:`naturev1.steering.extrapolation_track` does).
* ``xtrp6``: the same with the last 6 hours -- noisier, but closer to now.
* ``xtrp_gc``: the 12-hour motion continued along the great circle (constant speed, the initial bearing
  turning as the sphere requires) -- what extrapolation should mean at high latitude.
* ``persistence``: intensity stays what it is now.

:func:`strongest_extrapolation` returns, per lead, the best of the extrapolations on a given sample, the
baseline every claim in the benchmark is made against.
"""

from __future__ import annotations

import numpy as np

from .geometry import bearing_deg, destination, great_circle_km, wrap_longitude


def _past(case, hours: int):
    from .data import HISTORY_HOURS

    k = HISTORY_HOURS.index(-hours)
    lat, lon = case.history["lat"][k], case.history["lon"][k]
    return (lat, lon) if np.isfinite(lat) and np.isfinite(lon) else None


def motion_back(case, hours: int):
    """``(dlat, dlon)`` degrees per hour over the last ``hours``, falling back to the other window."""
    for h in (hours, 12 if hours == 6 else 6):
        past = _past(case, h)
        if past is not None:
            dlon = wrap_longitude(case.lon - past[1])
            return (case.lat - past[0]) / h, dlon / h
    return None


def xtrp_latlon(case, leads, hours: int = 12) -> np.ndarray:
    """Linear extrapolation in latitude and longitude; ``(L, 2)``."""
    rate = motion_back(case, hours)
    out = np.full((len(leads), 2), np.nan)
    if rate is None:
        return out
    for k, lead in enumerate(leads):
        out[k] = case.lat + rate[0] * lead, wrap_longitude(case.lon + rate[1] * lead)
    return out


def xtrp_great_circle(case, leads, hours: int = 12) -> np.ndarray:
    past = _past(case, hours) or _past(case, 6 if hours == 12 else 12)
    out = np.full((len(leads), 2), np.nan)
    if past is None:
        return out
    span = hours if _past(case, hours) is not None else (6 if hours == 12 else 12)
    speed = great_circle_km(past[0], past[1], case.lat, case.lon) / span
    # Bearing of motion *at* the current position: the back-bearing to the past fix, reversed.
    heading = (bearing_deg(case.lat, case.lon, past[0], past[1]) + 180.0) % 360.0
    for k, lead in enumerate(leads):
        out[k] = destination(case.lat, case.lon, heading, speed * lead)
    return out


def persistence_intensity(case, leads) -> np.ndarray:
    return np.full(len(leads), case.wind_kt)


def baseline_tracks(cases, leads) -> dict[str, np.ndarray]:
    return {"xtrp12": np.stack([xtrp_latlon(c, leads, 12) for c in cases]),
            "xtrp6": np.stack([xtrp_latlon(c, leads, 6) for c in cases]),
            "xtrp_gc": np.stack([xtrp_great_circle(c, leads, 12) for c in cases])}


def strongest_extrapolation(errors: dict[str, np.ndarray], names=("xtrp12", "xtrp6", "xtrp_gc")) -> np.ndarray:
    """
    Per case and lead, the error of whichever extrapolation has the lowest *mean* error at that lead on
    this sample -- chosen per lead with hindsight, which only makes the baseline harder to beat.
    """
    mask = None
    for n in names:
        ok = np.isfinite(errors[n])
        mask = ok if mask is None else mask & ok
    out = np.full_like(errors[names[0]], np.nan)
    for k in range(out.shape[1]):
        means = [np.nanmean(np.where(mask[:, k], errors[n][:, k], np.nan)) for n in names]
        best = names[int(np.nanargmin(means))]
        out[:, k] = errors[best][:, k]
    return out


__all__ = ["baseline_tracks", "motion_back", "persistence_intensity", "strongest_extrapolation", "xtrp_great_circle",
           "xtrp_latlon"]
