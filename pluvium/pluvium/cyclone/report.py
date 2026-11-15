# Copyright 2026 Nathan. Apache-2.0.
"""
A forecast a person can read: the bulletin for one storm.

:func:`bulletin` turns :class:`pluvium.cyclone.products.Guidance` into text: where the centre is expected at each
lead and how far off that may be, which places it may pass and with what chance, and whether the systems split into
scenarios. :func:`to_json` gives the same content for programs, :func:`track_csv` the track as a table.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .geodesy import bearing_deg, distance_km
from .products import Guidance

COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")


def compass(bearing: float) -> str:
    return COMPASS[int((float(bearing) % 360.0 + 11.25) // 22.5) % 16]


def position(lat: float, lon: float) -> str:
    lon = (lon + 180.0) % 360.0 - 180.0
    return f"{abs(lat):.1f}{'N' if lat >= 0 else 'S'} {abs(lon):.1f}{'W' if lon < 0 else 'E'}"


def stamp(hour: int) -> str:
    return str(np.datetime64(int(hour), "h")).replace("T", " ") + " UTC"


def motion(g: Guidance) -> str:
    """The expected motion over the first 24 hours, in words."""
    if g.start is None or 24 not in g.track:
        return ""
    a, b = g.start, g.track[24]
    km = float(distance_km(a[0], a[1], b[0], b[1]))
    if km < 20.0:
        return "nearly stationary over the next 24 hours"
    return f"moving {compass(bearing_deg(a[0], a[1], b[0], b[1]))} at about {km / 24.0:.0f} km/h over the next 24 hours"


def bulletin(g: Guidance, name: str = "", radius_km: float = 150.0, top_places: int = 8, min_chance: float = 0.05) -> str:
    """The forecast as text."""
    lines = [f"PLUVIUM CYCLONE: {name or g.storm}", f"Forecast from {stamp(g.init)}"]
    if g.start is not None:
        lines.append(f"Centre at the start: {position(*g.start)}; {motion(g)}".rstrip("; "))
    lines += ["", "Hours  Valid                 Centre          Two-in-three circle"]
    for lead in sorted(g.track):
        lat, lon = g.track[lead]
        r = g.cone.get(lead, float("nan"))
        lines.append(f"{lead:>5}  {stamp(g.init + lead):<20}  {position(lat, lon):<14}  {r:>4.0f} km ({r / 1.852:.0f} nm)")
    places = sorted(((v["chance"], k, v) for k, v in g.places.items() if v["chance"] >= min_chance), reverse=True)[:top_places]
    if places:
        lines += ["", f"Chance that the centre passes within {radius_km:.0f} km in the next five days:"]
        for chance, place, v in places:
            when = "" if not np.isfinite(v["track_km"]) else \
                f"  (the forecast track comes within {v['track_km']:.0f} km at {v['track_hour']:.0f} h)"
            lines.append(f"  {place:<18} {100 * chance:>3.0f}%{when}")
    elif g.places:
        lines += ["", f"No listed place has a chance of {100 * min_chance:.0f}% or more of the centre passing within "
                      f"{radius_km:.0f} km in the next five days."]
    if len(g.scenarios) > 1:
        lines += ["", f"The systems split into {len(g.scenarios)} scenarios:"]
        for k, s in enumerate(g.scenarios):
            last = max(s.points)
            lines.append(f"  {k + 1}. {100 * s.weight:.0f}% of the weight: centre near {position(*s.points[last])} at {last} h")
    lines += ["", "The circle should hold the centre two times in three; it does not show the storm's size. "
                  "Track only: this is not a forecast of wind, surge or rain."]
    return "\n".join(lines) + "\n"


def to_json(g: Guidance, name: str = "") -> dict:
    """The forecast as plain data."""
    return {"storm": g.storm, "name": name, "init": stamp(g.init), "start": None if g.start is None else list(g.start),
            "track": {str(k): list(v) for k, v in sorted(g.track.items())},
            "cone_km": {str(k): float(v) for k, v in sorted(g.cone.items())},
            "spread_km": {str(k): (None if not np.isfinite(v) else float(v)) for k, v in sorted(g.spread.items())},
            "places": {k: {a: (None if isinstance(b, float) and not np.isfinite(b) else b) for a, b in v.items()}
                       for k, v in g.places.items()},
            "scenarios": [{"weight": s.weight, "track": {str(k): list(v) for k, v in sorted(s.points.items())}}
                          for s in g.scenarios]}


def track_csv(g: Guidance, path) -> Path:
    """The track as a table: lead, valid time, latitude, longitude, cone radius."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lead_hours", "valid_utc", "lat", "lon", "cone_km"])
        for lead in sorted(g.track):
            lat, lon = g.track[lead]
            w.writerow([lead, stamp(g.init + lead), f"{lat:.2f}", f"{lon:.2f}", f"{g.cone.get(lead, float('nan')):.0f}"])
    return path


__all__ = ["bulletin", "compass", "motion", "position", "stamp", "to_json", "track_csv"]
