# Copyright 2026 Nathan. Apache-2.0.
"""
Ocean basins where tropical cyclones form, as boxes the tracker reads fields for.

Each basin has the box ``(south, north, west, east)`` that holds its storms from genesis to extratropical transition,
the ATCF prefix of its decks, and the agency that issues its best tracks. The North Atlantic and the eastern North
Pacific have public real-time decks from the US National Hurricane Center, so forecasts there can be measured against
the official forecast. Other basins are tracked the same way and verified against IBTrACS (:mod:`.ibtracs`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Basin:
    name: str
    box: tuple                 # south, north, west, east (degrees; west < east, longitudes in -180..180 or 0..360)
    atcf: str | None           # deck prefix ("al", "ep", ...) or None when no public deck exists
    agency: str


BASINS = {
    "natl": Basin("North Atlantic", (5.0, 60.0, -105.0, -5.0), "al", "NHC"),
    "epac": Basin("Eastern North Pacific", (5.0, 45.0, -180.0, -80.0), "ep", "NHC"),
    "cpac": Basin("Central North Pacific", (0.0, 45.0, -180.0, -140.0), "cp", "CPHC"),
    "wpac": Basin("Western North Pacific", (0.0, 55.0, 100.0, 180.0), "wp", "JTWC"),
    "nio": Basin("North Indian Ocean", (0.0, 35.0, 45.0, 100.0), "io", "JTWC"),
    "sio": Basin("South Indian Ocean", (-45.0, 0.0, 20.0, 135.0), "sh", "JTWC"),
    "spac": Basin("South Pacific", (-45.0, 0.0, 135.0, 240.0), "sh", "JTWC"),
}


def basin_of(lat: float, lon: float) -> str | None:
    """The basin whose box holds a position (the first match; ``None`` over land far from any)."""
    lon180 = (lon + 180.0) % 360.0 - 180.0
    for key, basin in BASINS.items():
        s, n, w, e = basin.box
        for candidate in (lon180, lon180 % 360.0):
            if s <= lat <= n and w <= candidate <= e:
                return key
    return None


__all__ = ["BASINS", "Basin", "basin_of"]
