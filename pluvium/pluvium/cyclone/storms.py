# Copyright 2026 Nathan. Apache-2.0.
"""
A season's storms, and the forecast cases they define.

A case is one storm at one start time. Following the National Hurricane Center's verification rules, a case counts
only when the storm is a tropical or subtropical cyclone in the best track at the start time, and each lead is scored
only when it still is one at the verifying time. The start position and motion come from ``CARQ`` (what was known in
real time), never from the best track written after the season.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .atcf import BestTrack, aids, best_track


@dataclass
class Storm:
    track: BestTrack
    adeck: Path | None

    @property
    def id(self) -> str:
        return self.track.storm

    @property
    def name(self) -> str:
        return self.track.name

    def aids(self, techs=None) -> dict:
        return {} if self.adeck is None else aids(self.adeck, techs)


def load_season(folder, year: int, basin: str = "al") -> list[Storm]:
    """Every storm of a season with a b-deck in ``folder`` (see :func:`pluvium.cyclone.atcf.fetch_decks`)."""
    folder = Path(folder).expanduser()
    storms = []
    for b in sorted(folder.glob(f"b{basin}??{year}.dat*")):
        a = b.with_name("a" + b.name[1:])
        try:
            storms.append(Storm(best_track(b), a if a.exists() else None))
        except ValueError:
            continue
    return storms


def start_hours(storms: list[Storm], cycles=(0,), min_hours: int = 24) -> np.ndarray:
    """Start times (hours since the epoch) at the given UTC cycles when at least one storm is a tropical cyclone and
    will still be tracked ``min_hours`` later."""
    out = set()
    for storm in storms:
        t = storm.track
        for hour in t.hours:
            if int(hour) % 24 in cycles and t.tropical(int(hour)) and t.index(int(hour) + min_hours) is not None:
                out.add(int(hour))
    return np.array(sorted(out), np.int64)


def active(storms: list[Storm], hour: int) -> list[Storm]:
    """Storms that are tropical cyclones at ``hour``."""
    return [s for s in storms if s.track.tropical(int(hour))]


def first_guess(storm: Storm, hour: int, carq: dict | None = None) -> dict | None:
    """Where the storm is and how it moves at ``hour``, as known in real time: ``CARQ`` at lead 0 and 12 hours before.
    Falls back to the best track only when the a-deck has no ``CARQ`` for that hour (flagged in ``source``)."""
    from .geodesy import offset_km

    carq = storm.aids({"CARQ"}).get("CARQ", {}) if carq is None else carq
    rows = carq.get(int(hour), {})
    if 0 in rows:
        lat, lon = rows[0][0], rows[0][1]
        back = rows.get(-12) or rows.get(-6)
        span = 12 if -12 in rows else 6
        source = "CARQ"
    else:
        here = storm.track.position(int(hour))
        if here is None:
            return None
        lat, lon = here
        prior = storm.track.position(int(hour) - 12)
        back, span, source = (None if prior is None else (prior[0], prior[1], np.nan)), 12, "best track"
    if back is None:
        east = north = 0.0
    else:
        east, north = offset_km(back[0], back[1], lat, lon)
        east, north = float(east) / span, float(north) / span          # km per hour
    return {"lat": float(lat), "lon": float(lon), "east_kmh": east, "north_kmh": north, "source": source}


__all__ = ["Storm", "active", "first_guess", "load_season", "start_hours"]
