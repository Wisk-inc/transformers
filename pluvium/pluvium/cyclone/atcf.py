# Copyright 2026 Nathan. Apache-2.0.
"""
The ATCF decks of the US National Hurricane Center: best tracks (b-decks) and every forecast aid (a-decks).

A b-deck holds a storm's best track: position, maximum wind, central pressure and type every six hours. An a-deck holds
every forecast made for the storm, one technique per code. The ones Pluvium is measured against:

* ``OFCL``, the official NHC forecast;
* ``TVCN`` and ``HCCA``, NHC's consensus aids;
* ``AVNO``/``AVNI`` (GFS), ``EMX``/``EMXI`` (ECMWF), ``EEMN`` and ``AEMN`` (the ensemble means);
* ``GDMN``/``GDMI`` (Google DeepMind's cyclone model);
* ``CLP5`` and ``OCD5``, the climatology-and-persistence baselines that define "no skill";
* ``CARQ``, the position and motion known in real time, which is what a tracker may start from.

Codes ending in ``I`` are the "early" versions: the previous cycle's run, shifted to the current position, which is
what a forecaster has in hand at advisory time.

Decks are plain comma-separated text: https://ftp.nhc.noaa.gov/atcf/archive/YYYY/ (documented in the ATCF manual).
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..util import hour_of

#: types that count as a tropical or subtropical cyclone (NHC verifies only these, at both ends of a forecast)
TROPICAL = frozenset({"TD", "TS", "HU", "SD", "SS", "TY", "ST", "TC"})
ARCHIVE = "https://ftp.nhc.noaa.gov/atcf/archive/{year}/{kind}{basin}{number:02d}{year}.dat.gz"


def _open(path):
    path = Path(path)
    return gzip.open(path, "rt", encoding="ascii", errors="replace") if path.suffix == ".gz" else path.open(errors="replace")


def _coordinate(text: str) -> float:
    """``140N`` -> 14.0, ``700W`` -> -70.0 (tenths of a degree with a hemisphere letter)."""
    text = text.strip()
    if not text or text[-1] not in "NSEW":
        return float("nan")
    value = float(text[:-1] or 0) / 10.0
    return -value if text[-1] in "SW" else value


def _hour(stamp: str) -> int:
    s = stamp.strip()
    return hour_of(f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}")


def read_rows(path) -> list[dict]:
    """Every usable line of a deck: one dict per (technique, start time, lead), the first wind-radius row only."""
    rows, seen = [], set()
    with _open(path) as handle:
        for line in handle:
            p = [x.strip() for x in line.split(",")]
            if len(p) < 9 or not p[2][:10].isdigit():
                continue
            try:
                tau = int(p[5] or 0)
            except ValueError:
                continue
            key = (p[4], p[2], tau)
            if key in seen:                                   # 34, 50 and 64 kt radii repeat the position
                continue
            seen.add(key)
            lat, lon = _coordinate(p[6]), _coordinate(p[7])
            if not np.isfinite(lat) or not np.isfinite(lon) or (lat == 0.0 and lon == 0.0):
                continue
            rows.append({"basin": p[0], "number": int(p[1]), "hour": _hour(p[2]), "tech": p[4], "tau": tau,
                         "lat": lat, "lon": lon, "vmax": float(p[8] or 0) if p[8].lstrip("-").isdigit() else np.nan,
                         "mslp": float(p[9]) if len(p) > 9 and p[9].isdigit() and int(p[9]) > 0 else np.nan,
                         "kind": p[10] if len(p) > 10 else "", "name": p[27] if len(p) > 27 else ""})
    return rows


@dataclass
class BestTrack:
    """One storm's best track, six-hourly."""

    storm: str                      # "AL132025"
    name: str
    hours: np.ndarray               # hours since the epoch (util.hour_of)
    lat: np.ndarray
    lon: np.ndarray
    vmax: np.ndarray                # kt
    mslp: np.ndarray                # hPa
    kind: np.ndarray                # "TD", "TS", "HU", "EX", ...

    def index(self, hour: int) -> int | None:
        i = int(np.searchsorted(self.hours, hour))
        return i if i < self.hours.size and self.hours[i] == hour else None

    def position(self, hour: int) -> tuple[float, float] | None:
        i = self.index(hour)
        return None if i is None else (float(self.lat[i]), float(self.lon[i]))

    def tropical(self, hour: int) -> bool:
        i = self.index(hour)
        return i is not None and str(self.kind[i]) in TROPICAL

    def heading(self, hour: int, span: int = 12) -> float:
        """The storm's heading at ``hour`` from the positions ``span`` hours before and after (degrees)."""
        from .geodesy import bearing_deg

        a = self.index(hour - span)
        b = self.index(hour + span)
        a = self.index(hour) if a is None else a
        b = self.index(hour) if b is None else b
        if a is None or b is None or a == b:
            return float("nan")
        return float(bearing_deg(self.lat[a], self.lon[a], self.lat[b], self.lon[b]))


def best_track(path) -> BestTrack:
    rows = [r for r in read_rows(path) if r["tech"] == "BEST"]
    if not rows:
        raise ValueError(f"{path}: no best-track rows")
    rows.sort(key=lambda r: r["hour"])
    names = [r["name"] for r in rows if r["name"] and r["name"] not in ("INVEST", "GENESIS") and not r["name"].isdigit()]
    first = rows[0]
    year = str(np.datetime64(int(first["hour"]), "h"))[:4]
    return BestTrack(f"{first['basin']}{first['number']:02d}{year}", names[-1] if names else "UNNAMED",
                     np.array([r["hour"] for r in rows], np.int64), np.array([r["lat"] for r in rows]),
                     np.array([r["lon"] for r in rows]), np.array([r["vmax"] for r in rows]),
                     np.array([r["mslp"] for r in rows]), np.array([r["kind"] for r in rows]))


def aids(path, techs=None) -> dict:
    """``{technique: {start hour: {lead hours: (lat, lon, vmax)}}}`` from an a-deck."""
    want = None if techs is None else set(techs)
    out = {}
    for r in read_rows(path):
        if want is not None and r["tech"] not in want:
            continue
        out.setdefault(r["tech"], {}).setdefault(r["hour"], {})[r["tau"]] = (r["lat"], r["lon"], r["vmax"])
    return out


def fetch_decks(folder, year: int, basin: str = "al", numbers=range(1, 32)) -> list[Path]:
    """Download a season's decks into ``folder`` (skipping those already there); returns the b-deck paths found."""
    from ..util import fetch

    folder = Path(folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    found = []
    for number in numbers:
        got = True
        for kind in ("b", "a"):
            path = folder / f"{kind}{basin}{number:02d}{year}.dat.gz"
            if not path.exists():
                blob = fetch(ARCHIVE.format(year=year, kind=kind, basin=basin, number=number), attempts=3, missing_ok=True)
                if blob is None:
                    got = False
                    break
                path.write_bytes(blob)
        if not got:
            break
        found.append(folder / f"b{basin}{number:02d}{year}.dat.gz")
    return found


__all__ = ["ARCHIVE", "TROPICAL", "BestTrack", "aids", "best_track", "fetch_decks", "read_rows"]
