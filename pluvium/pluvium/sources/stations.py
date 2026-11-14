# Copyright 2026 Nathan. Apache-2.0.
"""
Point truth: hourly weather reports from about 250 Caribbean airports and stations (NOAA's Integrated Surface
Database, ``noaa-global-hourly-pds`` on AWS Open Data).

A gridded rain estimate says how much fell in an 8 km box; an airport report says whether it was raining at the
airport at 14:55, how hard, whether it was drizzle, rain or showers, and whether there was thunder. For *when* rain
arrives in a town and *what kind* of rain it is, that is the sharper truth, and the model is trained against both.

Each station-hour is stored as the packed code of :mod:`pluvium.weather` (``-1`` when nothing was reported): the
raw METAR in the remarks is read first, then SYNOP present-weather codes and one-hour gauge totals. A report at
14:00 describes the end of hour 13 (reports are assigned to the hour five minutes before their time).
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np

from ..domain import TARGET_BOX
from ..util import atomic_save, fetch, log, parallel_map, year_hours
from ..weather import UNKNOWN, code, code_from_gauge, code_from_metar, code_from_wmo, decode, merge

HISTORY = "https://noaa-isd-pds.s3.amazonaws.com/isd-history.csv"
HOURLY = "https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/{station}.csv"
#: version of the per-hour codes in the staged arrays (2: packed weather codes of :mod:`pluvium.weather`)
FORMAT = 2


def station_list(text: str, box=TARGET_BOX, first_year: int = 2000, last_year: int = 2100) -> list[dict]:
    """Stations of ``isd-history.csv`` inside ``box`` that reported at some point in ``[first_year, last_year]``."""
    s, n, w, e = box
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        try:
            lat, lon = float(r["LAT"]), float(r["LON"])
        except (TypeError, ValueError):
            continue
        if not (s <= lat <= n and w <= lon <= e) or (lat == 0 and lon == 0):
            continue
        if r["END"][:4] < str(first_year) or r["BEGIN"][:4] > str(last_year):
            continue
        out.append({"id": r["USAF"] + r["WBAN"], "name": r["STATION NAME"].strip().title(), "country": r["CTRY"],
                    "icao": (r.get("ICAO") or "").strip(), "lat": lat, "lon": lon,
                    "elevation": float(r["ELEV(M)"] or 0.0), "begin": r["BEGIN"], "end": r["END"]})
    return out


def _code(value: str) -> int | None:
    try:
        return int(value.split(",")[0])
    except (ValueError, AttributeError, IndexError):
        return None


def _hour(date: str, first: int) -> int | None:
    """The hour a report describes: its time minus five minutes (a report at 14:00 describes the end of hour 13)."""
    try:
        t = np.datetime64(date[:16], "m") - np.timedelta64(5, "m")
    except ValueError:
        return None
    return int((t.astype("datetime64[h]") - np.datetime64("1970-01-01T00", "h")).astype(np.int64)) - first


def parse_hourly(text: str, year: int) -> np.ndarray:
    """
    One station-year of ISD CSV -> int8 per hour of ``year``: the packed weather code of :mod:`pluvium.weather`
    (intensity class, thunder, drizzle/rain/showers), ``-1`` where nothing was reported.

    The raw METAR in the remarks is read first (it says light/moderate/heavy, showers, drizzle, thunder); SYNOP-style
    present-weather codes (manual WMO 4677, automatic 4680) and one-hour gauge totals fill in where there is none.
    """
    first, end = year_hours(year)
    n = end - first
    out = np.full(n, UNKNOWN, np.int16)

    def put(h, c):
        if c is not None and c >= 0 and 0 <= h < n:
            out[h] = merge(int(out[h]), int(c))

    for r in csv.DictReader(io.StringIO(text)):
        date = r.get("DATE", "")
        if len(date) < 13:
            continue
        h = _hour(date if len(date) >= 16 else date[:13] + ":00", first)
        if h is None:
            continue
        rem = r.get("REM") or ""
        got = code_from_metar(rem) if ("METAR" in rem or "SPECI" in rem) else None
        if got is not None:
            put(h, got[0])
            put(h - 1, got[1])
            continue
        said = False
        for key in ("MW1", "MW2", "MW3", "MW4", "MW5", "MW6", "MW7"):
            c = _code(r.get(key, ""))
            if c is None:
                continue
            now, before = code_from_wmo(c, automated=False)
            if now is not None:
                said = True
                put(h, now)
            put(h - 1, before)
        for key in ("AW1", "AW2", "AW3", "AW4"):
            c = _code(r.get(key, ""))
            if c is None:
                continue
            now, before = code_from_wmo(c, automated=True)
            if now is not None:
                said = True
                put(h, now)
            put(h - 1, before)
        for key in ("AA1", "AA2", "AA3", "AA4"):
            parts = (r.get(key) or "").split(",")
            if len(parts) < 2:
                continue
            try:
                period, depth = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if period != 1 or depth == 9999:
                continue
            put(h, code_from_gauge(depth / 10.0))
            said = True
        if not said and r.get("REPORT_TYPE", "").strip() in ("FM-15", "FM-16"):
            put(h, code(0))
    return out.astype(np.int8)


class StationCache:
    """Hourly wet/dry at stations: ``meta`` (ids, names, lat, lon) and ``read(hours) -> (len(hours), n) int8``."""

    def __init__(self, root) -> None:
        self.root = Path(root).expanduser() / "stations"
        self.meta = json.loads((self.root / "stations.json").read_text())
        if int(self.meta.get("format", 1)) < FORMAT:
            raise RuntimeError(f"{self.root} holds airport reports in an older format; stage them into a new folder")
        self.lat = np.array([s["lat"] for s in self.meta["stations"]], np.float64)
        self.lon = np.array([s["lon"] for s in self.meta["stations"]], np.float64)
        self._open = {}

    def __len__(self) -> int:
        return self.lat.size

    def read(self, hours) -> np.ndarray:
        from ..util import calendar

        hours = np.asarray(hours, np.int64)
        out = np.full((hours.size, len(self)), -1, np.int8)
        years = calendar(hours)["year"]
        for year in np.unique(years):
            path = self.root / f"{int(year)}.npy"
            if not path.exists():
                continue
            if int(year) not in self._open:
                self._open[int(year)] = np.load(path, mmap_mode="r")
            first, _ = year_hours(int(year))
            sel = np.nonzero(years == year)[0]
            out[sel] = self._open[int(year)][hours[sel] - first]
        return out


def stage_stations(root, years, box=TARGET_BOX, workers: int = 16, verbose: bool = True) -> StationCache:
    """Stations in ``box`` with their hourly wet/dry for ``years`` (one array per year)."""
    root = Path(root).expanduser() / "stations"
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / "stations.json"
    if meta_path.exists():
        stations = json.loads(meta_path.read_text())["stations"]
    else:
        stations = station_list(fetch(HISTORY).decode("utf-8", "replace"), box, min(years), max(years))
        meta_path.write_text(json.dumps({"stations": stations, "format": FORMAT}))
    for year in sorted(years):
        path = root / f"{year}.npy"
        if path.exists():
            continue

        def one(station, y=year):
            blob = fetch(HOURLY.format(year=y, station=station["id"]), missing_ok=True)
            if blob is None:
                return None
            return parse_hourly(blob.decode("utf-8", "replace"), y)

        rows = parallel_map(one, stations, workers=workers, progress=f"stations {year}" if verbose else None)
        first, end = year_hours(year)
        arr = np.full((end - first, len(stations)), -1, np.int8)
        for k, r in enumerate(rows):
            if r is not None:
                arr[:, k] = r
        atomic_save(path, lambda p, a=arr: np.save(p, a))
        if verbose:
            d = decode(arr)
            known = int(d["known"].sum())
            log(f"  stations {year}: {sum(r is not None for r in rows)} of {len(stations)} reporting, "
                f"{known:,} known station-hours, {d['wet'].sum() / max(known, 1):.1%} wet, "
                f"{d['thunder'].sum() / max(known, 1):.2%} with thunder")
    return StationCache(root.parent)


__all__ = ["HISTORY", "HOURLY", "StationCache", "parse_hourly", "stage_stations", "station_list"]
