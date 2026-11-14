# Copyright 2026 Nathan. Apache-2.0.
"""
Live airport reports: the last hours of METARs from every Caribbean airport, from the Aviation Weather Center's open
API (``aviationweather.gov``), turned into the same per-hour codes the model was trained on (:mod:`pluvium.weather`).

They do two jobs live: they are the "airport reports of the last three hours" the model reads into its mesh, and
they are the truth :mod:`pluvium.online` checks past forecasts against.
"""

from __future__ import annotations

import json

import numpy as np

from ..domain import TARGET_BOX
from ..util import fetch, great_circle_km, log
from ..weather import UNKNOWN, code_from_metar, merge

API = "https://aviationweather.gov/api/data/metar?bbox={s},{w},{n},{e}&format=json&hours={hours}"


def fetch_metars(box=TARGET_BOX, hours: int = 4) -> list[dict]:
    """Reports in ``box`` from the last ``hours``: ``[{"icao", "lat", "lon", "time" (unix s), "raw"}]``."""
    s, n, w, e = box
    blob = fetch(API.format(s=s, w=w, n=n, e=e, hours=int(hours)), timeout=60.0, attempts=3)
    rows = json.loads(blob.decode("utf-8", "replace") or "[]")
    out = []
    for r in rows:
        try:
            out.append({"icao": str(r.get("icaoId", "")).upper(), "lat": float(r["lat"]), "lon": float(r["lon"]),
                        "time": int(r["obsTime"]), "raw": str(r.get("rawOb", ""))})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def report_hour(unix_seconds: int) -> int:
    """The hour a report describes (its time minus five minutes), as integer hours since 1970."""
    return int((int(unix_seconds) - 300) // 3600)


def station_codes(stations: list[dict], metars: list[dict], hours, max_km: float = 3.0) -> np.ndarray:
    """``(len(hours), len(stations))`` packed codes from live reports, matched to the model's stations by ICAO code
    (or by position within ``max_km``); ``-1`` where no report covers the hour."""
    hours = [int(h) for h in hours]
    where = {h: i for i, h in enumerate(hours)}
    out = np.full((len(hours), len(stations)), UNKNOWN, np.int16)
    by_icao = {str(st.get("icao", "")).upper(): k for k, st in enumerate(stations) if st.get("icao")}
    lat = np.array([st["lat"] for st in stations])
    lon = np.array([st["lon"] for st in stations])
    for m in metars:
        k = by_icao.get(m["icao"])
        if k is None and len(stations):
            d = great_circle_km(lat, lon, m["lat"], m["lon"])
            j = int(np.argmin(d))
            k = j if d[j] <= max_km else None
        if k is None:
            continue
        got = code_from_metar(m["raw"])
        if got is None:
            continue
        h = report_hour(m["time"])
        for hour, c in ((h, got[0]), (h - 1, got[1])):
            if hour in where and c >= 0:
                out[where[hour], k] = merge(int(out[where[hour], k]), int(c))
    return out.astype(np.int8)


def live_station_codes(stations: list[dict], hours, box=TARGET_BOX) -> np.ndarray | None:
    """Fetch and match; None when the API cannot be reached."""
    try:
        span = max(1, max(hours) - min(hours) + 3)
        return station_codes(stations, fetch_metars(box, hours=span), hours)
    except Exception as error:
        log(f"airport reports not read ({type(error).__name__}: {error}); forecasting without them")
        return None


__all__ = ["API", "fetch_metars", "live_station_codes", "report_hour", "station_codes"]
