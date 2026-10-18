# Copyright 2026 Nathan. Apache-2.0.
"""
The satellite path into the swarm: every centre fix NHC receives, as it arrives.

The ATCF fix deck (``f``-deck) is the record of every independent estimate of where a storm's centre is: subjective
and objective Dvorak fixes from geostationary imagery (``DVTS``, ``DVTO``), microwave imagery (``MICR``, ``AMSR``,
``GMI``, ``SSMI``, ``MWI``), scatterometer winds (``ASCT``), satellite centre fixes (``SATC``), aircraft
reconnaissance vortex fixes (``AIRC``), dropsondes (``DRPS``), land radar (``RDRT``, ``RDRD``) and analyses
(``ANAL``). During a recon mission they arrive several times an hour. The best-track deck (``b``-deck) is the
6-hourly working best track.

Each fix becomes a likelihood with its own width: recon and radar know the centre to a few kilometres, a Dvorak fix
on a cloud-covered centre to tens of kilometres, and the fix's own position-confidence code (1 good, 2 fair, 3
poor) widens it further. :func:`assimilate` streams them into a :class:`naturev1.swarm.core.Swarm` in time order,
advancing the worlds in micro-segments to each fix's exact minute.

No credentials: NHC's servers are public. Live: :func:`active_storms`, :func:`fetch_deck`. Archive: the same with
``archive=True`` (``ftp.nhc.noaa.gov/atcf/archive/<year>``).
"""

from __future__ import annotations

import gzip
import json
import urllib.request
from dataclasses import dataclass

import numpy as np

NHC = "https://www.nhc.noaa.gov/CurrentStorms.json"
ATCF = "https://ftp.nhc.noaa.gov/atcf"
#: One-sigma centre uncertainty (km) of each fix type at position confidence 1; confidence 2 and 3 multiply it.
FIX_SIGMA_KM = {"AIRC": 8.0, "DRPS": 15.0, "RDRT": 12.0, "RDRD": 12.0, "RDRC": 12.0, "MICR": 25.0, "AMSR": 25.0,
                "GMI": 25.0, "SSMI": 25.0, "SSMS": 25.0, "MWI": 25.0, "TMI": 25.0, "ASCT": 25.0, "SATC": 30.0,
                "DVTS": 25.0, "DVTO": 30.0, "ANAL": 20.0, "BEST": 20.0}
CONFIDENCE_FACTOR = {1: 1.0, 2: 1.6, 3: 2.4}


@dataclass
class Fix:
    time: np.datetime64
    lat: float
    lon: float
    sigma_km: float
    source: str
    wind_kt: float = float("nan")


def _get(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data = response.read()
    return gzip.decompress(data) if url.endswith(".gz") else data


def active_storms() -> list[dict]:
    """NHC's active storms (Atlantic, east and central Pacific): id, name, class, intensity, position, motion."""
    out = []
    for s in json.loads(_get(NHC)).get("activeStorms", []):
        out.append({"id": s["id"], "name": s.get("name"), "classification": s.get("classification"),
                    "wind_kt": float(s.get("intensity") or "nan"), "pressure_hpa": float(s.get("pressure") or "nan"),
                    "lat": float(s.get("latitudeNumeric")), "lon": float(s.get("longitudeNumeric")),
                    "heading_deg": s.get("movementDir"), "speed_mph": s.get("movementSpeed"),
                    "time": np.datetime64(s.get("lastUpdate", "").replace("Z", "").split(".")[0])})
    return out


def fetch_deck(storm_id: str, kind: str = "b", archive: bool = False) -> str:
    """The text of a b- or f-deck, e.g. ``fetch_deck("al132025", "f", archive=True)``."""
    sid = storm_id.lower()
    year = sid[-4:]
    folder = {"b": "btk", "f": "fix", "a": "aid_public"}[kind]
    names = [f"{ATCF}/archive/{year}/{kind}{sid}.dat.gz", f"{ATCF}/archive/{year}/{kind}{sid}.dat"] if archive else \
        [f"{ATCF}/{folder}/{kind}{sid}.dat", f"{ATCF}/{folder}/{kind}{sid}.dat.gz"]
    last = None
    for url in names:
        try:
            return _get(url).decode("latin-1")
        except Exception as error:
            last = error
    raise RuntimeError(f"no {kind}-deck for {storm_id}: {last}")


def _coord(text: str, tenths: bool) -> float:
    text = text.strip()
    if not text or text[-1] not in "NSEW":
        return float("nan")
    value = float(text[:-1]) / (10.0 if tenths else 100.0)
    return -value if text[-1] in "SW" else value


def _time(text: str) -> np.datetime64:
    t = text.strip()
    t = t + "00" if len(t) == 10 else t
    return np.datetime64(f"{t[:4]}-{t[4:6]}-{t[6:8]}T{t[8:10]}:{t[10:12]}")


def parse_bdeck(text: str) -> list[Fix]:
    """Best-track positions (one per time; the wind-radii repeats are folded)."""
    seen, out = set(), []
    for line in text.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 9 or not f[2].isdigit():
            continue
        t = _time(f[2])
        if t in seen:
            continue
        lat, lon = _coord(f[6], True), _coord(f[7], True)
        if not np.isfinite(lat):
            continue
        seen.add(t)
        out.append(Fix(t, lat, lon, FIX_SIGMA_KM["BEST"], "BEST", float(f[8]) if f[8].isdigit() else float("nan")))
    return sorted(out, key=lambda x: x.time)


def parse_fdeck(text: str) -> list[Fix]:
    """Centre fixes from an f-deck: rows flagged as centre fixes with a position; intensity kept when given."""
    out = []
    for line in text.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 11 or not f[2].isdigit():
            continue
        kind, flags = f[4], f[5]
        if flags and "C" not in flags:
            continue
        lat, lon = _coord(f[7], False), _coord(f[8], False)
        if not (np.isfinite(lat) and np.isfinite(lon)):
            continue
        conf = int(f[10]) if f[10].isdigit() else 2
        sigma = FIX_SIGMA_KM.get(kind, 35.0) * CONFIDENCE_FACTOR.get(conf, 1.6)
        wind = float(f[11]) if len(f) > 11 and f[11].isdigit() else float("nan")
        out.append(Fix(_time(f[2]), lat, lon, sigma, kind, wind))
    return sorted(out, key=lambda x: x.time)


def merge(fixes: list[Fix], minutes: float = 10.0) -> list[Fix]:
    """Fixes within ``minutes`` of each other combined by inverse-variance weighting (they are not independent
    information about where the worlds should be at the same instant)."""
    out: list[Fix] = []
    group: list[Fix] = []

    def flush():
        if not group:
            return
        w = np.array([1.0 / g.sigma_km ** 2 for g in group])
        lat = float(np.sum(w * [g.lat for g in group]) / w.sum())
        lon = float(np.sum(w * [g.lon for g in group]) / w.sum())
        winds = [g.wind_kt for g in group if np.isfinite(g.wind_kt)]
        t = group[int(np.argmax(w))].time
        out.append(Fix(t, lat, lon, float(1.0 / np.sqrt(w.sum())), "+".join(sorted({g.source for g in group})),
                       float(np.mean(winds)) if winds else float("nan")))

    for fx in sorted(fixes, key=lambda x: x.time):
        if group and (fx.time - group[0].time) / np.timedelta64(1, "m") > minutes:
            flush()
            group = []
        group.append(fx)
    flush()
    return out


def window(fixes: list[Fix], start, end) -> list[Fix]:
    start, end = np.datetime64(start), np.datetime64(end)
    return [f for f in fixes if start <= f.time <= end]


def assimilate(swarm, fixes: list[Fix], analysis_time, sigma_floor_km: float = 8.0) -> list[dict]:
    """
    Stream ``fixes`` into ``swarm`` in time order: the worlds advance in micro-segments to each fix's minute (hours
    relative to ``analysis_time``) and are re-weighted by it. Fixes earlier than the swarm's own clock are skipped.
    Returns the filter's history (time, effective sample size, median distance of the worlds from each fix).
    """
    t0 = np.datetime64(analysis_time)
    for fx in sorted(fixes, key=lambda x: x.time):
        hours = float((fx.time - t0) / np.timedelta64(1, "m")) / 60.0
        if hours < swarm.hours - 1e-6:
            continue
        swarm.advance_to(hours)
        swarm.update(fx.lat, fx.lon, max(fx.sigma_km, sigma_floor_km), fx.wind_kt if np.isfinite(fx.wind_kt) else None)
        swarm.history[-1]["source"] = fx.source
    return swarm.history




def reweight_forecast(forecast, fixes: list[Fix], analysis_time, sigma_floor_km: float = 8.0) -> float:
    """
    Fixes that arrived after the analysis re-weight the forecast worlds directly: each world's position at the
    fix's minute (interpolated along its track) against the fix. Between analysis cycles this is how the swarm
    keeps up with the satellites in real time. Returns the effective sample size as a fraction of the worlds.
    """
    from .core import _haversine_np

    t0 = np.datetime64(analysis_time)
    hours = forecast.track_hours if forecast.track is not None else np.asarray(forecast.leads, float)
    track = forecast.track if forecast.track is not None else forecast.positions
    logw = np.log(np.maximum(forecast.weights, 1e-300))
    for fx in fixes:
        h = float((fx.time - t0) / np.timedelta64(1, "m")) / 60.0
        if h <= 0 or h > hours[-1]:
            continue
        k = int(np.clip(np.searchsorted(hours, h) - 1, 0, hours.size - 2))
        a = (h - hours[k]) / (hours[k + 1] - hours[k])
        lat = track[:, k, 0] * (1 - a) + track[:, k + 1, 0] * a
        lon = track[:, k, 1] * (1 - a) + track[:, k + 1, 1] * a
        d = _haversine_np(lat, lon, fx.lat, fx.lon)
        logw = logw - 0.5 * (d / max(fx.sigma_km, sigma_floor_km)) ** 2
    logw -= logw.max()
    w = np.exp(logw)
    w /= w.sum()
    forecast.weights = w
    return float(1.0 / np.sum(w * w)) / w.size


__all__ = ["CONFIDENCE_FACTOR", "FIX_SIGMA_KM", "Fix", "active_storms", "assimilate", "fetch_deck", "merge",
           "parse_bdeck", "parse_fdeck", "reweight_forecast", "window"]
