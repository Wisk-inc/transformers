# Copyright 2026 Nathan. Apache-2.0.
"""
The operational systems Pluvium is measured against, read exactly as their producers publish them.

* **ECMWF ENS**: the European Centre's 51-member ensemble (one control and 50 perturbed members), 0.25 degrees,
  from ECMWF open data.
* **ECMWF HRES**: the European Centre's single high-resolution IFS run, at the same resolution.
* **ECMWF AIFS**: ECMWF's operational AI weather model (AIFS Single).
* **NOAA GEFS**: the 31-member Global Ensemble Forecast System, 0.5 degrees. Its 6-hour rain buckets are summed.
* **NOAA GraphCast**: NOAA's real-time run of DeepMind's GraphCast, started from GFS analyses, 0.25 degrees. NOAA
  published it as GraphCastGFS until 5 May 2026 and publishes its successor, AIGFS, since; both are read here.
* **NOAA GFS**: the deterministic model that drives Pluvium in the benchmark, 0.5 degrees.

Every reader returns ``{"values": (members, 3, H, W), "lat", "lon"}``: rain accumulated from the start of the run
to 24, 48 and 72 hours, in mm, cropped to a box. Results are cached per run, so a benchmark can be re-run without
downloading again. Only the rain messages are fetched, by byte range from each producer's public bucket. ECMWF's
bucket throttles bursts (HTTP 503 "SlowDown"); :func:`pluvium.util.fetch` retries with backoff.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..util import fetch, hour_to_datetime64, log, parallel_map
from .nwp import GFS05, _index_ncep, decode

#: ECMWF open data has two archives with the same layout: Google Cloud first (it takes bursts), then AWS (which
#: answers bursts with HTTP 503 "SlowDown")
ECMWF_ARCHIVES = ("https://storage.googleapis.com/ecmwf-open-data", "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com")
ECMWF = "{base}/{day}/{hour:02d}z/{path}/{day}{hour:02d}0000-{step}h-{kind}"
GEFS = ("https://noaa-gefs-pds.s3.amazonaws.com/gefs.{day}/{hour:02d}/atmos/pgrb2ap5/"
        "{member}.t{hour:02d}z.pgrb2a.0p50.f{step:03d}")
GRAPHCAST = ("https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/graphcastgfs.{day}/{hour:02d}/forecasts_13_levels/"
             "graphcastgfs.t{hour:02d}z.pgrb2.0p25.f{step:03d}")
#: NOAA's successor to GraphCastGFS, in the same bucket: GraphCast's architecture with newer weights. Rain since the
#: start of the run and sea-level pressure are in its surface files, written as GraphCastGFS wrote them.
AIGFS = ("https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/aigfs.{day}/{hour:02d}/model/atmos/grib2/"
         "aigfs.t{hour:02d}z.sfc.f{step:03d}.grib2")
#: the first run read from AIGFS. GraphCastGFS runs end on 5 May 2026; AIGFS has 00 UTC runs from 22 April 2026
AIGFS_FROM = int(np.datetime64("2026-05-06T00", "h").astype(np.int64))
STEPS = (24, 48, 72)


def graphcast_template(init: int) -> str:
    """The address pattern of NOAA's GraphCast run started at ``init`` (hours since 1970): GraphCastGFS until
    5 May 2026, AIGFS after. Both take ``day``, ``hour`` and ``step``."""
    return AIGFS if int(init) >= AIGFS_FROM else GRAPHCAST

#: name -> (label, members, kind of product)
SYSTEMS = {
    "ens": ("ECMWF ENS", 51, "ensemble"),
    "gefs": ("NOAA GEFS", 31, "ensemble"),
    "ifs": ("ECMWF HRES", 1, "deterministic"),
    "aifs": ("ECMWF AIFS", 1, "deterministic (AI)"),
    "graphcast": ("NOAA GraphCast", 1, "deterministic (AI)"),
    "gfs": ("NOAA GFS", 1, "deterministic"),
}


def _stamp(init: int) -> tuple[str, int]:
    s = str(hour_to_datetime64(init))
    return s[:10].replace("-", ""), int(s[11:13])


def _crop(item, box, pad: float = 0.75):
    values, lat, lon = item
    s, n, w, e = box
    i = (lat >= s - pad) & (lat <= n + pad)
    j = (lon >= w - pad) & (lon <= e + pad)
    return values[np.ix_(i, j)].astype(np.float32), lat[i], lon[j]


def _ecmwf_rows(text: str, param: str = "tp") -> list[tuple[int, int, int]]:
    """``(member number, first byte, last byte)`` of every ``param`` message in an ECMWF ``.index`` (the control,
    which has no number, is member 0). Sorted by member, because the files list members in no particular order."""
    rows = []
    for line in text.splitlines():
        if line.startswith("{"):
            r = json.loads(line)
            if r.get("param") == param:
                start = int(r["_offset"])
                rows.append((int(r.get("number", 0) or 0), start, start + int(r["_length"]) - 1))
    return sorted(rows)


def _first(urls, suffix: str, start: int | None = None, end: int | None = None, trust_missing: bool = False) -> bytes:
    """The first archive that serves ``url + suffix`` (each tried with retries, then the next). With
    ``trust_missing``, a 404 from the first archive is final: AWS answers requests for files that do not exist with
    HTTP 503 "SlowDown", which would otherwise cost a round of retries every time a path is probed."""
    last = None
    for url in urls:
        try:
            return fetch(url + suffix, start, end, timeout=120.0, attempts=4)
        except Exception as error:
            last = error
            if trust_missing and "HTTP 404" in str(error):
                break
    raise RuntimeError(f"no archive serves {urls[0]}{suffix}: {last}")


class BaselineReader:
    """Reads and caches the operational systems' rain for a box ``(south, north, west, east)``."""

    def __init__(self, box, cache, workers: int = 6) -> None:
        self.box = tuple(float(x) for x in box)
        self.cache = Path(cache).expanduser()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.workers = workers

    def read(self, system: str, init: int) -> dict | None:
        """One system's run, from the cache or the archive; ``None`` when the archive does not have it."""
        if system not in SYSTEMS:
            raise ValueError(f"unknown system {system!r}; choose from {', '.join(SYSTEMS)}")
        day, hour = _stamp(init)
        path = self.cache / f"{system}_{day}{hour:02d}.npz"
        if path.exists():
            z = np.load(path)
            return {"values": z["values"], "lat": z["lat"], "lon": z["lon"]}
        try:
            got = getattr(self, "_" + system)(init)
        except Exception as error:
            log(f"{SYSTEMS[system][0]} {str(hour_to_datetime64(init))[:13]}Z: not read ({type(error).__name__}: {error})")
            return None
        values = np.asarray(got["values"], np.float32)
        if values.ndim != 4 or values.shape[1] != len(STEPS):
            raise RuntimeError(f"{system}: unexpected shape {values.shape}")
        if np.nanmin(values) < -0.5 or np.nanmax(values) > 3000:
            raise RuntimeError(f"{system}: rain outside physical range ({np.nanmin(values)}, {np.nanmax(values)}) mm")
        got = {"values": np.maximum(values, 0.0), "lat": np.asarray(got["lat"]), "lon": np.asarray(got["lon"])}
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **got)
        tmp.replace(path)
        return got

    # ------------------------------------------------------------------------------------ ECMWF --
    def _ecmwf(self, init: int, path: str, kind: str, scale: float = 1000.0) -> dict:
        """Every member's total precipitation (metres from the start of the run, converted to mm)."""
        day, hour = _stamp(init)
        jobs = []
        for k, step in enumerate(STEPS):
            urls = [ECMWF.format(base=b, day=day, hour=hour, path=path, step=step, kind=kind) for b in ECMWF_ARCHIVES]
            rows = _ecmwf_rows(_first(urls, ".index").decode())
            if not rows:
                raise RuntimeError(f"no tp in {urls[0]}.index")
            jobs += [(k, m, urls, a, b) for m, a, b in rows]
        members = sorted({m for _, m, _, _, _ in jobs})
        slot = {m: i for i, m in enumerate(members)}

        def one(job):
            k, m, urls, a, b = job
            return k, slot[m], _crop(decode(_first(urls, ".grib2", a, b)), self.box)

        got = parallel_map(one, jobs, workers=self.workers)
        lat, lon = got[0][2][1], got[0][2][2]
        values = np.full((len(members), len(STEPS), lat.size, lon.size), np.nan, np.float32)
        for k, i, item in got:
            values[i, k] = item[0] * scale
        if np.isnan(values).any():
            raise RuntimeError("some members or steps are missing")
        return {"values": values, "lat": lat, "lon": lon}

    def _ens(self, init: int) -> dict:
        return self._ecmwf(init, "ifs/0p25/enfo", "enfo-ef")

    def _ifs(self, init: int) -> dict:
        return self._ecmwf(init, "ifs/0p25/oper", "oper-fc")

    def _aifs(self, init: int) -> dict:
        return self._ecmwf(init, "aifs-single/0p25/oper", "oper-fc", scale=1.0)     # AIFS publishes mm (kg m-2)

    # -------------------------------------------------------------------------------------- NOAA --
    def _gefs(self, init: int, members: int = 31) -> dict:
        """Six-hour buckets of the control and 30 perturbed members, summed to 24, 48 and 72 hours."""
        day, hour = _stamp(init)
        names = ["gec00"] + [f"gep{k:02d}" for k in range(1, members)]
        jobs = [(m, name, step) for m, name in enumerate(names) for step in range(6, max(STEPS) + 1, 6)]

        def one(job):
            m, name, step = job
            url = GEFS.format(day=day, hour=hour, member=name, step=step)
            rows = [r for r in _index_ncep(fetch(url + ".idx", timeout=60.0, attempts=8).decode())
                    if r["var"] == "APCP" and r["level"] == "surface" and f"{step - 6}-{step} hour acc" in r["desc"]]
            if not rows:
                raise RuntimeError(f"no 6-hour rain bucket in {url}.idx")
            return m, step, _crop(decode(fetch(url, rows[0]["start"], rows[0]["end"], attempts=8)), self.box)

        got = parallel_map(one, jobs, workers=self.workers * 2)
        lat, lon = got[0][2][1], got[0][2][2]
        buckets = np.zeros((members, max(STEPS) // 6, lat.size, lon.size), np.float32)
        for m, step, item in got:
            buckets[m, step // 6 - 1] = item[0]
        cum = np.cumsum(buckets, axis=1)[:, [s // 6 - 1 for s in STEPS]]
        return {"values": cum, "lat": lat, "lon": lon}

    def _ncep_cumulative(self, init: int, template: str, **fmt) -> dict:
        """Rain from the start of the run, published as "0-N day acc fcst" at whole days."""
        day, hour = _stamp(init)
        fields = []
        for step in STEPS:
            url = template.format(day=day, hour=hour, step=step, lead=step, **fmt)
            want = f"0-{step // 24} day acc"
            rows = [r for r in _index_ncep(fetch(url + ".idx", timeout=60.0, attempts=8).decode())
                    if r["var"] == "APCP" and r["level"] == "surface" and want in r["desc"]]
            if not rows:
                raise RuntimeError(f"no '{want}' rain in {url}.idx")
            fields.append(_crop(decode(fetch(url, rows[0]["start"], rows[0]["end"], attempts=8)), self.box))
        return {"values": np.stack([f[0] for f in fields])[None], "lat": fields[0][1], "lon": fields[0][2]}

    def _graphcast(self, init: int) -> dict:
        return self._ncep_cumulative(init, graphcast_template(init))

    def _gfs(self, init: int) -> dict:
        return self._ncep_cumulative(init, GFS05)


def at_points(field: dict, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Bilinear values ``(members, 3, P)`` of a system's rain at points."""
    from ..dataset import bilinear

    return bilinear(field["values"], np.asarray(field["lat"], np.float64), np.asarray(field["lon"], np.float64),
                    np.asarray(lat), np.asarray(lon))


def daily(values: np.ndarray) -> np.ndarray:
    """Accumulations to 24/48/72 h ``(..., 3, P)`` -> the rain of days 1, 2 and 3 ``(..., 3, P)``."""
    out = np.diff(values, axis=-2, prepend=np.zeros_like(values[..., :1, :]))
    return np.maximum(out, 0.0)


__all__ = ["SYSTEMS", "STEPS", "BaselineReader", "at_points", "daily"]
