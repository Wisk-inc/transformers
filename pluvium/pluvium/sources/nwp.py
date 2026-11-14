# Copyright 2026 Nathan. Apache-2.0.
"""
Live driving fields: GFS (NOAA, hourly to 5 days), ECMWF IFS and AIFS (ECMWF open data, CC-BY-4.0), read by byte
range from their public buckets and averaged onto the exact grids the model was trained on (ERA5's 1 degree hourly
and 1.5 degree upper-air), so live input looks like training input.

Only the messages needed are fetched: rain (cumulative, differenced into each frame), water vapour, 10 m wind, 2 m
temperature, sea-level pressure and cloud cover for every frame; wind, humidity and temperature at 850/700/500/200
hPa every three hours (interpolated between). Where a model lacks a field (AIFS has no column water vapour), it is
derived from what it has, and the substitution is recorded. A run is cached by model and cycle.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..features import HOURLY_SHORT, LEVELS, UPPER_SHORT
from ..schedule import DEFAULT, Schedule
from ..util import fetch, hour_to_datetime64, log, parallel_map

GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{day}/{hour:02d}/atmos/gfs.t{hour:02d}z.pgrb2.0p25.f{lead:03d}"
ECMWF = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/{day}/{hour:02d}z/{model}/0p25/oper/{day}{hour:02d}0000-{lead}h-oper-fc.grib2"
ECMWF_SITE = "https://data.ecmwf.int/forecasts/{day}/{hour:02d}z/{model}/0p25/oper/{day}{hour:02d}0000-{lead}h-oper-fc.grib2"

NCEP_SURFACE = {"tcwv": ("PWAT", "entire atmosphere (considered as a single layer)"), "u10": ("UGRD", "10 m above ground"),
                "v10": ("VGRD", "10 m above ground"), "t2m": ("TMP", "2 m above ground"), "msl": ("PRMSL", "mean sea level"),
                "tcc": ("TCDC", "entire atmosphere")}
NCEP_UPPER = {"u": "UGRD", "v": "VGRD", "q": "SPFH", "t": "TMP"}
ECMWF_SURFACE = {"tcwv": "tcwv", "u10": "10u", "v10": "10v", "t2m": "2t", "msl": "msl", "tcc": "tcc"}
#: Output steps of each model (hours) and how far it runs.
STEPS = {"gfs": (1, 120), "ifs": (3, 144), "aifs": (6, 240)}


def _index_ncep(text: str) -> list:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    offsets = [int(ln.split(":")[1]) for ln in lines]
    out = []
    for k, ln in enumerate(lines):
        p = ln.split(":")
        out.append({"var": p[3], "level": p[4], "desc": p[5] if len(p) > 5 else "", "start": offsets[k],
                    "end": offsets[k + 1] - 1 if k + 1 < len(lines) else None})
    return out


def _index_ecmwf(text: str) -> list:
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            row = json.loads(ln)
            out.append({"var": row["param"], "level": str(row.get("levelist", row.get("levtype", ""))),
                        "start": int(row["_offset"]), "end": int(row["_offset"]) + int(row["_length"]) - 1})
    return out


def decode(message: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One GRIB2 message -> ``(values (lat ascending, lon ascending -180..180), lat, lon)``."""
    from naturev1.swarm.models import decode_grib_native

    values, (lat0, dlat, lon0, dlon) = decode_grib_native(message)
    lat = lat0 - dlat * np.arange(values.shape[0])
    lon = (lon0 + dlon * np.arange(values.shape[1]) + 180.0) % 360.0 - 180.0
    order = np.argsort(lon)
    values, lon = values[:, order], lon[order]
    if lat[0] > lat[-1]:
        values, lat = values[::-1], lat[::-1]
    return values, lat, lon


def box_average(values: np.ndarray, lat: np.ndarray, lon: np.ndarray, dst_lat: np.ndarray, dst_lon: np.ndarray,
                half: float) -> np.ndarray:
    """Average a regular-grid field onto destination cell centres, each the box ``+-half`` degrees, by overlap
    (weights cos latitude); the same reduction the ERA5 stores used for their coarse grids."""
    d_lat, d_lon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])

    def overlap(src, dst, width):
        lo = np.maximum(dst[:, None] - half, src[None] - width / 2)
        hi = np.minimum(dst[:, None] + half, src[None] + width / 2)
        return np.clip(hi - lo, 0, None) / width

    rows = np.nonzero((lat >= dst_lat.min() - half - d_lat) & (lat <= dst_lat.max() + half + d_lat))[0]
    cols = np.nonzero((lon >= dst_lon.min() - half - d_lon) & (lon <= dst_lon.max() + half + d_lon))[0]
    sub = values[np.ix_(rows, cols)]
    wl = overlap(lat[rows], dst_lat, d_lat) * np.cos(np.radians(lat[rows]))[None]
    wo = overlap(lon[cols], dst_lon, d_lon)
    ok = np.isfinite(sub)
    num = wl @ np.where(ok, sub, 0.0) @ wo.T
    den = wl @ ok.astype(np.float64) @ wo.T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan).astype(np.float32)


class NWPReader:
    """One model's run read onto the training grids. ``frames(init_hour)`` returns what the dataset builds from ERA5:
    ``{"fields": {short: (F, H1, W1)}, "upper": {name: (F, H2, W2)}, "notes": [...]}``."""

    def __init__(self, model: str, lat1, lon1, lat2, lon2, cache="~/.cache/pluvium/nwp", schedule: Schedule = DEFAULT,
                 workers: int = 24) -> None:
        if model not in STEPS:
            raise ValueError(f"unknown model {model!r}; choose from {', '.join(STEPS)}")
        self.model, self.schedule, self.workers = model, schedule, workers
        self.lat1, self.lon1, self.lat2, self.lon2 = (np.asarray(x, np.float64) for x in (lat1, lon1, lat2, lon2))
        self.cache = Path(cache).expanduser()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.notes: list[str] = []

    # ------------------------------------------------------------------------------------- reading --
    def _urls(self, init: int, lead: int) -> list[str]:
        s = str(hour_to_datetime64(init))
        day, hour = s[:10].replace("-", ""), int(s[11:13])
        if self.model == "gfs":
            return [GFS.format(day=day, hour=hour, lead=lead)]
        name = "ifs" if self.model == "ifs" else "aifs-single"
        return [u.format(day=day, hour=hour, lead=lead, model=name) for u in (ECMWF, ECMWF_SITE)]

    def _index(self, init: int, lead: int):
        last = None
        for url in self._urls(init, lead):
            suffix = ".idx" if self.model == "gfs" else ".index"
            try:
                text = fetch(url + suffix, timeout=60.0, attempts=3).decode()
                return url, (_index_ncep(text) if self.model == "gfs" else _index_ecmwf(text))
            except Exception as error:
                last = error
        raise RuntimeError(f"{self.model} {str(hour_to_datetime64(init))[:13]}Z +{lead} h: no index ({last})")

    def available(self, init: int, lead: int | None = None) -> bool:
        lead = self.schedule.three_hourly_until if lead is None else lead
        step = STEPS[self.model][0]
        try:
            self._index(init, int(np.ceil(lead / step) * step))
            return True
        except Exception:
            return False

    def _message(self, url, index, var, level, desc_contains: str | None = None):
        for row in index:
            if row["var"] == var and row["level"] == level and (desc_contains is None or desc_contains in row["desc"]):
                return decode(fetch(url, row["start"], row["end"]))
        return None

    def _surface_at(self, init: int, lead: int) -> dict:
        url, index = self._index(init, lead)
        out = {}
        for short in ("tcwv", "u10", "v10", "t2m", "msl", "tcc"):
            if self.model == "gfs":
                var, level = NCEP_SURFACE[short]
                got = self._message(url, index, var, level)
            else:
                got = self._message(url, index, ECMWF_SURFACE[short], "sfc")
            if got is not None:
                out[short] = got
        # cumulative rain to this lead (mm)
        if lead > 0:
            if self.model == "gfs":
                want = f"0-{lead // 24} day acc fcst" if lead % 24 == 0 else f"0-{lead} hour acc fcst"
                got = self._message(url, index, "APCP", "surface", want)
                out["tp_cum"] = got
            else:
                got = self._message(url, index, "tp", "sfc")
                out["tp_cum"] = None if got is None else (got[0] * 1000.0, got[1], got[2])
        return out

    def _upper_at(self, init: int, lead: int) -> dict:
        url, index = self._index(init, lead)
        out = {}
        for short in UPPER_SHORT:
            for p in LEVELS:
                if self.model == "gfs":
                    got = self._message(url, index, NCEP_UPPER[short], f"{p} mb")
                else:
                    got = self._message(url, index, short, str(p))
                if got is not None:
                    out[f"{short}{p}"] = got
        return out

    # ------------------------------------------------------------------------------------- frames --
    def frames(self, init: int, offset: int = 0) -> dict:
        """The schedule's frames for a forecast issued ``offset`` hours after the run initialised at ``init``
        (integer hours), on the training grids. With ``offset`` >= 3 the history frames are the run's own first hours;
        with less they repeat its analysis."""
        stamp = str(hour_to_datetime64(init))[:13].replace(':', '')
        path = self.cache / f"{self.model}_{stamp}{f'+{offset}' if offset else ''}.npz"
        if path.exists():
            z = np.load(path, allow_pickle=False)
            return {"fields": {k[2:]: z[k] for k in z.files if k.startswith("f_")},
                    "upper": {k[2:]: z[k] for k in z.files if k.startswith("u_")},
                    "notes": json.loads(str(z["notes"]))}
        sch = self.schedule
        step, horizon = STEPS[self.model]
        need = offset + sch.three_hourly_until
        if need > horizon:
            raise RuntimeError(f"{self.model} runs to {horizon} h, the forecast needs {need} h")
        steps = np.arange(0, need + step, step)
        steps = steps[steps <= horizon]
        log(f"{self.model}: reading {steps.size} steps of the {str(hour_to_datetime64(init))[:13]}Z run")
        surface = dict(zip(steps.tolist(), parallel_map(lambda L: self._surface_at(init, int(L)), steps,
                                                         workers=self.workers)))
        up_steps = steps[(steps % max(3, step)) == 0]
        upper = dict(zip(up_steps.tolist(), parallel_map(lambda L: self._upper_at(init, int(L)), up_steps,
                                                          workers=self.workers)))
        notes = []

        def on1(item):
            values, lat, lon = item
            return box_average(values, lat, lon, self.lat1, self.lon1, 0.5)

        def on2(item):
            values, lat, lon = item
            return box_average(values, lat, lon, self.lat2, self.lon2, 0.75)

        # surface fields at every model step, then interpolated to the frame centres
        grid1 = {short: {} for short in HOURLY_SHORT}
        for L, got in surface.items():
            for short in ("tcwv", "u10", "v10", "t2m", "msl", "tcc"):
                if short in got:
                    grid1[short][L] = on1(got[short])
            grid1["tp"][L] = np.zeros_like(on1(next(iter(got.values())))) if L == 0 else (
                on1(got["tp_cum"]) if got.get("tp_cum") is not None else None)
        grid2 = {f"{s}{p}": {} for s in UPPER_SHORT for p in LEVELS}
        for L, got in upper.items():
            for name, item in got.items():
                grid2[name][L] = on2(item)

        def interp(series: dict, at: float):
            keys = sorted(k for k, v in series.items() if v is not None)
            if not keys:
                return None
            if at <= keys[0]:
                return series[keys[0]]
            if at >= keys[-1]:
                return series[keys[-1]]
            hi = int(np.searchsorted(keys, at))
            a, b = keys[hi - 1], keys[hi]
            w = (at - a) / (b - a)
            return series[a] * (1 - w) + series[b] * w

        centres = np.maximum(offset + sch.centres, 0)
        fields, uppers = {}, {}
        for short in HOURLY_SHORT:
            if short == "tp":
                continue
            frames = [interp(grid1[short], float(c)) for c in centres]
            if any(f is None for f in frames):
                frames = None
            fields[short] = frames
        # rain: cumulative interpolated in time (uniform rate inside a model step), differenced over each frame span
        starts = np.maximum(offset + sch.leads, 0)
        ends = starts + sch.spans
        cum0 = [interp(grid1["tp"], float(a)) for a in starts]
        cum1 = [interp(grid1["tp"], float(b)) for b in ends]
        if any(x is None for x in cum0 + cum1):
            fields["tp"] = None
        else:
            fields["tp"] = [np.clip((b - a) / s, 0, None) for a, b, s in zip(cum0, cum1, sch.spans)]
        for name in grid2:
            frames = [interp(grid2[name], float(c)) for c in centres]
            uppers[name] = None if any(f is None for f in frames) else frames
        # substitutes for what a model does not publish
        if fields.get("tcwv") is None and all(uppers.get(f"q{p}") is not None for p in (850, 700, 500)):
            fields["tcwv"] = [(22500.0 * a + 17500.0 * b + 25000.0 * c) / 9.81
                              for a, b, c in zip(uppers["q850"], uppers["q700"], uppers["q500"])]
            notes.append("column water vapour derived from 850/700/500 hPa humidity")
        if fields.get("tcc") is None:
            fields["tcc"] = [np.full((self.lat1.size, self.lon1.size), 0.5, np.float32) for _ in centres]
            notes.append("cloud cover not published: held at 0.5")
        missing = [k for k, v in list(fields.items()) + list(uppers.items()) if v is None]
        if missing:
            raise RuntimeError(f"{self.model}: fields missing from the run: {', '.join(missing)}")
        out = {"fields": {k: np.stack(v).astype(np.float32) for k, v in fields.items()},
               "upper": {k: np.stack(v).astype(np.float32) for k, v in uppers.items()}, "notes": notes}
        if out["fields"]["tcc"].max() > 1.5:
            out["fields"]["tcc"] = out["fields"]["tcc"] / 100.0
        np.savez_compressed(path, notes=json.dumps(notes), **{f"f_{k}": v for k, v in out["fields"].items()},
                            **{f"u_{k}": v for k, v in out["upper"].items()})
        return out


def latest_cycle(model: str, now_hour: int, reader: NWPReader, look_back: int = 30) -> int | None:
    """The newest cycle of ``model`` that has published far enough to cover a forecast issued at ``now_hour``."""
    step = 6 if model == "gfs" else 12 if model == "ifs" else 6
    start = now_hour - now_hour % step
    for back in range(0, look_back + 1, step):
        init = start - back
        if reader.available(init, now_hour - init + reader.schedule.three_hourly_until):
            return init
    return None


__all__ = ["ECMWF", "GFS", "NWPReader", "STEPS", "box_average", "decode", "latest_cycle"]
