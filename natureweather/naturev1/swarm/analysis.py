# Copyright 2026 Nathan. Apache-2.0.
"""
Real-time and recent analyses for the physics and the swarm: the GFS 0.25-degree analysis from NOAA's public AWS
bucket, cut into exactly the storm-centred box the ERA5 hindcasts used.

ERA5 arrives days late and WeatherBench 2's copy ends in 2023, so a live storm (or one since 2021) needs another
analysis. Only the GRIB messages the physics reads are fetched, with byte ranges from the ``.idx`` index (about
60 MB instead of 500): u, v, geopotential, temperature and specific humidity on the 13 levels, sea-level pressure,
skin temperature (as SST over water), precipitable water, 10 m winds, the land mask and the surface height -- and
the winds, 500 hPa height and pressure of the analysis 12 hours earlier, for the tendencies. Each field is averaged
conservatively onto the 1.5-degree grid of the training data (the same grid the shipped coefficients were fitted on).

GRIB decoding needs ``eccodes`` (``pip install eccodes``; the wheel carries the library).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from ..cyclone.data import (HALF_LAT, HALF_LON, HISTORY_HOURS, LEADS, LEVELS, PREVIOUS_LEVELS, STEP, Case,
                            _box_indices)
from ..cyclone.fields import Environment

GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{day}/{hour:02d}/atmos/gfs.t{hour:02d}z.pgrb2.0p25.f000"
G = 9.80665
LAT_AXIS = np.arange(-90.0, 90.0 + 1e-9, STEP)
LON_AXIS = np.arange(0.0, 360.0 - 1e-9, STEP)
_LEVEL_VARS = {"u": "UGRD", "v": "VGRD", "z": "HGT", "t": "TMP", "q": "SPFH"}
_SURFACE = {"msl": ("PRMSL", "mean sea level"), "skin": ("TMP", "surface"),
            "tcwv": ("PWAT", "entire atmosphere (considered as a single layer)"),
            "u10": ("UGRD", "10 m above ground"), "v10": ("VGRD", "10 m above ground"),
            "land": ("LAND", "surface"), "orog": ("HGT", "surface")}


def _get(url: str, start: int | None = None, end: int | None = None, timeout: float = 120.0) -> bytes:
    request = urllib.request.Request(url)
    if start is not None:
        request.add_header("Range", f"bytes={start}-{'' if end is None else end}")
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as error:
            last = error
            import time

            time.sleep(2 ** attempt)
    raise RuntimeError(f"could not fetch {url}: {last}")


def _coarsen(field: np.ndarray) -> np.ndarray:
    """0.25-degree global (721 x 1440, north to south, 0..359.75 E) to the 1.5-degree nodes (121 x 240, south to
    north, 0..358.5 E): each node the area-weighted mean of the 0.25-degree cells within 0.75 degrees of it."""
    f = np.asarray(field, np.float64)[::-1]                                   # south to north
    w1 = np.array([0.5, 1, 1, 1, 1, 1, 0.5])
    lat_q = np.arange(-90.0, 90.0 + 1e-9, 0.25)
    cos = np.cos(np.radians(lat_q))
    cols = np.concatenate([f[:, -3:], f, f[:, :3]], 1)                       # wrap in longitude
    out = np.zeros((LAT_AXIS.size, LON_AXIS.size))
    for k, lat in enumerate(LAT_AXIS):
        i = int(round((lat + 90.0) / 0.25))
        rows = np.arange(i - 3, i + 4)
        ok = (rows >= 0) & (rows < f.shape[0])
        wr = w1[ok] * np.maximum(cos[rows[ok]], 1e-3)
        block = cols[rows[ok]]                                                # (r, 1446)
        lat_mean = (block * wr[:, None]).sum(0) / wr.sum()
        idx = np.arange(LON_AXIS.size) * 6 + 3                                # node j -> padded column 6j + 3
        out[k] = sum(w1[d] * lat_mean[idx + d - 3] for d in range(7)) / w1.sum()
    return out


class GFSAnalysis:
    """GFS analyses on the 1.5-degree grid, cached as ``.npz`` under ``cache``."""

    def __init__(self, cache: str | Path = "~/.cache/naturev1/gfs") -> None:
        self.cache = Path(cache).expanduser()
        self.cache.mkdir(parents=True, exist_ok=True)

    def fields(self, when) -> dict:
        when = np.datetime64(when, "h")
        stamp = str(when).replace("-", "").replace("T", "")[:10]
        path = self.cache / f"gfs_{stamp}.npz"
        if path.exists():
            return dict(np.load(path))
        import eccodes

        day, hour = stamp[:8], int(stamp[8:10])
        base = GFS.format(day=day, hour=hour)
        lines = _get(base + ".idx").decode().splitlines()
        offsets = [int(line.split(":")[1]) for line in lines]
        index = {}
        for k, line in enumerate(lines):
            parts = line.split(":")
            end = offsets[k + 1] - 1 if k + 1 < len(lines) else None
            index[(parts[3], parts[4])] = (offsets[k], end)

        def read(var, level):
            start, end = index[(var, level)]
            gid = eccodes.codes_new_from_message(_get(base, start, end))
            try:
                values = eccodes.codes_get_values(gid).reshape(eccodes.codes_get(gid, "Nj"), eccodes.codes_get(gid, "Ni"))
            finally:
                eccodes.codes_release(gid)
            return _coarsen(values)

        out = {}
        for key, var in _LEVEL_VARS.items():
            out[key] = np.stack([read(var, f"{p} mb") for p in LEVELS])
        for key, (var, level) in _SURFACE.items():
            out[key] = read(var, level)
        out["z"] = out["z"] * G
        out["lsm"] = np.clip(out.pop("land"), 0.0, 1.0)
        out["zs"] = out.pop("orog") * G
        skin = out.pop("skin")
        out["sst"] = np.where(out["lsm"] < 0.5, skin, np.nan)
        np.savez_compressed(path, **out)
        return out

    def environment(self, when, lat0: float, lon0: float) -> Environment:
        """The storm-centred box at ``when`` (00/06/12/18 UTC), with the analysis 12 hours earlier as ``previous``."""
        when = np.datetime64(when, "h")
        now = self.fields(when)
        try:
            before = self.fields(when - np.timedelta64(12, "h"))
        except Exception:
            before = now
        rows, cols = int(round(2 * HALF_LAT / STEP)) + 1, int(round(2 * HALF_LON / STEP)) + 1
        rs, cs, box_lat, box_lon = _box_indices(lat0, lon0, LAT_AXIS, STEP, rows, cols)
        cut = lambda a: a[..., rs, :][..., cs]
        fields = {k: cut(now[k]) for k in ("u", "v", "z", "t", "q", "msl", "sst", "tcwv", "u10", "v10", "lsm", "zs")}
        keep = [LEVELS.index(p) for p in PREVIOUS_LEVELS]
        previous = {"u": cut(before["u"][keep]), "v": cut(before["v"][keep]),
                    "z500": cut(before["z"][LEVELS.index(500)]), "msl": cut(before["msl"])}
        env = Environment(lat=box_lat.astype(np.float64), lon=box_lon.astype(np.float64), levels=LEVELS,
                          fields={k: v.astype(np.float64) for k, v in fields.items()},
                          previous={k: v.astype(np.float64) for k, v in previous.items()}, time=np.datetime64(when, "s"))
        env.previous_levels = PREVIOUS_LEVELS
        return env


def case_from_bdeck(best: list, when, storm_id: str = "", name: str = "", basin: str = "NA") -> Case:
    """A :class:`naturev1.cyclone.data.Case` at ``when`` from b-deck fixes (:func:`naturev1.swarm.fixes.parse_bdeck`):
    the last 48 hours as history and, when the deck runs past ``when``, the truth at the standard leads."""
    when = np.datetime64(when, "m")
    times = np.array([f.time for f in best], dtype="datetime64[m]")
    lat = np.array([f.lat for f in best])
    lon = np.array([f.lon for f in best])
    wind = np.array([f.wind_kt for f in best])

    def at(hours):
        t = when + np.timedelta64(int(round(hours * 60)), "m")
        k = np.nonzero(times == t)[0]
        return (lat[k[0]], lon[k[0]], wind[k[0]]) if k.size else (np.nan, np.nan, np.nan)

    now = at(0)
    if not np.isfinite(now[0]):
        raise ValueError(f"no best-track fix at {when}")
    hist = {"lat": [], "lon": [], "wind": [], "pressure": []}
    for h in HISTORY_HOURS:
        la, lo, wi = at(h)
        hist["lat"].append(la)
        hist["lon"].append(lo)
        hist["wind"].append(wi)
        hist["pressure"].append(np.nan)
    truth = {"lat": [], "lon": [], "wind": [], "pressure": [], "valid": []}
    for h in LEADS:
        la, lo, wi = at(h)
        truth["lat"].append(la)
        truth["lon"].append(lo)
        truth["wind"].append(wi)
        truth["pressure"].append(np.nan)
        truth["valid"].append(bool(np.isfinite(la)))
    return Case(storm_id, name, basin, np.datetime64(when, "s"), float(now[0]), float(now[1]), float(now[2]), float("nan"),
                "HU" if now[2] >= 64 else "TS", {k: np.asarray(v, float) for k, v in hist.items()},
                truth={k: np.asarray(v, bool if k == "valid" else float) for k, v in truth.items()})


__all__ = ["GFSAnalysis", "case_from_bdeck"]
