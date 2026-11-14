# Copyright 2026 Nathan. Apache-2.0.
"""
Sea-level pressure from every operational system, cut to an ocean basin, for tracking storms.

A tropical cyclone is, first of all, a deep closed low. Its centre in a forecast is the pressure minimum, and that is
all a tracker needs from each run: mean sea-level pressure every 12 hours out to five days. This reader fetches only
those messages (one byte range each) from the public archives, cuts them to the basin and caches one small file per
system and run:

* ECMWF ENS (the control and the first 25 perturbed members by default), ECMWF HRES and ECMWF AIFS, 0.25 degrees;
* NOAA GEFS (31 members, 0.5 degrees), NOAA GFS and NOAA GraphCast (0.25 degrees).

Values are hPa. Files store ``hPa - 1000`` as 16-bit floats (about 0.05 hPa of resolution near 1000 hPa).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..sources.baselines import ECMWF, ECMWF_ARCHIVES, GEFS, GRAPHCAST, _ecmwf_rows, _first, _stamp
from ..sources.nwp import GFS, _index_ncep, decode
from ..util import fetch, hour_to_datetime64, log, parallel_map
from .basins import BASINS

STEPS = tuple(range(12, 121, 12))
SYSTEMS = ("ens", "gefs", "ifs", "aifs", "gfs", "graphcast")
LABELS = {"ens": "ECMWF ENS", "gefs": "NOAA GEFS", "ifs": "ECMWF HRES", "aifs": "ECMWF AIFS", "gfs": "NOAA GFS",
          "graphcast": "NOAA GraphCast"}
ENS_MEMBERS = 26


class FieldArchive:
    def __init__(self, cache, basin: str = "natl", workers: int = 12, ens_members: int = ENS_MEMBERS,
                 steps=STEPS) -> None:
        self.cache = Path(cache).expanduser()
        self.basin, self.box = basin, BASINS[basin].box
        self.workers, self.ens_members, self.steps = workers, ens_members, tuple(steps)

    # ------------------------------------------------------------------------------------ files --
    def path(self, system: str, init: int) -> Path:
        day, hour = _stamp(init)
        return self.cache / self.basin / system / f"{day}{hour:02d}.npz"

    def load(self, system: str, init: int) -> dict | None:
        p = self.path(system, init)
        if not p.exists():
            return None
        z = np.load(p)
        return {"msl": z["values"].astype(np.float32) + 1000.0, "lat": z["lat"], "lon": z["lon"], "steps": z["steps"]}

    def _save(self, system: str, init: int, values: np.ndarray, lat, lon) -> None:
        p = self.path(system, init)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, values=(values - 1000.0).astype(np.float16), lat=lat, lon=lon,
                            steps=np.asarray(self.steps))
        tmp.replace(p)

    # ---------------------------------------------------------------------------------- reading --
    def read(self, system: str, init: int) -> bool:
        """Fetch one system's run (skipped when cached); True when it is in the cache."""
        if self.path(system, init).exists():
            return True
        try:
            values, lat, lon = getattr(self, "_" + system)(int(init))
        except Exception as error:
            log(f"{system} {str(hour_to_datetime64(int(init)))[:13]}Z: no pressure fields ({type(error).__name__}: "
                f"{str(error)[:120]})")
            return False
        if np.isfinite(values).mean() < 0.5:
            log(f"{system} {str(hour_to_datetime64(int(init)))[:13]}Z: more than half the fields are missing; not kept")
            return False
        self._save(system, init, values, lat, lon)
        return True

    def _cut(self, blob: bytes):
        values, lat, lon = decode(blob)
        s, n, w, e = self.box
        i = (lat >= s) & (lat <= n)
        j = (lon >= w) & (lon <= e)
        out = values[np.ix_(i, j)].astype(np.float32)
        if np.nanmax(out) > 2000.0:                                # Pa -> hPa
            out = out / 100.0
        return out, lat[i], lon[j]

    def _assemble(self, pieces, members: int):
        pieces = [p for p in pieces if p is not None]
        if not pieces:
            raise RuntimeError("nothing was read")
        _, _, (first, lat, lon) = pieces[0]
        out = np.full((members, len(self.steps), lat.size, lon.size), np.nan, np.float32)
        for m, k, (values, _, _) in pieces:
            if values.shape == first.shape:
                out[m, k] = values
        return out, lat, lon

    # ------------------------------------------------------------------------------------ ECMWF --
    def _ecmwf(self, init: int, layouts, max_members: int):
        day, hour = _stamp(init)
        jobs = []
        for k, step in enumerate(self.steps):
            for path, kind in layouts:
                urls = [ECMWF.format(base=b, day=day, hour=hour, path=path, step=step, kind=kind) for b in ECMWF_ARCHIVES]
                try:
                    rows = _ecmwf_rows(_first(urls, ".index", trust_missing=True).decode(), param="msl")
                except Exception:
                    rows = []
                if rows:
                    jobs += [(m, k, urls, a, b) for m, a, b in rows if m < max_members]
                    break
        if not jobs:
            raise RuntimeError("no pressure messages in the archive")

        def one(job):
            m, k, urls, a, b = job
            try:
                return m, k, self._cut(_first(urls, ".grib2", a, b))
            except Exception:
                return None

        return self._assemble(parallel_map(one, jobs, workers=self.workers), max(j[0] for j in jobs) + 1)

    def _ens(self, init):
        return self._ecmwf(init, [("ifs/0p25/enfo", "enfo-ef")], self.ens_members)

    def _ifs(self, init):
        return self._ecmwf(init, [("ifs/0p25/oper", "oper-fc")], 1)

    def _aifs(self, init):
        return self._ecmwf(init, [("aifs-single/0p25/oper", "oper-fc"), ("aifs/0p25/oper", "oper-fc")], 1)

    # -------------------------------------------------------------------------------------- NOAA --
    def _ncep(self, init: int, urls_of_member: list):
        """``urls_of_member``: per member, a function step -> URL of the file holding that step."""

        def one(job):
            m, k, url = job
            try:
                rows = [r for r in _index_ncep(fetch(url + ".idx", timeout=60.0, attempts=4).decode())
                        if r["var"] == "PRMSL" and r["level"] == "mean sea level"]
                return m, k, self._cut(fetch(url, rows[0]["start"], rows[0]["end"], attempts=4))
            except Exception:
                return None

        jobs = [(m, k, make(step)) for m, make in enumerate(urls_of_member) for k, step in enumerate(self.steps)]
        return self._assemble(parallel_map(one, jobs, workers=self.workers), len(urls_of_member))

    def _gfs(self, init):
        day, hour = _stamp(init)
        return self._ncep(init, [lambda step: GFS.format(day=day, hour=hour, lead=step)])

    def _graphcast(self, init):
        day, hour = _stamp(init)
        return self._ncep(init, [lambda step: GRAPHCAST.format(day=day, hour=hour, step=step)])

    def _gefs(self, init):
        day, hour = _stamp(init)
        names = ["gec00"] + [f"gep{k:02d}" for k in range(1, 31)]
        return self._ncep(init, [lambda step, name=name: GEFS.format(day=day, hour=hour, member=name, step=step)
                                 for name in names])


def stage_fields(cache, inits, systems=SYSTEMS, basin: str = "natl", workers: int = 12, parallel_runs: int = 2) -> dict:
    """Fetch pressure fields for every start time and system; returns how many runs each system has."""
    archive = FieldArchive(cache, basin, workers=workers)
    counts = {}
    for system in systems:
        ok = parallel_map(lambda init, system=system: archive.read(system, int(init)), list(inits),
                          workers=parallel_runs, progress=f"{system} pressure runs", every=120.0)
        counts[system] = int(sum(ok))
        log(f"cyclone fields: {system}: {counts[system]} of {len(inits)} runs")
    return counts


__all__ = ["ENS_MEMBERS", "LABELS", "STEPS", "SYSTEMS", "FieldArchive", "stage_fields"]
