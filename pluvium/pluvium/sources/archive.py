# Copyright 2026 Nathan. Apache-2.0.
"""
Every operational system's rain, read once per run and cut to many regions at the same time.

A GRIB message holds the whole globe, so reading one region costs as much as reading ten. This reader decodes each
rain message once, cuts it to every region in :data:`REGIONS`, and caches one small file per region and run. That is
how Pluvium's blend can be trained and scored in the tropics, the mid-latitudes and on complex terrain without
downloading the archive again for each place.

Systems and how far they reach (00 UTC runs):

* ECMWF ENS (51 members), ECMWF HRES and ECMWF AIFS: 0.25 degrees, out to 14 days. Before 25 February 2025 AIFS
  was published as an experiment (``aifs``); after, as AIFS Single (``aifs-single``). Both are read.
* NOAA GEFS (31 members), 0.5 degrees, from 6-hour buckets. Beyond day 3 only the control and the first 10 members are
  read (``ext_members``): 31 members x 56 buckets per run is more than the archive serves in reasonable time.
* NOAA GraphCast (GFS-initialised), 0.25 degrees, out to 10 days.
* NOAA GFS, 0.5 degrees, out to 14 days.

Values are rain accumulated from the start of the run to each 24-hour step, in mm. Units are checked per message from
the global maximum: ECMWF publishes metres for IFS and kg m-2 for AIFS, and a 24-hour global maximum below 5 can only
be metres.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..util import hour_to_datetime64, log, parallel_map
from .baselines import ECMWF, ECMWF_ARCHIVES, GEFS, _ecmwf_rows, _first, _stamp, graphcast_template
from .nwp import GFS05, _index_ncep, decode

#: name -> (south, north, west, east): tropical islands, tropical mountains, mid-latitude mountains, coasts and plains,
#: in both hemispheres. CMORPH, the truth, covers 60 S to 60 N, so polar regions cannot be scored.
REGIONS = {
    "caribbean": (8.0, 29.0, -91.0, -57.0),
    "maritime": (-9.0, 7.0, 95.0, 120.0),
    "alps": (43.5, 48.5, 4.5, 16.5),
    "pacific_nw": (40.5, 50.0, -125.5, -115.5),
    "great_plains": (33.0, 43.0, -103.0, -91.0),
    "andes_south": (-50.0, -33.0, -76.0, -68.0),
}
DAYS = 14
LIMIT = {"graphcast": 10}                     # days a system's runs reach in the archive
ARCHIVE_SYSTEMS = ("ens", "gefs", "ifs", "aifs", "graphcast", "gfs")


class ArchiveReader:
    def __init__(self, cache, regions: dict | None = None, workers: int = 8, pad: float = 0.75) -> None:
        self.cache = Path(cache).expanduser()
        self.regions = dict(regions or REGIONS)
        self.workers, self.pad = workers, pad

    # ------------------------------------------------------------------------------------ files --
    def path(self, region: str, system: str, init: int) -> Path:
        day, hour = _stamp(init)
        return self.cache / region / system / f"{day}{hour:02d}.npz"

    def days_cached(self, system: str, init: int) -> int:
        """How many days of this run every region has (0 when any region is missing)."""
        got = []
        for region in self.regions:
            p = self.path(region, system, init)
            if not p.exists():
                return 0
            with np.load(p) as z:
                got.append(int(z["steps"].size))
        return min(got) if got else 0

    def load(self, region: str, system: str, init: int) -> dict | None:
        p = self.path(region, system, init)
        if not p.exists():
            return None
        z = np.load(p)
        return {"values": z["values"].astype(np.float32), "lat": z["lat"], "lon": z["lon"], "steps": z["steps"]}

    def _save(self, system: str, init: int, cut: dict, steps) -> None:
        for region, (values, lat, lon) in cut.items():
            p = self.path(region, system, init)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, values=values.astype(np.float16), lat=lat, lon=lon, steps=np.asarray(steps))
            tmp.replace(p)

    # ----------------------------------------------------------------------------------- reading --
    def read(self, system: str, init: int, days: int = 3, ext_members: int = 11) -> bool:
        """Fetch ``days`` days of a run for every region (skipped when cached); True when the run is cached."""
        days = min(days, LIMIT.get(system, DAYS))
        if self.days_cached(system, init) >= days:
            return True
        steps = [24 * d for d in range(1, days + 1)]
        try:
            cut = getattr(self, "_" + system)(init, steps, ext_members)
        except Exception as error:
            log(f"{system} {str(hour_to_datetime64(init))[:13]}Z: not read ({type(error).__name__}: {str(error)[:160]})")
            return False
        self._save(system, init, cut, steps)
        return True

    def _cut(self, item) -> dict:
        values, lat, lon = item
        out = {}
        for region, (s, n, w, e) in self.regions.items():
            i = (lat >= s - self.pad) & (lat <= n + self.pad)
            j = (lon >= w - self.pad) & (lon <= e + self.pad)
            out[region] = (values[np.ix_(i, j)].astype(np.float32), lat[i], lon[j])
        return out

    def _decode_cut(self, blob: bytes, metres_possible: bool) -> dict:
        values, lat, lon = decode(blob)
        if metres_possible and np.nanmax(values) < 5.0:          # metres (IFS); AIFS is already kg m-2 = mm
            values = values * 1000.0
        return self._cut((values, lat, lon))

    def _assemble(self, pieces, members: int, steps) -> dict:
        """``pieces``: (member, step index, cut) -> per region ``(members, steps, H, W)`` with NaN where missing."""
        pieces = [p for p in pieces if p is not None]
        if not pieces:
            raise RuntimeError("nothing was read")
        out = {}
        for region in self.regions:
            _, _, first = pieces[0]
            _, lat, lon = first[region]
            arr = np.full((members, len(steps), lat.size, lon.size), np.nan, np.float32)
            for m, k, cut in pieces:
                arr[m, k] = cut[region][0]
            out[region] = (arr, lat, lon)
        return out

    # ------------------------------------------------------------------------------------ ECMWF --
    def _ecmwf_run(self, init: int, steps, layouts) -> dict:
        """``layouts``: (path, kind) pairs tried in turn for each step (AIFS changed path in February 2025)."""
        day, hour = _stamp(init)
        jobs, members = [], set()
        for k, step in enumerate(steps):
            for path, kind in layouts:
                urls = [ECMWF.format(base=b, day=day, hour=hour, path=path, step=step, kind=kind) for b in ECMWF_ARCHIVES]
                try:
                    rows = _ecmwf_rows(_first(urls, ".index", trust_missing=True).decode())
                except Exception:
                    rows = []
                if rows:
                    jobs += [(m, k, urls, a, b) for m, a, b in rows]
                    members |= {m for m, _, _ in rows}
                    break
        if not jobs:
            raise RuntimeError("no rain messages in the archive")
        slot = {m: i for i, m in enumerate(sorted(members))}

        def one(job):
            m, k, urls, a, b = job
            try:
                return slot[m], k, self._decode_cut(_first(urls, ".grib2", a, b), True)
            except Exception as error:
                log(f"  message lost ({type(error).__name__}); left as missing")
                return None

        return self._assemble(parallel_map(one, jobs, workers=self.workers), len(slot), steps)

    def _ens(self, init, steps, ext_members):
        return self._ecmwf_run(init, steps, [("ifs/0p25/enfo", "enfo-ef")])

    def _ifs(self, init, steps, ext_members):
        return self._ecmwf_run(init, steps, [("ifs/0p25/oper", "oper-fc")])

    def _aifs(self, init, steps, ext_members):
        return self._ecmwf_run(init, steps, [("aifs-single/0p25/oper", "oper-fc"), ("aifs/0p25/oper", "oper-fc")])

    # -------------------------------------------------------------------------------------- NOAA --
    def _gefs(self, init, steps, ext_members):
        """Six-hour buckets summed; all 31 members to day 3, the control and ``ext_members - 1`` perturbed beyond."""
        day, hour = _stamp(init)
        names = ["gec00"] + [f"gep{k:02d}" for k in range(1, 31)]
        last = max(steps)
        jobs = [(m, name, b) for m, name in enumerate(names) for b in range(6, last + 1, 6)
                if b <= 72 or m < ext_members]

        def one(job):
            m, name, b = job
            from ..util import fetch

            url = GEFS.format(day=day, hour=hour, member=name, step=b)
            try:
                rows = [r for r in _index_ncep(fetch(url + ".idx", timeout=60.0, attempts=6).decode())
                        if r["var"] == "APCP" and r["level"] == "surface" and f"{b - 6}-{b} hour acc" in r["desc"]]
                return m, b, self._decode_cut(fetch(url, rows[0]["start"], rows[0]["end"], attempts=6), False)
            except Exception as error:
                log(f"  GEFS {name} +{b} h lost ({type(error).__name__}); left as missing")
                return None

        got = [g for g in parallel_map(one, jobs, workers=self.workers * 2) if g is not None]
        if not got:
            raise RuntimeError("nothing was read")
        out = {}
        for region in self.regions:
            _, lat, lon = got[0][2][region]
            buckets = np.full((31, last // 6, lat.size, lon.size), np.nan, np.float32)
            for m, b, cut in got:
                buckets[m, b // 6 - 1] = cut[region][0]
            cum = np.cumsum(buckets, axis=1)                    # NaN propagates: a lost bucket spoils what follows
            out[region] = (cum[:, [s // 6 - 1 for s in steps]], lat, lon)
        return out

    def _ncep_cumulative(self, init, steps, template) -> dict:
        from ..util import fetch

        day, hour = _stamp(init)

        def one(item):
            k, step = item
            url = template.format(day=day, hour=hour, step=step, lead=step)
            want = f"0-{step // 24} day acc"
            try:
                rows = [r for r in _index_ncep(fetch(url + ".idx", timeout=60.0, attempts=6).decode())
                        if r["var"] == "APCP" and r["level"] == "surface" and want in r["desc"]]
                return 0, k, self._decode_cut(fetch(url, rows[0]["start"], rows[0]["end"], attempts=6), False)
            except Exception as error:
                log(f"  +{step} h lost ({type(error).__name__}); left as missing")
                return None

        return self._assemble(parallel_map(one, list(enumerate(steps)), workers=self.workers), 1, steps)

    def _graphcast(self, init, steps, ext_members):
        return self._ncep_cumulative(init, steps, graphcast_template(init))

    def _gfs(self, init, steps, ext_members):
        return self._ncep_cumulative(init, steps, GFS05)


def run_set(start: str, end: str, every: int = 2, long_every: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """00 UTC runs from ``start`` to ``end`` every ``every`` days, and the subset read to 14 days (every
    ``long_every`` days)."""
    from ..util import hour_of

    a = hour_of(start)
    a += (-a) % 24
    runs = np.arange(a, hour_of(end), 24 * every, dtype=np.int64)
    long = runs[((runs - runs[0]) // 24) % long_every == 0]
    return runs, long


def stage_archive(cache, runs, long_runs, systems=ARCHIVE_SYSTEMS, regions: dict | None = None, workers: int = 8,
                  parallel_runs: int = 2) -> dict:
    """Read every system's runs (14 days for ``long_runs``, 3 otherwise) for every region; returns counts per system."""
    reader = ArchiveReader(cache, regions, workers=workers)
    longs = set(int(r) for r in long_runs)
    counts = {}
    for system in systems:
        def one(init, system=system):
            return reader.read(system, int(init), DAYS if int(init) in longs else 3)

        ok = parallel_map(one, runs, workers=parallel_runs, progress=f"{system} runs", every=120.0)
        counts[system] = int(sum(ok))
        log(f"archive: {system}: {counts[system]} of {len(runs)} runs")
    return counts


__all__ = ["ARCHIVE_SYSTEMS", "DAYS", "REGIONS", "ArchiveReader", "run_set", "stage_archive"]
