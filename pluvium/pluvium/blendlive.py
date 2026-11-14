# Copyright 2026 Nathan. Apache-2.0.
"""
Pluvium Blend for any country, every day.

``train`` learns the blend for an area (:class:`pluvium.area.Area`) from a year of the six operational systems'
archived runs and CMORPH: about two hours of downloads, minutes of CPU, no GPU. ``forecast`` then reads the newest run
of every system and writes, for every town, the chance of at least 1, 10, 25 and 50 mm and the likely and heavy
amounts on each of the next 14 days.

A system whose newest run is not out yet (ECMWF ENS appears about 7 hours after its start time) is read from its
previous run; the blend knows how old each input is only through the lead, so older runs are used at the lead
matching the forecast day. A system that is missing altogether is left out: the blend was trained with each system's
``present`` flag and still forecasts from the others.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import blend as B
from .sources.archive import ARCHIVE_SYSTEMS, DAYS, LIMIT, ArchiveReader, run_set, stage_archive
from .util import hour_of, hour_to_datetime64, log

#: hours after its start time that each system's run is complete in the archives (a little margin included)
LATENCY = {"ens": 8, "gefs": 6, "ifs": 7, "aifs": 7, "graphcast": 6, "gfs": 5}


class BlendModel:
    def __init__(self, net: B.BlendNet, systems: list, area_json: dict, clim: dict, grid: tuple, meta: dict) -> None:
        self.net, self.systems, self.area_json, self.clim, self.grid, self.meta = net, systems, area_json, clim, grid, meta

    # ------------------------------------------------------------------------------------- train --
    @classmethod
    def train(cls, area, root, start: str = "2024-04-01T00", end: str = "2025-04-01T00", workers: int = 8,
              max_cells: int = 1500, max_rows: int = 400_000) -> BlendModel:
        from .benchmarks.fullyear import CellSet, build_table, land_mask, rows_for
        from .sources.cmorph_daily import DailyTruth, stage_daily

        root = Path(root).expanduser()
        region = {area.name: tuple(area.target_box)}
        runs, long = run_set(start, end, every=2, long_every=4)
        log(f"blend for {area.name}: reading {len(runs)} runs of {len(ARCHIVE_SYSTEMS)} systems ...")
        stage_archive(root / "archive", runs, long, regions=region, workers=workers)
        a = hour_of(start)
        stage_daily(root / "truth", region, start, str(hour_to_datetime64(hour_of(end) + 24 * (DAYS + 1)))[:13], workers=4)
        truth = DailyTruth(root / "truth")
        lat, lon = truth.grid(area.name)
        reader = ArchiveReader(root / "archive", region)
        mask = land_mask(reader, area.name, lat, lon, root / "masks")
        ii, jj = np.nonzero(mask)
        if ii.size > max_cells:
            keep = np.linspace(0, ii.size - 1, max_cells).astype(int)
            ii, jj = ii[keep], jj[keep]
        cells = CellSet(lat[ii], lon[jj], lambda h: truth.days(area.name, [h])[0][ii, jj], area.name)
        cells.climatology(np.arange(a, hour_of(end), 24))
        table = build_table(cells, reader, area.name, runs, set(int(r) for r in long))
        x, y, _ = rows_for(table, np.arange(len(table["run"])), list(ARCHIVE_SYSTEMS), len(cells), max_rows=max_rows)
        net = B.fit(x, y, max_rows=max_rows)
        # the climatology on the whole grid (every cell, not only the scored ones), by month
        full = CellSet(np.repeat(lat, lon.size), np.tile(lon, lat.size),
                       lambda h: truth.days(area.name, [h])[0].ravel(), area.name)
        full.climatology(np.arange(a, hour_of(end), 24))
        clim = {m: {k: v.reshape(lat.size, lon.size) for k, v in full.clim[m].items()} for m in full.clim}
        model = cls(net, list(ARCHIVE_SYSTEMS), area.to_json(), clim, (lat, lon),
                    {"trained": [start, end], "runs": int(len(runs)), "rows": int(x.shape[0]), "cells": int(len(cells))})
        model.save(root / "blend")
        return model

    # ---------------------------------------------------------------------------------------- io --
    def save(self, directory) -> Path:
        d = Path(directory).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict(), "n_in": self.net.mu.numel()}, d / "blend.pt")
        np.savez_compressed(d / "climatology.npz", lat=self.grid[0], lon=self.grid[1],
                            **{f"{k}_{m:02d}": v for m, entry in self.clim.items() for k, v in entry.items()})
        (d / "blend.json").write_text(json.dumps({"systems": self.systems, "area": self.area_json, "meta": self.meta,
                                                  "thresholds": B.THRESHOLDS.tolist()}, indent=1))
        return d

    @classmethod
    def load(cls, directory) -> BlendModel:
        d = Path(directory).expanduser()
        meta = json.loads((d / "blend.json").read_text())
        state = torch.load(d / "blend.pt", map_location="cpu", weights_only=False)
        net = B.BlendNet(int(state["n_in"]))
        net.load_state_dict(state["state"])
        net.eval()
        z = np.load(d / "climatology.npz")
        clim = {}
        for key in z.files:
            if key in ("lat", "lon"):
                continue
            name, m = key.rsplit("_", 1)
            clim.setdefault(int(m), {})[name] = z[key]
        return cls(net, meta["systems"], meta["area"], clim, (z["lat"], z["lon"]), meta["meta"])

    # ---------------------------------------------------------------------------------- forecast --
    def _clim_at(self, lat: np.ndarray, lon: np.ndarray, month: int) -> dict:
        glat, glon = self.grid
        i = np.clip(np.round((lat - glat[0]) / (glat[1] - glat[0])).astype(int), 0, glat.size - 1)
        j = np.clip(np.round((lon - glon[0]) / (glon[1] - glon[0])).astype(int), 0, glon.size - 1)
        return {k: v[i, j] for k, v in self.clim[month].items()}

    def forecast(self, lat, lon, cache, now: int | None = None, days: int = DAYS) -> dict:
        """Daily chances and amounts at points for days 1..``days`` from 00 UTC of the day of ``now`` (UTC hour)."""
        from .benchmarks.fullyear import CellSet, system_at_cells

        lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        now = int(datetime.now(timezone.utc).timestamp() // 3600) if now is None else int(now)
        day0 = now - now % 24
        s, n, w, e = self.area_json["target_box"]
        reader = ArchiveReader(cache, {self.area_json["name"]: (s, n, w, e)})
        cells = CellSet(lat, lon, None, self.area_json["name"])
        runs = {}
        for system in self.systems:
            for back in range(0, 4):                                    # newest complete 00 UTC run
                init = day0 - 24 * back
                if now - init < LATENCY.get(system, 8):
                    continue
                if reader.read(system, init, min(DAYS, LIMIT.get(system, DAYS))):
                    runs[system] = (init, reader.load(self.area_json["name"], system, init))
                    break
        if not runs:
            raise RuntimeError("no operational run could be read")
        out = {"day0": str(hour_to_datetime64(day0))[:10], "runs": {k: str(hour_to_datetime64(v[0]))[:13] for k, v in runs.items()},
               "days": []}
        for d in range(1, days + 1):
            valid = day0 + 24 * (d - 1)
            per = {}
            for system, (init, z) in runs.items():
                lead = (valid - init) // 24 + 1
                if lead <= LIMIT.get(system, DAYS):
                    f = system_at_cells(z, int(lead), cells)
                    if f is not None:
                        f.pop("_members", None)
                        per[system] = f
            clim = self._clim_at(lat, lon, int(str(hour_to_datetime64(valid))[5:7]))
            x = B.assemble(per, self.systems, {"ens", "gefs"}, clim, float(d), lat.size)
            p = B.predict(self.net, x)
            q = B.quantiles(p)
            out["days"].append({"date": str(hour_to_datetime64(valid))[:10], "systems": sorted(per),
                                **{f"p_ge_{int(t)}": B.exceed(p, t) for t in B.SCORED},
                                "median_mm": q[:, 49], "p90_mm": q[:, 89], "mean_mm": q.mean(1)})
        return out


def write_table(result: dict, names, path) -> Path:
    """One row per place and day: chances of 1/10/25/50 mm, median, 90th percentile and expected rain."""
    import csv

    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["place", "date", "p_ge_1mm", "p_ge_10mm", "p_ge_25mm", "p_ge_50mm", "median_mm", "p90_mm", "mean_mm",
                    "systems"])
        for d in result["days"]:
            for k, name in enumerate(names):
                w.writerow([name, d["date"], *(f"{d[f'p_ge_{int(t)}'][k]:.2f}" for t in B.SCORED),
                            f"{d['median_mm'][k]:.1f}", f"{d['p90_mm'][k]:.1f}", f"{d['mean_mm'][k]:.1f}",
                            "+".join(d["systems"])])
    return path


__all__ = ["LATENCY", "BlendModel", "write_table"]
