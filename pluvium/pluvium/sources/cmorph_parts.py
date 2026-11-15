# Copyright 2026 Nathan. Apache-2.0.
"""
Rain truth for parts of the day: CMORPH summed over each six-hour period of the UTC day (00-06, 06-12, 12-18,
18-24) and averaged onto the 0.25-degree grid, the same way :mod:`pluvium.sources.cmorph_daily` makes daily totals.

The daily truth says whether it rained. This says when. It is what the time-of-day forecast
(:mod:`pluvium.timing`) is checked against, and where the usual time of day of rain at a place comes from.

A pixel's period counts when at least 10 of its 12 half-hours are known; a 0.25-degree cell counts when at least
80% of its pixels do.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from ..util import fetch, hour_of, hour_to_datetime64, log
from .cmorph import URL, url_for
from .cmorph_daily import _Plan

PARTS = 4                    # six-hour periods in a UTC day
HOURS = 24 // PARTS


def _read_day(args) -> tuple[int, dict]:
    """One UTC day: the 24 hourly files, every region's total in each period on its 0.25-degree grid, ``(4, H, W)``
    mm (NaN where unknown)."""
    import h5py

    day_hour, plans = args
    sums = {r: None for r in plans}
    counts = {r: None for r in plans}
    for h in range(24):
        blob = fetch(url_for(day_hour + h, URL), timeout=120.0, attempts=6, missing_ok=True)
        if blob is None:
            continue
        part = h // HOURS
        with h5py.File(io.BytesIO(blob), "r") as f:
            var = f["cmorph"]
            scale = float(np.ravel(var.attrs.get("scale_factor", [0.01]))[0])
            for r, p in plans.items():
                raw = var[:, p.rows, :][:, :, p.cols].astype(np.float32)          # (2 half-hours, rows, cols)
                ok = raw >= 0
                rate = np.where(ok, raw * scale, 0.0)
                if sums[r] is None:
                    sums[r] = np.zeros((PARTS,) + rate.shape[1:], np.float32)
                    counts[r] = np.zeros((PARTS,) + rate.shape[1:], np.int32)
                sums[r][part] += rate.sum(0)
                counts[r][part] += ok.sum(0)
    out = {}
    for r, p in plans.items():
        cells = np.full((PARTS, p.lat.size, p.lon.size), np.nan, np.float32)
        if sums[r] is not None:
            pixels = np.bincount(p.cell, minlength=p.n_cells).astype(np.float64)
            for k in range(PARTS):
                known = counts[r][k] >= 2 * HOURS - 2                              # 10 of 12 half-hours
                total = np.where(known, sums[r][k] / np.maximum(counts[r][k], 1) * HOURS, 0.0).ravel()   # mean rate x 6 h
                good = np.bincount(p.cell, known.ravel().astype(np.float64), p.n_cells)
                summed = np.bincount(p.cell, total, p.n_cells)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cell = np.where(good >= 0.8 * pixels, summed / np.maximum(good, 1), np.nan)
                cells[k] = cell.reshape(p.lat.size, p.lon.size)
        out[r] = cells
    return day_hour, out


class PartsTruth:
    """Six-hour CMORPH totals per region: ``days(region, day_hours) -> (n, 4, H, W)`` mm, NaN where unknown."""

    def __init__(self, root) -> None:
        self.root = Path(root).expanduser()
        self._cache = {}

    def grid(self, region: str) -> tuple[np.ndarray, np.ndarray]:
        meta = json.loads((self.root / region / "grid.json").read_text())
        return np.asarray(meta["lat"]), np.asarray(meta["lon"])

    def _month(self, region: str, month: str):
        key = (region, month)
        if key not in self._cache:
            path = self.root / region / f"{month}.npz"
            self._cache[key] = np.load(path) if path.exists() else None
        return self._cache[key]

    def days(self, region: str, day_hours) -> np.ndarray:
        lat, lon = self.grid(region)
        day_hours = np.atleast_1d(np.asarray(day_hours, np.int64))
        out = np.full((day_hours.size, PARTS, lat.size, lon.size), np.nan, np.float32)
        for i, h in enumerate(day_hours):
            z = self._month(region, str(hour_to_datetime64(int(h)))[:7])
            if z is None:
                continue
            k = np.nonzero(z["days"] == h)[0]
            if k.size:
                out[i] = z["rain"][k[0]]
        return out

    def held(self, region: str) -> np.ndarray:
        """Every day (as its first UTC hour) this region has on disk, in order."""
        days = []
        for path in sorted((self.root / region).glob("????-??.npz")):
            with np.load(path) as z:
                days.append(z["days"])
        return np.sort(np.concatenate(days)) if days else np.zeros(0, np.int64)


def stage_parts(root, regions: dict, first_day: str, end_day: str, workers: int = 4) -> PartsTruth:
    """Stage six-hour totals for ``regions`` from ``first_day`` up to ``end_day`` (exclusive), month by month. Days
    already on disk are not read again, except those stored with nothing known (asked for before NOAA had published
    them)."""
    from concurrent.futures import ProcessPoolExecutor

    root = Path(root).expanduser()
    plans = {r: _Plan(box) for r, box in regions.items()}
    for r, p in plans.items():
        (root / r).mkdir(parents=True, exist_ok=True)
        (root / r / "grid.json").write_text(json.dumps({"box": regions[r], "lat": p.lat.tolist(), "lon": p.lon.tolist()}))
    a, b = hour_of(first_day), hour_of(end_day)
    a -= a % 24
    months = sorted({str(hour_to_datetime64(int(h)))[:7] for h in range(a, b, 24)})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for month in months:
            days = [h for h in range(a, b, 24) if str(hour_to_datetime64(h))[:7] == month]
            held = {}
            for r in regions:
                path = root / r / f"{month}.npz"
                if path.exists():
                    with np.load(path) as z:
                        held[r] = {int(h): z["rain"][i] for i, h in enumerate(z["days"])}
                else:
                    held[r] = {}
            wanted = [h for h in days if any(h not in held[r] or not np.isfinite(held[r][h]).any() for r in regions)]
            if not wanted:
                continue
            got = dict(pool.map(_read_day, [(h, plans) for h in wanted], chunksize=1))
            for r in plans:
                merged = {**held[r], **{h: got[h][r] for h in wanted}}
                order = sorted(merged)
                tmp = root / r / f"{month}.tmp.npz"
                np.savez_compressed(tmp, days=np.asarray(order, np.int64),
                                    rain=np.stack([merged[h] for h in order]).astype(np.float32))
                tmp.replace(root / r / f"{month}.npz")
            known = np.mean([np.isfinite(got[h][r]).mean() for h in wanted for r in plans])
            log(f"six-hour truth {month}: {len(wanted)} days, {known:.0%} of cells known")
    return PartsTruth(root)


__all__ = ["HOURS", "PARTS", "PartsTruth", "stage_parts"]
