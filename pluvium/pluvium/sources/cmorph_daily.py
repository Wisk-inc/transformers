# Copyright 2026 Nathan. Apache-2.0.
"""
Daily rain truth for any region of the world between 60 S and 60 N: CMORPH, summed over each UTC day and averaged onto
the 0.25-degree grid the operational systems publish.

Every hourly CMORPH file covers the whole band, so one pass over the files serves every region at once (the same idea
as :mod:`pluvium.sources.archive`). A pixel's day counts when at least 42 of its 48 half-hours are known; a 0.25-degree
cell counts when at least 80% of its pixels do.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from ..domain import cmorph_axes
from ..util import fetch, hour_of, hour_to_datetime64, log
from .cmorph import URL, url_for

STEP = 0.25


def region_grid(box, step: float = STEP) -> tuple[np.ndarray, np.ndarray]:
    """Cell centres on the 0.25-degree grid (multiples of 0.25, as ECMWF and NOAA publish) inside ``box``."""
    s, n, w, e = box
    lat = np.arange(np.ceil(s / step) * step, n + 1e-9, step)
    lon = np.arange(np.ceil(w / step) * step, e + 1e-9, step)
    return lat.round(4), lon.round(4)


class _Plan:
    """For one region: the CMORPH window and, for each pixel in it, the 0.25-degree cell it falls in."""

    def __init__(self, box) -> None:
        lat_c, lon_c = cmorph_axes()
        s, n, w, e = box
        half = STEP / 2
        rows = np.nonzero((lat_c >= s - half) & (lat_c < n + half))[0]
        lon_signed = (lon_c + 180.0) % 360.0 - 180.0
        cols = np.nonzero((lon_signed >= w - half) & (lon_signed < e + half))[0]
        self.rows = slice(int(rows[0]), int(rows[-1]) + 1)
        self.cols = np.sort(cols)
        self.lat, self.lon = region_grid(box)
        gi = np.clip(np.round((lat_c[rows] - self.lat[0]) / STEP).astype(int), 0, self.lat.size - 1)
        gj = np.clip(np.round((lon_signed[self.cols] - self.lon[0]) / STEP).astype(int), 0, self.lon.size - 1)
        self.cell = (gi[:, None] * self.lon.size + gj[None, :]).ravel()
        self.n_cells = self.lat.size * self.lon.size


def _read_day(args) -> tuple[int, dict]:
    """One UTC day: the 24 hourly files, every region's daily total on its 0.25-degree grid (NaN where unknown)."""
    import h5py

    day_hour, plans = args
    sums = {r: None for r in plans}
    counts = {r: None for r in plans}
    for h in range(24):
        blob = fetch(url_for(day_hour + h, URL), timeout=120.0, attempts=6, missing_ok=True)
        if blob is None:
            continue
        with h5py.File(io.BytesIO(blob), "r") as f:
            var = f["cmorph"]
            scale = float(np.ravel(var.attrs.get("scale_factor", [0.01]))[0])
            for r, p in plans.items():
                raw = var[:, p.rows, :][:, :, p.cols].astype(np.float32)          # (2 half-hours, rows, cols)
                ok = raw >= 0
                rate = np.where(ok, raw * scale, 0.0)
                sums[r] = rate.sum(0) if sums[r] is None else sums[r] + rate.sum(0)
                counts[r] = ok.sum(0) if counts[r] is None else counts[r] + ok.sum(0)
    out = {}
    for r, p in plans.items():
        if sums[r] is None:
            out[r] = np.full((p.lat.size, p.lon.size), np.nan, np.float32)
            continue
        known = counts[r] >= 42                                     # of 48 half-hours
        daily = np.where(known, sums[r] / np.maximum(counts[r], 1) * 24.0, 0.0).ravel()   # mean rate x 24 h = mm
        good = np.bincount(p.cell, known.ravel().astype(np.float64), p.n_cells)
        total = np.bincount(p.cell, daily, p.n_cells)
        pixels = np.bincount(p.cell, minlength=p.n_cells).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            cell = np.where(good >= 0.8 * pixels, total / np.maximum(good, 1), np.nan)
        out[r] = cell.reshape(p.lat.size, p.lon.size).astype(np.float32)
    return day_hour, out


class DailyTruth:
    """Daily CMORPH totals per region: ``days(region, first_day_hour, n) -> (n, H, W)`` mm, NaN where unknown."""

    def __init__(self, root) -> None:
        self.root = Path(root).expanduser()
        self._cache = {}

    def _file(self, region: str, month: str) -> Path:
        return self.root / region / f"{month}.npz"

    def grid(self, region: str) -> tuple[np.ndarray, np.ndarray]:
        meta = json.loads((self.root / region / "grid.json").read_text())
        return np.asarray(meta["lat"]), np.asarray(meta["lon"])

    def _month(self, region: str, month: str):
        key = (region, month)
        if key not in self._cache:
            p = self._file(region, month)
            self._cache[key] = np.load(p) if p.exists() else None
        return self._cache[key]

    def days(self, region: str, day_hours) -> np.ndarray:
        lat, lon = self.grid(region)
        day_hours = np.atleast_1d(np.asarray(day_hours, np.int64))
        out = np.full((day_hours.size, lat.size, lon.size), np.nan, np.float32)
        for i, h in enumerate(day_hours):
            z = self._month(region, str(hour_to_datetime64(int(h)))[:7])
            if z is None:
                continue
            k = np.nonzero(z["days"] == h)[0]
            if k.size:
                out[i] = z["rain"][k[0]]
        return out


def stage_daily(root, regions: dict, first_day: str, end_day: str, workers: int = 4) -> DailyTruth:
    """Stage daily totals for ``regions`` from ``first_day`` up to ``end_day`` (exclusive), month by month (resumes)."""
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
            # what each region already holds for this month. A day stored with nothing known counts as not held:
            # it was asked for before NOAA had published it, and may be there now.
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
            log(f"daily truth {month}: {len(wanted)} days, {known:.0%} of cells known")
    return DailyTruth(root)


#: the days NOAA has published for one month, as the bucket lists them
LISTING = "https://noaa-cdr-precip-cmorph-pds.s3.amazonaws.com/?list-type=2&delimiter=/&prefix=data/30min/8km/{y}/{m}/"


def truth_until(now: int | None = None, months: int = 18, enough: int = 25, listing: str = LISTING) -> int | None:
    """The last UTC day (hours since 1970, at 00 UTC) for which the CMORPH record can be relied on: the last day
    published in the newest month that has at least ``enough`` days. NOAA adds to this record months after the
    fact and fills a month in over several weeks, so a month with a few days is not counted yet. None when the
    bucket cannot be listed (no network)."""
    import re
    from datetime import datetime, timezone

    now = int(datetime.now(timezone.utc).timestamp() // 3600) if now is None else int(now)
    stamp = str(hour_to_datetime64(now))
    year, month = int(stamp[:4]), int(stamp[5:7])
    for _ in range(months):
        try:
            text = fetch(listing.format(y=year, m=f"{month:02d}"), attempts=3).decode()
        except Exception:
            return None
        days = sorted(int(d) for d in re.findall(r"<Prefix>data/30min/8km/\d{4}/\d{2}/(\d{2})/</Prefix>", text))
        if len(days) >= enough:
            last = hour_of(f"{year}-{month:02d}-{days[-1]:02d}T00")
            return min(last, now - now % 24)
        year, month = (year, month - 1) if month > 1 else (year - 1, 12)
    return None


__all__ = ["DailyTruth", "STEP", "region_grid", "stage_daily", "truth_until"]
