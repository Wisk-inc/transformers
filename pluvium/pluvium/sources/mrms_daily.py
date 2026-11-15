# Copyright 2026 Nathan. Apache-2.0.
"""
A second truth for the United States: MRMS radar and gauge rain.

CMORPH, the satellite estimate every Pluvium benchmark is scored against, is weak where precipitation is frozen: in
a continental winter it misses snow and light stratiform rain. NOAA's Multi-Radar Multi-Sensor analysis (MRMS)
combines radars with rain gauges on a 0.01 degree grid over the contiguous United States. Its 24-hour "pass 2"
product, issued each hour, is public on AWS.

This module reads the 24 hours ending at 00 UTC, which is the same day CMORPH's daily total covers, and averages it
onto forecast cells. It is used to score winter forecasts a second time (:mod:`pluvium.benchmarks.winter_bench`).

(:mod:`pluvium.sources.mrms` is the live reader for the Caribbean radar domain; this one is for verification over
the contiguous United States.)
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from ..util import fetch, hour_to_datetime64, log, parallel_map

URL = ("https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_24H_Pass2_00.00/{day}/"
       "MRMS_MultiSensor_QPE_24H_Pass2_00.00_{day}-000000.grib2.gz")
BOX = (20.0, 55.0, -130.0, -60.0)             # south, north, west, east of the analysis


def _stamp(hour: int) -> str:
    return str(hour_to_datetime64(int(hour)))[:10].replace("-", "")


def decode(blob: bytes) -> tuple:
    """A gzipped MRMS GRIB2 message -> ``(values (lat ascending, lon ascending), lat, lon)`` in mm, NaN where the
    analysis has no coverage (it marks those with negative values)."""
    from .nwp import decode as decode_grib

    values, lat, lon = decode_grib(gzip.decompress(blob))
    return np.where(values >= 0, values, np.nan).astype(np.float32), lat, lon


def box_means(values: np.ndarray, lat: np.ndarray, lon: np.ndarray, cell_lat: np.ndarray, cell_lon: np.ndarray,
              half: float = 0.125, min_share: float = 0.5) -> np.ndarray:
    """The mean of a fine regular grid over the box ``+-half`` degrees around every cell, NaN where less than
    ``min_share`` of the box is covered. Uses summed-area tables, so the cost does not grow with the box."""
    ok = np.isfinite(values)
    total = np.zeros((values.shape[0] + 1, values.shape[1] + 1), np.float64)
    count = np.zeros_like(total)
    total[1:, 1:] = np.cumsum(np.cumsum(np.where(ok, values, 0.0), 0, dtype=np.float64), 1)
    count[1:, 1:] = np.cumsum(np.cumsum(ok, 0, dtype=np.float64), 1)
    i0 = np.searchsorted(lat, cell_lat - half, side="left")
    i1 = np.searchsorted(lat, cell_lat + half, side="left")
    j0 = np.searchsorted(lon, cell_lon - half, side="left")
    j1 = np.searchsorted(lon, cell_lon + half, side="left")

    def area(table):
        return table[i1, j1] - table[i0, j1] - table[i1, j0] + table[i0, j0]

    n, size = area(count), np.maximum((i1 - i0) * (j1 - j0), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n >= min_share * size, area(total) / np.maximum(n, 1), np.nan)
    return np.where(size > 0, out, np.nan).astype(np.float32)


class MRMSDaily:
    """Daily MRMS rain at a set of cells, kept on disk one small file per day."""

    def __init__(self, cache, cell_lat: np.ndarray, cell_lon: np.ndarray, name: str, half: float = 0.125) -> None:
        self.folder = Path(cache).expanduser() / name
        self.lat, self.lon, self.half = np.asarray(cell_lat, np.float64), np.asarray(cell_lon, np.float64), half

    def covered(self) -> np.ndarray:
        """Which cells lie inside the analysis."""
        s, n, w, e = BOX
        return (self.lat - self.half >= s) & (self.lat + self.half <= n) & (self.lon - self.half >= w) & (self.lon + self.half <= e)

    def day(self, day_hour: int, download: bool = True) -> np.ndarray | None:
        """Rain (mm) at the cells for the day starting at ``day_hour`` (00 UTC); ``None`` when it cannot be had."""
        path = self.folder / f"{_stamp(day_hour)}.npy"
        if path.exists():
            return np.load(path)
        if not download:
            return None
        blob = fetch(URL.format(day=_stamp(int(day_hour) + 24)), attempts=3, missing_ok=True, timeout=120.0)
        if blob is None:
            return None
        try:
            values, lat, lon = decode(blob)
        except Exception as error:
            log(f"mrms {_stamp(day_hour)}: unreadable ({type(error).__name__})")
            return None
        out = box_means(values, lat, lon, self.lat, self.lon, self.half)
        self.folder.mkdir(parents=True, exist_ok=True)
        np.save(path, out)
        return out

    def stage(self, day_hours, workers: int = 4) -> int:
        """Fetch every day (skipping those on disk); returns how many are there."""
        got = parallel_map(lambda h: self.day(int(h)) is not None, list(day_hours), workers=workers,
                           progress="mrms days", every=120.0)
        return int(sum(got))


__all__ = ["BOX", "URL", "MRMSDaily", "box_means", "decode"]
