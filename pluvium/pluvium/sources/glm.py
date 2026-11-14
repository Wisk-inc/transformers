# Copyright 2026 Nathan. Apache-2.0.
"""
Lightning now: flashes seen by GOES-East's Geostationary Lightning Mapper in the last minutes (``noaa-goes19``,
``GLM-L2-LCFA``, one file per 20 seconds), counted around every town. It is the one direct observation of thunder
over the whole Caribbean, so live it answers "is there lightning near me right now" and sharpens the first hour's
thunder chance (flashes within 10 km in the last 15 minutes almost always mean thunder audible now).
"""

from __future__ import annotations

import io
import time
from datetime import datetime, timezone

import numpy as np

from ..util import fetch, log, parallel_map

BUCKET = "noaa-goes19"
PRODUCT = "GLM-L2-LCFA"
LIST = "https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={prefix}"
OBJECT = "https://{bucket}.s3.amazonaws.com/{key}"


def _keys(bucket: str, prefix: str) -> list[str]:
    import re

    text = fetch(LIST.format(bucket=bucket, prefix=prefix), timeout=60.0, attempts=3).decode()
    return re.findall(r"<Key>([^<]+)</Key>", text)


def recent_files(minutes: int = 15, now: float | None = None, bucket: str = BUCKET) -> list[str]:
    now = time.time() if now is None else now
    keys = []
    for back in (0, 1):
        t = datetime.fromtimestamp(now - back * 3600, tz=timezone.utc)
        keys += _keys(bucket, f"{PRODUCT}/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/")
    cutoff = datetime.fromtimestamp(now - minutes * 60, tz=timezone.utc).strftime("%Y%j%H%M%S")
    return sorted(k for k in keys if k.split("_s")[-1][:13] >= cutoff)


def flashes(keys: list[str], bucket: str = BUCKET, workers: int = 16) -> np.ndarray:
    """``(N, 2)`` flash latitudes and longitudes from GLM files."""
    import h5py

    def one(key):
        try:
            with h5py.File(io.BytesIO(fetch(OBJECT.format(bucket=bucket, key=key), timeout=60.0, attempts=3)), "r") as f:
                return np.stack([f["flash_lat"][:].astype(np.float64), f["flash_lon"][:].astype(np.float64)], -1)
        except Exception:
            return np.zeros((0, 2))

    got = [g for g in parallel_map(one, keys, workers=workers) if g is not None and len(g)]
    return np.concatenate(got) if got else np.zeros((0, 2))


def flashes_near(lat: np.ndarray, lon: np.ndarray, minutes: int = 15, radius_km: float = 10.0,
                 satellite: str = "goes-east") -> tuple[np.ndarray, str]:
    """Number of flashes within ``radius_km`` of each point in the last ``minutes``, and a note of the source."""
    from scipy.spatial import cKDTree

    try:
        bucket = {"goes-east": "noaa-goes19", "goes-west": "noaa-goes18"}[satellite]
        keys = recent_files(minutes, bucket=bucket)
        pts = flashes(keys, bucket=bucket)
    except Exception as error:
        log(f"lightning not read ({type(error).__name__}: {error})")
        return np.zeros(np.size(lat), np.int32), "none"
    if len(pts) == 0:
        return np.zeros(np.size(lat), np.int32), f"GOES-East GLM, {len(keys)} files, no flashes"

    def xyz(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)

    tree = cKDTree(xyz(pts[:, 0], pts[:, 1]))
    counts = tree.query_ball_point(xyz(np.asarray(lat), np.asarray(lon)), r=radius_km / 6371.0, return_length=True)
    return np.asarray(counts, np.int32), f"GOES-East GLM, {len(keys)} files, {len(pts):,} flashes in the region"


__all__ = ["flashes", "flashes_near", "recent_files"]
