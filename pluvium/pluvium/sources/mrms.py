# Copyright 2026 Nathan. Apache-2.0.
"""
Radar where the Caribbean has it: NOAA's Multi-Radar Multi-Sensor system, Caribbean domain (``noaa-mrms-pds``,
``CARIB/``), covering Puerto Rico, the Virgin Islands and the waters around them.

Radar sees rain at 1 km every 2 minutes, far sharper than the satellite's estimate. Where it reaches, Pluvium uses
it for the rain falling now (``PrecipRate``: the instantaneous rate, mm/h) and for how wet the ground is
(``MultiSensor_QPE_72H``: gauge-corrected 72-hour totals, for the antecedent index), and falls back to GOES
elsewhere. Files are small (about 15 kB compressed) GRIB2, decoded with the same reader as the driving models.
"""

from __future__ import annotations

import gzip
import re
import time
from datetime import datetime, timezone

import numpy as np

from ..util import fetch, log

BUCKET = "https://noaa-mrms-pds.s3.amazonaws.com"
PRODUCTS = {"rate": "PrecipRate_00.00", "qpe1h": "MultiSensor_QPE_01H_Pass2_00.00",
            "qpe72h": "MultiSensor_QPE_72H_Pass2_00.00"}


def latest_key(product: str, now: float | None = None, max_age_minutes: float = 30.0) -> str | None:
    """The newest file of ``product`` no older than ``max_age_minutes``."""
    now = time.time() if now is None else now
    name = PRODUCTS.get(product, product)
    for back in (0, 1):
        day = datetime.fromtimestamp(now - back * 86400, tz=timezone.utc).strftime("%Y%m%d")
        text = fetch(f"{BUCKET}/?list-type=2&prefix=CARIB/{name}/{day}/", timeout=60.0, attempts=3).decode()
        keys = re.findall(r"<Key>([^<]+)</Key>", text)
        while "<IsTruncated>true</IsTruncated>" in text:
            token = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", text).group(1)
            from urllib.parse import quote

            text = fetch(f"{BUCKET}/?list-type=2&prefix=CARIB/{name}/{day}/&continuation-token={quote(token)}",
                         timeout=60.0, attempts=3).decode()
            keys += re.findall(r"<Key>([^<]+)</Key>", text)
        if keys:
            key = sorted(keys)[-1]
            stamp = re.search(r"(\d{8}-\d{6})", key).group(1)
            t = datetime.strptime(stamp, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc).timestamp()
            return key if now - t <= max_age_minutes * 60 else None
    return None


def read_field(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(values, lat ascending, lon ascending)`` of one MRMS file (missing values are NaN)."""
    from .nwp import decode

    values, lat, lon = decode(gzip.decompress(fetch(f"{BUCKET}/{key}", timeout=60.0, attempts=3)))
    values = np.where(values < 0, np.nan, values)                 # -1 / -3: no coverage
    return values.astype(np.float32), lat, lon


def at_points(product: str, lat: np.ndarray, lon: np.ndarray, half_width_km: float = 4.0) -> tuple[np.ndarray, str]:
    """The newest ``product`` averaged within ``half_width_km`` of each point (NaN outside radar coverage)."""
    try:
        key = latest_key(product)
        if key is None:
            return np.full(np.size(lat), np.nan, np.float32), "none recent"
        values, glat, glon = read_field(key)
    except Exception as error:
        log(f"MRMS {product} not read ({type(error).__name__}: {error})")
        return np.full(np.size(lat), np.nan, np.float32), "none"
    dlat, dlon = abs(glat[1] - glat[0]), abs(glon[1] - glon[0])
    r_i = max(1, int(round(half_width_km / 111.195 / dlat)))
    r_j = max(1, int(round(half_width_km / 111.195 / dlon)))
    out = np.full(np.size(lat), np.nan, np.float32)
    ii = np.searchsorted(glat, lat)
    jj = np.searchsorted(glon, lon)
    inside = (lat >= glat[0]) & (lat <= glat[-1]) & (lon >= glon[0]) & (lon <= glon[-1])
    for k in np.flatnonzero(inside):
        patch = values[max(0, ii[k] - r_i): ii[k] + r_i + 1, max(0, jj[k] - r_j): jj[k] + r_j + 1]
        if np.isfinite(patch).any():
            out[k] = float(np.nanmean(patch))
    return out, f"MRMS Caribbean radar ({key.split('/')[-1]})"


__all__ = ["PRODUCTS", "at_points", "latest_key", "read_field"]
