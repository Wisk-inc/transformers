# Copyright 2026 Nathan. Apache-2.0.
"""
Rain already falling, live, from GOES (the forecast's "history" frames).

In training, the last three hours of rain come from CMORPH. CMORPH's archive lags by months, so a live forecast
reads the newest GOES imagery instead: the ABI rain-rate product (``ABI-L2-RRQPEF``, 2 km every 10 minutes) when it
is being published, otherwise band 13 brightness temperature turned into a rain rate with the infrared
auto-estimator of Vicente et al. (1998). Both are cruder than CMORPH, so the forecast says which one it used; the
model was trained with the history frames sometimes withheld, so it also runs with none (``--no-satellite``).

Only the parts of each 20-30 MB file that cover the Caribbean are read, through HTTP range requests.
"""

from __future__ import annotations

import io
import re

import numpy as np

from ..util import fetch, hour_to_datetime64, log

BUCKETS = ("noaa-goes19", "noaa-goes16")
LIST = "https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={prefix}"
OBJECT = "https://{bucket}.s3.amazonaws.com/{key}"


class RangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests with a block cache, for h5py to read chunks from."""

    def __init__(self, url: str, block: int = 1 << 20) -> None:
        self.url, self.block, self.pos, self.blocks = url, block, 0, {}
        head = fetch(url, 0, 0)
        self.size = None
        self._probe = head
        import urllib.request

        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def _block(self, k: int) -> bytes:
        if k not in self.blocks:
            start = k * self.block
            self.blocks[k] = fetch(self.url, start, min(self.size, start + self.block) - 1)
        return self.blocks[k]

    def readinto(self, buffer) -> int:
        n = min(len(buffer), max(0, self.size - self.pos))
        out, got = memoryview(buffer), 0
        while got < n:
            k, off = divmod(self.pos + got, self.block)
            chunk = self._block(k)[off: off + n - got]
            out[got: got + len(chunk)] = chunk
            got += len(chunk)
        self.pos += got
        return got


def list_keys(bucket: str, prefix: str) -> list[str]:
    text = fetch(LIST.format(bucket=bucket, prefix=prefix), missing_ok=True)
    return re.findall(r"<Key>([^<]+)</Key>", text.decode()) if text else []


def scene_near(product: str, hour: int, band: str | None = None) -> tuple[str, str] | None:
    """``(bucket, key)`` of the full-disk scene closest to the middle of ``hour``, newest satellite first."""
    when = hour_to_datetime64(hour)
    s = str(when)
    doy = int((when.astype("datetime64[D]") - when.astype("datetime64[Y]")).astype(int)) + 1
    prefix = f"{product}/{s[:4]}/{doy:03d}/{s[11:13]}/"
    for bucket in BUCKETS:
        keys = list_keys(bucket, prefix)
        if band:
            keys = [k for k in keys if f"M6{band}_" in k or f"-{band}_" in k]
        if keys:
            return bucket, sorted(keys, key=lambda k: abs(int(re.search(r"_s\d{11}(\d{2})", k).group(1)) - 30))[0]
    return None


def ir_rain_rate(tb: np.ndarray) -> np.ndarray:
    """Vicente et al. (1998) auto-estimator: rain rate (mm/h) from 10.3 um brightness temperature (K)."""
    with np.errstate(over="ignore", invalid="ignore"):
        rate = 1.1183e11 * np.exp(-3.6382e-2 * np.power(np.clip(tb, 180.0, 330.0), 1.2))
    return np.where(np.isfinite(tb) & (tb < 250.0), np.clip(rate, 0, 150.0), 0.0).astype(np.float32)


def read_scene(bucket: str, key: str, lat: np.ndarray, lon: np.ndarray, half_width_deg: np.ndarray) -> np.ndarray:
    """Rain rate (mm/h) averaged around each point from one full-disk RRQPE or band-13 CMI scene."""
    import h5py
    from naturev1.stormsat import latlon_to_fixed_grid

    with h5py.File(RangeFile(OBJECT.format(bucket=bucket, key=key)), "r") as f:
        var = "RRQPE" if "RRQPE" in f else "CMI"
        proj = f["goes_imager_projection"].attrs
        sat_lon = float(np.ravel(proj["longitude_of_projection_origin"])[0])
        height = float(np.ravel(proj["perspective_point_height"])[0])
        xs = f["x"][:] * float(np.ravel(f["x"].attrs.get("scale_factor", [1.0]))[0]) + float(np.ravel(f["x"].attrs.get("add_offset", [0.0]))[0])
        ys = f["y"][:] * float(np.ravel(f["y"].attrs.get("scale_factor", [1.0]))[0]) + float(np.ravel(f["y"].attrs.get("add_offset", [0.0]))[0])
        x, y = latlon_to_fixed_grid(lat, lon, sat_lon, perspective_height=height)
        ok = np.isfinite(x) & np.isfinite(y)
        col = np.interp(x, xs, np.arange(xs.size)) if xs[1] > xs[0] else np.interp(-x, -xs, np.arange(xs.size))
        row = np.interp(-y, -ys, np.arange(ys.size)) if ys[1] < ys[0] else np.interp(y, ys, np.arange(ys.size))
        r0, r1 = int(np.nanmin(row[ok])) - 30, int(np.nanmax(row[ok])) + 30
        c0, c1 = int(np.nanmin(col[ok])) - 30, int(np.nanmax(col[ok])) + 30
        r0, c0 = max(r0, 0), max(c0, 0)
        ds = f[var]
        raw = ds[r0:r1, c0:c1].astype(np.float32)
        scale = float(np.ravel(ds.attrs.get("scale_factor", [1.0]))[0])
        offset = float(np.ravel(ds.attrs.get("add_offset", [0.0]))[0])
        fill = np.ravel(ds.attrs.get("_FillValue", [-1]))[0]
        values = np.where(raw == fill, np.nan, raw * scale + offset)
    rate = values if var == "RRQPE" else ir_rain_rate(values)
    out = np.full(lat.shape, np.nan, np.float32)
    px_deg = 0.02
    for k in np.nonzero(ok)[0]:
        h = max(1, int(round(half_width_deg[k] / px_deg)))
        i, j = int(round(row[k])) - r0, int(round(col[k])) - c0
        patch = rate[max(0, i - h): i + h + 1, max(0, j - h): j + h + 1]
        if patch.size and np.isfinite(patch).any():
            out[k] = float(np.nanmean(patch))
    return out


def recent_rain(lat: np.ndarray, lon: np.ndarray, half_width_deg: np.ndarray, hours: list[int]) -> tuple[np.ndarray, str]:
    """``(len(hours), N)`` rain rates at points for the given hours, and which product supplied them."""
    out = np.full((len(hours), lat.size), np.nan, np.float32)
    used = set()
    for k, h in enumerate(hours):
        hit = scene_near("ABI-L2-RRQPEF", h)
        product = "GOES rain rate (RRQPE)"
        if hit is None:
            hit = scene_near("ABI-L2-CMIPF", h, band="C13")
            product = "GOES infrared (band 13, auto-estimator)"
        if hit is None:
            continue
        try:
            out[k] = read_scene(*hit, lat, lon, half_width_deg)
            used.add(product + f" from {hit[0].replace('noaa-', '').upper()}")
        except Exception as error:
            log(f"  GOES {str(hour_to_datetime64(h))[:13]}Z: not read ({type(error).__name__}: {error})")
    return out, "; ".join(sorted(used)) or "none"


__all__ = ["RangeFile", "ir_rain_rate", "read_scene", "recent_rain", "scene_near"]
