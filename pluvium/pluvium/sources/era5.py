# Copyright 2026 Nathan. Apache-2.0.
"""
Training predictors: ERA5 reanalysis from the public ARCO and WeatherBench 2 stores on Google Cloud.

* **Hourly, 1 degree** (``1959-2022-1h-360x181``): rain, water vapour, 10 m wind, 2 m temperature, sea-level
  pressure and cloud cover, every hour. A chunk is eight hours of the whole globe for one variable (about 1 MB), so
  eight years of the seven fields cost about 70 GB to read and 2 GB to keep once cropped to the Caribbean.
* **Six-hourly, 1.5 degrees** (WeatherBench 2): wind, humidity and temperature at 850, 700, 500 and 200 hPa, for the
  fronts aloft, the tropical waves and the moisture through the column (about 6 GB a year to read).

Caches are float32 arrays per variable and year, ``(hours or steps, lat, lon)`` with latitude and longitude
ascending over :data:`pluvium.domain.DOMAIN`. Rain is stored as **mm in hour h** (ERA5's accumulation that ends at
h + 1), everything else as the value at the start of the hour.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..domain import DOMAIN, signed_lon
from ..features import HOURLY_SHORT, HOURLY_VARS, LEVELS, UPPER_SHORT, UPPER_VARS
from ..util import atomic_save, calendar, hour_of, log, year_hours
from ..zarrlite import ZarrStore, axis_window

HOURLY_STORE = ("https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
                "1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr")
UPPER_STORE = ("https://storage.googleapis.com/weatherbench2/datasets/era5/"
               "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr")
#: Last hour the hourly store holds. Its name says 1959-2022, and its time axis ends on 31 December 2021 (read from
#: the store on 15 November 2026). Until 0.4.3 this said 2022-12-31T23, and staging 2022 failed with an IndexError.
#: :func:`stage_era5` and :func:`hourly_years` ask the store itself; this is what they fall back on.
HOURLY_LAST = hour_of("2021-12-31T23")


def hourly_years(store: ZarrStore | None = None) -> tuple[int, int]:
    """``(first, last)`` whole calendar years the hourly store holds, read from its time axis (from
    :data:`HOURLY_LAST` when the store cannot be reached). The last year is the newest one training can hold out."""
    try:
        hours = (store or ZarrStore(HOURLY_STORE)).hours()
        first, last = int(hours[0]), int(hours[-1])
    except Exception:
        first, last = hour_of("1959-01-01T00"), HOURLY_LAST
    year_a, year_b = int(str(np.datetime64(first, "h"))[:4]), int(str(np.datetime64(last, "h"))[:4])
    if first > year_hours(year_a)[0]:
        year_a += 1
    if last < year_hours(year_b)[1] - 1:
        year_b -= 1
    return year_a, year_b


def _box_slices(store: ZarrStore, box, pad: float) -> tuple[slice, slice, np.ndarray, np.ndarray]:
    lat = store.coordinate("latitude")
    lon = store.coordinate("longitude")
    s, n, w, e = box
    if w - pad < 0 < e + pad:
        raise ValueError(f"box {box} crosses longitude 0, which ERA5 staging does not handle yet; train stage B on "
                         "GFS runs from an existing checkpoint instead")
    la = axis_window(lat, s - pad, n + pad)
    lo = axis_window(lon, w - pad, e + pad, wrap=360.0)
    return la, lo, lat[la].astype(np.float64), signed_lon(lon[lo]).astype(np.float64)


def _transpose(block: np.ndarray, dims: list) -> np.ndarray:
    """Arrange a ``(time, [level,] longitude, latitude)`` block as ``(time, [level,] latitude, longitude)``."""
    if dims and dims[-1] == "latitude" and dims[-2] == "longitude":
        return np.swapaxes(block, -1, -2)
    return block


class ERA5Cache:
    """Read access to staged ERA5: ``hourly(short, hours)`` and ``upper(name, hours)`` (time-interpolated)."""

    def __init__(self, root) -> None:
        self.root = Path(root).expanduser()
        meta = json.loads((self.root / "era5.json").read_text())
        self.lat1, self.lon1 = np.array(meta["lat1"]), np.array(meta["lon1"])
        self.lat2, self.lon2 = np.array(meta["lat2"]), np.array(meta["lon2"])
        self.years = sorted(meta["years"])
        self._open = {}

    def _array(self, kind: str, name: str, year: int) -> np.ndarray:
        key = (kind, name, year)
        if key not in self._open:
            self._open[key] = np.load(self.root / kind / f"{name}_{year}.npy", mmap_mode="r")
        return self._open[key]

    def hourly(self, short: str, hours: np.ndarray) -> np.ndarray:
        """``(len(hours), H1, W1)`` float32; hours outside the staged years read as NaN."""
        hours = np.asarray(hours, np.int64)
        out = np.full((hours.size, self.lat1.size, self.lon1.size), np.nan, np.float32)
        years = calendar(hours)["year"]
        for year in np.unique(years):
            if year not in self.years:
                continue
            first, _ = year_hours(int(year))
            sel = np.nonzero(years == year)[0]
            arr = self._array("era5_1h", short, int(year))
            idx = hours[sel] - first
            ok = (idx >= 0) & (idx < arr.shape[0])
            out[sel[ok]] = arr[idx[ok]]
        return out

    def upper(self, name: str, hours: np.ndarray) -> np.ndarray:
        """``(len(hours), H2, W2)`` at any hours, linear in time between the six-hourly analyses."""
        hours = np.asarray(hours, np.float64)
        lo = np.floor(hours / 6.0) * 6.0
        frac = (hours - lo) / 6.0
        a = self._upper_at(name, lo.astype(np.int64))
        b = self._upper_at(name, (lo + 6).astype(np.int64))
        return (a * (1 - frac)[:, None, None] + b * frac[:, None, None]).astype(np.float32)

    def _upper_at(self, name: str, hours: np.ndarray) -> np.ndarray:
        out = np.full((hours.size, self.lat2.size, self.lon2.size), np.nan, np.float32)
        years = calendar(hours)["year"]
        for k, h in enumerate(hours):
            year = int(years[k])
            if year not in self.years:
                continue
            first, _ = year_hours(year)
            arr = self._array("era5_6h", name, year)
            i = (int(h) - first) // 6
            if 0 <= i < arr.shape[0]:
                out[k] = arr[i]
        return out


def _merge(path: Path, shape: tuple, offset: int, block: np.ndarray) -> None:
    """Write ``block`` into the year array at ``path`` (created full of NaN) starting at ``offset``."""
    full = np.load(path) if path.exists() else np.full(shape, np.nan, np.float32)
    full[offset: offset + block.shape[0]] = block[: max(0, shape[0] - offset)]
    atomic_save(path, lambda p, a=full: np.save(p, a))


def _missing(path: Path, a: int, b: int) -> tuple[int, int] | None:
    """The smallest ``[lo, hi)`` inside ``[a, b)`` that covers every step of ``path`` not staged yet (None: all
    staged)."""
    if not path.exists():
        return (a, b)
    arr = np.load(path, mmap_mode="r")
    gap = np.flatnonzero(np.isnan(arr[a:b, 0, 0]))
    if gap.size == 0:
        return None
    return (a + int(gap[0]), a + int(gap[-1]) + 1)


def stage_era5(root, years=None, box=DOMAIN, workers: int = 32, include_upper: bool = True, period=None,
               verbose: bool = True) -> ERA5Cache:
    """
    Download and crop ERA5 for whole ``years``, or for ``period = (first_hour, end_hour)``. Hours already staged
    are not fetched again; a year array is created full of NaN and filled as periods arrive. Returns the cache.
    """
    root = Path(root).expanduser()
    (root / "era5_1h").mkdir(parents=True, exist_ok=True)
    (root / "era5_6h").mkdir(parents=True, exist_ok=True)
    spans = []
    if period is not None:
        a, b = int(period[0]), int(period[1])
        for year in range(int(str(np.datetime64(a, "h"))[:4]), int(str(np.datetime64(b - 1, "h"))[:4]) + 1):
            first, end = year_hours(year)
            spans.append((year, max(a, first), min(b, end)))
    else:
        for year in sorted(years or []):
            first, end = year_hours(year)
            spans.append((year, first, end))
    hourly = ZarrStore(HOURLY_STORE)
    la1, lo1, lat1, lon1 = _box_slices(hourly, box, pad=0.0)
    hours1 = hourly.hours()
    upper = ZarrStore(UPPER_STORE) if include_upper else None
    if upper is not None:
        la2, lo2, lat2, lon2 = _box_slices(upper, box, pad=1.5)
        hours2 = upper.hours()
        levels = [int(x) for x in upper.coordinate("level")]
    else:
        lat2, lon2 = np.zeros(0), np.zeros(0)
    staged = []
    last = int(hours1[-1])                                      # the store's own last hour, not a date kept in code
    for year, a, b in spans:
        first, end = year_hours(year)
        if a > last:
            log(f"  ERA5 {year}: past the hourly store's end ({str(np.datetime64(last, 'h'))[:10]}); skipped")
            continue
        b = min(b, last + 1)
        for var, short in zip(HOURLY_VARS, HOURLY_SHORT):
            path = root / "era5_1h" / f"{short}_{year}.npy"
            gap = _missing(path, a - first, b - first)
            if gap is None:
                continue
            i0, i1 = int(np.searchsorted(hours1, first + gap[0])), int(np.searchsorted(hours1, first + gap[1]))
            array = hourly[var]
            stop = min(i1 + 1, array.shape[0]) if short == "tp" else i1
            block = _transpose(array.read((slice(i0, stop), lo1, la1), workers=workers,
                                          progress=f"ERA5 {short} {year}" if verbose else None), array.dims)
            block = block.astype(np.float32)
            if short == "tp":   # accumulation ending at h + 1 -> rain in hour h, in mm
                pad = np.full_like(block[:1], np.nan)
                block = np.concatenate([block[1:], pad])[: i1 - i0] * 1000.0
            _merge(path, (end - first, lat1.size, lon1.size), int(hours1[i0] - first), block)
        if upper is not None:
            # 6-hourly steps covering [a, b), plus the one after (hours past the last step interpolate towards it)
            s0 = (a - first) // 6
            s1 = min((end - first) // 6, -(-(b - first) // 6) + 1)
            for var, short in zip(UPPER_VARS, UPPER_SHORT):
                gaps = {p: _missing(root / "era5_6h" / f"{short}{p}_{year}.npy", s0, s1) for p in LEVELS}
                want = [p for p, g in gaps.items() if g is not None]
                if not want:
                    continue
                g_lo = min(gaps[p][0] for p in want)
                g_hi = max(gaps[p][1] for p in want)
                j0 = int(np.searchsorted(hours2, first + 6 * g_lo))
                j1 = int(np.searchsorted(hours2, first + 6 * g_hi))
                if j1 <= j0:
                    continue
                array = upper[var]
                block = _transpose(array.read((slice(j0, j1), slice(None), lo2, la2), workers=workers,
                                              progress=f"ERA5 {short} aloft {year}" if verbose else None), array.dims)
                steps = (end - first) // 6
                for p in want:
                    k = levels.index(p)
                    start = int((hours2[j0] - first) // 6)
                    lo, hi = max(0, -start), max(0, min(block.shape[0], steps - start))
                    _merge(root / "era5_6h" / f"{short}{p}_{year}.npy", (steps, lat2.size, lon2.size), max(start, 0),
                           block[lo:hi, k].astype(np.float32))
        staged.append(year)
        if verbose:
            log(f"  ERA5 {year}: staged {str(np.datetime64(a, 'h'))[:13]} to {str(np.datetime64(b, 'h'))[:13]}")
    meta_path = root / "era5.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"years": []}
    meta.update({"lat1": lat1.tolist(), "lon1": lon1.tolist(), "lat2": lat2.tolist(), "lon2": lon2.tolist(),
                 "years": sorted(set(meta.get("years", [])) | set(staged))})
    atomic_save(meta_path, lambda p: Path(p).write_text(json.dumps(meta)))
    return ERA5Cache(root)


__all__ = ["ERA5Cache", "HOURLY_LAST", "HOURLY_STORE", "UPPER_STORE", "hourly_years", "stage_era5"]
