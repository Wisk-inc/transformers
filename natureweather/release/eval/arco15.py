"""
Rebuild WeatherBench 2's 1.5-degree, 6-hourly ERA5 (240x121, with poles) from ARCO-ERA5 at 0.25 degrees,
for dates the WeatherBench 2 store does not cover (it ends 2021-12-31).

Conservative, area-weighted regridding: each 1.5-degree cell is the area-weighted mean of the 0.25-degree
cells it overlaps (weights by overlap in sin(latitude) and in longitude), the scheme WeatherBench 2 used.
Accumulated fields (precipitation, top-of-atmosphere solar) are summed over the six hours ending at the
timestamp. Static fields are taken from the WeatherBench 2 store itself.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
from pathlib import Path

import numpy as np
import xarray as xr

ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
SURFACE = ["2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind",
           "sea_surface_temperature", "surface_pressure", "total_column_water_vapour"]
ACCUMULATED = {"total_precipitation_6hr": "total_precipitation",
               "toa_incident_solar_radiation_6hr": "toa_incident_solar_radiation"}
UPPER = ["geopotential", "specific_humidity", "temperature", "u_component_of_wind", "v_component_of_wind",
         "vertical_velocity"]
STATIC = ["land_sea_mask", "geopotential_at_surface"]


def _overlap_weights(src_centres, src_half, dst_centres, dst_half, periodic=False, sine=False):
    """(dst, src) overlap of 1-D cells; in sin(latitude) for latitude, so weights are areas."""
    s_lo, s_hi = src_centres - src_half, src_centres + src_half
    d_lo, d_hi = dst_centres - dst_half, dst_centres + dst_half
    if not periodic:
        s_lo, s_hi, d_lo, d_hi = (np.clip(x, -90, 90) for x in (s_lo, s_hi, d_lo, d_hi))
    weights = np.zeros((dst_centres.size, src_centres.size))
    shifts = (-360.0, 0.0, 360.0) if periodic else (0.0,)
    for shift in shifts:
        lo = np.maximum(d_lo[:, None], s_lo[None, :] + shift)
        hi = np.minimum(d_hi[:, None], s_hi[None, :] + shift)
        if sine:
            part = np.sin(np.deg2rad(hi)) - np.sin(np.deg2rad(lo))
        else:
            part = hi - lo
        weights += np.where(hi > lo, part, 0.0)
    return weights


class Regridder:
    def __init__(self, src_lat, src_lon, dst_lat, dst_lon):
        self.wlat = _overlap_weights(src_lat, 0.125, dst_lat, 0.75, sine=True)
        self.wlon = _overlap_weights(src_lon, 0.125, dst_lon, 0.75, periodic=True)

    def __call__(self, field):
        """(..., src_lat, src_lon) -> (..., dst_lat, dst_lon), NaN-aware (sea-surface temperature)."""
        valid = np.isfinite(field)
        filled = np.where(valid, field, 0.0)
        num = self.wlat @ filled @ self.wlon.T
        den = self.wlat @ valid.astype(float) @ self.wlon.T
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / den
        full = np.outer(self.wlat.sum(1), self.wlon.sum(1))
        out[den < 0.5 * full] = np.nan                     # mostly land: missing, as in the store
        return out


class ArcoTo15:
    def __init__(self, wb2, cache_dir, threads=32):
        self.threads = threads
        self.arco = xr.open_zarr(ARCO, storage_options={"token": "anon"}, chunks=None)
        self.wb2 = wb2
        self.regrid = Regridder(self.arco.latitude.values, self.arco.longitude.values,
                                wb2.latitude.values, wb2.longitude.values)
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)

    def _read(self, name, when, level=None):
        field = self.arco[name].sel(time=when)
        if level is not None:
            field = field.sel(level=level)
        return np.asarray(field.values, dtype=np.float64)

    def frame(self, when) -> dict[str, np.ndarray]:
        """One 6-hourly frame on the 1.5-degree grid, as (lat, lon) or (level, lat, lon) arrays. Cached."""
        when = np.datetime64(when, "h")
        path = self.cache / f"{str(when).replace(':', '')}.npz"
        if path.exists():
            return dict(np.load(path))
        jobs = {name: (name, when, None) for name in SURFACE}
        jobs.update({name: (name, when, LEVELS) for name in UPPER})
        for target, source in ACCUMULATED.items():
            for hour in range(6):
                jobs[f"{target}#{hour}"] = (source, when - np.timedelta64(hour, "h"), None)
        with cf.ThreadPoolExecutor(self.threads) as pool:
            raw = dict(zip(jobs, pool.map(lambda job: self._read(*job), jobs.values())))
        out = {}
        for name in SURFACE + UPPER:
            out[name] = self.regrid(raw[name]).astype(np.float32)
        for target in ACCUMULATED:
            out[target] = self.regrid(sum(raw[f"{target}#{hour}"] for hour in range(6))).astype(np.float32)
        np.savez(path, **out)
        return out

    def dataset(self, times) -> xr.Dataset:
        """A WeatherBench 2-shaped Dataset (time, [level,] longitude, latitude) for consecutive times."""
        times = np.asarray(times, dtype="datetime64[ns]")
        frames = [self.frame(t) for t in times]
        lat, lon = self.wb2.latitude.values, self.wb2.longitude.values
        data = {}
        for name in SURFACE + list(ACCUMULATED):
            stack = np.stack([f[name] for f in frames])                       # (T, lat, lon)
            data[name] = (("time", "longitude", "latitude"), stack.transpose(0, 2, 1))
        for name in UPPER:
            stack = np.stack([f[name] for f in frames])                       # (T, level, lat, lon)
            data[name] = (("time", "level", "longitude", "latitude"), stack.transpose(0, 1, 3, 2))
        for name in STATIC:
            data[name] = (("longitude", "latitude"), np.asarray(self.wb2[name].values, dtype=np.float32))
        return xr.Dataset(data, coords={"time": times, "level": LEVELS, "longitude": lon, "latitude": lat})
