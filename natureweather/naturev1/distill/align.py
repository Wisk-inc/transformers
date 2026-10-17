# Copyright 2026 Nathan. Apache-2.0.
"""
Putting every teacher into NatureV1's frame: space, vertical, time, names, units -- before a single loss.

The engineering problem of multi-teacher distillation is not the loss; it is that the teachers do not agree
on what a "grid point", a "level" or a "step" is. A logit-matching shortcut is impossible -- the models share no
output space. So every teacher forecast is first *resampled into NatureV1's own analysis space* -- the
1.5-degree, pole-inclusive equiangular grid its encoder reads onto the Fibonacci mesh, the pressure levels of
its channels, its 6-hour step, its channel names, physical units and normalisation -- and only then compared
with NatureV1's rollout. Each step here is the physically correct operation, not the convenient one:

* **Space, structured** (:class:`ConservativeRegrid`): area-weighted overlap of cells (overlap in
  ``sin(latitude)`` times overlap in longitude, separable, so a 0.25-degree global field regrids with two small
  matrix products), NaN-aware -- conservation matters because a teacher's sea-level-pressure minimum and its
  precipitation total must survive the change of grid as cell means, not as point samples. Handles either
  latitude ordering and either longitude convention.
* **Space, unstructured** (:func:`points_to_grid`): reduced Gaussian grids (ECMWF AIFS's N320), icosahedral
  mesh nodes, any point cloud -- each point's value is spread into the target cell it falls in with its own
  area as the weight; empty target cells (finer target than source) are filled by diffusion from their
  neighbours.
* **Vertical** (:func:`to_levels`): linear in log-pressure (the coordinate in which temperature and geopotential
  vary smoothly), never extrapolated -- a NatureV1 level outside a teacher's range is *missing* for that teacher,
  not invented; levels below the ground (pressure above the surface pressure) are masked.
* **Time** (:func:`to_leads`): teacher leads snapped to NatureV1's 6-hourly steps where they coincide,
  interpolated linearly between bracketing steps where they do not (GenCast's 12-hour steps), never
  extrapolated past a teacher's horizon.
* **Names and units** (:func:`canonical_name`, :func:`to_si`): archive quirks stripped (Aurora's
  ``'geopotential\\t'``), GRIB short names mapped (``gh`` geopotential height in metres becomes geopotential,
  ``msl``/``prmsl``, ``2t``, ``10u``...), hPa to Pa, Celsius to Kelvin, relative to specific humidity.
* **Channels** (:func:`to_channels`): everything assembled into NatureV1's ``variable@level`` channel order and
  normalised with NatureV1's own normaliser, with a mask of which channels this teacher actually provides.
"""

from __future__ import annotations

import numpy as np

#: NatureV1's analysis grid.
NATURE_LAT = np.linspace(-90.0, 90.0, 121)
NATURE_LON = np.arange(240) * 1.5

ALIASES = {
    "z": "geopotential", "t": "temperature", "u": "u_component_of_wind", "v": "v_component_of_wind",
    "q": "specific_humidity", "w": "vertical_velocity", "msl": "mean_sea_level_pressure",
    "prmsl": "mean_sea_level_pressure", "2t": "2m_temperature", "t2m": "2m_temperature",
    "10u": "10m_u_component_of_wind", "u10": "10m_u_component_of_wind", "10v": "10m_v_component_of_wind",
    "v10": "10m_v_component_of_wind", "sp": "surface_pressure", "tp": "total_precipitation_6hr",
    "sst": "sea_surface_temperature", "tcwv": "total_column_water_vapour", "gh": "geopotential_height",
    "r": "relative_humidity",
}


def canonical_name(name: str) -> str:
    """Archive or GRIB name -> the WeatherBench 2 / ERA5 long name NatureV1's channels use."""
    clean = name.strip().strip("\t").lower()
    return ALIASES.get(clean, clean)


def to_si(name: str, data: np.ndarray, units: str | None = None) -> tuple[str, np.ndarray]:
    """Convert to the units NatureV1 was trained on; returns ``(canonical name, data)``."""
    name = canonical_name(name)
    u = (units or "").strip().lower()
    data = np.asarray(data, np.float64)
    if name == "geopotential_height" or (name == "geopotential" and u in ("m", "gpm")):
        return "geopotential", data * 9.80665
    if u in ("hpa", "mb", "millibar") and name in ("mean_sea_level_pressure", "surface_pressure"):
        return name, data * 100.0
    if u in ("c", "degc", "celsius") and name in ("temperature", "2m_temperature", "sea_surface_temperature"):
        return name, data + 273.15
    if u in ("g/kg", "g kg-1") and name == "specific_humidity":
        return name, data / 1000.0
    return name, data


def specific_from_relative(rh_percent: np.ndarray, t_k: np.ndarray, p_hpa) -> np.ndarray:
    """Specific humidity (kg/kg) from relative humidity (%) via Bolton's saturation vapour pressure."""
    es = 6.112 * np.exp(17.67 * (t_k - 273.15) / (t_k - 29.65))
    e = np.clip(rh_percent, 0.0, 110.0) / 100.0 * es
    p = np.asarray(p_hpa, np.float64)
    return 0.622 * e / np.maximum(p - 0.378 * e, 1.0)


# ------------------------------------------------------------------------------------------- space --

def cell_edges(centres: np.ndarray, periodic: bool) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper edges of 1-D cells around ``centres`` (ascending); latitude clipped to the poles."""
    c = np.asarray(centres, np.float64)
    mid = 0.5 * (c[1:] + c[:-1])
    if periodic:
        step = 360.0 / c.size if np.allclose(np.diff(c), np.diff(c)[0]) else float(np.median(np.diff(c)))
        lo = np.concatenate([[c[0] - step / 2], mid])
        hi = np.concatenate([mid, [c[-1] + step / 2]])
        return lo, hi
    lo = np.concatenate([[c[0] - (mid[0] - c[0])], mid])
    hi = np.concatenate([mid, [c[-1] + (c[-1] - mid[-1])]])
    return np.clip(lo, -90.0, 90.0), np.clip(hi, -90.0, 90.0)


def overlap_matrix(src: np.ndarray, dst: np.ndarray, periodic: bool, sine: bool) -> np.ndarray:
    """``(dst, src)`` overlap of 1-D cells, rows normalised -- latitude overlaps measured in sin(latitude)."""
    s_lo, s_hi = cell_edges(src, periodic)
    d_lo, d_hi = cell_edges(dst, periodic)
    weights = np.zeros((dst.size, src.size))
    for shift in ((-360.0, 0.0, 360.0) if periodic else (0.0,)):
        lo = np.maximum(d_lo[:, None], s_lo[None, :] + shift)
        hi = np.minimum(d_hi[:, None], s_hi[None, :] + shift)
        part = (np.sin(np.radians(hi)) - np.sin(np.radians(lo))) if sine else (hi - lo)
        weights += np.where(hi > lo, part, 0.0)
    return weights


class ConservativeRegrid:
    """Area-conservative regridding between any two regular latitude-longitude grids (separable)."""

    def __init__(self, src_lat, src_lon, dst_lat=NATURE_LAT, dst_lon=NATURE_LON) -> None:
        src_lat, src_lon = np.asarray(src_lat, np.float64), np.asarray(src_lon, np.float64) % 360.0
        self.flip = src_lat[0] > src_lat[-1]
        self.lon_order = np.argsort(src_lon)
        lat = src_lat[::-1] if self.flip else src_lat
        self.wlat = overlap_matrix(lat, np.asarray(dst_lat, np.float64), periodic=False, sine=True)
        self.wlon = overlap_matrix(src_lon[self.lon_order], np.asarray(dst_lon, np.float64) % 360.0, periodic=True, sine=False)
        self.full = np.outer(self.wlat.sum(1), self.wlon.sum(1))

    def __call__(self, field: np.ndarray, min_coverage: float = 0.5) -> np.ndarray:
        """``(..., src_lat, src_lon)`` -> ``(..., dst_lat, dst_lon)``; NaN where under half the cell is observed."""
        x = np.asarray(field, np.float64)
        if self.flip:
            x = x[..., ::-1, :]
        x = x[..., self.lon_order]
        valid = np.isfinite(x)
        num = self.wlat @ np.where(valid, x, 0.0) @ self.wlon.T
        den = self.wlat @ valid.astype(np.float64) @ self.wlon.T
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / den
        return np.where(den >= min_coverage * self.full, out, np.nan)


def points_to_grid(values: np.ndarray, lat: np.ndarray, lon: np.ndarray, weights: np.ndarray | None = None,
                   dst_lat=NATURE_LAT, dst_lon=NATURE_LON, fill_passes: int = 50) -> np.ndarray:
    """
    Unstructured points (reduced Gaussian, icosahedral nodes...) to a regular grid: each point's value goes to the
    target cell containing it, weighted by its area (``weights``; default ``cos(lat)``). ``values``: ``(..., N)``.
    Target cells no point falls in are filled by repeated neighbour averaging (periodic in longitude).
    """
    values = np.asarray(values, np.float64)
    lat = np.asarray(lat, np.float64)
    lon = np.asarray(lon, np.float64) % 360.0
    w = np.cos(np.radians(lat)) if weights is None else np.asarray(weights, np.float64)
    lat_lo, lat_hi = cell_edges(np.asarray(dst_lat, np.float64), periodic=False)
    lon_lo, _ = cell_edges(np.asarray(dst_lon, np.float64) % 360.0, periodic=True)
    i = np.clip(np.searchsorted(lat_hi, lat), 0, len(dst_lat) - 1)
    step = 360.0 / len(dst_lon)
    j = np.floor(((lon - lon_lo[0]) % 360.0) / step).astype(int) % len(dst_lon)
    cell = i * len(dst_lon) + j
    size = len(dst_lat) * len(dst_lon)
    lead_shape = values.shape[:-1]
    flat = values.reshape(-1, values.shape[-1])
    out = np.empty((flat.shape[0], size))
    for r in range(flat.shape[0]):
        ok = np.isfinite(flat[r])
        num = np.bincount(cell[ok], weights=(flat[r] * w)[ok], minlength=size)
        den = np.bincount(cell[ok], weights=w[ok], minlength=size)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[r] = np.where(den > 0, num / den, np.nan)
    grid = out.reshape(*lead_shape, len(dst_lat), len(dst_lon))
    for _ in range(fill_passes):
        missing = ~np.isfinite(grid)
        if not missing.any():
            break
        padded = np.concatenate([grid[..., :, -1:], grid, grid[..., :, :1]], -1)
        padded = np.concatenate([padded[..., :1, :], padded, padded[..., -1:, :]], -2)
        stack = np.stack([padded[..., :-2, 1:-1], padded[..., 2:, 1:-1], padded[..., 1:-1, :-2], padded[..., 1:-1, 2:]])
        finite = np.isfinite(stack)
        count = finite.sum(0)
        mean = np.where(count > 0, np.where(finite, stack, 0.0).sum(0) / np.maximum(count, 1), np.nan)
        grid = np.where(missing & np.isfinite(mean), mean, grid)
    return grid


def reduced_gaussian_points(points_per_row: list[int], latitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Latitudes, longitudes and areas of a reduced Gaussian grid's points (row by row, starting at longitude 0)."""
    lat, lon, area = [], [], []
    for row, (n, phi) in enumerate(zip(points_per_row, latitudes)):
        lon.append(np.arange(n) * 360.0 / n)
        lat.append(np.full(n, phi))
        area.append(np.full(n, np.cos(np.radians(phi)) * 360.0 / n))
    return np.concatenate(lat), np.concatenate(lon), np.concatenate(area)


# ---------------------------------------------------------------------------------------- vertical --

def to_levels(data: np.ndarray, src_levels, dst_levels, surface_pressure_hpa: np.ndarray | None = None) -> np.ndarray:
    """
    ``(L_src, ...)`` -> ``(L_dst, ...)`` linear in log-pressure. Destination levels outside the source range are
    NaN (never extrapolated); with ``surface_pressure_hpa`` levels below the ground are NaN too.
    """
    src = np.asarray(src_levels, np.float64)
    order = np.argsort(src)
    src, data = src[order], np.asarray(data, np.float64)[order]
    logs = np.log(src)
    out = np.full((len(dst_levels),) + data.shape[1:], np.nan)
    for k, p in enumerate(dst_levels):
        if p < src[0] - 1e-9 or p > src[-1] + 1e-9:
            continue
        j = int(np.clip(np.searchsorted(src, p) - 1, 0, len(src) - 2))
        if abs(p - src[j + 1]) < 1e-9:
            out[k] = data[j + 1]
            continue
        w = (np.log(p) - logs[j]) / (logs[j + 1] - logs[j])
        out[k] = (1.0 - w) * data[j] + w * data[j + 1]
        if surface_pressure_hpa is not None:
            out[k] = np.where(np.asarray(surface_pressure_hpa) < p, np.nan, out[k])
    return out


# -------------------------------------------------------------------------------------------- time --

def to_leads(data: np.ndarray, src_leads_h, dst_leads_h) -> tuple[np.ndarray, np.ndarray]:
    """
    ``(S_src, ...)`` -> ``(S_dst, ...)``: exact where a teacher lead coincides, linear between bracketing leads,
    NaN beyond the teacher's range. Returns ``(data, exact)`` with ``exact`` a ``(S_dst,)`` flag.
    """
    src = np.asarray(src_leads_h, np.float64)
    data = np.asarray(data, np.float64)
    out = np.full((len(dst_leads_h),) + data.shape[1:], np.nan)
    exact = np.zeros(len(dst_leads_h), bool)
    for k, h in enumerate(dst_leads_h):
        hit = np.nonzero(np.abs(src - h) < 1e-6)[0]
        if hit.size:
            out[k], exact[k] = data[hit[0]], True
            continue
        if h < src.min() or h > src.max():
            continue
        j = int(np.searchsorted(src, h) - 1)
        w = (h - src[j]) / (src[j + 1] - src[j])
        out[k] = (1.0 - w) * data[j] + w * data[j + 1]
    return out, exact


# ---------------------------------------------------------------------------------------- channels --

def to_channels(fields: dict, channels: list[str], normalizer=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble aligned fields into NatureV1's channel layout.

    Args:
        fields: ``{canonical name: (S, H, W)}`` for single-level and ``{canonical name: {level: (S, H, W)}}``
            for pressure-level variables, already on NatureV1's grid, levels and leads.
        channels: NatureV1's channel names (``variable`` or ``variable@level``).
        normalizer: NatureV1's :class:`naturev1.era5.Normalizer` (its ``variables`` must be ``channels``); when
            given, the output is normalised (NaN kept where missing).

    Returns:
        ``(values (S, H, W, C), provided (C,) bool)``.
    """
    sample = next(iter(v if not isinstance(v, dict) else next(iter(v.values())) for v in fields.values()))
    shape = sample.shape
    out = np.full(shape + (len(channels),), np.nan, np.float64)
    provided = np.zeros(len(channels), bool)
    for c, name in enumerate(channels):
        base, _, level = name.partition("@")
        entry = fields.get(base)
        if entry is None:
            continue
        if level:
            if isinstance(entry, dict) and int(level) in entry:
                out[..., c] = entry[int(level)]
                provided[c] = np.isfinite(entry[int(level)]).any()
        elif not isinstance(entry, dict):
            out[..., c] = entry
            provided[c] = np.isfinite(entry).any()
    if normalizer is not None:
        out = normalizer.prepare(out.astype(np.float32))
    return out, provided


__all__ = ["ALIASES", "ConservativeRegrid", "NATURE_LAT", "NATURE_LON", "canonical_name", "cell_edges",
           "overlap_matrix", "points_to_grid", "reduced_gaussian_points", "specific_from_relative", "to_channels",
           "to_leads", "to_levels", "to_si"]
