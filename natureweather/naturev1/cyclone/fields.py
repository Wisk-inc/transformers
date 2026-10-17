# Copyright 2026 Nathan. Apache-2.0.
"""
The atmosphere around one storm: a latitude-longitude box of analysis fields with spherical calculus.

The idea: every physical diagnostic a track model needs -- layer-mean winds, vorticity and its gradient,
divergence, the Laplacian for inverting vorticity -- is computed with the sphere's metric terms (the
``1 / (a cos(lat))`` on every east-west derivative, the ``cos(lat)`` inside the curl and the divergence),
not as finite differences in degrees. The box is stored with continuous (unwrapped) longitudes, so a storm
box straddling the dateline is just a box. Sampling between grid points is bilinear in the box's own
coordinates, with any longitude mapped into the box first; points outside the box clamp to its edge rather
than wrapping onto the far side of the planet.

Layer means are pressure-weighted (trapezoid in pressure), the vertical average a steering-layer argument
actually calls for; levels between the stored ones interpolate in log-pressure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import EARTH_RADIUS_KM, coriolis, to_local

A_M = EARTH_RADIUS_KM * 1000.0

#: Variables stored on pressure levels, and single-level ones.
LEVEL_FIELDS = ("u", "v", "z", "t", "q")
SURFACE_FIELDS = ("msl", "sst", "tcwv", "lsm", "zs")


@dataclass
class Environment:
    """
    Analysis fields in a box around a storm.

    Attributes:
        lat: ``(H,)`` ascending latitudes, degrees.
        lon: ``(W,)`` ascending, continuous longitudes (may run past 180), degrees.
        levels: pressure levels, hPa, in the order the level fields are stored.
        fields: ``name -> (L, H, W)`` for level fields, ``(H, W)`` for single-level ones. SI units (m/s,
            m^2/s^2 geopotential, K, kg/kg, Pa, 0-1 land fraction).
        previous: the same kind of mapping 12 hours earlier, for the fields that were stored (u, v, z).
        time: the analysis time.
    """

    lat: np.ndarray
    lon: np.ndarray
    levels: tuple[int, ...]
    fields: dict[str, np.ndarray]
    previous: dict[str, np.ndarray] = field(default_factory=dict)
    time: np.datetime64 | None = None

    # ------------------------------------------------------------------------------- access --

    @property
    def shape(self) -> tuple[int, int]:
        return self.lat.size, self.lon.size

    @property
    def spacing_deg(self) -> float:
        return float(self.lat[1] - self.lat[0])

    def has(self, name: str) -> bool:
        return name in self.fields

    def level(self, name: str, pressure: float, previous: bool = False) -> np.ndarray:
        """A level field at ``pressure`` hPa, interpolated in log-pressure between stored levels."""
        source = self.previous if previous else self.fields
        data = source[name]
        levels = np.asarray(self.levels if data.shape[0] == len(self.levels) else self.previous_levels, np.float64)
        if pressure in levels:
            return data[int(np.nonzero(levels == pressure)[0][0])]
        order = np.argsort(levels)
        lv = levels[order]
        p = float(np.clip(pressure, lv[0], lv[-1]))
        j = int(np.clip(np.searchsorted(lv, p) - 1, 0, len(lv) - 2))
        w = (np.log(p) - np.log(lv[j])) / (np.log(lv[j + 1]) - np.log(lv[j]))
        return (1.0 - w) * data[order[j]] + w * data[order[j + 1]]

    #: Levels of the stored previous-time level fields (set by the loader when they differ).
    previous_levels: tuple[int, ...] = ()

    def layer_mean(self, name: str, top: float, bottom: float, previous: bool = False,
                   weights: dict[float, float] | None = None) -> np.ndarray:
        """
        Pressure-weighted mean of a level field between ``top`` and ``bottom`` hPa (trapezoid in pressure),
        or a weighted mean over explicit ``weights`` ``{level: weight}`` when given.
        """
        if weights is not None:
            total = sum(weights.values())
            return sum(w * self.level(name, p, previous) for p, w in weights.items()) / total
        source = self.previous if previous else self.fields
        levels = np.asarray(self.levels if source[name].shape[0] == len(self.levels) else self.previous_levels,
                            np.float64)
        inside = sorted({float(p) for p in levels if top <= p <= bottom} | {float(top), float(bottom)})
        values = [self.level(name, p, previous) for p in inside]
        total = np.zeros_like(values[0], dtype=np.float64)
        span = 0.0
        for k in range(len(inside) - 1):
            dp = inside[k + 1] - inside[k]
            total = total + 0.5 * dp * (values[k] + values[k + 1])
            span += dp
        return total / max(span, 1e-9)

    # ----------------------------------------------------------------------------- sampling --

    def _fractional_index(self, lat, lon):
        lat = np.asarray(lat, np.float64)
        lon = np.asarray(lon, np.float64)
        step = self.lat[1] - self.lat[0]
        fi = (lat - self.lat[0]) / step
        offset = (lon - self.lon[0]) % 360.0
        fj = offset / (self.lon[1] - self.lon[0])
        fi = np.clip(fi, 0.0, self.lat.size - 1.000001)
        fj = np.clip(fj, 0.0, self.lon.size - 1.000001)
        return fi, fj

    def sample(self, grid: np.ndarray, lat, lon) -> np.ndarray:
        """Bilinear value of a ``(H, W)`` field at points (degrees); outside the box clamps to its edge."""
        fi, fj = self._fractional_index(lat, lon)
        i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
        di, dj = fi - i0, fj - j0
        return ((1 - di) * (1 - dj) * grid[i0, j0] + (1 - di) * dj * grid[i0, j0 + 1]
                + di * (1 - dj) * grid[i0 + 1, j0] + di * dj * grid[i0 + 1, j0 + 1])

    def inside(self, lat, lon, margin_deg: float = 0.0):
        """Whether points lie inside the box (with a margin)."""
        lat = np.asarray(lat, np.float64)
        offset = (np.asarray(lon, np.float64) - self.lon[0]) % 360.0
        return ((lat >= self.lat[0] + margin_deg) & (lat <= self.lat[-1] - margin_deg)
                & (offset >= margin_deg) & (offset <= self.lon[-1] - self.lon[0] - margin_deg))

    # ---------------------------------------------------------------------------- calculus --

    @property
    def coslat(self) -> np.ndarray:
        return np.cos(np.radians(self.lat))[:, None]

    def ddx(self, f: np.ndarray) -> np.ndarray:
        """East derivative, per metre: (1 / (a cos(lat))) d/dlambda."""
        dl = np.radians(self.lon[1] - self.lon[0])
        g = np.gradient(f, dl, axis=-1)
        return g / (A_M * np.maximum(self.coslat, 1e-3))

    def ddy(self, f: np.ndarray) -> np.ndarray:
        """North derivative, per metre: (1 / a) d/dphi."""
        dp = np.radians(self.lat[1] - self.lat[0])
        return np.gradient(f, dp, axis=-2) / A_M

    def vorticity(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Relative vorticity on the sphere: (1 / (a cos)) [dv/dlambda - d(u cos)/dphi]."""
        dl = np.radians(self.lon[1] - self.lon[0])
        dp = np.radians(self.lat[1] - self.lat[0])
        c = np.maximum(self.coslat, 1e-3)
        return (np.gradient(v, dl, axis=-1) - np.gradient(u * c, dp, axis=-2)) / (A_M * c)

    def divergence(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Horizontal divergence on the sphere: (1 / (a cos)) [du/dlambda + d(v cos)/dphi]."""
        dl = np.radians(self.lon[1] - self.lon[0])
        dp = np.radians(self.lat[1] - self.lat[0])
        c = np.maximum(self.coslat, 1e-3)
        return (np.gradient(u, dl, axis=-1) + np.gradient(v * c, dp, axis=-2)) / (A_M * c)

    def laplacian(self, f: np.ndarray) -> np.ndarray:
        """Spherical Laplacian: div(grad f)."""
        return self.divergence(self.ddx(f), self.ddy(f))

    def coriolis(self) -> np.ndarray:
        """f on the box, ``(H, 1)``."""
        return coriolis(self.lat)[:, None]

    def absolute_vorticity(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.vorticity(u, v) + self.coriolis()

    def advect(self, u: np.ndarray, v: np.ndarray, f: np.ndarray) -> np.ndarray:
        """-(u . grad) f."""
        return -(u * self.ddx(f) + v * self.ddy(f))

    # ------------------------------------------------------------------------------ frames --

    def local_coordinates(self, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
        """``(x, y)`` km of every grid point in the azimuthal-equidistant frame centred on (lat0, lon0)."""
        lat_g, lon_g = np.meshgrid(self.lat, self.lon, indexing="ij")
        return to_local(lat0, lon0, lat_g, lon_g)

    def distance_km(self, lat0: float, lon0: float) -> np.ndarray:
        x, y = self.local_coordinates(lat0, lon0)
        return np.hypot(x, y)

    def cell_area_m2(self) -> np.ndarray:
        """Area of each grid cell, ``(H, W)``."""
        dl = np.radians(self.lon[1] - self.lon[0])
        dp = np.radians(self.lat[1] - self.lat[0])
        return np.broadcast_to(A_M ** 2 * dl * dp * self.coslat, self.shape).copy()


def synthetic_environment(lat0: float = 20.0, lon0: float = -60.0, half_lat: float = 30.0, half_lon: float = 45.0,
                          step: float = 1.5, levels=(1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50),
                          u=None, v=None) -> Environment:
    """
    A constructed environment for tests and scenarios: uniform or functional winds ``u(lat, lon, p)`` and
    ``v(lat, lon, p)`` (callables or constants), a standard-atmosphere column and a warm ocean.
    """
    lat = np.arange(lat0 - half_lat, lat0 + half_lat + 1e-9, step)
    lon = np.arange(lon0 - half_lon, lon0 + half_lon + 1e-9, step)
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    levels = tuple(int(p) for p in levels)
    shape = (len(levels), lat.size, lon.size)

    def build(spec):
        if spec is None:
            return np.zeros(shape)
        if callable(spec):
            return np.stack([np.broadcast_to(spec(lat_g, lon_g, p), lat_g.shape) for p in levels]).astype(np.float64)
        return np.full(shape, float(spec))

    p = np.asarray(levels, np.float64)
    t_col = np.clip(288.15 * (p / 1013.25) ** 0.190263, 210.0, None)
    z_col = 9.80665 * 44330.8 * (1 - (p / 1013.25) ** 0.190263)
    q_col = 0.018 * (p / 1000.0) ** 3
    fields = {"u": build(u), "v": build(v),
              "z": np.broadcast_to(z_col[:, None, None], shape).copy(),
              "t": np.broadcast_to(t_col[:, None, None], shape).copy(),
              "q": np.broadcast_to(q_col[:, None, None], shape).copy(),
              "msl": np.full(lat_g.shape, 101300.0), "sst": np.full(lat_g.shape, 302.0),
              "tcwv": np.full(lat_g.shape, 50.0), "lsm": np.zeros(lat_g.shape), "zs": np.zeros(lat_g.shape)}
    return Environment(lat=lat, lon=lon, levels=levels, fields=fields)


__all__ = ["Environment", "LEVEL_FIELDS", "SURFACE_FIELDS", "synthetic_environment"]
