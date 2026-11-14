# Copyright 2026 Nathan. Apache-2.0.
"""
A synthetic Caribbean for tests and demonstrations: an island with a mountain spine, trade winds, afternoon storms,
cold fronts that sweep south in the cool season, and rain made where the wind meets the slope -- with "reanalysis"
fields and "CMORPH" truth produced by the same weather, so the whole pipeline can be trained and scored without a
network. A model that learns anything should beat this world's climatology; the tests check that it does.
"""

from __future__ import annotations

import numpy as np

from .domain import Geography
from .features import cos_zenith, local_solar_hour
from .util import calendar, hour_of, year_hours
from .weather import code, rate_class

BOX = (15.0, 21.5, -81.0, -73.5)


def synthetic_geography(verbose: bool = False) -> Geography:
    from naturev1.swarm.geography import synthetic_island

    region = synthetic_island(lat0=18.1, lon0=-77.3, step=0.03, parishes=4, peak_m=2000.0)
    return Geography.build(region=region, verbose=verbose, mesh_box=BOX, target_box=(16.0, 20.5, -80.0, -74.5),
                           ocean_cells=150, land_step=0.15, coast_step=0.3, ocean_step=1.0)


class SyntheticWorld:
    """The weather of the synthetic Caribbean, hour by hour, deterministic given ``seed``. Provides the caches'
    interfaces: ``hourly``/``upper`` (ERA5), ``targets``/``mesh`` (CMORPH), and stations via :meth:`station_cache`."""

    def __init__(self, geo: Geography, start: str = "2020-10-01T00", days: int = 120, seed: int = 0) -> None:
        self.geo = geo
        self.first = hour_of(start)
        self.end = self.first + 24 * days
        self.rng = np.random.default_rng(seed)
        s, n, w, e = BOX
        self.lat1, self.lon1 = np.arange(s, n + 1e-9, 1.0), np.arange(w, e + 1e-9, 1.0)
        self.lat2, self.lon2 = np.arange(s - 1.5, n + 1.6, 1.5), np.arange(w - 1.5, e + 1.6, 1.5)
        self.years = sorted({int(calendar(np.array([h]))["year"][0]) for h in (self.first, self.end - 1)})
        # fronts: arrival times at 21.5 N, every 4-7 days from November to March
        t, fronts = self.first, []
        while t < self.end:
            t += int(self.rng.integers(96, 168))
            month = int(calendar(np.array([t]))["month"][0])
            if month in (11, 12, 1, 2, 3):
                fronts.append((t, float(self.rng.uniform(12.0, 22.0))))   # (time at 21.5 N, speed km/h)
        self.fronts = fronts
        self.daily_moisture = self.rng.uniform(0.6, 1.4, size=days + 4)
        self.n_targets = len(geo.targets)
        self.station_lat = np.array([17.94, 18.50, 18.20, 18.40, 17.98])
        self.station_lon = np.array([-76.79, -77.92, -76.60, -77.10, -77.50])
        self._cache = {}

    # -------------------------------------------------------------------------------------- weather --
    def front_lat(self, hour: float) -> list[tuple[float, float]]:
        """Latitude of each active front at ``hour`` and its strength (fronts decay south of 16 N)."""
        out = []
        for t, speed in self.fronts:
            lat = 21.5 - (hour - t) * speed / 111.0
            if 14.0 < lat < 26.0:
                out.append((lat, float(np.clip((lat - 15.0) / 4.0, 0.0, 1.0))))
        return out

    def rain(self, hour: int, lat: np.ndarray, lon: np.ndarray, land: np.ndarray, elev: np.ndarray,
             slope_e: np.ndarray) -> np.ndarray:
        """Rain (mm/h) in ``hour`` at points."""
        key = (int(hour), lat.size, float(lat[0]), float(lon[0]))
        if key in self._cache:
            return self._cache[key]
        rng = np.random.default_rng(int(hour) * 7919 + lat.size)
        moist = self.daily_moisture[int((hour - self.first) // 24) % self.daily_moisture.size]
        out = np.zeros(lat.size)
        for flat, strength in self.front_lat(hour + 0.5):
            band = np.exp(-0.5 * ((lat - flat - 0.15) / 0.35) ** 2) * strength
            out += 5.0 * band * (rng.random(lat.size) < 0.75) * moist
        lst = local_solar_hour(lon, hour)
        afternoon = np.exp(-0.5 * ((lst - 15.0) / 2.0) ** 2)
        p_conv = land * afternoon * (0.25 + 0.5 * np.clip(elev / 1500.0, 0, 1)) * moist
        cells = rng.random(lat.size) < p_conv
        out += cells * rng.gamma(2.0, 3.0, lat.size)
        trade = 1.0 if not self.front_lat(hour + 0.5) else 0.3
        upslope = np.clip(-slope_e, 0, None)          # east-facing slopes face the trade wind
        out += land * trade * (rng.random(lat.size) < 0.12 + 0.6 * np.clip(upslope * 3, 0, 1)) * 1.5 * moist
        out += (rng.random(lat.size) < 0.02) * rng.gamma(1.0, 1.0, lat.size)
        out = out.astype(np.float32)
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[key] = out
        return out

    def _points(self, which: str):
        if which == "targets":
            p = self.geo.targets
        else:
            p = self.geo.mesh
        land = (p.kind == 0).astype(np.float64)
        return p.lat, p.lon, land, p.statics[:, 0] * 1000.0, p.statics[:, 3]

    # ------------------------------------------------------------------------- the caches' interfaces --
    def targets(self, hours, columns=None) -> np.ndarray:
        lat, lon, land, elev, slope = self._points("targets")
        if columns is not None:
            lat, lon, land, elev, slope = lat[columns], lon[columns], land[columns], elev[columns], slope[columns]
        out = np.full((len(hours), lat.size), np.nan, np.float32)
        for k, h in enumerate(hours):
            if self.first <= h < self.end:
                out[k] = self.rain(int(h), lat, lon, land, elev, slope)
        return out

    def mesh(self, hours) -> np.ndarray:
        lat, lon, land, elev, slope = self._points("mesh")
        out = np.full((len(hours), lat.size), np.nan, np.float32)
        for k, h in enumerate(hours):
            if self.first <= h < self.end:
                out[k] = self.rain(int(h), lat, lon, land, elev, slope)
        return out

    def hourly(self, short: str, hours) -> np.ndarray:
        hours = np.asarray(hours)
        la, lo = np.meshgrid(self.lat1, self.lon1, indexing="ij")
        out = np.full((hours.size,) + la.shape, np.nan, np.float32)
        for k, h in enumerate(hours):
            if not self.first <= h < self.end:
                continue
            fronts = self.front_lat(h + 0.5)
            behind = np.zeros(la.shape)
            near = np.zeros(la.shape)
            for flat, strength in fronts:
                behind = np.maximum(behind, (la > flat).astype(float) * strength)
                near = np.maximum(near, np.exp(-0.5 * ((la - flat - 0.15) / 0.6) ** 2) * strength)
            moist = self.daily_moisture[int((h - self.first) // 24) % self.daily_moisture.size]
            if short == "tp":
                land_frac = np.exp(-(((la - 18.1) / 0.6) ** 2 + ((lo + 77.3) / 1.5) ** 2))
                lst = local_solar_hour(lo, h)
                val = 1.2 * near * moist + land_frac * np.exp(-0.5 * ((lst - 15) / 2.5) ** 2) * moist * 1.5 + 0.05
            elif short == "tcwv":
                val = 45 * moist - 15 * behind + 5 * near
            elif short == "u10":
                val = -7.0 + 2.0 * behind
            elif short == "v10":
                val = 0.5 - 6.0 * behind
            elif short == "t2m":
                val = 300.0 - 3.0 * behind + 2.0 * cos_zenith(la, lo, h)
            elif short == "msl":
                val = 101300.0 + 400.0 * behind
            else:
                val = np.clip(0.3 + 0.6 * near, 0, 1)
            out[k] = val
        return out

    def upper(self, name: str, hours) -> np.ndarray:
        hours = np.asarray(hours, np.float64)
        la, lo = np.meshgrid(self.lat2, self.lon2, indexing="ij")
        out = np.full((hours.size,) + la.shape, np.nan, np.float32)
        level = int(name[1:])
        for k, h in enumerate(hours):
            if not self.first <= h < self.end:
                continue
            behind = np.zeros(la.shape)
            for flat, strength in self.front_lat(h + 0.5):
                behind = np.maximum(behind, 1.0 / (1.0 + np.exp(-(la - flat) / 0.3)) * strength)
            if name[0] == "t":
                val = {850: 291.0, 700: 283.0, 500: 268.0, 200: 221.0}[level] - 4.0 * behind
            elif name[0] == "q":
                val = {850: 0.013, 700: 0.007, 500: 0.003, 200: 0.0001}[level] * (1 - 0.5 * behind)
            elif name[0] == "u":
                val = -8.0 + 3.0 * behind
            else:
                val = 0.5 - 5.0 * behind
            out[k] = val
        return out

    # --------------------------------------------------------------------------------------- stations --
    def station_cache(self):
        world = self

        class Stations:
            lat, lon = world.station_lat, world.station_lon

            def __len__(self):
                return world.station_lat.size

            def read(self, hours):
                stat = world.geo.terrain.descriptors(world.station_lat, world.station_lon)
                land = np.ones(world.station_lat.size)
                out = np.full((len(hours), world.station_lat.size), -1, np.int8)
                for k, h in enumerate(hours):
                    if world.first <= h < world.end:
                        r = world.rain(int(h), world.station_lat, world.station_lon, land, stat[:, 0] * 1000.0,
                                       stat[:, 3])
                        cls = rate_class(r)
                        lst = local_solar_hour(world.station_lon, h)
                        thunder = (r >= 7.6) & (np.abs(lst - 15.0) < 4.0)
                        kind = np.where(r < 0.5, 1, np.where(r >= 2.5, 3, 2))
                        out[k] = [code(int(c), bool(t), int(q) if c > 0 else 0) for c, t, q in zip(cls, thunder, kind)]
                return out

        return Stations()

    def hours(self) -> np.ndarray:
        return np.arange(self.first, self.end)


__all__ = ["BOX", "SyntheticWorld", "synthetic_geography", "year_hours"]
