# Copyright 2026 Nathan. Apache-2.0.
"""
Training examples: one forecast issued at hour ``t0``, built from the staged caches.

Each example is what the model would have had at ``t0`` and what then happened:

* **driving fields** for every frame of :mod:`pluvium.schedule` -- ERA5 in training (perfect prognosis; with the
  lead-growing errors of :func:`pluvium.features.emulate_forecast_error` on most examples), GFS/ECMWF live -- as the
  14 coarse hourly channels and 16 upper-air channels;
* **rain already falling** in the three hours before ``t0`` (CMORPH around every mesh point, and at each target the
  hour before ``t0``), withheld on a share of examples so the model also works without it;
* **per-point timing context** for every frame: sun, season, island-heating potential, the driving model's own rain
  at the point, and the climatological chance of rain for that month and local hour;
* **truth** for hours 0-71: CMORPH at a subset of the 8 km target cells (every land cell, a sample of the sea) and
  wet/dry at every reporting airport, and the weak rain-cause labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import (
    HOURLY_SHORT,
    LEVELS,
    UPPER_SHORT,
    emulate_forecast_error,
    heating_potential,
    hourly_channels,
    regime_flags,
    regime_labels,
    time_features,
    upper_channels,
)
from .schedule import DEFAULT, Schedule
from .util import calendar

WET_MM = 0.2          # a wet hour: at least 0.2 mm
#: per-point inputs of every frame: version 1's thirteen, then the advected rain of :mod:`pluvium.advection` (v2)
TARGET_BASE = ("cos_zenith", "lst_sin", "lst_cos", "doy_sin", "doy_cos", "cold_season", "heat_1h", "heat_3h",
               "driving_rain", "clim_wet", "clim_rate", "memory_rain", "memory_known")
TARGET_DYNAMIC = TARGET_BASE + ("advected_rain", "advected_known")
#: upper-air fields regridded to the 1-degree grid for the hourly channels (fronts, regimes, convection)
UPPER_ON_HOURLY = ("t850", "q850", "u700", "v700", "u850", "v850", "t700", "q700", "t500", "q500", "u200", "v200")
MESH_DYNAMIC = ("memory_rain", "memory_known", "cos_zenith", "lst_sin", "lst_cos", "doy_sin", "doy_cos", "cold_season",
                "heat_1h", "heat_3h")


def bilinear(field: np.ndarray, lat_axis: np.ndarray, lon_axis: np.ndarray, lat, lon) -> np.ndarray:
    """Sample ``(..., H, W)`` regular-grid fields at points (edge-clamped); returns ``(..., N)``."""
    from scipy import ndimage

    i = np.clip((np.asarray(lat) - lat_axis[0]) / (lat_axis[1] - lat_axis[0]), 0, lat_axis.size - 1)
    j = np.clip((np.asarray(lon) - lon_axis[0]) / (lon_axis[1] - lon_axis[0]), 0, lon_axis.size - 1)
    lead = field.shape[:-2]
    flat = field.reshape((-1,) + field.shape[-2:])
    out = np.stack([ndimage.map_coordinates(np.nan_to_num(f), [i, j], order=1, mode="nearest") for f in flat])
    return out.reshape(lead + (np.size(lat),)).astype(np.float32)


def regrid(field: np.ndarray, src_lat, src_lon, dst_lat, dst_lon) -> np.ndarray:
    """``(..., h, w)`` on one regular grid -> ``(..., H, W)`` on another, bilinear."""
    la, lo = np.meshgrid(dst_lat, dst_lon, indexing="ij")
    return bilinear(field, src_lat, src_lon, la.ravel(), lo.ravel()).reshape(field.shape[:-2] + la.shape)


@dataclass
class Climatology:
    """Hourly chance of a wet hour and mean rate by month and local solar hour, per point (from training years)."""

    wet: np.ndarray                  # (12, 24, N) chance of at least 0.2 mm in the hour
    rate: np.ndarray                 # (12, 24, N) mean rate, mm/h
    moderate: np.ndarray | None = None   # (12, 24, N) chance of at least 2.5 mm/h
    heavy: np.ndarray | None = None      # (12, 24, N) chance of at least 7.6 mm/h

    def __post_init__(self) -> None:
        if self.moderate is None:
            self.moderate = (0.3 * self.wet).astype(np.float32)
        if self.heavy is None:
            self.heavy = (0.08 * self.wet).astype(np.float32)

    def _index(self, hours, lon):
        month = calendar(hours)["month"] - 1
        lst = ((np.asarray(hours)[:, None] + 0.5 + lon[None, :] / 15.0) % 24).astype(np.int64)
        return month[:, None], lst, np.arange(lon.size)[None, :]

    def at(self, hours: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(F, N)`` climatological wet chance and rate for frame hours ``(F,)`` at points with longitudes ``(N,)``."""
        i = self._index(hours, lon)
        return self.wet[i], self.rate[i]

    def classes_at(self, hours: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """``(F, N, 4)`` climatological chances of dry, light, moderate and heavy."""
        i = self._index(hours, lon)
        wet, mod, heavy = self.wet[i], np.minimum(self.moderate[i], self.wet[i]), self.heavy[i]
        heavy = np.minimum(heavy, mod)
        return np.stack([1 - wet, wet - mod, mod - heavy, heavy], -1)

    def subset(self, index) -> Climatology:
        return Climatology(self.wet[:, :, index], self.rate[:, :, index], self.moderate[:, :, index],
                           self.heavy[:, :, index])

    def save(self, path) -> None:
        np.savez_compressed(path, wet=self.wet.astype(np.float16), rate=self.rate.astype(np.float16),
                            moderate=self.moderate.astype(np.float16), heavy=self.heavy.astype(np.float16))

    @classmethod
    def load(cls, path) -> Climatology:
        d = np.load(path)
        get = lambda k: d[k].astype(np.float32) if k in d.files else None   # noqa: E731
        return cls(get("wet"), get("rate"), get("moderate"), get("heavy"))

    @classmethod
    def flat(cls, n: int, wet: float = 0.08, rate: float = 0.15) -> Climatology:
        return cls(np.full((12, 24, n), wet, np.float32), np.full((12, 24, n), rate, np.float32))


def compute_climatology(read, lon: np.ndarray, hours: np.ndarray, chunk: int = 744) -> Climatology:
    """From ``read(hours) -> (len(hours), N)`` mm/h (NaN missing) over training ``hours``: per month and local hour,
    the share of wet hours and the mean rate (smoothed over neighbouring hours)."""
    n = lon.size
    wet = np.zeros((12, 24, n))
    rate = np.zeros((12, 24, n))
    mod = np.zeros((12, 24, n))
    heavy = np.zeros((12, 24, n))
    count = np.zeros((12, 24, n))
    idx = np.arange(n)[None, :]
    for start in range(0, hours.size, chunk):
        h = hours[start: start + chunk]
        x = read(h)
        ok = np.isfinite(x)
        month = calendar(h)["month"] - 1
        lst = ((h[:, None] + 0.5 + lon[None, :] / 15.0) % 24).astype(np.int64)
        m = np.broadcast_to(month[:, None], lst.shape)
        where = (m[ok], lst[ok], np.broadcast_to(idx, lst.shape)[ok])
        np.add.at(wet, where, (x[ok] >= WET_MM).astype(np.float64))
        np.add.at(mod, where, (x[ok] >= 2.5).astype(np.float64))
        np.add.at(heavy, where, (x[ok] >= 7.6).astype(np.float64))
        np.add.at(rate, where, x[ok])
        np.add.at(count, where, 1.0)

    def smooth(a):
        return (np.roll(a, 1, axis=1) + 2 * a + np.roll(a, -1, axis=1)) / 4.0

    c = smooth(count)
    prior_w, prior_r = 0.08, 0.15
    wet_f = (smooth(wet) + 5 * prior_w) / (c + 5)
    rate_f = (smooth(rate) + 5 * prior_r) / (c + 5)
    mod_f = (smooth(mod) + 5 * prior_w * 0.3) / (c + 5)
    heavy_f = (smooth(heavy) + 5 * prior_w * 0.08) / (c + 5)
    return Climatology(wet_f.astype(np.float32), rate_f.astype(np.float32), mod_f.astype(np.float32),
                       heavy_f.astype(np.float32))


class SampleBuilder:
    """
    Builds examples from caches. ``hourly(short, hours)``, ``upper(name, hours)`` (on the 1.5-degree grid),
    ``cmorph.targets(hours, columns)``, ``cmorph.mesh(hours)`` and ``stations.read(hours)`` are the only things it
    asks of the data, so the synthetic world in :mod:`pluvium.synthetic` drives it exactly as the real caches do.
    """

    def __init__(self, geo, era5, cmorph, stations=None, climatology: Climatology | None = None,
                 station_climatology: Climatology | None = None, schedule: Schedule = DEFAULT,
                 ocean_targets: int = 1500, max_land_targets: int = 6000, memory_dropout: float = 0.35,
                 emulate: float = 0.75, seed: int = 0, driver=None) -> None:
        self.geo, self.era5, self.cmorph, self.stations = geo, era5, cmorph, stations
        #: ``driver(t0) -> (fields, upper)`` replaces ERA5 as the driving fields (stage B: real GFS runs)
        self.driver = driver
        self.schedule = schedule
        self.ocean_targets, self.max_land = ocean_targets, max_land_targets
        self.memory_dropout, self.emulate = memory_dropout, emulate
        self.rng = np.random.default_rng(seed)
        t = geo.targets
        self.land_idx = np.flatnonzero(t.kind <= 1)
        self.sea_idx = np.flatnonzero(t.kind == 2)
        self.clim = climatology or Climatology.flat(len(t))
        self.station_clim = station_climatology or Climatology.flat(len(stations) if stations is not None else 0)
        self.station_statics = (geo.terrain.descriptors(stations.lat, stations.lon) if stations is not None
                                else np.zeros((0, 13), np.float32))
        self.lat1, self.lon1, self.lat2, self.lon2 = era5.lat1, era5.lon1, era5.lat2, era5.lon2
        self.tile_lat = np.array([tl.lat0 for tl in geo.tiles], np.float64)
        self.tile_lon = np.array([tl.lon0 for tl in geo.tiles], np.float64)
        from .advection import Advector

        # the observed rain field is read at every target cell and mesh point, and carried along the steering wind
        self.obs_lat = np.concatenate([t.lat, geo.mesh.lat])
        self.obs_lon = np.concatenate([t.lon, geo.mesh.lon])
        self.advector = Advector(self.obs_lat, self.obs_lon)
        self.station_dropout = (0.2, 0.3)       # (whole-network, per-station) dropout of reports given as inputs

    # ----------------------------------------------------------------------------------- choosing --
    def choose_targets(self, train: bool) -> np.ndarray:
        if train:
            land = self.land_idx
            if land.size > self.max_land:
                land = np.sort(self.rng.choice(land, self.max_land, replace=False))
            sea = (np.sort(self.rng.choice(self.sea_idx, min(self.ocean_targets, self.sea_idx.size), replace=False))
                   if self.sea_idx.size else self.sea_idx)
            return np.concatenate([land, sea])
        land = self.land_idx[:: max(1, self.land_idx.size // self.max_land)][: self.max_land]
        sea = self.sea_idx[:: max(1, self.sea_idx.size // max(1, self.ocean_targets))][: self.ocean_targets]
        return np.concatenate([land, sea])

    # ------------------------------------------------------------------------------------ driving --
    def driving(self, t0: int) -> tuple[dict, dict]:
        """Raw ERA5 fields (frames at the schedule's centres; rain averaged over each frame's span), or the driver's."""
        if self.driver is not None:
            return self.driver(t0)
        sch = self.schedule
        centres = t0 + sch.centres
        fields = {}
        for short in HOURLY_SHORT:
            if short == "tp":
                hours = (t0 + sch.leads[:, None] + np.arange(3)[None, :])
                rain = self.era5.hourly("tp", hours.ravel()).reshape(hours.shape + (self.lat1.size, self.lon1.size))
                mask = np.arange(3)[None, :] < sch.spans[:, None]
                fields["tp"] = (np.nansum(rain * mask[..., None, None], axis=1) / sch.spans[:, None, None]).astype(np.float32)
            else:
                fields[short] = self.era5.hourly(short, centres)
        upper = {f"{s}{p}": self.era5.upper(f"{s}{p}", centres) for s in UPPER_SHORT for p in LEVELS}
        return fields, upper

    def frames_from(self, fields: dict, upper: dict, t0: int, rng=None, emulate: bool = False) -> dict:
        """Driving fields -> encoder channels, tile drivers, flags (shared by training and live forecasts)."""
        sch = self.schedule
        centres = t0 + sch.centres
        fields = {k: np.nan_to_num(v, nan=_FILL.get(k, 0.0)).astype(np.float32) for k, v in fields.items()}
        upper = {k: np.nan_to_num(v, nan=_FILL.get(k[0], 0.0)).astype(np.float32) for k, v in upper.items()}
        hourly_raw = np.stack([fields[s] for s in HOURLY_SHORT], 1)
        upper_raw = np.stack([upper[f"{s}{p}"] for s in UPPER_SHORT for p in LEVELS], 1)
        scale = 0.0
        if emulate and rng is not None:
            cell1 = (self.lat1[1] - self.lat1[0]) * 111.195
            cell2 = (self.lat2[1] - self.lat2[0]) * 111.195
            raw = hourly_raw.copy()
            raw[:, 0] = np.log1p(np.clip(raw[:, 0], 0, None))
            raw, upper_raw, scale = emulate_forecast_error(raw, upper_raw, sch.leads, rng, cell1, cell2)
            raw[:, 0] = np.expm1(np.clip(raw[:, 0], 0, None))
            hourly_raw = raw
            fields = {s: hourly_raw[:, k] for k, s in enumerate(HOURLY_SHORT)}
            upper = {f"{s}{p}": upper_raw[:, k] for k, (s, p) in enumerate((s, p) for s in UPPER_SHORT for p in LEVELS)}
        upper1 = {k: regrid(upper[k], self.lat2, self.lon2, self.lat1, self.lon1) for k in UPPER_ON_HOURLY}
        hourly = hourly_channels(fields, upper1, self.lat1, self.lon1, centres.astype(np.float64))
        flags = regime_flags(fields, upper1, self.lat1, self.lon1, centres)
        tiles = np.zeros((sch.num_frames, len(self.tile_lat), 3), np.float32)
        if len(self.tile_lat):
            tiles[..., 0] = bilinear(upper["u850"], self.lat2, self.lon2, self.tile_lat, self.tile_lon)
            tiles[..., 1] = bilinear(upper["v850"], self.lat2, self.lon2, self.tile_lat, self.tile_lon)
            tiles[..., 2] = bilinear(fields["tcwv"], self.lat1, self.lon1, self.tile_lat, self.tile_lon)
        now = int(np.nonzero(sch.leads == 0)[0][0])
        steer = np.stack([0.5 * (upper["u850"][now] + upper["u700"][now]), 0.5 * (upper["v850"][now] + upper["v700"][now])])
        return {"hourly": hourly, "upper": upper_channels(upper), "flags": flags, "tiles": tiles, "fields": fields,
                "steer": steer.astype(np.float32),
                "error_scale": scale}

    # ------------------------------------------------------------------------------------ context --
    def point_context(self, lat, lon, land, frames: dict, t0: int, clim: Climatology, memory: np.ndarray | None,
                      memory_known: bool, observed: np.ndarray | None = None) -> np.ndarray:
        """``(F, N, len(TARGET_DYNAMIC))`` per-frame context at forecast points. ``observed`` is the last hour's rain
        at :attr:`obs_lat`/:attr:`obs_lon` (target cells, then mesh points), carried along the steering wind."""
        sch = self.schedule
        centres = (t0 + sch.centres).astype(np.float64)
        f = sch.num_frames
        tf = time_features(lat[None, :], lon[None, :], centres[:, None])
        u = bilinear(frames["fields"]["u10"], self.lat1, self.lon1, lat, lon)
        v = bilinear(frames["fields"]["v10"], self.lat1, self.lon1, lat, lon)
        heat = heating_potential(self.geo.terrain, lat, lon, u, v)
        rain = bilinear(np.clip(frames["fields"]["tp"], 0, None), self.lat1, self.lon1, lat, lon)
        cw, cr = clim.at(t0 + sch.centres, lon)
        mem = np.zeros((f, lat.size), np.float32) if memory is None else np.broadcast_to(memory[None, :], (f, lat.size))
        known = np.full((f, lat.size), float(memory_known and memory is not None), np.float32)
        adv = np.zeros((f, lat.size), np.float32)
        adv_known = np.zeros((f, lat.size), np.float32)
        if observed is not None and memory_known:
            from .advection import HOURS

            us = bilinear(frames["steer"][0], self.lat2, self.lon2, lat, lon)
            vs = bilinear(frames["steer"][1], self.lat2, self.lon2, lat, lon)
            which = np.nonzero((sch.leads >= 0) & (sch.leads < HOURS))[0]
            moved, found = self.advector.extrapolate(observed, lat, lon, us, vs, sch.leads[which])
            adv[which] = np.log1p(np.clip(moved, 0, None))
            adv_known[which] = found
        out = np.concatenate([tf, heat, np.log1p(rain)[..., None], cw[..., None], np.log1p(cr)[..., None] * 3.0,
                              np.log1p(np.nan_to_num(mem))[..., None], known[..., None], adv[..., None],
                              adv_known[..., None]], axis=-1)
        del land
        return out.astype(np.float32)

    def mesh_context(self, frames: dict, t0: int, memory: np.ndarray | None) -> np.ndarray:
        """``(F, N_mesh, len(MESH_DYNAMIC))``: rain observed in the history frames (log1p, known flag), sun, season,
        island heating."""
        sch = self.schedule
        mesh = self.geo.mesh
        centres = (t0 + sch.centres).astype(np.float64)
        f, n = sch.num_frames, len(mesh)
        mem = np.zeros((f, n, 2), np.float32)
        if memory is not None:
            hist = np.nonzero(sch.leads < 0)[0]
            ok = np.isfinite(memory)
            mem[hist, :, 0] = np.log1p(np.where(ok, memory, 0.0))
            mem[hist, :, 1] = ok.astype(np.float32)
        tf = time_features(mesh.lat[None, :], mesh.lon[None, :], centres[:, None])
        u = bilinear(frames["fields"]["u10"], self.lat1, self.lon1, mesh.lat, mesh.lon)
        v = bilinear(frames["fields"]["v10"], self.lat1, self.lon1, mesh.lat, mesh.lon)
        heat = heating_potential(self.geo.terrain, mesh.lat, mesh.lon, u, v)
        return np.concatenate([mem, tf, heat], axis=-1).astype(np.float32)

    # ------------------------------------------------------------------------------------- examples --
    def example(self, t0: int, train: bool = True, targets: np.ndarray | None = None,
                emulate: bool | None = None) -> dict:
        """One forecast issued at ``t0``. ``emulate`` forces forecast-error emulation on or off (default: on for a
        share ``self.emulate`` of training examples, off otherwise)."""
        sch = self.schedule
        rng = self.rng if train else np.random.default_rng(int(t0))
        fields, upper = self.driving(t0)
        if emulate is None:
            emulate = train and self.driver is None and rng.random() < self.emulate
        frames = self.frames_from(fields, upper, t0, rng, emulate=emulate)
        idx = self.choose_targets(train) if targets is None else np.asarray(targets)
        t = self.geo.targets
        hist_hours = t0 + sch.leads[sch.leads < 0]
        keep_memory = not (train and rng.random() < self.memory_dropout)
        mesh_memory = self.cmorph.mesh(hist_hours) if keep_memory else None
        observed = last_hour = None
        if keep_memory:
            everywhere = self.cmorph.targets(np.array([t0 - 1]))[0]
            last_hour = everywhere[idx]
            observed = np.concatenate([everywhere, mesh_memory[-1]])
        hours = t0 + np.arange(sch.num_hours)
        land = t.kind[idx] == 0
        target_ctx = self.point_context(t.lat[idx], t.lon[idx], land, frames, t0, _Indexed(self.clim, idx), last_hour,
                                        keep_memory, observed)
        flags = np.stack([np.stack([bilinear(frames["flags"][k, j].astype(np.float32), self.lat1, self.lon1,
                                             t.lat[idx], t.lon[idx]) > 0.5 for j in range(4)])
                          for k in range(sch.num_frames)])
        out = {
            "t0": np.int64(t0), "targets": idx.astype(np.int64), "hourly": frames["hourly"], "upper": frames["upper"],
            "tiles": frames["tiles"], "mesh_ctx": self.mesh_context(frames, t0, mesh_memory),
            "target_ctx": target_ctx, "truth": self.cmorph.targets(hours, idx),
            "regime": regime_labels(flags, land, t.lon[idx], (t0 + sch.centres).astype(np.float64)),
            "error_scale": np.float32(frames["error_scale"]),
        }
        if self.stations is not None and len(self.stations):
            from .weather import decode, obs_features

            st = self.stations
            reports = st.read(hist_hours).astype(np.int8)                          # (3, S) packed codes
            if train:                                                               # sometimes no reports, often gaps
                if rng.random() < self.station_dropout[0]:
                    reports[:] = -1
                else:
                    reports[:, rng.random(len(st)) < self.station_dropout[1]] = -1
            d = decode(reports[-1])
            last = np.where(d["known"], d["wet"].astype(np.float32), np.nan) if keep_memory else None
            out["station_ctx"] = self.point_context(st.lat, st.lon, np.ones(len(st), bool), frames, t0,
                                                    self.station_clim, last, keep_memory, observed)
            out["station_obs"] = obs_features(reports)
            out["station_truth"] = st.read(hours)
        return out

    def valid_starts(self, hours: np.ndarray) -> np.ndarray:
        """Issue times among ``hours`` whose whole window lies inside staged data (checked on CMORPH's first target
        and ERA5 rain at the first and last frames)."""
        sch = self.schedule
        ok = []
        for t0 in hours:
            a = self.cmorph.targets(np.array([t0 + sch.leads[0], t0 + sch.num_hours - 1]), np.array([0]))
            e = self.era5.hourly("tp", np.array([t0 + sch.leads[0], t0 + sch.leads[-1] + sch.spans[-1] - 1]))
            ok.append(bool(np.isfinite(a).all() and np.isfinite(e[:, 0, 0]).all()))
        return hours[np.array(ok, bool)] if len(hours) else hours


#: Physical fill values for gaps in the driving fields (never zero pressure or zero temperature).
_FILL = {"tp": 0.0, "tcwv": 40.0, "u10": 0.0, "v10": 0.0, "t2m": 298.0, "msl": 101500.0, "tcc": 0.5, "u": 0.0, "v": 0.0,
         "q": 0.008, "t": 280.0}


class _Indexed:
    """A climatology restricted to some of its points."""

    def __init__(self, clim: Climatology, idx: np.ndarray) -> None:
        self.clim, self.idx = clim, idx

    def at(self, hours, lon):
        return self.clim.subset(self.idx).at(hours, lon)


def collate(examples: list[dict]) -> dict:
    """Stack examples into a batch of tensors (all examples of a batch share their target count)."""
    import torch

    out = {}
    for key in examples[0]:
        values = [e[key] for e in examples]
        arr = np.stack([np.asarray(v) for v in values])
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        out[key] = torch.from_numpy(np.ascontiguousarray(arr))
    return out


__all__ = ["Climatology", "MESH_DYNAMIC", "SampleBuilder", "TARGET_BASE", "TARGET_DYNAMIC", "UPPER_ON_HOURLY", "WET_MM", "bilinear", "collate",
           "compute_climatology", "regrid"]
