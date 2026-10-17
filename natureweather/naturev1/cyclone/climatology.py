# Copyright 2026 Nathan. Apache-2.0.
"""
The flow a storm will meet in three days is mostly the flow it would meet on an average year -- plus what is
left of today's anomaly.

The idea: a steering flow frozen at the initial time is the right answer at hour zero and an increasingly
wrong one after: troughs move on, ridges rebuild, and by day five the analysis's particular pattern has lost
most of its memory. What is *not* lost is the seasonal circulation -- the subtropical ridge in September is
where it always is in September. So the flow is split into climatology plus anomaly, and the anomaly is
**damped with lead time** (the persistence of steering anomalies is a few days; Neumann's CLIPER is the
fully-damped limit, the frozen flow the undamped one):

``V(x, t) = V_clim(x, doy) + exp(-t / tau) * (V_analysis(x) - V_clim(x, doy))``

The climatology is ERA5 1981-2010 (WeatherBench 2's daily climatology, a 7-day window) -- years that do not
overlap the 2016-2022 hindcast seasons, so the benchmark is not flattered by its own future. It is cached as a
day-of-year stack of the winds the steering layers use, plus the 500 hPa height (for the ridge's anomaly) and
the day-to-day standard deviation of the 500 hPa wind (the natural size of an analysis's flow uncertainty,
used by the ensemble).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fields import Environment

STORE = ("gs://weatherbench2/datasets/era5-daily-climatology/"
         "daily_clim_daily_mean_1981_to_2010_7_dw-240x121_equiangular_with_poles_conservative.zarr")
LEVELS = (200, 250, 300, 400, 500, 600, 700, 850)
CHANNELS = [f"u@{p}" for p in LEVELS] + [f"v@{p}" for p in LEVELS] + ["z@500", "ustd@500", "vstd@500"]


def build_climatology(path: str | Path, log=print) -> Path:
    """Read the day-of-year climatology into ``path`` (``(366, C, 121, 240)`` float16 + ``.json``); idempotent."""
    import json

    import xarray as xr

    path = Path(path)
    if path.exists() and path.with_suffix(".json").exists():
        return path
    ds = xr.open_zarr(STORE, storage_options={"token": "anon"}, chunks=None)
    lat = ds.latitude.values
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(366, len(CHANNELS), 121, 240))
    levels = list(ds.level.values.astype(int))
    order = [levels.index(p) for p in LEVELS]

    def oriented(array):
        array = np.swapaxes(np.asarray(array, np.float32), -1, -2)
        return array[..., ::-1, :] if lat[0] > lat[-1] else array

    for name, slot, idx in (("u_component_of_wind_mean", 0, order), ("v_component_of_wind_mean", len(LEVELS), order)):
        data = oriented(ds[name].values)                            # (366, L, lat, lon)
        out[:, slot:slot + len(LEVELS)] = data[:, idx]
        log(f"[climatology] {name}")
    out[:, 2 * len(LEVELS)] = oriented(ds["geopotential_mean"].sel(level=500).values) / 100.0   # dam-scaled for fp16
    out[:, 2 * len(LEVELS) + 1] = oriented(ds["u_component_of_wind_std"].sel(level=500).values)
    out[:, 2 * len(LEVELS) + 2] = oriented(ds["v_component_of_wind_std"].sel(level=500).values)
    out.flush()
    path.with_suffix(".json").write_text(json.dumps({"channels": CHANNELS, "levels": list(LEVELS), "z_scale": 100.0}))
    return path


class Climatology:
    """Day-of-year climatology, cut into the same boxes as the hindcast analyses."""

    def __init__(self, path: str | Path) -> None:
        self.data = np.load(path, mmap_mode="r")
        self.lat = np.linspace(-90.0, 90.0, 121)
        self.slot = {n: k for k, n in enumerate(CHANNELS)}

    def environment(self, like: Environment, when: np.datetime64, shift_days: float = 0.0) -> Environment:
        """An :class:`Environment` on ``like``'s box holding the climatological winds (and z500) for ``when``."""
        day = int((np.datetime64(when, "D") - np.datetime64(str(when)[:4] + "-01-01", "D")).astype(int) + shift_days) % 366
        rows = np.rint((like.lat - self.lat[0]) / 1.5).astype(int)
        cols = np.rint((like.lon % 360.0) / 1.5).astype(int) % 240
        block = self.data[day][:, rows][:, :, cols].astype(np.float64)
        fields = {"u": np.stack([block[self.slot[f"u@{p}"]] for p in LEVELS]),
                  "v": np.stack([block[self.slot[f"v@{p}"]] for p in LEVELS]),
                  "z": np.repeat(block[self.slot["z@500"]][None] * 100.0, len(LEVELS), 0),
                  "ustd500": block[self.slot["ustd@500"]], "vstd500": block[self.slot["vstd@500"]]}
        return Environment(lat=like.lat, lon=like.lon, levels=LEVELS, fields=fields, time=like.time)


class DampedAnomalyFlow:
    """``V_clim + exp(-t / tau) (V_analysis - V_clim)`` from two steering flows (``velocity(lat, lon)`` objects)."""

    def __init__(self, analysis_flow, climate_flow, tau_h: float = 72.0) -> None:
        self.analysis, self.climate, self.tau_h = analysis_flow, climate_flow, tau_h

    def __call__(self, lat, lon, hours):
        ua, va = self.analysis.velocity(lat, lon)
        uc, vc = self.climate.velocity(lat, lon)
        memory = np.exp(-np.asarray(hours, np.float64) / max(self.tau_h, 1e-6))
        return uc + memory * (ua - uc), vc + memory * (va - vc)


__all__ = ["CHANNELS", "Climatology", "DampedAnomalyFlow", "LEVELS", "STORE", "build_climatology"]
