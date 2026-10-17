# Copyright 2026 Nathan. Apache-2.0.
"""
The models to beat, on the same storms: tracks from GraphCast, Pangu-Weather, IFS HRES, Aurora, FGN (the
model behind WeatherNext 2), FuXi and GenCast, as archived by WeatherBench 2 (NeuralGCM's archive has no
sea-level pressure, so it is left out rather than tracked on a different field).

The idea: "better than GraphCast on hurricanes" is a measurement, not a slogan. WeatherBench 2 keeps each
model's forecasts conservatively regridded to the same 1.5-degree grid NatureV1 runs on, initialised at
00 and 12 UTC -- the same initial times as the hindcast cases. Every teacher track here is made the same way:
starting from the best-track position at the initial time, following the sea-level pressure minimum through
the teacher's own forecast (:func:`naturev1.cyclone.field_tracker.track_in_fields`), so differences between
models are differences in their forecasts, not in how they were tracked.

Caveat stated where it belongs: tracking at 1.5 degrees smooths every model's low (each of them ran at 0.25
degrees and was regridded); the sub-grid quadratic refinement recovers most of the position, but a teacher
tracked at 0.25 degrees would score a little better than here.
"""

from __future__ import annotations

import numpy as np

from .field_tracker import track_in_fields

WB2 = "gs://weatherbench2/datasets/"
#: name -> list of (store, first year, last year).
STORES = {
    "graphcast": [("graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr", 2018, 2018),
                  ("graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr", 2020, 2020)],
    "pangu": [("pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr", 2018, 2022)],
    "hres": [("hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr", 2016, 2022)],
    "aurora": [("aurora/2022-240x121_equiangular_with_poles_conservative.zarr", 2022, 2022)],
    "fgn": [("fgn/2022-240x121_equiangular_with_poles_conservative_mean.zarr", 2022, 2022)],
    "fuxi": [("fuxi/2020-240x121_equiangular_with_poles_conservative.zarr", 2020, 2020)],
    "gencast": [("gencast/2020-240x121_equiangular_with_poles_conservative_mean.zarr", 2020, 2020)],
}


class TeacherStore:
    """One WeatherBench 2 forecast store: sea-level pressure at chosen leads for an initial time."""

    def __init__(self, path: str) -> None:
        import xarray as xr

        self.ds = xr.open_zarr(WB2 + path, storage_options={"token": "anon"}, chunks=None)
        self.times = self.ds.time.values.astype("datetime64[s]")
        lead = self.ds.prediction_timedelta
        if np.issubdtype(lead.dtype, np.timedelta64):
            self.leads_h = (lead.values / np.timedelta64(1, "h")).astype(int)
        else:                                             # stored as integers with a units attribute
            scale = {"hours": 1.0, "minutes": 1 / 60.0, "seconds": 1 / 3600.0, "days": 24.0}.get(lead.attrs.get("units", "hours"), 1.0)
            self.leads_h = np.round(lead.values.astype(np.float64) * scale).astype(int)
        self.lat = self.ds.latitude.values.astype(np.float64)
        self.lon = self.ds.longitude.values.astype(np.float64)
        msl = self.ds["mean_sea_level_pressure"]
        self.order = msl.dims
        self.has_member = "number" in msl.dims or "sample" in msl.dims or "realization" in msl.dims

    def covers(self, when: np.datetime64) -> bool:
        return bool(np.any(self.times == np.datetime64(when, "s")))

    def msl(self, when: np.datetime64, leads) -> np.ndarray | None:
        """``(L, H, W)`` Pa, latitude ascending; None when the initial time or a lead is missing."""
        index = np.nonzero(self.times == np.datetime64(when, "s"))[0]
        if index.size == 0:
            return None
        wanted = [int(np.nonzero(self.leads_h == h)[0][0]) if np.any(self.leads_h == h) else -1 for h in leads]
        present = [w for w in wanted if w >= 0]
        if not present:
            return None
        lo, hi = min(present), max(present) + 1
        sel = self.ds["mean_sea_level_pressure"].isel(time=int(index[0]), prediction_timedelta=slice(lo, hi))
        for dim in ("number", "sample", "realization"):
            if dim in sel.dims:
                sel = sel.isel({dim: 0})
        data = np.asarray(sel.values, np.float32)
        dims = [d for d in sel.dims]
        data = np.moveaxis(data, [dims.index("prediction_timedelta"), dims.index("latitude"), dims.index("longitude")], [0, 1, 2])
        out = np.full((len(wanted),) + data.shape[1:], np.nan, np.float32)
        for k, w in enumerate(wanted):
            if w >= 0:
                out[k] = data[w - lo]
        if self.lat[0] > self.lat[-1]:
            out = out[:, ::-1]
        return out

    @property
    def lat_ascending(self) -> np.ndarray:
        return self.lat if self.lat[0] < self.lat[-1] else self.lat[::-1]


def teacher_tracks(cases, name: str, leads, log=None) -> np.ndarray:
    """``(N, L, 2)`` lat/lon of teacher ``name`` for every case it covers (NaN elsewhere)."""
    from .baselines import motion_back
    from .geometry import local_displacement_km

    out = np.full((len(cases), len(leads), 2), np.nan)
    stores = []
    for path, first, last in STORES[name]:
        try:
            stores.append((TeacherStore(path), first, last))
        except Exception as error:                     # a store that is unreachable is reported, not fatal
            if log:
                log(f"[teachers] {name}: {path} unavailable ({error})")
    cache: dict = {}
    for i, case in enumerate(cases):
        store = next((s for s, first, last in stores if first <= case.year <= last and s.covers(case.time)), None)
        if store is None:
            continue
        key = (id(store), int(np.datetime64(case.time, "s").astype(np.int64)))
        if key not in cache:
            cache.clear()
            cache[key] = store.msl(case.time, leads)
        fields = cache[key]
        if fields is None:
            continue
        rate = motion_back(case, 6)
        first = None
        if rate is not None:
            east, north = local_displacement_km(case.lat - rate[0] * 6, case.lon - rate[1] * 6, case.lat, case.lon)
            first = (float(east) / 6.0, float(north) / 6.0)
        out[i] = track_in_fields((case.lat, case.lon), leads, fields, store.lat_ascending, store.lon, first_motion_kmh=first)
        if log and i % 250 == 0:
            log(f"[teachers] {name}: {i}/{len(cases)}")
    return out


__all__ = ["STORES", "TeacherStore", "teacher_tracks"]
