# Copyright 2026 Nathan. Apache-2.0.
"""
Reading a teacher's forecast and handing back NatureV1-aligned fields, whatever the teacher.

Three kinds of source, one output (:func:`naturev1.distill.align.to_channels`-ready fields on NatureV1's grid,
levels and leads):

* :class:`ArchiveTeacher` -- WeatherBench 2 archives (Pangu, GraphCast, HRES, Aurora, FGN/WeatherNext 2, FuXi,
  GenCast, NeuralGCM): dimension orders differ, lead times are stored as timedeltas or as integer hours,
  names may carry stray whitespace, ensembles come as a ``sample`` dimension (reduced here to a mean *and* a
  variance -- the spread is a target too). The 1.5-degree archives are on NatureV1's grid already; anything
  else is regridded conservatively.
* :class:`OpenDataAIFS` -- ECMWF's AIFS from the open-data service (GRIB on the N320 reduced Gaussian grid, the
  last few days only): binned onto the grid by :func:`naturev1.distill.align.points_to_grid`. Needs
  ``ecmwf-opendata`` and ``cfgrib``.
* :class:`Earth2Teacher` -- an NVIDIA Earth2Studio model (SFNO, FourCastNet v2...) run locally on the GPU from
  the same initial conditions NatureV1 sees. Needs ``earth2studio``.
"""

from __future__ import annotations

import numpy as np

from .align import NATURE_LAT, NATURE_LON, ConservativeRegrid, canonical_name, to_leads, to_levels, to_si
from .catalog import TEACHERS, WB2


def _leads_hours(coord) -> np.ndarray:
    values = coord.values
    if np.issubdtype(values.dtype, np.timedelta64):
        return (values / np.timedelta64(1, "h")).astype(np.float64)
    scale = {"hours": 1.0, "minutes": 1 / 60.0, "seconds": 1 / 3600.0, "days": 24.0}.get(coord.attrs.get("units", "hours"), 1.0)
    return values.astype(np.float64) * scale


class ArchiveTeacher:
    """One teacher-year from WeatherBench 2, read and aligned per initial time."""

    def __init__(self, name: str, year: int, resolution: str = "1p5") -> None:
        import xarray as xr

        self.teacher = TEACHERS[name]
        archive = self.teacher.archive_1p5 if resolution == "1p5" else self.teacher.archive_0p25
        if year not in archive:
            raise KeyError(f"{name} has no {resolution} archive for {year}")
        self.ds = xr.open_zarr(WB2 + archive[year], storage_options={"token": "anon"}, chunks=None)
        self.names = {canonical_name(v): v for v in self.ds.data_vars}
        self.times = self.ds.time.values.astype("datetime64[s]")
        self.leads_h = _leads_hours(self.ds.prediction_timedelta)
        self.lat = self.ds.latitude.values.astype(np.float64)
        self.lon = self.ds.longitude.values.astype(np.float64)
        self.levels = [int(v) for v in self.ds.level.values] if "level" in self.ds.dims else []
        same = (self.lat.size == NATURE_LAT.size and self.lon.size == NATURE_LON.size
                and np.allclose(np.sort(self.lat), NATURE_LAT) and np.allclose(np.sort(self.lon % 360), NATURE_LON))
        self.regrid = None if same else ConservativeRegrid(self.lat, self.lon)

    def covers(self, when) -> bool:
        return bool(np.any(self.times == np.datetime64(when, "s")))

    def _orient(self, data: np.ndarray, dims: tuple) -> np.ndarray:
        """``(..., lat, lon)`` on NatureV1's grid (latitude ascending, longitude 0..358.5)."""
        order = [d for d in dims if d not in ("latitude", "longitude")] + ["latitude", "longitude"]
        data = np.transpose(data, [dims.index(d) for d in order])
        if self.regrid is not None:
            return self.regrid(data)
        if self.lat[0] > self.lat[-1]:
            data = data[..., ::-1, :]
        return data[..., np.argsort(self.lon % 360.0)]

    def read(self, when, leads_h, variables: dict) -> tuple[dict, dict]:
        """
        Aligned fields for one initial time.

        Args:
            variables: ``{canonical name: [levels] or None}`` wanted.

        Returns:
            ``(mean, variance)`` each ``{name: (S, H, W)}`` or ``{name: {level: (S, H, W)}}``; variance only for
            ensembles (else empty). Missing variables, levels or leads are simply absent / NaN.
        """
        index = np.nonzero(self.times == np.datetime64(when, "s"))[0]
        if index.size == 0:
            return {}, {}
        wanted_leads = [h for h in leads_h]
        in_range = [k for k, h in enumerate(self.leads_h) if min(wanted_leads) - self.teacher.lead_step_h <= h <= max(wanted_leads) + self.teacher.lead_step_h]
        if not in_range:
            return {}, {}
        lo, hi = min(in_range), max(in_range) + 1
        mean, var = {}, {}
        for name, levels in variables.items():
            source = self.names.get(name)
            if source is None:
                continue
            arr = self.ds[source].isel(time=int(index[0]), prediction_timedelta=slice(lo, hi))
            if levels is not None and "level" in arr.dims:
                keep = sorted(set(int(p) for p in levels) | set(self.levels))
                arr = arr.sel(level=[p for p in self.levels if p in keep])
            data = np.asarray(arr.values, np.float64)
            dims = tuple(arr.dims)
            units = arr.attrs.get("units")
            ens = self.teacher.ensemble_dim if self.teacher.ensemble_dim in dims else None
            members = None
            if ens is not None:
                axis = dims.index(ens)
                members = data
                data = np.nanmean(members, axis=axis)
                spread = np.nanvar(members, axis=axis)
                dims = tuple(d for d in dims if d != ens)
            canonical, data = to_si(name, data, units)
            data = self._orient(data, dims)                                    # (S, [L], H, W)
            spread_aligned = None
            if members is not None:
                spread_aligned = self._orient(spread * (9.80665 ** 2 if canonical != name else 1.0), dims)
            data, _ = to_leads(data, self.leads_h[lo:hi], wanted_leads)
            if spread_aligned is not None:
                spread_aligned, _ = to_leads(spread_aligned, self.leads_h[lo:hi], wanted_leads)
            if "level" in dims:
                present = [int(p) for p in arr.level.values]
                targets = [p for p in (levels or present)]
                moved = np.moveaxis(data, 1, 0)                                # (L, S, H, W)
                aligned = to_levels(moved, present, targets)
                mean[canonical] = {p: aligned[i] for i, p in enumerate(targets) if np.isfinite(aligned[i]).any()}
                if spread_aligned is not None:
                    va = to_levels(np.moveaxis(spread_aligned, 1, 0), present, targets)
                    var[canonical] = {p: va[i] for i, p in enumerate(targets) if np.isfinite(va[i]).any()}
            else:
                mean[canonical] = data
                if spread_aligned is not None:
                    var[canonical] = spread_aligned
        return mean, var


class OpenDataAIFS:
    """
    ECMWF AIFS Single from the open-data service, aligned like an archive teacher. Real-time only (the service
    keeps a few days): suitable for live distillation top-ups and for a live comparison, not for a hindcast.
    """

    PARAMS = {"gh": "geopotential", "t": "temperature", "u": "u_component_of_wind", "v": "v_component_of_wind",
              "q": "specific_humidity", "msl": "mean_sea_level_pressure", "2t": "2m_temperature",
              "10u": "10m_u_component_of_wind", "10v": "10m_v_component_of_wind"}

    def __init__(self, directory: str = "aifs_opendata") -> None:
        try:
            from ecmwf.opendata import Client  # noqa: F401
            import cfgrib  # noqa: F401
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError("OpenDataAIFS needs `pip install ecmwf-opendata cfgrib eccodes`") from error
        self.directory = directory

    def read(self, when, leads_h, levels=(1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)) -> dict:  # pragma: no cover - network + GRIB
        import os

        import xarray as xr
        from ecmwf.opendata import Client

        from .align import points_to_grid

        os.makedirs(self.directory, exist_ok=True)
        stamp = np.datetime64(when, "h").astype(object)
        target = os.path.join(self.directory, f"aifs_{stamp:%Y%m%d%H}.grib2")
        if not os.path.exists(target):
            Client(source="ecmwf", model="aifs-single").retrieve(
                date=f"{stamp:%Y%m%d}", time=stamp.hour, step=list(leads_h), type="fc",
                param=list(self.PARAMS), levelist=list(levels), target=target)
        out: dict = {}
        for level_type in ("isobaricInhPa", "surface", "meanSea", "heightAboveGround"):
            try:
                ds = xr.open_dataset(target, engine="cfgrib", backend_kwargs={"filter_by_keys": {"typeOfLevel": level_type}})
            except Exception:
                continue
            lat, lon = ds.latitude.values.ravel(), ds.longitude.values.ravel()
            for var in ds.data_vars:
                name, data = to_si(self.PARAMS.get(var, var), ds[var].values, ds[var].attrs.get("units"))
                grid = points_to_grid(data, lat, lon)
                if "isobaricInhPa" in ds[var].dims:
                    out[name] = {int(p): grid[:, k] for k, p in enumerate(ds.isobaricInhPa.values)}
                else:
                    out[name] = grid
        return out


class Earth2Teacher:
    """An Earth2Studio model run locally; ``model`` e.g. ``"SFNO"`` or ``"FCN"`` (see Earth2Studio's model zoo)."""

    def __init__(self, model: str = "SFNO", device: str = "cuda") -> None:
        try:
            import earth2studio  # noqa: F401
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError("Earth2Teacher needs `pip install earth2studio`") from error
        self.model_name, self.device = model, device

    def read(self, when, leads_h, data_source=None) -> dict:  # pragma: no cover - GPU + model download
        import earth2studio.models.px as px
        from earth2studio.data import ARCO
        from earth2studio.io import XarrayBackend
        from earth2studio.run import deterministic

        model_cls = getattr(px, self.model_name)
        model = model_cls.load_model(model_cls.load_default_package()).to(self.device)
        io = XarrayBackend()
        steps = int(max(leads_h) // 6)
        deterministic([str(np.datetime64(when, "s"))], steps, model, data_source or ARCO(), io)
        ds = io.root
        regrid = ConservativeRegrid(ds.lat.values, ds.lon.values)
        out: dict = {}
        hours = (ds.lead_time.values / np.timedelta64(1, "h")).astype(float)
        for var in ds.data_vars:
            base, level = var.rstrip("0123456789"), var[len(var.rstrip("0123456789")):]
            name, data = to_si(base, ds[var].values[0], ds[var].attrs.get("units"))
            data, _ = to_leads(regrid(data), hours, leads_h)
            if level:
                out.setdefault(name, {})[int(level)] = data
            else:
                out[name] = data
        return out


__all__ = ["ArchiveTeacher", "Earth2Teacher", "OpenDataAIFS"]
