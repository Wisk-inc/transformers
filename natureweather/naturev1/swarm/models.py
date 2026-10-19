# Copyright 2026 Nathan. Apache-2.0.
"""
The live models, from public buckets, all read the same way: forecast steering and the model's own storm track.

===============  ====================================================================================================
``gfs``          NOAA GFS, 0.25 degrees, ``noaa-gfs-bdp-pds`` (4 cycles a day, about 4 hours after the analysis)
``ifs``          ECMWF IFS, 0.25 degrees, ECMWF open data (CC-BY-4.0): the strongest track model in NHC's verification
``aifs``         ECMWF AIFS Single, 0.25 degrees, ECMWF open data: ECMWF's operational machine-learning model
``graphcastgfs`` NOAA's GraphCast run from the GFS analysis, 0.25 degrees, ``noaa-nws-graphcastgfs-pds``
``gefs``         NOAA GEFS ensemble mean, 0.5 degrees (31 members), with the ensemble's sea-level-pressure spread
===============  ====================================================================================================

For each model, only the GRIB messages the physics needs are fetched by byte range from the index file (NCEP's
``.idx`` text or ECMWF's ``.index`` JSON lines): the 850-200 hPa winds and the sea-level pressure every 12 hours to
120 h, plus the pressure at the standard leads -- about 150 MB per model at 0.25 degrees, cached as ``.npz``. Every
field is averaged onto the 1.5-degree nodes of the training data, whatever its native grid, so the physics sees
the same thing it was fitted on. The model's storm is followed through its pressure field, removed from its winds,
and the layer-mean environment becomes a :class:`naturev1.cyclone.forecastflow.ForecastFlow`; the followed positions
are its track.

Why several: the equal-weight mean of independent models errs less than the best of them (NHC's TVCN), measured
here as 11-23% on held-out seasons (:mod:`naturev1.cyclone.multimodel`). Any model that is not yet published for a
cycle, or cannot be read, is skipped and the rest are used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..cyclone.data import HALF_LAT, HALF_LON, LEADS, PREVIOUS_LEVELS, STEP, _box_indices
from ..cyclone.fields import Environment
from .analysis import FLOW_HOURS, LAT_AXIS, LON_AXIS, _get

GFS_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{day}/{hour:02d}/atmos/gfs.t{hour:02d}z.pgrb2.0p25.f{lead:03d}"
GRAPHCAST_URL = ("https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/graphcastgfs.{day}/{hour:02d}/forecasts_13_levels/"
                 "graphcastgfs.t{hour:02d}z.pgrb2.0p25.f{lead:03d}")
GEFS_MEAN_URL = "https://noaa-gefs-pds.s3.amazonaws.com/gefs.{day}/{hour:02d}/atmos/pgrb2ap5/geavg.t{hour:02d}z.pgrb2a.0p50.f{lead:03d}"
GEFS_SPREAD_URL = "https://noaa-gefs-pds.s3.amazonaws.com/gefs.{day}/{hour:02d}/atmos/pgrb2ap5/gespr.t{hour:02d}z.pgrb2a.0p50.f{lead:03d}"
ECMWF_AWS = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/{day}/{hour:02d}z/MODEL/0p25/oper/{day}{hour:02d}0000-{lead}h-oper-fc.grib2"
ECMWF_SITE = "https://data.ecmwf.int/forecasts/{day}/{hour:02d}z/MODEL/0p25/oper/{day}{hour:02d}0000-{lead}h-oper-fc.grib2"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    title: str
    kind: str                       # "ncep": NCEP .idx text index; "ecmwf": ECMWF .index JSON lines
    urls: tuple                     # format strings ({day}, {hour}, {lead}), tried in order
    levels: tuple = PREVIOUS_LEVELS
    spread_urls: tuple = ()         # the ensemble spread of the same fields (GEFS), same layout
    index_suffix: str = ".idx"


MODELS = {
    "gfs": ModelSpec("gfs", "GFS", "ncep", (GFS_URL,)),
    "ifs": ModelSpec("ifs", "ECMWF IFS", "ecmwf", (ECMWF_AWS.replace("MODEL", "ifs"), ECMWF_SITE.replace("MODEL", "ifs")), index_suffix=".index"),
    "aifs": ModelSpec("aifs", "ECMWF AIFS", "ecmwf", (ECMWF_AWS.replace("MODEL", "aifs-single"), ECMWF_SITE.replace("MODEL", "aifs-single")),
                      index_suffix=".index"),
    "graphcastgfs": ModelSpec("graphcastgfs", "GraphCast-GFS", "ncep", (GRAPHCAST_URL,)),
    "gefs": ModelSpec("gefs", "GEFS mean", "ncep", (GEFS_MEAN_URL,), levels=(200, 250, 300, 400, 500, 700, 850), spread_urls=(GEFS_SPREAD_URL,)),
}
#: The models a live forecast tries, in order (GEFS adds the ensemble spread as a diagnostic).
DEFAULT_MODELS = ("gfs", "ifs", "aifs", "graphcastgfs", "gefs")
_NCEP = {"u": "UGRD", "v": "VGRD", "msl": ("PRMSL", "mean sea level")}
_ECMWF = {"u": "u", "v": "v", "msl": ("msl", "sfc")}


# --------------------------------------------------------------------------------------------------- indexes --

def _index_ncep(text: str) -> dict:
    """``{(variable, level): (first byte, last byte)}`` from an NCEP ``.idx``."""
    lines = text.splitlines()
    offsets = [int(line.split(":")[1]) for line in lines]
    out = {}
    for k, line in enumerate(lines):
        parts = line.split(":")
        out[(parts[3], parts[4])] = (offsets[k], offsets[k + 1] - 1 if k + 1 < len(lines) else None)
    return out


def _index_ecmwf(text: str) -> dict:
    """``{(param, level): (first byte, last byte)}`` from an ECMWF open-data ``.index`` (one JSON object per line)."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        level = str(row.get("levelist", row.get("levtype", "")))
        out[(row["param"], level)] = (int(row["_offset"]), int(row["_offset"]) + int(row["_length"]) - 1)
    return out


# ------------------------------------------------------------------------------------------- grids and GRIB --

def _overlap(nodes: np.ndarray, centres: np.ndarray, width: float, wrap: float | None = None, half: float = 0.75) -> np.ndarray:
    """``(nodes, cells)`` fraction of each source cell inside each node's box of half-width ``half`` degrees."""
    lo, hi = centres - width / 2.0, centres + width / 2.0
    out = np.zeros((nodes.size, centres.size))
    for s in ((0.0,) if wrap is None else (-wrap, 0.0, wrap)):
        a = np.maximum(nodes[:, None] - half, lo[None] + s)
        b = np.minimum(nodes[:, None] + half, hi[None] + s)
        out += np.clip(b - a, 0.0, None) / width
    return out


def coarsen_to_nodes(values: np.ndarray, lat_first: float, dlat: float, lon_first: float, dlon: float) -> np.ndarray:
    """
    Any regular latitude-longitude grid (rows north to south from ``lat_first``, columns east from ``lon_first``)
    averaged onto the 1.5-degree nodes: each node takes the source cells within 0.75 degrees of it, a cell straddling
    the edge by its overlap, weighted by the cosine of latitude. On the 0.25-degree grid this is exactly
    :func:`naturev1.swarm.analysis._coarsen`.
    """
    nj, ni = values.shape
    lat = lat_first - dlat * np.arange(nj)
    lon = (lon_first + dlon * np.arange(ni)) % 360.0
    wl = _overlap(LAT_AXIS, lat, dlat) * np.maximum(np.cos(np.radians(lat)), 1e-3)[None]
    wo = _overlap(LON_AXIS, lon, dlon, wrap=360.0)
    finite = np.isfinite(values)
    num = wl @ np.where(finite, values, 0.0) @ wo.T
    den = wl @ finite.astype(np.float64) @ wo.T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def decode_grib_native(message: bytes) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """One GRIB message on its own grid: ``(values (rows north to south, columns east from the first longitude >= 0),
    (lat_first, dlat, lon_first, dlon))``."""
    import eccodes

    gid = eccodes.codes_new_from_message(message)
    try:
        ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
        lat0 = float(eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees"))
        lon0 = float(eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees"))
        dlat = abs(float(eccodes.codes_get(gid, "jDirectionIncrementInDegrees")))
        dlon = abs(float(eccodes.codes_get(gid, "iDirectionIncrementInDegrees")))
        north_first = not bool(eccodes.codes_get(gid, "jScansPositively"))
        missing = float(eccodes.codes_get(gid, "missingValue"))
        values = np.asarray(eccodes.codes_get_values(gid), np.float64).reshape(nj, ni)
    finally:
        eccodes.codes_release(gid)
    values = np.where(values == missing, np.nan, values)
    if not north_first:
        values, lat0 = values[::-1], lat0 + dlat * (nj - 1)
    lon = (lon0 + dlon * np.arange(ni)) % 360.0
    shift = int(np.argmin(lon))                                   # start the columns at the first longitude >= 0
    values = np.roll(values, -shift, axis=1)
    return values, (lat0, dlat, float(lon[shift]), dlon)


def decode_grib(message: bytes) -> np.ndarray:
    """One GRIB message (any regular grid, either row order, longitudes from -180 or 0) onto the 1.5-degree nodes."""
    values, (lat0, dlat, lon0, dlon) = decode_grib_native(message)
    return coarsen_to_nodes(values, lat0, dlat, lon0, dlon)


def native_window(values: np.ndarray, grid: tuple, lat0: float, lon0: float, half_lat: float = 25.0,
                  half_lon: float = 35.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(field, lat_axis, lon_axis)`` of a native-grid field within a box around ``(lat0, lon0)``, longitudes
    continuous across the dateline."""
    lat_first, dlat, lon_first, dlon = grid
    nj, ni = values.shape
    rows = np.arange(nj)
    lat_axis = lat_first - dlat * rows
    keep = (lat_axis >= lat0 - half_lat) & (lat_axis <= lat0 + half_lat)
    j0 = int(round(((lon0 - lon_first) % 360.0) / dlon))
    half = int(round(half_lon / dlon))
    cols = (j0 + np.arange(-half, half + 1)) % ni
    lon_axis = lon_first + dlon * (j0 + np.arange(-half, half + 1))
    lon_axis = lon_axis - 360.0 * np.round((lon_axis - lon0) / 360.0)        # continuous around the storm
    field = values[keep][:, cols]
    return field[::-1], lat_axis[keep][::-1], lon_axis                           # latitude ascending for the tracker


# ---------------------------------------------------------------------------------------------- the models --

class LiveModel:
    """One model's forecast run, read by byte range and cached under ``cache`` as ``.npz`` per lead."""

    def __init__(self, spec: ModelSpec, cache: str | Path = "~/.cache/naturev1/models") -> None:
        self.spec = spec
        self.cache = Path(cache).expanduser()
        self.cache.mkdir(parents=True, exist_ok=True)

    def _stamp(self, when) -> tuple[str, int]:
        when = np.datetime64(when, "h")
        s = str(when).replace("-", "").replace("T", "")[:10]
        return s[:8], int(s[8:10])

    def _index(self, urls, when, lead: int):
        """``(base url, index)`` of the first source that has this lead."""
        day, hour = self._stamp(when)
        last = None
        for fmt in urls:
            base = fmt.format(day=day, hour=hour, lead=int(lead))
            try:
                text = _get(base + self.spec.index_suffix, timeout=60.0).decode()
                return base, (_index_ecmwf(text) if self.spec.kind == "ecmwf" else _index_ncep(text))
            except Exception as error:
                last = error
        raise RuntimeError(f"{self.spec.title}: no index for {day} {hour:02d}Z +{lead} h ({last})")

    def available(self, when, lead: int = 120) -> bool:
        """Whether the run initialised at ``when`` has published ``lead`` hours."""
        try:
            self._index(self.spec.urls, when, lead)
            return True
        except Exception:
            return False

    def _read(self, base: str, index: dict, key: tuple) -> np.ndarray:
        if key not in index:
            raise KeyError(f"{self.spec.title}: {key} not in {base}")
        start, end = index[key]
        return decode_grib(_get(base, start, end))

    def _keys(self, var: str):
        table = _ECMWF if self.spec.kind == "ecmwf" else _NCEP
        if var == "msl":
            return [table["msl"]]
        name = table[var]
        return [(name, str(p) if self.spec.kind == "ecmwf" else f"{p} mb") for p in self.spec.levels]

    def fields(self, when, lead: int) -> dict:
        """``u``, ``v`` ``(levels, 121, 240)`` and ``msl`` ``(121, 240)`` Pa at ``lead`` hours (plus ``msl_spread`` for an
        ensemble), on the 1.5-degree nodes, from the cache when there."""
        day, hour = self._stamp(when)
        path = self.cache / f"{self.spec.name}_{day}{hour:02d}_f{int(lead):03d}.npz"
        if path.exists():
            return dict(np.load(path))
        base, index = self._index(self.spec.urls, when, lead)
        start, end = index[self._keys("msl")[0]]
        msl_native, grid = decode_grib_native(_get(base, start, end))
        out = {"u": np.stack([self._read(base, index, k) for k in self._keys("u")]),
               "v": np.stack([self._read(base, index, k) for k in self._keys("v")]),
               "msl": coarsen_to_nodes(msl_native, *grid), "levels": np.asarray(self.spec.levels, np.float64),
               # the pressure on the model's own grid as well: the storm is followed there, not on the smoothed nodes
               "msl_native": msl_native.astype(np.float32), "native_grid": np.asarray(grid, np.float64)}
        if self.spec.spread_urls:
            try:
                sbase, sindex = self._index(self.spec.spread_urls, when, lead)
                out["msl_spread"] = self._read(sbase, sindex, self._keys("msl")[0])
            except Exception:
                pass
        np.savez_compressed(path, **out)
        return out

    def forecast_flow(self, when, case, leads=LEADS, hours=FLOW_HOURS):
        """``(ForecastFlow, track (L, 2), extras)`` -- see :func:`flow_and_track`."""
        return flow_and_track(lambda lead: self.fields(when, lead), case, leads, hours,
                              source=f"{self.spec.title} {str(np.datetime64(when, 'h'))[:13]}Z", levels=self.spec.levels)


def flow_and_track(fields, case, leads=LEADS, hours=FLOW_HOURS, source: str = "", levels=PREVIOUS_LEVELS):
    """
    A model's forecast steering for ``case`` and its own track of the storm, from ``fields(lead) -> {u, v, msl}``
    on the 1.5-degree nodes.

    The model's storm is followed through its sea-level pressure (:func:`naturev1.cyclone.field_tracker.track_in_fields`,
    seeded with the storm's observed motion); at each forecast hour its vortex is removed from the winds around where
    the model has it, and the 850-700, 850-400 and 850-200 hPa environmental winds are kept -- the construction the
    HRES hindcast flows were made with. Returns ``(ForecastFlow, model_track (L, 2), extras)``; a lead the model has
    lost the storm at is NaN. ``extras`` carries the model's central pressure along its track (``msl_min_hpa``) and,
    for an ensemble, its pressure spread there (``spread_hpa``).
    """
    from ..cyclone.baselines import motion_back
    from ..cyclone.field_tracker import track_in_fields
    from ..cyclone.forecastflow import LAYERS, ForecastFlow
    from ..cyclone.geometry import great_circle_km, local_displacement_km
    from ..cyclone.steering import SteeringFlow
    from ..cyclone.vortex import remove_vortex

    need = sorted(set(int(h) for h in hours) | set(int(h) for h in leads))
    rows, cols = int(round(2 * HALF_LAT / STEP)) + 1, int(round(2 * HALF_LON / STEP)) + 1
    rs, cs, box_lat, box_lon = _box_indices(case.lat, case.lon, LAT_AXIS, STEP, rows, cols)
    cut = lambda a: np.asarray(a)[..., rs, :][..., cs].astype(np.float64)
    got = {}
    for h in need:
        f = fields(h)
        got[h] = {k: cut(f[k]) for k in ("u", "v", "msl") if k in f}
        if "msl_spread" in f:
            got[h]["msl_spread"] = cut(f["msl_spread"])
    first = None
    rate = motion_back(case, 6)
    if rate is not None:
        east, north = local_displacement_km(case.lat - rate[0] * 6, case.lon - rate[1] * 6, case.lat, case.lon)
        first = (float(east) / 6.0, float(north) / 6.0)
    msl = np.stack([got[h]["msl"] for h in need])
    tracked = track_in_fields((case.lat, case.lon), need, msl, box_lat, box_lon, first_motion_kmh=first)
    position = {h: tuple(tracked[k]) for k, h in enumerate(need)}
    native = {}
    for h in need:
        f = fields(h)
        if "msl_native" in f:
            native[h] = native_window(np.asarray(f["msl_native"], np.float64), tuple(np.asarray(f["native_grid"], float)), case.lat, case.lon)
    pmin_native = np.full(len(need), np.nan)
    fine = np.full((len(need), 2), np.nan)
    if len(native) == len(need):
        # On the model's own grid (0.25 or 0.5 degrees) the low is sharp where the 1.5-degree nodes smear it. Followed
        # there, Melissa's GFS and GraphCast-GFS tracks were no better than on the smoothed grid (the sharp minimum
        # wobbles; the smoothed one is a steadier centroid), so the smoothed track stays the reported one and the
        # native one travels in the extras with the model's true central pressure.
        axes = native[need[0]][1], native[need[0]][2]
        stack = np.stack([native[h][0] for h in need])
        fine = track_in_fields((case.lat, case.lon), need, stack, axes[0], axes[1], first_motion_kmh=first)
        lon2n, lat2n = np.meshgrid(axes[1], axes[0])
        for k, h in enumerate(need):
            if np.isfinite(fine[k, 0]):
                near = great_circle_km(lat2n, lon2n, fine[k, 0], fine[k, 1]) <= 150.0
                pmin_native[k] = np.nanmin(stack[k][near]) / 100.0
    lon2, lat2 = np.meshgrid(box_lon, box_lat)
    extras = {"msl_min_hpa": np.full(len(need), np.nan), "msl_env_hpa": np.full(len(need), np.nan),
              "spread_hpa": np.full(len(need), np.nan), "hours": np.asarray(need, float), "msl_min_native_hpa": pmin_native,
              "native_track": len(native) == len(need), "native_positions": fine}
    for k, h in enumerate(need):
        at = (case.lat, case.lon) if h == 0 else position[h]
        if np.isfinite(at[0]):
            d = great_circle_km(lat2, lon2, at[0], at[1])
            near, ring = d <= 250.0, (d > 600.0) & (d <= 900.0)
            extras["msl_min_hpa"][k] = np.nanmin(got[h]["msl"][near]) / 100.0           # the model's central pressure
            extras["msl_env_hpa"][k] = np.nanmean(got[h]["msl"][ring]) / 100.0          # and its environment's
            if "msl_spread" in got[h]:
                extras["spread_hpa"][k] = np.nanmean(got[h]["msl_spread"][near]) / 100.0
    data = np.zeros((len(hours), 2 * len(LAYERS), box_lat.size, box_lon.size), np.float32)
    last = (case.lat, case.lon)
    for k, h in enumerate(hours):
        at = (case.lat, case.lon) if h == 0 else position[int(h)]
        if np.isfinite(at[0]):
            last = at
        env = Environment(lat=box_lat.astype(np.float64), lon=box_lon.astype(np.float64), levels=tuple(levels),
                          fields={"u": got[int(h)]["u"], "v": got[int(h)]["v"]})
        try:
            clean = remove_vortex(env, *last).environment
        except Exception:                                # a degenerate box: steer with the raw winds rather than fail
            clean = env
        for j, (_, top, bottom) in enumerate(LAYERS):
            flow = SteeringFlow(clean, top, bottom)
            data[k, 2 * j], data[k, 2 * j + 1] = flow.u, flow.v
    model_track = np.array([position[int(h)] for h in leads], np.float64)
    return ForecastFlow(box_lat, box_lon, hours, data, source=source), model_track, extras


def fetch_models(case, when, names=DEFAULT_MODELS, cache: str | Path = "~/.cache/naturev1/models", leads=LEADS,
                 hours=FLOW_HOURS, log=print) -> dict:
    """
    Every model that can be read for the cycle at ``when``: ``{name: {"title", "flow", "track", "extras"}}``. A model
    that is not published yet or fails to read is reported through ``log`` and left out.
    """
    out = {}
    for name in names:
        spec = MODELS[name]
        try:
            flow, track, extras = LiveModel(spec, cache).forecast_flow(when, case, leads, hours)
            out[name] = {"title": spec.title, "flow": flow, "track": track, "extras": extras}
            followed = [h for h, p in zip(leads, track) if np.isfinite(np.asarray(p)[0])]
            log(f"{spec.title}: read; its storm followed to {followed[-1] if followed else 0} h")
        except Exception as error:
            log(f"{spec.title}: not used ({type(error).__name__}: {error})")
    return out



def fetch_qpf(init, bounds, hours: int = 120, step: int = 6, cache=None, log=print) -> dict | None:
    """
    The GFS cycle's own accumulated precipitation over ``bounds = (lat_min, lat_max, lon_min, lon_max)`` as 6-hourly
    buckets (mm) to ``hours``, read by byte range from the cycle's cumulative APCP records (about 0.4 MB each, 20 records
    for five days). Against gauge-corrected radar on four held-out US landfalls (eight cycles) the model's own rain beat
    the parametric R-CLIPER swath along the *observed* track on three (Ida: cell MAE 23 against 45 mm, correlation 0.78
    against 0.49) and lost on one (Ian), and a 50/50 blend of the two, each on the model's track, was the best or
    within noise of it on all four -- which is how :func:`naturev1.swarm.impact.paint` uses it: every world's rain is
    half its own R-CLIPER and half the model's field moved with the world (``ImpactSettings.qpf_weight``).

    Returns ``{"source", "init", "hours" (bucket end hours), "buckets" (T, ny, nx) mm, "lat" ascending, "lon" ascending}``
    or None when the cycle has no readable APCP yet. Cached under ``cache`` by cycle and bounds.
    """
    from .analysis import _get

    init = np.datetime64(init, "h")
    day, hour = str(init)[:10].replace("-", ""), int(str(init)[11:13])
    key = f"gfs_{str(init).replace(':', '')}_{bounds[0]:.1f}_{bounds[1]:.1f}_{bounds[2]:.1f}_{bounds[3]:.1f}.npz"
    path = Path(cache).expanduser() / key if cache is not None else None
    if path is not None and path.exists():
        z = np.load(path)
        return {"source": "GFS", "init": str(init), "hours": z["hours"], "buckets": z["buckets"], "lat": z["lat"], "lon": z["lon"]}

    def cumulative(lead):
        url = GFS_URL.format(day=day, hour=hour, lead=lead)
        want = f"0-{lead // 24} day acc fcst" if lead % 24 == 0 else f"0-{lead} hour acc fcst"
        lines = _get(url + ".idx").decode().splitlines()
        offsets = [int(line.split(":")[1]) for line in lines]
        for k, line in enumerate(lines):
            p = line.split(":")
            if len(p) > 5 and p[3] == "APCP" and p[5].strip() == want:
                raw = _get(url, offsets[k], offsets[k + 1] - 1 if k + 1 < len(lines) else None)
                return decode_grib_native(raw)
        raise KeyError(f"{url}: no APCP '{want}'")

    buckets, ends, prev, grid = [], [], None, None
    for lead in range(step, hours + 1, step):
        try:
            cum, grid = cumulative(lead)
        except Exception as error:
            log(f"GFS rain: stopped at {lead - step} h ({type(error).__name__})")
            break
        buckets.append(cum if prev is None else cum - prev)
        ends.append(lead)
        prev = cum
    if not buckets:
        return None
    lat0, dlat, lon0, dlon = grid
    stack = np.stack(buckets)
    lat = lat0 - dlat * np.arange(stack.shape[1])
    lon = (lon0 + dlon * np.arange(stack.shape[2]) + 180.0) % 360.0 - 180.0
    order = np.argsort(lon)
    lon, stack = lon[order], stack[:, :, order]
    if lat[0] > lat[-1]:
        lat, stack = lat[::-1], stack[:, ::-1, :]
    i = (lat >= bounds[0]) & (lat <= bounds[1])
    j = (lon >= bounds[2]) & (lon <= bounds[3])
    out = {"source": "GFS", "init": str(init), "hours": np.asarray(ends, float), "buckets": np.clip(stack[:, i][:, :, j], 0.0, None).astype(np.float32),
           "lat": lat[i], "lon": lon[j]}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **out)
    log(f"GFS rain: {len(ends)} six-hour buckets to {ends[-1]} h over {out['lat'].size} x {out['lon'].size} cells")
    return out

__all__ = ["fetch_qpf", "DEFAULT_MODELS", "MODELS", "LiveModel", "ModelSpec", "coarsen_to_nodes", "decode_grib", "decode_grib_native",
           "fetch_models", "flow_and_track", "native_window"]
