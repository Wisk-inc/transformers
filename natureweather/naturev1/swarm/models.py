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


def decode_grib(message: bytes) -> np.ndarray:
    """One GRIB message (any regular grid, either row order, longitudes from -180 or 0) onto the 1.5-degree nodes."""
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
    return coarsen_to_nodes(values, lat0, dlat, float(lon[shift]), dlon)


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
        out = {"u": np.stack([self._read(base, index, k) for k in self._keys("u")]),
               "v": np.stack([self._read(base, index, k) for k in self._keys("v")]),
               "msl": self._read(base, index, self._keys("msl")[0]), "levels": np.asarray(self.spec.levels, np.float64)}
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
    lon2, lat2 = np.meshgrid(box_lon, box_lat)
    extras = {"msl_min_hpa": np.full(len(need), np.nan), "msl_env_hpa": np.full(len(need), np.nan),
              "spread_hpa": np.full(len(need), np.nan), "hours": np.asarray(need, float)}
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
            lost = int(np.isfinite(track[:, 0]).sum())
            log(f"{spec.title}: read; its storm followed to {leads[lost - 1] if lost else 0} h")
        except Exception as error:
            log(f"{spec.title}: not used ({type(error).__name__}: {error})")
    return out


__all__ = ["DEFAULT_MODELS", "MODELS", "LiveModel", "ModelSpec", "coarsen_to_nodes", "decode_grib", "fetch_models",
           "flow_and_track"]
