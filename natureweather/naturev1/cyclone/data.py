# Copyright 2026 Nathan. Apache-2.0.
"""
Hindcast cases: every best-track fix that a forecast centre would have issued a forecast from, with the
analysis around the storm at that moment and what the storm actually did afterwards.

The idea: a track model is judged the way the National Hurricane Center judges its own -- homogeneous
samples, forecasts only from tropical or subtropical cyclones (depressions included), verified only while
the system is still one, at fixed leads -- on thousands of cases across every basin, never on a single
storm. Each case stores only what was knowable at issue time (the best track up to now and the analysis
now and 12 hours ago), so nothing downstream can peek at the future; the verifying positions are kept
apart, in ``truth``.

Analysis boxes come from WeatherBench 2's 1.5-degree ERA5 (the grid NatureV1 runs on). Reading is
chunk-aware -- the store keeps 8 times per chunk, so cases are processed in time order and every chunk is
read once for all the storms active in it -- and threaded. Boxes are stored as float16 after a per-case,
per-channel standardisation (geopotential would overflow float16 otherwise), in one memory-mapped array per
season, resumable season by season.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .fields import Environment

WB2_ERA5 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
PREVIOUS_LEVELS = (200, 250, 300, 400, 500, 600, 700, 850)
LEVEL_VARIABLES = {"u": "u_component_of_wind", "v": "v_component_of_wind", "z": "geopotential",
                   "t": "temperature", "q": "specific_humidity"}
SURFACE_VARIABLES = {"msl": "mean_sea_level_pressure", "sst": "sea_surface_temperature",
                     "tcwv": "total_column_water_vapour", "u10": "10m_u_component_of_wind",
                     "v10": "10m_v_component_of_wind"}
STATIC_VARIABLES = {"lsm": "land_sea_mask", "zs": "geopotential_at_surface"}
#: Leads verified, hours.
LEADS = (6, 12, 24, 36, 48, 72, 96, 120)
#: Box half-widths, degrees, and the grid step.
HALF_LAT, HALF_LON, STEP = 30.0, 45.0, 1.5
TROPICAL = ("TD", "TS", "HU", "SS")
HISTORY_HOURS = (-48, -42, -36, -30, -24, -18, -12, -6, 0)


def channel_names() -> list[str]:
    names = [f"{k}@{p}" for k in LEVEL_VARIABLES for p in LEVELS]
    names += list(SURFACE_VARIABLES) + list(STATIC_VARIABLES)
    names += [f"prev_{k}@{p}" for k in ("u", "v") for p in PREVIOUS_LEVELS] + ["prev_z@500", "prev_msl"]
    return names


@dataclass
class Case:
    """One forecast opportunity. Everything outside ``truth`` was known at ``time``."""

    storm_id: str
    name: str
    basin: str
    time: np.datetime64
    lat: float
    lon: float
    wind_kt: float
    pressure_hpa: float
    status: str
    history: dict = field(default_factory=dict)      # HISTORY_HOURS -> lat, lon, wind, pressure (NaN where absent)
    rmw_nmi: float = float("nan")
    r34_nmi: float = float("nan")                     # mean of the reported quadrants
    distance_to_land_km: float = float("nan")
    truth: dict = field(default_factory=dict)         # LEADS -> lat, lon, wind, pressure, valid
    index: int = -1                                   # row in the season's box array

    @property
    def year(self) -> int:
        return int(str(self.time)[:4])

    @property
    def key(self) -> str:
        return f"{self.storm_id}@{str(self.time)[:13]}"


def _fix_lookup(track) -> dict[int, int]:
    seconds = track.time.astype("datetime64[s]").astype(np.int64)
    return {int(s): k for k, s in enumerate(seconds)}


def cases_from_tracks(tracks, years, init_hours=(0, 12), leads=LEADS) -> list[Case]:
    """
    Every fix at ``init_hours`` UTC in ``years`` (calendar year of the fix) where the system is a tropical or
    subtropical cyclone and at least one earlier fix 6 or 12 hours back exists (so extrapolation is defined).
    """
    years = set(int(y) for y in years)
    out = []
    for track in tracks:
        lookup = _fix_lookup(track)
        seconds = track.time.astype("datetime64[s]").astype(np.int64)
        for k in range(len(track)):
            t = int(seconds[k])
            if (t // 3600) % 24 not in init_hours or track.status[k] not in TROPICAL:
                continue
            if int(str(track.time[k])[:4]) not in years:
                continue
            if not (np.isfinite(track.latitude[k]) and np.isfinite(track.longitude[k])):
                continue
            if lookup.get(t - 6 * 3600) is None and lookup.get(t - 12 * 3600) is None:
                continue
            history = {"lat": [], "lon": [], "wind": [], "pressure": []}
            for h in HISTORY_HOURS:
                j = lookup.get(t + h * 3600)
                for key, arr in (("lat", track.latitude), ("lon", track.longitude), ("wind", track.max_wind_kt),
                                 ("pressure", track.min_pressure_hpa)):
                    history[key].append(float(arr[j]) if j is not None else float("nan"))
            truth = {"lat": [], "lon": [], "wind": [], "pressure": [], "valid": []}
            for lead in leads:
                j = lookup.get(t + lead * 3600)
                ok = j is not None and track.status[j] in TROPICAL and np.isfinite(track.latitude[j])
                for key, arr in (("lat", track.latitude), ("lon", track.longitude), ("wind", track.max_wind_kt),
                                 ("pressure", track.min_pressure_hpa)):
                    truth[key].append(float(arr[j]) if ok else float("nan"))
                truth["valid"].append(bool(ok))
            radii = track.wind_radii_nmi[k, 0]
            out.append(Case(
                storm_id=track.storm_id, name=track.name, basin=track.basin, time=track.time[k].astype("datetime64[s]"),
                lat=float(track.latitude[k]), lon=float(track.longitude[k]), wind_kt=float(track.max_wind_kt[k]),
                pressure_hpa=float(track.min_pressure_hpa[k]), status=str(track.status[k]),
                history={key: np.asarray(v, np.float64) for key, v in history.items()},
                rmw_nmi=float(track.rmw_nmi[k]), r34_nmi=float(np.nanmean(radii)) if np.isfinite(radii).any() else float("nan"),
                distance_to_land_km=float(track.distance_to_land_km[k]) if track.distance_to_land_km is not None else float("nan"),
                truth={key: np.asarray(v, np.float64 if key != "valid" else bool) for key, v in truth.items()},
            ))
    out.sort(key=lambda c: (c.time, c.storm_id))
    return out


# ------------------------------------------------------------------------------------- serialise --

_SCALARS = ("storm_id", "name", "basin", "lat", "lon", "wind_kt", "pressure_hpa", "status", "rmw_nmi", "r34_nmi",
            "distance_to_land_km", "index")


def save_cases(cases: list[Case], path: str | Path) -> None:
    payload = {key: np.asarray([getattr(c, key) for c in cases]) for key in _SCALARS}
    payload["time"] = np.asarray([c.time for c in cases], dtype="datetime64[s]").astype(np.int64)
    for key in ("lat", "lon", "wind", "pressure"):
        payload[f"history_{key}"] = np.stack([c.history[key] for c in cases]) if cases else np.zeros((0, 9))
    for key in ("lat", "lon", "wind", "pressure", "valid"):
        payload[f"truth_{key}"] = np.stack([c.truth[key] for c in cases]) if cases else np.zeros((0, len(LEADS)))
    np.savez(path, **payload)


def load_cases(path: str | Path) -> list[Case]:
    data = np.load(path, allow_pickle=False)
    out = []
    for i in range(len(data["time"])):
        out.append(Case(
            storm_id=str(data["storm_id"][i]), name=str(data["name"][i]), basin=str(data["basin"][i]),
            time=np.datetime64(int(data["time"][i]), "s"), lat=float(data["lat"][i]), lon=float(data["lon"][i]),
            wind_kt=float(data["wind_kt"][i]), pressure_hpa=float(data["pressure_hpa"][i]), status=str(data["status"][i]),
            history={key: data[f"history_{key}"][i] for key in ("lat", "lon", "wind", "pressure")},
            rmw_nmi=float(data["rmw_nmi"][i]), r34_nmi=float(data["r34_nmi"][i]),
            distance_to_land_km=float(data["distance_to_land_km"][i]),
            truth={key: data[f"truth_{key}"][i] for key in ("lat", "lon", "wind", "pressure", "valid")},
            index=int(data["index"][i]),
        ))
    return out


# --------------------------------------------------------------------------------------- reading --

class WB2Reader:
    """
    Chunk-cached, threaded reads of WeatherBench 2 ERA5. ``fields(t)`` returns every variable at time ``t`` as
    ``(level, lat, lon)`` / ``(lat, lon)`` arrays (latitude ascending, longitude 0..358.5).
    """

    def __init__(self, store: str = WB2_ERA5, threads: int = 8, cache_chunks: int = 3) -> None:
        import xarray as xr

        self.ds = xr.open_zarr(store, storage_options={"token": "anon"}, chunks=None)
        self.t0 = np.datetime64(self.ds.time.values[0], "s").astype(np.int64)
        self.step = int((np.datetime64(self.ds.time.values[1], "s") - np.datetime64(self.ds.time.values[0], "s"))
                        / np.timedelta64(1, "s"))
        self.chunk = int(self.ds["u_component_of_wind"].encoding.get("chunks", (8,))[0])
        self.levels = tuple(int(p) for p in self.ds.level.values)
        self.lat = self.ds.latitude.values.astype(np.float64)
        self.lon = self.ds.longitude.values.astype(np.float64)
        self.pool = cf.ThreadPoolExecutor(threads)
        self.cache: OrderedDict[int, dict] = OrderedDict()
        self.cache_chunks = cache_chunks
        self.static = {k: self._orient(np.asarray(self.ds[v].values, np.float32)) for k, v in STATIC_VARIABLES.items()}

    def _orient(self, array: np.ndarray) -> np.ndarray:
        """(..., lon, lat) -> (..., lat, lon), latitude ascending."""
        array = np.swapaxes(array, -1, -2)
        if self.lat[0] > self.lat[-1]:
            array = array[..., ::-1, :]
        return array

    def _load_chunk(self, number: int) -> dict:
        start, stop = number * self.chunk, (number + 1) * self.chunk

        def read(item):
            key, name = item
            return key, self._orient(np.asarray(self.ds[name].isel(time=slice(start, stop)).values, np.float32))

        items = list(LEVEL_VARIABLES.items()) + list(SURFACE_VARIABLES.items())
        return dict(self.pool.map(read, items))

    def fields(self, when: np.datetime64) -> dict:
        index = int((np.datetime64(when, "s").astype(np.int64) - self.t0) // self.step)
        number, offset = divmod(index, self.chunk)
        if number not in self.cache:
            self.cache[number] = self._load_chunk(number)
            while len(self.cache) > self.cache_chunks:
                self.cache.popitem(last=False)
        self.cache.move_to_end(number)
        chunk = self.cache[number]
        out = {key: value[offset] for key, value in chunk.items()}
        out.update(self.static)
        return out


def _box_indices(lat0: float, lon0: float, lat_axis: np.ndarray, lon_step: float, rows: int, cols: int):
    """Row range (kept inside the globe) and wrapped column indices with continuous box longitudes."""
    step = lat_axis[1] - lat_axis[0]
    centre = int(round((lat0 - lat_axis[0]) / step))
    start = int(np.clip(centre - rows // 2, 0, lat_axis.size - rows))
    j0 = int(round((lon0 % 360.0) / lon_step))
    offsets = np.arange(cols) - cols // 2
    columns = (j0 + offsets) % int(round(360.0 / lon_step))
    box_lon = (j0 + offsets) * lon_step
    if box_lon[cols // 2] - lon0 > 180.0:
        box_lon = box_lon - 360.0
    return slice(start, start + rows), columns, lat_axis[start:start + rows], box_lon


def build_box(now: dict, before: dict | None, lat0: float, lon0: float, lat_axis: np.ndarray, levels) -> tuple:
    """``(boxes (C, H, W) float32, lat (H,), lon (W,))`` for one case, channels in :func:`channel_names` order."""
    rows, cols = int(round(2 * HALF_LAT / STEP)) + 1, int(round(2 * HALF_LON / STEP)) + 1
    rs, cs, box_lat, box_lon = _box_indices(lat0, lon0, lat_axis, STEP, rows, cols)
    order = [list(levels).index(p) for p in LEVELS]
    planes = []
    for key in LEVEL_VARIABLES:
        planes.extend(now[key][order][:, rs][:, :, cs])
    for key in list(SURFACE_VARIABLES) + list(STATIC_VARIABLES):
        planes.append(now[key][rs][:, cs])
    previous_order = [list(levels).index(p) for p in PREVIOUS_LEVELS]
    source = before if before is not None else now
    for key in ("u", "v"):
        planes.extend(source[key][previous_order][:, rs][:, :, cs])
    planes.append(source["z"][list(levels).index(500)][rs][:, cs])
    planes.append(source["msl"][rs][:, cs])
    return np.stack(planes).astype(np.float32), box_lat, box_lon


def _standardise(box: np.ndarray):
    mean = np.nanmean(box, axis=(1, 2))
    std = np.nanstd(box, axis=(1, 2))
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    scaled = (box - mean[:, None, None]) / std[:, None, None]
    return scaled.astype(np.float16), np.stack([mean, std], -1).astype(np.float32)


def build_season(cases: list[Case], directory: str | Path, reader: WB2Reader | None = None, log=print) -> Path:
    """
    Read the analysis box for every case (in time order, one chunk read for all storms in it) into
    ``directory``: ``cases.npz``, ``boxes.npy`` (N, C, H, W) float16, ``affine.npy`` (N, C, 2), ``grid.npz``
    (per-case box latitude and longitude), ``channels.json``. Resumable: a finished directory is left alone.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "done").exists():
        return directory
    reader = reader or WB2Reader()
    names = channel_names()
    rows, cols = int(round(2 * HALF_LAT / STEP)) + 1, int(round(2 * HALF_LON / STEP)) + 1
    boxes = np.lib.format.open_memmap(directory / "boxes.npy", mode="w+", dtype=np.float16,
                                      shape=(len(cases), len(names), rows, cols))
    affine = np.zeros((len(cases), len(names), 2), np.float32)
    grid_lat = np.zeros((len(cases), rows), np.float32)
    grid_lon = np.zeros((len(cases), cols), np.float32)
    for i, case in enumerate(cases):
        now = reader.fields(case.time)
        before = reader.fields(case.time - np.timedelta64(12, "h"))
        box, lat, lon = build_box(now, before, case.lat, case.lon, reader.lat, reader.levels)
        boxes[i], affine[i] = _standardise(box)
        grid_lat[i], grid_lon[i] = lat, lon
        case.index = i
        if log and (i % 200 == 0 or i == len(cases) - 1):
            log(f"[cyclone-data] {directory.name}: {i + 1}/{len(cases)} {case.key}")
    boxes.flush()
    np.save(directory / "affine.npy", affine)
    np.savez(directory / "grid.npz", lat=grid_lat, lon=grid_lon)
    (directory / "channels.json").write_text(json.dumps(names))
    save_cases(cases, directory / "cases.npz")
    (directory / "done").write_text("ok")
    return directory


class Season:
    """A built season: its cases and, on demand, each case's :class:`Environment`."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.cases = load_cases(self.directory / "cases.npz")
        self.boxes = np.load(self.directory / "boxes.npy", mmap_mode="r")
        self.affine = np.load(self.directory / "affine.npy")
        grid = np.load(self.directory / "grid.npz")
        self.grid_lat, self.grid_lon = grid["lat"], grid["lon"]
        self.names = json.loads((self.directory / "channels.json").read_text())
        self.slot = {n: k for k, n in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.cases)

    def environment(self, i: int) -> Environment:
        case = self.cases[i]
        box = self.boxes[case.index].astype(np.float32)
        a = self.affine[case.index]
        box = box * a[:, 1, None, None] + a[:, 0, None, None]
        take = lambda name: box[self.slot[name]].astype(np.float64)
        fields = {k: np.stack([take(f"{k}@{p}") for p in LEVELS]) for k in LEVEL_VARIABLES}
        for k in list(SURFACE_VARIABLES) + list(STATIC_VARIABLES):
            fields[k] = take(k)
        previous = {k: np.stack([take(f"prev_{k}@{p}") for p in PREVIOUS_LEVELS]) for k in ("u", "v")}
        previous["z500"] = take("prev_z@500")
        previous["msl"] = take("prev_msl")
        env = Environment(lat=self.grid_lat[case.index].astype(np.float64), lon=self.grid_lon[case.index].astype(np.float64),
                          levels=LEVELS, fields=fields, previous=previous, time=case.time)
        env.previous_levels = PREVIOUS_LEVELS
        return env


def build_seasons(ibtracs_paths: dict[str, str], root: str | Path, years, basins=("NA", "EP", "WP", "SI", "SP"),
                  log=print) -> dict[int, Path]:
    """Build (or reuse) one :func:`build_season` directory per calendar year, all basins together."""
    from ..ibtracs import parse_ibtracs

    years = sorted(int(y) for y in years)
    tracks = []
    for basin in basins:
        tracks += parse_ibtracs(ibtracs_paths[basin], since=years[0] - 1, until=years[-1] + 1, basins=(basin,))
    reader = None
    out = {}
    for year in years:
        directory = Path(root) / str(year)
        if not (directory / "done").exists():
            reader = reader or WB2Reader()
            cases = cases_from_tracks(tracks, [year])
            log(f"[cyclone-data] {year}: {len(cases)} cases")
            build_season(cases, directory, reader, log)
        out[year] = directory
    return out


__all__ = ["Case", "HISTORY_HOURS", "LEADS", "LEVELS", "Season", "WB2Reader", "build_box", "build_season",
           "build_seasons", "cases_from_tracks", "channel_names", "load_cases", "save_cases"]
