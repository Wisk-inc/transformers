# Copyright 2026 Nathan. Apache-2.0.
"""
Storm-centred infrared satellite imagery: what a 1.5-degree reanalysis cannot see.

ERA5 at 1.5 degrees smears a 30 km eyewall into a broad low, so a tropical storm and a category 5 look
alike to the backbone -- the red team measured intensity biased low by 10-18 kt for major hurricanes and
blind to explosive intensification. Forecasters estimate intensity from exactly what is missing: the
pattern of cold cloud tops in geostationary infrared (the Dvorak technique). This module supplies that
pattern, as one consistent product from 1978 to today:

* **HURSAT-B1** (NOAA NCEI), storm-centred, 1978-2016: small per-storm archives, used for training.
* **GridSat-B1** (NOAA NCEI), global, 1980-present: the same ISCCP B1 record on the same 0.07-degree grid,
  used for 2017 onward -- held-out seasons, later seasons, recent dates. On Katrina 2005-08-29 12Z the two
  agree with correlation 0.989 and a mean difference of 1 K.
* **GOES ABI band 13** (10.3 um), live: resampled onto the same storm-centred 0.07-degree patch.

Every source becomes the same thing: a ``PATCH_PIXELS`` square of 11-micron brightness temperature in
kelvin, 0.07 degrees per pixel (about 8 km, about 18 degrees across), centred on the storm, north up, NaN
where there is no data. Patches are stored as one byte per pixel (0.67 K steps) so 16,000 training samples
fit in two gigabytes.
"""

from __future__ import annotations

import concurrent.futures as cf
import csv
import datetime as dt
import hashlib
import math
import re
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

#: Pixels on a side of a storm patch, and degrees per pixel (HURSAT-B1 and GridSat-B1 native spacing).
PATCH_PIXELS = 256
PATCH_DEGREES = 0.07
#: Brightness temperatures outside this range are not cloud or surface; they are bad data.
BT_RANGE_K = (160.0, 330.0)
#: One byte per pixel: 0 is missing, 1-255 cover BT_RANGE_K in 2/3 K steps.
_BT_STEP = (BT_RANGE_K[1] - BT_RANGE_K[0]) / 254.0

HURSAT_URL = "https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06"
GRIDSAT_URL = "https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/access"
IBTRACS_NA_URL = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
                  "v04r01/access/csv/ibtracs.NA.list.v04r01.csv")
#: Last season HURSAT-B1 v06 covers; later seasons come from GridSat-B1.
HURSAT_LAST_SEASON = 2016


# ------------------------------------------------------------------------------------ patch geometry --

def patch_axes(center: tuple[float, float], pixels: int = PATCH_PIXELS, step: float = PATCH_DEGREES):
    """Latitudes (south to north) and longitudes (west to east, -180..180) of a storm patch's pixels."""
    offsets = (np.arange(pixels) - (pixels - 1) / 2.0) * step
    latitudes = center[0] + offsets
    longitudes = (center[1] + offsets + 180.0) % 360.0 - 180.0
    return latitudes, longitudes


def crop_regular(lat_axis: np.ndarray, lon_axis: np.ndarray, field: np.ndarray,
                 center: tuple[float, float], pixels: int = PATCH_PIXELS) -> np.ndarray:
    """
    Cut a storm patch out of a field on a regular latitude/longitude grid, nearest pixel.

    ``lat_axis`` may run either way; ``lon_axis`` is any regular axis, and the patch wraps across the
    date line when the grid is global. Pixels outside the grid come back NaN.
    """
    lat_axis = np.asarray(lat_axis, np.float64)
    lon_axis = np.asarray(lon_axis, np.float64)
    want_lat, want_lon = patch_axes(center, pixels)
    lat_step = lat_axis[1] - lat_axis[0]
    lon_step = lon_axis[1] - lon_axis[0]
    rows = np.rint((want_lat - lat_axis[0]) / lat_step).astype(np.int64)
    rows_ok = (rows >= 0) & (rows < lat_axis.size)
    span = lon_step * lon_axis.size
    global_lon = abs(abs(span) - 360.0) < 2 * abs(lon_step)
    # Distance east of the grid's first column, wrapped so a regional grid near the date line still works.
    offset = (want_lon - lon_axis[0]) % 360.0 if global_lon else (want_lon - lon_axis[0] + 180.0) % 360.0 - 180.0
    cols = np.rint(offset / lon_step).astype(np.int64)
    if global_lon:
        cols %= lon_axis.size
    cols_ok = (cols >= 0) & (cols < lon_axis.size)
    out = np.full((pixels, pixels), np.nan, np.float32)
    good_rows, good_cols = np.where(rows_ok)[0], np.where(cols_ok)[0]
    if good_rows.size and good_cols.size:
        out[np.ix_(good_rows, good_cols)] = field[np.ix_(rows[good_rows], cols[good_cols])]
    return _clean(out)


def _clean(patch: np.ndarray) -> np.ndarray:
    patch = np.asarray(patch, np.float32)
    return np.where((patch >= BT_RANGE_K[0]) & (patch <= BT_RANGE_K[1]), patch, np.nan).astype(np.float32)


def encode_bt(patch: np.ndarray) -> np.ndarray:
    """Kelvin -> one byte per pixel (0 = missing)."""
    scaled = np.rint((np.asarray(patch, np.float64) - BT_RANGE_K[0]) / _BT_STEP) + 1
    scaled = np.where(np.isfinite(scaled), np.clip(scaled, 1, 255), 0)
    return scaled.astype(np.uint8)


def decode_bt(codes) -> np.ndarray:
    """One byte per pixel -> kelvin (NaN where missing). Works on numpy arrays and torch tensors."""
    if hasattr(codes, "float"):                                  # torch
        import torch

        values = (codes.float() - 1) * _BT_STEP + BT_RANGE_K[0]
        return torch.where(codes > 0, values, torch.full_like(values, float("nan")))
    codes = np.asarray(codes)
    return np.where(codes > 0, (codes.astype(np.float32) - 1) * _BT_STEP + BT_RANGE_K[0], np.nan).astype(np.float32)


# --------------------------------------------------------------------------------------------- GOES --

def latlon_to_fixed_grid(latitude, longitude, satellite_longitude: float,
                         perspective_height: float = 35786023.0, semi_major: float = 6378137.0,
                         semi_minor: float = 6356752.31414):
    """
    GOES fixed-grid scan angles ``(x, y)`` in radians for geodetic latitude/longitude in degrees.

    The forward projection of the GOES-R Product User Guide (Vol. 3, 5.1.2.8.2), the inverse of
    :func:`naturev1.satellite.fixed_grid_to_latlon`. Points the satellite cannot see come back NaN.
    """
    lat, lon = np.radians(latitude), np.radians(longitude)
    height = perspective_height + semi_major
    e2 = (semi_major**2 - semi_minor**2) / semi_major**2
    lat_c = np.arctan((semi_minor**2 / semi_major**2) * np.tan(lat))
    r_c = semi_minor / np.sqrt(1.0 - e2 * np.cos(lat_c) ** 2)
    dlon = lon - np.radians(satellite_longitude)
    sx = height - r_c * np.cos(lat_c) * np.cos(dlon)
    sy = -r_c * np.cos(lat_c) * np.sin(dlon)
    sz = r_c * np.sin(lat_c)
    visible = height * (height - sx) >= sy**2 + (semi_major**2 / semi_minor**2) * sz**2
    x = np.arcsin(-sy / np.sqrt(sx**2 + sy**2 + sz**2))
    y = np.arctan(sz / sx)
    return np.where(visible, x, np.nan), np.where(visible, y, np.nan)


def goes_patch(path: str | Path, center: tuple[float, float], pixels: int = PATCH_PIXELS,
               footprint: int = 5) -> np.ndarray:
    """
    A storm patch from a GOES ABI band 13 Cloud and Moisture Imagery file (``ABI-L2-CMIP*``, variable
    ``CMI``), on the same 0.07-degree storm-centred grid as HURSAT-B1 and GridSat-B1.

    Each patch pixel takes the mean of the ``footprint`` x ``footprint`` ABI pixels nearest its centre:
    ABI band 13 is 2 km at nadir, the B1 record about 8 km, so a small average keeps a single 2 km
    pixel from standing in for an 8 km one.
    """
    import xarray as xr

    with xr.open_dataset(path) as dataset:
        projection = dataset["goes_imager_projection"].attrs
        x_axis = dataset["x"].values.astype(np.float64)
        y_axis = dataset["y"].values.astype(np.float64)
        cmi = dataset["CMI"].values.astype(np.float32)
        lats, lons = patch_axes(center, pixels)
        grid_lon, grid_lat = np.meshgrid(lons, lats)
        x, y = latlon_to_fixed_grid(grid_lat, grid_lon, float(projection["longitude_of_projection_origin"]),
                                    float(projection["perspective_point_height"]),
                                    float(projection["semi_major_axis"]), float(projection["semi_minor_axis"]))
    col = np.rint((x - x_axis[0]) / (x_axis[1] - x_axis[0]))
    row = np.rint((y - y_axis[0]) / (y_axis[1] - y_axis[0]))
    ok = np.isfinite(col) & np.isfinite(row)
    total = np.zeros(col.shape, np.float64)
    count = np.zeros(col.shape, np.float64)
    half = footprint // 2
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            r = np.where(ok, row + dr, -1).astype(np.int64)
            c = np.where(ok, col + dc, -1).astype(np.int64)
            inside = ok & (r >= 0) & (r < cmi.shape[0]) & (c >= 0) & (c < cmi.shape[1])
            values = np.full(col.shape, np.nan, np.float32)
            values[inside] = cmi[r[inside], c[inside]]
            good = np.isfinite(values)
            total[good] += values[good]
            count[good] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        patch = np.where(count > 0, total / count, np.nan)
    return _clean(patch)


# ------------------------------------------------------------------------------------------ archives --

def _fetch(url: str, attempts: int = 4, timeout: float = 120.0) -> bytes:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2.0 * 2**attempt)
    raise RuntimeError(url)


def ibtracs_atcf_map(csv_path: str | Path) -> dict[str, str]:
    """``{"AL122005": "2005236N23285", ...}``: HURDAT2 / ATCF storm ids to IBTrACS SIDs, which HURSAT uses."""
    mapping = {}
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        next(reader, None)                                         # the units row
        for row in reader:
            atcf = row.get("USA_ATCF_ID", "").strip()
            if atcf and atcf not in mapping:
                mapping[atcf.upper()] = row["SID"].strip()
    return mapping


class HursatArchive:
    """
    HURSAT-B1 v06, one tarball per storm, downloaded once into ``cache`` and read from there.

    A synoptic time usually has several satellites; the one viewing the storm most directly (the smallest
    view angle, which is in the file name) is used.
    """

    _NAME = re.compile(r"^(?P<sid>\d{7}[NS]\d{5})\.[^.]+\.(?P<y>\d{4})\.(?P<m>\d\d)\.(?P<d>\d\d)\.(?P<hm>\d{4})\."
                       r"(?P<angle>\d+)\.(?P<sat>[^.]+)\.")

    def __init__(self, cache: str | Path, atcf_to_sid: dict[str, str]):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.atcf_to_sid = atcf_to_sid
        self._listing: dict[int, list[str]] = {}
        self._index: dict[str, dict[np.datetime64, str] | None] = {}

    def _year_files(self, year: int) -> list[str]:
        if year not in self._listing:
            page = _fetch(f"{HURSAT_URL}/{year}/").decode("utf-8", "replace")
            self._listing[year] = sorted(set(re.findall(r'HURSAT_b1_v06_[^"<>]+?\.tar\.gz', page)))
        return self._listing[year]

    def _tarball(self, sid: str, year: int) -> Path | None:
        local = self.cache / f"{sid}.tar.gz"
        if local.exists() and local.stat().st_size > 0:
            return local
        names = [n for n in self._year_files(year) if f"_{sid}_" in n]
        if not names:
            return None
        partial = local.with_suffix(".part")
        partial.write_bytes(_fetch(f"{HURSAT_URL}/{year}/{names[0]}"))
        partial.replace(local)
        return local

    def _members(self, sid: str, year: int) -> dict[np.datetime64, str] | None:
        if sid not in self._index:
            path = self._tarball(sid, year)
            if path is None:
                self._index[sid] = None
            else:
                best: dict[np.datetime64, tuple[int, str]] = {}
                with tarfile.open(path) as archive:
                    for member in archive.getnames():
                        match = self._NAME.match(Path(member).name)
                        if not match:
                            continue
                        when = np.datetime64(f"{match['y']}-{match['m']}-{match['d']}T{match['hm'][:2]}:{match['hm'][2:]}", "m")
                        angle = int(match["angle"])
                        if when not in best or angle < best[when][0]:
                            best[when] = (angle, member)
                self._index[sid] = {when: member for when, (_, member) in best.items()}
        return self._index[sid]

    def patch(self, storm_id: str, when, center: tuple[float, float], tolerance_minutes: int = 90) -> np.ndarray | None:
        """The storm patch nearest ``when`` (within the tolerance), or None if HURSAT has no image."""
        return self.storm_patches(storm_id, [(when, center)], tolerance_minutes)[0]

    def storm_patches(self, storm_id: str, requests: list[tuple[np.datetime64, tuple[float, float]]],
                      tolerance_minutes: int = 90) -> list[np.ndarray | None]:
        """
        Every requested ``(time, centre)`` patch of one storm, reading its archive once.

        A storm's tarball is compressed as a whole, so pulling images out one at a time decompresses it
        again for each; one pass over it for all of a storm's samples is what makes 30,000 images minutes.
        """
        import xarray as xr

        sid = self.atcf_to_sid.get(storm_id.upper())
        if sid is None or not requests:
            return [None] * len(requests)
        year = int(str(np.datetime64(requests[0][0], "Y")))
        members = self._members(sid, year)
        if not members:
            return [None] * len(requests)
        wanted: dict[str, list[int]] = {}
        for i, (when, _) in enumerate(requests):
            target = np.datetime64(when, "m")
            nearest = min(members, key=lambda t: abs(t - target))
            if abs(nearest - target) <= np.timedelta64(tolerance_minutes, "m"):
                wanted.setdefault(members[nearest], []).append(i)
        out: list[np.ndarray | None] = [None] * len(requests)
        with tarfile.open(self._tarball(sid, year)) as archive:
            for member in archive:
                if member.name not in wanted:
                    continue
                raw = archive.extractfile(member).read()
                # Through a temporary file rather than memory: any installed netCDF engine can read a path.
                with tempfile.NamedTemporaryFile(suffix=".nc") as handle:
                    handle.write(raw)
                    handle.flush()
                    with xr.open_dataset(handle.name) as dataset:
                        field = dataset["IRWIN"].values[0].astype(np.float32)
                        lat, lon = dataset["lat"].values, dataset["lon"].values
                for i in wanted[member.name]:
                    out[i] = crop_regular(lat, lon, field, requests[i][1])
        return out


class GridSatArchive:
    """
    GridSat-B1 v02r01: one global 0.07-degree file per three hours (about 35 MB). Each file is downloaded,
    every storm patch wanted at that time is cut from it, and it is deleted -- only the patches are kept.
    """

    def __init__(self, workers: int = 4):
        self.workers = workers

    @staticmethod
    def url(when) -> str:
        stamp = np.datetime64(when, "h").astype(dt.datetime)
        return f"{GRIDSAT_URL}/{stamp:%Y}/GRIDSAT-B1.{stamp:%Y.%m.%d.%H}.v02r01.nc"

    def _patches_at(self, when, centers: list[tuple[float, float]]) -> list[np.ndarray | None]:
        import xarray as xr

        try:
            raw = _fetch(self.url(when))
        except Exception:
            return [None] * len(centers)
        with tempfile.NamedTemporaryFile(suffix=".nc") as handle:
            handle.write(raw)
            handle.flush()
            with xr.open_dataset(handle.name) as dataset:
                field = dataset["irwin_cdr"].values[0].astype(np.float32)
                lat, lon = dataset["lat"].values, dataset["lon"].values
        return [crop_regular(lat, lon, field, c) for c in centers]

    def patches(self, requests: list[tuple[np.datetime64, tuple[float, float]]],
                progress: bool = True) -> list[np.ndarray | None]:
        """One patch per ``(time, centre)`` request, None where GridSat has no file."""
        by_time: dict[np.datetime64, list[int]] = {}
        for i, (when, _) in enumerate(requests):
            by_time.setdefault(np.datetime64(when, "h"), []).append(i)
        out: list[np.ndarray | None] = [None] * len(requests)
        times = sorted(by_time)
        started = time.time()
        with cf.ThreadPoolExecutor(self.workers) as pool:
            futures = {pool.submit(self._patches_at, t, [requests[i][1] for i in by_time[t]]): t for t in times}
            for done, future in enumerate(cf.as_completed(futures), start=1):
                for i, patch in zip(by_time[futures[future]], future.result()):
                    out[i] = patch
                if progress and (done % 25 == 0 or done == len(times)):
                    left = (time.time() - started) / done * (len(times) - done) / 60
                    print(f"[stormsat] GridSat {done:,}/{len(times):,} times ({left:.0f} min left)", flush=True)
        return out


# ----------------------------------------------------------------------------------- satellite bank --

def sample_keys(targets) -> list[tuple[str, np.datetime64, float, float]]:
    """``(storm id, time, lat, lon)`` of each storm sample, in bank order."""
    keys = []
    for target in targets:
        track, point = target.track, target.point
        keys.append((track.storm_id, np.datetime64(track.time[point], "m"),
                     float(track.latitude[point]), float(track.longitude[point])))
    return keys


def _fingerprint(keys) -> str:
    text = "|".join(f"{s}@{t}@{a:.2f}@{o:.2f}" for s, t, a, o in keys)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _earlier_position(target, hours: int = 6):
    """Where the storm was ``hours`` before the sample, from its own best track (None if not recorded)."""
    track, point = target.track, target.point
    when = np.datetime64(track.time[point], "m") - np.timedelta64(hours * 60, "m")
    times = track.time.astype("datetime64[m]")
    match = np.where(times == when)[0]
    if match.size == 0:
        return None
    i = int(match[0])
    return when, (float(track.latitude[i]), float(track.longitude[i]))


def satellite_bank(targets, cache: str | Path, hursat: HursatArchive | None = None,
                   gridsat: GridSatArchive | None = None, hursat_last_season: int = HURSAT_LAST_SEASON,
                   progress: bool = True) -> dict:
    """
    Infrared patches for every storm sample, aligned with :func:`naturev1.storm_feature_bank` by order.

    Two images per sample, both storm-centred on the best-track position at their own time: now, and six
    hours earlier -- the change is what shows a convective burst or an eye clearing. Seasons up to
    ``hursat_last_season`` come from HURSAT-B1, later ones from GridSat-B1. Saved to ``cache``; reused
    while the samples are the same (a fingerprint of ids, times and positions); a damaged file is rebuilt.

    Returns:
        ``{"ir": uint8 (N, 2, P, P), "available": bool (N, 2), "fingerprint": str, "source": [str]}``.
    """
    import torch

    from .safeio import atomic_torch_save

    cache = Path(cache)
    keys = sample_keys(targets)
    fingerprint = _fingerprint(keys)
    if cache.exists():
        try:
            stored = torch.load(cache, map_location="cpu", weights_only=False)
            if stored.get("fingerprint") == fingerprint:
                if progress:
                    print(f"[stormsat] satellite bank reused: {cache} ({len(keys):,} samples)", flush=True)
                return stored
        except Exception as error:
            print(f"[stormsat] {cache.name} is damaged ({type(error).__name__}); rebuilding it", flush=True)

    count = len(keys)
    ir = np.zeros((count, 2, PATCH_PIXELS, PATCH_PIXELS), np.uint8)
    available = np.zeros((count, 2), bool)
    source = [""] * count
    jobs = []                                                   # (sample, slot, storm, time, centre)
    for i, target in enumerate(targets):
        storm, when, lat, lon = keys[i]
        jobs.append((i, 0, storm, when, (lat, lon)))
        earlier = _earlier_position(target)
        if earlier is not None:
            jobs.append((i, 1, storm, earlier[0], earlier[1]))

    old = [j for j in jobs if int(str(np.datetime64(j[3], "Y"))) <= hursat_last_season]
    new = [j for j in jobs if int(str(np.datetime64(j[3], "Y"))) > hursat_last_season]
    started = time.time()
    if hursat is not None and old:
        by_storm: dict[str, list] = {}
        for job in old:
            by_storm.setdefault(job[2], []).append(job)

        def one_storm(jobs):
            try:
                return jobs, hursat.storm_patches(jobs[0][2], [(when, centre) for *_, when, centre in jobs])
            except Exception:
                return jobs, [None] * len(jobs)

        with cf.ThreadPoolExecutor(4) as pool:
            for done, (jobs, patches) in enumerate(pool.map(one_storm, by_storm.values()), start=1):
                for (i, slot, *_), patch in zip(jobs, patches):
                    if patch is not None and np.isfinite(patch).mean() > 0.5:
                        ir[i, slot], available[i, slot], source[i] = encode_bt(patch), True, "HURSAT-B1"
                if progress and (done % 25 == 0 or done == len(by_storm)):
                    left = (time.time() - started) / done * (len(by_storm) - done) / 60
                    print(f"[stormsat] HURSAT {done:,}/{len(by_storm):,} storms ({left:.0f} min left)", flush=True)
    if gridsat is not None and new:
        patches = gridsat.patches([(when, centre) for _, _, _, when, centre in new], progress=progress)
        for (i, slot, *_), patch in zip(new, patches):
            if patch is not None and np.isfinite(patch).mean() > 0.5:
                ir[i, slot], available[i, slot], source[i] = encode_bt(patch), True, "GridSat-B1"

    bank = {"ir": torch.from_numpy(ir), "available": torch.from_numpy(available), "fingerprint": fingerprint,
            "source": source, "pixels": PATCH_PIXELS, "degrees": PATCH_DEGREES}
    atomic_torch_save(bank, cache)
    if progress:
        print(f"[stormsat] satellite bank: {int(available[:, 0].sum()):,} of {count:,} samples have an image now, "
              f"{int(available[:, 1].sum()):,} six hours earlier", flush=True)
    return bank


def describe_patch(patch: np.ndarray, center_pixels: int = 8) -> str:
    """One line a forecaster would read: coldest cloud top near the centre, and how much of it is deep."""
    patch = np.asarray(patch, np.float32)
    mid = patch.shape[0] // 2
    core = patch[mid - center_pixels: mid + center_pixels, mid - center_pixels: mid + center_pixels]
    cold = np.nanmean(patch < 213.15) if np.isfinite(patch).any() else float("nan")
    return (f"coldest cloud top near the centre {np.nanmin(core) - 273.15:.0f} C, warmest {np.nanmax(core) - 273.15:.0f} C; "
            f"{100 * cold:.0f}% of the patch colder than -60 C")


# ------------------------------------------------------------------------- NatureV2: infrared --

#: NatureV2 (0.17): the infrared structure the operational rapid-intensification guidance reads (Kaplan, DeMaria &
#: Knaff 2010; :func:`naturev1.swarm.infrared.ir_metrics` for one live scene), computed from the same storm-centred
#: patches the satellite encoder sees, with their six-hour trend. Order one; zeros when the image is missing.
STORM_IR_FEATURES = ("min_bt", "pc30", "pc50", "sdo", "mean_bt", "ring_asymmetry", "eye_warm", "d_pc50_6h",
                     "d_min_bt_6h", "present")


def storm_ir_features(codes, available, latitude=None, degrees: float = PATCH_DEGREES):
    """
    Explicit infrared predictors from storm-centred patches, ``(B, len(STORM_IR_FEATURES))``.

    Args:
        codes: ``(B, 2, P, P)`` uint8 brightness temperatures (:func:`encode_bt`): now, and six hours earlier; rows
            south to north, columns west to east, centred on the storm.
        available: ``(B, 2)`` whether each image exists.
        latitude: ``(B,)`` degrees, for the east-west pixel size (cos latitude); None treats pixels as square.

    Returns (in this order): the coldest top within 100 km ((C + 50) / 30), the shares of the 50-200 km annulus colder
    than -30 and -50 C, the standard deviation 100-300 km (/10 K), the 50-200 km mean ((C + 40) / 30), the spread of
    the eight 30-100 km octant means (/10 K), how much warmer the inner 25 km is than the 40-80 km ring (/30 K), the
    six-hour change of the -50 C share and of the coldest top (/10 K) where both images exist, and whether the current
    image was usable.
    """
    import torch

    kelvin = decode_bt(codes)                                                     # (B, 2, P, P), NaN missing
    batch, pixels = kelvin.shape[0], kelvin.shape[-1]
    device = kelvin.device
    step_km = degrees * 111.2
    offsets = (torch.arange(pixels, device=device, dtype=torch.float32) - (pixels - 1) / 2.0) * step_km
    north = offsets.view(1, -1, 1).expand(batch, pixels, pixels)
    scale = torch.ones(batch, device=device)
    if latitude is not None:
        scale = torch.cos(torch.deg2rad(torch.nan_to_num(torch.as_tensor(latitude, device=device, dtype=torch.float32)))).clamp_min(0.05)
    east = offsets.view(1, 1, -1) * scale.view(-1, 1, 1)
    radius = torch.sqrt(north ** 2 + east ** 2)                                   # (B, P, P) km
    angle = torch.atan2(north, east)
    available = torch.as_tensor(available, device=device).bool()

    def stats(image):
        valid = torch.isfinite(image)
        filled = torch.nan_to_num(image)

        def mean(mask):
            m = (mask & valid).float()
            n = m.sum((-1, -2))
            return (filled * m).sum((-1, -2)) / n.clamp_min(1.0), n

        def share_below(mask, kelvin_cut):
            m = (mask & valid).float()
            n = m.sum((-1, -2))
            return ((filled < kelvin_cut).float() * m).sum((-1, -2)) / n.clamp_min(1.0), n

        core = radius < 100.0
        coldest = torch.where(core & valid, filled, torch.full_like(filled, float("inf"))).amin((-1, -2))
        annulus = (radius >= 50.0) & (radius < 200.0)
        pc30, n_ann = share_below(annulus, 273.15 - 30.0)
        pc50, _ = share_below(annulus, 273.15 - 50.0)
        mean_ann, _ = mean(annulus)
        outer = (radius >= 100.0) & (radius < 300.0)
        mo, n_outer = mean(outer)
        m = (outer & valid).float()
        variance = (((filled - mo.view(-1, 1, 1)) ** 2) * m).sum((-1, -2)) / n_outer.clamp_min(1.0)
        sdo = variance.sqrt()
        ring = (radius >= 30.0) & (radius < 100.0)
        octants = []
        for k in range(8):
            lo = -math.pi + k * math.pi / 4.0
            sector = ring & (angle >= lo) & (angle < lo + math.pi / 4.0)
            octants.append(mean(sector)[0])
        asymmetry = torch.stack(octants, -1).std(-1)
        eye, _ = mean(radius < 25.0)
        wall, n_wall = mean((radius >= 40.0) & (radius < 80.0))
        usable = (n_ann > 50) & torch.isfinite(coldest)
        return {"coldest": coldest, "pc30": pc30, "pc50": pc50, "mean": mean_ann, "sdo": sdo,
                "asymmetry": asymmetry, "eye_warm": eye - wall, "usable": usable}

    now = stats(kelvin[:, 0])
    before = stats(kelvin[:, 1])
    present = available[:, 0] & now["usable"]
    both = present & available[:, 1] & before["usable"]
    zero = torch.zeros(batch, device=device)
    columns = [
        torch.where(present, (now["coldest"] - 273.15 + 50.0) / 30.0, zero),
        torch.where(present, now["pc30"], zero),
        torch.where(present, now["pc50"], zero),
        torch.where(present, now["sdo"] / 10.0, zero),
        torch.where(present, (now["mean"] - 273.15 + 40.0) / 30.0, zero),
        torch.where(present, now["asymmetry"] / 10.0, zero),
        torch.where(present, now["eye_warm"] / 30.0, zero),
        torch.where(both, now["pc50"] - before["pc50"], zero),
        torch.where(both, (now["coldest"] - before["coldest"]) / 10.0, zero),
        present.float(),
    ]
    return torch.nan_to_num(torch.stack(columns, -1).float(), nan=0.0, posinf=0.0, neginf=0.0)
