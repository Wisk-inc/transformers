# Copyright 2026 Nathan. Apache-2.0.
"""
The truth Pluvium learns rain from: NOAA's CMORPH Climate Data Record, 8 km every 30 minutes since 1998
(``noaa-cdr-precip-cmorph-pds`` on AWS Open Data, about 1.6 MB an hour for the whole band 60 S - 60 N).

CMORPH morphs passive-microwave rain estimates along cloud motion seen in geostationary infrared, then adjusts them
to gauge analyses (the ``ADJ`` in the file names). It is the best free gridded rain record with both 8 km cells and
half-hour timing over the Caribbean, and it has known faults the model is told about rather than hidden from: it
misses some light rain and under-reads rain made by mountains (see :mod:`pluvium.losses`, where each truth source has
its own learned error).

Staging keeps two arrays per year (uint16 in 0.01 mm/h, 65535 = missing): the rain at every target cell
(``targets_<year>.npy``, hours x targets) and the rain averaged around every mesh point (``mesh_<year>.npy``), which
is what the model is shown as "rain observed in the last hours" at the start of a forecast.
"""

from __future__ import annotations

import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..domain import DOMAIN, PointSet, cmorph_axes, cmorph_window, window_lon
from ..util import FetchError, atomic_save, fetch, hour_to_datetime64, log, year_hours

URL = ("https://noaa-cdr-precip-cmorph-pds.s3.amazonaws.com/data/30min/8km/{y}/{m}/{d}/"
       "CMORPH_V1.0_ADJ_8km-30min_{y}{m}{d}{h}.nc")
MISSING = 65535
SCALE = 100.0  # stored units per mm/h


def url_for(hour: int, template: str = URL) -> str:
    s = str(hour_to_datetime64(hour))
    return template.format(y=s[:4], m=s[5:7], d=s[8:10], h=s[11:13])


def read_hour(blob: bytes, rows: slice, cols: slice) -> np.ndarray:
    """One CMORPH file -> the hour's mean rate (mm/h) over the window, NaN where both half-hours are missing."""
    import h5py

    with h5py.File(io.BytesIO(blob), "r") as f:
        raw = read_window(f["cmorph"], rows, cols).astype(np.float32)
        scale = float(np.ravel(f["cmorph"].attrs.get("scale_factor", [0.01]))[0])
    valid = raw >= 0
    rate = np.where(valid, raw * scale, 0.0)
    count = valid.sum(0)
    with np.errstate(invalid="ignore"):
        return np.where(count > 0, rate.sum(0) / np.maximum(count, 1), np.nan).astype(np.float32)


def read_window(var, rows: slice, cols) -> np.ndarray:
    """``var[:, rows, cols]`` where ``cols`` is a slice or (across the prime meridian) an index array in two runs."""
    if isinstance(cols, slice):
        return var[:, rows, cols]
    cols = np.asarray(cols)
    breaks = np.nonzero(np.diff(cols) != 1)[0] + 1
    return np.concatenate([var[:, rows, int(run[0]):int(run[-1]) + 1] for run in np.split(cols, breaks)], axis=-1)


def mesh_operator(mesh: PointSet, window: tuple[slice, slice], half_widths_deg: np.ndarray):
    """Sparse ``(mesh points, window cells)`` averaging operator: each point takes the cells in a box of
    ``half_widths_deg`` around it."""
    from scipy import sparse

    lat, _ = cmorph_axes()
    lat, lon = lat[window[0]], window_lon(window[1])
    rows, cols, vals = [], [], []
    for k in range(len(mesh)):
        hw = float(half_widths_deg[k])
        r = np.nonzero(np.abs(lat - mesh.lat[k]) <= hw)[0]
        c = np.nonzero(np.abs(lon - mesh.lon[k]) <= hw / max(np.cos(np.radians(mesh.lat[k])), 0.2))[0]
        if r.size == 0 or c.size == 0:
            r = np.array([int(np.abs(lat - mesh.lat[k]).argmin())])
            c = np.array([int(np.abs(lon - mesh.lon[k]).argmin())])
        flat = (r[:, None] * lon.size + c[None, :]).ravel()
        rows.append(np.full(flat.size, k))
        cols.append(flat)
        vals.append(np.full(flat.size, 1.0 / flat.size))
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(mesh), lat.size * lon.size))


def default_half_widths(mesh: PointSet) -> np.ndarray:
    """Half the spacing of each mesh tier (land 0.1, coastal 0.2, open sea 0.75 degrees)."""
    return np.choose(np.clip(mesh.kind, 0, 2), [0.05, 0.10, 0.375]).astype(np.float64)


_WORKER = {}


def _init(template, window, target_rows, target_cols, operator) -> None:
    _WORKER.update(template=template, window=window, rows=target_rows, cols=target_cols, op=operator)


def _encode(rate: np.ndarray) -> np.ndarray:
    out = np.where(np.isnan(rate), MISSING, np.clip(np.round(rate * SCALE), 0, MISSING - 1))
    return out.astype(np.uint16)


def _one_hour(hour: int) -> tuple[int, np.ndarray, np.ndarray]:
    w = _WORKER
    try:
        blob = fetch(url_for(hour, w["template"]), missing_ok=True, attempts=4)
    except FetchError:
        blob = None
    n_t, n_m = w["rows"].size, w["op"].shape[0]
    if blob is None:
        return hour, np.full(n_t, MISSING, np.uint16), np.full(n_m, MISSING, np.uint16)
    try:
        rate = read_hour(blob, *w["window"])
    except Exception:
        return hour, np.full(n_t, MISSING, np.uint16), np.full(n_m, MISSING, np.uint16)
    at_targets = rate[w["rows"], w["cols"]]
    flat = rate.ravel()
    valid = np.isfinite(flat)
    num = w["op"] @ np.where(valid, flat, 0.0)
    den = w["op"] @ valid.astype(np.float64)
    with np.errstate(invalid="ignore"):
        mesh_rate = np.where(den > 0.5, num / np.maximum(den, 1e-9), np.nan)
    return hour, _encode(at_targets), _encode(mesh_rate)


class CMORPHCache:
    """``targets(hours)`` and ``mesh(hours)`` as float32 mm/h (NaN missing) from the staged arrays."""

    def __init__(self, root) -> None:
        self.root = Path(root).expanduser() / "cmorph"
        self.meta = json.loads((self.root / "cmorph.json").read_text())
        self._open = {}

    @property
    def years(self) -> list:
        return sorted(int(y) for y in self.meta.get("years", []))

    def _array(self, kind: str, year: int):
        key = (kind, year)
        if key not in self._open:
            self._open[key] = np.load(self.root / f"{kind}_{year}.npy", mmap_mode="r")
        return self._open[key]

    def _read(self, kind: str, hours: np.ndarray, columns=None) -> np.ndarray:
        from ..util import calendar

        hours = np.asarray(hours, np.int64)
        width = self.meta["n_targets" if kind == "targets" else "n_mesh"] if columns is None else len(columns)
        out = np.full((hours.size, width), np.nan, np.float32)
        years = calendar(hours)["year"]
        for year in np.unique(years):
            if int(year) not in self.years:
                continue
            first, _ = year_hours(int(year))
            sel = np.nonzero(years == year)[0]
            arr = self._array(kind, int(year))
            idx = hours[sel] - first
            raw = arr[idx] if columns is None else arr[idx][:, columns]
            out[sel] = np.where(raw == MISSING, np.nan, raw.astype(np.float32) / SCALE)
        return out

    def targets(self, hours, columns=None) -> np.ndarray:
        return self._read("targets", hours, columns)

    def mesh(self, hours) -> np.ndarray:
        return self._read("mesh", hours)


def stage_cmorph(root, targets: PointSet, mesh: PointSet, years=None, period=None, workers: int = 16,
                 template: str = URL, verbose: bool = True, cmorph_box=None) -> CMORPHCache:
    """
    Fetch and reduce CMORPH for whole ``years`` or ``period = (first_hour, end_hour)`` into the cache (hours already
    staged are skipped). ``workers`` processes each fetch and decode their own hours.
    """
    root = Path(root).expanduser() / "cmorph"
    root.mkdir(parents=True, exist_ok=True)
    window = cmorph_window(tuple(cmorph_box) if cmorph_box is not None else DOMAIN)   # the geography's own window
    op = mesh_operator(mesh, window, default_half_widths(mesh))
    meta_path = root / "cmorph.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"years": []}
    if meta.get("n_targets") not in (None, len(targets)) or meta.get("n_mesh") not in (None, len(mesh)):
        raise RuntimeError("the CMORPH cache was staged for a different geography; stage into a new folder")
    spans = []
    if period is not None:
        a, b = int(period[0]), int(period[1])
        for year in range(int(str(hour_to_datetime64(a))[:4]), int(str(hour_to_datetime64(b - 1))[:4]) + 1):
            first, end = year_hours(year)
            spans.append((year, max(a, first), min(b, end)))
    else:
        spans = [(y, *year_hours(y)) for y in sorted(years or [])]
    for year, a, b in spans:
        first, end = year_hours(year)
        paths = {k: root / f"{k}_{year}.npy" for k in ("targets", "mesh")}
        arrays = {k: (np.load(p) if p.exists() else np.full((end - first, n), MISSING, np.uint16))
                  for (k, p), n in zip(paths.items(), (len(targets), len(mesh)))}
        # an hour is still to do when either array lacks it. Until 0.5.5 only the targets were asked: a folder that
        # held ``targets_2016.npy`` and no ``mesh_2016.npy`` (half a year copied from the Hub) was taken as staged,
        # the mesh file was never made, and training stopped when it first needed it
        lacking = ((arrays["targets"][a - first: b - first] == MISSING).all(1)
                   | (arrays["mesh"][a - first: b - first] == MISSING).all(1))
        todo = [a + int(k) for k in np.nonzero(lacking)[0]]
        if not todo and not all(p.exists() for p in paths.values()):
            for k, p in paths.items():                         # nothing to fetch for this period, but a file to make
                if not p.exists():
                    atomic_save(p, lambda q, arr=arrays[k]: np.save(q, arr))
        if todo:
            args = (template, window, targets.extra["row"], targets.extra["col"], op)
            if workers <= 1:
                _init(*args)
                results = map(_one_hour, todo)
                pool = None
            else:
                import multiprocessing

                pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork"),
                                           initializer=_init, initargs=args)
                results = pool.map(_one_hour, todo, chunksize=8)
            done = 0
            import time

            last = started = time.time()
            for hour, at_t, at_m in results:
                arrays["targets"][hour - first] = at_t
                arrays["mesh"][hour - first] = at_m
                done += 1
                if verbose and time.time() - last > 30:
                    rate = done / (time.time() - started)
                    log(f"  CMORPH {year}: {done:,} of {len(todo):,} hours ({rate:.1f}/s, "
                        f"about {(len(todo) - done) / max(rate, 1e-9) / 60:.0f} min left)")
                    last = time.time()
            if pool is not None:
                pool.shutdown()
            for k, p in paths.items():
                atomic_save(p, lambda q, arr=arrays[k]: np.save(q, arr))
        missing = int((arrays["targets"][a - first: b - first] == MISSING).all(1).sum())
        if verbose:
            log(f"  CMORPH {year}: {b - a:,} hours staged ({missing:,} missing in the archive)")
        meta["years"] = sorted(set(meta.get("years", [])) | {year})
    meta.update({"n_targets": len(targets), "n_mesh": len(mesh)})
    atomic_save(meta_path, lambda p: Path(p).write_text(json.dumps(meta)))
    return CMORPHCache(root.parent)


__all__ = ["CMORPHCache", "MISSING", "URL", "default_half_widths", "mesh_operator", "read_hour", "stage_cmorph", "url_for"]
