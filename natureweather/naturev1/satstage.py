# Copyright 2026 Nathan. Apache-2.0.
"""
Staging the satellite record: four terabytes of GridSat-B1 into what the networks actually read.

GridSat-B1 v02r01 is every geostationary satellite since 1980 -- GOES, Meteosat, GMS/MTSAT/Himawari and
the rest -- recalibrated against each other and merged onto one 0.07-degree global grid every three hours.
1980-2022 is about 125,000 files of 35-57 MB: four to six terabytes. Reading that once, carefully, is the
whole job of this module. Every file is used twice:

* **a global frame**: the 11-micron window and 6.7-micron water-vapour channels on a quarter-degree band
  from 70 S to 70 N, regridded *conservatively* (:class:`ihelix.ConservativeRegrid`), with what an average
  would hide kept alongside it -- the coldest pixel in each cell (an overshooting top), the spread (cumulus
  field or anvil), how much of the cell was observed, and which satellite saw it at what view angle. Eight
  bytes per cell: 6.5 MB per time, 0.8 TB for the record. This is what the retrieval learns the
  atmosphere's state from.
* **storm patches**: for every tropical cyclone on the planet at that hour (IBTrACS), a 320 x 320 pixel
  square of the native 0.07-degree infrared and water vapour centred on it. 205 kB per storm per time.
  This is what the storm network learns intensity, structure and rapid intensification from.

**Quality comes first.** GridSat stores brightness temperature as scaled int16 with a ``valid_range``
attribute that netCDF4 cannot apply to the packed values -- read naively, 99.95% of every image comes back
masked (found and fixed here; the fields are decoded by hand). Then: values outside 160-330 K are
removed; isolated pixels more than 50 K from all four of their neighbours (bit errors, scan-line glitches)
are removed; water vapour more than 10 K warmer than the window channel is removed (only a very strong
surface inversion can do that, and in this band it marks a bad pixel far more often than a real one); files with less than 30% of the band observed are rejected outright and recorded (partial coverage
is normal in the early years and is carried as a coverage channel). Every month written carries its
own quality report.

**Which satellite is looking matters.** Calibration differs between instruments and a pixel at 70 degrees
view angle sees through a slanted, colder column. GridSat uses the nadir-most satellite at each pixel, so
the view angle and the satellite family are recomputed exactly from the satellites' positions in the file
and kept as channels: the model can learn each instrument's quirks instead of averaging over them.

Months are independent, written atomically, and skipped if already done -- an interrupted staging resumes
where it stopped.
"""

from __future__ import annotations

import calendar
import concurrent.futures as cf
import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from .stormsat import BT_RANGE_K, PATCH_DEGREES, crop_regular, encode_bt

#: Public mirrors of GridSat-B1 v02r01. AWS (NOAA Open Data Dissemination) is fast and free to read.
GRIDSAT_AWS = "https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com/data"
GRIDSAT_NCEI = "https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/access"
#: Channels of a staged global frame, one byte each. See :func:`encode_global` / :func:`decode_global`.
GLOBAL_CHANNELS = ("ir_mean", "ir_min", "ir_std", "wv_mean", "ir_coverage", "wv_coverage", "view_zenith",
                   "satellite_family")
#: Satellite families: calibration and spectral response differ between them, not within them much.
SATELLITE_FAMILIES = ("none", "GOES", "Meteosat", "GMS/MTSAT/Himawari", "FY", "INSAT", "Electro", "other")
#: Storm patches are staged wider than the network reads them, so the training crop can move.
STAGED_PATCH_PIXELS = 320
STD_STEP_K = 0.25
GEOSTATIONARY_RATIO = 6371.0 / 42164.0
FILL = -31999


def band_axes(resolution: float = 0.25, south: float = -70.0, north: float = 70.0):
    """Cell centres of the staged global band, degrees, south to north and west to east."""
    lat = np.arange(south + resolution / 2, north, resolution)
    lon = np.arange(-180.0 + resolution / 2, 180.0, resolution)
    return lat, lon


def gridsat_url(when, mirror: str = "aws") -> str:
    stamp = np.datetime64(when, "h").astype("datetime64[s]").astype(object)
    name = f"GRIDSAT-B1.{stamp:%Y.%m.%d.%H}.v02r01.nc"
    base = GRIDSAT_AWS if mirror == "aws" else GRIDSAT_NCEI
    return f"{base}/{stamp:%Y}/{name}"


def satellite_family(name: str) -> int:
    upper = name.strip().upper()
    for index, prefixes in ((1, ("GOE", "GOES")), (2, ("MET", "MSG")), (3, ("GMS", "MTS", "HIM")),
                            (4, ("FY",)), (5, ("INS", "KAL")), (6, ("ELE", "GOMS"))):
        if upper.startswith(prefixes):
            return index
    return 7 if upper else 0


def _decode(variable, scale: float = 0.01, offset: float = 200.0) -> np.ndarray:
    raw = np.asarray(variable[0])
    return np.where(raw == FILL, np.nan, raw.astype(np.float32) * scale + offset).astype(np.float32)


def read_gridsat(path: str | Path) -> dict:
    """
    One GridSat-B1 file: window and water-vapour brightness temperature (K, NaN missing), axes, satellites.

    Decoded by hand from the packed int16: the files declare ``valid_range`` in physical units on an
    int16 variable, which netCDF4's automatic masking applies to the *packed* values and masks nearly
    everything. xarray happens not to apply it; netCDF4 does. Neither is relied on.
    """
    import netCDF4

    with netCDF4.Dataset(path) as dataset:
        dataset.set_auto_maskandscale(False)
        ir = _decode(dataset["irwin_cdr"])
        wv = _decode(dataset["irwvp"]) if "irwvp" in dataset.variables else np.full_like(ir, np.nan)
        lat = np.asarray(dataset["lat"][:], np.float64)
        lon = np.asarray(dataset["lon"][:], np.float64)
        satlon = np.asarray(dataset["satlon"][0], np.float64) if "satlon" in dataset.variables else np.zeros(0)
        names = []
        if "satname" in dataset.variables:
            for chars in np.asarray(dataset["satname"][0]):
                names.append(b"".join(bytes(c) for c in chars).decode("ascii", "ignore").strip("\x00 "))
    valid = np.isfinite(satlon) & (np.abs(satlon) <= 360.0) & (np.array([bool(n) for n in names]) if names
                                                                 else np.ones(satlon.shape, bool))
    return {"ir": ir, "wv": wv, "lat": lat, "lon": lon, "satlon": satlon[valid],
            "satname": [n for n, ok in zip(names, valid) if ok] if names else []}


def quality_control(ir: np.ndarray, wv: np.ndarray, spike_k: float = 50.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Remove what is not a measurement, and say how much went.

    Returns ``(ir, wv, report)`` with removed pixels set to NaN.
    """
    low, high = BT_RANGE_K
    report = {"ir_observed": float(np.isfinite(ir).mean()), "wv_observed": float(np.isfinite(wv).mean())}
    ir = np.where((ir >= low) & (ir <= high), ir, np.nan).astype(np.float32)
    wv = np.where((wv >= low) & (wv <= high), wv, np.nan).astype(np.float32)
    # Isolated spikes: a pixel more than ``spike_k`` colder -- or warmer -- than *every* one of its four
    # neighbours. "Every", not "their mean": at an anvil's edge a 210 K pixel beside three 300 K ones is
    # 67 K from their mean and perfectly real; a bit error is far from all of them.
    padded = np.pad(ir, 1, constant_values=np.nan)
    sides = (padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:])
    with np.errstate(invalid="ignore"):
        highest = np.fmax(np.fmax(sides[0], sides[1]), np.fmax(sides[2], sides[3]))
        lowest = np.fmin(np.fmin(sides[0], sides[1]), np.fmin(sides[2], sides[3]))
        spikes = np.isfinite(ir) & np.isfinite(highest) & ((ir - highest > spike_k) | (lowest - ir > spike_k))
    ir = np.where(spikes, np.nan, ir)
    with np.errstate(invalid="ignore"):
        impossible = (wv - ir) > 10.0
    wv = np.where(impossible, np.nan, wv)
    report.update({"spikes_removed": int(spikes.sum()), "wv_warmer_than_window": int(impossible.sum()),
                   "ir_kept": float(np.isfinite(ir).mean()), "wv_kept": float(np.isfinite(wv).mean())})
    return ir, wv, report


def nadir_view(lat: np.ndarray, lon: np.ndarray, satlon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ``(view zenith angle in degrees, index of the nadir-most satellite)`` on a regular grid, from geometry.

    For a geostationary satellite at longitude s, a point at (lat, lon) is at central angle g with
    cos g = cos(lat) cos(lon - s), and is seen at zenith angle atan(sin g / (cos g - R / (R + h))).
    """
    if satlon.size == 0:
        shape = (lat.size, lon.size)
        return np.full(shape, np.nan, np.float32), np.full(shape, -1, np.int16)
    phi = np.radians(lat)[:, None, None]
    delta = np.radians(lon[None, :, None] - satlon[None, None, :])
    cos_g = np.cos(phi) * np.cos(delta)
    sin_g = np.sqrt(np.clip(1.0 - cos_g ** 2, 0.0, 1.0))
    zenith = np.degrees(np.arctan2(sin_g, cos_g - GEOSTATIONARY_RATIO))
    zenith = np.where(cos_g > GEOSTATIONARY_RATIO, zenith, np.inf)              # beyond the limb
    best = zenith.argmin(-1)
    angle = np.take_along_axis(zenith, best[..., None], -1)[..., 0]
    angle = np.where(np.isfinite(angle), angle, np.nan).astype(np.float32)
    return angle, np.where(np.isfinite(angle), best, -1).astype(np.int16)


def encode_fraction(values: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(np.nan_to_num(values), 0.0, 1.0) * 255).astype(np.uint8)


def encode_global(ir_mean, ir_min, ir_std, wv_mean, ir_cov, wv_cov, zenith, family) -> np.ndarray:
    """Eight fields on the band grid into ``(8, H, W)`` uint8, in :data:`GLOBAL_CHANNELS` order."""
    std = np.where(np.isfinite(ir_std), np.clip(np.rint(ir_std / STD_STEP_K), 0, 255), 0).astype(np.uint8)
    angle = np.where(np.isfinite(zenith), np.clip(np.rint(zenith / 90.0 * 254) + 1, 1, 255), 0).astype(np.uint8)
    return np.stack([encode_bt(ir_mean), encode_bt(ir_min), std, encode_bt(wv_mean), encode_fraction(ir_cov),
                     encode_fraction(wv_cov), angle, np.clip(family, 0, 255).astype(np.uint8)])


def decode_global(codes):
    """
    ``(..., 8, H, W)`` uint8 into physical fields: a dict of float tensors (or arrays), NaN where missing.

    ``ir_mean``, ``ir_min``, ``wv_mean`` in kelvin; ``ir_std`` in kelvin; coverages in [0, 1];
    ``view_zenith`` in degrees; ``satellite_family`` as an integer index into :data:`SATELLITE_FAMILIES`.
    """
    from .stormsat import decode_bt

    torch_input = hasattr(codes, "float")
    take = (lambda i: codes[..., i, :, :])
    out = {"ir_mean": decode_bt(take(0)), "ir_min": decode_bt(take(1)), "wv_mean": decode_bt(take(3))}
    if torch_input:
        out["ir_std"] = take(2).float() * STD_STEP_K
        out["ir_coverage"] = take(4).float() / 255.0
        out["wv_coverage"] = take(5).float() / 255.0
        angle = take(6).float()
        out["view_zenith"] = (angle - 1.0) / 254.0 * 90.0
        out["view_zenith"] = out["view_zenith"].masked_fill(angle == 0, float("nan"))
        out["satellite_family"] = take(7).long()
    else:
        out["ir_std"] = take(2).astype(np.float32) * STD_STEP_K
        out["ir_coverage"] = take(4).astype(np.float32) / 255.0
        out["wv_coverage"] = take(5).astype(np.float32) / 255.0
        angle = take(6).astype(np.float32)
        out["view_zenith"] = np.where(angle > 0, (angle - 1.0) / 254.0 * 90.0, np.nan).astype(np.float32)
        out["satellite_family"] = take(7).astype(np.int64)
    return out


class GlobalFrameEncoder:
    """GridSat file -> one staged global frame, with the conservative regrid built once and reused."""

    def __init__(self, source_lat, source_lon, resolution: float = 0.25, device: str = "cpu") -> None:
        import torch
        from ihelix import ConservativeRegrid

        self.lat, self.lon = band_axes(resolution)
        self.regrid = ConservativeRegrid(source_lat, source_lon, self.lat, self.lon).to(device)
        self.device = device
        self.torch = torch
        self.source_shape = (len(source_lat), len(source_lon))

    def encode(self, fields: dict) -> np.ndarray:
        torch = self.torch
        ir = torch.from_numpy(fields["ir"]).to(self.device)
        wv = torch.from_numpy(fields["wv"]).to(self.device)
        ir_mean, ir_std, ir_cov = self.regrid.moments(ir)
        ir_min, _ = self.regrid.extremes(ir)
        wv_mean, wv_cov = self.regrid(wv)
        zenith, nadir = nadir_view(self.lat, self.lon, fields["satlon"])
        names = fields["satname"]
        families = np.array([satellite_family(n) for n in names] + [0], np.uint8)
        family = families[np.where(nadir >= 0, nadir, len(names))]
        cover = ir_cov.cpu().numpy()
        zenith = np.where(cover > 0, zenith, np.nan)
        family = np.where(cover > 0, family, 0)
        as_np = (lambda t: t.float().cpu().numpy())
        return encode_global(as_np(ir_mean), as_np(ir_min), as_np(ir_std), as_np(wv_mean), cover,
                             as_np(wv_cov), zenith, family)


def storm_crops(fields: dict, centres, pixels: int = STAGED_PATCH_PIXELS) -> np.ndarray:
    """``(n, 2, pixels, pixels)`` uint8 infrared and water vapour, centred on each ``(lat, lon)``."""
    out = np.zeros((len(centres), 2, pixels, pixels), np.uint8)
    for i, centre in enumerate(centres):
        out[i, 0] = encode_bt(crop_regular(fields["lat"], fields["lon"], fields["ir"], centre, pixels))
        out[i, 1] = encode_bt(crop_regular(fields["lat"], fields["lon"], fields["wv"], centre, pixels))
    return out


def month_times(year: int, month: int, hours: int = 3) -> np.ndarray:
    days = calendar.monthrange(year, month)[1]
    start = np.datetime64(f"{year:04d}-{month:02d}-01T00", "h")
    return start + np.arange(0, days * 24, hours).astype("timedelta64[h]")


def _download(url: str, destination: Path, attempts: int = 5, timeout: float = 180.0) -> bool:
    for attempt in range(attempts):
        try:
            temporary = destination.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=timeout) as response, open(temporary, "wb") as handle:
                while True:
                    block = response.read(1 << 22)
                    if not block:
                        break
                    handle.write(block)
            temporary.replace(destination)
            return True
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return False                                            # no file for that hour
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return False


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(payload))
    os.replace(temporary, path)


def stage_month(root: str | Path, year: int, month: int, table: dict | None = None, resolution: float = 0.25,
                raw_dir: str | Path | None = None, keep_raw: bool = False, mirror: str = "aws",
                min_observed: float = 0.3, prefetch: int = 4, device: str = "cpu", download: bool = True) -> dict:
    """
    Stage one month: its global frames and the storm patches of every storm active in it.

    Writes ``global/YYYY-MM.npy`` ``(times, 8, H, W)`` uint8 and ``storms/YYYY-MM.npy``
    ``(fixes, 2, 320, 320)`` uint8, each with a ``.json`` written last (so a month with a ``.json`` is
    complete). Raw files are read from ``raw_dir`` when present there, else downloaded into it (unless
    ``download`` is False) and deleted after use unless ``keep_raw``.

    Returns the month's report.
    """
    import torch

    torch.set_num_threads(1)
    root = Path(root)
    tag = f"{year:04d}-{month:02d}"
    frames_meta = root / "global" / f"{tag}.json"
    if frames_meta.exists():
        return json.loads(frames_meta.read_text())
    (root / "global").mkdir(parents=True, exist_ok=True)
    (root / "storms").mkdir(parents=True, exist_ok=True)
    raw_dir = Path(raw_dir or tempfile.mkdtemp(prefix="gridsat-"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    times = month_times(year, month)
    seconds = times.astype("datetime64[s]").astype(np.int64)
    rows_at: dict[int, list[int]] = {}
    position: dict[int, tuple[float, float]] = {}
    if table is not None and len(table["time"]):
        ids = table.get("row", np.arange(len(table["time"])))
        inside = np.where((table["time"] >= seconds[0]) & (table["time"] <= seconds[-1]))[0]
        for local in inside:
            row = int(ids[local])
            rows_at.setdefault(int(table["time"][local]), []).append(row)
            position[row] = (float(table["lat"][local]), float(table["lon"][local]))
    storm_rows = sorted(r for rows in rows_at.values() for r in rows)
    storm_slot = {row: i for i, row in enumerate(storm_rows)}

    lat, lon = band_axes(resolution)
    frames = np.lib.format.open_memmap(root / "global" / f"{tag}.npy.part", mode="w+", dtype=np.uint8,
                                       shape=(len(times), len(GLOBAL_CHANNELS), len(lat), len(lon)))
    patches = np.lib.format.open_memmap(root / "storms" / f"{tag}.npy.part", mode="w+", dtype=np.uint8,
                                        shape=(max(len(storm_rows), 1), 2, STAGED_PATCH_PIXELS, STAGED_PATCH_PIXELS))
    available = np.zeros(len(times), bool)
    patch_ok = np.zeros(len(storm_rows), bool)
    reports, encoder = [], None

    def fetch(when) -> Path | None:
        name = Path(gridsat_url(when, mirror)).name
        local = raw_dir / name
        if local.exists() and local.stat().st_size > 0:
            return local
        if not download:
            return None
        return local if _download(gridsat_url(when, mirror), local) else None

    with cf.ThreadPoolExecutor(prefetch) as pool:
        futures = [pool.submit(fetch, when) for when in times]
        for index, future in enumerate(futures):
            path = future.result()
            stamp = int(seconds[index])
            if path is None:
                reports.append({"time": stamp, "status": "missing"})
                continue
            try:
                fields = read_gridsat(path)
            except Exception as error:                                   # a truncated or corrupt file
                reports.append({"time": stamp, "status": f"unreadable: {type(error).__name__}"})
                path.unlink(missing_ok=True)
                continue
            fields["ir"], fields["wv"], report = quality_control(fields["ir"], fields["wv"])
            report["time"] = stamp
            if report["ir_kept"] < min_observed:                         # share of the 70S-70N band observed
                report["status"] = f"rejected: under {100 * min_observed:.0f}% observed"
                reports.append(report)
            else:
                if encoder is None or encoder.source_shape != fields["ir"].shape:
                    encoder = GlobalFrameEncoder(fields["lat"], fields["lon"], resolution, device)
                frames[index] = encoder.encode(fields)
                available[index] = True
                wanted = rows_at.get(stamp, [])
                if wanted:
                    centres = [position[r] for r in wanted]
                    crops = storm_crops(fields, centres)
                    for r, crop in zip(wanted, crops):
                        patches[storm_slot[r]] = crop
                        patch_ok[storm_slot[r]] = bool((crop[0] > 0).mean() > 0.5)
                report["status"] = "ok"
                reports.append(report)
            if not keep_raw:
                path.unlink(missing_ok=True)

    frames.flush()
    patches.flush()
    del frames, patches
    os.replace(root / "global" / f"{tag}.npy.part", root / "global" / f"{tag}.npy")
    os.replace(root / "storms" / f"{tag}.npy.part", root / "storms" / f"{tag}.npy")
    _atomic_json(root / "storms" / f"{tag}.json", {"rows": storm_rows, "ok": patch_ok.tolist(),
                                                     "pixels": STAGED_PATCH_PIXELS, "degrees": PATCH_DEGREES})
    summary = {
        "month": tag, "times": seconds.tolist(), "available": available.tolist(), "resolution": resolution,
        "channels": list(GLOBAL_CHANNELS), "shape": [len(lat), len(lon)],
        "observed_times": int(available.sum()), "storm_fixes": len(storm_rows),
        "storm_patches_ok": int(patch_ok.sum()),
        "spikes_removed": int(sum(r.get("spikes_removed", 0) for r in reports)),
        "rejected": [r["time"] for r in reports if r.get("status", "").startswith(("rejected", "unreadable"))],
        "missing": [r["time"] for r in reports if r.get("status") == "missing"],
    }
    _atomic_json(frames_meta, summary)
    return summary


def month_rows(table: dict | None, year: int, month: int) -> dict | None:
    """The label-table rows inside one month -- positions and their global row numbers, nothing else, so a
    worker process is sent kilobytes rather than the whole table."""
    if table is None:
        return None
    seconds = month_times(year, month).astype("datetime64[s]").astype(np.int64)
    inside = np.where((table["time"] >= seconds[0]) & (table["time"] <= seconds[-1]))[0]
    return {"time": table["time"][inside], "lat": table["lat"][inside], "lon": table["lon"][inside], "row": inside}


def stage_gridsat(root: str | Path, years, table: dict | None = None, workers: int = 32, resolution: float = 0.25,
                  raw_dir: str | Path | None = None, keep_raw: bool = False, mirror: str = "aws",
                  progress: bool = True) -> list[dict]:
    """
    Stage every month of ``years`` with ``workers`` processes; finished months are skipped.

    Each worker downloads, decodes, checks, regrids and crops one month at a time with one CPU thread, so
    ``workers`` should be about the number of physical cores you can spare. Prints a running total.
    """
    months = [(y, m) for y in years for m in range(1, 13)]
    root = Path(root)
    pending = [(y, m) for y, m in months if not (root / "global" / f"{y:04d}-{m:02d}.json").exists()]
    results, started = [], time.time()
    if progress:
        print(f"[satstage] {len(months) - len(pending)} of {len(months)} months already staged; "
              f"{len(pending)} to go with {workers} workers", flush=True)
    with cf.ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(stage_month, root, y, m, month_rows(table, y, m), resolution,
                               None if raw_dir is None else Path(raw_dir) / f"{y:04d}-{m:02d}", keep_raw, mirror): (y, m)
                   for y, m in pending}
        for done, future in enumerate(cf.as_completed(futures), start=1):
            year, month = futures[future]
            try:
                summary = future.result()
            except Exception as error:
                print(f"\n[satstage] {year}-{month:02d} failed: {type(error).__name__}: {error}; "
                      "it will be retried on the next run", flush=True)
                continue
            results.append(summary)
            if progress:
                rate = (time.time() - started) / done
                print(f"[satstage] {summary['month']}: {summary['observed_times']}/{len(summary['times'])} times, "
                      f"{summary['storm_patches_ok']}/{summary['storm_fixes']} storm patches  "
                      f"[{done}/{len(pending)}, about {rate * (len(pending) - done) / 60:.0f} min left]", flush=True)
    return results


def staging_plan(years, resolution: float = 0.25, storm_fixes: int = 0) -> str:
    """What staging ``years`` will download and write -- print it before starting."""
    years = list(years)
    times = sum(len(month_times(y, m)) for y in years for m in range(1, 13))
    lat, lon = band_axes(resolution)
    frame_bytes = len(GLOBAL_CHANNELS) * len(lat) * len(lon)
    patch_bytes = 2 * STAGED_PATCH_PIXELS ** 2
    return "\n".join([
        f"GridSat-B1 {years[0]}-{years[-1]}: {times:,} three-hourly files, about "
        f"{times * 45e6 / 1e12:.1f} TB to read (35-57 MB each)",
        f"  global frames at {resolution} deg ({len(lat)} x {len(lon)} x {len(GLOBAL_CHANNELS)} bytes): "
        f"{times * frame_bytes / 1e12:.2f} TB on disk",
        f"  storm patches ({STAGED_PATCH_PIXELS} px, IR + WV): {storm_fixes:,} fixes, "
        f"{storm_fixes * patch_bytes / 1e9:.0f} GB on disk",
    ])


def main(argv=None) -> None:
    """``python -m naturev1.satstage --root /data/sat --ibtracs ibtracs.since1980.csv --years 1980-2022``"""
    import argparse

    from .ibtracs import describe_table, parse_ibtracs, storm_label_table

    parser = argparse.ArgumentParser(description="Stage GridSat-B1 into global frames and storm patches")
    parser.add_argument("--root", required=True)
    parser.add_argument("--ibtracs", required=True, help="IBTrACS v04 CSV (ibtracs.since1980.list.v04r01.csv)")
    parser.add_argument("--years", default="1980-2022", help="first-last, inclusive")
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 8) - 8, 1))
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--raw-dir", default=None, help="keep or reuse raw GridSat files here")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--mirror", default="aws", choices=("aws", "ncei"))
    args = parser.parse_args(argv)
    first, last = (int(y) for y in args.years.split("-"))
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    table_path = root / "labels.npz"
    if table_path.exists():
        table = dict(np.load(table_path, allow_pickle=False))
    else:
        table = storm_label_table(parse_ibtracs(args.ibtracs, since=first, until=last))
        np.savez(table_path, **table)
    print(describe_table(table), flush=True)
    print(staging_plan(range(first, last + 1), args.resolution, len(table["time"])), flush=True)
    results = stage_gridsat(root, range(first, last + 1), table, workers=args.workers, resolution=args.resolution,
                            raw_dir=args.raw_dir, keep_raw=args.keep_raw, mirror=args.mirror)
    done = sorted(root.glob("global/*.json"))
    print(f"[satstage] {len(done)} months staged in {root}; this run added {len(results)}", flush=True)


__all__ = ["GLOBAL_CHANNELS", "GRIDSAT_AWS", "GlobalFrameEncoder", "SATELLITE_FAMILIES", "STAGED_PATCH_PIXELS",
           "band_axes", "decode_global", "encode_global", "gridsat_url", "month_times", "nadir_view",
           "quality_control", "read_gridsat", "stage_gridsat", "stage_month", "staging_plan", "storm_crops"]


if __name__ == "__main__":
    main()
