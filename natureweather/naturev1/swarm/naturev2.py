# Copyright 2026 Nathan. Apache-2.0.
"""
NatureV2 on a live storm: the GFS analysis turned into exactly the state the backbone was trained on, and the trained
model run on it -- its storm heads and its own full-state rollout -- beside the swarm.

* :func:`gfs_state` -- one GFS 0.25-degree analysis as the 89 ERA5 channels on the 1.5-degree training grid, in
  physical units: geopotential, temperature, winds, specific humidity and vertical velocity on the 13 levels; 2 m
  temperature, sea-level and surface pressure, 10 m winds, skin temperature as the sea surface (over water only, as
  in ERA5), precipitable water; the six-hour precipitation from the previous cycle's six-hour forecast (ERA5
  accumulates over the six hours ending at the valid time); top-of-atmosphere sunlight from the orbit, exactly as the
  rollout computes it; and the land mask and surface height -- ERA5's own, from the staged record, when given,
  because those are the constants the model learned with. Each field is averaged onto the 1.5-degree nodes the way
  the hindcast boxes are. Cached as ``.npz``, about 10 MB a time.
* :func:`gfs_window` -- six of them, six hours apart, normalised with the training statistics: the model's input.
* :func:`bdeck_track` -- NHC's best track (b-deck) as a :class:`naturev1.besttrack.Track`: wind, pressure, status,
  gale radii, radius of maximum wind -- what the storm-state vector and the ocean features read.
* :func:`goes_storm_inputs` -- the storm-centred GOES-East / West infrared (band 13) and water vapour (band 9) on the
  same 0.07-degree, 256-pixel patch grid as GridSat-B1, now and 3, 6, 9 hours earlier: the in-model satellite
  encoder's two images and SatNet-Storm 2.0's loop. The full-disk files are deleted once their patches are cut.
* :func:`naturev2_forecast` -- the trained model on all of it: track scenarios, the multi-member track consensus,
  intensity, rapid intensification, landfall (:func:`naturev1.forecast.build_forecast`), and the full-state rollout
  followed for the storm (vortex and steering tracks, rollout wind and pressure).

GFS is not ERA5: its analysis differs a little everywhere, and the model never saw GFS in training. The difference is
measured, not assumed -- :func:`compare_to_era5` scores a GFS state against the ERA5 of the same time (both exist for
2021) channel by channel, in the normalised units the model reads.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import torch

from .analysis import GFS, GFS_FORECAST, LAT_AXIS, LON_AXIS, G, _index, _read


LEVEL_FIELDS = {"geopotential": ("HGT", G), "temperature": ("TMP", 1.0), "u_component_of_wind": ("UGRD", 1.0),
                "v_component_of_wind": ("VGRD", 1.0), "specific_humidity": ("SPFH", 1.0),
                "vertical_velocity": ("VVEL", 1.0)}
SURFACE_FIELDS = {"2m_temperature": ("TMP", "2 m above ground", 1.0),
                  "mean_sea_level_pressure": ("PRMSL", "mean sea level", 1.0),
                  "10m_u_component_of_wind": ("UGRD", "10 m above ground", 1.0),
                  "10m_v_component_of_wind": ("VGRD", "10 m above ground", 1.0),
                  "sea_surface_temperature": ("TMP", "surface", 1.0),
                  "surface_pressure": ("PRES", "surface", 1.0),
                  "total_column_water_vapour": ("PWAT", "entire atmosphere (considered as a single layer)", 1.0),
                  "land_sea_mask": ("LAND", "surface", 1.0),
                  "geopotential_at_surface": ("HGT", "surface", G)}
PRECIP = ("APCP", "surface", 1e-3)                  # kg m-2 (mm) -> m, ERA5's unit
STATIC = ("land_sea_mask", "geopotential_at_surface")
SUN = "toa_incident_solar_radiation_6hr"


def _stamp(when) -> str:
    return str(np.datetime64(when, "h")).replace("-", "").replace("T", "")[:10]


def _unix(when) -> int:
    return int(np.datetime64(when, "s").astype(np.int64))


def gfs_reader(when, lead: int, keys) -> dict:
    """``{(var, level): (121, 240) field}`` from the GFS cycle ``when``, forecast hour ``lead`` (0: the analysis)."""
    stamp = _stamp(when)
    base = (GFS if lead == 0 else GFS_FORECAST).format(day=stamp[:8], hour=int(stamp[8:10]), lead=int(lead))
    index = _index(base)
    missing = [k for k in keys if k not in index]
    if missing:
        raise KeyError(f"GFS {stamp} f{lead:03d} has no {missing}")
    return {k: _read(base, index, *k) for k in keys}


def gfs_state(when, variables, cache="~/.cache/naturev1/gfs", static: dict | None = None, reader=gfs_reader) -> np.ndarray:
    """
    The GFS analysis at ``when`` (00/06/12/18 UTC) as ``(121, 240, len(variables))`` float32 in ERA5's channels and
    units, south to north from the pole and 0-358.5 E -- the layout of the staged training record. ``variables`` are
    the record's flat channel names (``"temperature@850"``). ``static``: ``{name: (121, 240)}`` to use instead of
    GFS's land mask and surface height (pass ERA5's, :func:`static_from_staging`).
    """
    when = np.datetime64(when, "h")
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"gfs_state_{_stamp(when)}.npz"
    fields = dict(np.load(path)) if path.exists() else {}
    if not fields:
        keys = [(var, f"{int(level)} mb") for var, _ in LEVEL_FIELDS.values() for level in _levels(variables)]
        keys += [(var, level) for var, level, _ in SURFACE_FIELDS.values()]
        raw = reader(when, 0, sorted(set(keys)))
        for name, (var, scale) in LEVEL_FIELDS.items():
            for level in _levels(variables):
                fields[f"{name}@{level}"] = raw[(var, f"{level} mb")] * scale
        for name, (var, level, scale) in SURFACE_FIELDS.items():
            fields[name] = raw[(var, level)] * scale
        rain = reader(when - np.timedelta64(6, "h"), 6, [PRECIP[:2]])
        fields["total_precipitation_6hr"] = np.maximum(rain[PRECIP[:2]] * PRECIP[2], 0.0)
        tmp = path.with_name(".tmp-" + path.name)
        with open(tmp, "wb") as handle:
            np.savez_compressed(handle, **{k: v.astype(np.float32) for k, v in fields.items()})
        tmp.replace(path)
    fields = dict(fields)
    if static:
        fields.update({k: np.asarray(v, np.float64) for k, v in static.items() if k in STATIC})
    fields["land_sea_mask"] = np.clip(fields["land_sea_mask"], 0.0, 1.0)
    fields["sea_surface_temperature"] = np.where(fields["land_sea_mask"] < 0.5, fields["sea_surface_temperature"], np.nan)
    if SUN in variables:
        from ..state import toa_accumulated

        lat, lon = np.meshgrid(LAT_AXIS, LON_AXIS, indexing="ij")
        fields[SUN] = toa_accumulated(lat, lon, _unix(when))
    absent = [v for v in variables if v not in fields]
    if absent:
        raise KeyError(f"GFS gives no {absent}")
    return np.stack([np.asarray(fields[v], np.float32) for v in variables], -1)


def _levels(variables) -> list[int]:
    return sorted({int(v.split("@")[1]) for v in variables if "@" in v})


def static_from_staging(era5_path) -> dict:
    """ERA5's land mask and surface geopotential (physical units) from a staged record (:func:`naturev1.b300.stage_era5_record`)."""
    from ..era5 import Normalizer

    era5_path = Path(era5_path)
    meta = json.loads(era5_path.with_suffix(".json").read_text())
    normalizer = Normalizer.from_dict(meta["statistics"])
    values = np.load(era5_path, mmap_mode="r")
    last = np.asarray(values[len(values) - 1], np.float32)
    variables = list(meta["variables"])
    return {name: np.asarray(normalizer.decode(last[..., variables.index(name)], name), np.float64)
            for name in STATIC if name in variables}


def gfs_window(when, variables, normalizer, channels: int, history: int = 6, cache="~/.cache/naturev1/gfs",
               static: dict | None = None, reader=gfs_reader) -> dict:
    """
    The model's input at ``when``: ``history`` GFS states six hours apart ending at ``when``, normalised with the
    training statistics and laid out as the staged record is read (``analysis (1, T, P, C)``, ``calendar``,
    ``valid_time``) -- the batch :func:`naturev1.tracker.rollout_member_tracks` and the model take.
    """
    from ..model import calendar_features
    from ..satdata import prepared_to_state

    when = np.datetime64(when, "h")
    times = [when - np.timedelta64(6 * (history - 1 - k), "h") for k in range(history)]
    frames = [prepared_to_state(normalizer.prepare(gfs_state(t, variables, cache, static, reader)), normalizer, channels)
              for t in times]
    stamps = torch.tensor([float(_unix(t)) for t in times], dtype=torch.float64)
    return {"analysis": torch.stack(frames)[None], "calendar": calendar_features(stamps)[None],
            "valid_time": stamps[-1:].clone()}


def compare_to_era5(gfs: np.ndarray, era5: np.ndarray, normalizer) -> dict:
    """
    How far the GFS state is from ERA5's at the same time, per channel, in the normalised units the model reads:
    ``{channel: {"rmse": ..., "correlation": ...}}`` (area-weighted, NaN cells ignored), and the median over channels.
    """
    a, b = normalizer.prepare(gfs.astype(np.float32)), normalizer.prepare(era5.astype(np.float32))
    weight = np.cos(np.radians(LAT_AXIS))[:, None] * np.ones((1, LON_AXIS.size))
    out = {}
    for c, name in enumerate(normalizer.variables):
        x, y = a[..., c], b[..., c]
        ok = np.isfinite(x) & np.isfinite(y)
        w = weight[ok] / weight[ok].sum()
        dx, dy = x[ok] - (w * x[ok]).sum(), y[ok] - (w * y[ok]).sum()
        corr = float((w * dx * dy).sum() / math.sqrt(max((w * dx * dx).sum() * (w * dy * dy).sum(), 1e-30)))
        out[name] = {"rmse": float(math.sqrt((w * (x[ok] - y[ok]) ** 2).sum())), "correlation": corr}
    out["median"] = {k: float(np.median([v[k] for n, v in out.items() if n != "median"])) for k in ("rmse", "correlation")}
    return out


# ----------------------------------------------------------------------------------------------- best track --

def bdeck_track(text: str, storm_id: str = "", name: str = "", until=None, basin: str = ""):
    """
    NHC's best track (ATCF b-deck text, :func:`naturev1.swarm.fixes.fetch_deck`) as a :class:`naturev1.besttrack.Track`
    -- one fix per synoptic time with wind, pressure, status, the 34/50/64 kt radii (nmi, NE/SE/SW/NW) and the radius of
    maximum wind; fixes after ``until`` (a replay's "now") are left out.
    """
    from ..besttrack import Track
    from .fixes import _coord, _time

    rows: dict = {}
    for line in text.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 11 or not f[2].isdigit():
            continue
        when = _time(f[2])
        if until is not None and when > np.datetime64(until, "m"):
            continue
        lat, lon = _coord(f[6], True), _coord(f[7], True)
        if not np.isfinite(lat):
            continue
        row = rows.setdefault(when, {"lat": lat, "lon": lon, "wind": np.nan, "pressure": np.nan, "status": f[10],
                                     "rmw": np.nan, "radii": np.full((3, 4), np.nan)})
        number = lambda k: float(f[k]) if len(f) > k and f[k].lstrip("-").isdigit() and float(f[k]) > 0 else np.nan  # noqa: E731
        row["wind"], row["pressure"] = number(8), number(9)
        if len(f) > 19 and np.isfinite(number(19)):
            row["rmw"] = number(19)
        if len(f) >= 17 and f[11].isdigit() and int(f[11]) in (34, 50, 64):
            row["radii"][(34, 50, 64).index(int(f[11]))] = [number(k) for k in range(13, 17)]
    times = sorted(t for t in rows if (t.astype("datetime64[m]").astype(np.int64) % 360) == 0)
    if not times:
        raise ValueError("no synoptic best-track fixes")
    pick = lambda key: np.array([rows[t][key] for t in times])  # noqa: E731
    return Track(storm_id=storm_id, name=name, time=np.array(times, "datetime64[s]"),
                 latitude=pick("lat").astype(np.float32), longitude=(((pick("lon") + 180.0) % 360.0) - 180.0).astype(np.float32),
                 max_wind_kt=pick("wind").astype(np.float32), min_pressure_hpa=pick("pressure").astype(np.float32),
                 status=pick("status"), rmw_nmi=pick("rmw").astype(np.float32),
                 wind_radii_nmi=np.stack([rows[t]["radii"] for t in times]).astype(np.float32),
                 record=np.array([""] * len(times)), basin=basin)


# --------------------------------------------------------------------------------------------- satellite --

GOES_EAST = (("noaa-goes19", np.datetime64("2025-04-07")), ("noaa-goes16", np.datetime64("2017-12-18")))
GOES_WEST = (("noaa-goes18", np.datetime64("2023-01-04")), ("noaa-goes17", np.datetime64("2019-02-12")))
SUB_LONGITUDE = {"east": -75.2, "west": -137.2}


def goes_source(when, lon: float) -> tuple[str, str] | None:
    """``(bucket, side)`` of the GOES satellite that saw ``lon`` at ``when``, or None (no ABI imager there then)."""
    side = "east" if lon > -106.0 else "west"
    if abs(((lon - SUB_LONGITUDE[side]) + 180.0) % 360.0 - 180.0) > 70.0:
        return None
    when = np.datetime64(when, "D")
    for bucket, since in (GOES_EAST if side == "east" else GOES_WEST):
        if when >= since:
            return bucket, side
    return None


def goes_file(when, band: int, lon: float, directory, tolerance_minutes: float = 20.0) -> Path | None:
    """The full-disk ABI Cloud and Moisture Imagery file of ``band`` nearest ``when`` (within the tolerance)."""
    from ..live import download, list_objects

    source = goes_source(when, lon)
    if source is None:
        return None
    bucket, _ = source
    when = np.datetime64(when, "m")
    best, gap = None, tolerance_minutes
    for hour in (when - np.timedelta64(1, "h"), when, when + np.timedelta64(1, "h")):
        t = hour.astype(object)
        prefix = f"ABI-L2-CMIPF/{t.year}/{t.timetuple().tm_yday:03d}/{t.hour:02d}/"
        try:
            keys = list_objects(bucket, prefix)
        except Exception:
            continue
        for key in keys:
            if f"C{band:02d}_" not in key or "_s" not in key:
                continue
            s = key.split("_s")[1][:13]                                      # YYYYDDDHHMMSS
            start = (np.datetime64(f"{s[:4]}-01-01") + np.timedelta64(int(s[4:7]) - 1, "D")
                     + np.timedelta64(int(s[7:9]) * 60 + int(s[9:11]), "m"))
            minutes = abs(float((start - when) / np.timedelta64(1, "m")))
            if minutes <= gap:
                best, gap = key, minutes
    if best is None:
        return None
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return download(bucket, best, directory / Path(best).name)


def goes_storm_inputs(when, centre: tuple[float, float], directory="~/.cache/naturev1/goes_archive",
                      frame_hours=(0, 3, 6, 9), fetch=None) -> dict:
    """
    The storm-centred satellite inputs at ``when``, cut around the current centre on the GridSat-B1 patch grid:

    * ``satellite``: ``(codes (1, 2, 256, 256) uint8, available (1, 2) bool)`` -- infrared now and six hours earlier,
      the in-model encoder's input;
    * ``satnet``: ``{"codes" (1, 4, 2, 256, 256), "available" (1, 4), "latitude" (1,), "context" (1, 4)}`` -- infrared
      and water vapour now and 3, 6, 9 hours earlier, SatNet-Storm 2.0's input.

    ``fetch(time, band) -> (256, 256) brightness temperature (K) or None`` replaces the GOES download (tests).
    Patches are cached; each full-disk file is deleted as soon as its patch is cut.
    """
    from ..stormsat import PATCH_PIXELS, encode_bt, goes_patch

    directory = Path(directory).expanduser()
    when = np.datetime64(when, "m")

    def from_goes(t, band):
        key = directory / f"patch_{band}_{str(t)[:16].replace(':', '')}_{centre[0]:.2f}_{centre[1]:.2f}.npy"
        if key.exists():
            return np.load(key)
        path = goes_file(t, band, centre[1], directory)
        if path is None:
            return None
        try:
            patch = goes_patch(path, centre, PATCH_PIXELS)
        finally:
            Path(path).unlink(missing_ok=True)
        np.save(key, patch)
        return patch

    fetch = fetch or from_goes
    frames = np.zeros((1, len(frame_hours), 2, PATCH_PIXELS, PATCH_PIXELS), np.uint8)
    seen = np.zeros((1, len(frame_hours)), bool)
    for k, hours in enumerate(frame_hours):
        t = when - np.timedelta64(int(hours * 60), "m")
        ir, wv = fetch(t, 13), fetch(t, 9)
        if ir is not None:
            frames[0, k, 0] = encode_bt(ir)
            seen[0, k] = bool(np.isfinite(ir).mean() > 0.5)
        if wv is not None:
            frames[0, k, 1] = encode_bt(wv)
    local_hour = (when.astype(object).hour + when.astype(object).minute / 60.0 + centre[1] / 15.0) % 24.0
    context = torch.tensor([[abs(centre[0]) / 30.0, math.sin(2 * math.pi * local_hour / 24.0),
                             math.cos(2 * math.pi * local_hour / 24.0), float(seen.mean())]], dtype=torch.float32)
    k0, k6 = list(frame_hours).index(0), list(frame_hours).index(6)
    return {"satellite": (torch.from_numpy(frames[:, [k0, k6], 0].copy()), torch.from_numpy(seen[:, [k0, k6]].copy())),
            "satnet": {"codes": torch.from_numpy(frames), "available": torch.from_numpy(seen),
                       "latitude": torch.tensor([centre[0]], dtype=torch.float32), "context": context}}


# ---------------------------------------------------------------------------------------------- forecast --

def naturev2_forecast(model, window: dict, grid, variables, normalizer, stepper, track, when, satellite: dict | None = None,
                      leads=None, rollout: bool = True, precision: str = "bf16") -> dict:
    """
    The trained model on a live storm. ``window`` from :func:`gfs_window`; ``track`` from :func:`bdeck_track` (its fix
    at ``when`` is "now"); ``satellite`` from :func:`goes_storm_inputs` (or None). Every storm input the storm stage
    trained with is built the same way here: the storm-state vector, the environment and the steering prior from the
    analysis, the physics predictors, the ocean and gale radius, the infrared now and six hours earlier, and
    SatNet-Storm 2.0's estimates from the loop.

    Returns ``{"bundle": build_forecast(...), "rollout": {...} or None, "centre": (lat, lon), "inputs": {...}}``; the
    rollout entries are absolute positions ``(L, 2)`` and the rollout wind (kt) and pressure (hPa) per lead.
    """
    from ..forecast import build_forecast
    from ..steering import steering_track, storm_environment
    from ..storms import storm_state_vector
    from ..thermo import storm_ocean, storm_physics
    from ..tracker import rollout_member_tracks

    device = next(model.parameters()).device
    leads = tuple(leads or model.config.lead_times_hours)
    when = np.datetime64(when, "s")
    points = np.nonzero(track.time.astype("datetime64[s]") == when)[0]
    if not len(points):
        raise ValueError(f"the best track has no fix at {when}")
    point = int(points[0])
    lat, lon = float(track.latitude[point]), float(track.longitude[point])
    centre = torch.tensor([[lat, lon]], dtype=torch.float32, device=device)
    state = torch.from_numpy(storm_state_vector(track, point))[None].to(device)
    batch = {k: v.to(device) for k, v in window.items()}
    now = batch["analysis"][:, -1].float()
    model.eval()
    with torch.no_grad():
        environment = storm_environment(now, variables, normalizer, grid, centre)
        prior = steering_track(now, variables, normalizer, grid, centre, leads)
        physics = storm_physics(now, variables, normalizer, grid, centre, state)
        ocean = None
        if getattr(model, "storm_ocean_embed", None) is not None:
            radii = torch.from_numpy(np.asarray(track.wind_radii_nmi[point], np.float32))[None].to(device)
            ocean = storm_ocean(centre, batch["valid_time"], environment, radii)
        sat = satnet = None
        if satellite is not None and getattr(model, "storm_satellite", None) is not None:
            codes, seen = satellite["satellite"]
            sat = (codes.to(device), seen.to(device))
        if satellite is not None and model.storm_satnet is not None and bool(satellite["satnet"]["available"].any()):
            loop = satellite["satnet"]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and precision == "bf16"):
                satnet = model.satnet_outputs(loop["codes"], loop["available"], loop["latitude"], loop["context"])
            satnet = {k: v.float() for k, v in satnet.items()}
            satnet["present"] = torch.ones(1, dtype=torch.bool, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and precision == "bf16"):
            out = model(analysis=batch["analysis"], analysis_grid=grid, calendar=batch["calendar"], storm_center=centre,
                        storm_state=state, storm_environment=environment,
                        track_prior=prior if model.uses_track_prior() else None, storm_satellite=sat,
                        storm_satnet=satnet, storm_physics=physics, storm_ocean=ocean)
    out = {k: v.float().cpu() for k, v in out.items() if isinstance(v, torch.Tensor)}
    out.setdefault("track_prior", prior.float().cpu())

    rolled = None
    if rollout and model.trained_heads().get("state", 0) > 0:
        gale = np.nanmean(track.wind_radii_nmi[point][0]) * 1.852
        with torch.no_grad():
            members = rollout_member_tracks(model, batch, stepper, centre, leads, variables, precision,
                                            wind_kt=torch.tensor([float(np.nan_to_num(track.max_wind_kt[point], nan=50.0))],
                                                                 device=device),
                                            size_km=torch.tensor([float(gale) if np.isfinite(gale) else 200.0], device=device))
        members = {k: v.float().cpu() for k, v in members.items() if isinstance(v, torch.Tensor)}
        rolled = {"displacement": {k: members[k][0] for k in ("rollout_steering", "rollout_vortex") if k in members}}
        for k in ("rollout_steering", "rollout_vortex", "rollout_ensemble"):
            if k in members:
                d = members[k][0].numpy().astype(np.float64)
                rolled[k] = np.stack([lat + d[:, 0], ((lon + d[:, 1] + 180.0) % 360.0) - 180.0], -1)
        for k in ("rollout_wind_kt", "rollout_pressure_hpa"):
            if k in members:
                rolled[k] = members[k][0].numpy().astype(np.float64)
    issued = dt.datetime.fromtimestamp(int(when.astype(np.int64)), tz=dt.timezone.utc)
    consensus = {"weights": model.track_consensus.detach().cpu(), "storm_state": state[0].cpu()}
    if rolled is not None:
        consensus["rollout"] = rolled["displacement"]
    bundle = build_forecast(out, leads, issued, storm_center=(lat, lon), normalizer=normalizer,
                            trained=model.trained_heads(), inputs=("analysis",), consensus=consensus,
                            storm_physics=physics[0].cpu(), model=model)
    return {"bundle": bundle, "rollout": rolled, "centre": (lat, lon), "issued": issued, "leads": leads,
            "inputs": {"satellite": sat is not None and bool(sat[1].any()), "satnet": satnet is not None,
                       "ocean": ocean is not None}}


def _fmt(lat: float, lon: float) -> str:
    return f"{abs(lat):.1f}{'N' if lat >= 0 else 'S'} {abs(lon):.1f}{'W' if lon < 0 else 'E'}"


def _km(a, b) -> float:
    from ..cyclone.geometry import great_circle_km

    return float(great_circle_km(a[0], a[1], b[0], b[1]))


def naturev2_track(result: dict) -> np.ndarray | None:
    """NatureV2's headline track ``(L, 2)``: its fitted multi-member consensus, else its rollout vortex, else its top scenario."""
    bundle = result["bundle"]
    consensus = bundle.get("track_consensus")
    if isinstance(consensus, dict) and consensus.get("track"):
        return np.array([[p["latitude"], p["longitude"]] for p in consensus["track"]], np.float64)
    if result.get("rollout") and "rollout_vortex" in result["rollout"]:
        return np.asarray(result["rollout"]["rollout_vortex"], np.float64)
    scenarios = bundle.get("track_scenarios")
    if isinstance(scenarios, list) and scenarios:
        return np.array([[p["latitude"], p["longitude"]] for p in scenarios[0]["track"]], np.float64)
    return None


def describe_naturev2(result: dict, swarm_track=None, swarm_leads=None) -> str:
    """
    Markdown lines for a forecast panel: NatureV2's track, intensity, RI, its rollout, and how far it sits from the swarm's
    consensus ``swarm_track`` ``(L', 2)`` at ``swarm_leads`` (matched by forecast hour; NatureV2 also forecasts 18 h).
    """
    bundle, leads = result["bundle"], result["leads"]
    lines = []
    if not bundle.get("trustworthy", True):
        lines.append(f"**NatureV2 warning:** {bundle.get('warning', 'not trustworthy')}")
    track = naturev2_track(result)
    if track is not None:
        lines.append("**NatureV2 track:** " + "  ".join(f"{h} h {_fmt(*p)}" for h, p in zip(leads, track) if np.isfinite(p[0])))
    intensity = bundle.get("intensity")
    if isinstance(intensity, list):
        lines.append("**NatureV2 intensity:** " + "  ".join(f"{p['lead_hours']} h {p['max_wind_ms'] * 1.94384:.0f} kt"
                                                          for p in intensity))
    ri = bundle.get("rapid_intensification")
    if isinstance(ri, dict) and ri.get("available", True):
        probs = {k: v for k, v in ri.items() if k.startswith("probability_")}
        if probs:
            lines.append("**NatureV2 rapid intensification (24 h):** " + "; ".join(
                f"+{k.split('_')[1]} {v:.0%}" for k, v in probs.items()))
    roll = result.get("rollout")
    if roll and "rollout_vortex" in roll:
        text = "  ".join(f"{h} h {_fmt(*p)}" for h, p in zip(leads, roll["rollout_vortex"]) if np.isfinite(p[0]))
        if "rollout_wind_kt" in roll:
            text += "; rolled-forward wind " + " ".join(f"{w:.0f}" for w in roll["rollout_wind_kt"]) + " kt"
        lines.append("**NatureV2 full-state rollout:** " + text)
    if swarm_track is not None and track is not None:
        theirs = dict(zip(swarm_leads or leads, np.asarray(swarm_track, np.float64)))
        gaps = [f"{h} h {_km(p, theirs[h]):.0f}" for h, p in zip(leads, track)
                if h in theirs and np.all(np.isfinite([*p, *theirs[h]]))]
        if gaps:
            lines.append("**NatureV2 vs the swarm consensus (km):** " + "  ".join(gaps))
    used = [k for k, v in result["inputs"].items() if v]
    lines.append(f"<sub>NatureV2 read the GFS analysis (6 frames, 89 channels){' + ' + ', '.join(used) if used else ''}. "
                 "It was trained on ERA5; GFS differs a little everywhere.</sub>")
    return "  \n".join(lines)


__all__ = ["bdeck_track", "compare_to_era5", "describe_naturev2", "gfs_reader", "gfs_state", "gfs_window", "goes_file",
           "goes_source", "goes_storm_inputs", "naturev2_forecast", "naturev2_track", "static_from_staging"]
