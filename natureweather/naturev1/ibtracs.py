# Copyright 2026 Nathan. Apache-2.0.
"""
Every basin, not just the Atlantic: IBTrACS as global training labels.

Stage two learned from 584 Atlantic storms, and the red team showed what that costs. Southern-hemisphere
storms were tracked worse than the steering flow alone at 24 hours; storms that turned poleward from low
latitudes in the Pacific (Otis, Rai) were kept at the latitude they started; a 50-to-130-knot explosion
was called 45 knots. All of it is the signature of heads that have only ever seen one ocean.

IBTrACS v04 merges every agency's best track into one record: roughly ninety tropical cyclones a year,
1980 onward, in all seven basins -- about eight times the Atlantic alone. This module reads it into the
same :class:`naturev1.besttrack.Track` the rest of the package uses, and builds the per-fix label table the
satellite storm network trains on.

Three things about IBTrACS decide whether the labels mean anything:

**Wind is not one measurement.** The US agencies (NHC, CPHC, JTWC) report one-minute sustained wind;
most others report ten-minute, which runs about 12% lower for the same storm. Mixing them unconverted
teaches a model that West Pacific typhoons are weaker than they are. The US value is preferred wherever
it exists; otherwise a ten-minute value is divided by 0.88 (the WMO's own guidance, Harper et al. 2010),
and every fix records which it was.

**Rows between synoptic times are interpolated by IBTrACS itself.** They are kept -- they are where the
three-hourly satellite images are -- but flagged, and the six-hourly targets the storm heads train on are
read only at synoptic hours.

**Hemisphere is a reflection.** A southern-hemisphere cyclone turns clockwise and recurves south. Every
motion here is stored as it is; :func:`hemisphere_sign` is what the model uses to put both hemispheres in
one poleward frame.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .besttrack import Track, download

IBTRACS_BASE = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv"
#: Every storm since 1980 in one file (~ 150 MB) -- the satellite era.
IBTRACS_SINCE_1980 = f"{IBTRACS_BASE}/ibtracs.since1980.list.v04r01.csv"
BASINS = ("NA", "EP", "WP", "NI", "SI", "SP", "SA")
#: Ten-minute to one-minute sustained wind (WMO guidance; Harper et al. 2010, "at-sea" conditions).
TEN_TO_ONE_MINUTE = 1.0 / 0.88
#: Agencies that report one-minute winds as the WMO value: the US centres.
ONE_MINUTE_AGENCIES = {"hurdat_atl", "hurdat_epa", "cphc", "atcf"}
#: USA_STATUS codes that mean a tropical or subtropical cyclone.
TROPICAL_STATUS = {"TD", "TS", "HU", "TY", "ST", "TC", "SD", "SS"}
#: Leads of the intensity-change labels, hours.
CHANGE_HOURS = (12, 24, 36, 48, 72)
KM_PER_DEGREE = 111.195


def _number(value: str | None) -> float:
    value = (value or "").strip()
    if not value:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def hemisphere_sign(latitude):
    """+1 north of the equator (and on it), -1 south. Works on floats, numpy arrays and tensors."""
    if hasattr(latitude, "sign") and hasattr(latitude, "dtype") and not isinstance(latitude, np.ndarray):
        import torch

        return torch.where(latitude < 0, -torch.ones_like(latitude), torch.ones_like(latitude))
    return np.where(np.asarray(latitude) < 0, -1.0, 1.0)


def _wind(row: dict) -> tuple[float, int]:
    usa = _number(row.get("USA_WIND"))
    if np.isfinite(usa):
        return usa, 0
    wmo = _number(row.get("WMO_WIND"))
    if not np.isfinite(wmo):
        return float("nan"), -1
    if (row.get("WMO_AGENCY") or "").strip().lower() in ONE_MINUTE_AGENCIES:
        return wmo, 2
    return wmo * TEN_TO_ONE_MINUTE, 1


def _status(row: dict, wind: float) -> str:
    """HURDAT2-style status: TD/TS/HU for a tropical system by its wind, SS subtropical, EX otherwise."""
    usa = (row.get("USA_STATUS") or "").strip()
    nature = (row.get("NATURE") or "").strip()
    if usa in ("SD", "SS") or (not usa and nature == "SS"):
        return "SS"
    tropical = usa in TROPICAL_STATUS or (not usa and nature in ("TS", "MX"))
    if not tropical or not np.isfinite(wind):
        return "EX"
    return "TD" if wind < 34 else "TS" if wind < 64 else "HU"


def parse_ibtracs(path: str | Path, since: int = 1980, until: int | None = None, basins=BASINS,
                  min_peak_kt: float = 25.0, synoptic_only: bool = False) -> list[Track]:
    """
    Read an IBTrACS v04 CSV into :class:`Track` objects, one per storm, main tracks only.

    Args:
        since, until: seasons to keep (inclusive).
        basins: genesis basins to keep (IBTrACS ``BASIN`` of the storm's first fix).
        min_peak_kt: drop systems that never reach this wind -- disturbances and weak depressions that
            some agencies track and others do not, which would otherwise make labels basin-dependent.
        synoptic_only: keep only 00/06/12/18 UTC fixes.

    Returns:
        Tracks with ``basin``, ``interpolated``, ``wind_source`` and ``distance_to_land_km`` filled.
    """
    storms: dict[str, list[dict]] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        next(reader, None)                                         # the units row
        for row in reader:
            if (row.get("TRACK_TYPE") or "").strip() != "main":
                continue
            season = int(_number(row["SEASON"]))
            if season < since or (until is not None and season > until):
                continue
            stamp = row["ISO_TIME"]
            if stamp[14:16] != "00":
                continue
            if synoptic_only and stamp[11:13] not in ("00", "06", "12", "18"):
                continue
            storms.setdefault(row["SID"], []).append(row)

    tracks = []
    for sid, rows in storms.items():
        basin = (rows[0].get("BASIN") or "").strip()
        if basin not in basins:
            continue
        winds, sources = zip(*(_wind(r) for r in rows))
        wind = np.array(winds, np.float32)
        if not np.isfinite(wind).any() or np.nanmax(wind) < min_peak_kt:
            continue
        pressure = np.array([_number(r.get("USA_PRES")) if np.isfinite(_number(r.get("USA_PRES")))
                             else _number(r.get("WMO_PRES")) for r in rows], np.float32)
        latitude = np.array([_number(r["LAT"]) for r in rows], np.float32)
        longitude = np.array([_number(r["LON"]) for r in rows], np.float32)
        longitude = ((longitude + 180.0) % 360.0 - 180.0).astype(np.float32)
        distance = np.array([_number(r.get("DIST2LAND")) for r in rows], np.float32)
        record = np.array(["L" if i > 0 and distance[i] == 0 and distance[i - 1] > 0 else ""
                           for i in range(len(rows))])
        radii = np.full((len(rows), 3, 4), np.nan, np.float32)
        for i, r in enumerate(rows):
            for g, kt in enumerate((34, 50, 64)):
                for q, quadrant in enumerate(("NE", "SE", "SW", "NW")):
                    radii[i, g, q] = _number(r.get(f"USA_R{kt}_{quadrant}"))
        flags = [(r.get("IFLAG") or "") for r in rows]
        tracks.append(Track(
            storm_id=sid, name=(rows[0].get("NAME") or "").strip(),
            time=np.array([np.datetime64(r["ISO_TIME"][:16].replace(" ", "T"), "s") for r in rows]),
            latitude=latitude, longitude=longitude, max_wind_kt=wind, min_pressure_hpa=pressure,
            status=np.array([_status(r, w) for r, w in zip(rows, wind)]),
            rmw_nmi=np.array([_number(r.get("USA_RMW")) for r in rows], np.float32),
            wind_radii_nmi=radii, record=record, basin=basin,
            interpolated=np.array(["O" not in flag for flag in flags]),
            wind_source=np.array(sources, np.int8), distance_to_land_km=distance,
        ))
    tracks.sort(key=lambda t: (t.time[0], t.storm_id))
    return tracks


def displacement_km(lat0, lon0, lat1, lon1) -> np.ndarray:
    """``(..., 2)`` (north, east) kilometres from one position to another, east scaled at the mid-latitude."""
    lat0, lon0, lat1, lon1 = (np.asarray(x, np.float64) for x in (lat0, lon0, lat1, lon1))
    north = (lat1 - lat0) * KM_PER_DEGREE
    east = ((lon1 - lon0 + 180.0) % 360.0 - 180.0) * KM_PER_DEGREE * np.cos(np.radians((lat0 + lat1) / 2))
    return np.stack([north, east], -1)


def storm_label_table(tracks: list[Track], cadence_hours: int = 3, max_abs_latitude: float = 65.0) -> dict:
    """
    One row per (storm, fix) on the satellite cadence, with every label the storm networks train on.

    Rows are fixes at multiples of ``cadence_hours`` where the wind is known, the system is tropical or
    subtropical, and it is inside the geostationary band. Labels that need the future -- intensity change,
    motion, landfall -- are NaN where the track does not reach that far, never guessed.

    Returns:
        A dict of equal-length numpy columns:

        ``sid, time, lat, lon, basin, hemisphere, wind_kt, pressure_hpa, rmw_nmi, radii_nmi (N, 3, 4),
        wind_change_kt (N, len(CHANGE_HOURS)), tropical, past_motion_km (N, 2) over the previous 6 h,
        future_motion_km (N, 2) over the next 12 h, landfall_24h, landfall_48h (NaN unknown),
        wind_source, interpolated, distance_to_land_km``.
    """
    columns: dict[str, list] = {key: [] for key in (
        "sid", "time", "lat", "lon", "basin", "hemisphere", "wind_kt", "pressure_hpa", "rmw_nmi", "radii_nmi",
        "wind_change_kt", "tropical", "past_motion_km", "future_motion_km", "landfall_24h", "landfall_48h",
        "wind_source", "interpolated", "distance_to_land_km")}
    step = cadence_hours * 3600
    for track in tracks:
        seconds = track.time.astype("datetime64[s]").astype(np.int64)
        index = {int(s): i for i, s in enumerate(seconds)}
        landfall = track.landfall
        for i, now in enumerate(seconds):
            if now % step or not np.isfinite(track.max_wind_kt[i]) or track.status[i] == "EX":
                continue
            if abs(float(track.latitude[i])) > max_abs_latitude:
                continue

            def at(hours: int) -> int:
                return index.get(int(now) + hours * 3600, -1)

            changes = []
            for hours in CHANGE_HOURS:
                j = at(hours)
                ok = j >= 0 and np.isfinite(track.max_wind_kt[j])
                changes.append(float(track.max_wind_kt[j] - track.max_wind_kt[i]) if ok else np.nan)
            before, after = at(-6), at(12)
            past = (displacement_km(track.latitude[before], track.longitude[before], track.latitude[i],
                                    track.longitude[i]) if before >= 0 else np.full(2, np.nan))
            future = (displacement_km(track.latitude[i], track.longitude[i], track.latitude[after],
                                      track.longitude[after]) if after >= 0 else np.full(2, np.nan))
            outcomes = []
            for hours in (24, 48):
                last = at(hours)
                if last < 0:
                    # Known "no" only if the storm dissipated over water before then -- otherwise unknown.
                    end = int(np.searchsorted(seconds, now + hours * 3600, side="right")) - 1
                    reached = bool(landfall[i + 1 : end + 1].any())
                    outcomes.append(1.0 if reached else np.nan)
                else:
                    outcomes.append(float(landfall[i + 1 : last + 1].any()))
            columns["sid"].append(track.storm_id)
            columns["time"].append(int(now))
            columns["lat"].append(float(track.latitude[i]))
            columns["lon"].append(float(track.longitude[i]))
            columns["basin"].append(BASINS.index(track.basin) if track.basin in BASINS else -1)
            columns["hemisphere"].append(1.0 if track.latitude[i] >= 0 else -1.0)
            columns["wind_kt"].append(float(track.max_wind_kt[i]))
            columns["pressure_hpa"].append(float(track.min_pressure_hpa[i]))
            columns["rmw_nmi"].append(float(track.rmw_nmi[i]))
            columns["radii_nmi"].append(track.wind_radii_nmi[i])
            columns["wind_change_kt"].append(changes)
            columns["tropical"].append(track.status[i] in ("TD", "TS", "HU"))
            columns["past_motion_km"].append(past)
            columns["future_motion_km"].append(future)
            columns["landfall_24h"].append(outcomes[0])
            columns["landfall_48h"].append(outcomes[1])
            columns["wind_source"].append(int(track.wind_source[i]) if track.wind_source is not None else 0)
            columns["interpolated"].append(bool(track.interpolated[i]) if track.interpolated is not None else False)
            columns["distance_to_land_km"].append(float(track.distance_to_land_km[i])
                                                  if track.distance_to_land_km is not None else np.nan)
    dtypes = {"sid": str, "time": np.int64, "basin": np.int8, "tropical": bool, "wind_source": np.int8,
              "interpolated": bool}
    table = {}
    for key, values in columns.items():
        if key == "radii_nmi":
            table[key] = np.asarray(values, np.float32).reshape(-1, 3, 4)
        elif key in ("wind_change_kt",):
            table[key] = np.asarray(values, np.float32).reshape(-1, len(CHANGE_HOURS))
        elif key in ("past_motion_km", "future_motion_km"):
            table[key] = np.asarray(values, np.float32).reshape(-1, 2)
        else:
            table[key] = np.asarray(values, dtype=dtypes.get(key, np.float32))
    return table


def describe_table(table: dict) -> str:
    """What the global labels contain -- worth printing before trusting anything trained on them."""
    n = len(table["time"])
    if not n:
        return "no labelled fixes"
    lines = [f"{n:,} storm fixes from {len(set(table['sid'].tolist())):,} storms"]
    for b, name in enumerate(BASINS):
        count = int((table["basin"] == b).sum())
        if count:
            lines.append(f"  {name}  {count:>8,} fixes")
    change = table["wind_change_kt"][:, CHANGE_HOURS.index(24)]
    eligible = np.isfinite(change) & table["tropical"]
    for threshold in (25, 30, 35):
        rate = 100.0 * float((change[eligible] >= threshold).mean()) if eligible.any() else float("nan")
        lines.append(f"  RI >= {threshold} kt / 24 h: {rate:.2f}% of {int(eligible.sum()):,} eligible fixes")
    lines.append(f"  wind from US agencies {100 * float((table['wind_source'] == 0).mean()):.0f}%, "
                 f"converted 10-min {100 * float((table['wind_source'] == 1).mean()):.0f}%")
    lines.append(f"  wind radii observed  {100 * float(np.isfinite(table['radii_nmi'][:, 0]).any(-1).mean()):.0f}%  "
                 f"RMW {100 * float(np.isfinite(table['rmw_nmi']).mean()):.0f}%")
    lines.append(f"  southern hemisphere  {100 * float((table['hemisphere'] < 0).mean()):.0f}%")
    return "\n".join(lines)


def fetch_ibtracs(directory: str | Path, url: str = IBTRACS_SINCE_1980) -> Path:
    """Download the IBTrACS CSV once."""
    return download(url, Path(directory) / Path(url).name)


__all__ = ["BASINS", "CHANGE_HOURS", "IBTRACS_SINCE_1980", "TEN_TO_ONE_MINUTE", "describe_table", "displacement_km",
           "fetch_ibtracs", "hemisphere_sign", "parse_ibtracs", "storm_label_table"]
