# Copyright 2026 Nathan. Apache-2.0.
"""
One call from an NHC storm id to a Caribbean impact report: best track and fixes from NHC, the GFS analysis, the
NatureV1 physics consensus with the satellite-filtered swarm around it, and the parish-by-parish impacts.

* **Live** (``when=None``): the newest synoptic time that has both a best-track fix and a GFS analysis; the swarm is
  filtered on the 24 hours of centre fixes before it, launched, and then re-weighted by every fix that has arrived
  *since* (the satellite path keeps the forecast current between analysis cycles).
* **Replay** (``when="2025-10-27T00"``, ``archive=True``): exactly what was known at that time -- fixes after it are
  never used.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from .analysis import GFSAnalysis, case_from_bdeck
from .fixes import fetch_deck, merge, parse_bdeck, parse_fdeck, reweight_forecast, window

BASINS = {"al": "NA", "ep": "EP", "cp": "EP"}


def _gfs_available(gfs: GFSAnalysis, when: np.datetime64) -> bool:
    from .analysis import GFS, _get

    stamp = str(np.datetime64(when, "h")).replace("-", "").replace("T", "")[:10]
    try:
        _get(GFS.format(day=stamp[:8], hour=int(stamp[8:])) + ".idx", 0, 64, timeout=30.0)
        return True
    except Exception:
        return False


def forecast_storm(storm_id: str, when=None, archive: bool = False, particles: int = 100_000, device="cpu",
                   region=None, climatology=None, analogs=None, impact_particles: int = 20_000,
                   cache: str | Path = "~/.cache/naturev1", name: str = "") -> dict:
    """
    Args:
        storm_id: NHC id, e.g. ``"al132025"`` (Melissa) or ``"ep212026"``.
        when: analysis time for a replay (None: live).
        region: a :class:`naturev1.swarm.geography.RegionMap` for impacts (None: track only).
        climatology, analogs: for the full 13-member consensus (without them the basic consensus is used).

    Returns:
        ``{"case", "result" (physics_forecast output, swarm included), "impact", "fixes_before", "fixes_after",
        "best", "analysis_time", "ess_after"}``.
    """
    from ..cyclone.forecast import physics_forecast
    from .impact import impact_from_forecast

    cache = Path(cache).expanduser()
    best = parse_bdeck(fetch_deck(storm_id, "b", archive))
    try:
        fdeck = parse_fdeck(fetch_deck(storm_id, "f", archive))
    except Exception:
        fdeck = []
    gfs = GFSAnalysis(cache / "gfs")
    if when is None:
        synoptic = [f.time for f in best if int(str(f.time)[11:13]) % 6 == 0 and str(f.time)[14:16] == "00"]
        for t in sorted(synoptic, reverse=True)[:4]:
            if _gfs_available(gfs, t) and _gfs_available(gfs, t - np.timedelta64(12, "h")):
                when = t
                break
        if when is None:
            raise RuntimeError("no recent analysis with both a best-track fix and a GFS run yet")
    when = np.datetime64(when, "m")
    case = case_from_bdeck(best, when, storm_id.upper(), name, BASINS.get(storm_id[:2].lower(), "NA"))
    env = gfs.environment(when, case.lat, case.lon)
    before = merge(window(fdeck, when - np.timedelta64(24, "h"), when))
    result = physics_forecast(case, env, climatology, analogs, swarm=particles, swarm_fixes=before or None,
                              swarm_device=device)
    after, ess_after = [], None
    if not archive and fdeck:
        now = np.datetime64(dt.datetime.now(dt.timezone.utc).replace(tzinfo=None), "m")
        after = merge(window(fdeck, when + np.timedelta64(1, "m"), now))
        if after:
            ess_after = reweight_forecast(result["swarm"]["forecast"], after, when)
    impact = impact_from_forecast(region, result, case, particles=impact_particles, device=device) if region is not None else None
    return {"case": case, "result": result, "impact": impact, "fixes_before": before, "fixes_after": after,
            "best": best, "analysis_time": when, "ess_after": ess_after}


def goes_scene(lat: float, lon: float, directory: str | Path = "~/.cache/naturev1/goes", half_deg: float = 8.0,
               stride: int = 2):
    """The newest GOES infrared (band 13) scene around a point: GOES-19 east of 110 W, GOES-18 west of it."""
    from ..live import fetch_latest
    from ..satellite import scene_from_netcdf

    satellite = "east" if lon > -110.0 else "west"
    paths = fetch_latest(Path(directory).expanduser(), ("C13",), satellite, "fulldisk")
    return scene_from_netcdf({k: str(v) for k, v in paths.items()},
                             bounds=(lat - half_deg, lat + half_deg, lon - half_deg, lon + half_deg), stride=stride)


def consensus_inputs(directory: str | Path = "~/.cache/naturev1", log=print) -> tuple:
    """
    The climatology (ERA5 1981-2010 day-of-year means, from WeatherBench 2) and the track-analog database (IBTrACS
    1980-2014, every basin) that the full 13-member consensus needs -- built once into ``directory`` (about 2.5 GB and
    20-30 minutes the first time), then just loaded. Same construction as the shipped coefficients assume.
    """
    import pickle

    from ..cyclone.analogs import AnalogDatabase
    from ..cyclone.climatology import Climatology, build_climatology
    from ..cyclone.data import LEADS, cases_from_tracks
    from ..ibtracs import fetch_ibtracs, parse_ibtracs

    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    climatology = Climatology(build_climatology(directory / "climatology.npy", log=log))
    db_path = directory / "analogs.pkl"
    if db_path.exists():
        analogs = pickle.loads(db_path.read_bytes())
    else:
        tracks = parse_ibtracs(fetch_ibtracs(directory), since=1979, until=2015)
        history = cases_from_tracks(tracks, range(1980, 2015), init_hours=(0, 6, 12, 18))
        analogs = AnalogDatabase.from_cases(history, LEADS)
        db_path.write_bytes(pickle.dumps(analogs))
        log(f"analog database: {len(history):,} historical forecasts")
    return climatology, analogs


__all__ = ["consensus_inputs", "forecast_storm", "goes_scene"]
