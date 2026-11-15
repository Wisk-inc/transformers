# Copyright 2026 Nathan. Apache-2.0.
"""
Pluvium Cyclone for the storms that are out there now.

:func:`run` does everything for one moment in time:

1. reads the season's decks (the National Hurricane Center's public folders for the current year, its archive for a
   past one), so it knows which storms exist and where each was at every hour, as known then;
2. fetches sea-level pressure from the six systems for the recent 00 and 12 UTC runs and tracks every storm in every
   member (tracks are kept on disk, so each run is tracked once);
3. steps the forecaster through those forecasts in time order, starting from what was learned on the 2024 and 2025
   seasons (:func:`prior_state`, shipped with the package), and returns the guidance for the newest run.

Nothing is stored between calls except the decks, the fields and the tracks: the learned state is rebuilt by replaying
the recent forecasts, so a call for a past moment gives exactly what would have been issued then.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..util import fetch, log
from .atcf import fetch_decks
from .basins import BASINS
from .fields import SYSTEMS, FieldArchive
from .products import Forecaster, Guidance
from .report import bulletin, to_json
from .season import cases_of
from .storms import load_season
from .tracks import TrackStore

LIVE_A = "https://ftp.nhc.noaa.gov/atcf/aid_public/a{basin}{number:02d}{year}.dat.gz"
LIVE_B = "https://ftp.nhc.noaa.gov/atcf/btk/b{basin}{number:02d}{year}.dat"
PRIOR = Path(__file__).with_name("data") / "prior_2025.json"
#: hours after a run starts before all six systems' pressure fields are out
RUN_DELAY = 9


def prior_state() -> dict | None:
    """What the forecaster had learned by the end of the 2025 seasons (weights and cone), shipped with the package."""
    return json.loads(PRIOR.read_text()) if PRIOR.exists() else None


def fetch_live_decks(folder, year: int, basin: str = "al", numbers=range(1, 41), stop_after: int = 3) -> list:
    """Download the current season's decks into ``folder`` (refreshed on every call: they grow while a storm lives).
    Stops after ``stop_after`` storm numbers in a row do not exist. Returns the b-deck paths."""
    folder = Path(folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    found, missing = [], 0
    for number in numbers:
        b = fetch(LIVE_B.format(basin=basin, number=number, year=year), attempts=3, missing_ok=True)
        if b is None:
            missing += 1
            if missing >= stop_after:
                break
            continue
        missing = 0
        path = folder / f"b{basin}{number:02d}{year}.dat"
        path.write_bytes(b)
        a = fetch(LIVE_A.format(basin=basin, number=number, year=year), attempts=3, missing_ok=True)
        if a is not None:
            (folder / f"a{basin}{number:02d}{year}.dat").write_bytes(gzip.decompress(a))
        found.append(path)
    return found


def latest_cycle(now: int, delay: int = RUN_DELAY) -> int:
    """The newest 00 or 12 UTC start time whose fields should be complete at ``now`` (hours since the epoch)."""
    t = int(now) - delay
    return t - t % 12


def recent_starts(storms: list, cycle: int, replay_days: int = 10) -> np.ndarray:
    """The 00 and 12 UTC start times in the last ``replay_days`` up to ``cycle`` at which any storm was a tropical
    cyclone (a storm need not have a future yet, unlike in verification)."""
    out = set()
    for storm in storms:
        for hour in storm.track.hours:
            hour = int(hour)
            if hour % 12 == 0 and cycle - 24 * replay_days <= hour <= cycle and storm.track.tropical(hour):
                out.add(hour)
    return np.array(sorted(out), np.int64)


def run(root, now: int | None = None, basins=("al", "ep"), systems=SYSTEMS, download: bool = True,
        replay_days: int = 10, decks=None, fields=None, tracks=None, use_prior: bool = True, progress=log) -> list:
    """Guidance for every storm alive at the newest run: ``[(storm, Guidance)]``, also written under
    ``root/cyclone`` as a bulletin (.txt) and data (.json) per storm.

    ``decks``, ``fields`` and ``tracks`` default to folders under ``root``. With ``download=False`` only what is
    already there is used (for replaying a past moment from a staged archive)."""
    root = Path(root).expanduser()
    now = int(datetime.now(timezone.utc).timestamp() // 3600) if now is None else int(now)
    year = int(str(np.datetime64(now, "h"))[:4])
    this_year = datetime.now(timezone.utc).year
    decks = Path(decks) if decks else root / "decks"
    fields = Path(fields) if fields else root / "pressure"
    store = TrackStore(Path(tracks) if tracks else root / "tracks")
    cycle = latest_cycle(now)
    forecaster = Forecaster(list(systems))
    if use_prior and prior_state() is not None:
        forecaster.load_state(prior_state())
    cases, basin_of = [], {}
    for prefix in basins:
        key = next(k for k, b in BASINS.items() if b.atcf == prefix)
        if download:
            (fetch_live_decks if year == this_year else fetch_decks)(decks, year, prefix)
        storms = load_season(decks, year, prefix)
        starts = recent_starts(storms, cycle, replay_days)
        if starts.size == 0:
            continue
        archive = FieldArchive(fields, key)
        if download:
            for init in starts:
                for system in systems:
                    archive.read(system, int(init))
        found = cases_of(archive, storms, starts, store, systems)
        for c in found:
            basin_of[id(c)] = key
        cases += found
        if progress:
            progress(f"cyclone live: {BASINS[key].name}: {len(storms)} storms this season, {len(found)} recent forecasts")
    out = []
    folder = root / "cyclone"
    for case in sorted(cases, key=lambda c: (c.init, c.storm.id)):
        newest = case.init == cycle
        g = forecaster.step(case, basin_of[id(case)], clouds=newest)
        if newest:
            folder.mkdir(parents=True, exist_ok=True)
            stamp = str(np.datetime64(int(case.init), "h")).replace("-", "").replace("T", "")[:10]
            (folder / f"{case.storm.id}_{stamp}.txt").write_text(bulletin(g, case.storm.name.title()))
            (folder / f"{case.storm.id}_{stamp}.json").write_text(json.dumps(to_json(g, case.storm.name.title()), indent=1))
            out.append((case.storm, g))
    return out


__all__ = ["LIVE_A", "LIVE_B", "RUN_DELAY", "Guidance", "fetch_live_decks", "latest_cycle", "prior_state", "recent_starts",
           "run"]
