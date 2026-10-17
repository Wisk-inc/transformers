# Copyright 2026 Nathan. Apache-2.0.
"""
The benchmark: every member, the consensus and the baselines on whole seasons -- and the teachers on the same
storms -- verified homogeneously, with storm-block bootstrap intervals.

:func:`run_season` computes the member suite (:func:`naturev1.cyclone.members.run_members`) for every case of
a built season in parallel processes and caches the forecasts; :func:`benchmark` scores them against the
strongest extrapolation, CLIPER and any teacher tracks given, and :func:`report` writes the table a reader
needs: mean track error per lead, sample sizes, skill and whether its whole 90% interval clears zero.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np

from .data import LEADS, Season

_WORKER: dict = {}


def _init(season_dir: str, climatology_path: str | None, analogs_path: str | None, members) -> None:
    from .climatology import Climatology

    _WORKER["season"] = Season(season_dir)
    _WORKER["climatology"] = Climatology(climatology_path) if climatology_path else None
    _WORKER["analogs"] = pickle.load(open(analogs_path, "rb")) if analogs_path else None
    _WORKER["members"] = tuple(members)


def _one(i: int):
    from .members import run_members

    season = _WORKER["season"]
    case = season.cases[i]
    try:
        out = run_members(case, season.environment(i), _WORKER["climatology"], _WORKER["analogs"], names=_WORKER["members"])
    except Exception as error:                       # one bad case must not end a season
        return case.key, {"_error": repr(error)}
    keep = {k: v for k, v in out.items() if not k.startswith("_")}
    if "_ensemble" in out:
        keep["ensemble_spread_km"] = out["_ensemble"]["spread_km"]
    return case.key, keep


def run_season(season_dir: str | Path, out_path: str | Path, climatology_path=None, analogs_path=None, members=None,
               processes: int = 4, log=print) -> dict:
    """Member forecasts for every case of a season, cached in ``out_path`` (a pickle)."""
    from .members import DEFAULT_MEMBERS

    out_path = Path(out_path)
    if out_path.exists():
        return pickle.load(open(out_path, "rb"))
    members = members or DEFAULT_MEMBERS
    season = Season(season_dir)
    results = {}
    with mp.Pool(processes, initializer=_init, initargs=(str(season_dir), climatology_path and str(climatology_path),
                                                          analogs_path and str(analogs_path), members)) as pool:
        for n, (key, value) in enumerate(pool.imap(_one, range(len(season)), chunksize=8), start=1):
            results[key] = value
            if log and n % 250 == 0:
                log(f"[benchmark] {Path(season_dir).name}: {n}/{len(season)}")
    pickle.dump(results, open(out_path, "wb"))
    return results


def consensus_tracks(cases, results: dict, table: dict | None = None) -> np.ndarray:
    """``(N, L, 2)`` the shipped consensus (:data:`naturev1.cyclone.coefficients.CONSENSUS`) of cached members."""
    from .consensus import apply_consensus

    if table is None:
        from .coefficients import CONSENSUS as table
    nan = np.full((len(LEADS), 2), np.nan)
    stack = np.stack([np.stack([results.get(c.key, {}).get(m, nan) for c in cases]) for m in table["members"]], 2)
    return apply_consensus(cases, stack, table)


def benchmark(cases, forecasts: dict, baseline: str = "best_xtrp", samples: int = 1000) -> dict:
    """Homogeneous comparison of ``forecasts`` (``{name: (N, L, 2)}``) against the strongest extrapolation."""
    from .baselines import baseline_tracks, strongest_extrapolation
    from .verify import compare, track_errors

    base = baseline_tracks(cases, LEADS)
    errors = {k: track_errors(cases, v) for k, v in base.items()}
    errors = {"best_xtrp": strongest_extrapolation(errors)}
    for name, f in forecasts.items():
        errors[name] = track_errors(cases, f)
    return compare(cases, errors, baseline, LEADS, samples=samples)


def report(result: dict, title: str) -> str:
    from .verify import format_comparison

    return format_comparison(result, title)


def save_result(result: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, indent=1, default=float))


__all__ = ["benchmark", "consensus_tracks", "report", "run_season", "save_result"]
