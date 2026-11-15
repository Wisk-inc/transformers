# Copyright 2026 Nathan. Apache-2.0.
"""
The time-of-day forecast checked against six-hour satellite rain.

For past runs of the six systems, :mod:`pluvium.timing` says how the rain of each of the first three days is spread
over its four six-hour periods. CMORPH says how it really was. Three questions are scored at the area's land cells:

1. **On days when it rained, which periods were wet?** The chance that a period is wet given a wet day, scored
   with the Brier score on the days that were wet. Compared: what is usual at the place (the satellite record, with
   the days around the one scored left out), what the systems' members say, and mixtures of the two.
2. **Is the wettest period the one forecast?** How often the period given the largest share is the one that got
   the most rain. A coin with four sides gets 25%.
3. **The chance of rain in a period**, with no condition: the chance of a wet day times the answer to 1, against
   the usual chance of rain in that period at that time of year. Shown with the members' own chance of a wet day
   and with the blend's calibrated one. The blend was trained on these same days, so the second is flattered and is
   printed for comparison only.

    python -m pluvium.benchmarks.timing_bench --root ~/pluvium-JM

``--root`` is a folder with a trained blend (``pluvium now`` or ``pluvium blend train`` made it). Six-hour fields
and six-hour truth are downloaded into it once (about 45 minutes for 61 runs and 13 months).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..sources.archive import ARCHIVE_SYSTEMS, ArchiveReader
from ..timing import DAYS, PERIODS, WET_MM, at_places, conditional, member_shares, period_amounts, steps_for, usual_timing
from ..util import hour_of, hour_to_datetime64, log, parallel_map

MIXES = (0.0, 0.25, 0.5, 0.75, 1.0)          # weight of the members against what is usual


def month_of(hour: int) -> int:
    return int(str(hour_to_datetime64(int(hour)))[5:7])


def stage(root, region: dict, runs, systems=ARCHIVE_SYSTEMS, first: str | None = None, end: str | None = None) -> None:
    """Download what the benchmark needs into ``root``: six-hour fields of every run (``archive6``) and six-hour
    truth from ``first`` to ``end`` (``truth6``). What is already there is not fetched again."""
    from ..sources.cmorph_parts import stage_parts

    root = Path(root)
    reader = ArchiveReader(root / "archive6", region)
    for system in systems:
        ok = parallel_map(lambda init, s=system: reader.read_steps(s, int(init), steps_for(int(init), int(init))),
                          runs, workers=2, progress=f"{system} runs", every=120.0)
        log(f"six-hour archive: {system}: {int(sum(ok))} of {len(runs)} runs")
    first = first or str(hour_to_datetime64(int(runs[0])))[:13]
    end = end or str(hour_to_datetime64(int(runs[-1]) + 24 * (DAYS + 1)))[:13]
    stage_parts(root / "truth6", region, first, end)


def collect(root, name: str, lat: np.ndarray, lon: np.ndarray, pick: tuple, runs, systems=ARCHIVE_SYSTEMS) -> dict:
    """For every run and each of its first three days, at the cells ``pick`` (row and column indices into the truth
    grid): what was observed in each period and what each system's members said.

    Returns ``obs`` ``(runs, days, 4, cells)`` mm, and per system ``wet_day`` ``(runs, days, cells)``,
    ``wet_period`` and ``amount`` ``(runs, days, 4, cells)`` (NaN where the system has no such day).
    """
    from ..sources.cmorph_parts import PartsTruth

    root = Path(root)
    truth = PartsTruth(root / "truth6")
    reader = ArchiveReader(root / "archive6", {name: (0, 0, 0, 0)})
    ii, jj = pick
    cell_lat, cell_lon = lat[ii], lon[jj]
    n = ii.size
    obs = np.full((len(runs), DAYS, PERIODS, n), np.nan, np.float32)
    per = {s: {"wet_day": np.full((len(runs), DAYS, n), np.nan), "wet_period": np.full((len(runs), DAYS, PERIODS, n), np.nan),
               "amount": np.full((len(runs), DAYS, PERIODS, n), np.nan)} for s in systems}
    for r, init in enumerate(runs):
        obs[r] = truth.days(name, int(init) + 24 * np.arange(DAYS))[:, :, ii, jj]
        for system in systems:
            z = reader.load(name, f"{system}_6h", int(init))
            if z is None:
                continue
            got = at_places(period_amounts(z, int(init), int(init)), z, cell_lat, cell_lon)
            shares = member_shares({system: got})
            for d in range(DAYS):
                if shares["systems"][d] > 0:
                    per[system]["wet_day"][r, d] = shares["wet_day"][d]
                    per[system]["wet_period"][r, d] = shares["wet_period"][d]
                    per[system]["amount"][r, d] = shares["amount"][d]
    return {"obs": obs, "systems": per, "runs": np.asarray(runs, np.int64)}


def pooled(per: dict, systems) -> dict:
    """Shares pooled over ``systems``, each counting the same where it has the day (as
    :func:`pluvium.timing.member_shares` does for a live forecast)."""
    stack = lambda key: np.stack([per[s][key] for s in systems])              # noqa: E731
    have = np.isfinite(stack("wet_day"))                                      # (systems, runs, days, cells)
    count = np.maximum(have.sum(0), 1)
    return {"wet_day": np.nansum(stack("wet_day"), 0) / count,
            "wet_period": np.nansum(stack("wet_period"), 0) / count[:, :, None, :],
            "amount": np.nansum(stack("amount"), 0) / count[:, :, None, :],
            "any": have.any(0)}


def usual_for(truth_rain: np.ndarray, truth_days: np.ndarray, target: int, leave: int = 7) -> tuple:
    """What is usual at each cell around day ``target``, learned with the days within ``leave`` of it left out:
    ``(chance a period is wet on a wet day, chance a period is wet)``, each ``(4, cells)``."""
    keep = np.abs(truth_days - target) > 24 * leave
    months = np.array([month_of(h) for h in truth_days[keep]])
    rain = truth_rain[keep]
    given = usual_timing(rain, months)[month_of(target) - 1]
    near = np.isin(months, [((month_of(target) - 1 + k) % 12) + 1 for k in (-1, 0, 1)])
    known = np.isfinite(rain[near])
    plain = (np.where(known, rain[near], 0) >= WET_MM).sum(0) / np.maximum(known.sum(0), 1)
    return given, plain


def brier(p: np.ndarray, y: np.ndarray, ok: np.ndarray) -> float:
    return float(np.mean((p[ok] - y[ok]) ** 2)) if ok.any() else float("nan")


def score(data: dict, truth_rain: np.ndarray, truth_days: np.ndarray, calibrated: np.ndarray | None = None,
          systems=ARCHIVE_SYSTEMS, mixes=MIXES) -> dict:
    """Score what ``collect`` gathered. ``truth_rain`` ``(days, 4, cells)`` and ``truth_days`` are the whole truth
    record at the cells (for what is usual); ``calibrated`` is the blend's chance of a wet day ``(runs, days,
    cells)`` or None. Returns the three tables of the module docstring, by day and overall."""
    obs, runs = data["obs"], data["runs"]
    both = pooled(data["systems"], systems)
    known = np.isfinite(obs).all(2) & both["any"]                              # (runs, days, cells)
    wet_day = known & (np.nansum(obs, 2) >= WET_MM)
    y = (obs >= WET_MM).astype(np.float64)                                     # (runs, days, 4, cells)
    usual = np.zeros_like(y)
    usual_plain = np.zeros_like(y)
    for r, init in enumerate(runs):
        for d in range(DAYS):
            usual[r, d], usual_plain[r, d] = usual_for(truth_rain, truth_days, int(init) + 24 * d)
    members = np.stack([conditional({"wet_day": both["wet_day"][r], "wet_period": both["wet_period"][r]})
                        for r in range(len(runs))])                             # (runs, days, 4, cells)
    with_usual = np.stack([conditional({"wet_day": both["wet_day"][r], "wet_period": both["wet_period"][r]}, usual[r])
                           for r in range(len(runs))])
    out = {"runs": int(len(runs)), "cells": int(obs.shape[-1]), "wet_days": int(wet_day.sum()),
           "known_days": int(known.sum()), "given_wet": {}, "wettest": {}, "period": {}, "by_system": {}}
    spans = {"day 1": [0], "day 2": [1], "day 3": [2], "days 1-3": [0, 1, 2]}

    def over(span, mask):                                                      # (runs, days, cells) -> periods too
        m = np.zeros_like(mask)
        m[:, span] = mask[:, span]
        return np.broadcast_to(m[:, :, None, :], y.shape)

    for label, span in spans.items():
        on_wet = over(span, wet_day)
        row = {"base rate": float(y[on_wet].mean()) if on_wet.any() else float("nan"),
               "usual": brier(usual, y, on_wet), "members": brier(members, y, on_wet)}
        for w in mixes:
            row[f"mix {w:g}"] = brier(w * with_usual + (1 - w) * usual, y, on_wet)
        out["given_wet"][label] = row
        # the wettest period: forecast by the largest share of the day's rain (members), or the largest usual chance
        sel = np.zeros_like(wet_day)
        sel[:, span] = wet_day[:, span]
        seen = np.argmax(np.nan_to_num(obs), 2)                                # (runs, days, cells)
        hits = {"usual": np.argmax(usual, 2), "members": np.argmax(both["amount"], 2),
                "mix 0.5": np.argmax(0.5 * with_usual + 0.5 * usual, 2)}
        out["wettest"][label] = {k: float((v[sel] == seen[sel]).mean()) if sel.any() else float("nan")
                                 for k, v in hits.items()}
        every = over(span, known)
        raw_day = both["wet_day"][:, :, None, :]
        row = {"usual": brier(usual_plain, y, every), "members alone": brier(both["wet_period"], y, every),
               "members' day x mix 0.5": brier(np.clip(raw_day * (0.5 * with_usual + 0.5 * usual), 0, 1), y, every)}
        if calibrated is not None:
            day = np.nan_to_num(calibrated)[:, :, None, :]
            have = every & np.broadcast_to(np.isfinite(calibrated)[:, :, None, :], y.shape)
            row["blend's day x members"] = brier(np.clip(day * with_usual, 0, day), y, have)
            row["blend's day x mix 0.5"] = brier(np.clip(day * (0.5 * with_usual + 0.5 * usual), 0, day), y, have)
            row["blend's day x usual"] = brier(np.clip(day * usual, 0, day), y, have)
            row["usual, on the blend's days"] = brier(usual_plain, y, have)
        out["period"][label] = row
    all_wet = over([0, 1, 2], wet_day)
    for system in systems:
        one = pooled(data["systems"], [system])
        ok = all_wet & np.broadcast_to(one["any"][:, :, None, :], y.shape)
        cond = np.stack([conditional({"wet_day": one["wet_day"][r], "wet_period": one["wet_period"][r]})
                         for r in range(len(runs))])
        out["by_system"][system] = {"given wet": brier(cond, y, ok), "usual on the same days": brier(usual, y, ok),
                                    "days": int(ok.sum() // PERIODS)}
    return out


def report(result: dict) -> str:
    lines = [f"{result['runs']} runs, {result['cells']} land cells, {result['known_days']:,} cell-days scored, "
             f"{result['wet_days']:,} of them wet", "",
             "1. On wet days, which periods were wet? Brier score (lower is better)"]
    for label, row in result["given_wet"].items():
        lines.append(f"  {label:9s} " + "  ".join(f"{k} {v:.4f}" for k, v in row.items()))
    lines += ["", "2. The wettest period forecast is the wettest observed (a guess: 25%)"]
    for label, row in result["wettest"].items():
        lines.append(f"  {label:9s} " + "  ".join(f"{k} {100 * v:.1f}%" for k, v in row.items()))
    lines += ["", "3. Chance of rain in a period, Brier score (lower is better)"]
    for label, row in result["period"].items():
        lines.append(f"  {label:9s} " + "  ".join(f"{k} {v:.4f}" for k, v in row.items()))
    lines += ["", "Each system alone, on wet days (question 1)"]
    for system, row in result["by_system"].items():
        lines.append(f"  {system:10s} {row['given wet']:.4f}   usual on the same days {row['usual on the same days']:.4f}"
                     f"   ({row['days']:,} cell-days)")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Check the time-of-day forecast against six-hour satellite rain.")
    parser.add_argument("--root", required=True, help="a folder with a trained blend")
    parser.add_argument("--every", type=int, default=6, help="score a run every this many days")
    parser.add_argument("--start", default=None, help="first run (default: the start of the blend's training period)")
    parser.add_argument("--end", default=None, help="last run (default: the end of the blend's training period)")
    parser.add_argument("--no-download", action="store_true", help="score what is already in the folder")
    parser.add_argument("--blend", action="store_true",
                        help="also score periods with the blend's calibrated chance of a wet day. The blend was "
                             "trained on these days, and runs it has only three days of are read to fourteen")
    args = parser.parse_args(argv)
    from ..benchmarks.fullyear import land_mask
    from ..frontier.model import load_blend
    from ..sources.cmorph_parts import PartsTruth

    root = Path(args.root).expanduser()
    model = load_blend(root / "blend")
    name, box = model.area_json["name"], tuple(model.area_json["target_box"])
    trained = model.meta.get("trained", ["2025-05-16T00", "2026-05-16T00"])
    a, b = hour_of(args.start or trained[0]), hour_of(args.end or trained[1]) - 24 * DAYS
    runs = np.arange(a - a % 24, b, 24 * args.every)
    if not args.no_download:
        stage(root, {name: box}, runs, first=str(hour_to_datetime64(int(runs[0]) - 24 * 15))[:13],
              end=str(hour_to_datetime64(int(runs[-1]) + 24 * (DAYS + 15)))[:13])
    truth = PartsTruth(root / "truth6")
    lat, lon = truth.grid(name)
    mask = land_mask(ArchiveReader(root / "archive", {name: box}), name, lat, lon, root / "masks")
    ii, jj = np.nonzero(mask)
    data = collect(root, name, lat, lon, (ii, jj), runs)
    days = truth.held(name)
    rain = truth.days(name, days)[:, :, ii, jj]
    calibrated = np.full((len(runs), DAYS, ii.size), np.nan)
    for r, init in enumerate(runs if args.blend else []):                       # the blend's own chance of a wet day
        try:
            got = model.forecast(lat[ii], lon[jj], root / "archive", now=int(init) + 12, days=DAYS)
            calibrated[r] = np.stack([d["p_ge_1"] for d in got["days"]])
        except Exception as error:
            log(f"blend forecast for {str(hour_to_datetime64(int(init)))[:10]} not made ({type(error).__name__})")
    result = score(data, rain, days, calibrated if np.isfinite(calibrated).any() else None)
    result["area"], result["every"] = name, args.every
    result["period_covered"] = [str(hour_to_datetime64(int(runs[0])))[:10], str(hour_to_datetime64(int(runs[-1])))[:10]]
    (root / "timing_bench.json").write_text(json.dumps(result, indent=1))
    print(report(result))


__all__ = ["MIXES", "collect", "main", "pooled", "report", "score", "stage", "usual_for"]

if __name__ == "__main__":
    main()
