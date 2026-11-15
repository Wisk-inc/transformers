# Copyright 2026 Nathan. Apache-2.0.
"""Three things the cyclone benchmark's write-up needs beyond the benchmark itself, from the same staged data:

    python scripts/cyclone_checks.py tracker --decks decks --fields pressure --tracks cyclone_out/tracks
    python scripts/cyclone_checks.py storm --name MELISSA --out melissa.json --decks ... --fields ... --tracks ...
    python scripts/cyclone_checks.py prior --out pluvium/cyclone/data/prior_2025.json --decks ... --fields ... --tracks ...

``tracker`` compares Pluvium's tracker with the National Hurricane Center's on the same model runs. ``storm`` follows
one storm through every forecast (track error, cone, passage chances). ``prior`` writes what the forecaster had
learned by the end of the seasons, the state the live tool starts from.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pluvium.benchmarks import cyclone_bench as cb
from pluvium.cyclone.approach import observed_passage
from pluvium.cyclone.fields import SYSTEMS, FieldArchive
from pluvium.cyclone.places import PLACES
from pluvium.cyclone.products import Forecaster
from pluvium.cyclone.season import cases_of
from pluvium.cyclone.storms import load_season, start_hours
from pluvium.cyclone.tracks import TrackStore
from pluvium.cyclone.verify import Scoreboard, case_errors

PAIRS = (("ifs", "EMX"), ("gfs", "AVNO"), ("ens", "EEMN"), ("gefs", "AEMN"), ("graphcast", "GRPH"))
LEARNER = dict(method="minvar", memory_gain=1.0)


def season_cases(args, season):
    storms = load_season(args.decks, season.year, season.basin)
    store = TrackStore(args.tracks)
    return storms, cases_of(FieldArchive(args.fields, season.archive), storms, start_hours(storms, season.cycles), store, SYSTEMS)


def all_cases(args):
    cases, basin_of = [], {}
    for season in cb.DEFAULT_SEASONS:
        for c in season_cases(args, season)[1]:
            basin_of[id(c)] = season.archive
            cases.append(c)
    return sorted(cases, key=lambda c: (c.init, c.storm.id)), basin_of


def tracker(args) -> None:
    for season in cb.DEFAULT_SEASONS:
        storms, cases = season_cases(args, season)
        aids = {s.id: s.aids({t for _, t in PAIRS}) for s in storms}
        board = Scoreboard()
        for c in cases:
            for system, tech in PAIRS:
                if system in c.points:
                    board.add(f"P-{system}", c.storm.id, c.init, case_errors(c.points[system], c.storm.track, c.init))
                pts = cb.deck_points(aids[c.storm.id], tech, c.init)
                if pts:
                    board.add(tech, c.storm.id, c.init, case_errors(pts, c.storm.track, c.init))
        print(season.label)
        for system, tech in PAIRS:
            rows = board.mean_errors([f"P-{system}", tech])
            print(f"  {system:9s} ours / NHC ({tech}), nm: " +
                  " ".join(f"{k}h {v[f'P-{system}']:.0f}/{v[tech]:.0f} ({v['n']})" for k, v in rows.items()))


def storm(args) -> None:
    cases, basin_of = all_cases(args)
    forecaster = Forecaster(list(SYSTEMS), LEARNER)
    rows = []
    for case in cases:
        mine = case.storm.name == args.name.upper()
        g = forecaster.step(case, basin_of[id(case)], clouds=mine)
        if not mine:
            continue
        err = case_errors(g.track, case.storm.track, case.init)
        row = {"init": str(np.datetime64(int(case.init), "h"))[:13],
               "errors_nm": {str(k): round(v / 1.852) for k, v in err.items()},
               "cone_nm": {str(k): round(v / 1.852) for k, v in g.cone.items()},
               "scenarios": [round(s.weight, 2) for s in g.scenarios]}
        for name, v in g.places.items():
            plat, plon = PLACES[basin_of[id(case)]][name]
            row[name] = {"chance": round(v["chance"], 2), "raw": round(v["raw_chance"], 2),
                         "track_km": None if not np.isfinite(v["track_km"]) else round(v["track_km"]),
                         "happened": observed_passage(case.storm.track, case.init, plat, plon)}
        rows.append(row)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} forecasts of {args.name} -> {args.out}")


def prior(args) -> None:
    cases, basin_of = all_cases(args)
    forecaster = Forecaster(list(SYSTEMS), LEARNER)
    for case in cases:
        forecaster.step(case, basin_of[id(case)], clouds=False)
    last = max(c.init for c in cases)
    forecaster.learner.learn_until(last + 240)                       # everything issued has verified by then
    forecaster.cone.learn_until(last + 240)
    state = forecaster.state()
    state["about"] = {"seasons": [s.label for s in cb.DEFAULT_SEASONS], "forecasts": len(cases),
                      "learner": "minimum-variance weights + storm memory", "note": "errors against real-time fixes"}
    for part in ("prod", "count"):                                   # rounded: a small file
        state["learner"][part] = {g: [[round(x, 5) for x in row] for row in m] for g, m in state["learner"][part].items()}
    state["learner"]["logw"] = {g: [round(x, 5) for x in v] for g, v in state["learner"]["logw"].items()}
    state["cone"]["rows"] = {k: [[round(e, 1), None if s is None else round(s, 1)] for e, s in rows]
                             for k, rows in state["cone"]["rows"].items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(state, separators=(",", ":")))
    print(f"prior from {len(cases)} forecasts -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("what", choices=["tracker", "storm", "prior"])
    p.add_argument("--decks", required=True)
    p.add_argument("--fields", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--name", default="MELISSA")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.what != "tracker" and not a.out:
        p.error("--out is needed")
    {"tracker": tracker, "storm": storm, "prior": prior}[a.what](a)
