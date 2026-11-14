# Copyright 2026 Nathan. Apache-2.0.
"""
Pluvium Cyclone against the National Hurricane Center's decks: a season of track forecasts.

* **Training season** (2024 by default): every system's bias and the covariance of their errors, against the best
  track. The consensus variant that will carry Pluvium's name is chosen here as well, by leaving out one storm at a
  time.
* **Test season** (2025): every storm is tracked from the position known in real time, in every member of ECMWF ENS,
  NOAA GEFS, ECMWF HRES, ECMWF AIFS, NOAA GFS and NOAA GraphCast, and the consensus is scored against the best track.

Two comparisons, as the National Hurricane Center makes them:

* **late guidance**: Pluvium Cyclone from a run against the models' own trackers for the same run (``AVNO`` GFS,
  ``EMX`` ECMWF, ``EEMN`` and ``AEMN`` ensemble means, ``GDMN`` Google DeepMind, ``GRPH`` GraphCast);
* **early guidance**: the same forecast 12 hours later, slid to the storm's observed position
  (:mod:`pluvium.cyclone.interp`), against what a forecaster has at advisory time: the official forecast ``OFCL``,
  the consensus aids ``TVCN`` and ``HCCA``, and the early versions ``GDMI``, ``EMXI``, ``AVNI``.

Every comparison is on the cases both sides forecast, with a bootstrap over storms.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..cyclone.fields import LABELS, SYSTEMS, FieldArchive
from ..cyclone.interp import interpolate
from ..cyclone.season import Case, cases_of, forecast, learn
from ..cyclone.storms import load_season, start_hours
from ..cyclone.tracks import TrackStore
from ..cyclone.verify import LEADS, Scoreboard, case_errors
from ..util import log

LATE = ("GDMN", "EMX", "EEMN", "AVNO", "AEMN", "GRPH", "CMC", "HFSA", "HWRF")
EARLY = ("OFCL", "TVCN", "HCCA", "GDMI", "EMXI", "AVNI", "FSSE")
BASELINE = "OCD5"
VARIANTS = ("equal", "weighted", "corrected", "adaptive")
NAMES = {"OFCL": "NHC official forecast", "TVCN": "NHC consensus TVCN", "HCCA": "NHC corrected consensus HCCA",
         "GDMN": "Google DeepMind (late)", "GDMI": "Google DeepMind (early)", "EMX": "ECMWF HRES (NHC tracker)",
         "EMXI": "ECMWF HRES (early)", "EEMN": "ECMWF ENS mean (NHC tracker)", "AVNO": "NOAA GFS (NHC tracker)",
         "AVNI": "NOAA GFS (early)", "AEMN": "NOAA GEFS mean (NHC tracker)", "GRPH": "GraphCast (NHC tracker)",
         "CMC": "Canadian global model", "HFSA": "NOAA HAFS-A hurricane model", "HWRF": "NOAA HWRF hurricane model",
         "FSSE": "Florida State superensemble", "OCD5": "climatology and persistence",
         "PLUVIUM": "Pluvium Cyclone (late)", "PLVI": "Pluvium Cyclone (early)"}


def leave_one_storm_out(cases: list, systems=SYSTEMS, gains=(0.5, 1.0, 2.0)) -> dict:
    """Mean track error (km) of each consensus variant on a season, every storm forecast with bias and weights learned
    on the other storms: ``{variant: {lead: km}}``. ``adaptive`` is tried with each in-storm gain."""
    storms = sorted({c.storm.id for c in cases})
    board = Scoreboard()
    for sid in storms:
        rest = [c for c in cases if c.storm.id != sid]
        mine = [Case(c.storm, c.init, c.sets, c.points, carq=c.carq) for c in cases if c.storm.id == sid]
        if len({c.storm.id for c in rest}) < 3:
            continue
        bias, weights = learn(rest, systems)
        for gain in gains:
            forecast(mine, bias, weights, adaptive=True, memory_gain=gain)
            for c in mine:
                board.add(f"adaptive gain {gain:g}", sid, c.init, case_errors(c.consensus["adaptive"], c.storm.track, c.init))
        for c in mine:
            for v in ("equal", "weighted", "corrected"):
                board.add(v, sid, c.init, case_errors(c.consensus[v], c.storm.track, c.init))
    techs = ["equal", "weighted", "corrected"] + [f"adaptive gain {g:g}" for g in gains]
    table = board.mean_errors(techs, unit="km")
    return {t: {lead: row[t] for lead, row in table.items()} for t in techs}


def choose(scores: dict) -> str:
    """The variant with the smallest mean error over leads (leads count equally, as error ratios to ``equal``)."""
    ref = scores["equal"]
    best, best_v = None, None
    for name, by_lead in scores.items():
        ratio = float(np.mean([by_lead[k] / ref[k] for k in by_lead if k in ref and ref[k] > 0]))
        if best is None or ratio < best:
            best, best_v = ratio, name
    return best_v


def run(decks, fields, out, train_year: int = 2024, test_year: int = 2025, basin: str = "al", systems=SYSTEMS,
        train_cycles=(0,), test_cycles=(0, 12)) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    archive = FieldArchive(fields)
    store = TrackStore(out / "tracks")
    train_storms = load_season(decks, train_year, basin)
    test_storms = load_season(decks, test_year, basin)
    train = cases_of(archive, train_storms, start_hours(train_storms, train_cycles), store, systems, progress=log)
    test = cases_of(archive, test_storms, start_hours(test_storms, test_cycles), store, systems, progress=log)
    log(f"cyclone benchmark: {len(train)} training cases ({train_year}), {len(test)} test cases ({test_year})")

    inner = leave_one_storm_out(train, systems)
    chosen = choose(inner)
    gain = float(chosen.split()[-1]) if chosen.startswith("adaptive") else 1.0
    variant = "adaptive" if chosen.startswith("adaptive") else chosen
    log(f"cyclone benchmark: chosen on {train_year} by leaving one storm out: {chosen}")
    bias, weights = learn(train, systems)
    forecast(test, bias, weights, adaptive=True, memory_gain=gain)

    board = Scoreboard()
    early_board = Scoreboard()
    aid_cache = {s.id: s.aids(set(LATE) | set(EARLY) | {BASELINE, "CARQ"}) for s in test_storms}
    for c in test:
        best = c.storm.track
        for s, pts in c.points.items():
            board.add(f"P-{s}", c.storm.id, c.init, case_errors(pts, best, c.init))
        for v in VARIANTS:
            if v in c.consensus:
                board.add(f"consensus {v}", c.storm.id, c.init, case_errors(c.consensus[v], best, c.init))
        board.add("PLUVIUM", c.storm.id, c.init, case_errors(c.consensus[variant], best, c.init))
        aids = aid_cache[c.storm.id]
        for tech in LATE + (BASELINE,):
            rows = aids.get(tech, {}).get(c.init)
            if rows:
                board.add(tech, c.storm.id, c.init, case_errors({t: (v[0], v[1]) for t, v in rows.items()}, best, c.init))
        # early guidance: this forecast, 12 hours later, slid to the position observed then
        later = c.init + 12
        fix = aids.get("CARQ", {}).get(later, {}).get(0)
        if fix is not None and best.tropical(later):
            pts = interpolate(c.consensus[variant], (fix[0], fix[1]), 12)
            early_board.add("PLVI", c.storm.id, later, case_errors(pts, best, later))
            for tech in EARLY + (BASELINE,):
                rows = aids.get(tech, {}).get(later)
                if rows:
                    early_board.add(tech, c.storm.id, later,
                                    case_errors({t: (v[0], v[1]) for t, v in rows.items()}, best, later))

    def block(b: Scoreboard, mine: str, rivals) -> dict:
        res = {}
        for rival in rivals:
            rows = {}
            for lead in LEADS:
                v = b.versus(mine, rival, lead)
                if v is not None:
                    keys = b.common([mine, rival], lead)
                    v["mine_nm"] = float(np.mean([b.errors[mine][k] for k in keys])) / 1.852
                    v["rival_nm"] = float(np.mean([b.errors[rival][k] for k in keys])) / 1.852
                    rows[str(lead)] = v
            if rows:
                res[rival] = rows
        return res

    ours = [f"P-{s}" for s in systems] + [f"consensus {v}" for v in VARIANTS]
    result = {
        "train_year": train_year, "test_year": test_year, "systems": list(systems),
        "cases": {"train": len(train), "test": len(test), "storms": sorted({c.storm.name for c in test})},
        "chosen": chosen, "inner": {k: {str(a): b for a, b in v.items()} for k, v in inner.items()},
        "weights": weights.weights, "bias": {s: {str(k): v for k, v in t.items()} for s, t in bias.table().items()},
        "late": block(board, "PLUVIUM", LATE + tuple(ours)),
        "early": block(early_board, "PLVI", EARLY),
        "skill_late": {t: {str(k): v for k, v in board.skill(t, BASELINE).items()} for t in ("PLUVIUM",) + LATE},
        "skill_early": {t: {str(k): v for k, v in early_board.skill(t, BASELINE).items()} for t in ("PLVI",) + EARLY},
        "homogeneous_late": {str(k): v for k, v in board.mean_errors(["PLUVIUM", "GDMN", "EMX", "AVNO"]).items()},
        "homogeneous_early": {str(k): v for k, v in early_board.mean_errors(["PLVI", "OFCL", "TVCN", "HCCA"]).items()},
        "own": {str(k): v for k, v in board.mean_errors(ours + ["PLUVIUM"]).items()},
    }
    (out / "cyclone.json").write_text(json.dumps(result, indent=1, default=float))
    return result


def markdown(result: dict) -> str:
    """The cyclone benchmark as Markdown."""
    def versus(block: dict, mine: str, order) -> list:
        lines = ["| Against | " + " | ".join(f"{k} h" for k in LEADS) + " |", "|---|" + "---|" * len(LEADS)]
        for rival in order:
            rows = block.get(rival)
            if not rows:
                continue
            cells = []
            for lead in LEADS:
                v = rows.get(str(lead))
                if not v:
                    cells.append("-")
                    continue
                txt = f"{v['mine_nm']:.0f} / {v['rival_nm']:.0f} ({v['n']})"
                cells.append(f"**{txt}**" if v["ci95"][1] < 0 else (f"*{txt}*" if v["ci95"][0] > 0 else txt))
            lines.append(f"| {NAMES.get(rival, LABELS.get(rival[2:], rival))} | " + " | ".join(cells) + " |")
        return lines

    c = result["cases"]
    lines = [f"**Storms ({result['test_year']}):** {', '.join(s.title() for s in c['storms'])}. {c['test']} forecasts; "
             f"bias and weights from {c['train']} forecasts of {result['train_year']}. Consensus variant chosen on "
             f"{result['train_year']} by leaving one storm out: {result['chosen']}.", "",
             "Each cell: Pluvium's mean error / the other's mean error in nautical miles on the cases both forecast "
             "(number of cases). **Bold**: Pluvium better with 95% confidence (bootstrap over storms). *Italic*: worse "
             "with 95% confidence.", "", "**Early guidance: what is in hand at advisory time**", ""]
    lines += versus(result["early"], "PLVI", EARLY) + ["", "**Late guidance: the same run**", ""]
    lines += versus(result["late"], "PLUVIUM", LATE) + ["", "**Pluvium Cyclone against its own ingredients** (the same "
              "tracker on each system, and the consensus at each stage)", ""]
    own = [k for k in result["late"] if k.startswith("P-") or k.startswith("consensus")]
    lines += versus(result["late"], "PLUVIUM", own) + [""]
    lines += ["**Weights learned on " + str(result["train_year"]) + "**", "",
              "| Leads | " + " | ".join(LABELS[s] for s in result["systems"]) + " |",
              "|---|" + "---|" * len(result["systems"])]
    for group, w in result["weights"].items():
        lines.append(f"| {group} | " + " | ".join(f"{w.get(s, 0):.2f}" for s in result["systems"]) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="pluvium cyclone-bench", description="Pluvium Cyclone against the NHC decks.")
    p.add_argument("--decks", required=True)
    p.add_argument("--fields", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-year", type=int, default=2024)
    p.add_argument("--test-year", type=int, default=2025)
    a = p.parse_args(argv)
    result = run(a.decks, a.fields, a.out, a.train_year, a.test_year)
    (Path(a.out) / "CYCLONE_BENCHMARK.md").write_text(markdown(result))


if __name__ == "__main__":
    main()
