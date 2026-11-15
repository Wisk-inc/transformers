# Copyright 2026 Nathan. Apache-2.0.
"""
Pluvium Cyclone against the National Hurricane Center's decks, season by season.

Every storm is tracked from the position known in real time, in every member of ECMWF ENS, NOAA GEFS, ECMWF HRES,
ECMWF AIFS, NOAA GFS and NOAA GraphCast. The consensus learner (:mod:`pluvium.cyclone.learner`) runs through all
seasons in time order and learns from each forecast only after it has verified, so every forecast is made from the
past alone.

Seasons have one of two roles:

* **develop**: the learner's settings are chosen by their errors on these seasons;
* **test**: scored once, after the choice. Nothing about a test season enters any decision.

Two comparisons, as the National Hurricane Center makes them:

* **late guidance**: Pluvium Cyclone from a run against the models' own trackers for the same run (``AVNO`` GFS,
  ``EMX`` ECMWF, ``EEMN`` and ``AEMN`` ensemble means, ``GDMN`` Google DeepMind, ``GRPH`` GraphCast);
* **early guidance**: the same forecast 12 hours later, slid to the storm's observed position
  (:mod:`pluvium.cyclone.interp`), against what a forecaster has at advisory time: the official forecast ``OFCL``,
  the consensus aids ``TVCN`` and ``HCCA``, and the early versions ``GDMI``, ``EMXI``, ``AVNI``.

``PLUVIUM`` uses the six public systems only. ``PLUVIUM+`` also admits Google DeepMind's published track as a seventh
member: it has no record before 2025, so it starts as an average member and earns its weight online.

Every comparison is on the cases both sides forecast, with a bootstrap over storms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..cyclone.basins import BASINS
from ..cyclone.consensus import consensus
from ..cyclone.fields import LABELS, STEPS, SYSTEMS, FieldArchive
from ..cyclone.interp import interpolate
from ..cyclone.learner import SeasonLearner
from ..cyclone.season import cases_of
from ..cyclone.storms import load_season, start_hours
from ..cyclone.tracks import TrackStore
from ..cyclone.verify import LEADS, Scoreboard, case_errors
from ..util import log
from .cyclone_products import evaluate_products, products_markdown

LATE = ("GDMN", "EMX", "EEMN", "AVNO", "AEMN", "GRPH", "CMC", "HFSA", "HWRF")
EARLY = ("OFCL", "TVCN", "HCCA", "GDMI", "EMXI", "AVNI", "FSSE")
BASELINE = "OCD5"
EXTRA = {"gdm": "GDMN"}                         # extra members read from the decks: system name -> technique
LEARNERS = {                                    # simplest first: ties go to the earlier entry
    "minimum-variance weights": dict(method="minvar"),
    "gradient weights": dict(method="eg", eta=0.1),
    "minimum-variance weights + storm memory": dict(method="minvar", memory_gain=1.0),
    "gradient weights + storm memory": dict(method="eg", eta=0.1, memory_gain=1.0),
    "minimum-variance weights + online bias": dict(method="minvar", online_bias=True),
}
NAMES = {"OFCL": "NHC official forecast", "TVCN": "NHC consensus TVCN", "HCCA": "NHC corrected consensus HCCA",
         "GDMN": "Google DeepMind (late)", "GDMI": "Google DeepMind (early)", "EMX": "ECMWF HRES (NHC tracker)",
         "EMXI": "ECMWF HRES (early)", "EEMN": "ECMWF ENS mean (NHC tracker)", "AVNO": "NOAA GFS (NHC tracker)",
         "AVNI": "NOAA GFS (early)", "AEMN": "NOAA GEFS mean (NHC tracker)", "GRPH": "GraphCast (NHC tracker)",
         "CMC": "Canadian global model", "HFSA": "NOAA HAFS-A hurricane model", "HWRF": "NOAA HWRF hurricane model",
         "FSSE": "Florida State superensemble", "OCD5": "climatology and persistence",
         "PLUVIUM": "Pluvium Cyclone (late)", "PLVI": "Pluvium Cyclone (early)",
         "PLUVIUM+": "Pluvium Cyclone+ (late)", "PLVI+": "Pluvium Cyclone+ (early)"}
BASIN_LABEL = {"al": "North Atlantic", "ep": "East Pacific", "cp": "Central Pacific"}


@dataclass(frozen=True)
class Season:
    basin: str                                  # deck prefix: "al", "ep"
    year: int
    cycles: tuple = (0, 12)
    role: str = "test"                          # "develop" or "test"

    @property
    def label(self) -> str:
        return f"{BASIN_LABEL.get(self.basin, self.basin)} {self.year}"

    @property
    def archive(self) -> str:
        return next(k for k, b in BASINS.items() if b.atcf == self.basin)


DEFAULT_SEASONS = (Season("al", 2024, (0,), "develop"), Season("ep", 2024, (0,), "develop"),
                   Season("al", 2025, (0, 12), "develop"), Season("ep", 2025, (0, 12), "test"))


def prequential(cases: list, learner: SeasonLearner, name: str, extra: dict | None = None) -> None:
    """Run a learner through cases in time order, filling ``case.consensus[name]``. Each forecast is made from what
    verified before its start time. ``extra`` adds members per case: ``{case key: {system: points}}``."""
    for case in sorted(cases, key=lambda c: (c.init, c.storm.id)):
        learner.learn_until(case.init)
        points = dict(case.points)
        if extra:
            points.update(extra.get(case.key, {}))
        case.consensus[name] = learner.predict(case.storm.id, case.init, points, case.carq)


def season_errors(cases: list, name: str) -> dict:
    """Mean error (km) by lead of a consensus variant over cases, against the best track."""
    acc = {}
    for c in cases:
        if name in c.consensus:
            for lead, km in case_errors(c.consensus[name], c.storm.track, c.init).items():
                acc.setdefault(lead, []).append(km)
    return {lead: float(np.mean(v)) for lead, v in sorted(acc.items())}


def ratio_to(errors: dict, reference: dict) -> float:
    """Mean over leads of ``errors / reference`` (leads count equally)."""
    leads = [k for k in errors if k in reference and reference[k] > 0]
    return float(np.mean([errors[k] / reference[k] for k in leads])) if leads else float("nan")


def choose(ratios: dict, margin: float = 0.02, tie: float = 0.005) -> str:
    """The learner to carry Pluvium's name, from error ratios to the equal-weight mean on the development seasons.

    The best ratio must beat the equal-weight mean by ``margin``, or the equal-weight mean is kept: complexity has to
    pay for itself. Among learners within ``tie`` of the best, the one listed first (the simplest) is taken."""
    best = min(ratios.values())
    if not best <= 1.0 - margin:
        return "equal"
    return next(name for name, r in ratios.items() if r <= best + tie)


def deck_points(aids: dict, tech: str, init: int, leads=None) -> dict:
    rows = aids.get(tech, {}).get(int(init))
    if not rows:
        return {}
    return {int(t): (v[0], v[1]) for t, v in rows.items() if t > 0 and (leads is None or int(t) in leads)}


def _block(board: Scoreboard, mine: str, rivals) -> dict:
    res = {}
    for rival in rivals:
        rows = {}
        for lead in LEADS:
            v = board.versus(mine, rival, lead)
            if v is not None:
                keys = board.common([mine, rival], lead)
                v["mine_nm"] = float(np.mean([board.errors[mine][k] for k in keys])) / 1.852
                v["rival_nm"] = float(np.mean([board.errors[rival][k] for k in keys])) / 1.852
                rows[str(lead)] = v
        if rows:
            res[rival] = rows
    return res


def tally(block: dict) -> dict:
    """Per rival: at how many leads Pluvium's mean error is lower, and at how many the 95% interval settles it."""
    out = {}
    for rival, rows in block.items():
        out[rival] = {"leads": len(rows), "lower": sum(v["delta_km"] < 0 for v in rows.values()),
                      "better95": sum(v["ci95"][1] < 0 for v in rows.values()),
                      "worse95": sum(v["ci95"][0] > 0 for v in rows.values())}
    return out


def score_season(cases: list, aid_cache: dict, chosen: str, variants: list, systems) -> dict:
    """Every comparison for one season's cases (their consensus variants already filled)."""
    board, early_board = Scoreboard(), Scoreboard()
    for c in cases:
        best, aids = c.storm.track, aid_cache[c.storm.id]
        for s, pts in c.points.items():
            board.add(f"P-{s}", c.storm.id, c.init, case_errors(pts, best, c.init))
        for v in variants:
            board.add(f"consensus: {v}", c.storm.id, c.init, case_errors(c.consensus[v], best, c.init))
        mine = {"PLUVIUM": c.consensus[chosen], "PLUVIUM+": c.consensus["plus"]}
        for tech, pts in mine.items():
            board.add(tech, c.storm.id, c.init, case_errors(pts, best, c.init))
        for tech in LATE + (BASELINE,):
            pts = deck_points(aids, tech, c.init)
            if pts:
                board.add(tech, c.storm.id, c.init, case_errors(pts, best, c.init))
        later = c.init + 12                                            # early guidance: 12 hours on, slid to the fix
        fix = aids.get("CARQ", {}).get(later, {}).get(0)
        if fix is not None and best.tropical(later):
            for tech, early in (("PLUVIUM", "PLVI"), ("PLUVIUM+", "PLVI+")):
                pts = interpolate(mine[tech], (fix[0], fix[1]), 12, extend_hours=12)
                early_board.add(early, c.storm.id, later, case_errors(pts, best, later))
            for tech in EARLY + (BASELINE,):
                pts = deck_points(aids, tech, later)
                if pts:
                    early_board.add(tech, c.storm.id, later, case_errors(pts, best, later))
    ours = [f"P-{s}" for s in systems] + [f"consensus: {v}" for v in variants]
    late, early = _block(board, "PLUVIUM", LATE), _block(early_board, "PLVI", EARLY)
    late_plus, early_plus = _block(board, "PLUVIUM+", LATE), _block(early_board, "PLVI+", EARLY)
    return {
        "cases": len(cases), "storms": sorted({c.storm.name for c in cases}),
        "late": late, "early": early, "late_plus": late_plus, "early_plus": early_plus,
        "own": _block(board, "PLUVIUM", tuple(ours) + ("PLUVIUM+",)),
        "tally": {"late": tally(late), "early": tally(early), "late_plus": tally(late_plus),
                  "early_plus": tally(early_plus)},
        "skill_late": {t: {str(k): v for k, v in board.skill(t, BASELINE).items()}
                       for t in ("PLUVIUM", "PLUVIUM+") + LATE},
        "skill_early": {t: {str(k): v for k, v in early_board.skill(t, BASELINE).items()}
                        for t in ("PLVI", "PLVI+") + EARLY},
        "homogeneous_early": {str(k): v for k, v in
                              early_board.mean_errors(["PLVI", "OFCL", "TVCN", "HCCA"]).items()},
        "homogeneous_late": {str(k): v for k, v in board.mean_errors(["PLUVIUM", "EMX", "AVNO", "EEMN"]).items()},
    }


def run(decks, fields, out, seasons=DEFAULT_SEASONS, systems=SYSTEMS) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    store = TrackStore(out / "tracks")
    wanted = set(LATE) | set(EARLY) | {BASELINE, "CARQ"} | set(EXTRA.values())
    by_season, aid_cache = {}, {}
    for season in seasons:
        storms = load_season(decks, season.year, season.basin)
        archive = FieldArchive(fields, season.archive)
        cases = cases_of(archive, storms, start_hours(storms, season.cycles), store, systems, progress=log)
        if not cases:
            log(f"cyclone benchmark: {season.label}: no cases (no fields staged?); skipped")
            continue
        by_season[season] = cases
        aid_cache.update({s.id: s.aids(wanted) for s in storms})
        log(f"cyclone benchmark: {season.label} ({season.role}): {len(cases)} cases, "
            f"{len({c.storm.id for c in cases})} storms")
    everything = [c for cases in by_season.values() for c in cases]
    develop = [c for season, cases in by_season.items() if season.role == "develop" for c in cases]

    # ---------------------------------------------------- every learner, once through all seasons, causally --
    for c in everything:
        c.consensus["equal"] = consensus(c.points)
    learners, ratios, by = {}, {}, {}
    for name, kw in LEARNERS.items():
        learners[name] = SeasonLearner(list(systems), **kw)
        prequential(everything, learners[name], name)
        ratios[name] = ratio_to(season_errors(develop, name), season_errors(develop, "equal"))
        by[name] = {s.label: ratio_to(season_errors(cases, name), season_errors(cases, "equal"))
                    for s, cases in by_season.items()}
    chosen = choose(ratios)
    log("cyclone benchmark: error against the equal-weight mean on the development seasons: " +
        ", ".join(f"{k} {v:.3f}" for k, v in ratios.items()) + f" -> {chosen}")

    kw = LEARNERS.get(chosen, dict(method="equal"))
    extra = {c.key: {s: p for s, tech in EXTRA.items()
                     if (p := deck_points(aid_cache[c.storm.id], tech, c.init, set(STEPS)))} for c in everything}
    plus = SeasonLearner(list(systems) + list(EXTRA), **kw)
    prequential(everything, plus, "plus", extra)

    variants = ["equal"] + list(LEARNERS)
    result = {
        "systems": list(systems), "chosen": chosen, "develop_ratios": ratios, "ratios_by_season": by,
        "roles": {s.label: s.role for s in by_season},
        "seasons": {s.label: score_season(cases, aid_cache, chosen, variants, systems) for s, cases in by_season.items()},
        "weights_end": learners[chosen].table() if chosen in learners else {},
        "weights_plus": plus.table(),
        "products": evaluate_products(by_season, systems, kw),
    }
    (out / "cyclone.json").write_text(json.dumps(result, indent=1, default=float))
    return result


def _versus(block: dict, order) -> list:
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
        label = NAMES.get(rival) or (LABELS.get(rival[2:], rival) + " (Pluvium's tracker)" if rival.startswith("P-")
                                     else rival)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def season_markdown(label: str, role: str, s: dict) -> list:
    note = ("Scored once, after every setting was fixed." if role == "test" else
            "A development season: the learner's settings were chosen on it. Its weights were still learned causally.")
    lines = [f"## {label} ({role})", "", f"{note} Storms: {', '.join(x.title() for x in s['storms'])}. "
             f"{s['cases']} forecasts.", "", "### Early guidance: what is in hand at advisory time", "",
             "**Pluvium Cyclone (six public systems)**", ""]
    lines += _versus(s["early"], EARLY) + ["", "**Pluvium Cyclone+ (with Google DeepMind's published track as a "
                                               "seventh member)**", ""]
    lines += _versus(s["early_plus"], EARLY) + ["", "### Late guidance: the same run", "",
                                                 "**Pluvium Cyclone (six public systems)**", ""]
    lines += _versus(s["late"], LATE) + ["", "**Pluvium Cyclone+**", ""]
    lines += _versus(s["late_plus"], LATE) + ["", "### Pluvium Cyclone against its own ingredients", "",
              "The same tracker on each system, and the consensus under each learner.", ""]
    lines += _versus(s["own"], list(s["own"])) + [""]
    return lines


def markdown(result: dict) -> str:
    """The cyclone benchmark as Markdown."""
    ratios = ", ".join(f"{k} {v:.3f}" for k, v in result["develop_ratios"].items())
    lines = [f"**Chosen on the development seasons:** {result['chosen']}. Error relative to the equal-weight mean "
             f"there: {ratios}.", "",
             "Each cell: Pluvium's mean error / the other's mean error in nautical miles on the cases both forecast "
             "(number of cases). **Bold**: Pluvium better with 95% confidence (bootstrap over storms). *Italic*: worse "
             "with 95% confidence.", ""]
    order = sorted(result["seasons"], key=lambda k: (result["roles"][k] != "test", k))
    for label in order:
        lines += season_markdown(label, result["roles"][label], result["seasons"][label])
    if result.get("products"):
        lines += products_markdown(result["products"], result["roles"])
    lines += ["## Weights at the end", ""]
    for title, table in (("Pluvium Cyclone", result["weights_end"]), ("Pluvium Cyclone+", result["weights_plus"])):
        if not table:
            continue
        names = list(next(iter(table.values())))
        lines += [f"**{title}**", "", "| Leads | " + " | ".join(LABELS.get(s, "Google DeepMind") for s in names) + " |",
                  "|---|" + "---|" * len(names)]
        for group, w in table.items():
            lines.append(f"| {group} | " + " | ".join(f"{w.get(s, 0):.2f}" for s in names) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="pluvium cyclone-bench", description="Pluvium Cyclone against the NHC decks.")
    p.add_argument("--decks", required=True, help="folder with the a- and b-decks of every season")
    p.add_argument("--fields", required=True, help="folder the pressure fields were staged in")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    result = run(a.decks, a.fields, a.out)
    (Path(a.out) / "CYCLONE_BENCHMARK.md").write_text(markdown(result))


if __name__ == "__main__":
    main()
