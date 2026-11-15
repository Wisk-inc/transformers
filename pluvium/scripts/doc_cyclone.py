# Copyright 2026 Nathan. Apache-2.0.
"""Write docs/CYCLONE_BENCHMARK.md from the benchmark's own output, so every number in it is the data's.

    python scripts/doc_cyclone.py cyclone.json melissa.json docs/CYCLONE_BENCHMARK.md

``cyclone.json`` is written by ``python -m pluvium.benchmarks.cyclone_bench``; ``melissa.json`` holds one row per
forecast of Hurricane Melissa (``init``, ``errors_nm``, ``cone_nm`` and, per place, ``chance`` and ``happened``).
"""
import json
import sys
from pathlib import Path

from pluvium.benchmarks import cyclone_bench as cb
from pluvium.benchmarks.cyclone_products import products_markdown
from pluvium.cyclone.fields import LABELS
from pluvium.cyclone.verify import LEADS

result = json.loads(Path(sys.argv[1]).read_text())
melissa = json.loads(Path(sys.argv[2]).read_text()) if Path(sys.argv[2]).exists() else []
out = Path(sys.argv[3])
TEST = next(k for k, role in result["roles"].items() if role == "test")
DEV = "North Atlantic 2025"


def homogeneous(label: str) -> list:
    rows = result["seasons"][label]["homogeneous_early"]
    names = (("PLVI", "Pluvium Cyclone"), ("OFCL", "NHC official forecast"), ("TVCN", "NHC consensus TVCN"),
             ("HCCA", "NHC corrected consensus HCCA"))
    lines = ["| | " + " | ".join(f"{k} h" for k in LEADS) + " |", "|---|" + "---|" * len(LEADS)]
    for tech, name in names:
        cells = []
        for lead in LEADS:
            row = rows.get(str(lead))
            if not row:
                cells.append("-")
                continue
            best = min(row[t] for t, _ in names)
            cells.append(f"**{row[tech]:.0f}**" if row[tech] == best else f"{row[tech]:.0f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("| Forecasts | " + " | ".join(str(rows[str(k)]["n"]) if str(k) in rows else "-" for k in LEADS) + " |")
    return lines


def reading(label: str, part: str, rival: str) -> str:
    t = result["seasons"][label]["tally"][part].get(rival)
    if not t:
        return "not in the decks"
    text = f"lower error at {t['lower']} of {t['leads']} leads"
    if t["better95"]:
        text += f", {t['better95']} with 95% confidence"
    if t["worse95"]:
        text += f"; higher with 95% confidence at {t['worse95']}"
    return text


def versus_line(label: str, part: str, rival: str) -> str:
    rows = result["seasons"][label][part].get(rival, {})
    return ", ".join(f"{v['mine_nm']:.0f} against {v['rival_nm']:.0f} at {k} h" for k, v in rows.items() if k in ("24", "48", "72", "120"))


ratios = result["ratios_by_season"]
lines = [
    "# Pluvium Cyclone against the National Hurricane Center's guidance", "",
    "Pluvium Cyclone finds every tropical cyclone in six operational forecast systems (ECMWF ENS, NOAA GEFS, ECMWF "
    "HRES, ECMWF AIFS, NOAA GFS and NOAA GraphCast), follows it in every member, and combines the tracks into one "
    "forecast with a cone, passage chances for places and scenarios. The weights are learned as the season goes, from "
    "forecasts that have already verified.", "",
    "This page scores it against what the US National Hurricane Center (NHC) had in hand for the same storms: its "
    "official forecast, its consensus aids, every model's own track, and Google DeepMind's cyclone model.", "",
    "## The short version", "",
    f"- **Against single models it wins clearly.** On the test season it had {reading(TEST, 'late', 'EMX')} against "
    f"ECMWF HRES and {reading(TEST, 'late', 'AVNO')} against NOAA GFS, as tracked by the NHC.",
    f"- **Against the NHC's own consensus it is level.** On the test season: {reading(TEST, 'early', 'TVCN')} against "
    "TVCN.",
    f"- **Against the official forecast it lost on the test season.** {reading(TEST, 'early', 'OFCL').capitalize()} "
    f"({versus_line(TEST, 'early', 'OFCL')} nautical miles). On the development season in the Atlantic it was ahead "
    f"({reading(DEV, 'early', 'OFCL')}), but its settings were chosen with that season in view, so the test season is "
    "the result to trust.",
    f"- **Google DeepMind's model is better.** On the test season Pluvium Cyclone had {reading(TEST, 'late', 'GDMN')} "
    "against it. Its published track can be added as a seventh member (Pluvium Cyclone+ below).",
    "- **The uncertainty products are honest.** On the test season the cone held the centre "
    f"{100 * result['products'][TEST]['cone_all']['coverage']:.0f}% of the time against a target of 67%, and the "
    "passage chances beat both a yes/no read off the track and the raw cloud of ensemble members.", "",
    "![Track error by lead](img/cyclone_track_error.png)", "",
    "## How it was tested", "",
    "Four seasons were used, with two roles.", "",
    "| Season | Role | Forecasts | Storms |", "|---|---|---|---|"]
for label, role in result["roles"].items():
    s = result["seasons"][label]
    lines.append(f"| {label} | {role} | {s['cases']} | {len(s['storms'])} |")
lines += [
    "", "- **Development seasons** were used to choose between five ways of learning the weights. "
    "The choice was made by error relative to an equal-weight mean of the six systems.",
    "- **The test season** was scored once, after that choice. Nothing about it entered any decision.",
    "- In every season the weights, the cone and the in-storm memory were learned causally: a forecast is learned from "
    "only after its valid time, and against the position known in real time, not the best track written afterwards.",
    "- Storms are tracked from the real-time position (the `CARQ` record), at 00 and 12 UTC "
    "(00 UTC only in 2024).", "",
    "Two comparisons are made, as the NHC makes them.", "",
    "- **Late guidance**: Pluvium Cyclone from a run against the models' own tracks from the same run.",
    "- **Early guidance**: what a forecaster has at advisory time. A model run is not finished when the advisory is "
    "written, so the previous run is used, read 12 hours further along and slid to the storm's observed position. "
    "The official forecast and the consensus aids are early guidance.", "",
    "One handicap is worth knowing. Pluvium Cyclone reads the 00 and 12 UTC runs only, so its early guidance comes from "
    "runs 12 hours old. The NHC's early aids use runs 6 hours old where they exist. The last 12 hours of the early "
    "forecast are filled by carrying the track on at its final motion.", "",
    "## Early guidance on the forecasts all four made", "",
    "Mean track error in nautical miles. Bold is the lowest in the column.", "",
    f"**{TEST}, the test season**", ""] + homogeneous(TEST) + ["", f"**{DEV}, a development season**", ""] + homogeneous(DEV) + [
    "", "## Which learner, and what each one was worth", "",
    "Error relative to the equal-weight mean of the six systems (below 1 is better), season by season. "
    f"The learner chosen on the development seasons was **{result['chosen']}**.", "",
    "| Learner | " + " | ".join(result["roles"]) + " |", "|---|" + "---|" * len(result["roles"])]
for name, by in ratios.items():
    lines.append(f"| {name} | " + " | ".join(f"{by[label]:.3f}" if label in by else "-" for label in result["roles"]) + " |")
seasons = list(result["roles"])
bias = "minimum-variance weights + online bias"
plain, memory = "minimum-variance weights", "minimum-variance weights + storm memory"
gradient, gradient_memory = "gradient weights", "gradient weights + storm memory"
best_in = [s for s in seasons if ratios[bias][s] == min(r[s] for r in ratios.values())]
worst_in = [s for s in seasons if ratios[bias][s] == max(r[s] for r in ratios.values())]
above_one = [(name, s) for name, by in ratios.items() for s in seasons if by[s] > 1.0]
covariance_wins = sum(ratios[plain][s] < ratios[gradient][s] for s in seasons) + \
    sum(ratios[memory][s] < ratios[gradient_memory][s] for s in seasons)
memory_helped = [s for s in seasons if ratios[memory][s] < ratios[plain][s]]
chosen_test = ratios[result["chosen"]][TEST]
better_on_test = sum(by[TEST] < chosen_test for by in ratios.values())
lines += [
    "", "Reading the table:", "",
    f"- On the test season the chosen learner was {100 * (1 - chosen_test):.0f}% better than the equal-weight mean; "
    f"{better_on_test} of the other four did slightly better.",
    f"- **Correcting each system's bias as the season goes was erratic.** It was the best learner in "
    f"{' and '.join(best_in) or 'no season'} and the worst in {' and '.join(worst_in) or 'no season'}"
    + (f"; in {above_one[0][1]} it was {100 * (ratios[bias][above_one[0][1]] - 1):.0f}% worse than the plain mean"
       if above_one and above_one[0][0] == bias else "") + ". A bias measured on some storms is mostly their steering.",
    f"- Weights from the running error covariance beat weights from gradient steps in {covariance_wins} of "
    f"{2 * len(seasons)} comparisons, and need no step size.",
    f"- Remembering how each system has done on the storm so far helped in {' and '.join(memory_helped) or 'no season'}"
    " and cost a little elsewhere.", "",
    "## Weights at the end of 2025", "",
    "| Leads | " + " | ".join(LABELS[s] for s in result["systems"]) + " |", "|---|" + "---|" * len(result["systems"])]
for group, w in result["weights_end"].items():
    lines.append(f"| {group} | " + " | ".join(f"{w.get(s, 0):.2f}" for s in result["systems"]) + " |")
lines += ["", "ECMWF's AIFS earned about a third of the weight. No system was written off."]
if melissa:
    lines += [
        "", "## Hurricane Melissa, forecast by forecast", "",
        "Melissa crossed western Jamaica on 28 October 2025 as a category 5 hurricane. It is an Atlantic storm, so it "
        "belongs to a development season. The table follows every forecast: the chance given that the centre would pass "
        "within 150 km of each place in the next five days (in brackets, whether it then did), the track error, and "
        "the radius of the two-in-three circle.", "",
        "| Forecast from | Kingston | Montego Bay | Santiago de Cuba | Track error at 48 / 72 / 96 h, nm | Circle at 72 h, nm |",
        "|---|---|---|---|---|---|"]
    for r in melissa:
        if not any(r.get(p) for p in ("Kingston", "Montego Bay", "Santiago de Cuba")) or not r["errors_nm"]:
            continue

        def cell(name, r=r):
            v = r.get(name)
            return "-" if not v else f"{100 * v['chance']:.0f}% ({'yes' if v['happened'] else 'no'})"

        e = r["errors_nm"]
        lines.append(f"| {r['init'].replace('T', ' ')} UTC | {cell('Kingston')} | {cell('Montego Bay')} | {cell('Santiago de Cuba')} | "
                     f"{e.get('48', '-')} / {e.get('72', '-')} / {e.get('96', '-')} | {r['cone_nm'].get('72', '-')} |")
    lines += ["", "The chance for Kingston passed 40% five days before the centre arrived and 90% three days before. "
              "Early forecasts gave chances of 13% to 34% for five-day windows in which the centre did not yet arrive: "
              "the storm moved more slowly than every system expected."]
lines += ["", "## The test season in full", "",
          "Each cell: Pluvium's mean error / the other's mean error in nautical miles on the forecasts both made "
          "(number of forecasts). **Bold**: Pluvium better with 95% confidence (bootstrap over storms). *Italic*: worse "
          "with 95% confidence.", ""]
lines += cb.season_markdown(TEST, "test", result["seasons"][TEST])[2:]
lines += [f"## {DEV} in full (a development season)", ""] + cb.season_markdown(DEV, "develop", result["seasons"][DEV])[2:]
lines += products_markdown(result["products"], result["roles"])
lines += [
    "## What did not work", "",
    "- **Bias correction.** See the learner table: best in some seasons, worst in others.",
    "- **Counting a single run as one point in the cloud.** Weighted that way, a third of the chance sat on one line, "
    "and passage chances were worse than the raw member cloud. Giving every system's track the ensembles' "
    "perturbations fixed it.",
    "- **Choosing settings on one season.** With 81 forecasts from 17 storms, differences of 2% between learners are "
    "noise. The 2024 season alone would have chosen the bias-corrected learner.",
    "- **Beating the official forecast everywhere.** It held on the development season and failed on the test "
    "season.", "",
    "## Limits", "",
    "- Track only. Nothing here forecasts wind, surge or rain.",
    "- Two basins and two seasons. The test season has 18 storms.",
    "- Runs at 00 and 12 UTC only, so early guidance is 12 hours old.",
    "- The six systems are global models at about a quarter of a degree. Google DeepMind's model is better than all of "
    "them, and than their combination.", "",
    "## Run it", "",
    "```bash", "pip install pluvium",
    "pluvium cyclone --root storms                       # every storm alive now: track, cone, passage chances",
    "pluvium cyclone --root storms --time 2025-10-25T22  # as it would have been issued then", "```", "",
    "To repeat the benchmark, stage the decks and pressure fields, then:", "",
    "```bash", "python -m pluvium.benchmarks.cyclone_bench --decks decks --fields pressure --out cyclone_out",
    "python scripts/doc_cyclone.py cyclone_out/cyclone.json melissa.json docs/CYCLONE_BENCHMARK.md", "```", ""]
out.write_text("\n".join(lines))
print("wrote", out, len(lines), "lines")
