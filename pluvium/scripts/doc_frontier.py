# Copyright 2026 Nathan. Apache-2.0.
"""Write docs/FRONTIER_BENCHMARK.md from the benchmark's own output, so every number in it is the data's.

    python scripts/doc_frontier.py RESULTS FIRST EXPERIMENTS docs/FRONTIER_BENCHMARK.md

``RESULTS`` holds, per place, ``{place}.json`` (from :func:`pluvium.frontier.evaluate.evaluate`), ``{place}_tail.json``
and ``{place}_scenarios.json``. ``FIRST`` holds the first benchmark's ``{place}.json``. ``EXPERIMENTS`` holds the
withheld-storm results ``{place}.json`` (optional).
"""
import json
import sys
from pathlib import Path

import numpy as np

from pluvium.benchmarks import fullyear as F
from pluvium.benchmarks import scenario_bench, tail_bench
from pluvium.frontier import report as R

results_dir, first_dir, experiments_dir, out = (Path(p) for p in sys.argv[1:5])
ORDER = ("jamaica", "caribbean", "maritime", "alps", "pacific_nw", "great_plains", "andes_south")
SHORT = {"jamaica": "Jamaica", "caribbean": "Caribbean", "maritime": "Maritime Continent", "alps": "Alps",
         "pacific_nw": "US Pacific Northwest", "great_plains": "US Great Plains", "andes_south": "Southern Andes"}
WINTER = {"alps": "DJF", "pacific_nw": "DJF", "great_plains": "DJF", "andes_south": "JJA"}


def load(folder: Path, suffix: str = "") -> dict:
    return {k: json.loads((folder / f"{k}{suffix}.json").read_text()) for k in ORDER if (folder / f"{k}{suffix}.json").exists()}


new, old = load(results_dir), load(first_dir)
tails, scenarios = load(results_dir, "_tail"), load(results_dir, "_scenarios")
withheld = load(experiments_dir) if experiments_dir.exists() else {}
old = {k: v for k, v in old.items() if k in new}
rivals_only = R.without(new)
G = list(F.LEAD_GROUPS)


def count(key, t, higher=True, groups=None, season=None, places=None):
    use_new = {k: v for k, v in new.items() if places is None or k in places}
    return R.tally(old, use_new, key, t, higher, groups, season)


def mean_of(results, model, key, t, g, season=None):
    vals = []
    for res in results.values():
        row = res["groups"][g] if season is None else res["seasons"][g][season]
        name = F._top(res) if model == "top" else model
        v = F._value(row.get(name), key, t)
        if v == v:
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def confident(threshold: str, groups) -> tuple:
    """(comparisons, better with 95% confidence, worse with 95% confidence) of the Blend against every post-processed
    system at a threshold (bootstrap over runs), over places and lead groups."""
    n = better = worse = 0
    for res in new.values():
        for g in groups:
            for rival, v in res["versus"].get(g, {}).get(threshold, {}).items():
                if rival in R.ABLATIONS or rival in ("blend_ops", "neural_pp", "blend"):
                    continue
                n += 1
                better += v["ci95"][1] < 0
                worse += v["ci95"][0] > 0
    return n, better, worse


def board_rows() -> list:
    lines = ["| Score | " + " | ".join(G) + " |", "|---|" + "---|" * len(G)]
    for title, key, t, higher in F.BOARD:
        cells = []
        for g in G:
            c = count(key, t, higher, [g])
            cells.append(f"{c['new_leads']} of {c['cells']} (was {c['old_leads']})" if c["cells"] else "-")
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    return lines


def winter_rows() -> list:
    lines = ["| Place (its winter) | " + " | ".join(G) + " |", "|---|" + "---|" * len(G)]
    for place, season in WINTER.items():
        if place not in new:
            continue
        cells = []
        for g in G:
            row = new[place]["seasons"][g][season]
            b = F._value(row.get("stack"), "bss", "1")
            a = F._value(old[place]["seasons"][g][season].get(F._top(old[place])), "bss", "1") if place in old else float("nan")
            rival, who = R.best_rival(row, "bss", "1", True)
            if b != b:
                cells.append("-")
                continue
            mark = "**" if rival != rival or b > rival else ""
            cells.append(f"{a:+.3f} -> {mark}{b:+.3f}{mark} ({rival:+.3f}, {F.LABELS.get(str(who).replace('_raw', ''), who)})")
        lines.append(f"| {SHORT[place]} ({season}) | " + " | ".join(cells) + " |")
    return lines


def melissa_rows() -> list:
    """Jamaica on the landfall day: the first Blend, the new one, the raw consensus."""
    if "jamaica" not in tails:
        return []
    day = "2025-10-28"
    rows = sorted((c for c in tails["jamaica"]["cases"] if c["valid"] == day), key=lambda c: c["lead"])
    if not rows:
        return []
    first = {c["lead"]: c for c in old.get("jamaica", {}).get("cases", []) if c["valid"] == day and c["model"] == F._top(old["jamaica"])}
    o = rows[0]
    lines = [f"Observed over Jamaica on {day} (CMORPH): {o['observed_mean_mm']:.0f} mm on average, up to "
             f"{o['observed_max_mm']:.0f} mm; {100 * o['observed_share_50']:.0f}% of places had 50 mm or more and "
             f"{100 * o['observed_share_100']:.0f}% had 100 mm or more.", "",
             "| Days ahead | First Blend: chance of 50 mm, expected mm | New Blend: chance of 50 mm, of 100 mm, expected mm | "
             "Places under an extreme warning | Raw consensus: chance of 50 mm, mean mm |", "|---|---|---|---|---|"]
    for c in rows:
        f = first.get(c["lead"])
        was = "-" if f is None else f"{100 * f['p50']:.0f}%, {f['forecast_mean_mm']:.0f}"
        lines.append(f"| {c['lead']} | {was} | {100 * c['blend_p50']:.0f}%, {100 * c['blend_p100']:.0f}%, {c['blend_mean_mm']:.0f} | "
                     f"{100 * c['extreme_share']:.0f}% | {100 * c['raw_p50']:.0f}%, {c['raw_mean_mm']:.0f} |")
    return lines


def place_table(place: str) -> list:
    res = new[place]
    lines = [f"### {F.REGION_TITLES.get(place, place)}", "", f"{res['cells']:,} places.", "",
             "| Score | " + " | ".join(G) + " |", "|---|" + "---|" * len(G)]
    for title, key, t, higher in F.BOARD:
        cells = []
        for g in G:
            row = res["groups"][g]
            mine = F._value(row.get("stack"), key, t)
            rival, who = R.best_rival(row, key, t, higher)
            if mine != mine:
                cells.append("-")
                continue
            ahead = rival != rival or (mine > rival if higher else mine < rival)
            sign = "+" if key in ("bss", "acc") else ""
            name = str(who)
            label = F.LABELS.get(name.replace("_raw", ""), name) + (" raw" if name.endswith("_raw") else "")
            cells.append(f"{'**' if ahead else ''}{mine:{sign}.3f}{'**' if ahead else ''} ({rival:{sign}.3f}, {label})")
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    weights = R.online_weights(res)
    if weights:
        lines += ["", "Online weights for light rain on days 1-3, at the end of each quarter of the test year:", ""] + weights
    return lines + [""]


n_places = len(new)
c50 = count("bss", "50")
c50_short = count("bss", "50", groups=["days 1-3"])
c1 = count("bss", "1")
n_conf, better_conf, worse_conf = confident("50", G)
n1, better1, worse1 = confident("1", G[:2])
m50 = {g: (mean_of(old, "top", "bss", "50", g), mean_of(new, "stack", "bss", "50", g)) for g in G}
rmse = count("rmse", None, False)
acc = count("acc", None, True)
crps = count("crps", None, False)

lines = [
    "# Pluvium Blend on the frontier: the second full-year benchmark", "",
    "The first full-year benchmark ([FULL_YEAR_BENCHMARK.md](FULL_YEAR_BENCHMARK.md)) found three weak spots in Pluvium "
    "Blend: the heavy tail (50 mm), winter in mid-latitude continental climates, and structure at long range. This "
    "page scores a redesign aimed at them, on the same data, the same test year and the same rules.", "",
    "## Read this first", "",
    "- **The test year is the same as before** (April 2025 to March 2026), and the weak spots were found on it. So "
    "this is a second look at a year already seen, not a fresh test.",
    "- **To keep it honest**, every design choice was made inside the training year (April 2024 to March 2025), by "
    "cross-validation or by forecasting its later months from its earlier ones. The design was written down before "
    "the test year was scored, and scored once. The parts are shown separately below, including where they cost "
    "something.",
    "- **Truth is CMORPH**, a satellite estimate. It is weak on frozen precipitation and on the heaviest rain.", "",
    "## What changed", "",
    "1. **Components see every heavy day.** Each post-processing network is trained on all rows with observed or "
    "forecast rain of 25 mm or more, the rest thinned, with weights that undo the thinning.",
    "2. **The pool is fitted on the same kind of rows.**",
    "3. **Online weights.** The components and the pool are re-weighed after every verified day, with a 48-hour "
    "delay for the truth to arrive. One learning rate, chosen on the training year.",
    "4. **Record storms.** A forecast outside anything in the training period is handed back, in part, to what the "
    "raw systems say.",
    "5. **Scenarios.** Calibrated amounts are arranged in space like the ECMWF ensemble's members, so each scenario "
    "is as varied as real weather.", "",
    "## The short version", "",
    f"- **Heavy rain (50 mm).** Brier skill improved in {c50['improved']} of {c50['cells']} place and lead-group cells, "
    f"and in {c50_short['improved']} of {c50_short['cells']} on days 1-3. Mean skill over the places on days 1-3 went "
    f"from {m50[G[0]][0]:+.4f} to {m50[G[0]][1]:+.4f}. The new Blend leads every other system at 50 mm in "
    f"{c50['new_leads']} of {c50['cells']} cells (the first led in {c50['old_leads']}).",
    f"- **Against each post-processed system at 50 mm**, {better_conf} of {n_conf} comparisons are better with 95% "
    f"confidence and {worse_conf} worse.",
    f"- **Rain or no rain (1 mm).** The new Blend leads in {c1['new_leads']} of {c1['cells']} cells (the first: "
    f"{c1['old_leads']}), and improved on the first in {c1['improved']}. On days 1-7, {better1} of {n1} comparisons "
    f"with single post-processed systems are better with 95% confidence and {worse1} worse.",
    f"- **CRPS** leads in {crps['new_leads']} of {crps['cells']} cells (was {crps['old_leads']}); **RMSE** in "
    f"{rmse['new_leads']} of {rmse['cells']} (was {rmse['old_leads']}); **ACC** in {acc['new_leads']} of {acc['cells']} "
    f"(was {acc['old_leads']}).",
]
if scenarios:
    def sc(kind, key, g):
        vals = [r["groups"][g][kind][key] for r in scenarios.values() if r["groups"].get(g, {}).get(kind)]
        return float(np.mean(vals)) if vals else float("nan")

    last = G[-1]
    lines.append(
        f"- **Structure at long range.** On days 11-14 one coupled scenario has {sc('coupled', 'sharpness', last):.2f} of "
        f"the weather's spatial variability (raw ECMWF ENS: {sc('raw', 'sharpness', last):.2f}), and the CRPS of rain "
        f"averaged over two-degree boxes is {sc('coupled', 'areal_crps', last):.2f} mm against "
        f"{sc('uncoupled', 'areal_crps', last):.2f} for uncoupled draws and {sc('raw', 'areal_crps', last):.2f} for the "
        f"raw ensemble (means over {len(scenarios)} places).")
lines += ["", "## Scoreboard", "",
          f"In how many of the {n_places} places the new Blend beat every other system, raw and post-processed, on each "
          "score. In brackets: the first Blend's count. Pluvium's own variants are not counted as rivals.", ""]
lines += board_rows()
lines += ["", "## Weak spot 1: the heavy tail", "",
          "Brier skill score for 50 mm or more: first Blend -> new Blend (best other system). Bold where the new Blend "
          "is ahead of every other system. A trailing + where it improved on the first.", ""]
lines += R.before_after(old, new, "bss", "50")
if tails:
    lines += [""] + tail_bench.markdown([tails[k] for k in ORDER if k in tails]).splitlines()
if withheld:
    lines += ["", "**Storms withheld from training** (training year only). The wettest days were taken out of training, "
              "then forecast. The calibrated forecast shrinks them; the hand-over gives some of it back and leaves "
              "ordinary days alone.", "",
              "| Place | Withheld days | Observed mean, mm | Calibrated amount, mm | Brier at 50 mm: calibrated -> with hand-over | "
              "Brier at 100 mm | Ordinary days handed over |", "|---|---|---|---|---|---|---|"]
    for place, w in withheld.items():
        label = "q0.999 w0.69"
        s, c = w["storm"], w["control"]
        lines.append(f"| {SHORT.get(place, place)} | {', '.join(w['storms'])} | {s['observed_mean_mm']:.1f} | "
                     f"{s['calibrated']['amount_mm']:.1f} | {s['calibrated']['brier']['50']:.4f} -> {s[label]['brier']['50']:.4f} | "
                     f"{s['calibrated']['brier']['100']:.4f} -> {s[label]['brier']['100']:.4f} | "
                     f"{100 * c[label]['rows_handed_over']:.2f}% |")
lines += ["", "## Weak spot 2: winter in mid-latitude climates", "",
          "Brier skill score for 1 mm or more in each place's winter: first Blend -> new Blend (best other system and "
          "which). Bold where the new Blend is ahead.", ""] + winter_rows()
if scenarios:
    lines += ["", "## Weak spot 3: structure at long range", "",
              "The map of expected amounts is smooth at long range, and should be: nobody knows where the rain band will "
              "lie in twelve days. Its sharpness in the scoreboard's sense has not changed and will not. What a user "
              "needs at long range is a set of scenarios that each look like weather. Those are scored here.", ""]
    lines += scenario_bench.markdown([scenarios[k] for k in ORDER if k in scenarios]).splitlines()
lines += ["", "## What each part added", ""] + R.ablation(new)
mel = melissa_rows()
if mel:
    lines += ["## Hurricane Melissa over Jamaica", ""] + mel + [""]
lines += ["## Every place", "",
          "Each cell: the new Blend (best other system and which). Bold where the Blend is ahead.", ""]
for place in ORDER:
    if place in new:
        lines += place_table(place)
out.write_text("\n".join(lines) + "\n")
print("wrote", out, len(lines), "lines;", n_places, "places")
