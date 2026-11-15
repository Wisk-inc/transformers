# Copyright 2026 Nathan. Apache-2.0.
"""Write docs/FRONTIER_BENCHMARK.md from the benchmark's own output, so every number in it is the data's.

    python scripts/doc_frontier.py RESULTS FIRST EXPERIMENTS docs/FRONTIER_BENCHMARK.md [README.md hub/README.md ...]

``RESULTS`` holds, per place, ``{place}.json`` (from :func:`pluvium.frontier.evaluate.evaluate`), ``{place}_tail.json``,
``{place}_scenarios.json`` and, for the US places, ``{place}_winter_DJF.json``. ``FIRST`` holds the first benchmark's
``{place}.json``. ``EXPERIMENTS`` holds the withheld-storm results ``{place}.json`` (optional). Any further pages
named get the headline results written between their ``<!-- frontier-summary:start -->`` and ``:end`` markers.
"""
import sys
from pathlib import Path

import numpy as np

from pluvium.benchmarks import fullyear as F
from pluvium.benchmarks import scenario_bench, tail_bench, winter_bench
from pluvium.frontier import report as R

results_dir, first_dir, experiments_dir, out = (Path(p) for p in sys.argv[1:5])
ORDER, SHORT = R.PLACE_ORDER, R.PLACE_NAMES
WINTER = {"alps": "DJF", "pacific_nw": "DJF", "great_plains": "DJF", "andes_south": "JJA"}
#: the places scored before the rule that decides the hand-over was final, and the places scored after
SEEN, AFTER = R.SEEN_FIRST, R.SCORED_AFTER
load = R.load_results


def names(places) -> str:
    return R.join_names(places, the=True)


new, old = load(results_dir), load(first_dir)
tails, scenarios = load(results_dir, "_tail"), load(results_dir, "_scenarios")
winters = load(results_dir, "_winter_DJF")
withheld = load(experiments_dir) if experiments_dir.exists() else {}
late_dir = results_dir.parent / "results_late140"                       # the test year with the truth 140 days late
late = load(late_dir) if late_dir.exists() else {}
late = {k: v for k, v in late.items() if k in new}
old = {k: v for k, v in old.items() if k in new}
G = list(F.LEAD_GROUPS)
seen = [p for p in ORDER if p in new and p in SEEN]
after = [p for p in ORDER if p in new and p in AFTER]


def count(key, t, higher=True, groups=None, season=None, places=None):
    use_new = {k: v for k, v in new.items() if places is None or k in places}
    return R.tally(old, use_new, key, t, higher, groups, season)


def value(place, model, key, t, g):
    return F._value(new[place]["groups"][g].get(model), key, t)


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


def kept(place: str):
    """Whether the place's staged storms kept the hand-over (``None`` if its result does not say)."""
    return R.kept(new[place])


def weak_rows() -> list:
    """Where the new Blend is still behind something, from the data."""
    lines = []
    behind = []
    for place in new:
        for g in G:
            row = new[place]["groups"][g]
            mine = F._value(row.get("stack"), "bss", "50")
            rival, who = R.best_rival(row, "bss", "50", True)
            if mine == mine and rival == rival and mine < rival:
                behind.append((rival - mine, place, g, who))
    behind.sort(reverse=True)
    total = sum(1 for p in new for g in G if value(p, "stack", "bss", "50", g) == value(p, "stack", "bss", "50", g))
    if behind:
        worst = "; ".join(f"{SHORT[p]} {g} ({gap:.3f} behind {F.LABELS.get(str(w).replace('_raw', ''), w)}"
                          f"{' raw' if str(w).endswith('_raw') else ''})" for gap, p, g, w in behind[:4])
        short = sum(1 for _, _, g, _ in behind if g in G[:2])
        lines.append(f"- At 50 mm the Blend is behind some other system in {len(behind)} of {total} place and "
                     f"lead-group cells, {short} of them on days 1-7. The largest gaps: {worst}. Beyond a week the skill "
                     "of every system at 50 mm is close to zero and one storm can decide a cell.")
    acc = []
    for place in new:
        row = new[place]["groups"][G[-1]]
        mine = F._value(row.get("stack"), "acc", None)
        rival, who = R.best_rival(row, "acc", None, True)
        if mine == mine and rival == rival and mine < rival:
            acc.append((rival - mine, place, mine, rival, who))
    acc.sort(reverse=True)
    if acc:
        text = "; ".join(f"{SHORT[p]} {m:+.2f} against {r:+.2f} ({F.LABELS.get(str(w).replace('_raw', ''), w)}"
                         f"{' raw' if str(w).endswith('_raw') else ''})" for _, p, m, r, w in acc[:4])
        lines.append(f"- On days 11-14 the anomaly correlation of the Blend's expected amount is behind another system "
                     f"in {len(acc)} of {len(new)} places: {text}. A "
                     "calibrated expected amount is smooth by construction, and a raw model's field keeps its pattern "
                     "even when the pattern is in the wrong place. A correction that traded RMSE for this score was "
                     "tested and not adopted ([WHAT_DID_NOT_WORK.md](WHAT_DID_NOT_WORK.md)).")
    far = [(SHORT[p], value(p, "stack", "bss", "1", G[-1])) for p in new]
    far = [(n, v) for n, v in far if v == v]
    if far:
        lines.append("- Days 11-14 carry little skill for anyone. Brier skill at 1 mm on those days: "
                     + ", ".join(f"{n} {v:+.3f}" for n, v in far) + ". Zero is the climatology.")
    off = [p for p in new if kept(p) is False]
    if off:
        lines.append(f"- In {names(off)} the staged storms were not forecast better with the hand-over, so it is off "
                     "there, and a storm beyond the record is forecast by the calibration alone, which shrinks it. The "
                     "table still flags such a forecast.")
    agree, n = R.audit_agreement(new, G[0])
    if n:
        lines.append(f"- The verdict on the hand-over rests on four storms per place. It had the sign of what the "
                     f"hand-over then did in the test year in {agree} of {n} places. That is partly hindsight: the rule "
                     f"was fixed after {len(seen)} of the {len(new)} places had been scored. It can be overruled "
                     "(`pluvium blend train --handover always` or `never`).")
    lines.append("- The online weights were scored with each day's rain known two days later. The satellite record is "
                 "published about four and a half months late, so a live system learns that much later"
                 + (" ([what that costs](#with-the-rain-known-140-days-late))." if late else "."))
    lines.append("- The quarterly refits used every run before each quarter. A live system can train only on runs "
                 "whose rain has been published, which are about five months old. That gap was not tested.")
    lines.append("- The truth is one satellite product. CMORPH underestimates frozen and orographic precipitation "
                 "and the heaviest rain. Winter was scored against radar and gauges for the two US places only.")
    lines.append("- The test year is the one on which the weak spots were found. A fresh year is the test that "
                 "counts, and it does not exist yet.")
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
c50_after = count("bss", "50", places=after)
c1 = count("bss", "1")
n_conf, better_conf, worse_conf = confident("50", G)
n1, better1, worse1 = confident("1", G[:2])
m50 = {g: (mean_of(old, "top", "bss", "50", g), mean_of(new, "stack", "bss", "50", g)) for g in G}
rmse = count("rmse", None, False)
acc = count("acc", None, True)
crps = count("crps", None, False)
on = [p for p in new if kept(p) is True]
off = [p for p in new if kept(p) is False]
agree_all, n_all = R.audit_agreement(new, G[0])
agree_after, n_after = R.audit_agreement(new, G[0], after)
agree_yearly, n_yearly = R.audit_agreement(new, G[0], rule="yearly")

lines = [
    "# Pluvium Blend on the frontier: the second full-year benchmark", "",
    "The first full-year benchmark ([FULL_YEAR_BENCHMARK.md](FULL_YEAR_BENCHMARK.md)) found three weak spots in Pluvium "
    "Blend: the heavy tail (50 mm), winter in mid-latitude continental climates, and structure at long range. This "
    "page scores a redesign aimed at them, on the same data, the same test year and the same rules.", "",
    "## Read this first", "",
    "The test year is the same as before (April 2025 to March 2026), and the weak spots were found on it. This is "
    "a second look at a year already seen. Every design choice was made inside the training year (April 2024 to "
    "March 2025), by cross-validation or by forecasting its later months from its earlier ones, and the design was "
    "written down before the test year was scored. The parts are shown separately below, including where they cost "
    "something.", "",
    "One thing was changed after scoring began. The design as written down handed every record storm to the raw "
    "systems. The first places scored showed that this helps in some places and does harm in others. Each place now "
    "decides from its own training year: the hand-over is used only where it improved the forecast of storms "
    "withheld from training. A first remedy, an audit on out-of-fold forecasts, was tried and dropped. Its verdict "
    f"went against the test year in {n_yearly - agree_yearly} of {n_yearly} places, and its sign changes with "
    f"which calibrated forecast it is computed on. The rule now used was fixed when {len(seen)} of the {len(new)} "
    f"places had been scored ({names(seen)}). "
    + (f"Only {names(after)} {'was' if len(after) == 1 else 'were'} scored after it. " if after else "")
    + "The design as written down, the final one and the design with no hand-over are scored everywhere and shown "
    "side by side.", "",
    "One assumption of the test was too kind. The online weights were given each day's rain two days after it fell. "
    "The satellite record is published about four and a half months late"
    + (", and [a later section](#with-the-rain-known-140-days-late) repeats the test with that delay." if late else "."),
    "",
    "The truth throughout is CMORPH, a satellite estimate, which is weak on frozen precipitation and on the heaviest "
    "rain.", "",
    "## What changed", "",
    "1. Each post-processing network is trained on all rows with observed or forecast rain of 25 mm or more. The "
    "other rows are thinned, with weights that undo the thinning.",
    "2. The pool is fitted on the same kind of rows.",
    "3. The components and the pool are re-weighed online after every verified day, with a 48-hour delay for the "
    "truth to arrive and one learning rate, chosen on the training year.",
    "4. A forecast outside anything in the training period is handed back, in part, to what the raw systems say.",
    "5. Each place decides about that hand-over. The four wettest days of the training year are withheld from "
    "training and then forecast, and the hand-over is used in a place only if it lowered the Brier score at 50 mm "
    "on those storms.",
    "6. Scenarios arrange the calibrated amounts in space like the ECMWF ensemble's members, so each scenario is as "
    "varied as real weather.", "",
    "## The short version", "",
]
summary = [
    f"- At 50 mm, Brier skill improved on the first Blend in {c50['improved']} of {c50['cells']} place and "
    f"lead-group cells, and in {c50_short['improved']} of {c50_short['cells']} on days 1-3. Mean skill over the places "
    f"on days 1-3 went from {m50[G[0]][0]:+.4f} to {m50[G[0]][1]:+.4f}. The new Blend leads every other system at "
    f"50 mm in {c50['new_leads']} of {c50['cells']} cells (the first led in {c50['old_leads']}).",
]
if after:
    summary.append(f"- In the places scored after the design was final ({names(after)}), 50 mm skill improved on the "
                   f"first Blend in {c50_after['improved']} of {c50_after['cells']} cells, and the new Blend leads every "
                   f"other system in {c50_after['new_leads']} (the first: {c50_after['old_leads']}).")
summary += [
    f"- Against each post-processed system at 50 mm, {better_conf} of {n_conf} comparisons are better with 95% "
    f"confidence and {worse_conf} are worse.",
    f"- At 1 mm, which is rain or no rain, the new Blend leads in {c1['new_leads']} of {c1['cells']} cells (the first: "
    f"{c1['old_leads']}), and improved on the first in {c1['improved']}. On days 1-7, {better1} of {n1} comparisons "
    f"with single post-processed systems are better with 95% confidence and {worse1} are worse.",
    f"- The CRPS leads in {crps['new_leads']} of {crps['cells']} cells (was {crps['old_leads']}), the RMSE in "
    f"{rmse['new_leads']} of {rmse['cells']} (was {rmse['old_leads']}) and the anomaly correlation in "
    f"{acc['new_leads']} of {acc['cells']} (was {acc['old_leads']}).",
]
late1, late50 = count("bss", "1", groups=[G[-1]]), count("bss", "50", groups=[G[-1]])
if late1["new_leads"] < late1["old_leads"] or late50["new_leads"] < late50["old_leads"]:
    summary.append(
        f"- Days 11-14 got slightly worse. There the new Blend leads every other system at 1 mm in "
        f"{late1['new_leads']} of {late1['cells']} places (the first: {late1['old_leads']}) and at 50 mm in "
        f"{late50['new_leads']} (the first: {late50['old_leads']}). Mean Brier skill at 1 mm on those days went from "
        f"{mean_of(old, 'top', 'bss', '1', G[-1]):+.4f} to {mean_of(new, 'stack', 'bss', '1', G[-1]):+.4f}. The pool "
        f"alone scores {mean_of(new, 'pool', 'bss', '1', G[-1]):+.4f}: the online weights for that lead group learn "
        "from few forecasts that verify late, and cost a little.")
if late:
    g0 = G[0]
    summary.append(
        "- The online weights were scored with each day's rain known two days later, and the satellite record is "
        f"published about 140 days late. With that delay, mean Brier skill on days 1-3 over {len(late)} places is "
        f"{R.late_mean(new, late, '1', g0, 'late'):+.4f} at 1 mm and {R.late_mean(new, late, '50', g0, 'late'):+.4f} at "
        f"50 mm. The figures above correspond to {R.late_mean(new, late, '1', g0, 'bench'):+.4f} and "
        f"{R.late_mean(new, late, '50', g0, 'bench'):+.4f}, and with no online weights the Blend scores "
        f"{R.late_mean(new, late, '1', g0, 'none'):+.4f} and {R.late_mean(new, late, '50', g0, 'none'):+.4f}.")
if n_all:
    misses = R.audit_misses(new, G[0])
    summary.append("- Each place's staged storms "
                   + (f"kept the record-storm hand-over in {names(on)}" if on else "kept the record-storm hand-over nowhere")
                   + (f" and switched it off in {names(off)}" if off else "")
                   + f". That verdict had the sign of what the hand-over then did in the test year in {agree_all} of "
                   f"{n_all} places"
                   + (f" ({agree_after} of the {n_after} scored after the rule was final)" if n_after else "")
                   + f". The out-of-fold audit tried first would have been right in {agree_yearly} of {n_yearly}."
                   + (" " + misses if misses else ""))
if scenarios:
    def sc(kind, key, g):
        vals = [r["groups"][g][kind][key] for r in scenarios.values() if r["groups"].get(g, {}).get(kind)]
        return float(np.mean(vals)) if vals else float("nan")

    last = G[-1]
    summary.append(
        f"- On days 11-14 one coupled scenario has {sc('coupled', 'sharpness', last):.2f} of "
        f"the weather's spatial variability (raw ECMWF ENS: {sc('raw', 'sharpness', last):.2f}), and the CRPS of rain "
        f"averaged over two-degree boxes is {sc('coupled', 'areal_crps', last):.2f} mm against "
        f"{sc('uncoupled', 'areal_crps', last):.2f} for uncoupled draws and {sc('raw', 'areal_crps', last):.2f} for the "
        f"raw ensemble (means over {len(scenarios)} places).")
lines += summary
lines.append("- The weak spots that remain are listed [near the end](#what-is-still-weak).")
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
lines += ["", "### The hand-over: staged, checked, and scored three ways", ""]
if withheld:
    lines += ["Storms withheld from training (training year only). The wettest days were taken out of training, "
              "then forecast. The calibrated forecast shrinks them. The hand-over gives some of that back where the raw "
              "systems are the better guide, and does harm where they are not. Ordinary days are left alone. A place "
              "uses the hand-over only where the Brier score in the fifth column fell (bold).", "",
              "| Place | Withheld days | Observed mean, mm | Calibrated amount, mm | Brier at 50 mm: calibrated -> with hand-over | "
              "Raw systems alone | Ordinary days handed over |", "|---|---|---|---|---|---|---|"]
    for place, w in withheld.items():
        label = "q0.999 w0.69"
        s, c = w["storm"], w["control"]
        better = s[label]["brier"]["50"] < s["calibrated"]["brier"]["50"]
        lines.append(f"| {SHORT.get(place, place)} | {', '.join(w['storms'])} | {s['observed_mean_mm']:.1f} | "
                     f"{s['calibrated']['amount_mm']:.1f} | {s['calibrated']['brier']['50']:.4f} -> "
                     f"{'**' if better else ''}{s[label]['brier']['50']:.4f}{'**' if better else ''} | "
                     f"{s['raw']['brier']['50']:.4f} | {100 * c[label]['rows_handed_over']:.2f}% |")
    lines.append("")
rules = R.rules_table(new, G[0], SHORT)
if rules:
    lines += ["Two rules from training data, and what happened. The staged storms are the rule that decides. The "
              "yearly check was tried first as the switch: the hand-over is applied to all the training period's "
              "out-of-fold forecasts and the Brier score at 50 mm compared. It touches one row in a thousand, so its "
              "effect on a year's score is a few tenths of a percent, and the sign of so small a number tells us "
              "nothing. The last column is the test year: Brier skill at 50 mm on days 1-3 with the hand-over "
              "always on, minus without it. A verdict is bold where it had the sign of the test year.", ""] + rules + [""]
    lines += [f"The staged storms had the sign of the test year in {agree_all} of {n_all} places, the yearly check in "
              f"{agree_yearly} of {n_yearly}. The staged rule was the first one written down, was replaced by the yearly "
              f"check, and was restored when {len(seen)} of the {len(new)} places had been scored, so its count is partly "
              "hindsight.", ""]
    for g in G[:2]:
        lines += [f"Brier skill at 50 mm in the test year, {g}, three ways. Bold is the best of the three.", ""]
        lines += R.handover_three_ways(new, g, SHORT) + [""]
lines += ["## Weak spot 2: winter in mid-latitude climates", "",
          "Brier skill score for 1 mm or more in each place's winter: first Blend -> new Blend (best other system and "
          "which). Bold where the new Blend is ahead.", ""] + winter_rows()
if winters:
    lines += ["", "### Winter against a second truth", "",
              "CMORPH is weak on frozen precipitation, so a winter loss could be the forecast's or the truth's. The two "
              "US places were scored again for December to February against MRMS, the US radar and gauge analysis, on "
              "the same rows. Everything was trained on CMORPH, so every system carries the same handicap against "
              "MRMS, and the question is whether the order between the systems changes.", ""]
    labels = {k: v + ", post-processed" for k, v in F.LABELS.items()}
    for place in ORDER:
        if place in winters:
            lines += winter_bench.markdown(winters[place], labels).splitlines() + [""]
if scenarios:
    lines += ["", "## Weak spot 3: structure at long range", "",
              "The map of expected amounts is smooth at long range, and should be: nobody knows where the rain band will "
              "lie in twelve days. Its sharpness in the scoreboard's sense has not changed and will not. At that range "
              "a user is better served by a set of scenarios that each look like weather, and those are scored here.", ""]
    lines += scenario_bench.markdown([scenarios[k] for k in ORDER if k in scenarios]).splitlines()
lines += ["", "## What each part added", ""] + R.ablation(new)
mel = melissa_rows()
if mel:
    lines += ["## Hurricane Melissa over Jamaica", ""] + mel + [""]
    lines += R.rarity_lines(experiments_dir / "jamaica_returns.json")
if late:
    cells, lower, below = R.late_counts(new, late)
    where = f"all {n_places} places" if len(late) == n_places else f"{len(late)} of the {n_places} places ({names(late)})"
    lines += ["## With the rain known 140 days late", "",
              "The online weights learn from each day's rain, and in the test they were given it two days after it "
              "fell. The record used as truth, NOAA's CMORPH climate data record, is not available that soon. On 15 "
              "November 2026 its newest complete month was May 2026: the files for May had been written on 9 October "
              "and those for June were still arriving. A live system therefore learns from each day about 140 days "
              f"late. The test year was scored again with that delay and nothing else changed, in {where}. The delay "
              "was applied to the warm-up over the training period as well, which is a little harsher than live use, "
              "where all of the training period's rain is known by the time a blend is trained. The table gives the "
              "mean Brier skill over those places.", ""] + R.late_rows(new, late) + [
        "", f"With the delay, skill is lower than in the benchmark in {lower} of the {cells} cells (place, lead group, "
        f"1 or 50 mm) and lower than with no online weights at all in {below}. The scoreboard hardly moves: the Blend "
        f"is ahead of every other system at 1 mm in {R.late_leads(new, late, '1')[2]} of "
        f"{R.late_leads(new, late, '1')[0]} cells (with the two-day delay: {R.late_leads(new, late, '1')[1]}) and at "
        f"50 mm in {R.late_leads(new, late, '50')[2]} (with the two-day delay: {R.late_leads(new, late, '50')[1]}). "
        "What the delay takes away is most of the small gain the online weights gave at 50 mm on days 1-3. The other "
        "tables on this page keep the two-day delay, because that is the test as it was declared.", ""]
lines += ["## What is still weak", ""] + weak_rows() + [""]
lines += ["## Every place", "",
          "Each cell: the new Blend (best other system and which). Bold where the Blend is ahead.", ""]
for place in ORDER:
    if place in new:
        lines += place_table(place)
out.write_text("\n".join(lines) + "\n")
print("wrote", out, len(lines), "lines;", n_places, "places")

# the same headline results, written into any other page that carries the markers (the README, the model card)
START, END = "<!-- frontier-summary:start -->", "<!-- frontier-summary:end -->"
block = [f"On the test year, in {n_places} places (the same year the first Blend was scored on):", ""] + summary
for page in (Path(p) for p in sys.argv[5:]):
    text = page.read_text()
    if START not in text or END not in text:
        raise SystemExit(f"{page} has no {START} ... {END} markers")
    head, rest = text.split(START, 1)
    page.write_text(head + START + "\n" + "\n".join(block) + "\n" + END + rest.split(END, 1)[1])
    print("updated the summary in", page)
