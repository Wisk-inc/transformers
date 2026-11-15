# Copyright 2026 Nathan. Apache-2.0.
"""Write the three rain papers (docs/papers/01 to 03) from the benchmark's own output, so their numbers are the data's.

    python scripts/doc_papers.py RESULTS FIRST EXPERIMENTS docs/papers

The arguments are those of ``scripts/doc_frontier.py``. The text is fixed; every number and table is computed.
If ``RESULTS`` has a sibling folder ``results_late140`` (the test year scored again with the truth 140 days late),
paper 1 reports it.

The wording follows what ``scripts/check_style.py`` checks: plain statements, no bold phrase opening a list item,
every reference cited where it is used.
"""
import sys
from pathlib import Path

import numpy as np

from pluvium.benchmarks import fullyear as F
from pluvium.benchmarks import scenario_bench, tail_bench, winter_bench
from pluvium.frontier import report as R

results_dir, first_dir, experiments_dir, out_dir = (Path(p) for p in sys.argv[1:5])
ORDER, SHORT = R.PLACE_ORDER, R.PLACE_NAMES
new, old = R.load_results(results_dir), R.load_results(first_dir)


def names(places) -> str:
    return R.join_names(places, the=True)


tails, scenarios = R.load_results(results_dir, "_tail"), R.load_results(results_dir, "_scenarios")
winters = R.load_results(results_dir, "_winter_DJF")
withheld = R.load_results(experiments_dir) if experiments_dir.exists() else {}
late_dir = results_dir.parent / "results_late140"
late = R.load_results(late_dir) if late_dir.exists() else {}
late = {k: v for k, v in late.items() if k in new}
old = {k: v for k, v in old.items() if k in new}
G = list(F.LEAD_GROUPS)
seen = [p for p in ORDER if p in new and p in R.SEEN_FIRST]
after = [p for p in ORDER if p in new and p in R.SCORED_AFTER]
LABEL = "q0.999 w0.69"                                                    # the hand-over's declared setting
BYLINE = "Nathan, November 2026."
CAUTION = ("The rain test year (April 2025 to March 2026) is the year on which the weak spots were found. Every design "
           "choice was made inside the training year and written down before the test year was scored, but a second "
           "look at a year is weaker evidence than a fresh year.")
SAME_YEAR = "The test year is the one on which the weak spots were found (see the summary)."


def value(place, model, key, t, g):
    return F._value(new[place]["groups"][g].get(model), key, t)


def mean_over(model, key, t, g, results=None, places=None):
    vals = []
    for place, res in (results or new).items():
        if places is not None and place not in places:
            continue
        name = F._top(res) if model == "top" else model
        v = F._value(res["groups"][g].get(name), key, t)
        if v == v:
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def count(key, t, higher=True, groups=None, places=None):
    use = {k: v for k, v in new.items() if places is None or k in places}
    return R.tally(old, use, key, t, higher, groups)


def confident(threshold: str, groups) -> tuple:
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


def late_mean(t: str, g: str, which: str) -> float:
    return R.late_mean(new, late, t, g, which)


def write(name: str, lines: list) -> None:
    path = out_dir / name
    path.write_text("\n".join(lines) + "\n")
    print("wrote", path, len(lines), "lines")


n_places = len(new)
c50, c1 = count("bss", "50"), count("bss", "1")
c50_short = count("bss", "50", groups=[G[0]])
c50_after = count("bss", "50", places=after)
crps, rmse, acc = count("crps", None, False), count("rmse", None, False), count("acc", None, True)
n50, better50, worse50 = confident("50", G)
n1, better1, worse1 = confident("1", G[:2])
m50_old, m50_new = mean_over("top", "bss", "50", G[0], old), mean_over("stack", "bss", "50", G[0])
m1_old, m1_new = mean_over("top", "bss", "1", G[0], old), mean_over("stack", "bss", "1", G[0])
agree_all, n_all = R.audit_agreement(new, G[0])
agree_after, n_after = R.audit_agreement(new, G[0], after)
agree_yearly, n_yearly = R.audit_agreement(new, G[0], rule="yearly")
on = [p for p in new if R.kept(new[p]) is True]
off = [p for p in new if R.kept(new[p]) is False]
three = {m: mean_over(m, "bss", "50", G[0]) for m in ("pool_online", "declared", "stack")}
late1 = count("bss", "1", groups=[G[-1]])
LATE_WHERE = (f"all {n_places} places" if len(late) == n_places else f"{len(late)} of the {n_places} places ({names(late)})")

# ============================================================================================ paper 1 ==
p1 = [
    "# Tail-aware pooling with online weights: a multi-model rain forecast, second generation", "",
    f"Pluvium technical paper 1. {BYLINE}", "",
    "## Summary", "",
    "Pluvium Blend turns the forecasts of six operational systems (ECMWF ENS, HRES and AIFS; NOAA GEFS, GFS and "
    "GraphCast) into calibrated chances of daily rain for days 1 to 14. A benchmark over a full year found it weakest "
    "at 50 mm, in continental winters, and in the structure of its long-range maps. This paper describes the changes "
    "aimed at the first two of those and reports what they did on the same test year.", "",
    f"The components are now trained on every heavy-rain row, with weights that keep the chances unbiased, and the "
    f"pool is fitted on the same rows. The components and the pool are also re-weighed online after every verified "
    f"day. In {n_places} places, Brier skill at 50 mm improved on the first Blend in {c50['improved']} of "
    f"{c50['cells']} place and lead-group cells, and the Blend leads every other system, raw or post-processed, in "
    f"{c50['new_leads']} of them (the first led in {c50['old_leads']}). Against single post-processed systems at 50 mm, "
    f"{better50} of {n50} comparisons are better with 95% confidence and {worse50} are worse. At 1 mm it leads in "
    f"{c1['new_leads']} of {c1['cells']} cells (the first: {c1['old_leads']}).", "",
    f"The gains are small. Mean Brier skill at 50 mm on days 1-3 went from {m50_old:+.4f} to {m50_new:+.4f}, and at 1 mm "
    f"from {m1_old:+.4f} to {m1_new:+.4f}. Part of the gain at 50 mm comes from a hand-over for record storms, which is "
    f"the subject of paper 2; without it the changes described here score {three['pool_online']:+.4f}. On days 11-14 "
    f"the new Blend leads in fewer places than the first did (at 1 mm in {late1['new_leads']} of {late1['cells']}, the "
    f"first in {late1['old_leads']}).", "",
]
if late:
    p1 += [
        "One assumption of the test was too kind. The online weights were given each day's rain two days after it "
        "fell. The satellite record used as truth is published about four and a half months late, which we found out "
        f"only when running the system live in November 2026. With the rain held back 140 days (section 4.7, {LATE_WHERE}), "
        f"mean Brier skill on days 1-3 is {late_mean('1', G[0], 'late'):+.4f} at 1 mm and {late_mean('50', G[0], 'late'):+.4f} "
        f"at 50 mm. The benchmark's figures for the same places are {late_mean('1', G[0], 'bench'):+.4f} and "
        f"{late_mean('50', G[0], 'bench'):+.4f}, and with no online weights at all {late_mean('1', G[0], 'none'):+.4f} and "
        f"{late_mean('50', G[0], 'none'):+.4f}.", "",
    ]
p1 += [
    CAUTION, "",
    "## 1. What the first benchmark found", "",
    "The first Blend post-processed each system with a small network, fitted one more network over all six, and "
    "pooled the seven with weights per band of thresholds and lead. Scored over a year in six regions and Jamaica "
    "([FULL_YEAR_BENCHMARK.md](../FULL_YEAR_BENCHMARK.md)), it led every other system on most scores at most leads. "
    f"It was weakest at 50 mm, where it led in {c50['old_leads']} of {c50['cells']} cells, and in the Great Plains "
    "winter, where post-processed ECMWF ENS alone was better.", "",
    "We traced the weakness at 50 mm to two causes. A uniform sample of 400,000 training rows holds only a few "
    "thousand days with 50 mm, so the rarest thresholds were learned from very little. And a pool fitted once on last "
    "year's weather keeps last year's opinion of each system all through the next year.", "",
    "## 2. Method", "",
    "### 2.1 Components that see every heavy day", "",
    "A row is one place on one day at one lead. Rows are drawn unevenly on purpose. Every row where the observed "
    "rain, or any system's forecast, reaches 25 mm is kept (up to half the budget), and the rest are thinned to fill "
    "the budget. A row kept with probability `pi` carries the weight `1 / pi`, so the weighted loss", "",
    "    L = sum_i  (1 / pi_i) * loss_i", "",
    "is an unbiased estimate of the loss over all rows (Horvitz and Thompson, 1952). The chances stay calibrated "
    "and the tail is learned from several times more events. Rows that were forecast heavy and stayed dry are kept "
    "as well, because the networks have to learn false alarms at 50 mm as much as hits.", "",
    "### 2.2 The pool", "",
    "Each component `c` gives log-odds `z_ck` for threshold `k`. The pool is a weighted logarithmic pool with an "
    "offset (Ranjan and Gneiting, 2010):", "",
    "    logit p_k = W * ( sum_c w_c * present_c * z_ck ) / ( sum_c w_c * present_c ) + b_k + e_k * lead",
    "    w_c = softplus(a_c + d_c * lead),    W = sum_c w_c", "",
    "The weights `w` are shared within three bands of thresholds (below 5 mm, 5 to 25 mm, 35 mm and above) and "
    "change with lead. A missing component drops out of the average and the rest carry on. The pool is fitted by "
    "cross-entropy on out-of-fold rows, so that no component is judged on days it was trained on, with the row "
    "weights of section 2.1.", "",
    "### 2.3 Online weights", "",
    "The experts are the seven components and the pool. Each lead group (days 1-3, 4-7, 8-10, 11-14) and band of "
    "thresholds keeps its own weights. After every verified day, with `L_e` expert `e`'s mean Brier score on that "
    "day's rows, `L_mix` the mixture's and `S` a running mean of `L_mix`:", "",
    "    r_e = clip( (L_mix - L_e) / S, -2, 2 )",
    "    log w_e  <-  log w_e + eta * r_e - (eta * r_e)^2 / 2",
    "    w  <-  (1 - alpha) * w / sum(w) + alpha * prior", "",
    "The second-order term is that of Bernstein online aggregation (Wintenberger, 2017), which Berrisch and Ziel "
    "(2021) applied to probabilistic forecasts. Dividing by `S` lets one learning rate serve 1 mm and 50 mm. The last "
    "line is a fixed share (Herbster and Warmuth, 1998): a little weight returns to the prior at every step, so an "
    "expert that was written off can come back. The prior gives half the weight to the pool. The settings are "
    "`eta = 1` and `alpha = 0.02`.", "",
    "The weights do not start cold. Before the test they are run through the training period's out-of-fold "
    "forecasts in time order. In the benchmark a day's rows are then learned from 48 hours after the day ends. For "
    "the truth used here that delay is much too short"
    + (" (section 4.7)." if late else "; the record is published about four and a half months late."), "",
    "### 2.4 Record storms", "",
    "A forecast beyond anything in the training period is handed back, in part, to what the raw systems say, in "
    "the places where that improved the forecast of storms withheld from training. That is the subject of "
    "[paper 2](02_record_storms.md).", "",
    "### 2.5 How the design was chosen", "",
    "Every choice was made inside the training year, by cross-validation or by forecasting its later months from "
    "its earlier ones, in four regions. Most of what we tried was dropped. The numbers are in "
    "[WHAT_DID_NOT_WORK.md](../WHAT_DID_NOT_WORK.md).", "",
    "| Tried | Outcome |", "|---|---|",
    "| Weights for each of the 16 thresholds, tied by a smoothness penalty | No gain over three bands |",
    "| Weights that follow the season | Worse: one year cannot tell a season from that year's weather |",
    "| Weights that follow the regime (how wet and how split the components are) | Better at 50 mm, worse at 1 mm |",
    "| A learning rate of 2 in place of 1 | No better |",
    "| A separate forecast of the expected amount | Gains under 1% in RMSE |", "",
    "The design was then written down with its settings, and the test year was scored.", "",
    "## 3. Test", "",
    f"There are {n_places} places: " + ", ".join(F.REGION_TITLES.get(p, p) for p in new) + ". Training used runs from "
    "April 2024 to March 2025. The test is April 2025 to March 2026, with everything refitted every quarter on all "
    "data before it. Runs every second day are scored for days 1 to 3 and every fourth day to day 14. The truth is "
    "the CMORPH daily satellite estimate (Joyce et al., 2004) on land. Skill is against the training year's "
    "climatology by place and month. Confidence intervals come from a block bootstrap over runs. In the scoreboard a "
    "system counts as beaten only if the Blend is ahead of it both raw and post-processed; Pluvium's own variants "
    "are not counted as rivals.", "",
    "## 4. Results", "",
    "### 4.1 Scoreboard", "",
    f"In how many of the {n_places} places the new Blend beat every other system. In brackets: the first Blend.", "",
] + board_rows() + [
    "", "### 4.2 Heavy rain", "",
    "Brier skill score for 50 mm or more: first Blend -> new Blend (best other system). Bold where the new Blend is "
    "ahead of every other system. A trailing + where it improved on the first.", "",
] + R.before_after(old, new, "bss", "50") + [""]
if after:
    p1 += [f"In the places scored after the design was final ({names(after)}), skill at 50 mm improved on the first "
           f"Blend in {c50_after['improved']} of {c50_after['cells']} cells and the new Blend leads every other system in "
           f"{c50_after['new_leads']} (the first: {c50_after['old_leads']}).", ""]
p1 += ["### 4.3 What each part added", "",
       "The mean over the places of each score, for the pool alone and with each further part.", ""]
p1 += R.ablation(new, board=(("Brier skill, 1 mm", "bss", "1", True), ("Brier skill, 50 mm", "bss", "50", True),
                             ("CRPS", "crps", None, False)))
show = next((p for p in ("great_plains", "alps", "caribbean") if p in new and R.online_weights(new[p])), None)
if show:
    p1 += ["### 4.4 Where the weight went", "",
           f"The online weights for light rain on days 1-3 in the {SHORT[show]}, at the end of each quarter of the test "
           "year. The prior gives the pool 0.50 and each component 0.07.", ""] + R.online_weights(new[show]) + [""]
p1 += ["### 4.5 Other scores", "",
       f"CRPS leads in {crps['new_leads']} of {crps['cells']} cells (the first: {crps['old_leads']}), RMSE in "
       f"{rmse['new_leads']} of {rmse['cells']} ({rmse['old_leads']}) and the anomaly correlation in {acc['new_leads']} of "
       f"{acc['cells']} ({acc['old_leads']}). At 1 mm on days 1-7, {better1} of {n1} comparisons with single "
       f"post-processed systems are better with 95% confidence and {worse1} are worse.", ""]
if winters:
    p1 += ["### 4.6 Winter against a second truth", "",
           "CMORPH is weak on frozen precipitation, so a winter loss could be the forecast's or the truth's. The US "
           "places were scored again for December to February against MRMS, the US radar and gauge analysis (Zhang et "
           "al., 2016), on the same rows. Everything was trained on CMORPH, so every system carries the same handicap "
           "against MRMS, and the question is whether the order between the systems changes.", ""]
    labels = {k: v + ", post-processed" for k, v in F.LABELS.items()}
    for place in ORDER:
        if place in winters:
            p1 += winter_bench.markdown(winters[place], labels).splitlines() + [""]
if late:
    cells, lower, below = R.late_counts(new, late)
    p1 += ["### 4.7 With the rain known 140 days late", "",
           "The record we use as truth, NOAA's CMORPH climate data record, is not available two days after the fact. "
           "On 15 November 2026 its newest complete month was May 2026: the files for May had been written on 9 October "
           "and those for June were still arriving. A live system therefore learns from each day about 140 days late. "
           f"We scored the test year again with that delay and nothing else changed, in {LATE_WHERE}. The table gives "
           "the mean Brier skill over those places.", ""] + R.late_rows(new, late) + [
        "", f"With the delay, skill is lower than in the benchmark in {lower} of the {cells} cells (place, lead group, 1 or "
        f"50 mm) and lower than with no online weights at all in {below}. A second difference between the benchmark and "
        "live use is not measured here: the benchmark refitted every quarter on all runs before it, and a live system "
        "can train only on runs whose rain has been published, which are about five months old.", ""]
p1 += ["## 5. Limits", "",
       "- " + SAME_YEAR,
       "- The gains are small in absolute terms, and at long leads every system is close to the climatology.",
       "- The online weights were tested with rain known after two days. The record arrives about 140 days late"
       + (" (section 4.7)." if late else ".") + " The quarterly refits also used runs more recent than a live system "
       "could train on.",
       "- There is one truth, a satellite estimate, which underestimates orographic and frozen precipitation and the "
       "heaviest rain.",
       "- The scores are for daily totals on a quarter-degree grid (8 km for Jamaica). Nothing here speaks to hourly "
       "rain.",
       "- The anomaly correlation of the expected amount at long range is still behind raw models in mid-latitudes.",
       "- The full list is in [FRONTIER_BENCHMARK.md](../FRONTIER_BENCHMARK.md#what-is-still-weak).", "",
       "## 6. Reproducing", "",
       "```bash", "pip install pluvium",
       "python -m pluvium.benchmarks.frontier_bench --archive ARCHIVE --truth TRUTH --out OUT",
       "python scripts/doc_frontier.py OUT/results FIRST EXPERIMENTS docs/FRONTIER_BENCHMARK.md", "```", "",
       "`pluvium.sources.archive.stage_archive` downloads the runs and `pluvium.sources.cmorph_daily.stage_daily` the "
       "truth. The delay of the truth is the `latency` argument of `pluvium.frontier.evaluate.evaluate`, in hours. To "
       "train the same design for a country: `pluvium blend train` ([TRAINING.md](../TRAINING.md)).", "",
       "## References", "",
       "- Berrisch, J., and Ziel, F. (2021). CRPS learning. arXiv:2102.00968.",
       "- Herbster, M., and Warmuth, M. K. (1998). Tracking the best expert. *Machine Learning*, 32, 151-178. "
       "doi:10.1023/A:1007424614876",
       "- Horvitz, D. G., and Thompson, D. J. (1952). A generalization of sampling without replacement from a finite "
       "universe. *Journal of the American Statistical Association*, 47, 663-685. doi:10.1080/01621459.1952.10483446",
       "- Joyce, R. J., Janowiak, J. E., Arkin, P. A., and Xie, P. (2004). CMORPH: a method that produces global "
       "precipitation estimates from passive microwave and infrared data at high spatial and temporal resolution. "
       "*Journal of Hydrometeorology*, 5, 487-503. doi:10.1175/1525-7541(2004)005<0487:CAMTPG>2.0.CO;2",
       "- Ranjan, R., and Gneiting, T. (2010). Combining probability forecasts. *Journal of the Royal Statistical "
       "Society B*, 72, 71-91. doi:10.1111/j.1467-9868.2009.00726.x",
       "- Wintenberger, O. (2017). Optimal learning with Bernstein online aggregation. *Machine Learning*, 106, "
       "119-141. doi:10.1007/s10994-016-5592-6",
       "- Zhang, J., Howard, K., Langston, C., Kaney, B., and co-authors (2016). Multi-Radar Multi-Sensor (MRMS) "
       "quantitative precipitation estimation: initial operating capabilities. *Bulletin of the American "
       "Meteorological Society*, 97, 621-638. doi:10.1175/BAMS-D-14-00174.1"]
write("01_blend_frontier.md", p1)

# ============================================================================================ paper 2 ==
helped = [p for p, w in withheld.items() if w["storm"][LABEL]["brier"]["50"] < w["storm"]["calibrated"]["brier"]["50"]]
hurt = [p for p in withheld if p not in helped]
shrink = [(w["storm"]["calibrated"]["amount_mm"], w["storm"]["observed_mean_mm"]) for w in withheld.values()]
p2 = [
    "# Forecasting beyond the training record: a hand-over to the raw systems, decided by staged storms", "",
    f"Pluvium technical paper 2. {BYLINE}", "",
    "## Summary", "",
    "A calibrated forecast is a map learned from the forecasts of its training period. On a storm larger than "
    "anything in that period the map extrapolates the lesson of ordinary days, which is that big forecasts usually "
    "verify smaller. Two days before Hurricane Melissa crossed Jamaica, every raw system gave an 83% to 100% chance "
    "of 50 mm and the first Pluvium Blend gave 66%.", "",
    "We hand such a forecast back, in part, to the plain agreement of the raw systems. The share handed over is "
    "exactly zero inside the range of the training forecasts and rises beyond it, and it is scaled by how many "
    "systems agree. To test the idea without touching the test year, we stage record storms inside the training "
    "year: the wettest days are removed from training and then forecast.",
]
if withheld:
    p2 += [
        "",
        f"With the four wettest days withheld, the calibrated forecast gave on average "
        f"{np.mean([a for a, _ in shrink]):.0f} mm where {np.mean([b for _, b in shrink]):.0f} mm fell. The hand-over "
        f"improved the Brier score at 50 mm on those days in {len(helped)} of {len(withheld)} places ({names(helped)})"
        + (f". It made the score worse in {names(hurt)}." if hurt else "."),
    ]
p2 += [
    "",
    "That split is the main finding. The hand-over helps where the raw systems are the better guide to a storm the "
    "calibration has never seen, and it does harm where they overcall heavy rain against the truth. So each place "
    "decides from its own staged storms: the hand-over is used only where it lowered the Brier score at 50 mm on "
    "them.",
    "",
]
if n_all:
    p2 += [f"In the test year the hand-over was on in {names(on)}" + (f" and off in {names(off)}" if off else "")
           + f". Mean Brier skill at 50 mm on days 1-3 over the {n_places} places is {three['pool_online']:+.4f} with no "
           f"hand-over, {three['declared']:+.4f} with it always on and {three['stack']:+.4f} as the places decided. Each "
           f"place's verdict had the sign of what the hand-over then did in the test year in {agree_all} of {n_all} places"
           + (f", and in {agree_after} of the {n_after} scored after the rule was final." if n_after else ".")]
    if R.audit_misses(new, G[0]):
        p2 += ["", R.audit_misses(new, G[0])]
p2 += [
    "",
    "The paper also reports a rule that failed. Before settling on the staged storms we switched the hand-over by "
    "an audit of the training period's out-of-fold forecasts, a far larger sample. It would have had the right sign "
    f"in {agree_yearly} of {n_yearly} places. The hand-over touches one forecast in a thousand, so its effect on a "
    "year's score is a few tenths of a percent, and the sign of so small a number tells us nothing.", "",
    f"The staged-storm rule was the first one written down, was replaced by the audit, and was restored when "
    f"{len(seen)} of the {n_places} places ({names(seen)}) had been scored. For those places the rule was settled with "
    f"their test results known. {CAUTION}", "",
    "## 1. The problem", "",
    "Calibration shrinks forecasts towards what usually happens, and it should: a raw 80 mm forecast usually "
    "verifies well below 80 mm. A record storm is the case in which that lesson fails, and it is also the case in "
    "which a low forecast costs the most. Fitting more carefully does not help, because the training period does not "
    "hold the event.", "",
    "The raw systems are not bound in the same way. They are physical models, or machine-learning models trained on "
    "about forty years of reanalysis, and they can produce an event unlike anything in a one-year training set.", "",
    "## 2. Method", "",
    "### 2.1 The raw consensus", "",
    "Every system has one vote, however many members it runs. A system's chance of a threshold is the share of its "
    "members that reach it, 0 or 1 for a single run, and the consensus is the mean over the systems present. On "
    "ordinary days the consensus is poorly calibrated, which is why it is used only for the hand-over.", "",
    "### 2.2 Support: has the calibration seen a forecast like this one?", "",
    "    score = ( log(1 + amount) - log(1 + edge) ) / width", "",
    "`amount` is the raw consensus mean. `edge` is the 99.9th percentile of that quantity over the training rows, "
    "never below 20 mm, so one training row in a thousand lies beyond it. `width` is log 2: the score reaches one at "
    "about twice the edge.", "",
    "### 2.3 The hand-over", "",
    "    g = min( clip(score, 0, 1) * clip(2 * agree, 0, 1), 0.95 )",
    "    final = (1 - g) * calibrated + g * raw consensus", "",
    "`agree` is the share of systems that call for 25 mm or more. One system drawing a bullseye where the others "
    "are dry does not take the forecast away from the calibration. Inside the training range `g` is exactly zero, so "
    "ordinary days, and every score computed on them, are untouched by construction. An earlier version used a "
    "smooth gate, which is never exactly zero and touched 16% of ordinary rows.", "",
    "### 2.4 Staging record storms", "",
    "The remedy cannot be tuned on the test year, and ordinary days say nothing about it. So the situation is staged "
    "inside the training year. The four wettest days, at least five days apart, are removed from training along with "
    "the two days either side. Twelve ordinary days are removed as a control. A calibrated forecast is trained on "
    "the rest, and the withheld days are forecast with and without the hand-over.", "",
    "### 2.5 The verdict", "",
    "A place uses the hand-over if it lowered the Brier score at 50 mm on its withheld storms. The control days take "
    "no part in the verdict; they are there to show that ordinary days are left alone. With no storm to judge by, the "
    "hand-over is kept, because a shrunk record storm is the costlier error. `pluvium blend train` stages the storms, "
    "prints the verdict and saves it with the model. The owner can overrule it (`--handover always` or `never`).", "",
    "### 2.6 The yearly check, and why it does not decide", "",
    "The staged storms are four days. A larger sample is at hand: every training row has an out-of-fold forecast, "
    "made by networks that never saw that row's days. The check applies the hand-over to all of them and compares", "",
    "    Brier(with hand-over) against Brier(without),   at 50 mm, over all training rows", "",
    "with the rows' sampling weights undone. It costs nothing, and we first used it as the switch. It turned out to "
    "be useless for that. The hand-over touches about one row in a thousand, most of them heavy days of a kind the "
    "networks have seen. Over a year its effect on the score is a few tenths of a percent in either direction, and in "
    "three of seven places the sign changes when the check is run on the one-network component's forecast in place "
    "of the pool's ([WHAT_DID_NOT_WORK.md](../WHAT_DID_NOT_WORK.md)). The check is still run at every training and "
    "saved with the model, because it shows what the hand-over costs in an ordinary year.", "",
    "### 2.7 What the user is told", "",
    "The forecast table carries three more columns. The first is an alert tier for 50 mm: watch at a 10% chance, "
    "warning at 30%, extreme at 60%. The second is a flag for the days on which the raw systems call for more than "
    "anything in the training period; on those days the amounts should be read as a lower bound. The third says how "
    "rare the expected amount is at that place. It comes from a generalised Pareto fit to the area's daily record "
    "(Pickands, 1975; Hosking and Wallis, 1987), with one shape for the area and a threshold and scale per place, "
    "which is the index-flood idea of Dalrymple (1960). Beyond four times the length of the record the answer is "
    "only \"rarer than once in that many years\".", "",
    "## 3. Results", "",
]
if withheld:
    p2 += ["### 3.1 Staged storms (training year)", "",
           "| Place | Withheld days | Observed mean, mm | Calibrated amount, mm | Brier at 50 mm: calibrated -> with hand-over | "
           "Raw systems alone | Ordinary days handed over |", "|---|---|---|---|---|---|---|"]
    for place, w in withheld.items():
        s, c = w["storm"], w["control"]
        better = s[LABEL]["brier"]["50"] < s["calibrated"]["brier"]["50"]
        p2.append(f"| {SHORT.get(place, place)} | {', '.join(w['storms'])} | {s['observed_mean_mm']:.1f} | "
                  f"{s['calibrated']['amount_mm']:.1f} | {s['calibrated']['brier']['50']:.4f} -> "
                  f"{'**' if better else ''}{s[LABEL]['brier']['50']:.4f}{'**' if better else ''} | "
                  f"{s['raw']['brier']['50']:.4f} | {100 * c[LABEL]['rows_handed_over']:.2f}% |")
    raw_better = [p for p, w in withheld.items() if w["storm"]["raw"]["brier"]["50"] < w["storm"]["calibrated"]["brier"]["50"]]
    little = [p for p in helped if p not in raw_better]
    text = "The calibrated forecast shrinks every staged storm."
    if raw_better:
        text += (f" The hand-over helped most where the raw systems alone beat the calibration on those days "
                 f"({names(raw_better)}).")
    if little:
        text += (f" Where they did not, it still helped a little in {names(little)}, because it acts only on the rows "
                 "that lie beyond the training range and on which the systems agree.")
    if hurt:
        text += f" It did not help in {names(hurt)}, where the raw systems were well behind the calibration."
    if "jamaica" in withheld:
        text += " In Jamaica the withheld days include Hurricane Beryl (3 July 2024)."
    p2 += ["", text, ""]
rules = R.rules_table(new, G[0], SHORT)
if rules:
    p2 += ["### 3.2 Two rules from training data, and what happened", "",
           "The staged storms decide. The yearly check is shown beside them. The last column is the test year: Brier "
           "skill at 50 mm on days 1-3 with the hand-over always on, minus without it. A verdict is bold where it had the "
           "sign of the test year.", ""] + rules + [
        "", f"The staged storms had the sign of the test year in {agree_all} of {n_all} places and the yearly check in "
        f"{agree_yearly} of {n_yearly}. The first count is partly hindsight (see the summary). The effects in the test "
        "year are several times larger than anything the yearly check measures, in both directions, because a year's "
        "verdict on the hand-over is made by a few storms.", "",
        "### 3.3 The test year, three ways", ""]
    for g in G[:2]:
        p2 += [f"Brier skill at 50 mm, {g}. Bold is the best of the three.", ""] + R.handover_three_ways(new, g, SHORT) + [""]
if tails:
    p2 += ["### 3.4 The tail as a whole, and what a warning meant", "",
           "The scores in this section weight the tail of the distribution, following Gneiting and Ranjan (2011). "
           "Scoring forecasts only on the days when something extreme happened would favour a forecaster who always "
           "cries wolf (Lerch et al., 2017), so every day is scored.", ""]
    p2 += tail_bench.markdown([tails[k] for k in ORDER if k in tails]).splitlines() + [""]
if "jamaica" in tails:
    day = "2025-10-28"
    rows = sorted((c for c in tails["jamaica"]["cases"] if c["valid"] == day), key=lambda c: c["lead"])
    if rows:
        first = {c["lead"]: c for c in old.get("jamaica", {}).get("cases", [])
                 if c["valid"] == day and c["model"] == F._top(old["jamaica"])}
        o = rows[0]
        p2 += ["### 3.5 Hurricane Melissa over Jamaica", "",
               f"Observed on {day} (CMORPH): {o['observed_mean_mm']:.0f} mm on average, up to {o['observed_max_mm']:.0f} mm; "
               f"{100 * o['observed_share_50']:.0f}% of places had 50 mm or more and {100 * o['observed_share_100']:.0f}% had "
               "100 mm or more.", "",
               "| Days ahead | First Blend: chance of 50 mm, expected mm | New Blend: chance of 50 mm, of 100 mm, expected mm | "
               "Places under an extreme warning | Raw consensus: chance of 50 mm, mean mm |", "|---|---|---|---|---|"]
        for c in rows:
            f = first.get(c["lead"])
            was = "-" if f is None else f"{100 * f['p50']:.0f}%, {f['forecast_mean_mm']:.0f}"
            p2.append(f"| {c['lead']} | {was} | {100 * c['blend_p50']:.0f}%, {100 * c['blend_p100']:.0f}%, "
                      f"{c['blend_mean_mm']:.0f} | {100 * c['extreme_share']:.0f}% | {100 * c['raw_p50']:.0f}%, "
                      f"{c['raw_mean_mm']:.0f} |")
        p2.append("")
        p2 += R.rarity_lines(experiments_dir / "jamaica_returns.json", day)
p2 += ["## 4. What did not work", "",
       "### 4.1 Handing over everywhere", "",
       "That was the design as first written down. It was tested by staging storms in two regions, and both happened "
       "to be regions where it helps. Section 3.3 shows what it cost in the others.", "",
       "### 4.2 An audit of an ordinary year as the switch", "",
       "This is the yearly check of section 2.6. The sample is a hundred times larger than four storms, but nearly all "
       "of it is heavy days the networks have seen, and the hand-over exists for the storm they have not seen.", "",
       "### 4.3 Discounting the raw systems first", "",
       "We scaled the raw chances by how often they verified at the top of the training forecasts, and never let the "
       "hand-over lower a chance. That still did harm on the Alps' staged storms and gave away most of the gain where "
       "the plain hand-over helps. A switch per place is simpler and did better.", "",
       "### 4.4 Starting the hand-over further out", "",
       "We tried starting at the 99.99th percentile of the training forecasts, and at their maximum, so that no "
       "training row could speak against the hand-over. On the staged storms that gave nearly all of the gain away: "
       "the gain comes from forecasts just beyond what the training period held. The numbers are in "
       "[WHAT_DID_NOT_WORK.md](../WHAT_DID_NOT_WORK.md).", "",
       "## 5. Limits", "",
       "- " + SAME_YEAR,
       "- Where the staged storms switch the hand-over off, a record storm is forecast by the calibration alone, "
       "which shrinks it. The beyond-record flag and the count of raw systems calling for 50 mm are still shown.",
       "- The verdict rests on four storms of one training year, one amount (50 mm) and one truth. A place can pass "
       "and still be hurt in a given year, or fail and then meet the storm that would have justified the hand-over.",
       "- The rule was settled after most places had been scored. A fresh year in fresh places is the test it still "
       "needs.",
       "- The hand-over raises the RMSE of the expected amount slightly where it is used.",
       "- Whether the raw systems overcall depends on the truth. Against gauges in mountains the verdict could differ.",
       "- Return periods come from one to a few years of satellite estimates and are rough.", "",
       "## 6. Reproducing", "",
       "```bash", "pip install pluvium",
       "pluvium blend train --root ~/pluvium-JM        # stages the storms and prints the verdict",
       "```", "",
       "The staged-storm experiment is `pluvium.tails.withheld.withheld_experiment` and its verdict "
       "`pluvium.tails.withheld.verdict`. The yearly check is `pluvium.tails.audit.audit_handover`. The test-year "
       "tables come from `python -m pluvium.benchmarks.frontier_bench`.", "",
       "## References", "",
       "- Dalrymple, T. (1960). Flood-frequency analyses. Manual of Hydrology, part 3. *US Geological Survey "
       "Water-Supply Paper* 1543-A. doi:10.3133/wsp1543A",
       "- Gneiting, T., and Ranjan, R. (2011). Comparing density forecasts using threshold- and quantile-weighted "
       "scoring rules. *Journal of Business and Economic Statistics*, 29, 411-422. doi:10.1198/jbes.2010.08110",
       "- Hosking, J. R. M., and Wallis, J. R. (1987). Parameter and quantile estimation for the generalized Pareto "
       "distribution. *Technometrics*, 29, 339-349. doi:10.1080/00401706.1987.10488243",
       "- Lerch, S., Thorarinsdottir, T. L., Ravazzolo, F., and Gneiting, T. (2017). Forecaster's dilemma: extreme "
       "events and forecast evaluation. *Statistical Science*, 32, 106-127. doi:10.1214/16-STS588",
       "- Pickands, J. (1975). Statistical inference using extreme order statistics. *Annals of Statistics*, 3, "
       "119-131. doi:10.1214/aos/1176343003"]
write("02_record_storms.md", [line for line in p2 if line is not None])

# ============================================================================================ paper 3 ==


def sc(kind, key, g):
    vals = [r["groups"][g][kind][key] for r in scenarios.values() if r["groups"].get(g, {}).get(kind)]
    return float(np.mean(vals)) if vals else float("nan")


first_g, last_g = G[0], G[-1]
sharp_map = {g: (mean_over("stack", "sharpness", None, g), mean_over("ens_raw", "sharpness", None, g)) for g in G}
p3 = [
    "# Sharp scenarios from smooth chances: ensemble copula coupling for multi-model rain forecasts", "",
    f"Pluvium technical paper 3. {BYLINE}", "",
    "## Summary", "",
    "A calibrated chance of rain at each place makes a smooth map, and the map gets smoother the further ahead it "
    "looks, because it averages over every place a storm might be. For any one place that is the correct forecast. "
    "As a picture of a day it misleads, since no real day looks like the average of many. The first benchmark counted "
    "this as a weak spot: the sharpness of the expected-amount map fell with lead.", "",
    "We leave that map alone and add scenarios. Each scenario takes its amounts, place by place, from Pluvium "
    "Blend's calibrated distribution and its arrangement in space from one member of the ECMWF ensemble (ensemble "
    "copula coupling; Schefzik, Thorarinsdottir and Gneiting, 2013), with ties among dry members broken by rain "
    "nearby.",
]
if scenarios:
    p3 += [
        "",
        f"Over {len(scenarios)} places, a coupled scenario on days 11-14 has {sc('coupled', 'sharpness', last_g):.2f} of "
        f"the weather's spatial variability (a raw ENS member: {sc('raw', 'sharpness', last_g):.2f}). The CRPS of rain "
        f"averaged over two-degree boxes is {sc('coupled', 'areal_crps', last_g):.2f} mm for coupled scenarios, "
        f"{sc('uncoupled', 'areal_crps', last_g):.2f} mm for the same amounts with no arrangement and "
        f"{sc('raw', 'areal_crps', last_g):.2f} mm for the raw ensemble. On days 1-3 the three numbers are "
        f"{sc('coupled', 'areal_crps', first_g):.2f}, {sc('uncoupled', 'areal_crps', first_g):.2f} and "
        f"{sc('raw', 'areal_crps', first_g):.2f}. The variogram score, which rewards the right differences between "
        "nearby places, agrees at every lead.",
    ]
p3 += [
    "", CAUTION + " The scenarios have no settings fitted to that year.", "",
    "## 1. The problem", "",
    "Sharpness here is the spatial standard deviation of a map over that of the observed map. For the Blend's "
    "expected amount and for the mean of the raw ECMWF ensemble, averaged over the places:", "",
    "| Map | " + " | ".join(G) + " |", "|---|" + "---|" * len(G),
    "| Pluvium Blend, expected amount | " + " | ".join(f"{sharp_map[g][0]:.2f}" for g in G) + " |",
    "| Raw ECMWF ENS, ensemble mean | " + " | ".join(f"{sharp_map[g][1]:.2f}" for g in G) + " |", "",
    "Both fall with lead, as they must. At day 12 nobody knows where the rain band will lie, so every place gets a "
    "modest chance, and sharpening the map of expected amounts would make it wrong. At that range a user is better "
    "served by a set of days that could happen, each one looking like weather, which together carry the calibrated "
    "chances. A hydrologist needs such a set for rain over a basin, where the total depends on whether nearby places "
    "are wet together.", "",
    "## 2. Method", "",
    "### 2.1 Coupling", "",
    "For one valid day, take the ensemble's `M` members at the places.", "",
    "1. At every place, rank the members from driest to wettest.",
    "2. Give the member ranked `r` the calibrated distribution's quantile at level `(r + 0.5) / M`.", "",
    "Every scenario keeps its member's pattern: where that member was wet compared with the others, the scenario is "
    "wet. At every place the scenarios are an even sample of the calibrated distribution, so every score computed "
    "place by place is unchanged.", "",
    "### 2.2 Ties", "",
    "Rain needs one refinement. At most places most members are dry, so their ranks are ties. Broken at random, the "
    "ties scatter drizzle wherever calibration says rain is more likely than the ensemble did. Here a tie is broken "
    "by the member's rain in the neighbourhood: among dry members, the one with rain nearby is the next to be wet. "
    "Remaining ties are broken at random.", "",
    "### 2.3 Without an ensemble", "",
    "Where no physical ensemble is at hand, the arrangement can come from past observed days instead (the Schaake "
    "shuffle; Clark et al., 2004). `pluvium.scenarios.schaake` does that. It is not scored here.", "",
    "### 2.4 Scores", "",
    "Three sets of scenarios with the same number of members are compared: the raw ensemble, coupled scenarios, and "
    "uncoupled ones (the same amounts, each place drawn on its own).", "",
    "The sharpness of one scenario is its spatial standard deviation over the observed one, so that 1 is as varied "
    "as the weather. The CRPS is computed place by place, where coupled and uncoupled scenarios are equal by "
    "construction. The areal CRPS is the CRPS of the mean rain over boxes two degrees wide, which only scenarios "
    "with the right structure get right. The energy score (Gneiting et al., 2008) and the variogram score "
    "(Scheuerer and Hamill, 2015) reward the right joint behaviour of nearby places.", "",
    "## 3. Results", "",
]
if scenarios:
    p3 += scenario_bench.markdown([scenarios[k] for k in ORDER if k in scenarios]).splitlines() + [""]
    wins = total = 0
    for r in scenarios.values():
        for g in G:
            row = r["groups"].get(g, {})
            if row.get("coupled") and row.get("raw") and row.get("uncoupled"):
                total += 1
                wins += int(row["coupled"]["areal_crps"] < min(row["raw"]["areal_crps"], row["uncoupled"]["areal_crps"]))
    p3 += [f"Coupled scenarios have the lowest areal CRPS in {wins} of {total} place and lead-group cells. Against the "
           "raw ensemble the gain comes from calibration. Against uncoupled draws it comes from structure alone, since "
           "the amounts are the same and only their arrangement differs. That gap widens with lead, because independent "
           "draws average out over a box and real rain does not.", "",
           "A single uncoupled draw is as sharp as a coupled one, or sharper. Sharpness by itself therefore says "
           "nothing about structure, which is what the areal and variogram scores measure.", ""]
p3 += ["## 4. Limits", "",
       "- The arrangement is the ECMWF ensemble's. Where that ensemble puts the rain band in the wrong place, so does "
       "every scenario: coupling corrects the amounts and leaves the positions as the ensemble has them.",
       "- There are as many scenarios as the ensemble has members.",
       "- Only daily totals are coupled, and each day on its own. A scenario is not yet one story across the 14 days.",
       "- Everything is scored against one satellite estimate, on quarter-degree cells.",
       "- " + SAME_YEAR, "",
       "## 5. Reproducing", "",
       "```python",
       "from pluvium.scenarios.ecc import ecc, neighbourhood_tiebreak",
       "near = neighbourhood_tiebreak(fields, lat, lon, place_lat, place_lon)    # each member's rain nearby",
       "scenarios = ecc(chances, members, tiebreak=near)    # chances (places, thresholds), members (M, places)",
       "```", "",
       "The benchmark is `pluvium.benchmarks.scenario_bench`, run by `python -m pluvium.benchmarks.frontier_bench`.", "",
       "## References", "",
       "- Clark, M., Gangopadhyay, S., Hay, L., Rajagopalan, B., and Wilby, R. (2004). The Schaake shuffle: a method "
       "for reconstructing space-time variability in forecasted precipitation and temperature fields. *Journal of "
       "Hydrometeorology*, 5, 243-262. doi:10.1175/1525-7541(2004)005<0243:TSSAMF>2.0.CO;2",
       "- Gneiting, T., Stanberry, L. I., Grimit, E. P., Held, L., and Johnson, N. A. (2008). Assessing probabilistic "
       "forecasts of multivariate quantities, with an application to ensemble predictions of surface winds. *TEST*, "
       "17, 211-235. doi:10.1007/s11749-008-0114-x",
       "- Schefzik, R., Thorarinsdottir, T. L., and Gneiting, T. (2013). Uncertainty quantification in complex "
       "simulation models using ensemble copula coupling. *Statistical Science*, 28, 616-640. doi:10.1214/13-STS443",
       "- Scheuerer, M., and Hamill, T. M. (2015). Variogram-based proper scoring rules for probabilistic forecasts of "
       "multivariate quantities. *Monthly Weather Review*, 143, 1321-1334. doi:10.1175/MWR-D-14-00269.1"]
write("03_scenarios.md", p3)
