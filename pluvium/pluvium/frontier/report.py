# Copyright 2026 Nathan. Apache-2.0.
"""
The frontier benchmark as tables: where the new Blend stands, what each part added, and what changed since the first.

The results come from :func:`pluvium.frontier.evaluate.evaluate` (one per place) and have the form of the first
benchmark's, so its scoreboard reads them too. Three things are added here:

* **before and after**: the first benchmark's Blend against the new one, place by place, for any score;
* **ablation**: the pool alone, with the online layer, with the record-storm hand-over, with both as first declared,
  and with the hand-over decided place by place, so the worth of each part is on the page, losses included;
* **the online weights**: where the weight went over the year;
* **the hand-over's two rules**: what each place's staged storms said, what the yearly check would have said, and
  what the hand-over then did in the test year.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from ..benchmarks import fullyear as F

#: the benchmark's places, in the order the pages show them, with short names
PLACE_ORDER = ("jamaica", "caribbean", "maritime", "alps", "pacific_nw", "great_plains", "andes_south")
PLACE_NAMES = F.SHORT_TITLES
#: scored before the rule that decides the hand-over was final, and scored after
SEEN_FIRST = ("caribbean", "maritime", "alps", "pacific_nw", "andes_south")
SCORED_AFTER = ("great_plains", "jamaica")
ABLATIONS = ("declared", "pool", "pool_online", "pool_record")
PARTS = (("pool", "Pool alone"), ("pool_online", "Pool + online weights"),
         ("pool_record", "Pool + hand-over, always on"),
         ("declared", "Pool + online weights + hand-over, always on (as first declared)"),
         ("stack", "Pool + online weights + hand-over where the staged storms say so (Pluvium Blend)"))


def load_results(folder, suffix: str = "", places=PLACE_ORDER) -> dict:
    """``{place: result}`` for every place whose ``{place}{suffix}.json`` is in ``folder``, in the pages' order."""
    folder = Path(folder)
    return {k: json.loads((folder / f"{k}{suffix}.json").read_text()) for k in places if (folder / f"{k}{suffix}.json").exists()}


def join_names(places, titles: dict = PLACE_NAMES, the: bool = False) -> str:
    """"A", "A and B" or "A, B and C" from place keys. With ``the``, names that take an article in a sentence get
    one ("the Alps", "the US Great Plains"; not "Jamaica")."""
    words = [titles.get(p, p) for p in places]
    if the:
        words = [w if w == "Jamaica" else "the " + w for w in words]
    if not words:
        return "none"
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def without(results: dict, drop=ABLATIONS) -> dict:
    """The results with the ablation designs removed, so scoreboards count only real rivals."""
    out = copy.deepcopy(results)
    for res in out.values():
        for part in ("groups", "leads"):
            for row in res[part].values():
                for name in drop:
                    row.pop(name, None)
        for seasons in res["seasons"].values():
            for row in seasons.values():
                for name in drop:
                    row.pop(name, None)
    return out


def _cell(v: dict | None, key: str, t: str | None) -> float:
    return F._value(v, key, t) if v else float("nan")


def best_rival(row: dict, key: str, t: str | None, higher: bool) -> tuple:
    """``(value, name)`` of the best system that is not Pluvium's own, raw or post-processed."""
    ours = {"stack", "blend", "blend_ops", "neural_pp", "neural_raw", *ABLATIONS}
    vals = [(_cell(v, key, t), m) for m, v in row.items() if m not in ours and v]
    vals = [x for x in vals if x[0] == x[0]]
    if not vals:
        return float("nan"), None
    return max(vals) if higher else min(vals)


def before_after(old: dict, new: dict, key: str = "bss", t: str | None = "50", higher: bool = True,
                 season: str | None = None, digits: int = 3) -> list:
    """One row per place, one column per lead group: first Blend -> new Blend (best other system). Bold where the new
    Blend is ahead of every other system; a trailing ``+`` where it improved on the first."""
    lines = ["| Place | " + " | ".join(F.LEAD_GROUPS) + " |", "|---|" + "---|" * len(F.LEAD_GROUPS)]
    for place, res in new.items():
        cells = []
        for g in F.LEAD_GROUPS:
            row = res["groups"][g] if season is None else res["seasons"][g][season]
            before = old.get(place, {}).get("groups" if season is None else "seasons", {}).get(g, {})
            before = before if season is None else before.get(season, {})
            a, b = _cell(before.get(F._top(old[place])) if place in old else None, key, t), _cell(row.get("stack"), key, t)
            rival, _ = best_rival(row, key, t, higher)
            if b != b:
                cells.append("-")
                continue
            ahead = rival != rival or (b > rival if higher else b < rival)
            gained = a == a and (b > a if higher else b < a)
            text = f"{b:+.{digits}f}" if higher and key in ("bss", "acc") else f"{b:.{digits}f}"
            first = "-" if a != a else (f"{a:+.{digits}f}" if higher and key in ("bss", "acc") else f"{a:.{digits}f}")
            cells.append(f"{first} -> {'**' if ahead else ''}{text}{'**' if ahead else ''}"
                         + (f" ({rival:+.{digits}f})" if rival == rival and higher and key in ("bss", "acc")
                            else (f" ({rival:.{digits}f})" if rival == rival else "")))
            if gained:
                cells[-1] += " +"
        lines.append(f"| {F.REGION_TITLES.get(place, place)} | " + " | ".join(cells) + " |")
    return lines


def tally(old: dict, new: dict, key: str, t: str | None, higher: bool = True, groups=None, season: str | None = None) -> dict:
    """Counts over places and lead groups: how often the new Blend improved on the first, and how often each led
    every other system."""
    out = {"cells": 0, "improved": 0, "new_leads": 0, "old_leads": 0}
    for place, res in new.items():
        for g in groups or F.LEAD_GROUPS:
            row = res["groups"][g] if season is None else res["seasons"][g][season]
            b = _cell(row.get("stack"), key, t)
            if b != b:
                continue
            rival, _ = best_rival(row, key, t, higher)
            out["cells"] += 1
            out["new_leads"] += int(rival != rival or (b > rival if higher else b < rival))
            if place in old:
                orow = old[place]["groups"][g] if season is None else old[place]["seasons"][g][season]
                a = _cell(orow.get(F._top(old[place])), key, t)
                orival, _ = best_rival(orow, key, t, higher)
                if a == a:
                    out["improved"] += int(b > a if higher else b < a)
                    out["old_leads"] += int(orival != orival or (a > orival if higher else a < orival))
    return out


def ablation(results: dict, board=(("Brier skill, 1 mm", "bss", "1", True), ("Brier skill, 25 mm", "bss", "25", True),
                                   ("Brier skill, 50 mm", "bss", "50", True), ("CRPS", "crps", None, False),
                                   ("RMSE", "rmse", None, False))) -> list:
    """The mean over places of each score for each part of the design, by lead group."""
    lines = []
    for title, key, t, _higher in board:
        lines += [f"**{title}** (mean over {len(results)} places)", "",
                  "| Design | " + " | ".join(F.LEAD_GROUPS) + " |", "|---|" + "---|" * len(F.LEAD_GROUPS)]
        for name, label in PARTS:
            cells = []
            for g in F.LEAD_GROUPS:
                vals = [_cell(res["groups"][g].get(name), key, t) for res in results.values()]
                vals = [v for v in vals if v == v]
                cells.append(f"{np.mean(vals):+.4f}" if vals and key == "bss" else (f"{np.mean(vals):.4f}" if vals else "-"))
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def online_weights(res: dict, design: str = "stack") -> list:
    """The online layer's weights for light rain on days 1-3, at the end of each quarter of the test year."""
    w = res.get("online_weights", {}).get(design)
    if not w:
        return []
    names = [F.LABELS.get(n, {"blend_ops": "one network", "pool": "the pool", "neural_pp": "Pluvium's network",
                              "blend": "one network (all seven)"}.get(n, n)) for n in w["experts"]]
    lines = ["| After | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for q, row in enumerate(w["by_quarter_days_1_3_light"]):
        lines.append(f"| quarter {q + 1} | " + " | ".join(f"{x:.2f}" for x in row) + " |")
    return lines


RULES = ("staged", "yearly")


def kept(res: dict, rule: str = "staged") -> bool | None:
    """Whether a rule kept the hand-over in a place, or ``None`` if the result does not say.

    ``"staged"`` is the rule that decides: the place's staged storms (:func:`pluvium.tails.withheld.verdict`).
    ``"yearly"`` is the check that is only reported: the out-of-fold audit at the first refit
    (:mod:`pluvium.tails.audit`)."""
    if rule == "staged":
        verdict = res.get("staged")
        return None if not verdict else bool(verdict["keep"])
    checks = res.get("audits", {}).get("stack", [])
    return bool(checks[0]["keep"]) if checks else None


def handover_three_ways(results: dict, group: str, titles: dict | None = None, t: str = "50") -> list:
    """Brier skill at ``t`` mm with no hand-over, with it always on, and as the place's staged storms decided, place
    by place, with the mean. Bold is the best of the three in a row."""
    names = ("pool_online", "declared", "stack")
    lines = ["| Place | No hand-over | Always on (as first declared) | As the staged storms decided (Pluvium Blend) | "
             "Their verdict |", "|---|---|---|---|---|"]
    cols = {m: [] for m in names}
    for place, res in results.items():
        vals = {m: _cell(res["groups"][group].get(m), "bss", t) for m in names}
        if any(v != v for v in vals.values()):
            continue
        for m, v in vals.items():
            cols[m].append(v)
        best = max(vals.values())
        cells = [f"**{vals[m]:+.4f}**" if vals[m] == best else f"{vals[m]:+.4f}" for m in names]
        verdict = kept(res)
        said = "-" if verdict is None else ("keep" if verdict else "switch off")
        lines.append(f"| {(titles or F.REGION_TITLES).get(place, place)} | " + " | ".join(cells) + f" | {said} |")
    if cols["stack"]:
        lines.append("| **Mean** | " + " | ".join(f"{np.mean(cols[m]):+.4f}" for m in names) + " | |")
    return lines


def audit_outcomes(results: dict, group: str = "days 1-3", places=None, t: str = "50", rule: str = "staged") -> list:
    """Per place: ``(place, the rule's verdict, what the hand-over then did in the test year)``. The last is the
    change in Brier skill at ``t`` mm from having the hand-over always on instead of never."""
    out = []
    for place, res in results.items():
        if places is not None and place not in places:
            continue
        verdict = kept(res, rule)
        did = _cell(res["groups"][group].get("declared"), "bss", t) - _cell(res["groups"][group].get("pool_online"), "bss", t)
        if verdict is not None and did == did:
            out.append((place, verdict, did))
    return out


def audit_agreement(results: dict, group: str = "days 1-3", places=None, t: str = "50", rule: str = "staged") -> tuple:
    """In how many places a rule's verdict had the sign of what the hand-over then did in the test year (Brier skill
    at ``t`` mm: always on against none). Returns ``(agree, places counted)``."""
    outcomes = audit_outcomes(results, group, places, t, rule)
    return sum(int(verdict == (did >= 0)) for _, verdict, did in outcomes), len(outcomes)


def audit_misses(results: dict, group: str = "days 1-3", titles: dict = PLACE_NAMES, t: str = "50",
                 rule: str = "staged") -> str:
    """A sentence on the places where a rule's verdict went against the test year, or ``""`` if there are none."""
    parts = []
    for place, verdict, did in audit_outcomes(results, group, None, t, rule):
        if verdict == (did >= 0):
            continue
        name = titles.get(place, place)
        name = name if name == "Jamaica" else "the " + name
        parts.append(f"in {name} it {'kept' if verdict else 'switched off'} a hand-over that then "
                     f"{'cost' if did < 0 else 'would have added'} {abs(did):.4f} of Brier skill at {t} mm on {group}")
    if not parts:
        return ""
    return ("The verdict was wrong " if rule == "staged" else "The yearly check would have been wrong ") + "; ".join(parts) + "."


def rarity_lines(path, day: str = "2025-10-28", place: str = "Jamaica") -> list:
    """A paragraph on how rare a day's rain was, from a return-period file: ``{"span", "years", "horizon", "places",
    "level_mm": {years: mm}, "days": {day: {"share_ge_1y", "share_ge_5y", "share_beyond_horizon", "median_period_y"}}}``
    (:class:`pluvium.tails.returns.ReturnLevels` fitted on the record before the test year). Empty if there is none."""
    path = Path(path)
    if not path.exists():
        return []
    r = json.loads(path.read_text())
    d = r["days"].get(day)
    if not d:
        return []
    return [f"**How rare was it?** A generalised Pareto fit to {place}'s CMORPH record from {r['span'][0][:7]} to "
            f"{r['span'][1][:7]} ({r['years']:.1f} years, {r['places']} places, all before the test year) puts the daily "
            f"amount reached once a year at {r['level_mm']['1']:.0f} mm on average over the places, and once in five years "
            f"at {r['level_mm']['5']:.0f} mm. On {day}, {100 * d['share_ge_1y']:.0f}% of places had more than their "
            f"once-a-year amount, {100 * d['share_ge_5y']:.0f}% more than their once-in-five-years amount, and "
            f"{100 * d['share_beyond_horizon']:.0f}% an amount rarer than once in {r['horizon']:.0f} years, which is as far "
            f"as a record of that length can say. The median place had a day expected about once in "
            f"{d['median_period_y']:.0f} years. These are rough figures from five years of satellite estimates.", ""]


def rules_table(results: dict, group: str = "days 1-3", titles: dict | None = None, t: str = "50") -> list:
    """One row per place: what each of the two training-data rules said about the hand-over, and what the hand-over
    then did in the test year.

    The staged storms (the rule that decides): the change in the Brier score at 50 mm on the storms withheld from
    training. The yearly check (reported only): the change over all the training period's out-of-fold forecasts at
    the first refit, and at how many of the refits it would have kept the hand-over. The test year: Brier skill at
    ``t`` mm with the hand-over always on, minus without it. Bold marks a verdict with the sign of the test year."""
    lines = ["| Place | Staged storms: Brier at 50 mm on the withheld storms | Their verdict | Yearly check: Brier at 50 mm "
             "over the training period | It would have said | Test year: what the hand-over did |", "|---|---|---|---|---|---|"]
    for place, res in results.items():
        verdict, checks = res.get("staged"), res.get("audits", {}).get("stack", [])
        did = _cell(res["groups"][group].get("declared"), "bss", t) - _cell(res["groups"][group].get("pool_online"), "bss", t)
        if not verdict or did != did:
            continue

        def say(keep: bool) -> str:
            word = "keep" if keep else "switch off"
            return f"**{word}**" if keep == (did >= 0) else word

        staged = f"{-100 * verdict['gain']:+.1f}%" if verdict.get("evidence") else "no storms"
        if checks:
            a = checks[0]
            change = 100.0 * (a["handed"] - a["without"]) / max(a["without"], 1e-12)
            yearly = f"{change:+.2f}%"
            would = say(bool(a["keep"])) + f" (at {sum(bool(c['keep']) for c in checks)} of {len(checks)} refits: keep)"
        else:
            yearly, would = "-", "-"
        lines.append(f"| {(titles or F.REGION_TITLES).get(place, place)} | {staged} | {say(bool(verdict['keep']))} | {yearly} | "
                     f"{would} | {did:+.4f} |")
    return lines if len(lines) > 2 else []


__all__ = ["ABLATIONS", "PARTS", "PLACE_NAMES", "PLACE_ORDER", "RULES", "SCORED_AFTER", "SEEN_FIRST", "ablation",
           "audit_agreement", "audit_misses", "audit_outcomes", "before_after", "best_rival", "handover_three_ways",
           "join_names", "kept", "load_results", "online_weights", "rarity_lines", "rules_table", "tally", "without"]
