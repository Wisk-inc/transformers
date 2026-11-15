# Copyright 2026 Nathan. Apache-2.0.
"""
The frontier benchmark as tables: where the new Blend stands, what each part added, and what changed since the first.

The results come from :func:`pluvium.frontier.evaluate.evaluate` (one per place) and have the form of the first
benchmark's, so its scoreboard reads them too. Three things are added here:

* **before and after**: the first benchmark's Blend against the new one, place by place, for any score;
* **ablation**: the pool alone, with the online layer, with the record-storm hand-over, and with both, so the worth of
  each part is on the page, losses included;
* **the online weights**: where the weight went over the year.
"""

from __future__ import annotations

import copy

import numpy as np

from ..benchmarks import fullyear as F

ABLATIONS = ("pool", "pool_online", "pool_record")
PARTS = (("pool", "Pool alone"), ("pool_online", "Pool + online weights"), ("pool_record", "Pool + record-storm hand-over"),
         ("stack", "All three (Pluvium Blend)"))


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


__all__ = ["ABLATIONS", "PARTS", "ablation", "before_after", "best_rival", "online_weights", "tally", "without"]
