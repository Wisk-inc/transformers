# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
A new checkpoint against the last one on the same suite: every main metric that got worse by more than its
tolerance, with the paired test that says whether the change is more than noise. The molab program runs this after
each evaluation, so a training run that starts to break something is seen at the next checkpoint, not at the end.
"""

from __future__ import annotations

import math

from .leaderboard import per_item
from .metrics import better
from .report import MAIN
from .stats import paired, wilcoxon

TOLERANCE = {"f0_rmse_cents": 5.0, "mcd_db": 0.1, "speaker_similarity": 0.01, "note_accuracy": 0.01,
             "glitches_per_minute": 0.5, "quality_proxy": 0.02, "ffe": 0.005, "vuv_error": 0.005}


def compare(new: dict, old: dict, system: str = "tristream", alpha: float = 0.05) -> list[dict]:
    """``[{task, metric, old, new, worse_by, p, significant}]`` for the metrics that got worse."""
    found = []
    for task, metrics in MAIN.items():
        a = new["summary"].get(system, {}).get(task, {})
        b = old["summary"].get(system, {}).get(task, {})
        for m in metrics:
            if m not in a or m not in b or any(math.isnan(x) for x in (a[m][0], b[m][0])):
                continue
            delta = a[m][0] - b[m][0]
            worse = delta if better(m) == "lower" else -delta
            if worse <= TOLERANCE.get(m, 0.0):
                continue
            p = wilcoxon(paired(per_item(new["rows"], system, task, m), per_item(old["rows"], system, task, m)))
            found.append({"task": task, "metric": m, "old": b[m][0], "new": a[m][0], "worse_by": worse, "p": p,
                          "significant": p < alpha})
    return found


def describe(found: list[dict]) -> str:
    if not found:
        return "no metric got worse beyond its tolerance"
    return "; ".join(f"{f['task']}/{f['metric']} {f['old']:.3g} -> {f['new']:.3g}"
                     f"{' (significant)' if f['significant'] else ' (within noise)'}" for f in found)


__all__ = ["TOLERANCE", "compare", "describe"]
