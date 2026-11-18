# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Results of several runs side by side, and a ranking. A system is ranked on each task's main metrics by its mean;
two systems whose paired difference is not significant (Wilcoxon, p >= 0.05 over the items both were measured
on) share a rank rather than being ordered by noise.
"""

from __future__ import annotations

import json
from pathlib import Path

from .metrics import better
from .report import MAIN
from .stats import paired, wilcoxon


def load(paths) -> list[dict]:
    return [json.loads(Path(p).read_text()) for p in paths]


def per_item(rows: list[dict], system: str, task: str, metric: str) -> dict:
    return {r["item"]: r[metric] for r in rows if r.get("system") == system and r.get("task") == task
            and isinstance(r.get(metric), (int, float)) and r[metric] == r[metric]}


def rank(result: dict, task: str, metric: str, alpha: float = 0.05) -> list[tuple[str, int, float]]:
    """``[(system, rank, mean)]``, best first; ties where the difference is not significant."""
    rows = result["rows"]
    means = []
    for system, tasks in result["summary"].items():
        cell = tasks.get(task, {}).get(metric)
        if cell and cell[0] == cell[0]:
            means.append((system, cell[0]))
    lower = better(metric) == "lower"
    means.sort(key=lambda x: x[1] if lower else -x[1])
    out, current = [], 0
    for i, (system, mean) in enumerate(means):
        if i == 0:
            current = 1
        else:
            prev = means[i - 1][0]
            p = wilcoxon(paired(per_item(rows, system, task, metric), per_item(rows, prev, task, metric)))
            if p < alpha:
                current = i + 1
        out.append((system, current, mean))
    return out


def table(result: dict) -> str:
    lines = ["| task | metric | ranking (best first; = means not told apart) |", "|---|---|---|"]
    for task, metrics in MAIN.items():
        for metric in metrics:
            ranking = rank(result, task, metric)
            if len(ranking) < 2:
                continue
            text = ", ".join(f"{r}. {s} ({m:.3g})" for s, r, m in ranking)
            lines.append(f"| {task} | {metric} | {text} |")
    return "\n".join(lines) + "\n"


def merge(results: list[dict]) -> dict:
    """One result from several runs of the same suite (systems measured in separate runs)."""
    if not results:
        return {}
    out = {**results[0], "rows": [], "summary": {}, "systems": []}
    for r in results:
        out["rows"] += r["rows"]
        out["summary"].update(r["summary"])
        out["systems"] += [s for s in r["systems"] if s not in out["systems"]]
    return out


__all__ = ["load", "merge", "per_item", "rank", "table"]
