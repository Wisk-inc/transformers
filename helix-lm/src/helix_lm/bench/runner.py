# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
A benchmark run: every system on every task of a suite, rows of metrics written as they come, a summary per
system and task with confidence intervals, and the report. Outputs are cached in ``<out>/cache``, so a run stopped
in the middle (a molab session ending) goes on from where it was.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .cache import OutputCache
from .config import suite as get_suite
from .data import synthetic
from .stats import mean_ci
from .systems import make
from .tasks import runner

SKIP_COLUMNS = {"item", "set", "singer", "target", "technique", "degradation", "system", "task"}


def gather(suite, data_dirs: dict | None, out: Path, sample_rate: int) -> dict:
    """``{set: items}`` for the sets of the suite that are here (the synthetic set is made in ``out``)."""
    data_dirs = data_dirs or {}
    found = {}
    for name in suite.sets:
        if name == "synthetic":
            found[name] = synthetic.make_set(out / "sets" / "synthetic", suite.takes_per_singer,
                                             seconds=suite.seconds, sample_rate=sample_rate)
            continue
        root = data_dirs.get(name)
        if not root or not Path(root).exists():
            continue
        import importlib

        module = importlib.import_module(f"{__package__}.data.{name}")
        items = module.items(root)
        if items:
            found[name] = items[: suite.max_items]
    return found


def summarize(rows: list[dict]) -> dict:
    """``{system: {task: {metric: [mean, low, high, n]}}}``."""
    out: dict = {}
    for row in rows:
        cell = out.setdefault(row["system"], {}).setdefault(row["task"], {})
        for k, v in row.items():
            if k not in SKIP_COLUMNS and isinstance(v, (int, float)) and not isinstance(v, bool):
                cell.setdefault(k, []).append(v)
    for system in out.values():
        for task, metrics in system.items():
            system[task] = {k: [*mean_ci(v), len(v)] for k, v in metrics.items()}
    return out


def run_suite(suite="quick", systems=("identity", "harmonic-noise"), out="bench-out", sample_rate: int = 24000,
              root=None, size: str = "ts-765m", data_dirs: dict | None = None, log=print) -> dict:
    s = get_suite(suite)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sets = gather(s, data_dirs, out, sample_rate)
    cache = OutputCache(out / "cache")
    rows, skipped = [], []
    began = time.time()
    for name in systems:
        system = make(name, root, size)
        if system is None:
            skipped.append(f"{name}: no trained model under {root}")
            continue
        for task in s.tasks:
            if task not in system.supports:
                continue
            for set_name, items in sets.items():
                from .data.registry import get

                if task not in get(set_name).tasks and task not in ("speed", "robustness", "longform", "technique",
                                                                     "register"):
                    continue
                t0 = time.time()
                part = runner(task)(system, items[: s.max_items], sample_rate, cache)
                for row in part:
                    row.update(system=name, task=task)
                rows += part
                log(f"[bench] {name} / {task} / {set_name}: {len(part)} rows in {time.time() - t0:.1f} s")
    summary = summarize(rows)
    result = {"suite": s.name, "systems": list(systems), "sets": {k: len(v) for k, v in sets.items()},
              "sample_rate": sample_rate, "seconds": round(time.time() - began, 1), "skipped": skipped,
              "cache": {"hits": cache.hits, "made": cache.misses}, "summary": summary, "rows": rows}
    (out / "results.json").write_text(json.dumps(result, indent=1, default=float))
    from .report import markdown

    (out / "report.md").write_text(markdown(result))
    return result


__all__ = ["gather", "run_suite", "summarize"]
