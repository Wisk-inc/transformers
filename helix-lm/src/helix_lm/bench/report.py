# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The report of a run: one table per task (systems as rows, the task's main metrics as columns, each a mean with
its 95% interval), then the rows broken down by singer or technique where the task has them, and what was skipped.
Also CSV of every row, for a spreadsheet."""

from __future__ import annotations

import csv
import io
import math

from .metrics import INFO

MAIN = {
    "resynthesis": ("mcd_db", "f0_rmse_cents", "vuv_error", "mrstft", "si_snr_db", "glitches_per_minute"),
    "svc": ("speaker_similarity", "identity_margin", "f0_rmse_cents", "ffe", "consonant_clarity", "quality_proxy"),
    "svs": ("note_accuracy", "note_error_cents", "onset_error_ms", "mcd_db", "quality_proxy"),
    "pitch_control": ("shift_error_cents",),
    "technique": ("hnr_error_db", "vibrato_kept", "tilt_error_db", "jitter_error", "shimmer_error"),
    "robustness": ("f0_shift_cents", "voice_kept"),
    "register": ("register_pops_per_minute",),
    "longform": ("drift_cents_per_min", "timbre_drift"),
    "speed": ("rtf", "peak_memory_gb"),
}


def fmt(cell) -> str:
    if not cell or math.isnan(cell[0]):
        return "n/a"
    mean, lo, hi, n = cell
    digits = 3 if abs(mean) < 10 else 1
    if n < 2 or math.isnan(lo):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def header(metric: str) -> str:
    info = INFO.get(metric)
    arrow = "" if info is None else (" (lower)" if info.better == "lower" else " (higher)")
    return f"{metric}{arrow}"


def markdown(result: dict) -> str:
    lines = [f"# Benchmark: suite {result['suite']}", "",
             f"Sets: {', '.join(f'{k} ({v} items)' for k, v in result['sets'].items()) or 'none'}. "
             f"Sample rate {result['sample_rate']} Hz. Each cell: mean [95% bootstrap interval].", ""]
    summary = result["summary"]
    for task, metrics in MAIN.items():
        systems = [s for s in summary if task in summary[s]]
        if not systems:
            continue
        cols = [m for m in metrics if any(m in summary[s][task] for s in systems)]
        lines += [f"## {task}", "", "| system | " + " | ".join(header(m) for m in cols) + " | n |",
                  "|---|" + "---|" * (len(cols) + 1)]
        for s in systems:
            cell = summary[s][task]
            n = max((v[3] for v in cell.values()), default=0)
            lines.append(f"| {s} | " + " | ".join(fmt(cell.get(m)) for m in cols) + f" | {n} |")
        lines.append("")
    by = breakdown(result["rows"], "technique", "technique", "hnr_error_db")
    if by:
        lines += ["## technique, by technique (HNR error, dB)", "", *by, ""]
    if result.get("skipped"):
        lines += ["## skipped", "", *[f"- {s}" for s in result["skipped"]], ""]
    lines.append(f"Run time {result['seconds']} s; outputs made {result['cache']['made']}, "
                 f"from the cache {result['cache']['hits']}.")
    return "\n".join(lines) + "\n"


def breakdown(rows: list[dict], task: str, key: str, metric: str) -> list[str]:
    table: dict = {}
    for r in rows:
        if r.get("task") == task and key in r and isinstance(r.get(metric), float) and not math.isnan(r[metric]):
            table.setdefault(r["system"], {}).setdefault(r[key], []).append(r[metric])
    if not table:
        return []
    groups = sorted({g for t in table.values() for g in t})
    out = ["| system | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    for s, t in table.items():
        out.append(f"| {s} | " + " | ".join(f"{sum(t[g]) / len(t[g]):.2f}" if g in t else "" for g in groups) + " |")
    return out


def rows_csv(rows: list[dict]) -> str:
    keys = sorted({k for r in rows for k in r})
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in keys})
    return buf.getvalue()


__all__ = ["MAIN", "breakdown", "markdown", "rows_csv"]
