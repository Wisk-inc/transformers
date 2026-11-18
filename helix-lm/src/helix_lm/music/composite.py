# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
One number for "is this checkpoint better": each main metric mapped to [0, 1] between a floor (0) and a target (1)
and averaged with weights. The anchors are fixed here, not fitted to a run, so scores of different runs compare.
They are set at levels the literature and this suite's baselines bracket: 100 cents of pitch error is a wrong
note, 10 dB MCD is another voice, 0.95 similarity is the same voice.
"""

from __future__ import annotations

import math

ANCHORS = {                    # metric: (floor, target, weight)
    "f0_rmse_cents": (100.0, 5.0, 2.0),
    "ffe": (0.2, 0.0, 1.0),
    "vuv_error": (0.2, 0.0, 1.0),
    "mcd_db": (10.0, 2.0, 1.5),
    "speaker_similarity": (0.4, 0.9, 2.0),        # 0.4: another singer on the same vowel (fallback print)
    "note_accuracy": (0.5, 1.0, 1.5),
    "consonant_clarity": (0.3, 1.0, 1.0),
    "quality_proxy": (0.0, 1.0, 1.0),
    "glitches_per_minute": (10.0, 0.0, 1.0),
    "shift_error_cents": (100.0, 0.0, 0.5),
}


def scaled(metric: str, value: float) -> float:
    floor, target, _ = ANCHORS[metric]
    if value is None or math.isnan(value):
        return math.nan
    return min(1.0, max(0.0, (value - floor) / (target - floor)))


def composite(metrics: dict) -> float:
    """Weighted mean of the scaled metrics present; NaN when none is."""
    num = den = 0.0
    for name, (_, _, weight) in ANCHORS.items():
        v = metrics.get(name)
        if isinstance(v, (list, tuple)):
            v = v[0]
        s = scaled(name, v) if isinstance(v, (int, float)) else math.nan
        if not math.isnan(s):
            num, den = num + weight * s, den + weight
    return num / den if den else math.nan


def from_summary(summary: dict, system: str) -> float:
    """The composite of one system over every task of a benchmark summary (``runner.summarize``)."""
    merged: dict = {}
    for task in summary.get(system, {}).values():
        for k, cell in task.items():
            merged.setdefault(k, []).append(cell[0])
    return composite({k: sum(v) / len(v) for k, v in merged.items() if all(x == x for x in v)})


__all__ = ["ANCHORS", "composite", "from_summary", "scaled"]
