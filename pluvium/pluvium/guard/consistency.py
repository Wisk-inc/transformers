# Copyright 2026 Nathan. Apache-2.0.
"""
Outputs that contradict each other, found and repaired.

The products are computed coherently from one chain, but calibration, online corrections and mixing are applied
per product, and floating point is floating point. Before anything is published:

* probabilities are clipped to [0, 1];
* "any rain within h hours" is made non-decreasing in ``h``, and at least as large as the largest hourly chance in
  the window;
* the intensity classes are made non-negative and to sum to one; heavy never exceeds moderate-or-heavy, which never
  exceeds rain;
* the start window is ordered (early <= median <= late);
* totals quantiles are ordered (p10 <= median <= p90), and expected amounts are non-negative.

Each repair is counted and reported; repeated repairs are a symptom worth investigating.
"""

from __future__ import annotations

import numpy as np


def check_summary(s: dict) -> list[str]:
    """Repair ``s`` (a :func:`pluvium.products.summarize` result) in place; return what was repaired."""
    notes = []

    def note(text, count):
        if count:
            notes.append(f"{text}: {int(count):,} values repaired")

    for key in ("p_wet", "p_now", "torrential"):
        if key in s and s[key] is not None:
            bad = (s[key] < 0) | (s[key] > 1) | ~np.isfinite(s[key])
            note(key, bad.sum())
            s[key] = np.clip(np.nan_to_num(s[key]), 0, 1)
    if "classes" in s:
        c = np.clip(np.nan_to_num(s["classes"]), 0, None)
        total = c.sum(-1, keepdims=True)
        note("intensity classes not summing to one", (np.abs(total - 1) > 1e-3).sum())
        s["classes"] = c / np.maximum(total, 1e-12)
    windows = sorted(s.get("any_rain", {}))
    prev = None
    for h in windows:
        v = np.clip(s["any_rain"][h], 0, 1)
        floor = s["p_wet"][:, :h].max(-1) if "p_wet" in s else 0
        fixed = (v < floor - 1e-6).sum()
        v = np.maximum(v, floor)
        if prev is not None:
            fixed += (v < prev - 1e-6).sum()
            v = np.maximum(v, prev)
        note(f"any rain within {h} h", fixed)
        s["any_rain"][h] = v
        prev = v
    if all(k in s for k in ("start_early_24", "start_median_24", "start_late_24")):
        e, m, late = s["start_early_24"], s["start_median_24"], s["start_late_24"]
        bad = np.isfinite(m) & ((e > m) | (m > late))
        note("start window out of order", bad.sum())
        s["start_early_24"], s["start_late_24"] = np.fmin(e, m), np.fmax(late, m)
    for h, d in s.get("totals", {}).items():
        bad = (d["p10"] > d["median"]) | (d["median"] > d["p90"])
        note(f"{h} h totals quantiles", bad.sum())
        d["p10"] = np.minimum(d["p10"], d["median"])
        d["p90"] = np.maximum(d["p90"], d["median"])
    if "expected_mm" in s:
        note("negative expected rain", (s["expected_mm"] < 0).sum())
        s["expected_mm"] = np.clip(s["expected_mm"], 0, None)
    return notes


__all__ = ["check_summary"]
