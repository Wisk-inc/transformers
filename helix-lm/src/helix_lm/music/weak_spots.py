# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Where a model is weak, as sampling weights: evaluation rows grouped by register (the clip's pitch range, in
octave-wide buckets), technique and singer, each group's mean composite compared with the overall mean, and the
weaker groups weighted up for the next stretch of training (between 1 and ``max_weight``)."""

from __future__ import annotations

import math

from .composite import composite


def register_bucket(f0_high: float) -> str:
    if not f0_high or f0_high <= 0:
        return "unknown"
    edges = ((165, "low"), (330, "middle"), (660, "high"))
    for top, name in edges:
        if f0_high < top:
            return name
    return "very high"


def group_weights(rows: list[dict], key: str, max_weight: float = 2.5) -> dict:
    groups: dict = {}
    for r in rows:
        s = composite(r)
        if not math.isnan(s):
            groups.setdefault(str(r.get(key, "unknown")), []).append(s)
    if not groups:
        return {}
    overall = sum(sum(v) for v in groups.values()) / sum(len(v) for v in groups.values())
    out = {}
    for g, v in groups.items():
        gap = max(0.0, overall - sum(v) / len(v))
        out[g] = min(max_weight, 1.0 + 5.0 * gap)
    return out


def clip_weight(tags: dict, by_register: dict, by_technique: dict) -> float:
    """The weight of one clip from its tags (``f0_high``, ``technique``)."""
    w = by_register.get(register_bucket(float(tags.get("f0_high", 0.0) or 0.0)), 1.0)
    return w * by_technique.get(str(tags.get("technique", "unknown")), 1.0)


__all__ = ["clip_weight", "group_weights", "register_bucket"]
