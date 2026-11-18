# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Clips a model gets wrong, sampled more. From the per-clip rows of an evaluation (pitch error, glitches, voice
similarity), each clip gets a weight between 1 and ``max_weight`` by how far its composite score falls under the
median; the training sampler multiplies its own weights by these. Bounded, so a broken clip cannot take over."""

from __future__ import annotations

import math

from .composite import composite


def weights(rows: list[dict], key: str = "item", max_weight: float = 3.0) -> dict:
    scores = {}
    for r in rows:
        s = composite(r)
        if not math.isnan(s):
            scores[r[key]] = s
    if not scores:
        return {}
    ordered = sorted(scores.values())
    median = ordered[len(ordered) // 2]
    out = {}
    for k, s in scores.items():
        gap = max(0.0, median - s)
        out[k] = min(max_weight, 1.0 + 4.0 * gap)
    return out


def hardest(rows: list[dict], n: int = 10, key: str = "item") -> list[str]:
    w = weights(rows, key, max_weight=1e9)
    return sorted(w, key=w.get, reverse=True)[:n]


__all__ = ["hardest", "weights"]
