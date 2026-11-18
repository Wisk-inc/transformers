# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Which checkpoint to keep: the one with the best composite benchmark score, not the lowest training loss (a flow
loss that keeps falling while the pitch gets worse is a common way to lose a good model). The kept checkpoint
changes only when a newer one is better by more than ``margin``."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .composite import from_summary


def score_of(result: dict, system: str = "tristream") -> float:
    return from_summary(result.get("summary", {}), system)


def choose(candidates: list[tuple[str, dict]], current: str | None = None, margin: float = 0.01,
           system: str = "tristream") -> tuple[str | None, dict]:
    """``candidates``: ``[(checkpoint name, benchmark result)]``. Returns the name to keep and every score."""
    scores = {name: score_of(result, system) for name, result in candidates}
    finite = {k: v for k, v in scores.items() if not math.isnan(v)}
    if not finite:
        return current, scores
    best = max(finite, key=finite.get)
    if current in finite and finite[best] - finite[current] <= margin:
        return current, scores
    return best, scores


def record(path: str | Path, name: str, scores: dict) -> None:
    Path(path).write_text(json.dumps({"kept": name, "scores": scores}, indent=1))


__all__ = ["choose", "record", "score_of"]
