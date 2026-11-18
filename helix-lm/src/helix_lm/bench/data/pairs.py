# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Conversion pairs: each source take with a reference of another singer, chosen by a seeded generator so two runs
(and two systems) are measured on the same pairs. A pair never uses the source singer as the target."""

from __future__ import annotations

import random

from .registry import Item


def conversion_pairs(items: list[Item], per_source: int = 1, seed: int = 0, limit: int | None = None) -> list[tuple]:
    by_singer: dict = {}
    for it in items:
        by_singer.setdefault(it.singer, []).append(it)
    singers = sorted(by_singer)
    if len(singers) < 2:
        return []
    rng = random.Random(seed)
    out = []
    for src in sorted(items, key=lambda i: i.id):
        others = [s for s in singers if s != src.singer]
        for target in rng.sample(others, min(per_source, len(others))):
            ref = sorted(by_singer[target], key=lambda i: i.id)[rng.randrange(len(by_singer[target]))]
            out.append((src, ref))
    return out[:limit] if limit else out


__all__ = ["conversion_pairs"]
