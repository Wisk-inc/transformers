# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""High register: takes that reach above their singer's comfortable top, converted into their own voice, and the
pops at register crossings counted (:func:`helix_lm.music.register.register_break_report`). The comfortable top is
taken as 85% of the highest pitch the singer reaches across the set."""

from __future__ import annotations

from ...music.register import register_break_report
from . import F0_RATE, f0, load_item


def run(system, items, sample_rate: int, cache, seconds: float = 8.0) -> list[dict]:
    tops: dict = {}
    tracks = {}
    for it in items:
        source = load_item(it, sample_rate, seconds)
        track = f0(source, sample_rate)
        tracks[it.id] = (source, track)
        voiced = track[track > 0]
        if voiced.numel():
            tops[it.singer] = max(tops.get(it.singer, 0.0), float(voiced.quantile(0.98)))
    rows = []
    for it in items:
        source, track = tracks[it.id]
        top = 0.85 * tops.get(it.singer, 0.0)
        if top <= 0 or not bool((track > top).any()):
            continue
        out = cache.make(system.name, "register", it.id, {"sr": sample_rate, "s": seconds},
                         lambda s=source: system.convert(s, s, sample_rate))
        report = register_break_report(out, f0(out, sample_rate), sample_rate, top, F0_RATE)
        rows.append({"item": it.id, "set": it.set, "singer": it.singer,
                     "register_pops_per_minute": report["pops_per_minute"],
                     "crossings": int(report.get("crossings", 0))})       # a count, not a list
    return rows


__all__ = ["run"]
