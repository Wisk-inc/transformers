# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Technique: a take converted into its own voice, and whether what makes the technique survived: breathiness
(HNR), vibrato, brightness (tilt), roughness (jitter and shimmer). Rows carry the take's technique, so the report
can say which techniques a system flattens."""

from __future__ import annotations

from ..metrics.jitter_shimmer import jitter_shimmer_metrics
from ..metrics.periodicity import hnr_metrics
from ..metrics.tilt import tilt_metrics
from ..metrics.vibrato import vibrato_metrics
from . import F0_RATE, f0, load_item


def run(system, items, sample_rate: int, cache, seconds: float = 6.0) -> list[dict]:
    rows = []
    for it in items:
        source = load_item(it, sample_rate, seconds)
        out = cache.make(system.name, "technique", it.id, {"sr": sample_rate, "s": seconds},
                         lambda s=source: system.convert(s, s, sample_rate))
        n = min(out.numel(), source.numel())
        o, s = out[:n], source[:n]
        row = {"item": it.id, "set": it.set, "singer": it.singer, "technique": it.technique or "unknown"}
        row.update({"hnr_error_db": hnr_metrics(o, s, sample_rate)["hnr_error_db"]})
        row.update(vibrato_metrics(f0(o, sample_rate), f0(s, sample_rate), F0_RATE))
        row.update({"tilt_error_db": tilt_metrics(o, s, sample_rate)["tilt_error_db"]})
        row.update(jitter_shimmer_metrics(o, s, sample_rate))
        rows.append(row)
    return rows


__all__ = ["run"]
