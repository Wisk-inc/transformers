# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Resynthesis: a take through the system and back (for TriStream, the codec's round trip, which every output of
the model passes through). Measured against the take itself, frame for frame: MCD, pitch, voicing, spectral
distances, SI-SNR, loudness contour and glitches."""

from __future__ import annotations

from ..metrics.mcd import mcd
from ..metrics.spectral import spectral_metrics
from . import content, level, load_item, output_quality


def run(system, items, sample_rate: int, cache, seconds: float = 8.0) -> list[dict]:
    rows = []
    for it in items:
        ref = load_item(it, sample_rate, seconds)
        out = cache.make(system.name, "resynthesis", it.id, {"sr": sample_rate, "s": seconds},
                         lambda ref=ref: system.resynthesize(ref, sample_rate))
        n = min(out.numel(), ref.numel())
        o, r = out[:n], ref[:n]
        row = {"item": it.id, "set": it.set, "singer": it.singer}
        row.update(content(o, r, sample_rate, it.notes))
        row["mcd_db"] = mcd(level(o, sample_rate), level(r, sample_rate), sample_rate, align=False)
        row.update(spectral_metrics(level(o, sample_rate), level(r, sample_rate)))
        row.update(output_quality(o, sample_rate))
        rows.append(row)
    return rows


__all__ = ["run"]
