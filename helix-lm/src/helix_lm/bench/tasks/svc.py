# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Singing voice conversion: each source take sung again in another singer's voice. Content is measured against
the source (melody, voicing, dynamics, consonants, written notes), identity against the target and the source
(voice similarity to each and the margin between them), and the output on its own (quality proxy, glitches,
spectral tilt against the target's)."""

from __future__ import annotations

from ..data.pairs import conversion_pairs
from ..metrics.similarity import similarity_metrics
from ..metrics.tilt import tilt_metrics
from . import content, load_item, output_quality


def run(system, items, sample_rate: int, cache, seconds: float = 8.0, seed: int = 0, embed=None) -> list[dict]:
    rows = []
    for src, ref in conversion_pairs(items, per_source=1, seed=seed):
        source, voice = load_item(src, sample_rate, seconds), load_item(ref, sample_rate, seconds)
        out = cache.make(system.name, "svc", f"{src.id}__{ref.id}", {"sr": sample_rate, "s": seconds},
                         lambda s=source, v=voice: system.convert(s, v, sample_rate))
        n = min(out.numel(), source.numel())
        o, s = out[:n], source[:n]
        row = {"item": f"{src.id}->{ref.singer}", "set": src.set, "singer": src.singer, "target": ref.singer}
        row.update(content(o, s, sample_rate, src.notes))
        row.update(similarity_metrics(o, voice, sample_rate, source=s, embed=embed))
        row.update({k: v for k, v in tilt_metrics(o, voice, sample_rate).items() if k == "tilt_error_db"})
        row.update(output_quality(o, sample_rate))
        rows.append(row)
    return rows


__all__ = ["run"]
