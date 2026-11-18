# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Long takes: a singer's takes joined into one long performance, converted in one go, and cut into 10-second
pieces. A drift is the slope, per minute, of each piece's pitch error against the source and of its voice distance
from the target: a long conversion that wanders off key or off voice shows it here and nowhere else."""

from __future__ import annotations

import math

import torch

from ..metrics.pitch import f0_rmse_cents
from ..metrics.similarity import speaker_similarity
from . import f0, load_item


def _slope(xs: list[float], ys: list[float]) -> float:
    pts = [(x, y) for x, y in zip(xs, ys, strict=True) if not math.isnan(y)]
    if len(pts) < 2:
        return math.nan
    x = torch.tensor([p[0] for p in pts], dtype=torch.float64)
    y = torch.tensor([p[1] for p in pts], dtype=torch.float64)
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum().clamp(min=1e-12))


def run(system, items, sample_rate: int, cache, minutes: float = 1.0, piece_s: float = 10.0, embed=None) -> list[dict]:
    rows = []
    singers = sorted({i.singer for i in items})
    for singer in singers:
        takes = [i for i in items if i.singer == singer]
        others = [i for i in items if i.singer != singer]
        if not takes or not others:
            continue
        parts, total = [], 0
        while total < minutes * 60 * sample_rate:
            for it in takes:
                w = load_item(it, sample_rate)
                parts.append(w)
                total += w.numel()
        source = torch.cat(parts)[: int(minutes * 60 * sample_rate)]
        voice = load_item(others[0], sample_rate, 8.0)
        out = cache.make(system.name, "longform", singer, {"sr": sample_rate, "min": minutes},
                         lambda s=source, v=voice: system.convert(s, v, sample_rate))
        step = int(piece_s * sample_rate)
        times, err, dist = [], [], []
        for start in range(0, min(out.numel(), source.numel()) - step + 1, step):
            o, s = out[start : start + step], source[start : start + step]
            times.append(start / sample_rate / 60.0)
            err.append(f0_rmse_cents(f0(o, sample_rate), f0(s, sample_rate)))
            dist.append(1.0 - speaker_similarity(o, voice, sample_rate, embed))
        rows.append({"item": f"long-{singer}", "set": takes[0].set, "singer": singer, "pieces": len(times),
                     "drift_cents_per_min": _slope(times, err), "timbre_drift": _slope(times, dist)})
    return rows


__all__ = ["run"]
