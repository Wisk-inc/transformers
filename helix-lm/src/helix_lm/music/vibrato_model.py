# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Vibrato as singers do it: none on short notes, and on long ones a delayed onset, a rate of 5 to 7 Hz and a depth
that fades in. :func:`apply` adds it to a pitch curve that has none (a guide from a score, a flat converted note);
:func:`match` measures a reference's vibrato and gives the settings that reproduce it, so a conversion keeps the
singer's. Measured by the benchmark's vibrato metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class Vibrato:
    rate_hz: float = 5.8
    depth_cents: float = 35.0
    delay_s: float = 0.3
    fade_s: float = 0.25
    min_note_s: float = 0.45


def apply(f0: torch.Tensor, notes, frame_rate: float, v: Vibrato | None = None, seed: int = 0) -> torch.Tensor:
    """Vibrato on every note at least ``min_note_s`` long; the rate wanders by up to 4% from note to note."""
    v = v or Vibrato()
    out = f0.clone().double()
    g = torch.Generator().manual_seed(seed)
    for onset, offset, _ in notes:
        if offset - onset < v.min_note_s:
            continue
        a, b = int(round(onset * frame_rate)), min(out.numel(), int(round(offset * frame_rate)))
        t = torch.arange(b - a, dtype=torch.float64) / frame_rate
        rate = v.rate_hz * (1 + 0.04 * (2 * float(torch.rand((), generator=g)) - 1))
        depth = v.depth_cents * ((t - v.delay_s) / max(v.fade_s, 1e-3)).clamp(0, 1)
        cents = depth * torch.sin(2 * math.pi * rate * (t - v.delay_s).clamp(min=0))
        seg = out[a:b]
        out[a:b] = torch.where(seg > 0, seg * 2 ** (cents / 1200), seg)
    return out.to(f0.dtype)


def match(reference_f0: torch.Tensor, frame_rate: float) -> Vibrato | None:
    """The settings of the reference's vibrato, or None when it has none."""
    from ..bench.metrics.vibrato import vibrato

    found = vibrato(reference_f0, frame_rate)
    if not found["present"] or math.isnan(found["rate_hz"]):
        return None
    return Vibrato(rate_hz=found["rate_hz"], depth_cents=found["extent_cents"])


__all__ = ["Vibrato", "apply", "match"]
