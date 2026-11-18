# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Note-to-note transitions for a guide pitch curve. Between two slurred notes the pitch moves on a smooth S-curve
over ``glide_s`` centred on the boundary, and a leap upward gets a small preparation (a dip before the rise) and a
leap downward a small overshoot, both measured features of sung transitions. A step-shaped guide is what makes
synthetic singing sound like a keyboard.
"""

from __future__ import annotations

import math

import torch


def midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69.0) / 12.0)


def curve(notes, seconds: float, frame_rate: float, glide_s: float = 0.08, leap: float = 3.0,
          prepare_cents: float = 25.0) -> torch.Tensor:
    """F0 in Hz per frame from ``(onset_s, offset_s, midi)`` notes; rests are 0."""
    n = int(round(seconds * frame_rate))
    cents = torch.full((n,), float("nan"), dtype=torch.float64)
    for onset, offset, midi in notes:
        a, b = int(round(onset * frame_rate)), min(n, int(round(offset * frame_rate)))
        cents[a:b] = 100.0 * midi
    half = max(1, int(round(glide_s * frame_rate / 2)))
    for (_, e0, m0), (s1, _, m1) in zip(notes, notes[1:], strict=False):
        if abs(s1 - e0) > 1e-6:
            continue                                   # a rest between them: no glide
        c = int(round(s1 * frame_rate))
        a, b = max(0, c - half), min(n, c + half)
        t = torch.linspace(-3, 3, b - a, dtype=torch.float64)
        s = 1 / (1 + torch.exp(-t))
        jump = 100.0 * (m1 - m0)
        shape = 100.0 * m0 + jump * s
        if abs(m1 - m0) >= leap:                       # a leap: preparation against its direction
            shape = shape - math.copysign(prepare_cents, jump) * torch.exp(-((t + 1.5) ** 2))
        cents[a:b] = shape
    hz = torch.where(torch.isnan(cents), torch.zeros_like(cents), 440.0 * 2 ** ((cents / 100.0 - 69.0) / 12.0))
    return hz


__all__ = ["curve", "midi_hz"]
