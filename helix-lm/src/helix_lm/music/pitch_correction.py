# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Score-aware pitch correction that keeps the singing. Each note's centre (the median of its voiced frames) is moved
toward the written pitch by ``strength``; everything around the centre (vibrato, a scoop into the note, a fall at
its end) moves with it, unchanged. The first and last ``edge_s`` of a note are left as sung, so glides between
notes stay glides. At strength 1 the note-accuracy metric reads its best without the flat, robotic pitch of a
hard tuner.
"""

from __future__ import annotations

import math

import torch


def correct(f0: torch.Tensor, notes, frame_rate: float, strength: float = 0.7, edge_s: float = 0.04,
            max_cents: float = 300.0) -> torch.Tensor:
    """``f0`` (Hz per frame, 0 unvoiced) with each written note's centre pulled toward its pitch; notes sung more
    than ``max_cents`` away are left alone (a different note was sung on purpose, or the score is wrong there)."""
    out = f0.clone().double()
    edge = int(round(edge_s * frame_rate))
    for onset, offset, midi in notes:
        a, b = int(round(onset * frame_rate)), min(out.numel(), int(round(offset * frame_rate)))
        seg = out[a:b]
        voiced = seg > 0
        if int(voiced.sum()) < 3:
            continue
        target = 440.0 * 2 ** ((float(midi) - 69.0) / 12.0)
        off = 1200 * math.log2(float(seg[voiced].median()) / target)
        if abs(off) > max_cents:
            continue
        shift = 2 ** (-strength * off / 1200)
        inner = torch.zeros_like(voiced)
        inner[edge : max(edge, b - a - edge)] = True
        ramp = torch.ones(b - a, dtype=torch.float64)
        if edge:
            k = min(edge, (b - a) // 2)
            ramp[:k] = torch.linspace(0, 1, k, dtype=torch.float64)
            ramp[b - a - k :] = torch.linspace(1, 0, k, dtype=torch.float64)
        factor = shift ** ramp
        out[a:b] = torch.where(voiced, seg * factor, seg)
    return out.to(f0.dtype)


__all__ = ["correct"]
