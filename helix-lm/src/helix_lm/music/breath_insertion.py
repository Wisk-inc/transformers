# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Breaths before phrases. A synthetic or converted take that goes from silence straight to voice sounds pasted
together; a singer breathes in during any rest long enough to. A breath here is band-passed noise (most of it 1 to 6
kHz) with a swell-and-fade envelope, placed to end just before the next phrase, ``level_db`` under the voice."""

from __future__ import annotations

import torch


def breath(seconds: float, sample_rate: int, seed: int = 0) -> torch.Tensor:
    n = max(1, int(seconds * sample_rate))
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(n, generator=g)
    spec = torch.fft.rfft(noise)
    f = torch.fft.rfftfreq(n, 1.0 / sample_rate)
    band = torch.exp(-0.5 * ((torch.log(f.clamp(min=1.0)) - torch.log(torch.tensor(2500.0))) / 0.6) ** 2)
    shaped = torch.fft.irfft(spec * band, n=n)
    t = torch.linspace(0, 1, n)
    env = torch.sin(torch.pi * t.clamp(0, 1)) ** 2 * (1 - 0.5 * t)
    out = shaped * env
    return out / out.abs().max().clamp(min=1e-9)


def insert(wave: torch.Tensor, notes, sample_rate: int, min_rest_s: float = 0.25, length_s: float = 0.3,
           level_db: float = -24.0, seed: int = 0) -> torch.Tensor:
    """``wave`` with a breath in each rest of at least ``min_rest_s`` (and before the first phrase when it starts
    late enough), scaled ``level_db`` under the voice's peak."""
    out = wave.clone().float()
    peak = float(out.abs().max()) or 1.0
    gaps = []
    if notes and notes[0][0] >= min_rest_s:
        gaps.append((0.0, notes[0][0]))
    gaps += [(e0, s1) for (_, e0, _), (s1, _, _) in zip(notes, notes[1:], strict=False) if s1 - e0 >= min_rest_s]
    for k, (start, end) in enumerate(gaps):
        length = min(length_s, (end - start) * 0.8)
        b = int((end - 0.02) * sample_rate)
        a = max(0, b - int(length * sample_rate))
        if b <= a or b > out.numel():
            continue
        out[a:b] += breath((b - a) / sample_rate, sample_rate, seed + k)[: b - a] * peak * 10 ** (level_db / 20)
    return out


__all__ = ["breath", "insert"]
