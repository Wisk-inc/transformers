# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Jitter and shimmer, frame-based: the mean relative change of the pitch period and of the peak amplitude between
consecutive 10 ms frames of a voiced stretch. These are estimates from a pitch track, not Praat's period-mark
values; they are compared between a generated take and its reference, where the method's bias cancels.
"""

from __future__ import annotations

import math

import torch

from ...audio.singing import estimate_f0


def _runs(mask: torch.Tensor) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(mask.tolist() + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= 3:
                out.append((start, i))
            start = None
    return out


def jitter_shimmer(wave: torch.Tensor, sample_rate: int, frame_rate: float = 100.0) -> dict:
    wave = wave.flatten().float()
    f0 = estimate_f0(wave[None], sample_rate, frame_rate)[0].double()
    hop = int(round(sample_rate / frame_rate))
    peaks = wave[: f0.numel() * hop].reshape(-1, hop).abs().max(-1).values.double() if wave.numel() >= hop else None
    jit, shim = [], []
    for a, b in _runs(f0 > 0):
        period = 1.0 / f0[a:b]
        jit.append(float((period[1:] - period[:-1]).abs().mean() / period.mean()))
        if peaks is not None and b <= peaks.numel():
            amp = peaks[a:b]
            shim.append(float((amp[1:] - amp[:-1]).abs().mean() / amp.mean().clamp(min=1e-9)))
    return {"jitter": sum(jit) / len(jit) if jit else math.nan, "shimmer": sum(shim) / len(shim) if shim else math.nan}


def jitter_shimmer_metrics(pred: torch.Tensor, ref: torch.Tensor, sample_rate: int) -> dict:
    p, r = jitter_shimmer(pred, sample_rate), jitter_shimmer(ref, sample_rate)

    def gap(k):
        return abs(p[k] - r[k]) if not (math.isnan(p[k]) or math.isnan(r[k])) else math.nan

    return {"jitter_error": gap("jitter"), "shimmer_error": gap("shimmer")}


__all__ = ["jitter_shimmer", "jitter_shimmer_metrics"]
