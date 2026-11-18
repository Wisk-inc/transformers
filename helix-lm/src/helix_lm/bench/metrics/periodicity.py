# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Harmonics-to-noise ratio from the normalised autocorrelation at the pitch period (Boersma, 1993):
``HNR = 10 log10(r / (1 - r))`` on voiced frames. A breathy or buzzy output moves it away from the reference's."""

from __future__ import annotations

import math

import torch

from ...audio.singing import estimate_f0


def hnr_track(wave: torch.Tensor, sample_rate: int, frame_rate: float = 100.0) -> torch.Tensor:
    """HNR in dB per frame, NaN where unvoiced."""
    wave = wave.flatten().float()
    f0 = estimate_f0(wave[None], sample_rate, frame_rate)[0]
    hop = int(round(sample_rate / frame_rate))
    window = int(0.04 * sample_rate)
    padded = torch.nn.functional.pad(wave, (0, window))
    frames = padded.unfold(0, window, hop)[: f0.numel()]
    frames = frames - frames.mean(-1, keepdim=True)
    out = torch.full((f0.numel(),), float("nan"))
    for i in torch.nonzero(f0 > 0).flatten().tolist():
        lag = int(round(sample_rate / float(f0[i])))
        x = frames[i]
        if lag >= x.numel() - 1:
            continue
        r = float((x[:-lag] * x[lag:]).sum() / ((x[:-lag].pow(2).sum() * x[lag:].pow(2).sum()).sqrt() + 1e-12))
        r = min(max(r, 1e-4), 1 - 1e-4)
        out[i] = 10 * math.log10(r / (1 - r))
    return out


def hnr_db(wave: torch.Tensor, sample_rate: int) -> float:
    t = hnr_track(wave, sample_rate)
    t = t[~torch.isnan(t)]
    return float(t.median()) if t.numel() else math.nan


def hnr_metrics(pred: torch.Tensor, ref: torch.Tensor, sample_rate: int) -> dict:
    p, r = hnr_db(pred, sample_rate), hnr_db(ref, sample_rate)
    gap = abs(p - r) if not (math.isnan(p) or math.isnan(r)) else math.nan
    return {"hnr_db": p, "hnr_ref_db": r, "hnr_error_db": gap}


__all__ = ["hnr_db", "hnr_metrics", "hnr_track"]
