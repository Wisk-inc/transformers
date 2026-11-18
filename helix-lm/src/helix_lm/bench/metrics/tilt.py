# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Spectral tilt: the slope, in dB per octave, of the long-term average spectrum from 100 Hz to 8 kHz. Darker or
brighter than the reference is a difference listeners hear first."""

from __future__ import annotations

import math

import torch


def spectral_tilt(wave: torch.Tensor, sample_rate: int, lo: float = 100.0, hi: float = 8000.0) -> float:
    n_fft = 2048
    window = torch.hann_window(n_fft, dtype=torch.float64)
    spec = torch.stft(wave.flatten().double(), n_fft, n_fft // 4, window=window, return_complex=True).abs().pow(2)
    ltas = spec.mean(-1)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate).double()
    sel = (freqs >= lo) & (freqs <= min(hi, sample_rate / 2 - 1)) & (ltas > 0)
    if int(sel.sum()) < 4:
        return math.nan
    x, y = torch.log2(freqs[sel]), 10 * torch.log10(ltas[sel])
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def tilt_metrics(pred: torch.Tensor, ref: torch.Tensor, sample_rate: int) -> dict:
    p, r = spectral_tilt(pred, sample_rate), spectral_tilt(ref, sample_rate)
    return {"tilt_db_per_octave": p, "tilt_error_db": abs(p - r) if not (math.isnan(p) or math.isnan(r)) else math.nan}


__all__ = ["spectral_tilt", "tilt_metrics"]
