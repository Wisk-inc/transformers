# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The spectral envelope: the log magnitude with its cepstrum cut after ``lifter`` coefficients, which keeps the
formants and the tilt and drops the harmonics. Matching envelopes keeps vowels and brightness right without asking
the harmonics to line up, which a pitch error of a few cents already breaks."""

from __future__ import annotations

import torch


def log_envelope(x: torch.Tensor, n_fft: int = 1024, hop: int = 256, lifter: int = 30) -> torch.Tensor:
    """``(..., frames, bins)`` differentiable log-amplitude envelope of ``(..., samples)``."""
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(x.reshape(-1, x.shape[-1]), n_fft, hop, window=window, return_complex=True).abs()
    log = torch.log(spec.clamp(min=1e-6)).transpose(-1, -2)                  # (B, frames, bins)
    cep = torch.fft.irfft(log, n=n_fft)
    keep = torch.zeros(n_fft, device=x.device, dtype=cep.dtype)
    keep[:lifter] = 1
    keep[n_fft - lifter + 1 :] = 1
    return torch.fft.rfft(cep * keep, n=n_fft).real


def envelope_loss(pred: torch.Tensor, target: torch.Tensor, lifter: int = 30, floor_db: float = 50.0) -> torch.Tensor:
    """Mean absolute envelope difference over the frames where the target is within ``floor_db`` of its loudest."""
    p, t = log_envelope(pred, lifter=lifter), log_envelope(target, lifter=lifter)
    level = t.mean(-1)
    keep = (level > level.amax(-1, keepdim=True) - floor_db / 8.686).float()[..., None]
    return ((p - t).abs() * keep).sum() / (keep.sum() * p.shape[-1]).clamp(min=1.0)


__all__ = ["envelope_loss", "log_envelope"]
