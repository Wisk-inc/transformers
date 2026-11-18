# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The multi-resolution STFT loss (Yamamoto et al., 2020): spectral convergence plus log-magnitude L1 at three
resolutions. One short window sees the consonants, one long window the harmonics; the benchmark's MR-STFT distance
is the same quantity, measured."""

from __future__ import annotations

import torch

RESOLUTIONS = ((512, 128, 512), (1024, 256, 1024), (2048, 512, 2048))


def _mag(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    window = torch.hann_window(win, device=x.device, dtype=x.dtype)
    spec = torch.stft(x.reshape(-1, x.shape[-1]), n_fft, hop, win, window=window, return_complex=True)
    return spec.abs().clamp(min=1e-7)


def mrstft_loss(pred: torch.Tensor, target: torch.Tensor, resolutions=RESOLUTIONS) -> torch.Tensor:
    """Differentiable; ``pred`` and ``target`` are ``(..., samples)`` of the same length."""
    total = pred.new_zeros(())
    for n_fft, hop, win in resolutions:
        p, t = _mag(pred, n_fft, hop, win), _mag(target, n_fft, hop, win)
        sc = torch.linalg.norm(t - p, dim=(-2, -1)) / torch.linalg.norm(t, dim=(-2, -1)).clamp(min=1e-7)
        total = total + sc.mean() + (p.log() - t.log()).abs().mean()
    return total / len(resolutions)


__all__ = ["RESOLUTIONS", "mrstft_loss"]
