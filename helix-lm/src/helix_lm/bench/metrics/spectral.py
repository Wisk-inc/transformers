# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Spectral distances between a generated and a reference signal of the same length."""

from __future__ import annotations

import torch

RESOLUTIONS = ((512, 128), (1024, 256), (2048, 512))


def _mag(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, dtype=x.dtype, device=x.device)
    return torch.stft(x.flatten(), n_fft, hop, window=window, return_complex=True).abs()


def spectral_convergence(pred: torch.Tensor, ref: torch.Tensor, n_fft: int = 1024, hop: int = 256) -> float:
    p, r = _mag(pred, n_fft, hop), _mag(ref, n_fft, hop)
    n = min(p.shape[-1], r.shape[-1])
    p, r = p[..., :n], r[..., :n]
    return float((r - p).norm() / r.norm().clamp(min=1e-9))


def mrstft_distance(pred: torch.Tensor, ref: torch.Tensor, resolutions=RESOLUTIONS) -> float:
    """Mean over resolutions of spectral convergence plus the L1 distance of log magnitudes."""
    total = 0.0
    for n_fft, hop in resolutions:
        p, r = _mag(pred, n_fft, hop), _mag(ref, n_fft, hop)
        n = min(p.shape[-1], r.shape[-1])
        p, r = p[..., :n], r[..., :n]
        sc = (r - p).norm() / r.norm().clamp(min=1e-9)
        mag = (torch.log(p.clamp(min=1e-7)) - torch.log(r.clamp(min=1e-7))).abs().mean()
        total += float(sc + mag)
    return total / len(resolutions)


def log_spectral_distance(pred: torch.Tensor, ref: torch.Tensor, n_fft: int = 2048, hop: int = 512) -> float:
    """Root mean square over frequency of the dB difference, averaged over frames."""
    p, r = _mag(pred, n_fft, hop).pow(2), _mag(ref, n_fft, hop).pow(2)
    n = min(p.shape[-1], r.shape[-1])
    diff = 10 * torch.log10(p[..., :n].clamp(min=1e-10)) - 10 * torch.log10(r[..., :n].clamp(min=1e-10))
    return float(diff.pow(2).mean(0).sqrt().mean())


def si_snr(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """Scale-invariant SNR in dB (only meaningful when the timing matches, as in codec round trips)."""
    n = min(pred.numel(), ref.numel())
    p, r = pred.flatten()[:n].double(), ref.flatten()[:n].double()
    p, r = p - p.mean(), r - r.mean()
    target = (p @ r) / (r @ r).clamp(min=1e-12) * r
    noise = p - target
    return float(10 * torch.log10((target @ target).clamp(min=1e-12) / (noise @ noise).clamp(min=1e-12)))


def spectral_metrics(pred: torch.Tensor, ref: torch.Tensor) -> dict:
    return {"mrstft": mrstft_distance(pred, ref), "lsd_db": log_spectral_distance(pred, ref),
            "spectral_convergence": spectral_convergence(pred, ref), "si_snr_db": si_snr(pred, ref)}


__all__ = ["RESOLUTIONS", "log_spectral_distance", "mrstft_distance", "si_snr", "spectral_convergence",
           "spectral_metrics"]
