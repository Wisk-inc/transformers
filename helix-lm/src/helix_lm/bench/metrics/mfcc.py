# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Mel cepstra: the log amplitude of mel bands through a DCT, scaled as cepstral coefficients are (the cosine series
of the log spectrum), so distortions land on the scale singing papers report. This is the open-source stand-in for
the SPTK mel-generalised cepstrum those papers compute MCD on: numbers are comparable inside this suite, and of the
same size as the papers', not equal to them.
"""

from __future__ import annotations

import math

import torch


def hz_to_mel(hz):
    return 2595.0 * torch.log10(1.0 + torch.as_tensor(hz, dtype=torch.float64) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10 ** (torch.as_tensor(mel, dtype=torch.float64) / 2595.0) - 1.0)


def mel_filters(sample_rate: int, n_fft: int, n_mels: int, f_min: float = 0.0, f_max: float | None = None):
    """``(n_mels, n_fft // 2 + 1)`` triangular filters on the HTK mel scale."""
    f_max = f_max or sample_rate / 2
    bins = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1, dtype=torch.float64)
    points = mel_to_hz(torch.linspace(float(hz_to_mel(f_min)), float(hz_to_mel(f_max)), n_mels + 2))
    lower, centre, upper = points[:-2, None], points[1:-1, None], points[2:, None]
    up = (bins[None] - lower) / (centre - lower).clamp(min=1e-9)
    down = (upper - bins[None]) / (upper - centre).clamp(min=1e-9)
    return torch.clamp(torch.minimum(up, down), min=0.0).float()


def dct_matrix(n_in: int, n_out: int) -> torch.Tensor:
    """Orthonormal DCT-II as an ``(n_in, n_out)`` matrix."""
    k = torch.arange(n_out, dtype=torch.float64)[None]
    n = torch.arange(n_in, dtype=torch.float64)[:, None]
    m = torch.cos(math.pi / n_in * (n + 0.5) * k) * math.sqrt(2.0 / n_in)
    m[:, 0] /= math.sqrt(2.0)
    return m.float()


def mel_cepstrum(wave: torch.Tensor, sample_rate: int, n_coeffs: int = 25, n_fft: int = 1024, hop: int | None = None,
                 n_mels: int = 80, floor_db: float | None = None) -> torch.Tensor:
    """``(frames, n_coeffs)`` cepstra of a mono signal, ``c0`` (the level) first. ``hop`` defaults to 5 ms. With
    ``floor_db`` each frame's bands are held at least that far under its loudest band, so bands with next to no
    energy (a dark voice above 5 kHz) do not swing the cepstra on their rounding noise."""
    hop = hop or max(1, int(sample_rate * 0.005))
    window = torch.hann_window(n_fft, dtype=wave.dtype, device=wave.device)
    spec = torch.stft(wave.flatten(), n_fft, hop, window=window, return_complex=True).abs().pow(2)   # (bins, T)
    mel = mel_filters(sample_rate, n_fft, n_mels).to(spec.device) @ spec
    if floor_db is not None:
        mel = torch.maximum(mel, mel.amax(0, keepdim=True) * 10.0 ** (-floor_db / 10.0))
    log_amplitude = 0.5 * torch.log(mel.clamp(min=1e-10))
    return (log_amplitude.T @ dct_matrix(n_mels, n_coeffs).to(spec.device) / math.sqrt(2 * n_mels)).float()


__all__ = ["dct_matrix", "hz_to_mel", "mel_cepstrum", "mel_filters", "mel_to_hz"]
