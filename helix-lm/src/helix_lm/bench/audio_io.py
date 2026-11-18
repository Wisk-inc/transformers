# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Audio as the benchmarks take it: mono float32 at one rate, silence trimmed, two signals cut to one length."""

from __future__ import annotations

from pathlib import Path

import torch


def to_mono(wave: torch.Tensor) -> torch.Tensor:
    """``(samples,)`` from ``(samples,)``, ``(channels, samples)`` or ``(1, channels, samples)``."""
    wave = wave.float()
    while wave.dim() > 1:
        wave = wave.mean(0)
    return wave


def resample(wave: torch.Tensor, rate_from: int, rate_to: int) -> torch.Tensor:
    """Band-limited resampling through the spectrum: exact for whole-signal benchmarks, no extra dependency."""
    if rate_from == rate_to:
        return wave
    n = wave.shape[-1]
    m = max(1, int(round(n * rate_to / rate_from)))
    spec = torch.fft.rfft(wave, dim=-1)
    keep = min(spec.shape[-1], m // 2 + 1)
    out = torch.zeros(*spec.shape[:-1], m // 2 + 1, dtype=spec.dtype)
    out[..., :keep] = spec[..., :keep]
    return torch.fft.irfft(out, n=m, dim=-1) * (m / n)


def load(path: str | Path, sample_rate: int = 24000) -> torch.Tensor:
    """A file at ``sample_rate``, mono (the package's reader: soundfile when installed, plain WAV otherwise)."""
    from ..audio.data import read_wave

    return to_mono(read_wave(path, sample_rate))


def trim_silence(wave: torch.Tensor, sample_rate: int, threshold_db: float = -50.0, frame_ms: float = 20.0):
    """Leading and trailing frames quieter than ``threshold_db`` below the loudest frame are cut."""
    hop = max(1, int(sample_rate * frame_ms / 1000))
    frames = wave[: wave.shape[-1] // hop * hop].reshape(-1, hop)
    if frames.shape[0] == 0:
        return wave
    level = 10 * torch.log10(frames.pow(2).mean(-1) + 1e-12)
    loud = (level > level.max() + threshold_db).nonzero().flatten()
    if loud.numel() == 0:
        return wave[:0]
    return wave[int(loud[0]) * hop : (int(loud[-1]) + 1) * hop]


def match_length(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


def peak_normalize(wave: torch.Tensor, peak: float = 0.95) -> torch.Tensor:
    top = wave.abs().max()
    return wave if top <= 0 else wave * (peak / top)


__all__ = ["load", "match_length", "peak_normalize", "resample", "to_mono", "trim_silence"]
