# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Integrated loudness after ITU-R BS.1770-4 (K-weighting, 400 ms blocks with 75% overlap, an absolute gate at
-70 LUFS and a relative gate 10 LU below), for mono signals at any rate.

Benchmarks compare outputs at one loudness: a system that sings 3 dB louder than the reference is not better for
it, and several metrics (spectral distances, MCD on unnormalised cepstra) would say it is worse. The filters are
the two biquads of the standard, designed for the signal's own rate (the constants that give the published 48 kHz
coefficients) and applied through their magnitude response, which is all that an energy measure needs.
"""

from __future__ import annotations

import math

import torch

SHELF = (3.999843853973347, 0.7071752369554196, 1681.974450955533)     # gain dB, Q, centre Hz
HIGH_PASS = (0.5003270373238773, 38.13547087602444)                    # Q, centre Hz


def _biquads(sample_rate: int) -> list[tuple[list[float], list[float]]]:
    """The two K-weighting biquads for ``sample_rate``, normalised so ``a[0] = 1``. This parametrisation (De Man,
    2014) gives the coefficients printed in BS.1770 at 48 kHz."""
    gain, q, fc = SHELF
    k = math.tan(math.pi * fc / sample_rate)
    vh = 10 ** (gain / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf_b = [(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0]
    shelf_a = [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    q, fc = HIGH_PASS
    k = math.tan(math.pi * fc / sample_rate)
    a0 = 1 + k / q + k * k
    hp_b = [1.0, -2.0, 1.0]
    hp_a = [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    return [(shelf_b, shelf_a), (hp_b, hp_a)]


def k_weight(wave: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """The signal through the K-weighting magnitude response."""
    n = wave.shape[-1]
    spec = torch.fft.rfft(wave.double(), dim=-1)
    w = torch.linspace(0, math.pi, spec.shape[-1], dtype=torch.float64)
    z1, z2 = torch.exp(-1j * w), torch.exp(-2j * w)
    gain = torch.ones_like(w)
    for b, a in _biquads(sample_rate):
        h = (b[0] + b[1] * z1 + b[2] * z2) / (a[0] + a[1] * z1 + a[2] * z2)
        gain = gain * h.abs()
    return torch.fft.irfft(spec * gain, n=n, dim=-1).to(wave.dtype)


def integrated_loudness(wave: torch.Tensor, sample_rate: int) -> float:
    """LUFS of a mono signal; ``-inf`` for silence or a signal shorter than one 400 ms block."""
    y = k_weight(wave.flatten().double(), sample_rate)
    block, step = int(0.4 * sample_rate), int(0.1 * sample_rate)
    if y.numel() < block:
        return float("-inf")
    z = y.unfold(0, block, step).pow(2).mean(-1)
    level = -0.691 + 10 * torch.log10(z + 1e-20)
    gated = z[level > -70.0]
    if gated.numel() == 0:
        return float("-inf")
    relative = -0.691 + 10 * math.log10(float(gated.mean())) - 10.0
    gated = z[(level > -70.0) & (level > relative)]
    if gated.numel() == 0:
        return float("-inf")
    return -0.691 + 10 * math.log10(float(gated.mean()))


def normalize(wave: torch.Tensor, sample_rate: int, target: float = -23.0, peak: float = 0.99) -> torch.Tensor:
    """The signal at ``target`` LUFS (EBU R 128's -23 by default), never clipping past ``peak``."""
    now = integrated_loudness(wave, sample_rate)
    if not math.isfinite(now):
        return wave
    out = wave * 10 ** ((target - now) / 20)
    top = float(out.abs().max())
    return out * (peak / top) if top > peak else out


__all__ = ["integrated_loudness", "k_weight", "normalize"]
