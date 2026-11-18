# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Griffin-Lim: the magnitude spectrogram kept, the phase found by iteration (Griffin and Lim, 1984). What a
spectrogram model that gets the magnitudes exactly right but has no phase would sound like."""

from __future__ import annotations

import torch


class GriffinLim:
    name = "griffin-lim"
    supports = ("resynthesis", "speed")

    def __init__(self, n_fft: int = 1024, hop: int = 256, iterations: int = 32, momentum: float = 0.99) -> None:
        self.n_fft, self.hop, self.iterations, self.momentum = n_fft, hop, iterations, momentum

    def resynthesize(self, wave: torch.Tensor, sample_rate: int) -> torch.Tensor:
        window = torch.hann_window(self.n_fft)
        x = wave.flatten().float()
        mag = torch.stft(x, self.n_fft, self.hop, window=window, return_complex=True).abs()
        g = torch.Generator().manual_seed(0)
        angles = torch.exp(2j * torch.pi * torch.rand(mag.shape, generator=g))
        previous = torch.zeros_like(angles)
        out = x
        for _ in range(self.iterations):                   # fast Griffin-Lim (Perraudin et al., 2013)
            out = torch.istft(mag * angles, self.n_fft, self.hop, window=window, length=x.numel())
            rebuilt = torch.stft(out, self.n_fft, self.hop, window=window, return_complex=True)
            accelerated = rebuilt - self.momentum / (1 + self.momentum) * previous
            previous = rebuilt
            angles = accelerated / accelerated.abs().clamp(min=1e-8)
        return torch.istft(mag * angles, self.n_fft, self.hop, window=window, length=x.numel())


__all__ = ["GriffinLim"]
