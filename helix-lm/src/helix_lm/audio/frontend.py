# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Waveform in, frames out, and back: a causal SEANet-style encoder/decoder pair (EnCodec 2022, DAC 2023, Mimi
2024) that never computes a spectrogram. The encoder is strided convolutions with residual blocks between them;
the decoder mirrors it with transposed convolutions. All convolutions are causal (left padding only), so the
encoder can run on a live stream and a frame never depends on audio that comes after it.

Why these numbers: at 24 kHz the strides ``(4, 5, 6, 8)`` make 960 samples a frame, 25 frames a second. That is
twice Mimi's 12.5 Hz on purpose: an 80 ms frame is longer than a plosive burst and half a vibrato cycle, so a
12.5 Hz tokenizer has to average them away ("blur"); 40 ms keeps note onsets and vibrato resolvable and only
doubles the token count. The Snake activation (``x + sin²(αx)/α``, BigVGAN/DAC) is what lets a small decoder
produce clean periodic signals, which is most of what speech and singing are.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class Snake(nn.Module):
    """``x + (1/α) sin²(αx)`` with a learned α per channel: a periodic inductive bias for audio."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.clamp(min=1e-3)
        return x + torch.sin(alpha * x).pow(2) / alpha


class CausalConv1d(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.stride = stride
        self.conv = nn.Conv1d(cin, cout, kernel, stride=stride, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left padding only keeps causality; with (kernel - 1) * dilation on the left the output has exactly
        # ceil(T / stride) steps.
        return self.conv(F.pad(x, (self.pad, 0)))


class CausalConvTranspose1d(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.stride, self.kernel = stride, kernel
        self.conv = nn.ConvTranspose1d(cin, cout, kernel, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        trim = self.kernel - self.stride  # drop the non-causal tail
        return y[..., : y.shape[-1] - trim] if trim > 0 else y


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Snake(channels),
            CausalConv1d(channels, channels, 7, dilation=dilation),
            Snake(channels),
            CausalConv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class WaveEncoder(nn.Module):
    """``(B, 1, T)`` waveform -> ``(B, dim, ceil(T / hop))`` frames; ``hop = prod(strides)``."""

    def __init__(self, dim: int = 256, channels: int = 32, strides: tuple[int, ...] = (4, 5, 6, 8)) -> None:
        super().__init__()
        self.strides, self.hop = tuple(strides), math.prod(strides)
        layers: list[nn.Module] = [CausalConv1d(1, channels, 7)]
        c = channels
        for stride in strides:
            layers += [
                ResidualUnit(c, 1),
                ResidualUnit(c, 3),
                ResidualUnit(c, 9),
                Snake(c),
                CausalConv1d(c, 2 * c, 2 * stride, stride=stride),
            ]
            c *= 2
        layers += [Snake(c), CausalConv1d(c, dim, 3)]
        self.net = nn.Sequential(*layers)
        self.dim, self.channels = dim, c

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.dim() == 2:
            wave = wave[:, None]
        return self.net(wave)


class WaveDecoder(nn.Module):
    """``(B, dim, frames)`` -> ``(B, 1, frames * hop)`` waveform, in ``[-1, 1]`` through a final tanh."""

    def __init__(self, dim: int = 256, channels: int = 32, strides: tuple[int, ...] = (4, 5, 6, 8)) -> None:
        super().__init__()
        self.strides, self.hop = tuple(strides), math.prod(strides)
        c = channels * 2 ** len(strides)
        layers: list[nn.Module] = [CausalConv1d(dim, c, 7)]
        for stride in reversed(strides):
            layers += [
                Snake(c),
                CausalConvTranspose1d(c, c // 2, 2 * stride, stride),
                ResidualUnit(c // 2, 1),
                ResidualUnit(c // 2, 3),
                ResidualUnit(c // 2, 9),
            ]
            c //= 2
        layers += [Snake(c), CausalConv1d(c, 1, 7)]
        self.net = nn.Sequential(*layers)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(frames))


def receptive_field_is_causal(encoder: WaveEncoder, length: int = 4800) -> bool:
    """Perturbing the waveform after sample ``t`` must leave the frames up to ``t // hop`` unchanged."""
    encoder.eval()
    with torch.no_grad():
        x = torch.randn(1, 1, length)
        y = encoder(x)
        cut = length // 2
        z = x.clone()
        z[..., cut:] += torch.randn_like(z[..., cut:])
        w = encoder(z)
    frames_before = cut // encoder.hop
    return bool(torch.allclose(y[..., : frames_before - 1], w[..., : frames_before - 1], atol=1e-5))
