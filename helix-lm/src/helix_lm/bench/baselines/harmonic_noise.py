# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
A harmonic-plus-noise vocoder: YIN pitch, a cepstrally smoothed spectral envelope every 5 ms, harmonics at the
pitch shaped by the envelope, and noise shaped by it where the frame is unvoiced or breathy.

* resynthesis: the take's own pitch and envelope;
* conversion: the source's pitch (shifted on request) and articulation, with the envelope's long-term average
  moved from the source's to the target's (an envelope transplant, the oldest voice-conversion trick);
* singing: a pitch curve from the notes (glides and vibrato as :mod:`helix_lm.bench.data.synthetic` makes them)
  through the reference's average envelope.
"""

from __future__ import annotations

import math

import torch

from ...audio.yin import yin_f0

RATE = 200.0                    # analysis frames a second
LIFTER = 40                     # cepstral coefficients kept for the envelope


def envelope(wave: torch.Tensor, sample_rate: int, n_fft: int = 1024) -> torch.Tensor:
    """``(frames, bins)`` log-amplitude spectral envelope at ``RATE`` frames a second."""
    hop = int(round(sample_rate / RATE))
    window = torch.hann_window(n_fft, dtype=torch.float64)
    spec = torch.stft(wave.flatten().double(), n_fft, hop, window=window, return_complex=True).abs().T
    log = torch.log(spec.clamp(min=1e-7))
    cep = torch.fft.irfft(log, n=n_fft)
    cep[:, LIFTER : n_fft - LIFTER + 1] = 0
    return torch.fft.rfft(cep, n=n_fft).real                       # (frames, bins)


def synthesize(f0: torch.Tensor, env: torch.Tensor, sample_rate: int, n_fft: int = 1024, breath: float = 0.05,
               seed: int = 0) -> torch.Tensor:
    """Harmonics at ``f0`` (Hz per frame, 0 = unvoiced) with amplitudes read off ``env``, plus shaped noise."""
    frames = min(f0.numel(), env.shape[0])
    f0, env = f0[:frames].double(), env[:frames]
    hop = int(round(sample_rate / RATE))
    n = frames * hop
    amp_env = env.exp()                                            # (frames, bins)
    bin_hz = sample_rate / n_fft
    up = lambda x: torch.nn.functional.interpolate(x[None, None], size=n, mode="linear", align_corners=False)[0, 0]  # noqa: E731
    f0_s = up(f0)
    voiced_s = up((f0 > 0).double())
    phase = 2 * math.pi * torch.cumsum(f0_s, 0) / sample_rate
    out = torch.zeros(n, dtype=torch.float64)
    top = int(sample_rate / 2 / max(float(f0[f0 > 0].min()) if (f0 > 0).any() else 100.0, 40.0))
    for k in range(1, min(top, 120) + 1):
        idx = (f0 * k / bin_hz).clamp(max=env.shape[1] - 1)
        lo = idx.floor().long()
        frac = idx - lo
        hi = (lo + 1).clamp(max=env.shape[1] - 1)
        a = amp_env.gather(1, lo[:, None])[:, 0] * (1 - frac) + amp_env.gather(1, hi[:, None])[:, 0] * frac
        a = torch.where((f0 > 0) & (f0 * k < sample_rate / 2 * 0.95), a, torch.zeros_like(a))
        out += up(a) * torch.sin(k * phase)
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(n, generator=g, dtype=torch.float64)
    window = torch.hann_window(n_fft, dtype=torch.float64)
    spec = torch.stft(noise, n_fft, hop, window=window, return_complex=True)
    t = min(spec.shape[1], frames)
    level = (1 - voiced_s[:: hop][:t]) + breath
    spec[:, :t] = spec[:, :t] * amp_env[:t].T * level[None]
    shaped = torch.istft(spec, n_fft, hop, window=window, length=n)
    voice = out * 2.0 / math.sqrt(n_fft) + shaped
    return (voice / voice.abs().max().clamp(min=1e-9) * 0.5).float()


class HarmonicNoise:
    name = "harmonic-noise"
    supports = ("resynthesis", "svc", "svs", "pitch_control", "technique", "robustness", "longform", "speed",
                "register")

    def resynthesize(self, wave: torch.Tensor, sample_rate: int) -> torch.Tensor:
        f0 = yin_f0(wave.flatten(), sample_rate, RATE)
        out = synthesize(f0, envelope(wave, sample_rate), sample_rate)
        return _fit(out, wave)

    def convert(self, source: torch.Tensor, reference: torch.Tensor, sample_rate: int, shift_semitones: float = 0.0):
        f0 = yin_f0(source.flatten(), sample_rate, RATE) * 2 ** (shift_semitones / 12.0)
        env_src, env_ref = envelope(source, sample_rate), envelope(reference, sample_rate)
        moved = env_src - _loud_mean(env_src) + _loud_mean(env_ref)
        return _fit(synthesize(f0, moved, sample_rate), source)

    def sing(self, notes, sample_rate: int, reference: torch.Tensor, seconds: float | None = None) -> torch.Tensor:
        from ..data.synthetic import pitch_curve

        seconds = seconds or (max(e for _, e, _ in notes) + 0.2 if notes else 1.0)
        f0 = pitch_curve(notes, seconds, RATE)
        env = _loud_mean(envelope(reference, sample_rate))[None].repeat(f0.numel(), 1)
        env[f0 == 0] -= 8.0                         # rests are near silence (-70 dB), not the voice's noise
        return synthesize(f0, env, sample_rate)[: int(seconds * sample_rate)]


def _loud_mean(env: torch.Tensor) -> torch.Tensor:
    level = env.mean(1)
    keep = level > level.max() - 4.0
    return env[keep].mean(0)


def _fit(out: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    n = like.numel()
    return out[:n] if out.numel() >= n else torch.nn.functional.pad(out, (0, n - out.numel()))


__all__ = ["HarmonicNoise", "envelope", "synthesize"]
