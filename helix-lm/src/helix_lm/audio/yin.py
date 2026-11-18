# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
YIN pitch tracking (de Cheveigne and Kawahara, 2002), vectorised in torch, with frames centred on their time.

The package's plain autocorrelation tracker (:func:`helix_lm.audio.singing.estimate_f0`) takes the highest
correlation peak. When a vowel's first formant makes one harmonic loud, the correlation at a fraction of the period
is as high as at the period, and it reports the pitch an octave or more too high: on the synthetic benchmark voices
its error was 376 cents for a low voice and 920 cents for a high one. It also stamps each frame at the start of its
40 ms window, 20 ms late. YIN's cumulative-mean-normalised difference takes the first deep dip instead, which is
the period, not a fraction of it. The prepare stage makes TriStream's pitch targets with this tracker from 2.4 on.
"""

from __future__ import annotations

import torch

F0_MIN, F0_MAX = 50.0, 1100.0
THRESHOLD = 0.12           # the first dip of the normalised difference under this is the period
VOICED = 0.35              # a frame whose best dip stays above this is unvoiced
SUBHARMONIC_MARGIN = 0.05  # how much deeper a multiple of the lag must dip to be taken as the period


def _difference(frames: torch.Tensor, window: int, max_lag: int) -> torch.Tensor:
    """``d(tau) = sum_j (x_j - x_{j+tau})^2`` over ``window`` samples, for ``tau`` in ``0..max_lag``, by FFT."""
    n = frames.shape[-1]
    size = 1 << (2 * n - 1).bit_length()
    spec = torch.fft.rfft(frames, n=size)
    head = torch.fft.rfft(frames[..., :window], n=size)
    cross = torch.fft.irfft(spec * head.conj(), n=size)[..., : max_lag + 1]   # sum_j x_j x_{j+tau}
    sq = torch.nn.functional.pad(frames.pow(2).cumsum(-1), (1, 0))
    energy_head = sq[..., window : window + 1]                                   # sum_{j<window} x_j^2
    lags = torch.arange(max_lag + 1, device=frames.device)
    energy_shift = sq[..., lags + window] - sq[..., lags]                        # sum_j x_{j+tau}^2
    return (energy_head + energy_shift - 2 * cross).clamp(min=0)


def yin_f0(wave: torch.Tensor, sample_rate: int, frame_rate: float, f_min: float = F0_MIN, f_max: float = F0_MAX,
           threshold: float = THRESHOLD, voiced: float = VOICED, return_confidence: bool = False,
           centre: float = 0.0):
    """``(B, T)`` waveform -> ``(B, frames)`` F0 in Hz, 0 where unvoiced; frame ``i`` is centred at
    ``(i + centre) / frame_rate`` seconds (``centre=0.5``: the middle of codec frame ``i``). With
    ``return_confidence`` also ``1 - d'(period)`` per frame."""
    squeeze = wave.dim() == 1
    wave = wave[None] if squeeze else wave
    wave = wave.float()
    hop = int(round(sample_rate / frame_rate))
    max_lag = int(sample_rate / f_min)
    min_lag = max(2, int(sample_rate / f_max))
    window = max_lag
    span = window + max_lag + 1
    n_frames = wave.shape[-1] // hop + 1
    lead = max(0, span // 2 - int(round(centre * hop)))
    padded = torch.nn.functional.pad(wave, (lead, span))
    frames = padded.unfold(-1, span, hop)[:, :n_frames]                         # (B, T, span)
    frames = frames - frames.mean(-1, keepdim=True)
    d = _difference(frames, window, max_lag)                                     # (B, T, max_lag + 1)
    cum = d[..., 1:].cumsum(-1)
    lags = torch.arange(1, max_lag + 1, device=wave.device, dtype=d.dtype)
    cmnd = torch.ones_like(d)
    cmnd[..., 1:] = d[..., 1:] * lags / cum.clamp(min=1e-12)
    search = cmnd[..., min_lag : max_lag]                                        # lags min_lag .. max_lag-1
    left = torch.nn.functional.pad(search[..., :-1], (1, 0), value=float("inf"))
    right = torch.nn.functional.pad(search[..., 1:], (0, 1), value=float("inf"))
    dip = (search < threshold) & (search <= left) & (search <= right)
    first = torch.where(dip.any(-1), dip.float().argmax(-1), search.argmin(-1))
    best = first + min_lag
    value = cmnd.gather(-1, best[..., None])[..., 0]
    # a strong harmonic (a formant on it) can make a fraction of the period dip first: when a multiple of the lag
    # found dips clearly deeper, that multiple is the period
    for k in (3, 2):
        multiple = best * k
        ok = multiple + 1 <= max_lag
        near = torch.stack([cmnd.gather(-1, (multiple + o).clamp(1, max_lag)[..., None])[..., 0] for o in (-1, 0, 1)])
        deeper, where = near.min(0)
        take = ok & (deeper < value - SUBHARMONIC_MARGIN)
        best = torch.where(take, (multiple + where - 1).clamp(1, max_lag), best)
        value = torch.where(take, deeper, value)
    # parabolic interpolation of the dip for a sub-sample period
    lo = cmnd.gather(-1, (best - 1).clamp(min=1)[..., None])[..., 0]
    hi = cmnd.gather(-1, (best + 1).clamp(max=max_lag)[..., None])[..., 0]
    denom = lo - 2 * value + hi
    shift = torch.where(denom.abs() > 1e-9, 0.5 * (lo - hi) / denom, torch.zeros_like(denom)).clamp(-1, 1)
    f0 = sample_rate / (best.float() + shift)
    energy = frames[..., :window].pow(2).mean(-1)
    is_voiced = (value < voiced) & (energy > 1e-6)
    f0 = torch.where(is_voiced, f0, torch.zeros_like(f0))
    if squeeze:
        f0, value = f0[0], value[0]
    return (f0, (1 - value).clamp(0, 1)) if return_confidence else f0


__all__ = ["F0_MAX", "F0_MIN", "SUBHARMONIC_MARGIN", "THRESHOLD", "VOICED", "yin_f0"]
