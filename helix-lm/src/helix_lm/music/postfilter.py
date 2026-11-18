# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
A post-filter for generated singing, in the STFT domain:

* formant emphasis: the cepstral envelope's detail (coefficients 2 to ``lifter``) scaled by ``beta`` > 1, the
  classic remedy for the muffled, over-smoothed envelopes generative models produce;
* tilt correction: the long-term spectrum tilted to the reference's slope (dB per octave), so a converted voice is
  as bright as the target singer.

The excitation (harmonics and noise) is untouched: only the envelope changes, so pitch metrics do not move.
"""

from __future__ import annotations

import math

import torch

from ..bench.metrics.tilt import spectral_tilt


def postfilter(wave: torch.Tensor, sample_rate: int, beta: float = 1.2, lifter: int = 30,
               target_tilt_db: float | None = None, n_fft: int = 1024, hop: int = 256) -> torch.Tensor:
    x = wave.flatten().float()
    window = torch.hann_window(n_fft)
    spec = torch.stft(x, n_fft, hop, window=window, return_complex=True)              # (bins, frames)
    mag = spec.abs().clamp(min=1e-7)
    log = mag.log().T                                                                  # (frames, bins)
    cep = torch.fft.irfft(log, n=n_fft)
    env_cep = torch.zeros_like(cep)
    env_cep[:, :lifter] = cep[:, :lifter]
    env_cep[:, n_fft - lifter + 1 :] = cep[:, n_fft - lifter + 1 :]
    boosted = env_cep.clone()
    boosted[:, 2:lifter] *= beta
    boosted[:, n_fft - lifter + 1 : n_fft - 1] *= beta
    gain_log = torch.fft.rfft(boosted - env_cep, n=n_fft).real                       # (frames, bins)
    if target_tilt_db is not None:
        now = spectral_tilt(x, sample_rate)
        if not math.isnan(now):
            freqs = torch.fft.rfftfreq(n_fft, 1.0 / sample_rate).clamp(min=100.0)
            octaves = torch.log2(freqs / 1000.0)
            gain_log = gain_log + ((target_tilt_db - now) * octaves * math.log(10) / 20)[None]
    gain_log = gain_log - gain_log.mean(-1, keepdim=True)                             # keep the level
    out = torch.istft(spec * gain_log.T.exp(), n_fft, hop, window=window, length=x.numel())
    return out * (x.abs().max() / out.abs().max().clamp(min=1e-9))


__all__ = ["postfilter"]
