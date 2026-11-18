# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Voice measures on signals with known answers: harmonics to noise, jitter, formants, tilt, glitches and the
quality proxy."""

import math

import numpy as np
import torch

from helix_lm.bench.data.synthetic import VOWELS, render
from helix_lm.bench.metrics.formants import formants
from helix_lm.bench.metrics.glitch import count_glitches
from helix_lm.bench.metrics.jitter_shimmer import jitter_shimmer
from helix_lm.bench.metrics.periodicity import hnr_db
from helix_lm.bench.metrics.quality import quality_proxy
from helix_lm.bench.metrics.tilt import spectral_tilt

SR = 24000


def test_a_clean_tone_has_more_harmonics_to_noise_than_a_noisy_one():
    t = torch.arange(SR) / SR
    clean = 0.5 * torch.sin(2 * math.pi * 200 * t)
    noisy = clean + 0.2 * torch.randn(SR, generator=torch.Generator().manual_seed(0))
    assert hnr_db(clean, SR) > hnr_db(noisy, SR) + 10


def test_jitter_is_low_for_a_steady_voice():
    wave = render([(0.0, 2.0, 57)], SR, "a", "mid", 2.0, vibrato_cents=0.0)[0]
    js = jitter_shimmer(wave, SR)
    assert js["jitter"] < 0.01 and js["shimmer"] < 0.05


def _source_filter_vowel(f0: float, formant_hz, bandwidths=(80.0, 100.0, 120.0)) -> torch.Tensor:
    """A pulse train through one two-pole resonator per formant and a glottal low-pass."""
    x = np.zeros(SR)
    x[:: int(SR / f0)] = 1.0
    for f, b in zip(formant_hz, bandwidths, strict=False):
        r, th = math.exp(-math.pi * b / SR), 2 * math.pi * f / SR
        a1, a2 = 2 * r * math.cos(th), -r * r
        y = np.zeros_like(x)
        for i in range(len(x)):
            y[i] = (1 - r) * x[i] + a1 * (y[i - 1] if i else 0.0) + a2 * (y[i - 2] if i > 1 else 0.0)
        x = y
    for i in range(1, len(x)):
        x[i] += 0.95 * x[i - 1]
    return torch.tensor(0.5 * x / np.abs(x).max(), dtype=torch.float32)


def test_the_first_two_formants_of_a_vowel_are_found():
    # LPC formants hold for speech-range pitch; above about 250 Hz the harmonics are too far apart (documented)
    for f0 in (100.0, 220.0):
        for vowel in ("a", "u"):
            f = formants(_source_filter_vowel(f0, VOWELS[vowel]), SR)
            f1, f2 = float(np.nanmedian(f[:, 0])), float(np.nanmedian(f[:, 1]))
            assert abs(f1 - VOWELS[vowel][0]) / VOWELS[vowel][0] < 0.12, (f0, vowel, f1)
            assert abs(f2 - VOWELS[vowel][1]) / VOWELS[vowel][1] < 0.12, (f0, vowel, f2)


def test_tilt_of_white_and_pink_noise():
    n = 4 * SR
    white = torch.randn(n, generator=torch.Generator().manual_seed(2))
    spec = torch.fft.rfft(white)
    freqs = torch.fft.rfftfreq(n, 1 / SR)
    pink = torch.fft.irfft(spec / freqs.clamp(min=20).sqrt(), n=n)
    assert abs(spectral_tilt(white, SR)) < 1.0
    assert abs(spectral_tilt(pink, SR) + 3.01) < 0.7


def test_clicks_and_dropouts_are_glitches_and_a_bright_voice_is_not():
    for singer, midi in (("low", 45), ("mid", 57)):
        wave = render([(0.0, 2.0, midi)], SR, "a", singer, 2.0)[0]
        n, _ = count_glitches(wave, SR)          # a sharp edge every period: part of the voice, not a click
        dropped = wave.clone()
        dropped[SR : SR + SR // 20] = 0
        clicked = wave.clone()
        clicked[30000] += 0.6
        assert n == 0 and count_glitches(dropped, SR)[0] >= 1 and count_glitches(clicked, SR)[0] >= 1, singer


def test_the_quality_proxy_prefers_a_voice_to_noise_and_clipping():
    voice = render([(0.1, 1.9, 57)], SR, "a", "mid", 2.0)[0]
    clipped = (voice * 20).clamp(-0.5, 0.5)
    noise = torch.randn(2 * SR, generator=torch.Generator().manual_seed(3)) * 0.2
    q = quality_proxy(voice, SR)["quality_proxy"]
    assert q > quality_proxy(clipped, SR)["quality_proxy"] and q > quality_proxy(noise, SR)["quality_proxy"]
