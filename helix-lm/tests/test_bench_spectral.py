# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Spectral distances, mel cepstral distortion, DTW and loudness: zero for the same signal, larger the further
the signals are apart, and BS.1770 loudness on its reference tone."""

import math

import numpy as np
import torch

from helix_lm.bench.data.synthetic import render
from helix_lm.bench.loudness import integrated_loudness, normalize
from helix_lm.bench.metrics.dtw import cost_matrix, dtw
from helix_lm.bench.metrics.mcd import mcd
from helix_lm.bench.metrics.spectral import log_spectral_distance, mrstft_distance, si_snr, spectral_metrics

SR = 24000


def _voice(singer="mid", vowel="a", seed=0):
    return render([(0.1, 1.4, 57)], SR, vowel, singer, 1.5, seed=seed)[0]


def test_the_same_signal_is_at_zero_distance():
    a = _voice()
    assert mcd(a, a, SR) < 1e-6
    assert mrstft_distance(a, a) < 1e-6 and log_spectral_distance(a, a) < 1e-6
    assert si_snr(a, a) > 60


def test_distances_grow_with_the_difference():
    a = _voice()
    close = _voice(vowel="e")
    noise = torch.randn(a.shape, generator=torch.Generator().manual_seed(1)) * 0.1
    assert 0 < mcd(close, a, SR) < mcd(noise, a, SR)
    s = spectral_metrics(close, a)
    assert {"mrstft", "lsd_db", "spectral_convergence", "si_snr_db"} <= set(s)


def test_dtw_finds_a_shifted_copy():
    a = np.sin(np.linspace(0, 6, 60))[:, None]
    b = np.concatenate([np.zeros((5, 1)), a])
    path_a, path_b, cost = dtw(cost_matrix(a, b))
    assert cost < 1e-6 and len(path_a) == len(path_b) >= 60


def test_loudness_of_the_bs1770_reference_tone():
    for sr in (24000, 48000):
        t = torch.arange(10 * sr) / sr
        tone = torch.sin(2 * math.pi * 997 * t)                      # 0 dBFS peak sine: -3.01 LUFS
        assert abs(integrated_loudness(tone, sr) + 3.01) < 0.05
    quiet = normalize(0.01 * _voice(), SR, target=-23.0)
    assert abs(integrated_loudness(quiet, SR) + 23.0) < 0.5
