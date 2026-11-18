# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""YIN against the old tracker on voices where the right answer is known."""

import math

import torch

from helix_lm.audio.singing import estimate_f0, estimate_f0_autocorr
from helix_lm.audio.yin import yin_f0
from helix_lm.bench.data.synthetic import TRUTH_RATE, render

SR = 24000


def _cents(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    both = (a > 0) & (b > 0)
    return (1200 * torch.log2(a[both] / b[both])).abs()


def test_a_steady_tone_is_found_to_a_cent():
    t = torch.arange(SR) / SR
    for hz in (82.0, 220.0, 440.0, 880.0):
        wave = 0.5 * torch.sin(2 * math.pi * hz * t)
        f0 = yin_f0(wave, SR, 100.0)[5:-5]
        assert (f0 > 0).all()
        assert float(_cents(f0, torch.full_like(f0, hz)).median()) < 1.0


def test_silence_and_noise_are_unvoiced():
    assert (yin_f0(torch.zeros(SR), SR, 100.0) == 0).all()
    noise = torch.randn(SR, generator=torch.Generator().manual_seed(0)) * 0.1
    assert float((yin_f0(noise, SR, 100.0) > 0).float().mean()) < 0.2


def _errors(track: torch.Tensor, truth: torch.Tensor) -> tuple[float, float]:
    n = min(len(track), len(truth))
    e = _cents(track[:n], truth[:n])
    return float((e > 600).float().mean()), float(e.pow(2).mean().sqrt())


def test_sung_vowels_have_no_octave_errors_where_the_old_tracker_had_many():
    notes = [(0.1, 0.9, 45), (1.0, 1.9, 52), (2.0, 2.9, 57)]
    for singer, shift in (("low", 0), ("mid", 12), ("high", 24)):
        wave, truth = render([(a, b, m + shift) for a, b, m in notes], SR, "a", singer, 3.0)
        gross, rms = _errors(yin_f0(wave, SR, TRUTH_RATE, centre=0.5), truth)
        old_gross, old_rms = _errors(estimate_f0_autocorr(wave[None], SR, TRUTH_RATE)[0], truth)
        assert gross < 0.02 and rms < 15, (singer, gross, rms)
        assert rms < old_rms, singer
        if singer == "low":
            assert old_gross > 0.1                     # the reason for the change: 29% octave errors here


def test_the_package_tracker_is_yin_and_keeps_its_shape():
    wave = torch.randn(2, SR // 2) * 0.01
    f0 = estimate_f0(wave, SR, 25)
    assert f0.shape[0] == 2 and f0.dim() == 2
    f0, conf = yin_f0(wave[0], SR, 25.0, return_confidence=True)
    assert f0.shape == conf.shape and float(conf.min()) >= 0 and float(conf.max()) <= 1
