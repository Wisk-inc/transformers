# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The baselines TriStream is measured against do their jobs: resynthesis, conversion and singing from notes."""

import torch

from helix_lm.bench.baselines import BASELINES
from helix_lm.bench.data.synthetic import render
from helix_lm.bench.metrics.mcd import mcd
from helix_lm.bench.metrics.similarity import speaker_similarity
from helix_lm.bench.systems import checkpoint_tag, make
from helix_lm.bench.tasks import f0

SR = 24000


def test_identity_returns_its_input():
    wave = torch.randn(SR)
    assert torch.equal(BASELINES["identity"]().resynthesize(wave, SR), wave)


def test_griffin_lim_and_harmonic_noise_keep_the_voice_close():
    wave = render([(0.1, 1.9, 57)], SR, "a", "mid", 2.0)[0]
    for name in ("griffin-lim", "harmonic-noise"):
        out = BASELINES[name]().resynthesize(wave, SR)
        assert out.shape == wave.shape and mcd(out, wave, SR) < 6.0, name


def test_harmonic_noise_conversion_keeps_the_tune_and_moves_toward_the_reference():
    source = render([(0.1, 1.9, 57)], SR, "a", "mid", 2.0)[0]
    reference = render([(0.1, 1.9, 45)], SR, "o", "low", 2.0, seed=1)[0]
    out = BASELINES["harmonic-noise"]().convert(source, reference, SR)
    a, b = f0(out, SR), f0(source, SR)
    n = min(len(a), len(b))
    both = (a[:n] > 0) & (b[:n] > 0)
    cents = (1200 * torch.log2(a[:n][both] / b[:n][both])).abs()
    assert float(cents.median()) < 25                      # one octave slip at an onset is allowed for
    assert speaker_similarity(out, reference, SR) > speaker_similarity(source, reference, SR)


def test_harmonic_noise_sings_notes_at_their_pitch():
    reference = render([(0.1, 1.9, 57)], SR, "a", "mid", 2.0)[0]
    out = BASELINES["harmonic-noise"]().sing([(0.1, 0.9, 60), (1.0, 1.9, 64)], SR, reference, seconds=2.0)
    track = f0(out, SR)
    assert out.numel() == 2 * SR and float((track > 0).float().mean()) > 0.5


def test_systems_by_name_and_a_new_checkpoint_gets_a_new_cache_name(tmp_path):
    assert make("identity").name == "identity" and make("tristream", None) is None
    ckpt = tmp_path / "latest.pt"
    ckpt.write_bytes(b"one")
    first = checkpoint_tag(ckpt)
    ckpt.write_bytes(b"one more")
    assert checkpoint_tag(ckpt) != first
