# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Voice similarity and the Frechet distance between sets of voices."""

import math

import torch

from helix_lm.bench.data.synthetic import render
from helix_lm.bench.metrics.fad import fad, frechet_distance
from helix_lm.bench.metrics.similarity import similarity_metrics, speaker_similarity

SR = 24000


def _take(singer, vowel="a", midi=57, seed=0):
    return render([(0.1, 1.9, midi)], SR, vowel, singer, 2.0, seed=seed)[0]


def test_a_voice_is_closer_to_itself_than_to_another_singer():
    mid, mid2 = _take("mid", "a", 57), _take("mid", "a", 62, seed=1)
    low, high = _take("low", "a", 45, seed=2), _take("high", "a", 69, seed=3)
    assert math.isclose(speaker_similarity(mid, mid, SR), 1.0)
    same = speaker_similarity(mid, mid2, SR)
    assert same > speaker_similarity(mid, low, SR) and same > speaker_similarity(mid, high, SR)
    assert similarity_metrics(mid2, mid, SR, source=low)["identity_margin"] > 0


def test_a_copy_of_a_take_is_nearer_the_take_than_another_singer_is():
    from helix_lm.bench.baselines import BASELINES

    mid, low = _take("mid", "a", 57), _take("low", "a", 45, seed=2)
    copy = BASELINES["harmonic-noise"]().resynthesize(mid, SR)
    assert speaker_similarity(copy, mid, SR) > speaker_similarity(low, mid, SR)


def test_frechet_distance_is_zero_for_one_distribution_and_grows_with_a_shift():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(2000, 8, generator=g)
    b = torch.randn(2000, 8, generator=g)
    assert frechet_distance(a, b) < 0.1
    assert frechet_distance(a + 2.0, b) > 25.0


def test_fad_of_a_set_against_itself_is_lower_than_against_another_singer():
    lows = [_take("low", v, 45, seed=k) for k, v in enumerate("aeio")]
    highs = [_take("high", v, 69, seed=k) for k, v in enumerate("aeio")]
    assert fad(lows[:2], lows[2:], SR) < fad(lows, highs, SR)
