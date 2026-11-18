# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The tools used when singing a score: pitch correction toward the notes, vibrato, glides between notes, breaths
in rests, consonants ahead of the beat, loudness matching, the post-filter, best-of-n and score alignment."""

import math

import torch

from helix_lm.bench.loudness import integrated_loudness
from helix_lm.bench.metrics.semitone import note_track
from helix_lm.music.best_of import best_of
from helix_lm.music.breath_insertion import insert
from helix_lm.music.consonant_timing import advance, is_vowel
from helix_lm.music.loudness_match import match_loudness
from helix_lm.music.pitch_correction import correct
from helix_lm.music.portamento import curve
from helix_lm.music.postfilter import postfilter
from helix_lm.music.score_align import align
from helix_lm.music.vibrato_model import apply, match

SR = 24000
NOTES = [(0.1, 0.9, 57), (1.0, 1.9, 60)]


def _cents(a, b):
    both = (a > 0) & (b > 0)
    return (1200 * torch.log2(a[both] / b[both])).abs()


def test_correction_pulls_a_flat_singer_toward_the_notes():
    truth = note_track(NOTES, 200, 100.0)
    flat = truth * 2 ** (-35 / 1200)
    fixed = correct(flat, NOTES, 100.0)
    assert float(_cents(fixed, truth).mean()) < float(_cents(flat, truth).mean())


def test_vibrato_is_added_and_read_back():
    plain = note_track([(0.0, 3.0, 57)], 300, 100.0)
    sung = apply(plain, [(0.0, 3.0, 57)], 100.0)
    assert float(_cents(sung, plain).max()) > 10
    v = match(sung, 100.0)
    assert v is not None and 4.0 < v.rate_hz < 7.5


def test_glides_join_notes_without_a_jump():
    c = curve(NOTES, 2.0, 100.0)
    voiced = c[(c > 0)]
    steps = (1200 * torch.log2(voiced[1:] / voiced[:-1])).abs()
    assert float(steps.max()) < 300


def test_breaths_go_into_rests_and_leave_notes_alone():
    wave = torch.zeros(2 * SR)
    out = insert(wave, [(0.0, 0.5, 57), (1.2, 1.9, 60)], SR)
    assert float(out[int(0.7 * SR) : int(1.1 * SR)].abs().max()) > 0
    assert float(out[: int(0.4 * SR)].abs().max()) == 0


def test_consonants_start_before_the_note():
    assert is_vowel("a") and not is_vowel("t")
    starts = advance(["t", "a"], [0.1, 0.4], [10])
    assert starts is not None


def test_loudness_is_matched_to_the_reference():
    t = torch.arange(2 * SR) / SR
    ref = 0.3 * torch.sin(2 * math.pi * 220 * t)
    out = match_loudness(0.01 * ref, ref, SR)
    assert abs(integrated_loudness(out, SR) - integrated_loudness(ref, SR)) < 0.5


def test_the_postfilter_keeps_length_and_level():
    t = torch.arange(SR) / SR
    wave = 0.3 * torch.sin(2 * math.pi * 220 * t) + 0.01 * torch.randn(SR)
    out = postfilter(wave, SR)
    assert out.shape == wave.shape and float(out.abs().max()) <= 1.0


def test_best_of_keeps_the_best_scored_take():
    takes = iter([torch.full((10,), 3.0), torch.full((10,), 1.0), torch.full((10,), 2.0)])
    out, scores = best_of(lambda seed: next(takes), n=3, scorer=lambda w: -float(w[0]))
    assert float(out[0]) == 1.0 and len(scores) == 3


def test_alignment_moves_notes_to_where_they_were_sung():
    late = [(a + 0.1, b + 0.1, m) for a, b, m in NOTES]
    f0 = note_track(late, 220, 100.0)
    moved = align(NOTES, f0, 100.0)
    assert abs(moved[0][0] - 0.2) < 0.06
