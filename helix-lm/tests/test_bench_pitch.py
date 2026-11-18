# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Pitch, voicing, gross pitch error and note metrics on tracks whose errors are set by hand."""

import math

import torch

from helix_lm.bench.metrics.ffe import ffe_metrics
from helix_lm.bench.metrics.pitch import f0_rmse_cents, f0_rmse_hz, log_f0_corr, pitch_metrics
from helix_lm.bench.metrics.semitone import midi_to_hz, note_metrics, note_track
from helix_lm.bench.metrics.vibrato import vibrato
from helix_lm.bench.metrics.voicing import vuv_metrics


def test_a_track_a_semitone_sharp_is_100_cents_off():
    ref = torch.full((200,), 220.0)
    sharp = ref * 2 ** (1 / 12)
    assert math.isclose(f0_rmse_cents(sharp, ref), 100.0, abs_tol=1e-3)
    assert f0_rmse_cents(ref, ref) == 0.0
    assert math.isclose(f0_rmse_hz(sharp, ref), float(sharp[0] - ref[0]), rel_tol=1e-5)


def test_the_contour_correlation_ignores_a_constant_offset():
    t = torch.arange(300) / 100.0
    ref = 220 * 2 ** (t / 12)                                  # a glide up a minor third
    assert log_f0_corr(ref * 1.5, ref) > 0.999
    assert log_f0_corr(torch.flip(ref, [0]), ref) < -0.99      # the same glide down


def test_voicing_errors_and_gross_errors_are_counted_per_frame():
    ref = torch.full((100,), 200.0)
    ref[:20] = 0
    pred = ref.clone()
    pred[:10] = 200.0                        # 10 false voiced frames
    pred[50:60] = 400.0                      # 10 octave errors
    v = vuv_metrics(pred, ref)
    assert math.isclose(v["vuv_error"], 0.10, abs_tol=1e-6)
    f = ffe_metrics(pred, ref)
    assert math.isclose(f["gpe"], 10 / 80, abs_tol=1e-6) and math.isclose(f["ffe"], 0.20, abs_tol=1e-6)
    assert "f0_rmse_cents" in pitch_metrics(pred, ref)


def test_notes_sung_within_50_cents_count_as_right():
    notes = [(0.0, 1.0, 57), (1.0, 2.0, 60)]
    track = note_track(notes, 200, 100.0)
    assert math.isclose(float(track[50]), midi_to_hz(57), rel_tol=1e-6)
    good = note_metrics(track * 2 ** (30 / 1200), notes, 100.0)
    bad = note_metrics(track * 2 ** (80 / 1200), notes, 100.0)
    assert good["note_accuracy"] == 1.0 and bad["note_accuracy"] == 0.0


def test_vibrato_rate_and_depth_are_measured():
    t = torch.arange(400) / 100.0
    f0 = 220 * 2 ** (40 * torch.sin(2 * math.pi * 5.5 * t) / 1200)
    v = vibrato(f0, 100.0)
    assert v["present"] and abs(v["rate_hz"] - 5.5) < 0.4 and 25 < v["extent_cents"] < 55
