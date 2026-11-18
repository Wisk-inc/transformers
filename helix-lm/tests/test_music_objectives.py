# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The training losses aimed at the benchmark: blurred pitch targets decoded below a bin, multi-resolution STFT,
the spectral envelope and voicing at note edges."""

import math

import torch

from helix_lm.music.objectives import blurred_targets, decode_pitch, pitch_distribution_loss
from helix_lm.music.objectives.formant_loss import envelope_loss
from helix_lm.music.objectives.pitch_loss import BIN_CENTS, PITCH_BINS
from helix_lm.music.objectives.stft_loss import mrstft_loss
from helix_lm.music.objectives.vuv_loss import boundary_weights, voicing_loss


def test_the_pitch_target_is_blurred_and_sums_to_one():
    f0 = torch.tensor([[0.0, 110.0, 220.5, 440.0]])
    target = blurred_targets(f0)
    assert torch.allclose(target.sum(-1), torch.ones(1, 4), atol=1e-5)
    assert float(target[0, 0, -1]) == 1.0                                    # unvoiced bin
    assert int((target[0, 2] > 0.01).sum()) > 2                              # spread over neighbours


def test_decoding_the_target_gives_the_pitch_back_below_a_bin():
    f0 = torch.tensor([[110.0, 233.3, 440.0, 0.0, 987.7]])
    logits = torch.log(blurred_targets(f0).clamp(min=1e-9))
    back = decode_pitch(logits)
    voiced = f0 > 0
    cents = (1200 * torch.log2(back[voiced] / f0[voiced])).abs()
    assert float(cents.max()) < BIN_CENTS / 4 and float(back[0, 3]) == 0.0


def test_the_loss_is_lowest_at_the_right_pitch():
    f0 = torch.full((1, 8), 220.0)
    right = torch.log(blurred_targets(f0).clamp(min=1e-9))
    wrong = torch.log(blurred_targets(f0 * 2).clamp(min=1e-9))
    assert float(pitch_distribution_loss(right, f0).mean()) < float(pitch_distribution_loss(wrong, f0).mean())


def test_stft_and_envelope_losses_are_zero_on_the_target_and_have_gradients():
    t = torch.arange(24000) / 24000
    target = (0.3 * torch.sin(2 * math.pi * 220 * t))[None]
    pred = (target + 0.05 * torch.randn_like(target)).requires_grad_()
    assert float(mrstft_loss(target, target)) < 1e-4
    loss = mrstft_loss(pred, target) + envelope_loss(pred, target)
    loss.backward()
    assert pred.grad is not None and float(pred.grad.abs().sum()) > 0


def test_voicing_errors_at_note_edges_weigh_more():
    f0 = torch.tensor([[0.0] * 5 + [200.0] * 10 + [0.0] * 5])
    w = boundary_weights(f0 > 0)
    assert float(w[0, 5]) > float(w[0, 10])
    logits = torch.zeros(1, f0.shape[1], PITCH_BINS + 1, requires_grad=True)   # the pitch head's output
    voicing_loss(logits, f0).backward()
    assert logits.grad is not None
