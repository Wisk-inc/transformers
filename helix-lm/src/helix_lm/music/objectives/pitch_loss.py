# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Pitch over TriStream's 256 log-spaced bins (about 21 cents each) plus one unvoiced bin.

Until 2.4 the target was the nearest bin, one-hot, and the decoder took the argmax: every pitch the model sang was
rounded to 21 cents, and a prediction one bin off cost as much as one an octave off. The target here is a Gaussian
over the bins around the exact pitch (CREPE's choice, 25 cents wide), and the decoder averages the log frequency of
the bins around the peak, which gives back sub-bin pitch.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ...audio.singing import F0_MAX, F0_MIN, PITCH_BINS

BIN_CENTS = 1200.0 * math.log2(F0_MAX / F0_MIN) / (PITCH_BINS - 1)


def continuous_bins(f0_hz: torch.Tensor) -> torch.Tensor:
    """The exact (fractional) bin of each pitch; NaN where unvoiced."""
    scaled = (torch.log(f0_hz.clamp(min=F0_MIN, max=F0_MAX)) - math.log(F0_MIN)) / math.log(F0_MAX / F0_MIN)
    return torch.where(f0_hz > 0, scaled * (PITCH_BINS - 1), torch.full_like(f0_hz, float("nan")))


def blurred_targets(f0_hz: torch.Tensor, sigma_bins: float = 1.2) -> torch.Tensor:
    """``(..., PITCH_BINS + 1)`` target distributions: a normalised Gaussian around the exact bin when voiced, the
    unvoiced bin otherwise."""
    pos = continuous_bins(f0_hz)
    grid = torch.arange(PITCH_BINS, device=f0_hz.device, dtype=torch.float32)
    voiced = ~torch.isnan(pos)
    g = torch.exp(-0.5 * ((grid - torch.nan_to_num(pos, nan=0.0)[..., None]) / sigma_bins) ** 2)
    g = g / g.sum(-1, keepdim=True).clamp(min=1e-9)
    out = torch.zeros(*f0_hz.shape, PITCH_BINS + 1, device=f0_hz.device)
    out[..., :PITCH_BINS] = torch.where(voiced[..., None], g, torch.zeros_like(g))
    out[..., PITCH_BINS] = (~voiced).float()
    return out


def pitch_distribution_loss(logits: torch.Tensor, f0_hz: torch.Tensor, sigma_bins: float = 1.2,
                            reduction: str = "none") -> torch.Tensor:
    """Cross-entropy against :func:`blurred_targets`; ``sigma_bins = 0`` gives the old one-hot target."""
    if sigma_bins <= 0:
        from ...audio.singing import hz_to_bins

        target = F.one_hot(hz_to_bins(f0_hz), PITCH_BINS + 1).float()
    else:
        target = blurred_targets(f0_hz, sigma_bins)
    loss = -(target * F.log_softmax(logits[..., : PITCH_BINS + 1].float(), -1)).sum(-1)
    return loss.mean() if reduction == "mean" else loss


def decode_pitch(logits: torch.Tensor, window: int = 4, unvoiced_above: float = 0.5) -> torch.Tensor:
    """F0 in Hz from pitch logits: unvoiced where the unvoiced bin's probability exceeds ``unvoiced_above``, else
    the probability-weighted mean bin of the ``window`` bins either side of the peak, in log frequency."""
    probs = F.softmax(logits[..., : PITCH_BINS + 1].float(), -1)
    voiced_probs = probs[..., :PITCH_BINS]
    peak = voiced_probs.argmax(-1)
    offsets = torch.arange(-window, window + 1, device=logits.device)
    idx = (peak[..., None] + offsets).clamp(0, PITCH_BINS - 1)
    w = voiced_probs.gather(-1, idx)
    pos = (w * idx.float()).sum(-1) / w.sum(-1).clamp(min=1e-9)
    hz = torch.exp(math.log(F0_MIN) + pos / (PITCH_BINS - 1) * math.log(F0_MAX / F0_MIN))
    return torch.where(probs[..., PITCH_BINS] > unvoiced_above, torch.zeros_like(hz), hz)


__all__ = ["BIN_CENTS", "blurred_targets", "continuous_bins", "decode_pitch", "pitch_distribution_loss"]
