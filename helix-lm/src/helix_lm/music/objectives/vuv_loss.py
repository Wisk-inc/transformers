# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Voicing from the pitch head's unvoiced bin, with the frames around each voicing change weighted up: those few
frames are where most voiced/unvoiced errors are made (a consonant's edge, the end of a phrase)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...audio.singing import PITCH_BINS


def boundary_weights(voiced: torch.Tensor, radius: int = 2, boost: float = 3.0) -> torch.Tensor:
    """``boost`` on the frames within ``radius`` of a voicing change, 1 elsewhere."""
    v = voiced.float()
    change = (v[..., 1:] != v[..., :-1]).float()
    edge = F.pad(change, (1, 0)) + F.pad(change, (0, 1))
    near = F.max_pool1d(edge.reshape(-1, 1, edge.shape[-1]), 2 * radius + 1, stride=1, padding=radius)
    return 1.0 + (boost - 1.0) * near.reshape(edge.shape).clamp(max=1.0)


def voicing_loss(logits: torch.Tensor, f0_hz: torch.Tensor, radius: int = 2, boost: float = 3.0) -> torch.Tensor:
    """Weighted binary cross-entropy of "unvoiced" from the softmax over the pitch head's bins."""
    log_probs = F.log_softmax(logits[..., : PITCH_BINS + 1].float(), -1)
    unvoiced_logit = log_probs[..., PITCH_BINS] - torch.log1p(-log_probs[..., PITCH_BINS].exp().clamp(max=1 - 1e-6))
    target = (f0_hz <= 0).float()
    w = boundary_weights(f0_hz > 0, radius, boost)
    return (F.binary_cross_entropy_with_logits(unvoiced_logit, target, reduction="none") * w).sum() / w.sum()


__all__ = ["boundary_weights", "voicing_loss"]
