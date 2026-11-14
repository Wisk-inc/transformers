# Copyright 2026 Nathan. Apache-2.0.
"""
Knowing what it does not know.

A single trained network gives one answer even in situations it has barely seen (a hurricane over an island it has
never seen one over, an unusual winter front). Dropout before the output heads, kept active at forecast time and
sampled several times (Gal and Ghahramani 2016), gives a family of plausible networks; their chains are mixed like
the driving models', so where the network is unsure, the start window widens and the totals spread.
"""

from __future__ import annotations

import torch
from torch import nn


class HeadDropout(nn.Module):
    """Dropout that can be switched on at forecast time independently of the rest of the model."""

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = p
        self.sampling = False

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.p <= 0 or not (self.training or self.sampling):
            return h
        return nn.functional.dropout(h, self.p, training=True)


__all__ = ["HeadDropout"]
