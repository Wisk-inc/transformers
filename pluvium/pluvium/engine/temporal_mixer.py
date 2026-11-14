# Copyright 2026 Nathan. Apache-2.0.
"""
Attention along a point's own future.

The intensity chain is written hour by hour from each frame's decoded state. Without context across frames, a front
the model places at hour 30 can still nudge the transition logits at hour 29 and 31 inconsistently, which shows up
as jitter in start times. The temporal mixer runs a small transformer along each point's 40 future frames (every
frame already carries its lead-time embedding), so each hour's state knows what comes before and after it. All
frames are derived from inputs available at issue time, so attending in both directions leaks nothing.
"""

from __future__ import annotations

import torch
from torch import nn


class TemporalMixer(nn.Module):
    def __init__(self, hidden: int, heads: int = 4, layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=2 * hidden, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """``h (B, F, P, d)`` -> same shape."""
        b, f, p, d = h.shape
        x = h.permute(0, 2, 1, 3).reshape(b * p, f, d)
        y = self.encoder(x).view(b, p, f, d).permute(0, 2, 1, 3)
        return h + torch.tanh(self.gate) * (y - h)


__all__ = ["TemporalMixer"]
