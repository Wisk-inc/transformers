# Copyright 2026 Nathan. Apache-2.0.
"""
A mixture of experts for different kinds of rain.

A cold front's rain (a long band, steady, moving south-east), a tropical wave's (bursts in the afternoon and evening
behind the trough), island afternoon storms (fixed by terrain and sea breezes) and trade-wind showers (brief, on
windward slopes, any hour) follow different rules. A soft mixture of ``E`` expert networks with a learned gate
(Jacobs et al. 1991; Shazeer et al. 2017) lets each regime have its own mapping from the decoded state to the heads,
while sharing everything below. A small load-balancing term keeps every expert in use.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RegimeExperts(nn.Module):
    def __init__(self, hidden: int, experts: int = 4, expansion: int = 2) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden, experts)
        self.up = nn.Parameter(torch.randn(experts, hidden, expansion * hidden) * 0.02)
        self.down = nn.Parameter(torch.zeros(experts, expansion * hidden, hidden))
        self.norm = nn.LayerNorm(hidden)
        self.balance = torch.tensor(0.0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.norm(h)
        g = F.softmax(self.gate(x).float(), -1).to(h.dtype)                  # (..., E)
        hidden = F.gelu(torch.einsum("...d,edh->...eh", x, self.up.to(h.dtype)))
        out = torch.einsum("...eh,ehd->...ed", hidden, self.down.to(h.dtype))
        mean_use = g.reshape(-1, g.shape[-1]).float().mean(0)
        self.balance = (mean_use * mean_use).sum() * g.shape[-1] - 1.0        # 0 when experts share evenly
        return h + (g.unsqueeze(-1) * out).sum(-2)


__all__ = ["RegimeExperts"]
