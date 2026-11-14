# Copyright 2026 Nathan. Apache-2.0.
"""
Lead time as a first-class input, and predictability that decays.

Two pieces:

* **LeadFiLM** -- feature-wise linear modulation (Perez et al. 2018) of the decoder state by the lead-time embedding:
  the same features mean different things at hour 2 (mostly the rain already falling) and hour 60 (mostly the
  driving model's large-scale pattern);
* **LeadShrink** -- predictability theory says a forecast's information about a point decays towards climatology
  with lead time. The transition logits are pulled towards the prior (climatological) transitions with a weight
  ``1 - exp(-lead / tau)`` times a learned ceiling, with ``tau`` learned: the network can only be as confident at
  day three as the data have shown it can be.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class LeadFiLM(nn.Module):
    def __init__(self, hidden: int, embed: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed, 64), nn.GELU(), nn.Linear(64, 2 * hidden))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, frame_embed: torch.Tensor) -> torch.Tensor:
        """``h (B, F, P, d)``, ``frame_embed (F, 5)``."""
        gamma, beta = self.net(frame_embed.to(h.dtype)).chunk(2, -1)
        return h * (1 + gamma[None, :, None]) + beta[None, :, None]


class LeadShrink(nn.Module):
    def __init__(self, prior_logits: torch.Tensor, tau_hours: float = 48.0, ceiling: float = 0.3) -> None:
        super().__init__()
        self.register_buffer("prior", prior_logits.float(), persistent=False)       # (K, K)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_hours)))
        self.raw_ceiling = nn.Parameter(torch.tensor(math.log(ceiling / (1 - ceiling))))

    def forward(self, trans: torch.Tensor) -> torch.Tensor:
        """``trans (B, T, P, K, K)`` transition logits for hours 0..T-1."""
        t = torch.arange(trans.shape[1], device=trans.device, dtype=trans.dtype).view(1, -1, 1, 1, 1)
        w = torch.sigmoid(self.raw_ceiling) * (1 - torch.exp(-t / self.log_tau.exp()))
        prior = F.log_softmax(self.prior, -1).to(trans.dtype)
        return (1 - w) * trans + w * prior


__all__ = ["LeadFiLM", "LeadShrink"]
