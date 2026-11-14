# Copyright 2026 Nathan. Apache-2.0.
"""
Polyak averaging of the weights.

The weights at the end of training wander with the last few batches; their exponential moving average (decay
0.999, about the last thousand steps) sits nearer the centre of the basin and usually forecasts better on unseen
data. Kept on the side during training, swapped in for evaluation and saved for release.
"""

from __future__ import annotations

import contextlib

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))       # warm-up: short memory at the start
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    @contextlib.contextmanager
    def applied(self, model: nn.Module):
        """Temporarily load the averaged weights."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict({**model.state_dict(), **{k: self.shadow[k].to(backup[k].dtype) for k in backup}})
        try:
            yield model
        finally:
            model.load_state_dict({**model.state_dict(), **backup})

    def state_dict(self) -> dict:
        return {"decay": self.decay, "updates": self.updates, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay, self.updates = state["decay"], state["updates"]
        self.shadow = {k: v.float() for k, v in state["shadow"].items()}


__all__ = ["EMA"]
