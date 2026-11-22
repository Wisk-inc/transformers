# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The brushes. Each is a row of numbers that the renderer reads: how hard its edge is, how much it tapers, whether
its ends are round or square, how strongly its bristles show, how many bristle streaks cross its width, how fast
it runs out of paint along a stroke, how much the stroke fades as it runs out, and how thick it lays the paint.

The model does not pick a brush by name: it gives a weight to each, and the stroke is rendered with the weighted
row. While training the weights are soft, so the choice can be learned; when painting the largest one is used.
"""

from __future__ import annotations

import torch

BRUSHES = ("flat", "round", "filbert", "dry", "knife", "wash")
FIELDS = ("hardness", "taper", "end_round", "bristle", "bristle_count", "load_decay", "fade", "thickness")

_TABLE = (
    # hardness taper end_round bristle count decay fade thickness
    (0.85, 0.05, 0.10, 0.35, 9.0, 0.60, 0.15, 0.60),   # flat: square ends, clear bristle marks
    (0.60, 0.55, 1.00, 0.15, 5.0, 0.40, 0.10, 0.40),   # round: pointed, soft
    (0.70, 0.30, 0.80, 0.25, 7.0, 0.50, 0.12, 0.50),   # filbert: rounded flat, the all-purpose brush
    (0.75, 0.15, 0.30, 0.80, 13.0, 1.60, 0.55, 0.30),  # dry: little paint, broken streaks
    (0.97, 0.00, 0.00, 0.05, 3.0, 0.20, 0.05, 1.00),   # knife: hard slabs of thick paint
    (0.15, 0.35, 1.00, 0.05, 3.0, 0.20, 0.20, 0.05),   # wash: wide, soft, thin
)


def brush_table(device=None, dtype=torch.float32) -> torch.Tensor:
    """``(brushes, fields)``."""
    return torch.tensor(_TABLE, dtype=dtype, device=device)


def brush_profile(weights: torch.Tensor) -> torch.Tensor:
    """``(..., brushes)`` weights summing to 1 -> ``(..., fields)`` the row the renderer uses."""
    return weights @ brush_table(weights.device, weights.dtype)


def one_hot(index: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.one_hot(index, len(BRUSHES)).float()


__all__ = ["BRUSHES", "FIELDS", "brush_profile", "brush_table", "one_hot"]
