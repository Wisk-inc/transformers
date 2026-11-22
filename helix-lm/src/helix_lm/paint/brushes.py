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
    (0.93, 0.05, 0.10, 0.45, 11.0, 0.60, 0.15, 0.60),  # flat: square ends, clear bristle marks
    (0.84, 0.55, 1.00, 0.25, 6.0, 0.40, 0.10, 0.40),   # round: pointed
    (0.88, 0.30, 0.80, 0.35, 9.0, 0.50, 0.12, 0.50),   # filbert: rounded flat, the all-purpose brush
    (0.86, 0.15, 0.30, 0.85, 15.0, 1.60, 0.55, 0.30),  # dry: little paint, broken streaks
    (0.98, 0.00, 0.00, 0.08, 4.0, 0.20, 0.05, 1.00),   # knife: hard slabs of thick paint
    (0.60, 0.35, 1.00, 0.20, 7.0, 0.30, 0.20, 0.06),   # wash: wide and thin, its edge a little soft
)


def brush_table(device=None, dtype=torch.float32) -> torch.Tensor:
    """``(brushes, fields)``."""
    return torch.tensor(_TABLE, dtype=dtype, device=device)


def brush_profile(weights: torch.Tensor, soft: float = 1.0) -> torch.Tensor:
    """``(..., brushes)`` weights summing to 1 -> ``(..., fields)`` the row the renderer uses. ``soft`` above 1
    makes every edge that much softer (a style painted without hard edges, like Leonardo's)."""
    profile = weights @ brush_table(weights.device, weights.dtype)
    if soft == 1.0:
        return profile
    hard = 1.0 - ((1.0 - profile[..., :1]) * soft).clamp(max=0.95)
    return torch.cat([hard, profile[..., 1:]], -1)


def one_hot(index: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.one_hot(index, len(BRUSHES)).float()


__all__ = ["BRUSHES", "FIELDS", "brush_profile", "brush_table", "one_hot"]
