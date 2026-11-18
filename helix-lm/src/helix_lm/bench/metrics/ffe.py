# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Gross pitch error (GPE), fine pitch error (FPE) and F0 frame error (FFE), as pitch-tracking papers use them:

* GPE: among frames voiced in both, the share whose pitch is more than 20% off;
* FPE: the standard deviation, in cents, of the pitch error of the frames without a gross error;
* FFE: the share of all frames with either a voicing error or a gross pitch error.
"""

from __future__ import annotations

import math

import torch

GROSS = 0.2


def ffe_metrics(pred: torch.Tensor, ref: torch.Tensor, gross: float = GROSS) -> dict:
    pred, ref = pred.flatten().double(), ref.flatten().double()
    n = min(pred.numel(), ref.numel())
    pred, ref = pred[:n], ref[:n]
    if n == 0:
        return {"gpe": math.nan, "fpe_cents": math.nan, "ffe": math.nan}
    p, r = pred > 0, ref > 0
    both = p & r
    ratio = torch.ones_like(pred)
    ratio[both] = (pred[both] / ref[both] - 1).abs()
    gross_err = both & (ratio > gross)
    fine = both & ~gross_err
    cents = 1200 * torch.log2(pred[fine] / ref[fine]) if fine.any() else torch.empty(0, dtype=torch.float64)
    return {
        "gpe": float(gross_err.sum()) / int(both.sum()) if both.any() else math.nan,
        "fpe_cents": float(cents.std(unbiased=False)) if cents.numel() > 1 else math.nan,
        "ffe": float((p != r).sum() + gross_err.sum()) / n,
    }


__all__ = ["GROSS", "ffe_metrics"]
