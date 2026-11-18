# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Voiced/unvoiced decisions of a generated track against the reference's."""

from __future__ import annotations

import math

import torch


def vuv_metrics(pred: torch.Tensor, ref: torch.Tensor) -> dict:
    """``vuv_error`` (frames whose decision differs) and precision, recall and F1 of the voiced frames."""
    pred, ref = pred.flatten(), ref.flatten()
    n = min(pred.numel(), ref.numel())
    p, r = pred[:n] > 0, ref[:n] > 0
    if n == 0:
        return {"vuv_error": math.nan, "voiced_precision": math.nan, "voiced_recall": math.nan, "voiced_f1": math.nan}
    tp = int((p & r).sum())
    precision = tp / int(p.sum()) if int(p.sum()) else math.nan
    recall = tp / int(r.sum()) if int(r.sum()) else math.nan
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall and precision + recall > 0 else 0.0
    return {"vuv_error": float((p != r).float().mean()), "voiced_precision": precision, "voiced_recall": recall,
            "voiced_f1": f1}


__all__ = ["vuv_metrics"]
