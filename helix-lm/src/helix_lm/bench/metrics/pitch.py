# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Pitch error between two F0 tracks in Hz (0 = unvoiced), on the frames voiced in both."""

from __future__ import annotations

import math

import torch


def _pair(pred: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred, ref = pred.flatten().double(), ref.flatten().double()
    n = min(pred.numel(), ref.numel())
    return pred[:n], ref[:n]


def both_voiced(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    pred, ref = _pair(pred, ref)
    return (pred > 0) & (ref > 0)


def cents(f0: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """``1200 log2(f0 / ref)`` where both are voiced, NaN elsewhere."""
    f0, ref = _pair(f0, ref)
    out = torch.full_like(f0, float("nan"))
    mask = (f0 > 0) & (ref > 0)
    out[mask] = 1200.0 * torch.log2(f0[mask] / ref[mask])
    return out


def f0_rmse_hz(pred: torch.Tensor, ref: torch.Tensor) -> float:
    pred, ref = _pair(pred, ref)
    mask = (pred > 0) & (ref > 0)
    return float(((pred[mask] - ref[mask]) ** 2).mean().sqrt()) if mask.any() else float("nan")


def f0_rmse_cents(pred: torch.Tensor, ref: torch.Tensor) -> float:
    c = cents(pred, ref)
    c = c[~torch.isnan(c)]
    return float(c.pow(2).mean().sqrt()) if c.numel() else float("nan")


def log_f0_corr(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """Pearson correlation of log F0 over frames voiced in both; NaN with fewer than three such frames or a flat
    contour."""
    pred, ref = _pair(pred, ref)
    mask = (pred > 0) & (ref > 0)
    if int(mask.sum()) < 3:
        return float("nan")
    a, b = pred[mask].log(), ref[mask].log()
    a, b = a - a.mean(), b - b.mean()
    den = float(a.norm() * b.norm())
    return float((a * b).sum()) / den if den > 0 else float("nan")


def pitch_metrics(pred: torch.Tensor, ref: torch.Tensor) -> dict:
    pred_, ref_ = _pair(pred, ref)
    voiced_ref = int((ref_ > 0).sum())
    return {
        "f0_rmse_hz": f0_rmse_hz(pred, ref),
        "f0_rmse_cents": f0_rmse_cents(pred, ref),
        "log_f0_corr": log_f0_corr(pred, ref),
        "voiced_overlap": float(both_voiced(pred, ref).sum()) / voiced_ref if voiced_ref else math.nan,
    }


__all__ = ["both_voiced", "cents", "f0_rmse_cents", "f0_rmse_hz", "log_f0_corr", "pitch_metrics"]
