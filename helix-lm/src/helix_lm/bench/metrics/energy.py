# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Frame loudness (10 ms RMS in dB) of two time-aligned signals: the error and the correlation of the contours,
on the frames where the reference is sounding."""

from __future__ import annotations

import math

import torch


def rms_db(wave: torch.Tensor, sample_rate: int, hop_ms: float = 10.0) -> torch.Tensor:
    hop = max(1, int(sample_rate * hop_ms / 1000))
    x = wave.flatten()[: wave.numel() // hop * hop].double().reshape(-1, hop)
    return 10 * torch.log10(x.pow(2).mean(-1) + 1e-12)


def energy_metrics(pred: torch.Tensor, ref: torch.Tensor, sample_rate: int, floor_db: float = 50.0) -> dict:
    p, r = rms_db(pred, sample_rate), rms_db(ref, sample_rate)
    n = min(p.numel(), r.numel())
    p, r = p[:n], r[:n]
    keep = r > r.max() - floor_db if n else torch.zeros(0, dtype=torch.bool)
    if int(keep.sum()) < 3:
        return {"rms_error_db": math.nan, "loudness_corr": math.nan}
    a, b = p[keep] - p[keep].mean(), r[keep] - r[keep].mean()
    den = float(a.norm() * b.norm())
    return {"rms_error_db": float((p[keep] - r[keep]).abs().mean()),
            "loudness_corr": float((a * b).sum()) / den if den > 0 else math.nan}


__all__ = ["energy_metrics", "rms_db"]
