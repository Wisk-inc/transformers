# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Mel-cepstral distortion in dB, ``(10 / ln 10) * sqrt(2 * sum_d (c_d - c'_d)^2)`` per frame, without ``c0`` (the
level), averaged over frames aligned by DTW (or frame by frame when the timing is known to match).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .dtw import cost_matrix, dtw
from .mfcc import mel_cepstrum

SCALE = 10.0 / math.log(10.0) * math.sqrt(2.0)


def mcd_from_cepstra(a: torch.Tensor, b: torch.Tensor, align: bool = True, band: float | None = 0.2) -> float:
    a, b = np.asarray(a[:, 1:], np.float64), np.asarray(b[:, 1:], np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    if align:
        rows, cols, _ = dtw(cost_matrix(a, b), band=band)
        a, b = a[rows], b[cols]
    else:
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    return float(SCALE * np.sqrt(((a - b) ** 2).sum(1)).mean())


def _loud(c: torch.Tensor, below_db: float = 50.0) -> torch.Tensor:
    """Frames within ``below_db`` of the signal's loudest (``c0`` is a scaled log amplitude)."""
    if len(c) == 0:
        return torch.zeros(0, dtype=torch.bool)
    level_db = c[:, 0] * (20.0 / math.log(10.0))
    return level_db > level_db.max() - below_db


def mcd(pred: torch.Tensor, ref: torch.Tensor, sample_rate: int, align: bool = True, n_coeffs: int = 25) -> float:
    """MCD between two mono signals. Each signal's silent frames (50 dB under its own loudest) are left out, so
    a pause does not dominate the score, and with ``align`` the rest are matched by DTW."""
    cp = mel_cepstrum(pred, sample_rate, n_coeffs)
    cr = mel_cepstrum(ref, sample_rate, n_coeffs)
    if not align:
        n = min(len(cp), len(cr))
        keep = _loud(cr[:n])
        return mcd_from_cepstra(cp[:n][keep], cr[:n][keep], align=False)
    return mcd_from_cepstra(cp[_loud(cp)], cr[_loud(cr)], align=True)


__all__ = ["SCALE", "mcd", "mcd_from_cepstra"]
