# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Written notes placed on a recording: the pitch the notes call for, frame by frame, aligned to the measured pitch
by DTW (in cents, so a singer a little flat still aligns), and each note's onset and offset read off the path. Sets
that have a score but no timing (or timing from another performance) become usable for the note metrics.
"""

from __future__ import annotations

import numpy as np
import torch

from ..bench.metrics.dtw import dtw


def align(notes, f0: torch.Tensor, frame_rate: float, band: float = 0.3, unvoiced_cost: float = 600.0) -> list:
    """``notes`` with their times replaced by where they are sung in ``f0`` (Hz per frame, 0 unvoiced)."""
    if not notes:
        return []
    total = max(e for _, e, _ in notes)
    n = max(1, int(round(total * frame_rate)))
    expect = np.full(n, np.nan)
    owner = np.full(n, -1)
    for k, (s, e, m) in enumerate(notes):
        a, b = int(round(s * frame_rate)), min(n, int(round(e * frame_rate)))
        expect[a:b] = 100.0 * m
        owner[a:b] = k
    f = f0.flatten().double().numpy()
    sung = np.where(f > 0, 1200 * np.log2(np.maximum(f, 1e-9) / 440.0) + 6900.0, np.nan)
    e_col, s_row = expect[:, None], sung[None, :]
    cost = np.abs(e_col - s_row)
    cost = np.where(np.isnan(e_col) & np.isnan(s_row), 0.0, cost)
    cost = np.where(np.isnan(cost), unvoiced_cost, np.minimum(cost, 1200.0))
    rows, cols, _ = dtw(cost, band=band)
    out = []
    for k, (_, _, m) in enumerate(notes):
        sel = cols[owner[rows] == k]
        if sel.size:
            out.append((round(sel.min() / frame_rate, 4), round((sel.max() + 1) / frame_rate, 4), m))
    return out


__all__ = ["align"]
