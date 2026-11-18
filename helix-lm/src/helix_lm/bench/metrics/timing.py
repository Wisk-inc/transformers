# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Note onsets of a performance against the written onsets or a reference's: the error of each matched onset and
the share of onsets found within 100 ms."""

from __future__ import annotations

import math

import torch

from ...music.analysis import onset_envelope


def detect_onsets(wave: torch.Tensor, sample_rate: int, hop: int = 256, threshold: float = 0.3,
                  min_gap_s: float = 0.08) -> torch.Tensor:
    """Onset times in seconds: peaks of the spectral flux above ``threshold`` of its maximum."""
    env = onset_envelope(wave.flatten(), sample_rate, hop=hop)
    gap = max(1, int(min_gap_s * sample_rate / hop))
    peaks, last = [], -gap
    for i in range(1, env.numel() - 1):
        if env[i] >= threshold and env[i] >= env[i - 1] and env[i] > env[i + 1] and i - last >= gap:
            peaks.append(i)
            last = i
    return torch.tensor(peaks, dtype=torch.float64) * hop / sample_rate


def onset_metrics(found: torch.Tensor, expected: torch.Tensor, window_s: float = 0.1) -> dict:
    found, expected = torch.as_tensor(found, dtype=torch.float64), torch.as_tensor(expected, dtype=torch.float64)
    if expected.numel() == 0:
        return {"onset_error_ms": math.nan, "onset_hit_rate": math.nan}
    if found.numel() == 0:
        return {"onset_error_ms": math.nan, "onset_hit_rate": 0.0}
    gaps = (expected[:, None] - found[None]).abs().min(1).values
    hits = gaps <= window_s
    return {"onset_error_ms": float(gaps[hits].mean() * 1000) if hits.any() else math.nan,
            "onset_hit_rate": float(hits.double().mean())}


__all__ = ["detect_onsets", "onset_metrics"]
