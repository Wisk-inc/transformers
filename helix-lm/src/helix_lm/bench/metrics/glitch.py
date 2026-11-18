# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Clicks and dropouts per minute, clipping and the codec's frame-rate buzz, from the detectors the failsafe already
uses (:mod:`helix_lm.music.effects`, :mod:`helix_lm.music.artifacts`). Events in the first and last 50 ms are left
out: the dropout detector compares each point with the level 20 ms before it, wrapping around the array, so every
clip would otherwise get one at its start."""

from __future__ import annotations

import torch

from ...music.artifacts import frame_rate_buzz
from ...music.effects import detect_glitches

EDGE_SECONDS = 0.05


def count_glitches(wave: torch.Tensor, sample_rate: int, edge_s: float = EDGE_SECONDS) -> tuple[int, dict]:
    wave = wave.flatten().float()
    g = detect_glitches(wave, sample_rate)
    lo, hi = int(edge_s * sample_rate), wave.numel() - int(edge_s * sample_rate)

    def inside(pos: torch.Tensor) -> int:
        return int(((pos >= lo) & (pos < hi)).sum())

    return inside(g["clicks"]) + inside(g["dropouts"]), g


def glitch_metrics(wave: torch.Tensor, sample_rate: int, frame_rate: float = 25.0) -> dict:
    wave = wave.flatten().float()
    n, g = count_glitches(wave, sample_rate)
    minutes = max(wave.numel() / sample_rate / 60.0, 1e-9)
    buzz = frame_rate_buzz(wave, sample_rate, frame_rate)["buzz_db"]
    return {"glitches_per_minute": n / minutes, "clipped_share": g["clipped_fraction"], "buzz_db": buzz}


__all__ = ["EDGE_SECONDS", "count_glitches", "glitch_metrics"]
