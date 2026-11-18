# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Consonant clarity: the share of the reference's onset energy the output keeps at the reference's onsets (5 ms
spectral flux, :func:`helix_lm.music.artifacts.consonant_sharpness`), capped at 1. A smeared consonant is the
first thing that makes sung lyrics hard to follow."""

from __future__ import annotations

import math

import torch

from ...music.artifacts import consonant_sharpness


def clarity_metrics(pred: torch.Tensor, ref: torch.Tensor, sample_rate: int) -> dict:
    r = consonant_sharpness(pred.flatten().float(), ref.flatten().float(), sample_rate)
    if not r["onsets"]:
        return {"consonant_clarity": math.nan, "consonant_onsets": 0}
    return {"consonant_clarity": min(1.0, float(r["ratio"])), "consonant_onsets": int(r["onsets"])}


__all__ = ["clarity_metrics"]
