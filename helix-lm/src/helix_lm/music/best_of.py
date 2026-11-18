# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Best of N: a flow model's outputs differ from seed to seed, and the failures (a glitch, a wrong note, a lost
voice) are seed-specific. Generating a few and keeping the one with the best score on the benchmark's own checks
removes most of them at the cost of N generations."""

from __future__ import annotations

import math

import torch

from ..audio.yin import yin_f0
from ..bench.metrics.glitch import count_glitches
from ..bench.metrics.pitch import f0_rmse_cents
from ..bench.metrics.quality import quality_proxy


def score(out: torch.Tensor, sample_rate: int, target_f0: torch.Tensor | None = None, f0_rate: float = 100.0,
          similarity=None) -> float:
    """Higher is better: quality proxy, minus a pitch-error penalty (per 50 cents) and a glitch penalty, plus voice
    similarity when a ``similarity(out)`` is given."""
    s = quality_proxy(out, sample_rate)["quality_proxy"]
    if target_f0 is not None:
        err = f0_rmse_cents(yin_f0(out.flatten(), sample_rate, f0_rate), target_f0)
        if not math.isnan(err):
            s -= err / 50.0
    glitches, _ = count_glitches(out, sample_rate)
    s -= 0.2 * glitches
    if similarity is not None:
        s += similarity(out)
    return s


def best_of(generate, n: int = 4, scorer=None) -> tuple[torch.Tensor, list[float]]:
    """``generate(seed) -> audio``; returns the best output and every candidate's score."""
    outs, scores = [], []
    for seed in range(n):
        out = generate(seed)
        outs.append(out)
        scores.append(scorer(out) if scorer else 0.0)
    best = max(range(n), key=lambda i: scores[i])
    return outs[best], scores


__all__ = ["best_of", "score"]
