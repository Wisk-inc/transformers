# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
A signal-based quality proxy in [0, 1], and a hook for a learned MOS predictor.

The proxy multiplies four checks that each catch a failure listeners hear: clicks and dropouts, clipping, a
voice with no periodic energy (harmonics-to-noise ratio of the voiced frames), and a noisy silence (the floor
between phrases). It is not a listening test and is never reported as MOS. A real predictor (UTMOS for speech,
SingMOS-type models for singing) is loaded by :func:`load_predictor` when the machine can fetch it.
"""

from __future__ import annotations

import math

import torch

from ...audio.singing import estimate_f0
from .glitch import count_glitches
from .periodicity import hnr_db


def quality_proxy(wave: torch.Tensor, sample_rate: int) -> dict:
    wave = wave.flatten().float()
    seconds = max(wave.numel() / sample_rate, 1e-6)
    n, g = count_glitches(wave, sample_rate)
    glitches = n / seconds * 60
    hop = int(0.02 * sample_rate)
    frames = wave[: wave.numel() // hop * hop].reshape(-1, hop)
    level = 10 * torch.log10(frames.pow(2).mean(-1) + 1e-12) if frames.numel() else torch.zeros(1)
    voiced = estimate_f0(wave[None], sample_rate, sample_rate / hop)[0][: level.numel()] > 0
    pauses = level[~voiced] if voiced.numel() == level.numel() else level[:0]
    # the floor between phrases, from the unvoiced frames; a take with no pause has no floor to judge
    floor = float(pauses.median()) - float(level.max()) if pauses.numel() >= 5 else -90.0
    hnr = hnr_db(wave, sample_rate)
    parts = {
        "clean": math.exp(-glitches / 10.0),
        "unclipped": math.exp(-g["clipped_fraction"] * 1000.0),
        "periodic": 1.0 / (1.0 + math.exp(-(hnr - 5.0) / 3.0)) if not math.isnan(hnr) else 0.5,
        "quiet_floor": 1.0 / (1.0 + math.exp((floor + 40.0) / 6.0)),
    }
    score = 1.0
    for v in parts.values():
        score *= v
    return {"quality_proxy": score, **{f"quality_{k}": v for k, v in parts.items()}}


def load_predictor(name: str = "utmos22_strong"):
    """A learned MOS predictor ``f(wave (1, T) at 16 kHz) -> score``, or None when it cannot be loaded here. UTMOS
    was trained on speech: on singing it ranks, it does not rate."""
    try:
        return torch.hub.load("tarepan/SpeechMOS:v1.2.0", name, trust_repo=True)
    except Exception:
        return None


__all__ = ["load_predictor", "quality_proxy"]
