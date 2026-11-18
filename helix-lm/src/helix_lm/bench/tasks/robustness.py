# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Robustness: the source spoiled the ways real uploads are (noise at 20 and 10 dB SNR, a room, clipping), each
converted, and the result compared with the conversion of the clean source: how far its melody and its voice move.
A system that is robust gives nearly the same output."""

from __future__ import annotations

import math

import torch

from ..metrics.pitch import f0_rmse_cents
from ..metrics.similarity import speaker_similarity
from . import f0, load_item


def degrade(wave: torch.Tensor, sample_rate: int, kind: str, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = wave.flatten().float()
    if kind.startswith("noise"):
        snr = float(kind.split("-")[1])
        noise = torch.randn(x.numel(), generator=g)
        scale = x.pow(2).mean().sqrt() / noise.pow(2).mean().sqrt() / 10 ** (snr / 20)
        return x + scale * noise
    if kind == "room":
        from ...music.regularize import synthetic_room

        return synthetic_room(x, sample_rate, g)[0].flatten()[: x.numel()]
    if kind == "clip":
        top = float(x.abs().max()) * 0.3
        return x.clamp(-top, top)
    raise KeyError(kind)


KINDS = ("noise-20", "noise-10", "room", "clip")


def run(system, items, sample_rate: int, cache, seconds: float = 6.0, kinds=KINDS, embed=None) -> list[dict]:
    rows = []
    for it in items:
        source = load_item(it, sample_rate, seconds)
        reference = source

        def convert(x, key, it=it, reference=reference):
            return cache.make(system.name, "robustness", it.id, {"sr": sample_rate, "s": seconds, "kind": key},
                              lambda: system.convert(x, reference, sample_rate))

        clean = convert(source, "clean")
        fc = f0(clean, sample_rate)
        for k, kind in enumerate(kinds):
            out = convert(degrade(source, sample_rate, kind, seed=k), kind)
            n = min(out.numel(), clean.numel())
            rows.append({"item": it.id, "set": it.set, "singer": it.singer, "degradation": kind,
                         "f0_shift_cents": f0_rmse_cents(f0(out, sample_rate), fc),
                         "voice_kept": speaker_similarity(out[:n], clean[:n], sample_rate, embed)
                         if n > sample_rate // 10 else math.nan})
    return rows


__all__ = ["KINDS", "degrade", "run"]
