# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Speed: seconds of compute per second of audio (real-time factor) for a conversion, or a resynthesis for systems
that only resynthesise, and the peak accelerator memory. Never cached: a speed is measured every run."""

from __future__ import annotations

import math
import time

import torch

from . import load_item


def run(system, items, sample_rate: int, cache, seconds: float = 10.0, repeats: int = 2) -> list[dict]:
    rows = []
    for it in items[:3]:
        source = load_item(it, sample_rate, seconds)
        job = (lambda s=source: system.convert(s, s, sample_rate)) if "svc" in system.supports else \
              (lambda s=source: system.resynthesize(s, sample_rate))
        job()                                                       # warm-up: kernels, caches
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        began = time.perf_counter()
        for _ in range(repeats):
            job()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        took = (time.perf_counter() - began) / repeats
        rows.append({"item": it.id, "set": it.set, "singer": it.singer,
                     "rtf": took / max(source.numel() / sample_rate, 1e-9),
                     "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available()
                     else math.nan})
    return rows


__all__ = ["run"]
