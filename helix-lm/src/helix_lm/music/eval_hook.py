# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The benchmark during training: a few validation clips resynthesised and converted by the current checkpoint and
scored with the suite's metrics, in a couple of minutes on the training GPU. The molab program calls this at each
checkpoint and keeps the checkpoint with the best composite (:mod:`helix_lm.music.selection`)."""

from __future__ import annotations

import math

import torch

from ..bench.cache import OutputCache
from ..bench.data.registry import Item
from ..bench.tasks import runner
from .composite import composite


def quick_score(system, paths: list, sample_rate: int = 24000, seconds: float = 6.0) -> dict:
    """``{"composite", "rows", "means"}`` for a system over the clips at ``paths`` (singers taken from the parent
    folder names, so the conversion pairs cross singers)."""
    items = [Item(f"val-{k}", p, p.parent.name if hasattr(p, "parent") else "s", "validation")
             for k, p in enumerate(paths)]
    rows = []
    cache = OutputCache(None)
    with torch.no_grad():
        rows += runner("resynthesis")(system, items, sample_rate, cache, seconds=seconds)
        rows += runner("svc")(system, items, sample_rate, cache, seconds=seconds)
    means: dict = {}
    for r in rows:
        for k, v in r.items():
            if isinstance(v, float) and not math.isnan(v):
                means.setdefault(k, []).append(v)
    means = {k: sum(v) / len(v) for k, v in means.items()}
    return {"composite": composite(means), "rows": rows, "means": means}


__all__ = ["quick_score"]
