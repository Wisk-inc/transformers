# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The benchmark tasks. Each ``run(system, items, sample_rate, cache, **options)`` returns one row of metrics per item
(or pair), with ``item``, ``set`` and ``singer`` filled in. Shared measurements live here: pitch tracks are YIN at
100 frames a second, and signals are compared at -23 LUFS.
"""

from __future__ import annotations

import torch

from ...audio.yin import yin_f0
from ..audio_io import load
from ..loudness import normalize
from ..metrics.clarity import clarity_metrics
from ..metrics.energy import energy_metrics
from ..metrics.ffe import ffe_metrics
from ..metrics.glitch import glitch_metrics
from ..metrics.pitch import pitch_metrics
from ..metrics.quality import quality_proxy
from ..metrics.semitone import note_metrics
from ..metrics.voicing import vuv_metrics

F0_RATE = 100.0
TASK_NAMES = ("resynthesis", "svc", "svs", "pitch_control", "technique", "robustness", "longform", "speed", "register")


def f0(wave: torch.Tensor, sample_rate: int) -> torch.Tensor:
    return yin_f0(wave.flatten().float(), sample_rate, F0_RATE)


def load_item(item, sample_rate: int, seconds: float | None = None) -> torch.Tensor:
    wave = load(item.path, sample_rate)
    return wave[: int(seconds * sample_rate)] if seconds else wave


def level(wave: torch.Tensor, sample_rate: int) -> torch.Tensor:
    return normalize(wave.flatten().float(), sample_rate, -23.0)


def content(out: torch.Tensor, source: torch.Tensor, sample_rate: int, notes=None) -> dict:
    """What a conversion or resynthesis must keep of its source: the melody, the voicing, the dynamics, the
    consonants and, given notes, the written pitches."""
    fo, fs = f0(out, sample_rate), f0(source, sample_rate)
    row = {**pitch_metrics(fo, fs), **vuv_metrics(fo, fs), **ffe_metrics(fo, fs)}
    row.update(energy_metrics(level(out, sample_rate), level(source, sample_rate), sample_rate))
    row.update(clarity_metrics(out, source, sample_rate))
    if notes:
        row.update(note_metrics(fo, notes, F0_RATE))
    return row


def output_quality(out: torch.Tensor, sample_rate: int) -> dict:
    return {**quality_proxy(out, sample_rate), **glitch_metrics(out, sample_rate)}


def runner(name: str):
    import importlib

    if name not in TASK_NAMES:
        raise KeyError(f"no task {name!r}; there are {', '.join(TASK_NAMES)}")
    return importlib.import_module(f"{__name__}.{name}").run


__all__ = ["F0_RATE", "TASK_NAMES", "content", "f0", "level", "load_item", "output_quality", "runner"]
