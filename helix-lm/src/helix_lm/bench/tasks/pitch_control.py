# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Pitch control: a take converted into its own voice with the melody moved by whole semitones; the shift sung
(the median cent difference of the voiced frames) against the shift asked for."""

from __future__ import annotations

import math

import torch

from ..metrics.pitch import cents
from . import f0, load_item

SHIFTS = (-5.0, -2.0, 2.0, 5.0)


def run(system, items, sample_rate: int, cache, seconds: float = 6.0, shifts=SHIFTS) -> list[dict]:
    rows = []
    for it in items:
        source = load_item(it, sample_rate, seconds)
        fs = f0(source, sample_rate)
        errors = []
        for shift in shifts:
            out = cache.make(system.name, "pitch_control", it.id, {"sr": sample_rate, "s": seconds, "shift": shift},
                             lambda s=source, k=shift: system.convert(s, s, sample_rate, shift_semitones=k))
            c = cents(f0(out, sample_rate), fs)
            c = c[~torch.isnan(c)]
            if c.numel():
                errors.append(abs(float(c.median()) - 100.0 * shift))
        rows.append({"item": it.id, "set": it.set, "singer": it.singer,
                     "shift_error_cents": sum(errors) / len(errors) if errors else math.nan,
                     "shifts_measured": len(errors)})
    return rows


__all__ = ["SHIFTS", "run"]
