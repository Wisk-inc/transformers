# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The input, unchanged. Resynthesis by it is perfect, and a conversion by it keeps the source's voice: every
content metric reads its best and voice similarity to the target reads what a failed conversion would."""

from __future__ import annotations

import torch


class Identity:
    name = "identity"
    supports = ("resynthesis", "svc", "pitch_control", "technique", "robustness", "longform", "speed", "register")

    def resynthesize(self, wave: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return wave.clone()

    def convert(self, source: torch.Tensor, reference: torch.Tensor, sample_rate: int, shift_semitones: float = 0.0):
        return source.clone()


__all__ = ["Identity"]
