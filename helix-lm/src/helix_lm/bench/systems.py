# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The systems a benchmark runs: the baselines, and TriStream from a molab root (its codec and its singing model, as
``helix_lm.scripts.tristream_molab`` writes them).

TriStream's three jobs here: resynthesis is the codec's round trip (what every TriStream output passes through);
conversion is the converter's editing path; singing from notes is the guide-and-convert route (a neutral guide
voice rendered from the notes, then converted into the reference voice), which is how this package sings a score
until a dedicated score frontend is trained. A root without the trained files gives no TriStream system rather
than a pass-through that would be scored as if it were one.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .baselines import BASELINES


class TriStream:
    name = "tristream"
    supports = ("resynthesis", "svc", "svs", "pitch_control", "technique", "robustness", "longform", "speed",
                "register")

    def __init__(self, converter, steps: int = 16) -> None:
        self.converter, self.steps = converter, steps

    @classmethod
    def from_root(cls, root: str | Path, size: str = "ts-765m", device=None) -> TriStream | None:
        from ..music.conversion import load_converter

        converter = load_converter(root, size, device)
        return None if converter.model is None else cls(converter)

    def resynthesize(self, wave: torch.Tensor, sample_rate: int) -> torch.Tensor:
        codec = self.converter.codec
        device = next(codec.parameters()).device
        with torch.no_grad():
            frames = codec.listen(wave.flatten()[None].to(device))
            out = codec.decode_frames(frames)[0].float().cpu()
        return out.flatten()[: wave.numel()]

    def convert(self, source: torch.Tensor, reference: torch.Tensor, sample_rate: int, shift_semitones: float = 0.0):
        from ..audio.singing import estimate_f0

        rate = self.converter.codec.config.frame_rate
        f0 = estimate_f0(source.flatten()[None], sample_rate, rate)[0] * 2 ** (shift_semitones / 12.0)
        speaker = self.converter.embed(reference, sample_rate)
        out, _ = self.converter.convert(source.flatten(), f0, speaker, sample_rate, steps=self.steps,
                                        reference=reference)
        return out.flatten().float().cpu()

    def sing(self, notes, sample_rate: int, reference: torch.Tensor, seconds: float | None = None) -> torch.Tensor:
        from .data.synthetic import render

        seconds = seconds or (max(e for _, e, _ in notes) + 0.2 if notes else 1.0)
        guide, _ = render(notes, sample_rate, "a", "mid", seconds)
        return self.convert(guide, reference, sample_rate)


def make(name: str, root: str | Path | None = None, size: str = "ts-765m"):
    """A system by name: a baseline, or ``tristream`` (needs ``root``); None when TriStream is not trained yet."""
    if name in BASELINES:
        return BASELINES[name]()
    if name == "tristream":
        return TriStream.from_root(root, size) if root else None
    raise KeyError(f"no system {name!r}; there are {', '.join([*BASELINES, 'tristream'])}")


__all__ = ["TriStream", "make"]
