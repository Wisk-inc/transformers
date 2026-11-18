# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Pitch against the written notes: the share of sung frames within 50 cents of their note, and the error of each
note (the median over its frames, so vibrato around the note is not counted as error).
"""

from __future__ import annotations

import math

import torch


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69.0) / 12.0)


def note_track(notes, frames: int, frame_rate: float) -> torch.Tensor:
    """``notes`` as ``(onset_s, offset_s, midi)`` -> target Hz per frame, 0 between notes."""
    target = torch.zeros(frames, dtype=torch.float64)
    for onset, offset, midi in notes:
        a, b = int(round(onset * frame_rate)), int(round(offset * frame_rate))
        target[max(0, a) : max(0, min(frames, b))] = midi_to_hz(float(midi))
    return target


def note_metrics(f0: torch.Tensor, notes, frame_rate: float, tolerance_cents: float = 50.0) -> dict:
    f0 = f0.flatten().double()
    target = note_track(notes, f0.numel(), frame_rate)
    sung = (f0 > 0) & (target > 0)
    if not sung.any():
        return {"note_accuracy": math.nan, "note_error_cents": math.nan, "notes_sung": 0}
    err = 1200 * torch.log2(f0[sung] / target[sung])
    per_note = []
    for onset, offset, midi in notes:
        a, b = int(round(onset * frame_rate)), int(round(offset * frame_rate))
        seg = f0[max(0, a) : min(f0.numel(), b)]
        seg = seg[seg > 0]
        if seg.numel():
            per_note.append(abs(float(1200 * torch.log2(seg.median() / midi_to_hz(float(midi))))))
    return {"note_accuracy": float((err.abs() <= tolerance_cents).double().mean()),
            "note_error_cents": sum(per_note) / len(per_note) if per_note else math.nan, "notes_sung": len(per_note)}


__all__ = ["midi_to_hz", "note_metrics", "note_track"]
