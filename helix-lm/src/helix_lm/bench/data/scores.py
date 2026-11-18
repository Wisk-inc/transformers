# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Written notes: names to MIDI numbers, notes to per-frame targets, and a small reader for type 0 and 1 MIDI files
(note on and off, tempo changes), enough for the benchmark scores without a dependency.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import torch

NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
NOTE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")


def note_name_to_midi(name: str) -> int | None:
    """``C#4`` -> 61, ``Db4`` -> 61, ``A4`` -> 69; None for a rest or anything else."""
    m = NOTE.match(name.strip())
    if not m:
        return None
    letter, accidental, octave = m.group(1).upper(), m.group(2), int(m.group(3))
    return 12 * (octave + 1) + NAMES[letter] + (1 if accidental == "#" else -1 if accidental == "b" else 0)


def note_frames(notes, frames: int, frame_rate: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``(midi per frame (0 = rest), note onset per frame (bool))``."""
    midi = torch.zeros(frames)
    onset = torch.zeros(frames, dtype=torch.bool)
    for start, end, m in notes:
        a, b = int(round(start * frame_rate)), int(round(end * frame_rate))
        if 0 <= a < frames:
            midi[a : min(frames, b)] = float(m)
            onset[a] = True
    return midi, onset


def _varlen(data: bytes, i: int) -> tuple[int, int]:
    value = 0
    while True:
        b = data[i]
        i += 1
        value = (value << 7) | (b & 0x7F)
        if not b & 0x80:
            return value, i


def read_midi(path: str | Path, track: int | None = None) -> list[tuple[float, float, int]]:
    """``(onset_s, offset_s, midi)`` of every note (of one track, or all), tempo changes applied."""
    data = Path(path).read_bytes()
    if data[:4] != b"MThd":
        raise ValueError("not a MIDI file")
    _, ntracks, division = struct.unpack(">HHH", data[8:14])
    i, tracks = 14, []
    while i < len(data) and len(tracks) < ntracks:
        size = struct.unpack(">I", data[i + 4 : i + 8])[0]
        tracks.append(data[i + 8 : i + 8 + size])
        i += 8 + size
    events = []                                             # (tick, kind, a, b, track)
    for k, chunk in enumerate(tracks):
        j, tick, status = 0, 0, 0
        while j < len(chunk):
            delta, j = _varlen(chunk, j)
            tick += delta
            b = chunk[j]
            if b == 0xFF:
                kind, j = chunk[j + 1], j + 2
                size, j = _varlen(chunk, j)
                if kind == 0x51 and size == 3:
                    events.append((tick, "tempo", int.from_bytes(chunk[j : j + 3], "big"), 0, k))
                j += size
                continue
            if b in (0xF0, 0xF7):
                size, j = _varlen(chunk, j + 1)
                j += size
                continue
            if b & 0x80:
                status, j = b, j + 1
            kind = status & 0xF0
            if kind in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                a1, a2 = chunk[j], chunk[j + 1]
                j += 2
                if kind == 0x90 and a2 > 0:
                    events.append((tick, "on", a1, a2, k))
                elif kind == 0x80 or (kind == 0x90 and a2 == 0):
                    events.append((tick, "off", a1, 0, k))
            else:
                j += 1
    events.sort(key=lambda e: (e[0], e[1] != "tempo"))
    tempo, last_tick, seconds = 500000, 0, 0.0
    open_notes, out = {}, []
    for tick, kind, a, _, k in events:
        seconds += (tick - last_tick) * tempo / 1e6 / division
        last_tick = tick
        if kind == "tempo":
            tempo = a
        elif track is None or k == track:
            if kind == "on":
                open_notes[(a, k)] = seconds
            elif (a, k) in open_notes:
                out.append((round(open_notes.pop((a, k)), 4), round(seconds, 4), int(a)))
    return sorted(out)


__all__ = ["note_frames", "note_name_to_midi", "read_midi"]
