# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Scales, chords, ranges and harmony lines: the music theory the conductor needs, as small pure functions.

* a note never leaves the key (:func:`snap_to_scale`, :func:`nearest_chord_tone`);
* a melody never leaves the singer's range, or is moved by octaves and, as a last resort, compressed towards
  the range centre (:func:`fit_to_range`), which is how "sing it even if the voice cannot support it" is done
  honestly: the note is transposed or the register is changed, not faked;
* harmony parts are built from chord tones with voice-leading rules (small motion, no parallel octaves between
  adjacent parts, parts stay within an octave of the lead) (:func:`harmonize`);
* a rising line for a build-up (:func:`rising_line`) and an octave-leap climax figure (:func:`climax_figure`),
  the two vocal gestures the chills literature names most often.
"""

from __future__ import annotations

from .analysis import NOTE_NAMES, scale_pitch_classes

CHORD_INTERVALS = {
    "": (0, 4, 7),
    "m": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "m7": (0, 3, 7, 10),
    "maj7": (0, 4, 7, 11),
    "sus4": (0, 5, 7),
    "dim": (0, 3, 6),
}
# Typical ranges in MIDI: a conductor picks the singer's measured range when it has one.
VOICE_RANGES = {
    "bass": (40, 64),
    "baritone": (45, 67),
    "tenor": (48, 72),
    "alto": (53, 77),
    "mezzo": (55, 79),
    "soprano": (60, 84),
    "whistle": (84, 100),
}


def parse_chord(name: str) -> tuple[int, tuple[int, ...]]:
    """``"F#m7"`` -> (root pitch class, intervals)."""
    root_name = name[:2] if len(name) > 1 and name[1] == "#" else name[:1]
    quality = name[len(root_name) :]
    return NOTE_NAMES.index(root_name), CHORD_INTERVALS.get(quality, CHORD_INTERVALS[""])


def chord_tones(name: str) -> tuple[int, ...]:
    root, intervals = parse_chord(name)
    return tuple((root + i) % 12 for i in intervals)


def snap_to_scale(midi: int, tonic: int, mode: str, prefer_up: bool = False) -> int:
    allowed = set(scale_pitch_classes(tonic, mode))
    if midi % 12 in allowed:
        return midi
    for step in range(1, 7):
        candidates = (midi + step, midi - step) if prefer_up else (midi - step, midi + step)
        for c in candidates:
            if c % 12 in allowed:
                return c
    return midi


def nearest_chord_tone(midi: int, chord: str, prefer_up: bool = False) -> int:
    tones = set(chord_tones(chord))
    for step in range(0, 7):
        candidates = (midi + step, midi - step) if prefer_up else (midi - step, midi + step)
        for c in candidates:
            if c % 12 in tones:
                return c
    return midi


def fit_to_range(melody: list[int], low: int, high: int, allow_compress: bool = True) -> tuple[list[int], str]:
    """
    Keep a melody inside ``[low, high]``: first try whole-octave transpositions (the line keeps its shape),
    then compress the notes that still stick out towards the range (shape mostly kept, intervals smaller).
    Returns the new melody and what was done ("as is", "octave -1", "compressed").
    """
    notes = [m for m in melody if m >= 0]
    if not notes:
        return melody, "as is"
    if min(notes) >= low and max(notes) <= high:
        return melody, "as is"
    best, best_shift, best_out = None, 0, None
    for shift in (0, -12, 12, -24, 24):
        shifted = [m + shift if m >= 0 else m for m in melody]
        inside = [m for m in shifted if m >= 0]
        out = sum(max(0, low - m) + max(0, m - high) for m in inside)
        if best is None or out < best:
            best, best_shift, best_out = out, shift, shifted
    if best == 0:
        return best_out, f"octave {best_shift // 12:+d}" if best_shift else "as is"
    if not allow_compress:
        return best_out, f"octave {best_shift // 12:+d}, still out of range"
    centre = (low + high) / 2
    span = (high - low) / 2
    inside = [m for m in best_out if m >= 0]
    spread = max(abs(m - centre) for m in inside)
    factor = min(1.0, span / max(spread, 1e-6))
    compressed = [int(round(centre + (m - centre) * factor)) if m >= 0 else m for m in best_out]
    return compressed, "compressed"


def harmonize(
    lead: list[int], chords: list[str], parts: int = 2, tonic: int = 0, mode: str = "major", below: bool = True
) -> list[list[int]]:
    """
    ``parts`` harmony lines for a lead melody (one chord name per note): each part takes the nearest chord tone
    a third or more away from the part above it, moves as little as possible from its previous note, and never
    lands in unison or octave with its neighbour. Returns the parts, highest first (all below the lead when
    ``below``; the first above it otherwise).
    """
    lines: list[list[int]] = [[] for _ in range(parts)]
    previous = [None] * parts
    for note, chord in zip(lead, chords, strict=True):
        if note < 0:
            for line in lines:
                line.append(-1)
            continue
        tones = chord_tones(chord)
        upper = note
        for p in range(parts):
            direction = -1 if below or p > 0 else 1
            candidates = []
            for step in range(3, 13):
                cand = upper + direction * step
                if cand % 12 in tones and cand % 12 != upper % 12:
                    candidates.append(cand)
            if not candidates:
                cand = snap_to_scale(upper + direction * 4, tonic, mode)
                candidates.append(cand)
            prev = previous[p]
            choice = min(
                candidates, key=lambda c: (abs(c - prev) if prev is not None else abs(c - upper), abs(c - upper))
            )
            lines[p].append(choice)
            previous[p] = choice
            upper = choice
    return lines


def rising_line(start: int, end: int, steps: int, tonic: int, mode: str) -> list[int]:
    """A scale-wise ascent from ``start`` to ``end`` in ``steps`` notes: the build-up gesture before a chorus."""
    if steps <= 1:
        return [snap_to_scale(end, tonic, mode)]
    out = []
    for i in range(steps):
        target = start + (end - start) * i / (steps - 1)
        out.append(snap_to_scale(int(round(target)), tonic, mode, prefer_up=True))
    return out


def climax_figure(anchor: int, chord: str, tonic: int, mode: str) -> list[int]:
    """Approach, octave leap, held top note on a chord tone, step down: the figure most associated with chills."""
    top = nearest_chord_tone(anchor + 12, chord, prefer_up=True)
    approach = snap_to_scale(top - 12 - 2, tonic, mode)
    return [approach, top - 12, top, top, snap_to_scale(top - 2, tonic, mode)]


def range_name(low: int, high: int) -> str:
    centre = (low + high) / 2
    return min(VOICE_RANGES, key=lambda k: abs((VOICE_RANGES[k][0] + VOICE_RANGES[k][1]) / 2 - centre))
