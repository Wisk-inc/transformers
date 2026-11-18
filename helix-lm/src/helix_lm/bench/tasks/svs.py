# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Singing from written notes in a reference voice. Measured against the notes (the share of frames within 50
cents, each note's error, onsets of notes that follow a rest, vibrato present) and, where the set has the singer's
own recording of those notes, against it (MCD after DTW, pitch)."""

from __future__ import annotations

from ..metrics.mcd import mcd
from ..metrics.pitch import pitch_metrics
from ..metrics.timing import detect_onsets, onset_metrics
from ..metrics.vibrato import vibrato
from . import F0_RATE, f0, level, load_item, output_quality


def _voice_for(item, items):
    same = [i for i in items if i.singer == item.singer and i.id != item.id]
    return same[0] if same else item


def run(system, items, sample_rate: int, cache, seconds: float = 8.0) -> list[dict]:
    rows = []
    for it in [i for i in items if i.notes]:
        notes = [n for n in it.notes if n[0] < seconds]
        if not notes:
            continue
        voice = load_item(_voice_for(it, items), sample_rate, seconds)
        length = min(seconds, max(e for _, e, _ in notes) + 0.2)
        out = cache.make(system.name, "svs", it.id, {"sr": sample_rate, "s": seconds},
                         lambda v=voice, n=notes, length=length: system.sing(n, sample_rate, v, length))
        fo = f0(out, sample_rate)
        row = {"item": it.id, "set": it.set, "singer": it.singer}
        from ..metrics.semitone import note_metrics

        row.update(note_metrics(fo, notes, F0_RATE))
        after_rest = [s for k, (s, _, _) in enumerate(notes) if k == 0 or s - notes[k - 1][1] > 0.05]
        row.update(onset_metrics(detect_onsets(out, sample_rate), after_rest))
        row["vibrato_present"] = float(vibrato(fo, F0_RATE)["present"])
        recording = load_item(it, sample_rate, seconds)
        row["mcd_db"] = mcd(level(out, sample_rate), level(recording, sample_rate), sample_rate, align=True)
        row.update({k: v for k, v in pitch_metrics(fo, f0(recording, sample_rate)).items() if k != "voiced_overlap"})
        row.update(output_quality(out, sample_rate))
        rows.append(row)
    return rows


__all__ = ["run"]
