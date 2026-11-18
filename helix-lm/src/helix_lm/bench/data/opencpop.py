# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Opencpop items from ``segments/transcriptions.txt``, one line a segment:
``id|text|phonemes|notes|note_durations|phoneme_durations|slurs`` (notes such as ``C#4/Db4``, ``rest``), the audio
in ``segments/wavs/<id>.wav``. Notes are rebuilt as ``(onset_s, offset_s, midi)`` by adding up the note durations
of consecutive phonemes that share a note. ``test.txt`` lists the corpus's own test segments.
"""

from __future__ import annotations

from pathlib import Path

from .registry import Item
from .scores import note_name_to_midi


def parse_line(line: str) -> dict | None:
    parts = line.strip().split("|")
    if len(parts) < 6:
        return None
    seg, text, phones, notes, durs = parts[0], parts[1], parts[2].split(), parts[3].split(), parts[4].split()
    if not (len(phones) == len(notes) == len(durs)):
        return None
    events, t, last = [], 0.0, None
    for note, dur in zip(notes, durs, strict=True):
        d = float(dur)
        key = (note, d)
        if key != last:                             # a new note starts (phonemes of one note repeat its duration)
            if note != "rest":
                midi = note_name_to_midi(note.split("/")[0])
                if midi is not None:
                    events.append((round(t, 4), round(t + d, 4), midi))
            t += d
            last = key
    return {"id": seg, "text": text, "notes": events}


def items(root: str | Path, split: str = "test") -> list[Item]:
    root = Path(root)
    lines_file = root / "segments" / "transcriptions.txt"
    if not lines_file.exists():
        return []
    wanted = None
    listed = root / "segments" / f"{split}.txt"
    if split in ("test", "train") and listed.exists():
        wanted = {line.split("|")[0].strip() for line in listed.read_text().splitlines() if line.strip()}
    out = []
    for line in lines_file.read_text(encoding="utf-8", errors="replace").splitlines():
        seg = parse_line(line)
        if seg is None or (wanted is not None and seg["id"] not in wanted):
            continue
        wav = root / "segments" / "wavs" / f"{seg['id']}.wav"
        if wav.exists():
            out.append(Item(seg["id"], wav, "opencpop", "opencpop", notes=seg["notes"], lyrics=seg["text"]))
    return out


__all__ = ["items", "parse_line"]
