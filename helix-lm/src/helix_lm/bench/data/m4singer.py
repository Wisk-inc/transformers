# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""M4Singer items: folders ``<singer>#<song>/<n>.wav`` (singers named by voice type, ``Alto-1``, ``Tenor-6``), with
``meta.json`` at the root listing each item's notes (``notes``, ``notes_dur`` in seconds, ``is_slur``) when
present. Notes come back as ``(onset_s, offset_s, midi)``."""

from __future__ import annotations

import json
from pathlib import Path

from .registry import Item


def _notes(entry: dict) -> list:
    notes, durs = entry.get("notes") or [], entry.get("notes_dur") or []
    out, t = [], 0.0
    for midi, d in zip(notes, durs, strict=False):
        if int(midi) > 0:
            out.append((round(t, 4), round(t + float(d), 4), int(midi)))
        t += float(d)
    return out


def items(root: str | Path, singers: tuple | None = None) -> list[Item]:
    root = Path(root)
    meta = {}
    if (root / "meta.json").exists():
        for entry in json.loads((root / "meta.json").read_text(encoding="utf-8")):
            meta[str(entry.get("item_name", ""))] = entry
    out = []
    for wav in sorted(root.glob("*#*/*.wav")):
        singer, song = wav.parent.name.split("#", 1)
        if singers and singer not in singers:
            continue
        key = f"{singer}#{song}#{wav.stem}"
        entry = meta.get(key, {})
        out.append(Item(key, wav, singer, "m4singer", notes=_notes(entry), lyrics=entry.get("txt", "")))
    return out


__all__ = ["items"]
