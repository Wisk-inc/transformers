# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""NUS-48E items: ``<singer>/sing/<song>.wav`` paired with ``<singer>/read/<song>.wav`` (the same lyrics spoken).
The pair makes speech-to-singing and singing-to-singing conversions measurable on the same words."""

from __future__ import annotations

from pathlib import Path

from .registry import Item


def items(root: str | Path) -> list[Item]:
    out = []
    for sung in sorted(Path(root).glob("*/sing/*.wav")):
        singer = sung.parent.parent.name
        read = sung.parent.parent / "read" / sung.name
        lyrics_file = sung.with_suffix(".txt")
        lyrics = lyrics_file.read_text(errors="replace").strip() if lyrics_file.exists() else ""
        out.append(Item(f"{singer}-{sung.stem}", sung, singer, "nus48e", lyrics=lyrics,
                        pair=read if read.exists() else None))
    return out


__all__ = ["items"]
