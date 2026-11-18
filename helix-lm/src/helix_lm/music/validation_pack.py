# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""A fixed validation pack: clips of singers who are never trained on, chosen by a hash of their path (so the pack
is the same on every machine and in every session), balanced over singers, plus the synthetic set. Every
checkpoint is measured on exactly these clips, so scores of checkpoints compare."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _rank(path: str) -> str:
    return hashlib.sha1(path.encode()).hexdigest()


def choose(records: list[dict], held_singers: set, per_singer: int = 4, limit: int = 60) -> list[dict]:
    """From prepared clip records (``path``, ``tags.singer``, ``accepted``), the pack's clips."""
    by: dict = {}
    for r in records:
        singer = (r.get("tags") or {}).get("singer", "")
        if r.get("accepted") and singer in held_singers:
            by.setdefault(singer, []).append(r)
    out = []
    for singer in sorted(by):
        out += sorted(by[singer], key=lambda r: _rank(str(r["path"])))[:per_singer]
    return out[:limit]


def save(pack: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"path": str(r["path"]), "singer": (r.get("tags") or {}).get("singer", "")}
                                for r in pack], indent=1))
    return path


def load(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())


__all__ = ["choose", "load", "save"]
