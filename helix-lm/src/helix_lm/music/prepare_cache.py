# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Per-clip records of the prepare stage, kept beside each clip's outputs.

Until 2.4 a source's clips were all measured again whenever its ``manifest.json`` was missing, and that file is
written only after the last clip. On molab, where a session ends after 12 hours, a large source (300 hours of
Emilia speech) could then never be finished: every session started it again from its first file. Each clip now
leaves ``<stem>.record.json`` when it is done, and a clip whose record and outputs are on disk is read back
instead of being measured again.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

RECORD = ".record.json"
FIELDS = ("path", "seconds", "tags", "analysis", "f0_path", "accepted", "reason")


def record_path(out_dir: Path, stem: str) -> Path:
    return Path(out_dir) / f"{stem}{RECORD}"


def outputs_present(out_dir: Path, stem: str) -> bool:
    out_dir = Path(out_dir)
    return (out_dir / f"{stem}.vocal.wav").exists() and (out_dir / f"{stem}.f0.pt").exists()


def load_record(out_dir: Path, source: Path) -> dict | None:
    """The finished record of ``source``, or None when it has to be prepared (no record, a record of another file
    with the same name, a torn record, or an accepted clip whose audio or pitch file is gone)."""
    source = Path(source)
    path = record_path(out_dir, source.stem)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if data.get("path") != str(source) or not all(k in data for k in FIELDS):
        return None
    if data.get("accepted") and not outputs_present(out_dir, source.stem):
        return None
    return {k: data[k] for k in FIELDS}


def save_record(out_dir: Path, record: dict) -> Path:
    """Write the record through a temporary file, so a stop in the middle never leaves a torn one."""
    path = record_path(out_dir, Path(record["path"]).stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({k: record[k] for k in FIELDS}, default=str))
    os.replace(tmp, path)
    return path


def duplicate_stems(files: list) -> dict:
    """``{file: first file with the same name}`` for every file whose stem was already taken. Two files with one
    name would write over each other's audio and pitch."""
    seen, dup = {}, {}
    for f in files:
        stem = Path(f).stem
        if stem in seen:
            dup[f] = seen[stem]
        else:
            seen[stem] = f
    return dup


def progress(out_dir: Path, files: list) -> tuple[int, int]:
    """``(clips with a finished record, clips)`` for a source: how far an interrupted prepare got."""
    done = sum(load_record(out_dir, Path(f)) is not None for f in files)
    return done, len(files)


__all__ = ["FIELDS", "RECORD", "duplicate_stems", "load_record", "outputs_present", "progress", "record_path",
           "save_record"]
