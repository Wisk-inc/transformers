# Copyright 2026 Nathan. Apache-2.0.
"""
Writes that a kill cannot leave half-done.

A run that is stopped -- the Stop button, a supervisor restart, the machine losing power -- can be
stopped in the middle of writing a file. A half-written file that the run reads back on its next start
is worse than none: it crashes every restart, and a supervised run then gives up. So every file a run
reads back is written beside its destination first and renamed into place, which is atomic: readers see
the old file or the new one, never a mixture.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(obj, path: str | os.PathLike) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    try:
        torch.save(obj, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_copy(source: str | os.PathLike, destination: str | os.PathLike) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(destination)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: str | os.PathLike, default=None):
    """The file's JSON, or ``default`` if it is missing or damaged -- never an exception."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def remove_stale_temporaries(directory: str | os.PathLike) -> int:
    """Delete temporaries left by writes that were killed part-way (their process is gone). Returns bytes freed."""
    freed = 0
    for leftover in Path(directory).glob(".*.tmp"):
        try:
            pid = int(leftover.name.rsplit(".", 2)[-2])
        except (ValueError, IndexError):
            continue
        if pid == os.getpid() or Path(f"/proc/{pid}").exists():
            continue
        try:
            freed += leftover.stat().st_size
            leftover.unlink()
        except OSError:
            pass
    return freed
