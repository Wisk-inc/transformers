# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""A benchmark manifest: the exact items a result was measured on (path, singer, size and a checksum of the
audio), so two results can be compared only when they were measured on the same items."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from .registry import Item

VERSION = 1


def checksum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:16]


def build(items: list[Item], name: str) -> dict:
    rows = []
    for it in items:
        row = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(it).items()}
        row["sha"] = checksum(Path(it.path))
        rows.append(row)
    digest = hashlib.sha256(json.dumps([r["sha"] for r in rows]).encode()).hexdigest()[:16]
    return {"version": VERSION, "name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "items": rows,
            "digest": digest}


def save(manifest: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1))
    return path


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def same_items(a: dict, b: dict) -> bool:
    return a.get("digest") == b.get("digest")


def items_of(manifest: dict) -> list[Item]:
    keep = {"id", "path", "singer", "set", "notes", "lyrics", "technique", "pair"}
    out = []
    for row in manifest["items"]:
        r = {k: v for k, v in row.items() if k in keep}
        r["path"] = Path(r["path"])
        r["pair"] = Path(r["pair"]) if r.get("pair") else None
        r["notes"] = [tuple(n) for n in r.get("notes", [])]
        out.append(Item(**r))
    return out


__all__ = ["VERSION", "build", "checksum", "items_of", "load", "same_items", "save"]
