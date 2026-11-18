# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
VocalSet items, split by singer. The files are named ``<singer>_<context>_<technique>_<vowel>.wav`` (``f1`` to
``f9`` and ``m1`` to ``m11``). The test singers of this suite are fixed below, chosen so that both voice types
and the whole range are in the test half; a model meant to be benchmarked must not train on them.
"""

from __future__ import annotations

import re
from pathlib import Path

from .registry import Item

TEST_SINGERS = ("f2", "f8", "m3", "m5", "m10")
NAME = re.compile(r"^(?P<singer>[fm]\d+)_(?P<context>[a-z0-9]+)_(?P<rest>.+)$", re.IGNORECASE)


def parse(path: Path) -> dict | None:
    m = NAME.match(Path(path).stem)
    if not m:
        return None
    rest = m.group("rest").split("_")
    return {"singer": m.group("singer").lower(), "context": m.group("context").lower(),
            "technique": rest[0].lower() if rest else ""}


def items(root: str | Path, split: str = "test") -> list[Item]:
    """The takes of ``split`` (``test``: the singers above; ``train``: the others; ``all``)."""
    out = []
    for path in sorted(Path(root).rglob("*.wav")):
        info = parse(path)
        if info is None:
            continue
        held = info["singer"] in TEST_SINGERS
        if split == "all" or (split == "test") == held:
            out.append(Item(path.stem, path, info["singer"], "vocalset", technique=info["technique"]))
    return out


def is_test_singer(name: str) -> bool:
    return str(name).lower().split("/")[-1] in TEST_SINGERS


__all__ = ["TEST_SINGERS", "is_test_singer", "items", "parse"]
