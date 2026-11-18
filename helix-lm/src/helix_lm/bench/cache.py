# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Generated outputs kept by (system, task, item, settings), so a benchmark run that stopped, or one that adds a
system, only generates what is missing. Outputs are stored as float tensors, exactly as the system returned them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


class OutputCache:
    def __init__(self, root: str | Path | None) -> None:
        self.root = Path(root) if root else None
        self.hits = self.misses = 0

    def path(self, system: str, task: str, item: str, params: dict | None = None) -> Path | None:
        if self.root is None:
            return None
        key = hashlib.sha1(json.dumps(params or {}, sort_keys=True, default=str).encode()).hexdigest()[:10]
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in item)
        return self.root / system / task / f"{safe}-{key}.pt"

    def make(self, system: str, task: str, item: str, params: dict | None, produce) -> torch.Tensor:
        path = self.path(system, task, item, params)
        if path is not None and path.exists():
            try:
                self.hits += 1
                return torch.load(path)
            except Exception:                              # a torn file from a stopped run: made again
                self.hits -= 1
        self.misses += 1
        out = produce().detach().float().cpu().flatten()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            torch.save(out, tmp)
            tmp.replace(path)
        return out


__all__ = ["OutputCache"]
