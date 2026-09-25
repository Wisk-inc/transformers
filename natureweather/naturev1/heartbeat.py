# Copyright 2026 Nathan. Apache-2.0.
"""
A sign of life from a long run, for the supervisor that restarts a stalled one.

A process can be alive and doing nothing: a cloud read that never returns, a DataLoader worker that
deadlocked. Liveness is not progress. So every loop that makes progress -- a training step, a staged
chunk, a storm, a scored batch -- calls :func:`beat`, which touches a file the supervisor watches
(:func:`naturev1.launch`). No file configured, no cost: outside a supervised run this is a no-op.
"""

from __future__ import annotations

import os
import time


_last = [0.0]


def beat(note: str = "") -> None:
    """Record progress. Cheap enough to call every step: it writes at most once every five seconds."""
    path = os.environ.get("NATUREV1_HEARTBEAT")
    if not path:
        return
    now = time.time()
    if now - _last[0] < 5.0:
        return
    _last[0] = now
    try:
        with open(path, "w") as handle:
            handle.write(f"{now:.0f} {note}\n")
    except OSError:
        pass
