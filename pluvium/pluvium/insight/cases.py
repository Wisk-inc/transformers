# Copyright 2026 Nathan. Apache-2.0.
"""
Case replays: one day, forecast against truth, hour by hour.

For a chosen issue time and a few named places, the model is run on the staged data and the hourly chance of rain,
the words for each hour, and what the satellite and the nearest airport recorded are laid out side by side. This is
how a forecaster checks a model on a day they remember -- the flood, the front that came early -- rather than
through averages.
"""

from __future__ import annotations

import numpy as np
import torch

from ..dataset import collate
from ..products import summarize
from ..weather import rate_class


def replay(model, builder, t0: int, points: np.ndarray, names: list[str] | None = None, hours: int = 24) -> str:
    """Text table for target cells ``points`` (indices into the geography's targets) at issue time ``t0``."""
    model.eval()
    ex = builder.example(int(t0), False, np.asarray(points), emulate=False)
    with torch.no_grad():
        out = model(collate([ex]))
    s = summarize(out, int(t0), samples=50)
    truth = ex["truth"].T                                                     # (P, T)
    lines = []
    for k in range(len(points)):
        name = names[k] if names else f"cell {int(points[k])}"
        lines.append(f"{name}: chance of rain / satellite class (0 dry .. 3 heavy)")
        row = []
        for h in range(min(hours, truth.shape[1])):
            obs = int(rate_class(np.array([truth[k, h]]))[0])
            row.append(f"{h:02d}h {s['p_wet'][k, h]:4.0%}/{'?' if obs < 0 else obs}")
        for a in range(0, len(row), 6):
            lines.append("   " + "  ".join(row[a: a + 6]))
    return "\n".join(lines)


__all__ = ["replay"]
