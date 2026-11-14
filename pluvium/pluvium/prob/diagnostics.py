# Copyright 2026 Nathan. Apache-2.0.
"""
Is the uncertainty right?

* **PIT histogram** (probability integral transform): where each outcome fell within its forecast distribution. Flat
  means the spread is right; U-shaped means too narrow (overconfident); dome-shaped means too wide;
* **reliability with consistency bars** (Brocker and Smith 2007): the observed frequency in each probability bin, with
  the range a perfectly reliable forecast would show by chance given the bin's size, so a deviation is only called a
  problem when it is larger than sampling noise;
* **spread-skill**: the ensemble's spread against its error; a ratio near 1 means the stated uncertainty matches the
  actual error.
"""

from __future__ import annotations

import numpy as np


def pit_histogram(samples: np.ndarray, observed: np.ndarray, bins: int = 10, rng=None) -> dict:
    """``samples (S, N)``, ``observed (N,)``; randomised PIT for ties (many zeros in rain)."""
    rng = rng or np.random.default_rng(0)
    s = np.asarray(samples, np.float64)
    y = np.asarray(observed, np.float64)
    below = (s < y[None]).mean(0)
    equal = (s == y[None]).mean(0)
    pit = below + rng.random(y.size) * equal
    hist, _ = np.histogram(pit, bins=bins, range=(0, 1))
    freq = hist / max(hist.sum(), 1)
    return {"pit": pit, "frequencies": freq, "flatness": float(np.abs(freq - 1 / bins).sum())}


def reliability_bands(p: np.ndarray, y: np.ndarray, bins: int = 10, draws: int = 500, rng=None) -> list[dict]:
    rng = rng or np.random.default_rng(0)
    p, y = np.asarray(p, np.float64).ravel(), np.asarray(y, np.float64).ravel()
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for k in range(bins):
        m = which == k
        if not m.any():
            continue
        pk = p[m]
        sim = (rng.random((draws, pk.size)) < pk[None]).mean(1)
        out.append({"forecast": float(pk.mean()), "observed": float(y[m].mean()), "count": int(m.sum()),
                    "low": float(np.quantile(sim, 0.05)), "high": float(np.quantile(sim, 0.95))})
    return out


def spread_skill(samples: np.ndarray, observed: np.ndarray) -> float:
    s = np.asarray(samples, np.float64)
    y = np.asarray(observed, np.float64)
    spread = np.sqrt(np.mean(s.var(0)))
    error = np.sqrt(np.mean((s.mean(0) - y) ** 2))
    return float(spread / max(error, 1e-12))


__all__ = ["pit_histogram", "reliability_bands", "spread_skill"]
