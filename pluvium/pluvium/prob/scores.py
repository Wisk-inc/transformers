# Copyright 2026 Nathan. Apache-2.0.
"""
Scores that a single mean error hides.

* **Brier decomposition** (Murphy 1973): reliability (are 30% forecasts right 30% of the time?), resolution (do the
  forecasts separate wet from dry situations?) and uncertainty (how hard the climate is);
* **ROC area**: discrimination regardless of calibration;
* **equitable threat score**: yes/no skill corrected for hits by chance;
* **SEDI** (Ferro and Stephenson 2011): skill for rare events that, unlike threat scores, does not decay to zero
  as the event gets rarer -- the right score for heavy hours;
* **fractions skill score** (Roberts and Lean 2008) on scattered points: a forecast that puts a shower 10 km from
  where it fell is penalised twice by point scores (a miss and a false alarm); the FSS compares the fractions of
  rain within a neighbourhood, so it shows the scale at which the forecast becomes useful.
"""

from __future__ import annotations

import numpy as np


def brier_decomposition(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    p, y = np.asarray(p, np.float64).ravel(), np.asarray(y, np.float64).ravel()
    clim = y.mean()
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rel = res = 0.0
    for k in range(bins):
        m = which == k
        if m.any():
            pk, ok = p[m].mean(), y[m].mean()
            rel += m.sum() * (pk - ok) ** 2
            res += m.sum() * (ok - clim) ** 2
    n = p.size
    return {"brier": float(np.mean((p - y) ** 2)), "reliability": rel / n, "resolution": res / n,
            "uncertainty": float(clim * (1 - clim))}


def roc_auc(p: np.ndarray, y: np.ndarray) -> float:
    p, y = np.asarray(p).ravel(), np.asarray(y).ravel().astype(bool)
    pos, neg = y.sum(), (~y).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(p.size)
    ranks[order] = np.arange(1, p.size + 1)
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _table(yes: np.ndarray, obs: np.ndarray):
    yes, obs = np.asarray(yes).astype(bool), np.asarray(obs).astype(bool)
    return (yes & obs).sum(), (yes & ~obs).sum(), (~yes & obs).sum(), (~yes & ~obs).sum()


def ets(yes: np.ndarray, obs: np.ndarray) -> float:
    a, b, c, d = (float(x) for x in _table(yes, obs))
    n = a + b + c + d
    chance = (a + b) * (a + c) / max(n, 1)
    return (a - chance) / max(a + b + c - chance, 1e-12)


def sedi(yes: np.ndarray, obs: np.ndarray) -> float:
    a, b, c, d = (float(x) for x in _table(yes, obs))
    h = np.clip(a / max(a + c, 1), 1e-6, 1 - 1e-6)
    f = np.clip(b / max(b + d, 1), 1e-6, 1 - 1e-6)
    num = np.log(f) - np.log(h) - np.log(1 - f) + np.log(1 - h)
    den = np.log(f) + np.log(h) + np.log(1 - f) + np.log(1 - h)
    return float(num / den)


def fss_points(forecast: np.ndarray, observed: np.ndarray, lat: np.ndarray, lon: np.ndarray, radius_km: float) -> float:
    """Fractions skill score for binary (or probability) fields on scattered points, neighbourhood ``radius_km``.
    ``forecast``/``observed`` are ``(..., P)``."""
    from scipy.spatial import cKDTree

    la, lo = np.radians(np.asarray(lat)), np.radians(np.asarray(lon))
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    tree = cKDTree(xyz)
    groups = tree.query_ball_point(xyz, r=radius_km / 6371.0)
    f = np.asarray(forecast, np.float64).reshape(-1, len(lat))
    o = np.asarray(observed, np.float64).reshape(-1, len(lat))
    pf = np.stack([f[:, g].mean(1) for g in groups], 1)
    po = np.stack([o[:, g].mean(1) for g in groups], 1)
    mse = np.mean((pf - po) ** 2)
    ref = np.mean(pf**2) + np.mean(po**2)
    return float(1 - mse / max(ref, 1e-12))


__all__ = ["brier_decomposition", "ets", "fss_points", "roc_auc", "sedi"]
