# Copyright 2026 Nathan. Apache-2.0.
"""
Where the model fails: verification split by situation.

An overall skill score averages good and bad situations together. Splitting the same hourly Brier score (and its skill
against climatology) by the rain regime the model diagnosed, the lead time, the local hour, the place's elevation and
distance from the coast, and the season shows the weak spots -- for example "afternoon storms in the interior beyond
24 hours", where the next round of work should go.
"""

from __future__ import annotations

import numpy as np


def _skill(p, y, c):
    b, bc = np.mean((p - y) ** 2), np.mean((c - y) ** 2)
    return {"n": int(y.size), "brier": float(b), "brier_clim": float(bc), "skill": float(1 - b / max(bc, 1e-12)),
            "wet_share": float(y.mean())}


def stratified_scores(p: np.ndarray, wet: np.ndarray, clim: np.ndarray, known: np.ndarray, strata: dict) -> dict:
    """``p``, ``wet``, ``clim``, ``known`` are ``(N, T)``; ``strata`` maps a name to labels ``(N, T)`` (or ``(N,)``,
    broadcast over hours). Returns ``{stratum: {label: scores}}``, labels with fewer than 500 known hours left out."""
    out = {}
    for name, labels in strata.items():
        lab = np.broadcast_to(np.asarray(labels)[..., None] if np.ndim(labels) == 1 else labels, p.shape)
        table = {}
        for value in np.unique(lab[known]):
            m = known & (lab == value)
            if m.sum() >= 500:
                table[str(value)] = _skill(p[m], wet[m], clim[m])
        out[name] = table
    return out


def standard_strata(lat: np.ndarray, lon: np.ndarray, elevation_km: np.ndarray, coast_km: np.ndarray, t0: np.ndarray,
                    hours: int) -> dict:
    """Lead band, local-hour band, elevation band and coast band labels for ``N = len(t0) * P`` rows."""
    p = lat.size
    lead = np.broadcast_to(np.arange(hours)[None], (t0.size * p, hours))
    lead_band = np.digitize(lead, [6, 12, 24, 48])
    lst = (np.repeat(t0, p)[:, None] + np.arange(hours)[None] + 0.5 + np.tile(lon, t0.size)[:, None] / 15.0) % 24
    return {"lead (0-6/6-12/12-24/24-48/48-72 h)": lead_band,
            "local time (night/morning/afternoon/evening)": (lst // 6).astype(int),
            "elevation (<200 m / 200-600 m / >600 m)": np.tile(np.digitize(elevation_km, [0.2, 0.6]), t0.size),
            "coast (sea / <5 km / 5-20 km / >20 km inland)": np.tile(np.digitize(coast_km, [0.0, 5.0, 20.0]), t0.size)}


__all__ = ["standard_strata", "stratified_scores"]
