# Copyright 2026 Nathan. Apache-2.0.
"""
Scores for scenario fields: do they have the right structure, not just the right chances at each place?

Scores computed place by place cannot tell a coherent ensemble from the same values shuffled in space. These can:

* the **energy score** (Gneiting and Raftery 2007) is the CRPS for vectors: the mean distance from the scenarios to
  what happened, minus half the mean distance between scenarios. Proper, but mostly sensitive to the marginals;
* the **variogram score** (Scheuerer and Hamill 2015) compares differences between pairs of places in the scenarios
  and in the observed field. It is the one that punishes wrong spatial dependence;
* the **CRPS of an areal total**: the most practical test. Rain over a basin is the sum over its places, and its
  spread depends entirely on how the places vary together. Independent scenarios make areal totals far too certain;
* the **sharpness of a scenario**: its spatial variance against the observed field's, as for the means in the
  benchmark, but for single scenarios, which should stay near one at every lead.
"""

from __future__ import annotations

import numpy as np


def crps_ensemble(members: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """CRPS of an ensemble ``(M, ...)`` against ``obs`` ``(...)``, per element (fair to the ensemble size)."""
    m = np.sort(np.asarray(members, np.float64), axis=0)
    n = m.shape[0]
    a = np.abs(m - obs[None]).mean(0)
    k = np.arange(n).reshape((n,) + (1,) * (m.ndim - 1))
    spread = (m * (2 * k - n + 1)).sum(0) / (n * (n - 1)) if n > 1 else 0.0     # mean |x - x'| / 2, fair
    return a - spread


def energy_score(scenarios: np.ndarray, obs: np.ndarray) -> float:
    """Energy score of scenarios ``(M, N)`` against the observed vector ``(N,)`` (places with NaN truth dropped)."""
    ok = np.isfinite(obs)
    s, o = np.nan_to_num(scenarios[:, ok].astype(np.float64)), obs[ok].astype(np.float64)
    m = s.shape[0]
    to_obs = np.sqrt(((s - o[None]) ** 2).sum(1)).mean()
    sq = (s * s).sum(1)
    between = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * s @ s.T, 0.0)).sum() / (m * (m - 1))
    return float(to_obs - 0.5 * between)


def variogram_score(scenarios: np.ndarray, obs: np.ndarray, pairs: np.ndarray, power: float = 0.5,
                    weights: np.ndarray | None = None) -> float:
    """Variogram score of order ``power`` over the place pairs ``(P, 2)``: the squared difference between the
    observed ``|y_i - y_j| ** power`` and its scenario mean, averaged with optional pair ``weights``."""
    i, j = pairs[:, 0], pairs[:, 1]
    ok = np.isfinite(obs[i]) & np.isfinite(obs[j])
    i, j = i[ok], j[ok]
    s = np.nan_to_num(scenarios.astype(np.float64))
    vo = np.abs(obs[i] - obs[j]) ** power
    vf = (np.abs(s[:, i] - s[:, j]) ** power).mean(0)
    w = np.ones(i.size) if weights is None else np.asarray(weights, np.float64)[ok]
    return float((w * (vo - vf) ** 2).sum() / max(w.sum(), 1e-12))


def nearby_pairs(lat: np.ndarray, lon: np.ndarray, max_km: float = 150.0, limit: int = 20000,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Pairs of places within ``max_km`` of each other (a random ``limit`` of them when there are more)."""
    from ..cyclone.geodesy import distance_km

    rng = rng or np.random.default_rng(0)
    out = []
    for a in range(lat.size - 1):
        d = distance_km(lat[a], lon[a], lat[a + 1:], lon[a + 1:])
        near = np.flatnonzero(d <= max_km) + a + 1
        if near.size:
            out.append(np.stack([np.full(near.size, a), near], 1))
    pairs = np.concatenate(out) if out else np.zeros((0, 2), np.int64)
    if pairs.shape[0] > limit:
        pairs = pairs[np.sort(rng.choice(pairs.shape[0], limit, replace=False))]
    return pairs


def areal_crps(scenarios: np.ndarray, obs: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """CRPS of each area's mean rain: ``groups`` (N,) gives every place's area (negative: none). Returns one value
    per area, NaN where an area has no observed place."""
    out = []
    for g in np.unique(groups[groups >= 0]):
        m = (groups == g) & np.isfinite(obs)
        if m.sum() == 0:
            out.append(np.nan)
            continue
        out.append(float(crps_ensemble(np.nan_to_num(scenarios[:, m]).mean(1)[:, None], np.array([obs[m].mean()]))[0]))
    return np.array(out)


def box_groups(lat: np.ndarray, lon: np.ndarray, size_deg: float = 2.0, min_places: int = 4) -> np.ndarray:
    """Every place's area index for areas that are boxes ``size_deg`` wide (boxes with too few places get -1)."""
    gi = np.floor((lat - lat.min()) / size_deg).astype(np.int64)
    gj = np.floor((lon - lon.min()) / size_deg).astype(np.int64)
    key = gi * 10_000 + gj
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    relabel = np.where(counts >= min_places, np.cumsum(counts >= min_places) - 1, -1)
    return relabel[inv]


def member_sharpness(scenarios: np.ndarray, obs: np.ndarray) -> tuple[float, float]:
    """``(sum over members of spatial variance / M, observed spatial variance)`` for one day: accumulate both over
    days and take the root of their ratio (the benchmark's sharpness, for single scenarios)."""
    ok = np.isfinite(obs)
    if ok.sum() < 10:
        return 0.0, 0.0
    return float(np.nanvar(scenarios[:, ok], axis=1).mean()), float(np.var(obs[ok]))


__all__ = ["areal_crps", "box_groups", "crps_ensemble", "energy_score", "member_sharpness", "nearby_pairs",
           "variogram_score"]
