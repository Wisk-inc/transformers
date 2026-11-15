# Copyright 2026 Nathan. Apache-2.0.
"""
Scenarios: the handful of different futures in a cloud of tracks.

A mean track through a cloud that splits in two (recurve or not, stall or accelerate) runs between the two, where no
member goes. Grouping the members by how far apart their whole tracks are gives the scenarios, each with its share of
the weight and the mean track of its members.

The grouping is k-medoids on the mean distance between tracks over their common leads. The number of scenarios is
chosen by the silhouette, and a split is reported only when it is clear; otherwise there is one scenario.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cloud import Cloud
from .geodesy import distance_km, mean_position


@dataclass
class Scenario:
    weight: float                       # share of the cloud's weight
    members: np.ndarray                 # indices into the cloud
    medoid: int                         # the member nearest the middle of it
    points: dict                        # its mean track {lead: (lat, lon)}, where at least half its weight has a fix


def track_distances(cloud: Cloud, min_common: int = 3) -> np.ndarray:
    """``(M, M)`` mean distance (km) between members' tracks over the leads both have; inf with too few in common."""
    m = cloud.members
    d = np.full((m, m), np.inf)
    have = np.isfinite(cloud.lat)
    for i in range(m):
        both = have[i][None] & have                                             # (M, L)
        n = both.sum(1)
        dist = distance_km(cloud.lat[i][None], cloud.lon[i][None], cloud.lat, cloud.lon)
        mean = np.where(both, dist, 0.0).sum(1) / np.maximum(n, 1)
        d[i] = np.where(n >= min_common, mean, np.inf)
    np.fill_diagonal(d, 0.0)
    return d


def k_medoids(d: np.ndarray, k: int, weight: np.ndarray, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Weighted k-medoids: ``(medoids (k,), label (M,))``. Starts from the weighted medoid, then the points furthest
    from the medoids chosen so far."""
    big = np.where(np.isfinite(d), d, 1e9)
    medoids = [int(np.argmin((big * weight[None]).sum(1)))]
    while len(medoids) < k:
        medoids.append(int(np.argmax(big[:, medoids].min(1) * (weight > 0))))
    medoids = np.array(medoids)
    for _ in range(iters):
        label = np.argmin(big[:, medoids], 1)
        new = medoids.copy()
        for c in range(k):
            rows = np.flatnonzero(label == c)
            if rows.size:
                new[c] = rows[np.argmin((big[np.ix_(rows, rows)] * weight[rows][None]).sum(1))]
        if np.array_equal(new, medoids):
            break
        medoids = new
    return medoids, np.argmin(big[:, medoids], 1)


def silhouette(d: np.ndarray, label: np.ndarray, weight: np.ndarray) -> float:
    """Weighted mean silhouette (-1 to 1; near 0 means the groups are not really separate)."""
    big = np.where(np.isfinite(d), d, 1e9)
    groups = np.unique(label)
    if groups.size < 2:
        return 0.0
    s = np.zeros(label.size)
    for i in range(label.size):
        own = (label == label[i]) & (np.arange(label.size) != i)
        a = (big[i, own] * weight[own]).sum() / max(weight[own].sum(), 1e-12) if own.any() else 0.0
        b = min((big[i, label == g] * weight[label == g]).sum() / max(weight[label == g].sum(), 1e-12)
                for g in groups if g != label[i])
        s[i] = (b - a) / max(a, b, 1e-12) if own.any() else 0.0
    return float((s * weight).sum() / max(weight.sum(), 1e-12))


def scenarios(cloud: Cloud, max_k: int = 3, min_silhouette: float = 0.45, min_share: float = 0.1) -> list:
    """The cloud's scenarios, heaviest first. One scenario unless a split is clear (silhouette at least
    ``min_silhouette``) and every group holds at least ``min_share`` of the weight."""
    w = cloud.member_weight()
    keep = np.flatnonzero((w > 0) & np.isfinite(cloud.lat).any(1))
    if keep.size == 0:
        return []
    d = track_distances(cloud)[np.ix_(keep, keep)]
    wk = w[keep] / w[keep].sum()
    best = (np.array([int(np.argmin((np.where(np.isfinite(d), d, 1e9) * wk[None]).sum(1)))]), np.zeros(keep.size, int))
    best_score = min_silhouette
    for k in range(2, min(max_k, keep.size) + 1):
        medoids, label = k_medoids(d, k, wk)
        shares = np.array([wk[label == c].sum() for c in range(k)])
        score = silhouette(d, label, wk)
        if shares.min() >= min_share and score > best_score:
            best, best_score = (medoids, label), score
    medoids, label = best
    out = []
    for c, m in enumerate(medoids):
        rows = keep[label == c]
        pts = {}
        for j, lead in enumerate(cloud.leads):
            ok = np.isfinite(cloud.lat[rows, j])
            if w[rows][ok].sum() >= 0.5 * w[rows].sum() > 0:
                mlat, mlon = mean_position(cloud.lat[rows[ok], j], cloud.lon[rows[ok], j], w[rows][ok] / w[rows][ok].sum())
                pts[int(lead)] = (float(mlat), float(mlon))
        out.append(Scenario(float(wk[label == c].sum()), rows, int(keep[m]), pts))
    return sorted(out, key=lambda s: -s.weight)


__all__ = ["Scenario", "k_medoids", "scenarios", "silhouette", "track_distances"]
