# Copyright 2026 Nathan. Apache-2.0.
"""
Ensemble copula coupling: calibrated chances with a real ensemble's spatial structure.

Pluvium Blend gives every place its own calibrated distribution. The mean of those distributions is a smooth map,
smoother the further ahead it looks, because it averages over every place a storm might be. That is correct, and it
is not what a storm looks like. What is needed for rain over a river basin, or for a map people believe, is a set of
scenarios: each one sharp and spatially coherent, and together calibrated at every place.

Ensemble copula coupling (Schefzik, Thorarinsdottir and Gneiting 2013) builds them:

1. take a physical ensemble (ECMWF ENS): each member is a coherent field;
2. at every place, rank the members;
3. give the member ranked ``r`` of ``M`` the calibrated distribution's quantile at level ``(r + 0.5) / M``.

Every scenario keeps its member's pattern (where it is wet relative to the others), so storms stay storms; at every
place the scenarios are exactly a sample of the calibrated distribution.

Rain needs one refinement. Many members are dry at a place, so their ranks are ties. Broken at random, the ties
scatter drizzle incoherently wherever calibration says rain is more likely than the ensemble did. Here ties are broken
by each member's neighbourhood rain (a member with rain nearby is the next to be wet), then at random.
"""

from __future__ import annotations

import numpy as np

from ..blend import quantile_at


def ranks(members: np.ndarray, tiebreak: np.ndarray | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    """The rank (0 = driest) of every member at every place, ``(M, N)``. Ties are broken by ``tiebreak`` (same shape,
    larger ranks higher), then at random. Members that are NaN at a place rank lowest."""
    rng = rng or np.random.default_rng(0)
    m = np.where(np.isfinite(members), members, -1.0).astype(np.float64)
    key2 = np.zeros_like(m) if tiebreak is None else np.nan_to_num(np.asarray(tiebreak, np.float64))
    key3 = rng.random(m.shape)
    order = np.lexsort((key3, key2, m), axis=0)                    # last key is the primary one
    out = np.empty_like(order)
    np.put_along_axis(out, order, np.arange(m.shape[0])[:, None] * np.ones((1, m.shape[1]), np.int64), axis=0)
    return out


def ecc(p: np.ndarray, members: np.ndarray, tiebreak: np.ndarray | None = None, rng: np.random.Generator | None = None,
        jitter: bool = False) -> np.ndarray:
    """Calibrated scenarios ``(M, N)`` mm from calibrated chances ``p`` (N, K) and a template ensemble ``members``
    (M, N). With ``jitter`` the level within each rank's slot is random (ECC-R); without, its midpoint (ECC-Q)."""
    rng = rng or np.random.default_rng(0)
    r = ranks(members, tiebreak, rng)
    m = members.shape[0]
    u = (r + (rng.random(r.shape) if jitter else 0.5)) / m
    return quantile_at(p, u.T).T


def independent(p: np.ndarray, members: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Scenarios with the same calibrated marginals but no spatial structure (each place's quantiles shuffled on their
    own): the reference that shows what coupling adds."""
    rng = rng or np.random.default_rng(0)
    n = p.shape[0]
    r = np.argsort(rng.random((members, n)), axis=0)
    return quantile_at(p, ((r + 0.5) / members).T).T


def neighbourhood_tiebreak(day: np.ndarray, lat: np.ndarray, lon: np.ndarray, cell_lat: np.ndarray,
                           cell_lon: np.ndarray, radius_deg: float = 1.0) -> np.ndarray:
    """Each member's mean rain within ``radius_deg`` of every place, ``(M, N)``, from its field ``day`` (M, H, W)."""
    from scipy.ndimage import uniform_filter

    from ..dataset import bilinear

    k = 2 * int(round(radius_deg / abs(lon[1] - lon[0]))) + 1
    smooth = uniform_filter(np.nan_to_num(day), size=(1, k, k), mode="nearest")
    return bilinear(smooth, lat, lon, cell_lat, cell_lon)


__all__ = ["ecc", "independent", "neighbourhood_tiebreak", "ranks"]
