# Copyright 2026 Nathan. Apache-2.0.
"""
Rain over an area: a parish, a river basin, a country.

Floods answer to the rain over a catchment, not at a point. The chance that a basin's mean rain passes 50 mm cannot
be read off the chances at its places: it depends on whether they are wet together. Scenarios carry that dependence
(:mod:`.ecc`), so areal questions become counting:

* the distribution of the area's mean (or total volume) is the scenarios' areal means;
* the chance of passing a threshold is the share of scenarios that do;
* the same from places treated as independent is far too confident, which :func:`overconfidence` measures.

Places are weighted by the area they stand for: the cosine of latitude on a regular latitude-longitude grid, or
explicit weights (such as the cell areas of Pluvium's own mesh).
"""

from __future__ import annotations

import numpy as np


def area_weights(lat: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Normalised weights of places: ``weights`` when given, else the cosine of latitude."""
    w = np.cos(np.radians(lat)) if weights is None else np.asarray(weights, np.float64)
    return w / w.sum()


def areal_mean(scenarios: np.ndarray, lat: np.ndarray, mask: np.ndarray | None = None,
               weights: np.ndarray | None = None) -> np.ndarray:
    """Each scenario's mean rain over the area ``mask`` (all places by default): ``(M,)`` mm."""
    mask = np.ones(lat.size, bool) if mask is None else np.asarray(mask, bool)
    w = area_weights(lat[mask], None if weights is None else np.asarray(weights)[mask])
    return np.nan_to_num(scenarios[:, mask]) @ w


def areal_chance(scenarios: np.ndarray, lat: np.ndarray, thresholds, mask: np.ndarray | None = None,
                 weights: np.ndarray | None = None) -> dict:
    """``{threshold: chance}`` that the area's mean rain reaches each threshold."""
    means = areal_mean(scenarios, lat, mask, weights)
    return {float(t): float((means >= t).mean()) for t in thresholds}


def areal_summary(scenarios: np.ndarray, lat: np.ndarray, mask: np.ndarray | None = None,
                  weights: np.ndarray | None = None) -> dict:
    """Median, 10th and 90th percentile and mean of the area's mean rain, and the share of the area with at least
    25 mm in the median scenario."""
    means = areal_mean(scenarios, lat, mask, weights)
    mask = np.ones(lat.size, bool) if mask is None else np.asarray(mask, bool)
    w = area_weights(lat[mask], None if weights is None else np.asarray(weights)[mask])
    wet_share = (np.nan_to_num(scenarios[:, mask]) >= 25.0) @ w
    return {"mean_mm": float(means.mean()), "p10_mm": float(np.quantile(means, 0.1)),
            "median_mm": float(np.median(means)), "p90_mm": float(np.quantile(means, 0.9)),
            "share_25mm_median": float(np.median(wet_share))}


def overconfidence(coupled: np.ndarray, independent: np.ndarray, lat: np.ndarray, mask: np.ndarray | None = None) -> float:
    """How much too narrow independent places make the areal mean: the ratio of the coupled scenarios' spread
    (standard deviation of the areal mean) to the independent ones'. Well above one wherever rain is organised."""
    a = areal_mean(coupled, lat, mask).std()
    b = areal_mean(independent, lat, mask).std()
    return float(a / max(b, 1e-9))


__all__ = ["area_weights", "areal_chance", "areal_mean", "areal_summary", "overconfidence"]
