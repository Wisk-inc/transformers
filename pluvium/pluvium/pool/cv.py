# Copyright 2026 Nathan. Apache-2.0.
"""
Choosing a pool design without touching the test year: blocked cross-validation inside the training year.

Every row already carries out-of-fold component forecasts (:mod:`pluvium.frontier.oof`). A pool design is scored by
leaving out one calendar month at a time: fit on the other eleven, forecast the month left out. Every season is tested
and the pool never sees the month it is judged on. The design with the best scores here is the one that is then run,
once, on the test year.

Scores are Brier skill at 1, 10, 25, 50 and 100 mm (against the base rate of the rows scored) and a CRPS computed from
the same chances, each for all rows, by lead group and by season.
"""

from __future__ import annotations

import numpy as np

from ..blend import THRESHOLDS, exceed

SCORED = (1.0, 10.0, 25.0, 50.0, 100.0)
LEADS = {"days 1-3": (1, 3), "days 4-7": (4, 7), "days 8-14": (8, 14)}
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


def month_of(valid_hour: np.ndarray) -> np.ndarray:
    """Calendar month index (year * 12 + month - 1) of the day forecast."""
    d = np.asarray(valid_hour, np.int64).astype("datetime64[h]").astype("datetime64[M]").astype(np.int64)
    return d                                                    # months since 1970-01


def calendar_month(valid_hour: np.ndarray) -> np.ndarray:
    return month_of(valid_hour) % 12 + 1


def crps_from_chances(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS per row from chances of exceeding :data:`THRESHOLDS`: the Brier score integrated over the amount, with the
    chance taken as constant between thresholds (0 to the first threshold uses the first chance)."""
    edges = np.concatenate([[0.0], THRESHOLDS.astype(np.float64)])
    width = np.diff(edges)                                      # (K,)
    truth = (y[:, None] >= THRESHOLDS[None]).astype(np.float64)
    return ((p - truth) ** 2 * width[None]).sum(1)


def brier_skill(p: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None, scored=SCORED) -> dict:
    """``{threshold: skill}`` against the (weighted) base rate of the rows given, plus ``crps``."""
    w = np.ones(y.shape[0]) if weights is None else np.asarray(weights, np.float64)
    w = w / w.sum()
    out = {}
    for t in scored:
        yy = (y >= t).astype(np.float64)
        br = float((w * yy).sum())
        bs = float((w * (exceed(p, t) - yy) ** 2).sum())
        out[f"{t:g}"] = 1.0 - bs / max(br * (1 - br), 1e-12) if 0 < br < 1 else float("nan")
    out["crps"] = float((w * crps_from_chances(p, y)).sum())
    return out


def summarise(p: np.ndarray, y: np.ndarray, lead: np.ndarray, valid_hour: np.ndarray,
              weights: np.ndarray | None = None) -> dict:
    """Scores for all rows, by lead group and by season."""
    ok = np.isfinite(y)
    cm = calendar_month(valid_hour)
    out = {"all": brier_skill(p[ok], y[ok], None if weights is None else weights[ok])}
    for name, (lo, hi) in LEADS.items():
        m = ok & (lead >= lo) & (lead <= hi)
        if m.sum() > 100:
            out[name] = brier_skill(p[m], y[m], None if weights is None else weights[m])
    for name, months in SEASONS.items():
        m = ok & np.isin(cm, months) & (lead <= 3)
        if m.sum() > 100:
            out[f"{name} days 1-3"] = brier_skill(p[m], y[m], None if weights is None else weights[m])
    return out


def leave_month_out(fit, predict, months: np.ndarray, n_rows: int) -> np.ndarray:
    """Out-of-month chances for every row: ``fit(train mask) -> model`` and ``predict(model, test mask) -> (n, K)``
    are called once per calendar month present in ``months``."""
    out = np.zeros((n_rows, len(THRESHOLDS)), np.float32)
    for m in np.unique(months):
        test = months == m
        model = fit(~test)
        out[test] = predict(model, test)
    return out


__all__ = ["LEADS", "SCORED", "SEASONS", "brier_skill", "calendar_month", "crps_from_chances", "leave_month_out",
           "month_of", "summarise"]
