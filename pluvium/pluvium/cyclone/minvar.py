# Copyright 2026 Nathan. Apache-2.0.
"""
Minimum-variance weights that keep learning: a running error covariance.

:mod:`.weights` fits weights once, from a finished season. Here the same estimate is kept up to date after every
verified forecast, so it can be used causally through a season:

* each verified forecast adds the systems' scaled error vectors to running sums of products (one sum per pair of
  systems, counted on the forecasts both made);
* older forecasts fade with a half-life, because models are upgraded;
* a system with little or no record is given the typical error variance until its own record outweighs that prior
  (``pseudo`` forecasts' worth), so a new system starts as an average member and is neither trusted nor written off;
* covariances between systems are shrunk towards zero the same way, and by a fixed share on top;
* weights minimise the consensus error variance with no system subtracted, and are pulled part of the way back to
  equal weights.

With no forecasts verified the weights are equal. Unlike a learning-rate rule, this has no step size to tune: the
weights are a function of the evidence so far.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def solve_weights(cov: np.ndarray, to_equal: float = 0.25) -> np.ndarray:
    """Non-negative weights summing to one that minimise ``w' C w``, pulled ``to_equal`` of the way to equal."""
    k = cov.shape[0]
    typical = float(np.mean(np.diag(cov))) or 1.0
    chol = None
    for ridge in (1e-9, 1e-3, 1e-2, 1e-1):                        # pairwise estimates need not be positive definite
        try:
            chol = np.linalg.cholesky(cov + ridge * typical * np.eye(k))
            break
        except np.linalg.LinAlgError:
            continue
    if chol is None:
        chol = np.diag(np.sqrt(np.maximum(np.diag(cov), 1e-9)))
    big = 1e3 * np.sqrt(typical)
    w, _ = nnls(np.vstack([chol.T, big * np.ones((1, k))]), np.concatenate([np.zeros(k), [big]]))
    w = w / max(w.sum(), 1e-12)
    w = (1 - to_equal) * w + to_equal / k
    return w / w.sum()


class RunningCovariance:
    """The error covariance of ``systems`` so far, and the weights it implies."""

    def __init__(self, systems: list, shrink: float = 0.3, to_equal: float = 0.25, pseudo: float = 8.0,
                 halflife_days: float | None = 365.0) -> None:
        self.systems = list(systems)
        self.shrink, self.to_equal, self.pseudo, self.halflife = shrink, to_equal, pseudo, halflife_days
        k = len(self.systems)
        self.prod, self.count = np.zeros((k, k)), np.zeros((k, k))
        self.clock = None                                         # hour of the last forecast added
        self._weights = None

    def add(self, hour: int, vectors: dict) -> None:
        """One verified forecast: ``{system: scaled (east, north) error}`` valid at ``hour``."""
        if self.halflife and self.clock is not None and hour > self.clock:
            fade = 0.5 ** ((hour - self.clock) / 24.0 / self.halflife)
            self.prod *= fade
            self.count *= fade
        self.clock = hour if self.clock is None else max(self.clock, hour)
        idx = [self.systems.index(s) for s in vectors if s in self.systems]
        e = np.array([vectors[self.systems[i]] for i in idx], np.float64).reshape(len(idx), 2)
        self.prod[np.ix_(idx, idx)] += e @ e.T
        self.count[np.ix_(idx, idx)] += 1.0
        self._weights = None

    def covariance(self) -> np.ndarray:
        """The shrunk covariance: variances pulled to the typical variance, covariances pulled to zero."""
        n = np.diag(self.count)
        seen = n > 0
        raw = np.where(seen, np.diag(self.prod) / np.maximum(n, 1e-12), 0.0)
        typical = float(raw[seen].mean()) if seen.any() else 1.0
        cov = (1 - self.shrink) * self.prod / (self.count + self.pseudo)
        np.fill_diagonal(cov, (np.diag(self.prod) + self.pseudo * typical) / (n + self.pseudo))
        return cov

    def weights(self) -> np.ndarray:
        if self._weights is None:
            self._weights = solve_weights(self.covariance(), self.to_equal)
        return self._weights

    def rms(self) -> dict:
        """Each system's root-mean-square scaled error so far (NaN before its first forecast)."""
        n = np.diag(self.count)
        with np.errstate(invalid="ignore", divide="ignore"):
            return {s: float(np.sqrt(self.prod[i, i] / n[i])) if n[i] > 0 else float("nan")
                    for i, s in enumerate(self.systems)}


__all__ = ["RunningCovariance", "solve_weights"]
