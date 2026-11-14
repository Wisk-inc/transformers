# Copyright 2026 Nathan. Apache-2.0.
"""
How much each system counts in the consensus: weights from the covariance of their errors.

An equal-weight mean is hard to beat when the systems are equally good and independent. They are neither. ECMWF's
ensemble mean and its high-resolution run share most of their error; an AI model trained on reanalysis errs
differently from a physical model. The combination with the smallest expected squared error has weights proportional
to the inverse error covariance applied to a vector of ones (``w = C^-1 1 / 1' C^-1 1``): accurate systems count for
more, and systems whose errors repeat each other share one vote.

Three safeguards keep that from overfitting a season of storms:

* errors are pooled over leads after scaling by each lead's typical error, so every lead group has enough cases;
* the covariance is shrunk towards its diagonal;
* weights are found with a non-negativity constraint (no system is subtracted) and then pulled part of the way back
  to equal weights.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from .geodesy import offset_km

GROUPS = {"short": (12, 24, 36, 48), "long": (60, 72, 96, 120)}


def error_vectors(points: dict, observed: dict) -> dict:
    """``{lead: (east km, north km)}`` of forecast minus observed, where both exist."""
    out = {}
    for lead, obs in observed.items():
        if lead in points and np.isfinite(points[lead][0]):
            e, n = offset_km(obs[0], obs[1], points[lead][0], points[lead][1])
            out[int(lead)] = (float(e), float(n))
    return out


class ConsensusWeights:
    """Weights per lead group from training errors. ``add`` every training forecast, then ``fit``."""

    def __init__(self, systems: list, groups: dict | None = None, shrink: float = 0.3, to_equal: float = 0.25) -> None:
        self.systems = list(systems)
        self.groups = dict(groups or GROUPS)
        self.shrink, self.to_equal = shrink, to_equal
        self.cases = {}                              # (case key, lead) -> {system: (east, north)}
        self.weights = {}                            # group -> {system: weight}
        self.rms = {}                                # system -> {group: rms error km} (for reports)

    def add(self, key, system: str, vectors: dict) -> None:
        for lead, v in vectors.items():
            self.cases.setdefault((key, int(lead)), {})[system] = v

    def _covariance(self, leads) -> tuple[np.ndarray, np.ndarray]:
        """Covariance of the systems' error vectors over ``leads`` (each lead scaled by its typical error), and the
        number of cases behind every pair. Pairs are estimated on the cases both systems forecast."""
        k = len(self.systems)
        prod, count = np.zeros((k, k)), np.zeros((k, k))
        for lead in leads:
            rows = [v for (key, ld), v in self.cases.items() if ld == lead]
            if not rows:
                continue
            scale = np.sqrt(np.mean([e * e + n * n for row in rows for e, n in row.values()])) or 1.0
            for row in rows:
                for a, sa in enumerate(self.systems):
                    if sa not in row:
                        continue
                    for b, sb in enumerate(self.systems):
                        if sb in row:
                            prod[a, b] += (row[sa][0] * row[sb][0] + row[sa][1] * row[sb][1]) / scale ** 2
                            count[a, b] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = np.where(count > 0, prod / np.maximum(count, 1), 0.0)
        return cov, count

    def fit(self, min_cases: int = 8) -> ConsensusWeights:
        k = len(self.systems)
        for group, leads in self.groups.items():
            cov, count = self._covariance(leads)
            seen = np.diag(count) >= min_cases
            diag = np.where(seen, np.diag(cov), np.nan)
            typical = np.nanmean(diag) if seen.any() else 1.0
            cov = np.where(seen[:, None] & seen[None, :], cov, 0.0)
            np.fill_diagonal(cov, np.where(seen, np.diag(cov), typical * 4.0))      # unseen systems: little weight
            shrunk = (1 - self.shrink) * cov + self.shrink * np.diag(np.diag(cov))
            # minimise w' C w with w >= 0 and sum(w) = 1: non-negative least squares on a Cholesky factor
            try:
                chol = np.linalg.cholesky(shrunk + 1e-9 * typical * np.eye(k))
            except np.linalg.LinAlgError:
                chol = np.diag(np.sqrt(np.maximum(np.diag(shrunk), 1e-9)))
            big = 1e3 * np.sqrt(typical)
            a = np.vstack([chol.T, big * np.ones((1, k))])
            b = np.concatenate([np.zeros(k), [big]])
            w, _ = nnls(a, b)
            w = w / max(w.sum(), 1e-12)
            w = (1 - self.to_equal) * w + self.to_equal * np.where(seen, 1.0 / max(seen.sum(), 1), 0.0)
            self.weights[group] = {s: float(x) for s, x in zip(self.systems, w / w.sum())}
            for i, s in enumerate(self.systems):
                self.rms.setdefault(s, {})[group] = float(np.sqrt(max(diag[i], 0.0))) if seen[i] else float("nan")
        return self

    def at(self, lead: int) -> dict:
        """``{system: weight}`` for a lead (the group that holds it; equal weights before ``fit``)."""
        for group, leads in self.groups.items():
            if int(lead) in leads or int(lead) <= max(leads):
                if group in self.weights:
                    return self.weights[group]
                break
        if self.weights:
            return self.weights[list(self.weights)[-1]]
        return {s: 1.0 / len(self.systems) for s in self.systems}


__all__ = ["GROUPS", "ConsensusWeights", "error_vectors"]
