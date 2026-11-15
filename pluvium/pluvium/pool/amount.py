# Copyright 2026 Nathan. Apache-2.0.
"""
The expected amount as its own forecast: a stacked regression head.

The pool is fitted to chances. The amount it implies (the mean of its quantiles) is a by-product: nothing in its
training asks that mean to have the smallest squared error. Scores that are computed from one number per place, the
root-mean-square error and the anomaly correlation, deserve a forecast fitted for them.

:class:`AmountHead` is a non-negative linear combination of amounts already at hand: the pool's implied amount, each
component's implied amount, and the climatological mean. It is fitted by weighted non-negative least squares, one set
of coefficients per lead group, on the same out-of-fold rows as the pool. A component that is absent on a row is
replaced by the pool's amount, so its coefficient falls to the pool there.

Non-negative coefficients keep the head a blend, never a difference of forecasts, and make it as robust as its
ingredients: with a coefficient of one on the pool and zero elsewhere it is the pool's own amount.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from ..blend import quantiles

LEAD_GROUPS = ((1, 3), (4, 7), (8, 10), (11, 14))


def implied_amount(p: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """The amount a set of chances implies: the mean of its quantiles (``p`` is ``(N, K)`` or ``(N, C, K)``).
    Worked through in chunks of rows, because the quantiles of a million rows do not fit in memory at once."""
    flat = p.reshape(-1, p.shape[-1])
    out = np.empty(flat.shape[0], np.float32)
    for a in range(0, flat.shape[0], chunk):
        out[a:a + chunk] = quantiles(flat[a:a + chunk]).mean(1)
    return out.reshape(p.shape[:-1])


class AmountHead:
    def __init__(self, n_comp: int, lead_groups: tuple = LEAD_GROUPS, shrink: float = 0.02, n_extra: int = 0) -> None:
        self.n_comp, self.lead_groups, self.shrink, self.n_extra = n_comp, tuple(lead_groups), shrink, n_extra
        self.coef = np.zeros((len(self.lead_groups), n_comp + 2 + n_extra))
        self.coef[:, 0] = 1.0                                     # the pool's own amount until fitted

    def group_of(self, lead: np.ndarray) -> np.ndarray:
        lead = np.asarray(lead)
        out = np.full(lead.shape, len(self.lead_groups) - 1, np.int64)
        for g, (lo, hi) in enumerate(self.lead_groups):
            out[(lead >= lo) & (lead <= hi)] = g
        return out

    @staticmethod
    def design(pool: np.ndarray, comps: np.ndarray, present: np.ndarray, clim: np.ndarray,
               extra: np.ndarray | None = None) -> np.ndarray:
        """Columns: the pool's amount, each component's (the pool's where absent), the climatological mean, and any
        ``extra`` amounts ``(N, E)`` (the raw consensus mean, say; the pool's amount where not finite)."""
        cols = [pool[:, None], np.where(present, comps, pool[:, None]), clim[:, None]]
        if extra is not None:
            extra = np.asarray(extra, np.float64).reshape(pool.shape[0], -1)
            cols.append(np.where(np.isfinite(extra), extra, pool[:, None]))
        return np.concatenate(cols, 1).astype(np.float64)

    def fit(self, pool: np.ndarray, comps: np.ndarray, present: np.ndarray, clim: np.ndarray, lead: np.ndarray,
            y: np.ndarray, weights: np.ndarray | None = None, min_rows: int = 500,
            extra: np.ndarray | None = None) -> AmountHead:
        """Fit on rows with truth. ``pool`` (N,), ``comps`` (N, C), ``present`` (N, C), ``clim`` (N,) mm."""
        ok = np.isfinite(y)
        x = self.design(pool, comps, present, clim, extra)
        w = np.ones(y.shape[0]) if weights is None else np.asarray(weights, np.float64)
        group = self.group_of(lead)
        for g in range(len(self.lead_groups)):
            rows = ok & (group == g)
            if rows.sum() < min_rows:
                continue
            sw = np.sqrt(w[rows])
            a, b = x[rows] * sw[:, None], y[rows] * sw
            # a small pull towards "the pool's amount as it is": rows that say coefficient 1 on the pool, 0 elsewhere,
            # worth ``shrink`` of the pool column's own weight (one heavy-tailed column must not set the scale)
            scale = np.sqrt(self.shrink * (a[:, 0] ** 2).sum())
            prior = np.zeros(x.shape[1])
            prior[0] = 1.0
            a = np.concatenate([a, scale * np.eye(x.shape[1])])
            b = np.concatenate([b, scale * prior])
            self.coef[g], _ = nnls(a, b)
        return self

    def predict(self, pool: np.ndarray, comps: np.ndarray, present: np.ndarray, clim: np.ndarray,
                lead: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
        x = self.design(pool, comps, present, clim, extra)
        return (x * self.coef[self.group_of(np.broadcast_to(lead, pool.shape))]).sum(1).astype(np.float32)

    def table(self, names: list, extra_names: tuple = ()) -> dict:
        """``{lead group: {name: coefficient}}`` for reports."""
        cols = ["pool"] + list(names) + ["climatology"] + list(extra_names)
        return {f"days {lo}-{hi}": {c: float(v) for c, v in zip(cols, self.coef[g])}
                for g, (lo, hi) in enumerate(self.lead_groups)}


__all__ = ["LEAD_GROUPS", "AmountHead", "implied_amount"]
