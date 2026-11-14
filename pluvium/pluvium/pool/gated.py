# Copyright 2026 Nathan. Apache-2.0.
"""
The pool, second generation: weights that follow the threshold, the lead, the season and the weather.

The first pool gave each component one weight per band of thresholds, changing with lead. That left three things on
the table, each visible in the full-year benchmark:

* **the tail**: three bands are too coarse where it matters. Here every threshold has its own weights, tied to its
  neighbours by a smoothness penalty, so 50 mm is not forced to share weights with 150 mm;
* **the season**: the best mix in a continental winter is not the best mix in summer. Weights and offsets may depend
  on gates, such as the annual cycle;
* **the weather**: when the components disagree, or when all of them call for heavy rain, a different mix is right.
  Gates can carry such regime features too.

The form stays a weighted logarithmic pool (average the present components' log-odds, sharpen by the weights' total,
add an offset), so a missing component still drops out cleanly and the pool can fall back to its best component. With
no gates and three bands it is exactly the first pool.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..blend import THRESHOLDS


class GatedPool(nn.Module):
    def __init__(self, n_comp: int, n_thr: int = len(THRESHOLDS), n_gates: int = 0, bands: tuple | None = None) -> None:
        """``bands``: the first threshold index of each band, or ``None`` for one set of weights per threshold."""
        super().__init__()
        if bands is None:
            band = np.arange(n_thr)
        else:
            band = np.searchsorted(np.asarray(bands), np.arange(n_thr), side="right") - 1
        self.register_buffer("band", torch.as_tensor(band, dtype=torch.long))
        n_band = int(band.max()) + 1
        self.n_comp, self.n_thr, self.n_gates, self.n_band = n_comp, n_thr, n_gates, n_band
        start = float(np.log(np.expm1(1.0 / n_comp)))                                      # weights start at 1/C
        self.a = nn.Parameter(torch.full((n_band, n_comp), start))
        self.d = nn.Parameter(torch.zeros(n_band, n_comp))                                  # change with lead
        self.g = nn.Parameter(torch.zeros(n_band, n_comp, max(n_gates, 1)))                 # change with each gate
        self.b = nn.Parameter(torch.zeros(n_thr))
        self.e = nn.Parameter(torch.zeros(n_thr))
        self.h = nn.Parameter(torch.zeros(n_thr, max(n_gates, 1)))

    def weights(self, lead: torch.Tensor, gates: torch.Tensor | None = None) -> torch.Tensor:
        """``(rows, thresholds, components)``: each component's weight at each threshold."""
        raw = self.a[self.band][None] + self.d[self.band][None] * lead[:, None, None]
        if self.n_gates and gates is not None:
            raw = raw + torch.einsum("kcg,ng->nkc", self.g[self.band], gates)
        return nn.functional.softplus(raw)

    def logits(self, z: torch.Tensor, present: torch.Tensor, lead: torch.Tensor,
               gates: torch.Tensor | None = None) -> torch.Tensor:
        w = self.weights(lead, gates)
        wp = w * present[:, None, :]
        mean = (wp * z.transpose(1, 2)).sum(-1) / wp.sum(-1).clamp_min(1e-6)
        out = w.sum(-1) * mean + self.b[None] + self.e[None] * lead[:, None]
        if self.n_gates and gates is not None:
            out = out + gates @ self.h.T
        return out

    def forward(self, z, present, lead, gates=None) -> torch.Tensor:
        return torch.sigmoid(self.logits(z, present, lead, gates))

    def penalty(self, smooth: float = 0.0, ridge: float = 0.0) -> torch.Tensor:
        """Smoothness of the weights across neighbouring thresholds, and a ridge on everything a gate changes."""
        total = self.a.new_zeros(())
        if smooth > 0 and self.n_band > 1:
            total = total + smooth * ((self.a[1:] - self.a[:-1]) ** 2).mean()
            total = total + smooth * ((self.d[1:] - self.d[:-1]) ** 2).mean()
        if ridge > 0 and self.n_gates:
            total = total + ridge * ((self.g ** 2).mean() + (self.h ** 2).mean())
        return total


def fit_pool(z: np.ndarray, present: np.ndarray, lead: np.ndarray, y: np.ndarray, gates: np.ndarray | None = None,
             weights: np.ndarray | None = None, bands: tuple | None = None, smooth: float = 1.0, ridge: float = 0.1,
             iters: int = 200, threshold_weights: np.ndarray | None = None) -> GatedPool:
    """Fit a :class:`GatedPool` on out-of-fold rows.

    ``z`` (N, C, K) components' log-odds, ``present`` (N, C), ``lead`` (N,) days, ``y`` (N,) mm, ``gates`` (N, G) or
    ``None``, ``weights`` (N,) row weights (see :mod:`pluvium.tails.stratify`), ``threshold_weights`` (K,) the share of
    each threshold in the loss (uniform by default).

    ``smooth`` ties each threshold's weights to its neighbours'; ``ridge`` holds back what the gates may change. A
    ridge of 1 leaves a gate almost unused even when a component is far better in one season (a pool fitted on that
    season alone does twice as well in a synthetic test); 0.1 lets it act and is the default."""
    keep = np.isfinite(y)
    zt = torch.from_numpy(np.ascontiguousarray(z[keep], np.float32))
    pt = torch.from_numpy(present[keep].astype(np.float32))
    lt = torch.from_numpy((lead[keep] / 14.0).astype(np.float32))
    yt = torch.from_numpy((y[keep, None] >= THRESHOLDS[None]).astype(np.float32))
    gt = None if gates is None else torch.from_numpy(np.ascontiguousarray(gates[keep], np.float32))
    wt = torch.ones(zt.shape[0]) if weights is None else torch.from_numpy(np.asarray(weights, np.float32)[keep])
    wt = wt / wt.sum()
    tw = torch.ones(zt.shape[2]) if threshold_weights is None else torch.from_numpy(np.asarray(threshold_weights, np.float32))
    tw = tw / tw.sum()
    pool = GatedPool(z.shape[1], z.shape[2], 0 if gates is None else gates.shape[1], bands)
    opt = torch.optim.LBFGS(pool.parameters(), lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        rows = nn.functional.binary_cross_entropy_with_logits(pool.logits(zt, pt, lt, gt), yt, reduction="none") @ tw
        loss = (rows * wt).sum() + pool.penalty(smooth * 1e-3, ridge * 1e-3)
        loss.backward()
        return loss

    opt.step(closure)
    pool.eval()
    return pool


@torch.no_grad()
def predict_pool(pool: GatedPool, z: np.ndarray, present: np.ndarray, lead, gates: np.ndarray | None = None,
                 chunk: int = 400_000) -> np.ndarray:
    """P(rain >= THRESHOLDS) ``(N, K)``; never more likely at a higher threshold."""
    n = z.shape[0]
    lead = np.broadcast_to(np.asarray(lead, np.float32), (n,)) / 14.0
    out = []
    for i in range(0, n, chunk):
        s = slice(i, i + chunk)
        g = None if gates is None else torch.from_numpy(np.ascontiguousarray(gates[s], np.float32))
        out.append(pool(torch.from_numpy(np.ascontiguousarray(z[s], np.float32)),
                        torch.from_numpy(present[s].astype(np.float32)),
                        torch.from_numpy(np.ascontiguousarray(lead[s], np.float32)), g).numpy())
    p = np.concatenate(out) if out else np.zeros((0, len(THRESHOLDS)), np.float32)
    return np.minimum.accumulate(p, axis=1)


__all__ = ["GatedPool", "fit_pool", "predict_pool"]
