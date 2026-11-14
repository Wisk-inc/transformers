# Copyright 2026 Nathan. Apache-2.0.
"""
Re-weighing the experts as the season unfolds: online aggregation.

A pool fitted on last year's weather has last year's opinion of each system. Systems are upgraded, seasons differ, and
a component that was best in summer can trail in a continental winter. This aggregator keeps a weight per expert and
updates it after every verified day, by how much better or worse that expert did than the mixture (Bernstein online
aggregation, Wintenberger 2017; used for probabilistic forecasts by Berrisch and Ziel 2021).

* **Experts** are every component and the batch-fitted pool itself. If the batch pool is best, the weight goes to it
  and nothing changes; if one component pulls ahead for weeks, the mixture follows it.
* **Guarantee**: the mixture's cumulative loss is close to that of the best expert in hindsight, whatever happens.
  That is the property a forecast needs to have no weak season.
* **Separate weights** per lead group and band of thresholds: the best expert for drizzle tomorrow is not the best
  for 50 mm next week.
* **Losses** are Brier scores scaled by the mixture's own running score in that band, so one learning rate serves
  1 mm and 50 mm alike.
* **Fixed share**: a small part of the weight is returned to the prior at every update, so an expert that was written
  off can come back.
* **Causal**: weights used for a forecast come only from days already verified.
"""

from __future__ import annotations

import numpy as np

from ..blend import THRESHOLDS

LEAD_GROUPS = ((1, 3), (4, 7), (8, 10), (11, 14))
BANDS = (0, 5, 11)


class OnlineAggregator:
    def __init__(self, n_experts: int, eta: float = 2.0, share: float = 0.02, prior: np.ndarray | None = None,
                 lead_groups: tuple = LEAD_GROUPS, bands: tuple = BANDS, n_thr: int = len(THRESHOLDS)) -> None:
        self.eta, self.share = float(eta), float(share)
        self.lead_groups = tuple(lead_groups)
        self.band = np.searchsorted(np.asarray(bands), np.arange(n_thr), side="right") - 1
        self.n_band = int(self.band.max()) + 1
        prior = np.full(n_experts, 1.0 / n_experts) if prior is None else np.asarray(prior, np.float64)
        self.prior = prior / prior.sum()
        self.logw = np.tile(np.log(self.prior)[None, None], (len(self.lead_groups), self.n_band, 1))
        self.scale = np.full((len(self.lead_groups), self.n_band), np.nan)        # running mixture loss per band
        self.updates = 0

    # ------------------------------------------------------------------------------------ weights --
    def weights(self) -> np.ndarray:
        """``(lead groups, bands, experts)``, each row summing to one."""
        w = np.exp(self.logw - self.logw.max(-1, keepdims=True))
        return w / w.sum(-1, keepdims=True)

    def group_of(self, lead: np.ndarray) -> np.ndarray:
        lead = np.asarray(lead)
        out = np.full(lead.shape, len(self.lead_groups) - 1, np.int64)
        for g, (lo, hi) in enumerate(self.lead_groups):
            out[(lead >= lo) & (lead <= hi)] = g
        return out

    def combine(self, p: np.ndarray, present: np.ndarray, lead: np.ndarray) -> np.ndarray:
        """The mixture's chances ``(N, K)`` from the experts' ``p`` (N, E, K); absent experts' weight is shared out."""
        w = self.weights()[self.group_of(lead)][:, self.band, :]                    # (N, K, E)
        w = w * present[:, None, :]
        w = w / np.maximum(w.sum(-1, keepdims=True), 1e-12)
        out = (w * p.transpose(0, 2, 1)).sum(-1)
        return np.minimum.accumulate(out, axis=1)

    # ------------------------------------------------------------------------------------- update --
    def update(self, p: np.ndarray, present: np.ndarray, lead: np.ndarray, y: np.ndarray,
               weights: np.ndarray | None = None) -> None:
        """Learn from verified rows: ``p`` (N, E, K), ``present`` (N, E), ``lead`` (N,) days, ``y`` (N,) mm, and
        optional row ``weights`` (rows kept with unequal chances carry the inverse chance)."""
        ok = np.isfinite(y)
        if not ok.any():
            return
        p, present, lead, y = p[ok], present[ok], np.asarray(lead)[ok], y[ok]
        rw = np.ones(y.shape[0], np.float64) if weights is None else np.asarray(weights, np.float64)[ok]
        truth = (y[:, None] >= THRESHOLDS[None]).astype(np.float32)                 # (N, K)
        mix = self.combine(p, present, lead)
        group = self.group_of(lead)
        for g in range(len(self.lead_groups)):
            rows = group == g
            if not rows.any():
                continue
            err = (p[rows] - truth[rows][:, None, :]) ** 2                          # (n, E, K)
            err_mix = (mix[rows] - truth[rows]) ** 2                                # (n, K)
            here = present[rows] * rw[rows][:, None]                                # weighted presence
            for b in range(self.n_band):
                cols = self.band == b
                seen = here.sum(0) > 0
                if not seen.any():
                    continue
                n_e = np.maximum(here.sum(0), 1e-12)
                loss = (err[:, :, cols].mean(2) * here).sum(0) / n_e                # per expert, over its own rows
                loss_mix = float((err_mix[:, cols].mean(1) * rw[rows]).sum() / rw[rows].sum())
                old = self.scale[g, b]
                self.scale[g, b] = loss_mix if not np.isfinite(old) else 0.95 * old + 0.05 * loss_mix
                regret = (loss_mix - loss) / max(self.scale[g, b], 1e-9)            # positive: better than the mixture
                regret = np.clip(regret, -2.0, 2.0)
                step = self.eta * regret - (self.eta * regret) ** 2 * 0.5
                self.logw[g, b, seen] += step[seen]
                if self.share > 0:                                                  # fixed share: nobody is written off
                    w = np.exp(self.logw[g, b] - self.logw[g, b].max())
                    w = (1 - self.share) * w / w.sum() + self.share * self.prior
                    self.logw[g, b] = np.log(w)
        self.updates += 1

    def state(self) -> dict:
        return {"logw": self.logw.copy(), "scale": self.scale.copy(), "updates": self.updates}

    def load_state(self, state: dict) -> None:
        self.logw, self.scale, self.updates = state["logw"].copy(), state["scale"].copy(), int(state["updates"])


def run_online(agg: OnlineAggregator, p: np.ndarray, present: np.ndarray, lead: np.ndarray, y: np.ndarray,
               issued: np.ndarray, known: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Aggregate a whole period causally: rows are forecast in order of ``issued`` (hours), and each batch of
    forecasts sees only the rows whose truth was ``known`` (hours) before it was issued. Returns ``(N, K)``."""
    out = np.zeros((p.shape[0], p.shape[2]), np.float32)
    used = np.zeros(p.shape[0], bool)
    for t in np.unique(issued):
        ready = (~used) & (known <= t)
        if ready.any():
            for day in np.unique(known[ready]):                                    # one update per verified day
                rows = ready & (known == day)
                agg.update(p[rows], present[rows], lead[rows], y[rows], None if weights is None else weights[rows])
            used |= ready
        now = issued == t
        out[now] = agg.combine(p[now], present[now], lead[now])
    return out


__all__ = ["BANDS", "LEAD_GROUPS", "OnlineAggregator", "run_online"]
