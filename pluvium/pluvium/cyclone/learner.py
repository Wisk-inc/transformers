# Copyright 2026 Nathan. Apache-2.0.
"""
A consensus that learns as the season goes: weights from every verified forecast.

Weights fixed from last season are last season's opinion. Models are upgraded between seasons (ECMWF's AIFS became
operational in 2025), a model can have a poor year, and a new model has no record at all. The learner here starts from
equal weights (or any prior) and updates after every forecast that has verified. Two rules are offered:

``minvar`` (the default)
    Minimum-variance weights from a running error covariance (:mod:`.minvar`). The weights are a function of the
    evidence so far, with no step size.

``eg``
    Exponentiated gradient descent on the squared error of the consensus itself. For a verified forecast, with
    ``e_s`` the error vector of system ``s`` and ``E = sum w_s e_s`` the consensus error,

        log w_s  <-  log w_s  -  eta * (e_s . E) / scale^2

    A system whose error points the same way as the consensus error loses weight; one that would have pulled the
    consensus back gains it. A small fixed share of weight is returned to the prior at each step, so no system is
    ever written off.

Both reward accuracy and independence together, not accuracy alone. In both, every forecast is scaled by the systems'
own mean error on it (with a floor at half the typical error for that lead), so one wild case does not decide the
weights and every lead counts alike.

Everything is causal. A forecast is learned from only once its valid time has passed, and against the position known
in real time (``CARQ``), never the post-season best track. Optionally the learner also keeps an online bias (as
:mod:`.bias`, but learned from the season so far) and the in-storm memory of :mod:`.online`.
"""

from __future__ import annotations

import numpy as np

from .bias import TrackBias
from .consensus import consensus
from .geodesy import offset_km
from .minvar import RunningCovariance
from .online import StormMemory
from .weights import GROUPS

# typical track error by lead, km: about the National Hurricane Center's official five-year means
TYPICAL_KM = {12: 45.0, 24: 70.0, 36: 95.0, 48: 120.0, 60: 150.0, 72: 175.0, 96: 250.0, 120: 345.0}


def typical_km(lead: int) -> float:
    leads = sorted(TYPICAL_KM)
    return float(np.interp(lead, leads, [TYPICAL_KM[k] for k in leads]))


class SeasonLearner:
    def __init__(self, systems: list, method: str = "minvar", eta: float = 0.1, share: float = 0.01,
                 groups: dict | None = None, online_bias: bool = False, bias_shrink: float = 15.0,
                 memory_gain: float = 0.0, prior: dict | None = None, floor: float = 0.5, shrink: float = 0.3,
                 to_equal: float = 0.25, pseudo: float = 8.0, halflife_days: float | None = 365.0) -> None:
        if method not in ("minvar", "eg", "equal"):
            raise ValueError(f"unknown method {method!r}")
        self.systems, self.method = list(systems), method
        self.eta, self.share, self.floor = float(eta), float(share), float(floor)
        self.groups = dict(groups or GROUPS)
        base = np.array([(prior or {}).get(s, 1.0) for s in self.systems], np.float64)
        self.prior = base / base.sum()
        self.logw = {g: np.log(self.prior).copy() for g in self.groups}
        self.cov = {g: RunningCovariance(self.systems, shrink, to_equal, pseudo, halflife_days) for g in self.groups}
        self.bias = TrackBias(bias_shrink) if online_bias else None
        self.memory_gain = float(memory_gain)
        self.memory = {}                              # storm id -> StormMemory
        self.pending = []                             # (verify hour, lead, {system: (lat, lon)}, points, carq)
        self.learned = 0

    # ------------------------------------------------------------------------------------ weights --
    def group_of(self, lead: int) -> str:
        for name, leads in self.groups.items():
            if int(lead) <= max(leads):
                return name
        return list(self.groups)[-1]

    def at(self, lead: int) -> dict:
        """``{system: weight}`` at a lead (the form :func:`pluvium.cyclone.consensus.consensus` takes)."""
        group = self.group_of(lead)
        if self.method == "minvar":
            w = self.cov[group].weights()
        elif self.method == "eg":
            lw = self.logw[group]
            w = np.exp(lw - lw.max())
        else:
            w = self.prior
        return {s: float(x) for s, x in zip(self.systems, w / w.sum())}

    # ----------------------------------------------------------------------------------- forecast --
    def predict(self, storm_id: str, init: int, points: dict, carq: dict) -> dict:
        """The consensus for one case, from what has been learned so far. The case is queued to be learned from once
        its leads verify (call :meth:`learn_until` with each new start time first)."""
        pts = points if self.bias is None else {s: self.bias.correct(s, p) for s, p in points.items()}
        scale = None
        if self.memory_gain > 0:
            mem = self.memory.setdefault(storm_id, StormMemory(gain=self.memory_gain))
            scale = mem.scale(int(init), list(pts))
            mem.remember(int(init), pts, carq)
        out = consensus(pts, self, scale=scale)
        for lead in sorted({ld for p in points.values() for ld in p}):
            self.pending.append((int(init) + int(lead), int(lead), {s: p[lead] for s, p in points.items() if lead in p},
                                 points, carq))
        return out

    # -------------------------------------------------------------------------------------- learn --
    def learn_until(self, now: int) -> int:
        """Learn from every queued forecast whose valid time is not after ``now``; returns how many were used."""
        ready = [p for p in self.pending if p[0] <= now]
        if not ready:
            return 0
        self.pending = [p for p in self.pending if p[0] > now]
        used = 0
        for verify, lead, at_lead, points, carq in sorted(ready, key=lambda p: p[0]):
            fix = carq.get(int(verify), {}).get(0)
            if fix is None:
                continue                                              # the storm was no longer being fixed
            vectors = {}
            for s, (lat, lon) in at_lead.items():
                if s in self.systems and np.isfinite(lat):
                    pos = (lat, lon)
                    if self.bias is not None:                         # learn weights on what the consensus uses
                        pos = self.bias.correct(s, points[s]).get(lead, pos)
                    e, n = offset_km(fix[0], fix[1], pos[0], pos[1])
                    vectors[s] = np.array([float(e), float(n)])
            if len(vectors) >= 2:
                self._update(int(verify), lead, vectors)
                used += 1
            if self.bias is not None:
                for s in at_lead:
                    self.bias.learn(s, points[s], {lead: (fix[0], fix[1])})
        self.learned += used
        return used

    def _update(self, hour: int, lead: int, vectors: dict) -> None:
        group = self.group_of(lead)
        e = np.stack([vectors[s] for s in vectors])                   # (systems present, 2)
        scale = np.sqrt(max(float((e ** 2).sum(1).mean()), (self.floor * typical_km(lead)) ** 2))
        e = e / scale
        self.cov[group].add(hour, {s: e[i] for i, s in enumerate(vectors)})
        if self.method != "eg":
            return
        idx = [self.systems.index(s) for s in vectors]
        lw = self.logw[group]
        w = np.exp(lw[idx] - lw[idx].max())
        lw[idx] -= self.eta * (e @ ((w / w.sum()) @ e))
        if self.share > 0:
            full = np.exp(lw - lw.max())
            full = (1 - self.share) * full / full.sum() + self.share * self.prior
            self.logw[group] = np.log(full)

    def table(self) -> dict:
        """``{group: {system: weight}}`` as it stands."""
        return {g: self.at(max(leads)) for g, leads in self.groups.items()}

    def state(self) -> dict:
        """What has been learned, as plain lists (for a JSON file). Forecasts still waiting to verify and the
        in-storm memory are not part of it: they belong to storms in progress."""
        return {"systems": list(self.systems), "learned": self.learned,
                "logw": {g: v.tolist() for g, v in self.logw.items()},
                "prod": {g: c.prod.tolist() for g, c in self.cov.items()},
                "count": {g: c.count.tolist() for g, c in self.cov.items()},
                "clock": {g: c.clock for g, c in self.cov.items()}}

    def load_state(self, state: dict) -> None:
        """Take over what another learner learned. Systems are matched by name: one the state does not know starts
        as an average member, one this learner does not have is dropped."""
        theirs = list(state["systems"])
        both = [s for s in self.systems if s in theirs]
        i = [self.systems.index(s) for s in both]
        j = [theirs.index(s) for s in both]
        for g in self.groups:
            if g not in state["prod"]:
                continue
            prod, count = np.asarray(state["prod"][g], np.float64), np.asarray(state["count"][g], np.float64)
            self.cov[g].prod[np.ix_(i, i)] = prod[np.ix_(j, j)]
            self.cov[g].count[np.ix_(i, i)] = count[np.ix_(j, j)]
            self.cov[g].clock = state.get("clock", {}).get(g)
            self.cov[g]._weights = None
            lw = np.asarray(state["logw"][g], np.float64)
            self.logw[g][i] = lw[j]
        self.learned = int(state.get("learned", 0))


__all__ = ["TYPICAL_KM", "SeasonLearner", "typical_km"]
