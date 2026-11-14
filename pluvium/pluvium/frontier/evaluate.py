# Copyright 2026 Nathan. Apache-2.0.
"""
Scoring pool designs on a place's stored components.

Given a store (:mod:`.store`), every quarter's pool is fitted on that quarter's out-of-fold rows and applied to its
test blocks, exactly as the first benchmark did, but in minutes, because no component is refitted. A design is a pool
(:class:`pluvium.pool.GatedPool`) and, optionally, two steps after it:

* the online aggregator (:mod:`pluvium.pool.online`), which re-weighs the components and the pool after every
  verified day. It is warmed up on the training year's out-of-fold rows and then runs through the test year with a
  delay of ``latency_hours`` between a day and its verification;
* the record-storm hand-over (:mod:`pluvium.tails.override`), which gives a forecast back to the raw consensus when
  it leaves the range of the training forecasts.

The result has the same form as the first benchmark's, with the design named ``stack`` as Pluvium Blend, so the same
reports read both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import blend as B
from ..benchmarks import fullyear as F
from ..pool import OnlineAggregator, build_gates, fit_pool, predict_pool
from ..tails.override import agree_share_from, blend_raw, handover
from ..tails.support import Support
from ..util import log
from .store import Store

HEAVY_INDEX = int(np.searchsorted(B.THRESHOLDS, 25.0))


@dataclass
class PoolDesign:
    name: str = "stack"
    #: first threshold index of each band, or None for weights per threshold (tied by ``smooth``)
    bands: tuple | None = None
    smooth: float = 1.0
    gates: str = "none"                     # "none", "season", "regime" or "both"
    ridge: float = 0.1
    rows: int = 300_000
    iters: int = 120
    #: re-weigh the components and the pool online
    online: bool = False
    eta: float = 2.0
    share: float = 0.02
    pool_prior: float = 0.5
    #: hand record storms back to the raw consensus
    record: bool = False
    record_quantile: float = 0.999
    record_width: float = 0.6931


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z.astype(np.float32)))


def uniform_like(weights: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Row indices drawn in proportion to the rows' weights: a sample that looks like a uniform draw from all rows
    (for statistics that are not means, such as the CSI cut)."""
    p = np.asarray(weights, np.float64)
    return rng.choice(p.size, size=min(size, p.size), replace=True, p=p / p.sum())


def fit_design(design: PoolDesign, d: dict, thresholds, rng: np.random.Generator):
    """A quarter's pool for a design and its CSI cuts, from the quarter's out-of-fold rows."""
    z, present, y, lead, valid, w = d["oof_z"], d["oof_present"], d["oof_y"], d["oof_lead"], d["oof_valid"], d["oof_w"]
    pick = np.arange(y.size) if y.size <= design.rows else np.sort(rng.choice(y.size, design.rows, replace=False))
    zf = z[pick].astype(np.float32)
    gates = build_gates(design.gates, zf, present[pick], valid[pick])
    pool = fit_pool(zf, present[pick], lead[pick].astype(np.float32), y[pick], gates, w[pick], bands=design.bands,
                    smooth=design.smooth, ridge=design.ridge, iters=design.iters)
    rows = uniform_like(w, 100_000, rng)
    zr = z[rows].astype(np.float32)
    p = predict_pool(pool, zr, present[rows], lead[rows].astype(np.float32),
                     build_gates(design.gates, zr, present[rows], valid[rows]))
    cuts = {f"{t:g}": F.best_cut(B.exceed(p, t), y[rows] >= t) for t in thresholds}
    return pool, cuts


def warm_start(agg: OnlineAggregator, design: PoolDesign, pool, d: dict, run_of_block: np.ndarray,
               latency: int) -> None:
    """Run the aggregator through a quarter's out-of-fold rows in time order (the training period), so it starts the
    test period knowing how the experts did last year."""
    from ..pool import run_online

    z = d["oof_z"].astype(np.float32)
    present, lead, y, valid, w = d["oof_present"], d["oof_lead"], d["oof_y"], d["oof_valid"], d["oof_w"]
    p_pool = predict_pool(pool, z, present, lead.astype(np.float32), build_gates(design.gates, z, present, valid))
    p = np.concatenate([sigmoid(z), p_pool[:, None, :]], 1)
    pres = np.concatenate([present, np.ones((present.shape[0], 1), bool)], 1)
    issued = run_of_block[d["oof_block"]]
    run_online(agg, p, pres, lead, y, issued, valid + 24 + latency, w)


def evaluate(store: Store, designs: list, thresholds=B.SCORED, case_days=(), latency: int = 48,
             components: bool = True, top: str = "stack", keep: dict | None = None) -> dict:
    """Score every design (and the components) on the store's test blocks; returns the benchmark's result dict.

    ``keep``, when given, is filled with the top design's chances for every test block (``{block: (n, K) float16}``),
    for the scenario and tail benchmarks that start from them."""
    table = store.table()
    n, names = store.n, store.names
    scores = F.Scores(thresholds, case_days)
    rng = np.random.default_rng(0)
    aggs, pending, weights_log = {}, {}, {}
    for q in store.quarters():
        d = store.quarter(q)
        fitted = {design.name: fit_design(design, d, thresholds, rng) for design in designs}
        supports = {design.name: Support.from_rows(d["oof_raw_mean"], d["oof_w"], design.record_quantile,
                                                   design.record_width) for design in designs if design.record}
        for design in designs:
            if design.online and design.name not in aggs:
                prior = np.concatenate([np.full(len(names), (1 - design.pool_prior) / len(names)), [design.pool_prior]])
                aggs[design.name] = OnlineAggregator(len(names) + 1, design.eta, design.share, prior)
                warm_start(aggs[design.name], design, fitted[design.name][0], d, store.run, latency)
                pending[design.name] = []
        order = sorted(range(d["test"].size), key=lambda r: (store.run[d["test"][r]], store.lead[d["test"][r]]))
        log(f"  {store.folder.name}: quarter {q}: pools fitted; scoring {len(order)} test blocks")
        for r in order:
            i = int(d["test"][r])
            z = d["test_z"][r].astype(np.float32)
            here = d["test_present"][r]
            if not here.any():
                continue
            present = np.broadcast_to(here, (n, len(names)))
            lead, run = int(store.lead[i]), int(store.run[i])
            valid = np.full(n, run + 24 * (lead - 1), np.int64)
            p_comp = sigmoid(z)
            if components:
                for k, name in enumerate(names):
                    if here[k]:
                        F._score(scores, name, i, table, p=p_comp[:, k], q=B.quantiles(p_comp[:, k]), cuts=d["cuts"][name])
            for design in designs:
                pool, cuts = fitted[design.name]
                p = predict_pool(pool, z, present, float(lead), build_gates(design.gates, z, present, valid))
                if design.online:
                    agg, wait = aggs[design.name], pending[design.name]
                    ready = [w for w in wait if w[0] <= run]
                    if ready:
                        for known in sorted({w[0] for w in ready}):
                            batch = [w for w in ready if w[0] == known]
                            agg.update(np.concatenate([w[1] for w in batch]).astype(np.float32),
                                       np.concatenate([w[2] for w in batch]), np.concatenate([w[3] for w in batch]),
                                       np.concatenate([w[4] for w in batch]))
                        pending[design.name] = wait = [w for w in wait if w[0] > run]
                    p_exp = np.concatenate([p_comp, p[:, None, :]], 1)
                    pres = np.concatenate([present, np.ones((n, 1), bool)], 1)
                    wait.append((run + 24 * lead + latency, p_exp.astype(np.float16), pres, np.full(n, lead), store.y[i]))
                    p = agg.combine(p_exp, pres, np.full(n, lead))
                if design.record:
                    p_raw = store.raw_p(i)
                    g = handover(supports[design.name], store.raw_mean[i], agree_share_from(p_raw, HEAVY_INDEX))
                    if g.max() > 1e-4:
                        p = blend_raw(p, p_raw, g)
                F._score(scores, design.name, i, table, p=p, q=B.quantiles(p), cuts=cuts)
                if keep is not None and design.name == top:
                    keep[i] = p.astype(np.float16)
        for name, agg in aggs.items():
            weights_log.setdefault(name, []).append(agg.weights()[:, 0, :].round(3).tolist())
    result = report(store, scores, list(names) if components else [], [d_.name for d_ in designs], top, thresholds)
    result["online_weights"] = {k: {"experts": list(names) + ["pool"], "by_quarter_days_1_3_light": [w[0] for w in v]}
                                for k, v in weights_log.items()}
    return result


def report(store: Store, scores: F.Scores, components: list, designs: list, top: str, thresholds) -> dict:
    """The benchmark's result dict: raw systems (from the store), components and designs (just scored)."""
    out = {"cells": store.n, "leads": {}, "groups": {}, "seasons": {}, "versus": {},
           "cases": list(store.raw.get("cases", [])) + scores.cases}
    raw = store.raw
    scored = components + designs
    for lead in range(1, 15):
        row = {m: raw["leads"][m][str(lead)] for m in raw["leads"] if str(lead) in raw["leads"][m]}
        row.update({m: scores.summary(m, [lead]) for m in scored if scores.data.get(m, {}).get(lead)})
        out["leads"][str(lead)] = row
    for g, leads in F.LEAD_GROUPS.items():
        out["groups"][g] = {**{m: raw["groups"][m][g] for m in raw["groups"]},
                            **{m: scores.summary(m, leads) for m in scored}}
        out["seasons"][g] = {s: {**{m: raw["seasons"][m][g][s] for m in raw["seasons"]},
                                 **{m: scores.summary(m, leads, s) for m in scored}} for s in F.SEASONS}
    rivals = [m for m in scored if m != top]
    for g, leads in F.LEAD_GROUPS.items():
        for t in thresholds:
            mine = scores.by_run(top, leads, t)
            for rival in rivals:
                theirs = scores.by_run(rival, leads, t)
                common = sorted(set(mine) & set(theirs))
                if len(common) < 3:
                    continue
                diff = np.array([mine[r] - theirs[r] for r in common])
                mean, lo, hi = F.block_bootstrap(diff)
                out["versus"].setdefault(g, {}).setdefault(f"{t:g}", {})[rival] = {"delta_bs": mean, "ci95": [lo, hi],
                                                                                    "runs": len(common)}
    out["region"] = store.folder.name
    return out


__all__ = ["HEAVY_INDEX", "PoolDesign", "evaluate", "fit_design", "report", "uniform_like", "warm_start"]
