# Copyright 2026 Nathan. Apache-2.0.
"""
Learn the tail from every heavy-rain row, without biasing the chances.

A day with 50 mm or more is under 1% of days outside the wet tropics. A uniform sample of 400,000 training rows holds
only a few thousand of them, and the rarest thresholds are learned from noise. This module draws rows unevenly on
purpose: every row where the observed rain, or any system's forecast, reaches ``heavy`` mm is kept, and the rest are
thinned to fill the budget. Each kept row carries the inverse of its chance of being drawn as a weight, so the weighted
loss is an unbiased estimate of the loss over all rows (Horvitz-Thompson). The chances stay calibrated, while the tail
is learned from several times more events.

Forecast-heavy rows are kept as well as observed-heavy ones: a false alarm at 50 mm teaches as much as a hit.
"""

from __future__ import annotations

import numpy as np


def heavy_rows(y: np.ndarray, signal: np.ndarray, heavy: float = 25.0) -> np.ndarray:
    """Rows where the observed rain or the wettest system's forecast reaches ``heavy`` mm (NaN rain never counts)."""
    return (np.nan_to_num(y, nan=-1.0) >= heavy) | (np.nan_to_num(signal, nan=-1.0) >= heavy)


def draw_plan(n_heavy: int, n_rest: int, budget: int, heavy_share: float = 0.5) -> tuple[float, float]:
    """Chances of keeping a heavy row and any other row so that about ``budget`` rows are kept in all. Heavy rows are
    all kept unless they alone would take more than ``heavy_share`` of the budget."""
    p_heavy = 1.0 if n_heavy <= heavy_share * budget else heavy_share * budget / max(n_heavy, 1)
    left = max(budget - p_heavy * n_heavy, 0.0)
    p_rest = min(1.0, left / max(n_rest, 1))
    return float(p_heavy), float(p_rest)


def tail_sample(y: np.ndarray, signal: np.ndarray, budget: int, rng: np.random.Generator, heavy: float = 25.0,
                heavy_share: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the rows to train on and their weights (the inverse of each row's chance of being drawn)."""
    known = np.isfinite(y)
    is_heavy = heavy_rows(y, signal, heavy) & known
    rest = known & ~is_heavy
    p_heavy, p_rest = draw_plan(int(is_heavy.sum()), int(rest.sum()), budget, heavy_share)
    u = rng.random(y.shape[0])
    keep = (is_heavy & (u < p_heavy)) | (rest & (u < p_rest))
    index = np.flatnonzero(keep)
    weights = np.where(is_heavy[index], 1.0 / max(p_heavy, 1e-12), 1.0 / max(p_rest, 1e-12)).astype(np.float32)
    return index, weights


def signal_of(per_system: dict, n: int) -> np.ndarray:
    """The wettest system's mean forecast (mm) at each of ``n`` places, from the systems' features (missing systems and
    places count as zero). ``mean`` features are stored as log(1 + mm)."""
    out = np.zeros(n, np.float32)
    for feats in per_system.values():
        if feats is not None and "mean" in feats:
            out = np.maximum(out, np.nan_to_num(np.expm1(feats["mean"]), nan=0.0))
    return out


def table_rows(table: dict, blocks: np.ndarray, systems: list, n: int, budget: int, seed: int = 0, heavy: float = 25.0,
               stratified: bool = True, signal_systems: list | None = None) -> tuple:
    """Training rows from a benchmark table (see :func:`pluvium.benchmarks.fullyear.build_table`): features, rain, row
    weights and each row's block. With ``stratified`` every heavy row is kept; without, the same budget is drawn
    uniformly (weights of one), which is the earlier behaviour.

    ``signal_systems`` are the systems whose forecasts mark a row as heavy (default: all in the table), so a model of
    one system and the blend of all are trained on the same rows."""
    from .. import blend as B
    from ..benchmarks.fullyear import ENSEMBLES

    rng = np.random.default_rng(seed)
    blocks = np.asarray(blocks, np.int64)
    signal_systems = list(table["feats"]) if signal_systems is None else signal_systems
    total = blocks.size * n
    if total == 0:
        return np.zeros((0, 1), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    if stratified:
        heavy_mask = []
        for i in blocks:
            y = table["y"][i]
            sig = signal_of({s: table["feats"][s][i] for s in signal_systems}, n)
            heavy_mask.append(heavy_rows(y, sig, heavy) & np.isfinite(y))
        n_heavy = int(sum(int(m.sum()) for m in heavy_mask))
        n_rest = int(sum(int(np.isfinite(table["y"][i]).sum()) for i in blocks)) - n_heavy
        p_heavy, p_rest = draw_plan(n_heavy, n_rest, budget)
    else:
        heavy_mask = [None] * blocks.size
        p_heavy = p_rest = min(1.0, budget / max(total, 1))
    xs, ys, ws, bs = [], [], [], []
    for k, i in enumerate(blocks):
        y = table["y"][i]
        u = rng.random(n)
        if stratified:
            hm = heavy_mask[k]
            keep = np.isfinite(y) & np.where(hm, u < p_heavy, u < p_rest)
            w = np.where(hm, 1.0 / max(p_heavy, 1e-12), 1.0 / max(p_rest, 1e-12))
        else:
            keep = np.isfinite(y) & (u < p_rest)
            w = np.ones(n)
        sel = np.flatnonzero(keep)
        if sel.size == 0:
            continue
        x = B.assemble({s: table["feats"][s][i] for s in systems}, systems, ENSEMBLES, table["clim"][i],
                       float(table["lead"][i]), n)
        xs.append(x[sel])
        ys.append(y[sel])
        ws.append(w[sel].astype(np.float32))
        bs.append(np.full(sel.size, i, np.int64))
    if not xs:
        return np.zeros((0, 1), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws), np.concatenate(bs)


def table_order(table: dict, blocks: np.ndarray, n: int, budget: int, seed: int = 0, heavy: float = 25.0,
                stratified: bool = True, signal_systems: list | None = None) -> tuple[list, np.ndarray]:
    """Which places of which blocks to keep, without building features: ``[(block, places)]`` and one weight per
    kept row, in that order. The rows the pool is fitted on are chosen with this (every heavy row, a share of the
    rest), and their out-of-fold forecasts are then computed only where needed."""
    rng = np.random.default_rng(seed)
    blocks = np.asarray(blocks, np.int64)
    signal_systems = list(table["feats"]) if signal_systems is None else signal_systems
    masks, n_heavy, n_known = [], 0, 0
    for i in blocks:
        y = table["y"][i]
        known = np.isfinite(y)
        hm = heavy_rows(y, signal_of({s: table["feats"][s][i] for s in signal_systems}, n), heavy) & known
        masks.append((known, hm))
        n_heavy += int(hm.sum())
        n_known += int(known.sum())
    if stratified:
        p_heavy, p_rest = draw_plan(n_heavy, n_known - n_heavy, budget)
    else:
        p_heavy = p_rest = min(1.0, budget / max(n_known, 1))
    order, weights = [], []
    for i, (known, hm) in zip(blocks, masks):
        u = rng.random(n)
        keep = known & np.where(hm, u < p_heavy, u < p_rest)
        sel = np.flatnonzero(keep)
        if sel.size:
            order.append((int(i), sel))
            weights.append(np.where(hm[sel], 1.0 / max(p_heavy, 1e-12), 1.0 / max(p_rest, 1e-12)).astype(np.float32))
    return order, (np.concatenate(weights) if weights else np.zeros(0, np.float32))


__all__ = ["draw_plan", "heavy_rows", "signal_of", "table_order", "table_rows", "tail_sample"]
