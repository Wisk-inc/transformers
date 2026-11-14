# Copyright 2026 Nathan. Apache-2.0.
"""
Out-of-fold forecasts: what each component says about days it was not trained on.

The pool that combines Pluvium Blend's components must be fitted on forecasts the components made for days they never
saw, or it learns to trust whichever component memorised its training data best. The folds are alternate weeks of the
day forecast: a model trained on even weeks forecasts the odd weeks and the reverse. A week is longer than a rain
system lasts, so neighbouring days seldom straddle folds, and every season lands in both.

``oof_rows`` returns the log-odds of every component for a chosen set of rows. Everything downstream (the pool, the
record-storm hand-over, design choices by cross-validation) is fitted on these rows and nothing else.
"""

from __future__ import annotations

import numpy as np

from .. import blend as B
from ..benchmarks.fullyear import ENSEMBLES
from ..tails.stratify import table_rows
from ..util import log


def components_for(systems: list, with_neural: bool = False) -> dict:
    """Component name -> the systems it reads: each system alone, one network over all of them, and where Pluvium's
    network runs, the network alone and one network over all seven."""
    models = {s: [s] for s in systems}
    models["blend_ops"] = list(systems)
    if with_neural:
        models["neural_pp"] = ["neural"]
        models["blend"] = list(systems) + ["neural"]
    return models


def parity(table: dict) -> np.ndarray:
    """0 or 1 per block: the parity of the week of the day forecast."""
    valid = np.asarray(table["run"]) + 24 * (np.asarray(table["lead"]) - 1)
    return (valid // (24 * 7)) % 2


def has(table: dict, i: int, members: list, single: bool) -> bool:
    """Whether a component can forecast block ``i``: a single system must be there; a network over several needs any."""
    if single:
        return table["feats"][members[0]][i] is not None
    return any(table["feats"][s][i] is not None for s in members)


def usable(table: dict, blocks, members: list, single: bool) -> np.ndarray:
    return np.asarray([i for i in blocks if has(table, int(i), members, single)], np.int64)


def fit_component(table: dict, blocks: np.ndarray, members: list, n: int, max_rows: int = 400_000, seed: int = 0,
                  stratified: bool = True, heavy: float = 25.0, signal_systems: list | None = None):
    """One component on ``blocks``: the same small network for every component, trained on the same kind of rows."""
    x, y, w, _ = table_rows(table, blocks, members, n, max_rows, seed=seed, heavy=heavy, stratified=stratified,
                            signal_systems=signal_systems)
    if x.shape[0] == 0:
        return None
    return B.fit(x, y, seed=seed, max_rows=max_rows, weights=w if stratified else None)


def block_logits(net, table: dict, i: int, members: list, n: int, cells: np.ndarray | None = None) -> np.ndarray:
    """A component's log-odds of every threshold at one block's places (``cells`` picks a subset)."""
    x = B.assemble({s: table["feats"][s][i] for s in members}, members, ENSEMBLES, table["clim"][i],
                   float(table["lead"][i]), n)
    return B.to_logits(B.predict(net, x if cells is None else x[cells]))


def oof_rows(table: dict, order: list, models: dict, n: int, max_rows: int = 400_000, seed: int = 0,
             stratified: bool = True, verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold log-odds ``(rows, components, thresholds)`` and which components were present ``(rows, components)``.

    ``order`` lists ``(block, cells)`` pairs; rows follow that order. Each component is fitted twice, on the blocks of
    one week parity among those in ``order``, and forecasts the blocks of the other parity."""
    blocks = np.array([i for i, _ in order], np.int64)
    week = parity(table)
    offsets = np.cumsum([0] + [len(c) for _, c in order])
    z = np.zeros((int(offsets[-1]), len(models), len(B.THRESHOLDS)), np.float16)
    present = np.zeros((int(offsets[-1]), len(models)), bool)
    signal_systems = sorted({s for members in models.values() for s in members})
    for k, (name, members) in enumerate(models.items()):
        single = len(members) == 1
        mine = usable(table, blocks, members, single)
        for fold in (0, 1):
            net = fit_component(table, mine[week[mine] != fold], members, n, max_rows, seed, stratified,
                                signal_systems=signal_systems)
            if net is None:
                continue
            for r, (i, cells) in enumerate(order):
                if week[i] != fold or not has(table, i, members, single):
                    continue
                z[offsets[r]:offsets[r + 1], k] = block_logits(net, table, i, members, n, cells)
                present[offsets[r]:offsets[r + 1], k] = True
        if verbose:
            log(f"    out-of-fold: {name} done")
    return z, present


def row_arrays(table: dict, order: list) -> dict:
    """Per-row truth and context for ``order``: rain, lead (days), the hour the day forecast starts, block and place."""
    y = np.concatenate([table["y"][i][c] for i, c in order])
    lead = np.concatenate([np.full(len(c), table["lead"][i], np.int16) for i, c in order])
    valid = np.concatenate([np.full(len(c), table["run"][i] + 24 * (table["lead"][i] - 1), np.int64) for i, c in order])
    block = np.concatenate([np.full(len(c), i, np.int32) for i, c in order])
    cell = np.concatenate([np.asarray(c, np.int32) for _, c in order])
    return {"y": y.astype(np.float32), "lead": lead, "valid": valid, "block": block, "cell": cell}


__all__ = ["block_logits", "components_for", "fit_component", "has", "oof_rows", "parity", "row_arrays", "usable"]
