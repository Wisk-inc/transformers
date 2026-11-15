# Copyright 2026 Nathan. Apache-2.0.
"""
What happens when the storm is bigger than anything in training? The withheld-storm experiment.

A calibrated forecast pulls a record storm towards what it has seen. That cannot be measured on ordinary days, and
the test year must not be used to tune a remedy. So the situation is staged inside the training year:

1. the wettest days of the training year are found (:func:`storm_days`), each with the days around it;
2. a calibrated forecast is trained with those storms removed, so the largest storm it has seen is smaller;
3. the withheld storms are then forecast, with and without the record-storm hand-over
   (:mod:`pluvium.tails.override`), and scored. A set of ordinary days is withheld alongside as a control, to show
   the hand-over leaves them alone.

The hand-over's two settings (how far outside the training range a forecast must be, and how fast trust moves) can be
chosen here without touching the test year.
"""

from __future__ import annotations

import numpy as np

from .. import blend as B
from .override import agree_share_from, blend_raw, handover, tail_trust
from .stratify import table_rows
from .support import Support

HEAVY_INDEX = int(np.searchsorted(B.THRESHOLDS, 25.0))


def valid_days(table: dict) -> np.ndarray:
    """The day each block forecasts, as hours since the epoch at 00 UTC."""
    return np.asarray(table["run"], np.int64) + 24 * (np.asarray(table["lead"], np.int64) - 1)


def storm_days(table: dict, top: int = 3, gap_days: int = 5, measure: str = "mean") -> list:
    """The ``top`` wettest days in a table, at least ``gap_days`` apart (so they are different storms), wettest
    first. A day's size is the mean rain over the places, or with ``measure="wet"`` the share with 50 mm or more."""
    days, size = {}, {}
    for i, day in enumerate(valid_days(table)):
        if int(day) in days:
            continue
        y = table["y"][i]
        ok = np.isfinite(y)
        if ok.mean() < 0.5:
            continue
        days[int(day)] = i
        size[int(day)] = float(np.mean(y[ok] >= 50.0)) if measure == "wet" else float(np.mean(y[ok]))
    chosen = []
    for day in sorted(size, key=lambda d: -size[d]):
        if all(abs(day - other) >= 24 * gap_days for other in chosen):
            chosen.append(day)
        if len(chosen) == top:
            break
    return chosen


def split_blocks(table: dict, storms: list, control: list, halo_days: int = 2) -> tuple:
    """``(train, storm, control)`` block indices. Blocks forecasting a storm day, or a day within ``halo_days`` of
    one, are kept out of training; so are the control days."""
    valid = valid_days(table)
    near = np.zeros(valid.size, bool)
    for day in storms:
        near |= np.abs(valid - day) <= 24 * halo_days
    is_storm = np.isin(valid, np.asarray(storms, np.int64))
    is_control = np.isin(valid, np.asarray(control, np.int64)) & ~near
    return np.flatnonzero(~near & ~is_control), np.flatnonzero(is_storm), np.flatnonzero(is_control)


def control_days(table: dict, storms: list, count: int = 12, seed: int = 0, halo_days: int = 5) -> list:
    """Ordinary days, drawn at random away from the storms, to be withheld as a control."""
    rng = np.random.default_rng(seed)
    days = np.unique(valid_days(table))
    ok = np.ones(days.size, bool)
    for day in storms:
        ok &= np.abs(days - day) > 24 * halo_days
    pool = days[ok]
    return sorted(int(d) for d in rng.choice(pool, min(count, pool.size), replace=False))


def brier(p: np.ndarray, y: np.ndarray, t: float) -> float:
    ok = np.isfinite(y)
    return float(np.mean((B.exceed(p[ok], t) - (y[ok] >= t)) ** 2))


def score_blocks(table: dict, blocks: np.ndarray, net, systems: list, n: int, supports: dict,
                 thresholds=(25.0, 50.0, 100.0), trust: np.ndarray | None = None) -> dict:
    """Scores over ``blocks`` for the calibrated forecast, the raw consensus, and the hand-over under every support
    in ``supports`` (``{label: Support}``). With ``trust`` (:func:`pluvium.tails.override.tail_trust`) the guarded
    hand-over is scored too, under ``"<label> guarded"``."""
    from ..benchmarks.fullyear import ENSEMBLES

    guarded = {f"{label} guarded": support for label, support in supports.items()} if trust is not None else {}
    ps = {"calibrated": [], "raw": [], **{label: [] for label in supports}, **{label: [] for label in guarded}}
    ys, share = [], {label: [] for label in supports}
    for i in blocks:
        if not any(table["feats"][s][int(i)] is not None for s in systems):
            continue
        x = B.assemble({s: table["feats"][s][int(i)] for s in systems}, systems, ENSEMBLES, table["clim"][int(i)],
                       float(table["lead"][int(i)]), n)
        p_cal = B.predict(net, x)
        p_raw = table["raw_p"][int(i)].astype(np.float32)
        raw_mean = table["raw_mean"][int(i)]
        ps["calibrated"].append(p_cal)
        ps["raw"].append(p_raw)
        for label, support in supports.items():
            g = handover(support, raw_mean, agree_share_from(p_raw, HEAVY_INDEX))
            ps[label].append(blend_raw(p_cal, p_raw, g))
            share[label].append(g > 0)
            if trust is not None:
                ps[f"{label} guarded"].append(blend_raw(p_cal, p_raw, g, trust, only_raise=True))
        ys.append(table["y"][int(i)])
    if not ys:
        return {}
    y = np.concatenate(ys)
    ok = np.isfinite(y)
    out = {"rows": int(ok.sum()), "observed_mean_mm": float(y[ok].mean()),
           "observed_share": {f"{t:g}": float(np.mean(y[ok] >= t)) for t in thresholds}}
    for label, parts in ps.items():
        p = np.concatenate(parts)
        out[label] = {"brier": {f"{t:g}": brier(p, y, t) for t in thresholds},
                      "chance": {f"{t:g}": float(B.exceed(p[ok], t).mean()) for t in thresholds},
                      "amount_mm": float(B.quantiles(p[ok][::max(1, int(ok.sum()) // 20_000)]).mean())}
        if label in share:
            out[label]["rows_handed_over"] = float(np.concatenate(share[label])[ok].mean())
    return out


def withheld_experiment(table: dict, n: int, systems: list, top: int = 3, quantiles=(0.99, 0.995, 0.999),
                        widths=(0.4055, 0.6931, 1.0986), max_rows: int = 400_000, seed: int = 0, control: int = 12,
                        log=None) -> dict:
    """Run the experiment on a table built with ``keep_raw=True`` (the training year only). Returns the storms, and
    scores on the withheld storms and on the control days for every (quantile, width) pair."""
    storms = storm_days(table, top)
    ordinary = control_days(table, storms, control, seed)
    train, storm_blocks, control_blocks = split_blocks(table, storms, ordinary)
    x, y, w, block = table_rows(table, train, systems, n, max_rows, seed=seed, stratified=True)
    if log:
        log(f"withheld: {len(storms)} storms, {storm_blocks.size} storm blocks, {control_blocks.size} control blocks, "
            f"{x.shape[0]:,} training rows")
    net = B.fit(x, y, seed=seed, max_rows=max_rows, weights=w)
    rng = np.random.default_rng(seed + 1)                         # a uniform sample of the training rows
    share = min(1.0, 300_000 / max(train.size * n, 1))
    picks = [(int(i), rng.random(n) < share) for i in train]
    raw_train = np.concatenate([table["raw_mean"][i][m] for i, m in picks])
    supports = {f"q{q:g} w{wd:.2f}": Support.from_rows(raw_train, None, q, wd) for q in quantiles for wd in widths}
    trust = tail_trust(np.concatenate([table["raw_p"][i][m] for i, m in picks]).astype(np.float32),
                       np.concatenate([table["y"][i][m] for i, m in picks]), raw_train)
    largest = max(float(np.nanmean(table["y"][int(i)])) for i in train)
    return {"storms": [str(np.datetime64(int(d), "h"))[:10] for d in storms],
            "largest_training_day_mm": largest,
            "edges_mm": {label: s.edge_mm for label, s in supports.items()},
            "trust": {f"{t:g}": float(v) for t, v in zip(B.THRESHOLDS, trust)},
            "storm": score_blocks(table, storm_blocks, net, systems, n, supports, trust=trust),
            "control": score_blocks(table, control_blocks, net, systems, n, supports, trust=trust)}


__all__ = ["HEAVY_INDEX", "brier", "control_days", "score_blocks", "split_blocks", "storm_days", "valid_days",
           "withheld_experiment"]
