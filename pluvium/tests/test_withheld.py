# Copyright 2026 Nathan. Apache-2.0.
"""The withheld-storm experiment on a synthetic year with three planted storms far above everything else."""

import numpy as np

from pluvium.blend import THRESHOLDS
from pluvium.tails.withheld import control_days, split_blocks, storm_days, valid_days, withheld_experiment
from pluvium.util import hour_of

STORMS = (40, 90, 130)                                                    # the days with a storm


def make_table(days=160, n=80, seed=0):
    """One block per day at lead 1: ordinary days, and three storm days five times wetter than anything else."""
    rng = np.random.default_rng(seed)
    clim = {k: np.full(n, v, np.float32) for k, v in (("clim_p1", 0.4), ("clim_p10", 0.12), ("clim_p25", 0.03),
                                                       ("clim_p50", 0.005), ("clim_mean", np.log1p(3.0)))}
    table = {"run": hour_of("2024-04-01T00") + 24 * np.arange(days), "lead": np.ones(days, np.int64), "y": [],
             "clim": [], "feats": {"ens": [], "gfs": []}, "raw_p": [], "raw_mean": [], "raw_n": []}
    for d in range(days):
        scale = 60.0 if d in STORMS else 6.0 * rng.uniform(0.3, 1.6)
        truth = (rng.gamma(0.8, scale, n)).astype(np.float32)
        seen = truth * rng.lognormal(0, 0.25, n)                          # what the systems forecast: close to it
        table["y"].append(truth)
        table["clim"].append(clim)
        table["feats"]["ens"].append({"mean": np.log1p(seen).astype(np.float32)})
        table["feats"]["gfs"].append({"mean": np.log1p(truth * rng.lognormal(0, 0.6, n)).astype(np.float32)})
        table["raw_mean"].append(seen.astype(np.float32))
        raw_p = 1 / (1 + np.exp(-(np.log1p(seen)[:, None] - np.log1p(THRESHOLDS)[None]) * 6.0))
        table["raw_p"].append(raw_p.astype(np.float16))
        table["raw_n"].append(2)
    return table


def test_storm_days_are_found_and_kept_apart():
    table = make_table()
    days = valid_days(table)
    found = storm_days(table, top=3)
    assert sorted(found) == [int(days[d]) for d in STORMS]
    assert storm_days(table, top=3, measure="wet") and len(storm_days(table, top=5, gap_days=5)) == 5
    table["y"][41] = table["y"][40] * 0.99                                # the day after a storm is the same storm
    assert int(days[41]) not in storm_days(table, top=3, gap_days=5)
    control = control_days(table, found, count=10)
    assert len(control) == 10 and all(min(abs(c - s) for s in found) > 24 * 5 for c in control)
    train, storm, ordinary = split_blocks(table, found, control)
    assert storm.size == 3 and ordinary.size == 10 and train.size == 160 - 10 - 3 * 5   # two days either side go too
    assert not set(train) & set(storm) and not set(train) & set(ordinary)


def test_handover_restores_withheld_storms_and_leaves_ordinary_days_alone():
    table = make_table()
    res = withheld_experiment(table, 80, ["ens", "gfs"], top=3, quantiles=(0.999,), widths=(0.6931,), max_rows=30_000,
                              control=10)
    assert len(res["storms"]) == 3 and res["largest_training_day_mm"] < 12.0
    storm, control = res["storm"], res["control"]
    label = "q0.999 w0.69"
    observed = storm["observed_mean_mm"]
    calibrated, handed = storm["calibrated"]["amount_mm"], storm[label]["amount_mm"]
    assert calibrated < 0.8 * observed                                    # never having seen such a day, it shrinks it
    assert abs(handed - observed) < abs(calibrated - observed)           # the hand-over gives most of it back
    assert storm[label]["brier"]["50"] < storm["calibrated"]["brier"]["50"]
    assert storm[label]["brier"]["100"] < storm["calibrated"]["brier"]["100"]
    assert storm[label]["rows_handed_over"] > 0.2                         # the wettest places of the storm days
    assert control[label]["rows_handed_over"] < 0.01                      # ordinary days: untouched
    for t in ("25", "50"):
        assert abs(control[label]["brier"][t] - control["calibrated"]["brier"][t]) < 5e-4
    assert res["edges_mm"][label] >= 20.0
