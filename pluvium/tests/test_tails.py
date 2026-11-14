# Copyright 2026 Nathan. Apache-2.0.
"""The tail: unbiased heavy-row sampling, the raw consensus, the support score and the record-storm hand-over."""

import numpy as np

from pluvium.blend import THRESHOLDS
from pluvium.tails.consensus import agreement, raw_consensus, system_chances
from pluvium.tails.override import agree_share_from, blend_raw, handover
from pluvium.tails.stratify import draw_plan, heavy_rows, signal_of, table_order, table_rows, tail_sample
from pluvium.tails.support import Support, weighted_quantile


def test_every_heavy_row_is_kept_and_weights_undo_the_thinning():
    rng = np.random.default_rng(0)
    y = rng.gamma(0.4, 6.0, 200_000).astype(np.float32)
    y[:50] = np.nan
    signal = y * rng.lognormal(0, 0.5, y.size)
    index, w = tail_sample(y, signal, 20_000, rng)
    heavy = heavy_rows(y, signal) & np.isfinite(y)
    assert np.isin(np.flatnonzero(heavy), index).all()                   # all of them, none thinned
    assert not np.isnan(y[index]).any() and 18_000 < index.size < 22_000
    assert np.all(w[heavy[index]] == 1.0) and w[~heavy[index]].min() > 5.0
    for t in (1.0, 10.0, 50.0):                                           # weighted rates are unbiased
        true = np.nanmean(y >= t)
        est = (w * (y[index] >= t)).sum() / w.sum()
        assert abs(est - true) < 0.1 * true + 1e-4
    naive = (y[index] >= 50.0).mean()
    assert naive > 3 * np.nanmean(y >= 50.0)                              # unweighted, the sample is far too wet


def test_draw_plan_caps_heavy_rows_when_they_would_fill_the_budget():
    assert draw_plan(100, 10_000, 1_000) == (1.0, 0.09)
    p_heavy, p_rest = draw_plan(5_000, 10_000, 1_000)
    assert abs(p_heavy - 0.1) < 1e-12 and abs(p_rest - 0.05) < 1e-12
    assert draw_plan(0, 100, 1_000) == (1.0, 1.0)


def _table(blocks=30, n=200, seed=1):
    rng = np.random.default_rng(seed)
    table = {"run": np.arange(blocks) * 24, "lead": np.ones(blocks, np.int64), "y": [], "clim": [], "feats": {"ens": [], "gfs": []}}
    clim = {k: np.full(n, v, np.float32) for k, v in (("clim_p1", 0.3), ("clim_p10", 0.1), ("clim_p25", 0.03), ("clim_mean", 1.0))}
    for _ in range(blocks):
        truth = rng.gamma(0.4, 6.0, n).astype(np.float32)
        table["y"].append(truth)
        table["clim"].append(clim)
        table["feats"]["ens"].append({"mean": np.log1p(truth * rng.lognormal(0, 0.4, n)).astype(np.float32)})
        table["feats"]["gfs"].append(None if rng.random() < 0.2 else {"mean": np.log1p(truth * rng.lognormal(0, 1.0, n)).astype(np.float32)})
    return table


def test_table_rows_and_order_agree_with_the_plan():
    table = _table()
    n, blocks = 200, np.arange(30)
    x, y, w, b = table_rows(table, blocks, ["ens", "gfs"], n, budget=1500, stratified=True)
    assert x.shape[0] == y.size == w.size == b.size and x.shape[1] > 5
    assert abs((w * (y >= 25)).sum() / w.sum() - np.mean(np.concatenate(table["y"]) >= 25)) < 0.01
    xu, yu, wu, _ = table_rows(table, blocks, ["ens", "gfs"], n, budget=1500, stratified=False)
    assert np.all(wu == 1.0) and (y >= 25).sum() > 2 * (yu >= 25).sum()
    order, weights = table_order(table, blocks, n, 1500)
    assert sum(len(c) for _, c in order) == weights.size and all(len(c) for _, c in order)
    sig = signal_of({"ens": table["feats"]["ens"][0], "gfs": None}, n)
    assert sig.shape == (n,) and np.all(sig >= 0)


def test_raw_consensus_gives_every_system_one_vote():
    n = 4
    ens = np.array([[60.0, 0.0, 30.0, 0.0]] * 9 + [[0.0, 0.0, 30.0, 0.0]])   # ten members: 9 of 10 reach 50 mm at place 0
    single = np.array([[80.0, 0.0, 0.0, 0.0]])
    p, mean = raw_consensus({"ens": ens, "ifs": single}, n)
    k50 = int(np.searchsorted(THRESHOLDS, 50.0))
    assert abs(p[0, k50] - 0.95) < 1e-6 and p[1, k50] == 0.0                 # (0.9 + 1.0) / 2, not 10 of 11
    assert abs(mean[0] - (54.0 + 80.0) / 2) < 1e-4
    assert np.all(np.diff(p, axis=1) <= 1e-9)
    assert system_chances(np.array([[np.nan, 5.0]]))[0].sum() == 0.0
    np.testing.assert_array_equal(agreement({"ens": ens, "ifs": single}, 50.0), [2, 0, 0, 0])
    assert raw_consensus({}, n)[0].shape == (n, len(THRESHOLDS))


def test_support_and_handover_leave_ordinary_days_alone():
    rng = np.random.default_rng(2)
    train = rng.gamma(0.5, 8.0, 100_000)
    w = np.ones_like(train)
    support = Support.from_rows(train, w)
    assert abs(support.edge_mm - np.quantile(train, 0.999)) < 2.0
    assert abs(weighted_quantile(np.array([1.0, 2.0, 3.0, 4.0]), 0.5, np.array([1.0, 1.0, 1.0, 10.0])) - 4.0) < 1e-9
    assert Support.from_rows(np.full(100, 2.0)).edge_mm == 20.0             # the floor in a dry place
    g = handover(support, train)
    assert np.mean(g > 0) <= 0.0011 and np.all(g[train <= support.edge_mm] == 0.0)   # exactly nothing inside
    record = np.array([support.edge_mm, 1.4 * support.edge_mm, 2.1 * support.edge_mm, 6 * support.edge_mm])
    g = handover(support, record)
    assert g[0] == 0.0 and 0.3 < g[1] < 0.6 and g[2] >= 0.95 - 1e-6 and g[3] >= 0.95 - 1e-6
    lone = handover(support, record, agree_share=np.full(4, 0.1))            # one system in ten calls for it
    assert np.all(lone <= 0.2 + 1e-6) and lone[0] == 0.0                     # at most a fifth of the trust


def test_blend_raw_restores_a_shrunk_record_forecast():
    k50 = int(np.searchsorted(THRESHOLDS, 50.0))
    calibrated = np.linspace(0.9, 0.05, len(THRESHOLDS))[None].astype(np.float32)   # shrunk: 40% at 50 mm, say
    raw = np.ones((1, len(THRESHOLDS)), np.float32)
    raw[0, k50 + 3:] = 0.3
    out = blend_raw(calibrated, raw, np.array([0.9]))
    assert out[0, k50] > 0.9 and np.all(np.diff(out, axis=1) <= 1e-9)
    same = blend_raw(calibrated, raw, np.array([0.0]))
    np.testing.assert_allclose(same, calibrated, atol=1e-7)
    assert agree_share_from(raw, k50)[0] == 1.0
