# Copyright 2026 Nathan. Apache-2.0.
"""The deployable frontier blend: trained from a table, saved, loaded, forecasting and learning."""

import json

import numpy as np
import pytest

from pluvium.blend import THRESHOLDS, exceed
from pluvium.blendlive import BlendModel
from pluvium.frontier.model import BANDS, FrontierBlend, load_blend
from pluvium.util import hour_of

K = len(THRESHOLDS)


def make_table(days=120, n=60, seed=0):
    """A year in small: two systems (one sharp, one noisy) at lead 1-3 over ``days`` runs."""
    rng = np.random.default_rng(seed)
    clim = {k: np.full(n, v, np.float32) for k, v in (("clim_p1", 0.4), ("clim_p10", 0.12), ("clim_p25", 0.03),
                                                       ("clim_p50", 0.005), ("clim_mean", np.log1p(3.0)))}
    table = {"run": [], "lead": [], "y": [], "clim": [], "feats": {"ens": [], "gfs": []}, "raw_p": [], "raw_mean": [],
             "raw_n": []}
    for d in range(days):
        for lead in (1, 2, 3):
            truth = rng.gamma(0.6, 8.0 * rng.uniform(0.3, 1.6), n).astype(np.float32)
            seen = truth * rng.lognormal(0, 0.2 + 0.1 * lead, n)
            table["run"].append(hour_of("2024-04-01T00") + 48 * d)
            table["lead"].append(lead)
            table["y"].append(truth)
            table["clim"].append(clim)
            table["feats"]["ens"].append({"mean": np.log1p(seen).astype(np.float32)})
            table["feats"]["gfs"].append(None if d % 9 == 0 else
                                         {"mean": np.log1p(truth * rng.lognormal(0, 1.0, n)).astype(np.float32)})
            table["raw_mean"].append(seen.astype(np.float32))
            table["raw_p"].append((1 / (1 + np.exp(-(np.log1p(seen)[:, None] - np.log1p(THRESHOLDS)[None]) * 6.0))).astype(np.float16))
            table["raw_n"].append(2)
    table["run"], table["lead"] = np.asarray(table["run"], np.int64), np.asarray(table["lead"], np.int64)
    return table, clim


@pytest.fixture(scope="module")
def trained():
    table, clim = make_table()
    n = 60
    grid = (np.linspace(10.0, 12.0, 6), np.linspace(-80.0, -78.0, 10))
    grid_clim = {m: {k: np.full((6, 10), float(v[0]), np.float32) for k, v in clim.items()} for m in range(1, 13)}
    cells = (np.repeat(grid[0], 10), np.tile(grid[1], 6))
    model = FrontierBlend.from_table(table, n, ["ens", "gfs"], {"name": "testland", "target_box": [10, 12, -80, -78]},
                                     grid_clim, grid, cells, max_rows=20_000, pool_rows=12_000,
                                     meta={"trained": ["2024-04-01T00", "2024-11-27T00"], "runs": 120})
    return model, table, clim


def test_training_gives_components_pool_weights_and_a_support_edge(trained):
    model, table, clim = trained
    assert list(model.components()) == ["ens", "gfs", "blend_ops"] and model.meta["kind"] == "frontier"
    assert model.online.updates > 20 and 20.0 <= model.support.edge_mm < 400.0
    w = model.weights()
    assert list(w) == ["ens", "gfs", "blend_ops", "pool"] and abs(sum(w.values()) - 1.0) < 1e-6
    assert w["gfs"] < w["ens"]                                             # the noisy system has lost weight
    i = 30
    per = {s: table["feats"][s][i] for s in ("ens", "gfs")}
    p = model.chances(per, clim, float(table["lead"][i]), 60)
    assert p.shape == (60, K) and np.all(np.diff(p, axis=1) <= 1e-6) and 0 <= p.min() and p.max() <= 1
    truth = table["y"][i]
    wet, dry = truth >= np.quantile(truth, 0.8), truth <= np.quantile(truth, 0.2)
    assert exceed(p, 10.0)[wet].mean() > exceed(p, 10.0)[dry].mean() + 0.2  # it follows the weather
    alone = model.chances({"ens": per["ens"]}, clim, 1.0, 60)              # one system missing: still a forecast
    assert alone.shape == (60, K) and np.isfinite(alone).all()
    p_exp, present = model.experts({"ens": per["ens"]}, clim, 1.0, 60)
    assert p_exp.shape == (60, 4, K) and present[0].tolist() == [True, False, True, True]


def test_record_storm_is_handed_to_the_raw_systems(trained):
    model, table, clim = trained
    n = 60
    feats = {"mean": np.full(n, np.log1p(900.0), np.float32)}              # far beyond anything in training
    raw = {"ens": np.full((10, n), 900.0, np.float32), "gfs": np.full((1, n), 850.0, np.float32)}
    calm = model.chances({"ens": feats, "gfs": feats}, clim, 1.0, n)
    handed = model.chances({"ens": feats, "gfs": feats}, clim, 1.0, n, raw=raw)
    k100 = int(np.searchsorted(THRESHOLDS, 100.0))
    assert handed[:, k100].mean() > 0.9 and handed[:, k100].mean() >= calm[:, k100].mean()
    ordinary = {"ens": np.full((10, n), 4.0, np.float32), "gfs": np.full((1, n), 6.0, np.float32)}
    small = {"mean": np.full(n, np.log1p(4.0), np.float32)}
    np.testing.assert_allclose(model.chances({"ens": small, "gfs": small}, clim, 1.0, n, raw=ordinary),
                               model.chances({"ens": small, "gfs": small}, clim, 1.0, n), atol=1e-7)


def test_save_load_round_trip_and_learning_moves_the_weights(trained, tmp_path):
    model, table, clim = trained
    folder = model.save(tmp_path / "blend")
    meta = json.loads((folder / "blend.json").read_text())
    assert meta["kind"] == "frontier" and meta["bands"] == list(BANDS) and meta["support"]["edge_mm"] == model.support.edge_mm
    back = load_blend(folder)
    assert isinstance(back, FrontierBlend) and back.weights() == model.weights()
    np.testing.assert_allclose(back.cells[0], model.cells[0])
    i = 12
    per = {s: table["feats"][s][i] for s in ("ens", "gfs")}
    np.testing.assert_allclose(back.chances(per, clim, 1.0, 60), model.chances(per, clim, 1.0, 60), atol=1e-6)
    before = back.weights()
    rng = np.random.default_rng(5)
    for _ in range(40):                                                    # a month in which the noisy system is right
        truth = rng.gamma(0.6, 8.0, 60).astype(np.float32)
        per = {"ens": {"mean": np.log1p(truth * rng.lognormal(0, 1.5, 60)).astype(np.float32)},
               "gfs": {"mean": np.log1p(truth * rng.lognormal(0, 0.05, 60)).astype(np.float32)}}
        p_exp, present = back.experts(per, clim, 1.0, 60)
        back.learn_rows(p_exp, present, np.full(60, 1), truth)
    after = back.weights()
    assert after["gfs"] > before["gfs"] + 0.1 and after["ens"] < before["ens"]
    assert back.learn(tmp_path / "cache", None, now=hour_of("2025-06-01T00")) == []   # no truth to hand: nothing learned
    classic = BlendModel(model.nets, None, model.systems, model.area_json, model.clim, model.grid, {"runs": 1})
    assert classic.components() == {"ens": ["ens"], "gfs": ["gfs"]}        # the first Blend's naming, untouched
