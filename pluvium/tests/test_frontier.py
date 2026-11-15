# Copyright 2026 Nathan. Apache-2.0.
"""The frontier evaluation end to end on a small synthetic store, and the pieces it is built from."""

import json

import numpy as np

from pluvium.blend import THRESHOLDS, to_logits
from pluvium.frontier.evaluate import PoolDesign, evaluate, uniform_like
from pluvium.frontier.store import FrontierConfig, Store, refits
from pluvium.pool.amount import AmountHead, implied_amount
from pluvium.util import hour_of

K = len(THRESHOLDS)


def chances(signal, noise, rng):
    """A component: chances of every threshold from a noisy view of what will fall."""
    f = signal * rng.lognormal(0, noise, signal.shape)
    return np.clip(1 / (1 + np.exp(-(np.log1p(f)[..., None] - np.log1p(THRESHOLDS)) * 2.5)), 1e-4, 1 - 1e-4)


def make_store(folder, n=60, blocks=36, rows=6000, seed=0):
    """A store with three components (sharp, fair, poor), one quarter, in the format build_store writes."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True)
    names = ["sharp", "fair", "poor"]
    run = hour_of("2025-04-02T00") + 48 * (np.arange(blocks) // 3)
    lead = 1 + np.arange(blocks) % 3
    signal = rng.gamma(0.5, 8.0, (blocks, n))
    y = (signal * rng.lognormal(0, 0.2, (blocks, n))).astype(np.float32)
    y[0, :3] = np.nan
    raw_mean = (signal * rng.lognormal(0, 0.5, (blocks, n))).astype(np.float32)
    np.savez(folder / "meta.npz", run=run, lead=lead, y=y, raw_mean=raw_mean, raw_n=np.full(blocks, 6, np.int16),
             lat=np.linspace(10, 14, n), lon=np.linspace(-80, -70, n), names=np.array(names), systems=np.array(names))
    clim = {}
    for m in range(1, 13):
        for t, p in ((1, 0.4), (10, 0.15), (25, 0.05), (50, 0.015)):
            clim[f"clim_p{t}__{m:02d}"] = np.full(n, p, np.float32)
        clim[f"clim_mean__{m:02d}"] = np.full(n, np.log1p(4.0), np.float32)
    np.savez(folder / "clim.npz", **clim)
    (folder / "raw.json").write_text(json.dumps({"leads": {}, "groups": {}, "seasons": {}, "cases": []}))
    (folder / "config.json").write_text(json.dumps({"case_days": []}))
    np.savez_compressed(folder / "raw_test.npz", blocks=np.arange(blocks),
                        raw_p=chances(raw_mean, 0.01, rng).astype(np.float16))
    noises = (0.3, 0.8, 2.0)
    test_z = np.stack([to_logits(chances(signal, s, rng)) for s in noises], 2).astype(np.float16)   # (B, n, C, K)
    present = np.ones((blocks, 3), bool)
    present[::5, 2] = False                                               # the poor component misses some runs
    s_oof = rng.gamma(0.5, 8.0, rows)
    y_oof = (s_oof * rng.lognormal(0, 0.2, rows)).astype(np.float32)
    oof_z = np.stack([to_logits(chances(s_oof, s, rng)) for s in noises], 1).astype(np.float16)     # (R, C, K)
    oof_block = np.sort(rng.integers(0, 150, rows)).astype(np.int32)
    oof_lead = (1 + oof_block % 3).astype(np.int16)
    oof_valid = hour_of("2024-04-02T00") + 48 * (oof_block // 3).astype(np.int64) + 24 * (oof_lead - 1)
    cuts = {name: {f"{t:g}": 0.4 for t in (1.0, 10.0, 25.0, 50.0)} for name in names}
    np.savez(folder / "q0.npz", oof_z=oof_z, oof_present=np.ones((rows, 3), bool), oof_w=np.ones(rows),
             oof_raw_p=chances(s_oof, 0.5, rng).astype(np.float16), oof_raw_mean=s_oof.astype(np.float32),
             test=np.arange(blocks), test_z=test_z, test_present=present, cuts=json.dumps(cuts), oof_y=y_oof,
             oof_lead=oof_lead, oof_valid=oof_valid, oof_block=oof_block, oof_cell=rng.integers(0, n, rows).astype(np.int32))
    (folder / "done").write_text("ok")
    return Store(folder)


def test_store_reads_back_and_refit_dates_are_quarters(tmp_path):
    store = make_store(tmp_path / "place")
    assert store.n == 60 and store.names == ["sharp", "fair", "poor"] and store.quarters() == [0]
    d = store.quarter(0)
    assert d["oof_z"].shape == (6000, 3, K) and d["cuts"]["sharp"]["10"] == 0.4
    assert store.raw_p(5).shape == (60, K) and len(store.table()["clim"]) == 36
    cfg = FrontierConfig(archive="a", truth="t", out="o")
    dates = refits(cfg)
    assert len(dates) == 5 and dates[0] == hour_of("2025-04-01T00") and dates[1] == hour_of("2025-07-01T00")
    assert dates[-1] == hour_of("2026-04-01T00")
    picks = uniform_like(np.array([1.0, 1.0, 98.0]), 500, np.random.default_rng(0))
    assert np.mean(picks == 2) > 0.9                                      # heavy weights are drawn that much more


def test_evaluate_scores_designs_and_the_pool_beats_its_components(tmp_path):
    store = make_store(tmp_path / "place")
    designs = [PoolDesign(name="stack", bands=(0, 5, 11), smooth=0.0, online=True, eta=1.0, record=True, rows=4000, iters=40),
               PoolDesign(name="pool", bands=(0, 5, 11), smooth=0.0, rows=4000, iters=40)]
    keep = {}
    res = evaluate(store, designs, keep=keep)
    row = res["groups"]["days 1-3"]
    assert set(row) == {"sharp", "fair", "poor", "stack", "pool"}
    for t in ("1", "10", "25"):
        assert row["pool"][t]["bss"] >= row["sharp"][t]["bss"] - 0.01     # never clearly worse than the best one
        assert row["pool"][t]["bss"] > row["fair"][t]["bss"] > row["poor"][t]["bss"]
    assert row["stack"]["crps"] < row["fair"]["crps"] and 0 < row["stack"]["1"]["csi"] <= 1
    assert len(keep) == 36 and keep[3].shape == (60, K) and keep[3].dtype == np.float16
    weights = res["online_weights"]["stack"]
    assert weights["experts"] == ["sharp", "fair", "poor", "pool"]
    w = np.array(weights["by_quarter_days_1_3_light"][0])
    assert abs(w.sum() - 1.0) < 0.01 and w[2] < w[0]                      # the poor component has lost weight
    versus = res["versus"]["days 1-3"]["1"]
    assert versus["poor"]["delta_bs"] < 0 and versus["poor"]["ci95"][1] < 0   # better than it with confidence
    assert res["region"] == "place" and res["leads"]["2"]["stack"]["n"] > 0
    assert res["audits"] == {}                                            # no design asked for one


def test_the_staged_verdict_makes_the_design_the_declared_one_or_the_one_without_handover(tmp_path):
    from pluvium.frontier import report as R

    store = make_store(tmp_path / "place")
    common = dict(bands=(0, 5, 11), smooth=0.0, rows=4000, iters=40)
    designs = [PoolDesign(name="stack", record=True, staged=True, record_quantile=0.99, **common),
               PoolDesign(name="declared", record=True, record_quantile=0.99, **common), PoolDesign(name="pool", **common)]
    for keep in (True, False):                                            # what the place's staged storms said
        verdict = {"keep": keep, "evidence": True, "gain": 0.04 if keep else -0.02, "storms": ["2024-10-08"]}
        res = evaluate(store, designs, components=False, verdict=verdict)
        assert res["staged"] == verdict
        (check,) = res["audits"]["stack"]                                 # the yearly check: one refit, reported only
        assert check["rows"] > 10 and check["days"] > 3 and check["edge_mm"] >= 20.0
        row = res["groups"]["days 1-3"]
        twin, other = ("declared", "pool") if keep else ("pool", "declared")   # one or the other, exactly, all year
        assert row["stack"]["50"]["bss"] == row[twin]["50"]["bss"] and row["stack"]["crps"] == row[twin]["crps"]
        assert row["stack"]["50"]["bss"] != row[other]["50"]["bss"]       # and the hand-over does change the forecast
        assert R.kept(res) is keep and R.kept(res, "yearly") == check["keep"]
    free = evaluate(store, designs, components=False)                     # no verdict given: the hand-over stays on
    assert free["staged"] is None and free["groups"]["days 1-3"]["stack"]["crps"] == free["groups"]["days 1-3"]["declared"]["crps"]
    assert "declared" in R.ABLATIONS


def test_staged_verdict_is_read_from_the_file_beside_the_store(tmp_path):
    from pluvium.benchmarks import frontier_bench as FB

    folder = tmp_path / "store" / "place"
    folder.mkdir(parents=True)
    scores = lambda b: {"brier": {"50": b}, "amount_mm": 10.0, "rows_handed_over": 0.05}   # noqa: E731
    experiment = {"storms": ["2024-10-08"], "storm": {"observed_mean_mm": 20.0, "calibrated": scores(0.08), "raw": scores(0.07),
                                                       "q0.999 w0.69": scores(0.075)},
                  "control": {"q0.999 w0.69": {"rows_handed_over": 0.0001}}}
    (folder / "staged.json").write_text(json.dumps(experiment))
    cfg = FrontierConfig(archive="a", truth="t", out=str(tmp_path))
    v = FB.staged_verdict(cfg, "place")                                    # nothing is downloaded: the experiment is there
    assert v["keep"] and abs(v["gain"] - (1 - 0.075 / 0.08)) < 1e-9 and v["setting"] == "q0.999 w0.69"
    assert FB.POOL_RECORD == (FB.DESIGNS[0].record_quantile, FB.DESIGNS[0].record_width)


def test_frontier_bench_writes_results_and_resumes(tmp_path):
    from pluvium.benchmarks import frontier_bench as FB

    store = make_store(tmp_path / "store" / "place")
    designs = [PoolDesign(name="stack", online=True, eta=1.0, record=True, rows=4000, iters=30, **FB.POOL)]
    out = tmp_path / "results"
    res = FB.evaluate_place(store.folder, out, designs=designs)
    assert sorted(p.name for p in out.iterdir()) == ["place.json", "place_keep.npz", "place_tail.json"]
    assert res["groups"]["days 1-3"]["stack"]["1"]["bss"] > 0.2
    tail = json.loads((out / "place_tail.json").read_text())
    assert tail["groups"]["days 1-3"]["rows"] > 0 and "alerts" in tail["groups"]["days 1-3"]
    stamp = (out / "place.json").stat().st_mtime_ns
    again = FB.evaluate_place(store.folder, out, designs=designs)              # everything is there: nothing is redone
    assert (out / "place.json").stat().st_mtime_ns == stamp and again["cells"] == res["cells"]
    top = FB.DESIGNS[0]
    assert top.name == "stack" and top.online and top.record and top.staged and top.bands == (0, 5, 11) and top.eta == 1.0
    assert [d.name for d in FB.DESIGNS[1:]] == ["declared", "pool", "pool_online", "pool_record"]
    declared = FB.DESIGNS[1]                                                   # the same, with the hand-over always on
    assert declared.record and not declared.staged and (declared.online, declared.eta, declared.bands) == (True, 1.0, (0, 5, 11))


def test_a_half_written_quarter_is_not_taken_for_a_quarter(tmp_path):
    folder = tmp_path / "place"
    make_store(folder)
    (folder / "q1.part.npz").write_bytes(b"half a quarter")                    # what a killed build leaves behind
    assert Store(folder).quarters() == [0]


def test_amount_head_is_a_blend_and_falls_back_to_the_pool():
    rng = np.random.default_rng(1)
    n = 4000
    truth = rng.gamma(0.5, 8.0, n)
    pool = truth * rng.lognormal(0, 0.5, n) * 0.7                          # the pool's amount runs 30% low
    comps = np.stack([truth * rng.lognormal(0, 0.8, n), truth * rng.lognormal(0, 1.5, n)], 1)
    present = np.ones((n, 2), bool)
    present[:500, 1] = False
    clim = np.full(n, truth.mean())
    lead = rng.integers(1, 15, n)
    head = AmountHead(2).fit(pool, comps, present, clim, lead, truth)
    assert np.all(head.coef >= 0) and head.coef[0, 0] > 0.5               # no forecast is subtracted
    out = head.predict(pool, comps, present, clim, lead)
    rmse = lambda a: float(np.sqrt(np.mean((a - truth) ** 2)))             # noqa: E731
    assert rmse(out) < rmse(pool) and abs(out.mean() - truth.mean()) < abs(pool.mean() - truth.mean())
    fresh = AmountHead(2)
    np.testing.assert_allclose(fresh.predict(pool, comps, present, clim, lead), pool, rtol=1e-6)   # unfitted: the pool
    with_raw = AmountHead(2, n_extra=1).fit(pool, comps, present, clim, lead, truth, extra=truth * 1.02)
    assert with_raw.coef[:, -1].min() > 0.5                                # an extra column that is nearly the truth
    table = with_raw.table(["a", "b"], ("raw",))
    assert list(table["days 1-3"]) == ["pool", "a", "b", "climatology", "raw"]
    p = chances(truth[:300], 0.01, rng)
    amounts = implied_amount(p, chunk=100)
    assert amounts.shape == (300,) and np.corrcoef(amounts, truth[:300])[0, 1] > 0.95
    assert implied_amount(np.stack([p, p], 1)).shape == (300, 2)
