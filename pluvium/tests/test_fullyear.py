# Copyright 2026 Nathan. Apache-2.0.
"""The full-year benchmark's scores, on synthetic numbers with known answers."""

import numpy as np

from pluvium.benchmarks.fullyear import acc_per_run, best_cut, crps_quantiles, csi, daily_members, sharpness
from pluvium.verify import crps_samples


def test_daily_members_from_cumulative():
    cum = np.array([[[[1.0]], [[4.0]], [[4.0]]]])                    # (members, steps, H, W)
    z = {"values": cum}
    assert daily_members(z, 1)[0, 0, 0] == 1.0 and daily_members(z, 2)[0, 0, 0] == 3.0
    assert daily_members(z, 3)[0, 0, 0] == 0.0 and daily_members(z, 4) is None


def test_csi_and_best_cut():
    y = np.array([True, True, False, False, True])
    p = np.array([0.9, 0.6, 0.4, 0.1, 0.3])
    assert csi(p, y, 0.5) == 2 / 3
    assert best_cut(p, y) <= 0.3                                      # catching the third event pays


def test_acc_and_sharpness():
    rng = np.random.default_rng(0)
    ob = rng.gamma(1.0, 5.0, 400)
    clim = np.full(400, 5.0)
    blocks = np.repeat([1, 2], 200)
    assert acc_per_run(ob, ob, clim, blocks) > 0.999
    assert acc_per_run(rng.permutation(ob), ob, clim, blocks) < 0.2
    assert abs(sharpness(ob, ob, blocks) - 1.0) < 1e-9
    assert sharpness(0.5 * ob + 0.5 * ob.mean(), ob, blocks) < 0.6         # a blurred field is less sharp
    dry = np.concatenate([ob, np.r_[np.zeros(199), 0.1]])                   # a nearly flat (dry) day
    fc = np.concatenate([ob, rng.gamma(1.0, 1.0, 200)])
    assert sharpness(fc, dry, np.repeat([1, 2, 3], 200)) < 1.5              # a dry day cannot dominate


def test_crps_of_quantiles_matches_samples():
    rng = np.random.default_rng(1)
    draws = rng.gamma(0.8, 4.0, (2000, 50))
    levels = (np.arange(100) + 0.5) / 100
    q = np.quantile(draws, levels, axis=0).T                           # (50, 100)
    y = rng.gamma(0.8, 4.0, 50)
    np.testing.assert_allclose(crps_quantiles(q, y), crps_samples(draws, y), rtol=0.03, atol=0.02)


def test_the_refit_loop_scores_every_model_and_fills_the_scoreboard(monkeypatch):
    from functools import partial

    from pluvium import blend as B
    from pluvium.benchmarks import fullyear as F
    from pluvium.sources.archive import run_set

    monkeypatch.setattr(B, "fit", partial(B.fit, min_steps=40))                  # a quick learner for the test
    rng = np.random.default_rng(3)
    n = 40
    runs, long = run_set("2024-04-01T00", "2024-10-01T00", every=2, long_every=4)
    table = {"run": [], "lead": [], "y": [], "clim": [], "feats": {"ens": [], "gfs": []}}
    clim = {"clim_p1": np.full(n, 0.4, np.float32), "clim_p10": np.full(n, 0.1, np.float32),
            "clim_p25": np.full(n, 0.03, np.float32), "clim_p50": np.full(n, 0.01, np.float32),
            "clim_mean": np.full(n, 1.0, np.float32)}
    for run in runs:
        for lead in (range(1, 15) if run in set(long) else range(1, 4)):
            signal = rng.gamma(0.5, 6.0, n).astype(np.float32)
            noise = 0.2 + 0.1 * lead
            table["run"].append(int(run))
            table["lead"].append(lead)
            table["y"].append(np.where(rng.random(n) < 0.6, signal * rng.lognormal(0, 0.5, n), 0).astype(np.float32))
            table["clim"].append(clim)
            table["feats"]["ens"].append(B.system_features(signal[None] * rng.lognormal(0, noise, (10, n))))
            table["feats"]["gfs"].append(B.system_features(signal[None] * rng.lognormal(0, 3 * noise, (1, n))))
    table["run"], table["lead"] = np.asarray(table["run"]), np.asarray(table["lead"])
    cfg = F.FullYearConfig(archive="", truth="", out="", train_start="2024-04-01T00", test_start="2024-07-01T00",
                           test_end="2024-10-01T00", members=2, lead_ranges=((1, 3), (4, 14)), max_rows=20_000)
    cells = F.CellSet(np.zeros(n), np.zeros(n), None, "test")
    res = F.evaluate_cells(cells, table, cfg, ["ens", "gfs"])
    short, far = res["groups"]["days 1-3"], res["groups"]["days 11-14"]
    assert short["blend_ops"]["1"]["bss"] > short["gfs"]["1"]["bss"]             # the blend uses the better system
    assert np.isfinite(far["blend_ops"]["crps"]) and far["ens"]["n"] > 0          # long leads have their own nets
    board = F.scoreboard({"test": res})
    assert len(board) == 2 + len(F.BOARD) + 2 and all(" of " in row or "-" in row for row in board[2:])
    assert F.melissa(res) == []
