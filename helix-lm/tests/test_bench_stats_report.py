# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Confidence intervals, paired tests, the regression check, the leaderboard and the Hub upload."""

import math
from types import SimpleNamespace

from helix_lm.bench.hub import push_results
from helix_lm.bench.leaderboard import rank
from helix_lm.bench.regression import compare, describe
from helix_lm.bench.stats import effect_size, mean_ci, paired, sign_test, wilcoxon


def test_the_interval_holds_the_mean_and_narrows_with_more_items():
    few = mean_ci([1.0, 2.0, 3.0, 4.0])
    many = mean_ci([1.0, 2.0, 3.0, 4.0] * 25)
    assert few[1] <= few[0] <= few[2] and (many[2] - many[1]) < (few[2] - few[1])
    assert all(math.isnan(x) for x in mean_ci([]))


def test_paired_tests_see_a_steady_gain_and_not_noise():
    gain = [0.5 + 0.01 * k for k in range(20)]
    assert sign_test(gain) < 0.01 and wilcoxon(gain) < 0.01 and effect_size(gain) > 1
    noise = [(-1) ** k * 0.1 for k in range(20)]
    assert wilcoxon(noise) > 0.2
    assert paired({"a": 1.0, "b": 2.0}, {"a": 0.5, "c": 9.0}) == [0.5]


def _result(mcd_values):
    rows = [{"system": "tristream", "task": "resynthesis", "item": f"i{k}", "mcd_db": v}
            for k, v in enumerate(mcd_values)]
    mean = sum(mcd_values) / len(mcd_values)
    return {"suite": "quick", "rows": rows,
            "summary": {"tristream": {"resynthesis": {"mcd_db": [mean, mean, mean, len(rows)]}}}}


def test_a_worse_checkpoint_is_reported_and_a_better_one_is_not():
    old = _result([5.0 + 0.01 * k for k in range(12)])
    worse = _result([6.0 + 0.01 * k for k in range(12)])
    better = _result([4.0 + 0.01 * k for k in range(12)])
    found = compare(worse, old)
    assert found and found[0]["metric"] == "mcd_db" and found[0]["significant"]
    assert compare(better, old) == [] and "no metric got worse" in describe([])


def test_ranking_puts_the_lower_mcd_first():
    rows = [{"system": s, "task": "resynthesis", "item": f"i{k}", "mcd_db": base + 0.01 * k}
            for s, base in (("a", 3.0), ("b", 5.0)) for k in range(10)]
    result = {"rows": rows, "summary": {s: {"resynthesis": {"mcd_db": [m, m, m, 10]}} for s, m in
                                        (("a", 3.05), ("b", 5.05))}}
    order = rank(result, "resynthesis", "mcd_db")
    assert order[0][0] == "a"


def test_results_go_to_the_hub_in_one_commit_and_a_failure_does_not_raise(tmp_path):
    (tmp_path / "results.json").write_text('{"suite": "quick"}')
    (tmp_path / "report.md").write_text("# report")
    commits = []
    api = SimpleNamespace(create_commit=lambda **kw: commits.append(kw))
    assert push_results(tmp_path, "org/repo", None, api=api, log=lambda *_: None)
    paths = [op.path_in_repo for op in commits[0]["operations"]]
    assert "bench/latest.json" in paths and "bench/LATEST.md" in paths

    def broken(**kw):
        raise OSError("no network")

    said = []
    assert not push_results(tmp_path, "org/repo", None, api=SimpleNamespace(create_commit=broken), log=said.append)
    assert "not uploaded" in said[0]
