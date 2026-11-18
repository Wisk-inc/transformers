# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The tasks and the suite runner on the baselines: rows per item, a summary with confidence intervals, a report,
and a second run that takes its outputs from the cache."""

import json

import pytest

from helix_lm.bench.config import SUITES
from helix_lm.bench.runner import run_suite
from helix_lm.bench.tasks import TASK_NAMES, runner


def test_every_task_has_a_runner_and_every_suite_names_known_tasks():
    for name in TASK_NAMES:
        assert callable(runner(name))
    for suite in SUITES.values():
        assert set(suite.tasks) <= set(TASK_NAMES)
    with pytest.raises(KeyError):
        runner("no-such-task")


def test_the_quick_suite_runs_on_the_baselines_and_uses_its_cache(tmp_path):
    seen = []
    result = run_suite("quick", ("identity", "harmonic-noise"), tmp_path, log=seen.append)
    summary = result["summary"]
    assert {"identity", "harmonic-noise"} <= set(summary)
    mean, lo, hi, n = summary["identity"]["resynthesis"]["mcd_db"]
    assert mean < 1e-3 and lo <= mean <= hi and n > 0
    assert (tmp_path / "report.md").read_text().startswith("#")
    assert json.loads((tmp_path / "results.json").read_text())["suite"] == "quick"
    again = run_suite("quick", ("identity", "harmonic-noise"), tmp_path, log=seen.append)
    assert again["cache"]["made"] == 0 and again["cache"]["hits"] > 0


def test_tristream_without_a_trained_model_is_skipped_not_faked(tmp_path):
    result = run_suite("quick", ("tristream", "identity"), tmp_path, root=tmp_path / "empty", log=lambda *_: None)
    assert "tristream" not in result["summary"] and result["skipped"]


def test_items_are_spread_over_the_singers():
    from pathlib import Path

    from helix_lm.bench.data.registry import Item
    from helix_lm.bench.runner import balanced

    items = [Item(f"{s}_{k:03d}", Path(f"{s}/{k}.wav"), s, "vocalset") for s in ("f2", "f8", "m3") for k in range(50)]
    chosen = balanced(items, 9)
    assert sorted(i.singer for i in chosen) == ["f2"] * 3 + ["f8"] * 3 + ["m3"] * 3
    assert [i.id for i in balanced(items, 9)] == [i.id for i in chosen]          # seeded: the same every run


def test_a_failing_task_is_reported_and_the_others_still_run(tmp_path, monkeypatch):
    from helix_lm.bench import runner as R

    real = R.runner

    def broken(name):
        if name == "svs":
            def fail(*args, **kwargs):
                raise ValueError("broken task")
            return fail
        return real(name)

    monkeypatch.setattr(R, "runner", broken)
    result = R.run_suite("quick", ("harmonic-noise",), tmp_path, log=lambda *_: None)
    assert result["failed"] and "broken task" in result["failed"][0]
    assert "resynthesis" in result["summary"]["harmonic-noise"] and "svs" not in result["summary"]["harmonic-noise"]


def test_the_register_task_runs_and_a_clean_voice_has_no_pops(tmp_path):
    from helix_lm.bench.cache import OutputCache
    from helix_lm.bench.data.synthetic import make_set
    from helix_lm.bench.systems import make
    from helix_lm.bench.tasks import runner

    items = make_set(tmp_path / "set", takes_per_singer=3, seconds=6.0)
    rows = runner("register")(make("identity"), items, 24000, OutputCache(None))
    assert rows and all(isinstance(r["crossings"], int) for r in rows)
    assert sum(r["register_pops_per_minute"] for r in rows) == 0.0   # the identity output is the clean take
