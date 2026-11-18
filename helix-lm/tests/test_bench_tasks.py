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
