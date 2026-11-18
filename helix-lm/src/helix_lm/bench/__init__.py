# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
TriStream benchmarks: the objective metrics singing synthesis and conversion are measured with, the sets they are
measured on, the tasks, the baselines to compare against, and the statistics that say whether a difference is
real.

    from helix_lm.bench import run_suite
    result = run_suite("quick", systems=("identity", "harmonic-noise", "tristream"), root="/workspace/tristream")

The numbers in ``docs/bench/RESULTS.md`` were measured on the synthetic set and the baselines, on a CPU. Nothing
here claims a score for a trained TriStream before one has been measured.
"""

from .config import SUITES, Suite, suite
from .runner import run_suite

__all__ = ["SUITES", "Suite", "run_suite", "suite"]
