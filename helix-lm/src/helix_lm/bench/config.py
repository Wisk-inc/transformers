# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The suites: which sets and tasks a run covers. ``quick`` runs on a CPU in minutes on the synthetic set alone;
``standard`` adds VocalSet's test singers and the harder tasks; ``full`` adds long takes, speed and the research
sets that are present on disk (measured only, never trained on)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Suite:
    name: str
    sets: tuple
    tasks: tuple
    takes_per_singer: int = 2          # synthetic takes per synthetic singer
    max_items: int = 12                # items per set and task
    seconds: float = 6.0               # each take is cut to this


SUITES = {
    "quick": Suite("quick", ("synthetic",), ("resynthesis", "svc", "svs", "pitch_control"), 2, 6, 4.0),
    "standard": Suite("standard", ("synthetic", "vocalset"),
                      ("resynthesis", "svc", "svs", "pitch_control", "technique", "robustness", "register"),
                      4, 24, 6.0),
    "full": Suite("full", ("synthetic", "vocalset", "nus48e", "opencpop", "m4singer"),
                  ("resynthesis", "svc", "svs", "pitch_control", "technique", "robustness", "register", "longform",
                   "speed"), 6, 60, 8.0),
}


def suite(name_or_suite) -> Suite:
    if isinstance(name_or_suite, Suite):
        return name_or_suite
    if name_or_suite not in SUITES:
        raise KeyError(f"no suite {name_or_suite!r}; there are {', '.join(SUITES)}")
    return SUITES[name_or_suite]


__all__ = ["SUITES", "Suite", "suite"]
