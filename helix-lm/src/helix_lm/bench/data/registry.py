# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The benchmark sets, with the facts a result has to be read with. ``commercial`` says whether a model may be trained
on the set for a product; the research sets here are for measuring only, never for training a released model, in
line with the molab program's ``--include-research`` rule. Check each corpus's own terms before use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkSet:
    name: str
    what: str
    licence: str
    commercial: bool
    language: str
    url: str
    citation: str
    tasks: tuple = ("svc",)


@dataclass
class Item:
    """One benchmark item: a take, who sang it and, when the set has them, its notes ``(onset_s, offset_s, midi)``,
    lyrics and a paired take (the spoken version of the same lyrics, another singer's take)."""

    id: str
    path: Path
    singer: str
    set: str
    notes: list = field(default_factory=list)
    lyrics: str = ""
    technique: str = ""
    pair: Path | None = None


SETS = {
    "synthetic": BenchmarkSet(
        "synthetic", "voices built from a pitch curve, a vowel's formants and breath noise, with every answer known",
        "made here (Apache-2.0)", True, "none", "", "", ("svs", "svc", "resynthesis", "pitch_control")),
    "vocalset": BenchmarkSet(
        "vocalset", "20 professional singers, 17 techniques, about 10 hours", "CC BY 4.0", True, "vowels",
        "https://zenodo.org/records/1193957",
        "Wilkins, Seetharaman, Wahl, Pardo. VocalSet: A Singing Voice Dataset. ISMIR 2018.",
        ("svc", "resynthesis", "technique", "register")),
    "nus48e": BenchmarkSet(
        "nus48e", "12 singers, each song sung and its lyrics read", "research only (licence agreement)", False,
        "en", "https://smcnus.comp.nus.edu.sg/nus-48e-sung-and-spoken-lyrics-corpus/",
        "Duan, Fang, Li, Sim, Wang. The NUS sung and spoken lyrics corpus. APSIPA 2013.", ("svc", "resynthesis")),
    "opencpop": BenchmarkSet(
        "opencpop", "one singer, 100 Mandarin pop songs with notes and phoneme timing", "research only (CC BY-NC-ND)",
        False, "zh", "https://wenet.org.cn/opencpop/",
        "Wang et al. Opencpop: A High-Quality Open Source Chinese Popular Song Corpus. Interspeech 2022.",
        ("svs", "resynthesis")),
    "m4singer": BenchmarkSet(
        "m4singer", "20 singers in four voice types, Mandarin, with notes", "research only (CC BY-NC-SA)", False,
        "zh", "https://github.com/M4Singer/M4Singer",
        "Zhang et al. M4Singer: a Multi-Style, Multi-Singer and Musical Score Provided Mandarin Singing Corpus. "
        "NeurIPS 2022 Datasets and Benchmarks.", ("svs", "svc")),
}


def get(name: str) -> BenchmarkSet:
    if name not in SETS:
        raise KeyError(f"no benchmark set {name!r}; there are {', '.join(SETS)}")
    return SETS[name]


def usable_for(task: str, commercial_only: bool = False) -> list[str]:
    """The sets that have what ``task`` needs, the research-only ones left out with ``commercial_only``."""
    return [n for n, s in SETS.items() if task in s.tasks and (s.commercial or not commercial_only)]


__all__ = ["SETS", "BenchmarkSet", "Item", "get", "usable_for"]
