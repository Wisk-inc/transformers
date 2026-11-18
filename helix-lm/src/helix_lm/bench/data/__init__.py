# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The sets the TriStream benchmarks are measured on: what each one is, its licence, how it is split, and items
(audio, its singer, notes when written ones exist) in one shape for every task."""

from .registry import SETS, BenchmarkSet, Item, get, usable_for

__all__ = ["SETS", "BenchmarkSet", "Item", "get", "usable_for"]
