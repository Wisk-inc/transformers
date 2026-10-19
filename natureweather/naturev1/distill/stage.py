# Copyright 2026 Nathan. Apache-2.0.
"""
Staging aligned teacher forecasts on disk, once, so distillation trains at disk speed.

Each teacher-year becomes one memory-mapped array ``(inits, leads, lat, lon, channels)`` in NatureV1's
normalised channel space (float16, NaN where the teacher does not provide a channel), plus -- for ensemble
teachers -- the normalised ensemble variance, plus the list of initial times and the channels provided.
Resumable: a sidecar records the initial times already written, so an interrupted staging run continues.

Size planning (written into the sidecar): 7 leads x 29,040 points x ~85 channels x 2 bytes is ~35 MB per initial
time, ~25 GB per teacher-year at two initial times a day -- choose leads and teachers to fit the disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .align import to_channels
from .catalog import TEACHERS
from .sources import ArchiveTeacher

DEFAULT_LEADS = (6, 12, 18, 24, 36, 48, 72)


def wanted_variables(channels: list[str]) -> dict:
    """``{canonical base name: [levels] or None}`` covering NatureV1's channels."""
    out: dict = {}
    for name in channels:
        base, _, level = name.partition("@")
        if level:
            out.setdefault(base, [])
            if out[base] is not None:
                out[base].append(int(level))
        else:
            out.setdefault(base, None)
    return out


def stage_teacher(name: str, year: int, channels: list[str], normalizer, root: str | Path, leads=DEFAULT_LEADS,
                  inits=None, log=print, init_hours=None) -> Path:
    """Write ``root/<name>_<year>/{mean.npy, var.npy?, meta.json}``; returns the directory."""
    directory = Path(root) / f"{name}_{year}"
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / "meta.json"
    source = ArchiveTeacher(name, year)
    hours = TEACHERS[name].init_hours if init_hours is None else [h for h in init_hours if h in TEACHERS[name].init_hours]
    times = [t for t in source.times if int(str(t)[11:13]) in hours and int(str(t)[:4]) == year]
    if inits is not None:
        allowed = set(np.asarray(inits, "datetime64[s]").tolist())
        times = [t for t in times if t.tolist() in allowed]
    shape = (len(times), len(leads), 121, 240, len(channels))
    ensemble = TEACHERS[name].ensemble_dim is not None
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    if meta is None or meta.get("shape") != list(shape):
        mean = np.lib.format.open_memmap(directory / "mean.npy", mode="w+", dtype=np.float16, shape=shape)
        var = (np.lib.format.open_memmap(directory / "var.npy", mode="w+", dtype=np.float16, shape=shape)
               if ensemble else None)
        meta = {"teacher": name, "year": year, "shape": list(shape), "leads": list(leads), "channels": channels,
                "times": [str(t) for t in times], "done": [], "provided": [False] * len(channels),
                "gigabytes": float(np.prod(shape)) * 2 / 1e9 * (2 if ensemble else 1), "licence": TEACHERS[name].licence}
    else:
        mean = np.load(directory / "mean.npy", mmap_mode="r+")
        var = np.load(directory / "var.npy", mmap_mode="r+") if ensemble else None
    done = set(meta["done"])
    want = wanted_variables(channels)
    provided = np.asarray(meta["provided"], bool)
    std = np.asarray(normalizer.std, np.float64)
    for i, when in enumerate(times):
        if i in done:
            continue
        m, v = source.read(when, list(leads), want)
        if not m:
            continue
        values, got = to_channels(m, channels, normalizer)
        mean[i] = values.astype(np.float16)
        provided |= got
        if var is not None and v:
            spread, _ = to_channels(v, channels)
            var[i] = (spread / std ** 2).astype(np.float16)
        done.add(i)
        if i % 20 == 0 or i == len(times) - 1:
            meta["done"], meta["provided"] = sorted(done), provided.tolist()
            meta_path.write_text(json.dumps(meta))
            if log:
                log(f"[distill-stage] {name} {year}: {len(done)}/{len(times)} initial times")
    mean.flush()
    if var is not None:
        var.flush()
    meta["done"], meta["provided"] = sorted(done), provided.tolist()
    meta_path.write_text(json.dumps(meta))
    return directory


class StagedTeacher:
    """A staged teacher-year, read back by initial time."""

    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        self.meta = json.loads((directory / "meta.json").read_text())
        self.mean = np.load(directory / "mean.npy", mmap_mode="r")
        self.var = np.load(directory / "var.npy", mmap_mode="r") if (directory / "var.npy").exists() else None
        self.index = {np.datetime64(t, "s"): i for i, t in enumerate(self.meta["times"]) if i in set(self.meta["done"])}
        self.leads = list(self.meta["leads"])

    def get(self, when) -> tuple[np.ndarray, np.ndarray | None] | None:
        i = self.index.get(np.datetime64(when, "s"))
        if i is None:
            return None
        return np.asarray(self.mean[i], np.float32), (None if self.var is None else np.asarray(self.var[i], np.float32))


__all__ = ["DEFAULT_LEADS", "StagedTeacher", "stage_teacher", "wanted_variables"]
