# Copyright 2026 Nathan. Apache-2.0.
"""
Every member's track for a storm and a start time, and where they are kept.

A :class:`TrackSet` is one system's members for one case. The ensemble mean is a mean of positions on the sphere,
taken only where enough members still carry the storm: a mean over the few members that kept a storm the others
dissipated is biased towards wherever storms survive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fields import SYSTEMS, FieldArchive
from .geodesy import distance_km, mean_position
from .storms import Storm, first_guess
from .tracker import TrackerConfig, track_members


@dataclass
class TrackSet:
    storm: str
    init: int
    system: str
    steps: np.ndarray               # hours from the start, (S,)
    lat: np.ndarray                 # (members, S)
    lon: np.ndarray
    pressure: np.ndarray            # hPa

    @property
    def members(self) -> int:
        return self.lat.shape[0]

    def mean(self, min_share: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
        """The mean track ``(lat, lon)``; NaN at steps where fewer than ``min_share`` of the members have a fix."""
        lat, lon = mean_position(self.lat, self.lon, axis=0)
        have = np.isfinite(self.lat).mean(0)
        keep = have >= (min_share if self.members > 1 else 1.0)
        return np.where(keep, lat, np.nan), np.where(keep, lon, np.nan)

    def spread_km(self) -> np.ndarray:
        """Root-mean-square distance of the members from their mean, km (NaN with fewer than three fixes)."""
        lat, lon = mean_position(self.lat, self.lon, axis=0)
        d = distance_km(self.lat, self.lon, lat[None], lon[None])
        n = np.isfinite(d).sum(0)
        with np.errstate(invalid="ignore"):
            rms = np.sqrt(np.nansum(d ** 2, 0) / np.maximum(n, 1))
        return np.where(n >= 3, rms, np.nan)

    def points(self, min_share: float = 0.4) -> dict:
        """The mean track as ``{lead hours: (lat, lon)}``, the form the verification takes."""
        lat, lon = self.mean(min_share)
        return {int(s): (float(a), float(b)) for s, a, b in zip(self.steps, lat, lon) if np.isfinite(a)}


def track_case(archive: FieldArchive, storm: Storm, init: int, systems=SYSTEMS, cfg: TrackerConfig | None = None,
               carq: dict | None = None) -> dict:
    """Track one storm from one start time in every system that has fields: ``{system: TrackSet}``."""
    start = first_guess(storm, init, carq)
    out = {}
    if start is None:
        return out
    for system in systems:
        z = archive.load(system, init)
        if z is None:
            continue
        tr = track_members(z["msl"], z["lat"], z["lon"], z["steps"], start, cfg)
        out[system] = TrackSet(storm.id, int(init), system, np.asarray(z["steps"]), tr[..., 0], tr[..., 1], tr[..., 2])
    return out


class TrackStore:
    """Tracks on disk: one file per storm and start time, every system inside."""

    def __init__(self, root) -> None:
        self.root = Path(root).expanduser()

    def path(self, storm: str, init: int) -> Path:
        stamp = str(np.datetime64(int(init), "h")).replace("-", "").replace("T", "")[:10]
        return self.root / storm / f"{stamp}.npz"

    def save(self, tracks: dict) -> None:
        if not tracks:
            return
        any_set = next(iter(tracks.values()))
        p = self.path(any_set.storm, any_set.init)
        p.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for system, t in tracks.items():
            arrays[f"{system}_steps"] = t.steps
            arrays[f"{system}_track"] = np.stack([t.lat, t.lon, t.pressure], -1).astype(np.float32)
        tmp = p.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(p)

    def load(self, storm: str, init: int) -> dict | None:
        p = self.path(storm, init)
        if not p.exists():
            return None
        z = np.load(p)
        out = {}
        for key in z.files:
            if key.endswith("_track"):
                system = key[:-6]
                tr = z[key].astype(np.float64)
                out[system] = TrackSet(storm, int(init), system, z[f"{system}_steps"], tr[..., 0], tr[..., 1], tr[..., 2])
        return out

    def get(self, archive: FieldArchive, storm: Storm, init: int, systems=SYSTEMS, cfg: TrackerConfig | None = None,
            carq: dict | None = None) -> dict:
        """Tracks from the store, tracked and saved first when they are not there."""
        got = self.load(storm.id, init)
        if got is None:
            got = track_case(archive, storm, init, systems, cfg, carq)
            self.save(got)
        return got


__all__ = ["TrackSet", "TrackStore", "track_case"]
