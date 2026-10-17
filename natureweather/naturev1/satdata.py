# Copyright 2026 Nathan. Apache-2.0.
"""
Training samples from the staged satellite record (see :mod:`naturev1.satstage`).

Two datasets, one per network:

* :class:`StormSequenceDataset` -- four storm-centred frames (now, -3 h, -6 h, -9 h) of 11-micron and
  6.7-micron imagery, cut around a *first guess* of the centre that is deliberately wrong by up to
  ``jitter_km``: the network has to find the storm, not be handed it. All four frames are cut around the
  same first guess, so the storm moves across them the way it moves across a forecaster's loop. Labels
  are everything IBTrACS knows at that fix (:func:`naturev1.ibtracs.storm_label_table`).
* :class:`RetrievalDataset` -- three global frames (now, -3 h, -6 h) and an ERA5 state that is ``stale``
  hours old, with the true state now as the target. This is the honest version of "get the atmosphere
  from satellites": geostationary imagery sees cloud tops and upper-tropospheric moisture, not the wind at
  500 hPa, so it is asked to bring an aged state up to date rather than to conjure one from nothing.

Both read memory-mapped files and keep no handles across pickling, so they work with any number of
DataLoader workers on any number of GPUs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .ibtracs import CHANGE_HOURS, KM_PER_DEGREE
from .satstage import GLOBAL_CHANNELS, STAGED_PATCH_PIXELS
from .stormsat import PATCH_DEGREES

#: Storm network input: frames at these hours before now.
STORM_FRAME_HOURS = (0, 3, 6, 9)
#: Retrieval input: global frames at these hours before now.
GLOBAL_FRAME_HOURS = (0, 3, 6)
#: How stale the background state may be, hours.
STALE_HOURS = (6, 12, 24, 48)


class _Memmaps:
    """Lazily opened memory maps, dropped on pickling so each worker process opens its own."""

    def __init__(self) -> None:
        self._open: dict[str, np.ndarray] = {}

    def get(self, path: Path) -> np.ndarray:
        key = str(path)
        if key not in self._open:
            self._open[key] = np.load(path, mmap_mode="r")
        return self._open[key]

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self._open = {}


class GlobalFrames:
    """The staged global frames, looked up by valid time (unix seconds)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / "global"
        self.index: dict[int, tuple[str, int]] = {}
        self.shape = None
        for meta_path in sorted(self.root.glob("*.json")):
            meta = json.loads(meta_path.read_text())
            self.shape = tuple(meta["shape"])
            for row, (stamp, ok) in enumerate(zip(meta["times"], meta["available"])):
                if ok:
                    self.index[int(stamp)] = (meta["month"], row)
        self.maps = _Memmaps()

    def __len__(self) -> int:
        return len(self.index)

    def times(self) -> np.ndarray:
        return np.array(sorted(self.index), dtype=np.int64)

    def frame(self, stamp: int) -> np.ndarray | None:
        found = self.index.get(int(stamp))
        if found is None:
            return None
        tag, row = found
        return np.asarray(self.maps.get(self.root / f"{tag}.npy")[row])


class StormPatches:
    """The staged storm patches, looked up by row of the label table."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / "storms"
        self.index: dict[int, tuple[str, int]] = {}
        for meta_path in sorted(self.root.glob("*.json")):
            meta = json.loads(meta_path.read_text())
            tag = meta_path.stem
            for slot, (row, ok) in enumerate(zip(meta["rows"], meta["ok"])):
                if ok:
                    self.index[int(row)] = (tag, slot)
        self.maps = _Memmaps()

    def __contains__(self, row: int) -> bool:
        return int(row) in self.index

    def patch(self, row: int) -> np.ndarray | None:
        found = self.index.get(int(row))
        if found is None:
            return None
        tag, slot = found
        return np.asarray(self.maps.get(self.root / f"{tag}.npy")[slot])


def recentre(patch: np.ndarray, patch_centre: tuple[float, float], centre: tuple[float, float],
             pixels: int, step: float = PATCH_DEGREES) -> tuple[np.ndarray, float]:
    """
    Cut ``pixels`` square out of a staged ``(C, S, S)`` patch so it is centred on ``centre`` instead.

    Staged patches are regular in latitude and longitude, south to north, west to east, centred on the
    storm's own position at that time. Returns the crop (zeros -- "missing" -- where it runs off the staged
    square) and the share of the crop that was inside it.
    """
    size = patch.shape[-1]
    row_shift = (centre[0] - patch_centre[0]) / step
    col_shift = ((centre[1] - patch_centre[1] + 180.0) % 360.0 - 180.0) / step
    top = int(round((size - pixels) / 2 + row_shift))
    left = int(round((size - pixels) / 2 + col_shift))
    out = np.zeros((patch.shape[0], pixels, pixels), patch.dtype)
    r0, r1 = max(top, 0), min(top + pixels, size)
    c0, c1 = max(left, 0), min(left + pixels, size)
    if r1 > r0 and c1 > c0:
        out[:, r0 - top : r1 - top, c0 - left : c1 - left] = patch[:, r0:r1, c0:c1]
    inside = max(r1 - r0, 0) * max(c1 - c0, 0) / float(pixels * pixels)
    return out, inside


def _offset_degrees(latitude: float, north_km: float, east_km: float) -> tuple[float, float]:
    return north_km / KM_PER_DEGREE, east_km / (KM_PER_DEGREE * max(math.cos(math.radians(latitude)), 0.05))


class StormSequenceDataset(torch.utils.data.Dataset):
    """
    Storm-centred image sequences with every label the storm network learns from.

    Args:
        table: from :func:`naturev1.ibtracs.storm_label_table`.
        patches: the staged storm patches.
        rows: which table rows are samples (a season split, say). Default: every row with a patch now.
        pixels: crop size the network reads (the staged patches are wider).
        jitter_km: the first-guess centre is the true one moved by up to this much, uniformly over a disc.
            Held-out samples get a fixed perturbation per row, so a score is repeatable.
        train: random jitter and a random 2 K calibration offset when True.
    """

    def __init__(self, table: dict, patches: StormPatches, rows=None, pixels: int = 256, jitter_km: float = 150.0,
                 train: bool = True, seed: int = 0, frame_hours=STORM_FRAME_HOURS) -> None:
        self.table, self.patches, self.pixels = table, patches, pixels
        self.jitter_km, self.train, self.seed = jitter_km, train, seed
        self.frame_hours = tuple(frame_hours)
        candidates = np.arange(len(table["time"])) if rows is None else np.asarray(rows)
        self.rows = np.array([r for r in candidates if int(r) in patches], dtype=np.int64)
        self.lookup = {(sid, int(t)): i for i, (sid, t) in enumerate(zip(table["sid"], table["time"]))}
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _rng(self, item: int) -> np.random.Generator:
        if self.train:
            return np.random.default_rng((self.seed, self.epoch, item, int(torch.initial_seed() % (2 ** 31))))
        return np.random.default_rng((self.seed, int(self.rows[item])))

    def __getitem__(self, item: int) -> dict:
        table, row = self.table, int(self.rows[item])
        rng = self._rng(item)
        lat, lon = float(table["lat"][row]), float(table["lon"][row])
        radius = self.jitter_km * math.sqrt(rng.uniform())
        angle = rng.uniform(0, 2 * math.pi)
        north, east = radius * math.cos(angle), radius * math.sin(angle)
        dlat, dlon = _offset_degrees(lat, north, east)
        guess = (lat + dlat, (lon + dlon + 180.0) % 360.0 - 180.0)

        sid, now = table["sid"][row], int(table["time"][row])
        frames = np.zeros((len(self.frame_hours), 2, self.pixels, self.pixels), np.uint8)
        available = np.zeros(len(self.frame_hours), bool)
        for k, hours in enumerate(self.frame_hours):
            source = self.lookup.get((sid, now - hours * 3600))
            patch = None if source is None else self.patches.patch(source)
            if patch is None:
                continue
            crop, inside = recentre(patch, (float(table["lat"][source]), float(table["lon"][source])), guess,
                                    self.pixels)
            frames[k], available[k] = crop, inside > 0.5
        if self.train:
            shift = int(rng.integers(-3, 4))                         # ~2 K calibration spread between satellites
            frames = np.where(frames > 0, np.clip(frames.astype(np.int16) + shift, 1, 255), 0).astype(np.uint8)

        stamp = np.datetime64(now, "s").astype(object)
        local_hour = (stamp.hour + stamp.minute / 60.0 + lon / 15.0) % 24.0
        return {
            "codes": torch.from_numpy(frames),
            "available": torch.from_numpy(available),
            "latitude": torch.tensor(guess[0], dtype=torch.float32),
            "context": torch.tensor([abs(guess[0]) / 30.0, math.sin(2 * math.pi * local_hour / 24.0),
                                     math.cos(2 * math.pi * local_hour / 24.0), float(available.mean())],
                                    dtype=torch.float32),
            # The true centre relative to the first guess, kilometres (north, east): the centre-fix target.
            "centre_offset_km": torch.tensor([-north, -east], dtype=torch.float32),
            "wind_kt": torch.tensor(table["wind_kt"][row]),
            "pressure_hpa": torch.tensor(table["pressure_hpa"][row]),
            "rmw_nmi": torch.tensor(table["rmw_nmi"][row]),
            "radii_nmi": torch.from_numpy(np.asarray(table["radii_nmi"][row], np.float32)),
            "wind_change_kt": torch.from_numpy(np.asarray(table["wind_change_kt"][row], np.float32)),
            "tropical": torch.tensor(bool(table["tropical"][row])),
            "past_motion_km": torch.from_numpy(np.asarray(table["past_motion_km"][row], np.float32)),
            "future_motion_km": torch.from_numpy(np.asarray(table["future_motion_km"][row], np.float32)),
            "row": torch.tensor(row),
        }


def prepared_to_state(prepared: np.ndarray, normalizer, channels: int | None) -> torch.Tensor:
    """
    One staged ERA5 step ``(lat, lon, C_prepared)`` into the model's ``(points, channels)`` analysis layout.

    Exactly what :func:`naturev1.era5._build_window` does for one frame: NaN to zero with an observation
    mask channel for gappy variables, flattened, zero-padded to ``channels``.
    """
    from .era5 import split_prepared

    values, observed = split_prepared(np.asarray(prepared, np.float32))
    if normalizer.masked:
        values = np.concatenate([values, observed[..., normalizer.masked].astype(np.float32)], axis=-1)
    state = torch.from_numpy(np.ascontiguousarray(values.reshape(-1, values.shape[-1])))
    if channels is not None and channels > state.shape[-1]:
        state = torch.nn.functional.pad(state, (0, channels - state.shape[-1]))
    return state


class RetrievalDataset(torch.utils.data.Dataset):
    """
    (satellite now, -3 h, -6 h; ERA5 ``stale`` hours ago) -> ERA5 now.

    Args:
        frames: the staged global frames.
        era5: a :func:`naturev1.materialise` memmap covering the targets and their stale backgrounds.
        times: target valid times (unix seconds) to use; default every staged ERA5 step that has a current
            satellite frame.
        stale_hours: ages the background is drawn from. Training draws one at random per sample;
            held-out samples cycle through them deterministically.
        channels: the model's ``analysis_channels``.
    """

    def __init__(self, frames: GlobalFrames, era5: str | Path, times=None, stale_hours=STALE_HOURS,
                 channels: int | None = None, train: bool = True, seed: int = 0,
                 frame_hours=GLOBAL_FRAME_HOURS) -> None:
        from .era5 import Normalizer

        self.frames, self.path = frames, Path(era5)
        meta = json.loads(self.path.with_suffix(".json").read_text())
        self.variables = list(meta["variables"])
        self.normalizer = Normalizer.from_dict(meta["statistics"])
        stamps = np.asarray(meta["time"], dtype=np.int64)
        self.era_row = {int(t): i for i, t in enumerate(stamps)}
        self.stale_hours, self.channels = tuple(stale_hours), channels
        self.train, self.seed, self.frame_hours = train, seed, tuple(frame_hours)
        if times is None:
            times = stamps
        wanted = []
        for t in np.asarray(times, dtype=np.int64):
            if int(t) in frames.index and int(t) in self.era_row and any(
                    int(t) - h * 3600 in self.era_row for h in self.stale_hours):
                wanted.append(int(t))
        self.times = np.array(wanted, dtype=np.int64)
        self.maps = _Memmaps()
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.times)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, item: int) -> dict:
        from .model import calendar_features

        now = int(self.times[item])
        usable = [h for h in self.stale_hours if now - h * 3600 in self.era_row]
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, item, int(torch.initial_seed() % (2 ** 31))))
            stale = int(rng.choice(usable))
        else:
            stale = usable[item % len(usable)]
        values = self.maps.get(self.path)
        background = prepared_to_state(values[self.era_row[now - stale * 3600]], self.normalizer, self.channels)
        target = prepared_to_state(values[self.era_row[now]], self.normalizer, self.channels)
        shape = self.frames.shape
        sat = np.zeros((len(self.frame_hours), len(GLOBAL_CHANNELS), *shape), np.uint8)
        available = np.zeros(len(self.frame_hours), bool)
        for k, hours in enumerate(self.frame_hours):
            frame = self.frames.frame(now - hours * 3600)
            if frame is not None:
                sat[k], available[k] = frame, True
        stamp = torch.tensor([float(now)], dtype=torch.float64)
        return {
            "satellite": torch.from_numpy(sat),
            "satellite_available": torch.from_numpy(available),
            "background": background,
            "target": target,
            "stale_hours": torch.tensor(float(stale)),
            "calendar": calendar_features(stamp)[0],
            "valid_time": stamp[0],
        }


def season_rows(table: dict, seasons) -> np.ndarray:
    """Table rows whose fix falls in any of ``seasons`` (calendar years of the fix)."""
    years = table["time"].astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
    return np.where(np.isin(years, list(seasons)))[0]


__all__ = ["GLOBAL_FRAME_HOURS", "GlobalFrames", "RetrievalDataset", "STALE_HOURS", "STORM_FRAME_HOURS",
           "StormPatches", "StormSequenceDataset", "prepared_to_state", "recentre", "season_rows",
           "CHANGE_HOURS", "STAGED_PATCH_PIXELS"]
