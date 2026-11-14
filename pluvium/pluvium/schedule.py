# Copyright 2026 Nathan. Apache-2.0.
"""
The forecast's clock: which hours each frame stands for, shared by training, live input and the products.

A forecast issued at hour ``t0`` has three **history** frames (t0-3, t0-2, t0-1: the rain already observed),
then one frame per hour for the first day (when timing matters most and changes fastest), then one frame per three
hours to day three. Every frame carries the driving fields at its centre time and the driving model's rain rate
averaged over its span. Output is hourly throughout: a three-hour frame speaks for each of its three hours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Schedule:
    history: int = 3
    hourly_until: int = 24
    three_hourly_until: int = 72

    @property
    def leads(self) -> np.ndarray:
        """Lead (hours from t0) at the start of each frame, history included (negative)."""
        return np.concatenate([np.arange(-self.history, 0), np.arange(0, self.hourly_until),
                               np.arange(self.hourly_until, self.three_hourly_until, 3)]).astype(np.int64)

    @property
    def spans(self) -> np.ndarray:
        """Hours each frame stands for."""
        return np.where(self.leads >= self.hourly_until, 3, 1).astype(np.int64)

    @property
    def centres(self) -> np.ndarray:
        """Lead of the instant the driving fields are read at: the frame's start for hourly frames, its middle hour
        for three-hourly ones."""
        return self.leads + (self.spans - 1) // 2

    @property
    def num_frames(self) -> int:
        return int(self.leads.size)

    @property
    def num_hours(self) -> int:
        """Hourly outputs from t0: 0 .. three_hourly_until - 1."""
        return int(self.three_hourly_until)

    @property
    def future(self) -> np.ndarray:
        """Indices of the frames at or after t0."""
        return np.nonzero(self.leads >= 0)[0]

    def hour_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """For each output hour 0..num_hours-1: ``(frame index, sub-hour 0..2 within the frame)``."""
        frames, subs = [], []
        for k in self.future:
            for s in range(int(self.spans[k])):
                frames.append(int(k))
                subs.append(s)
        return np.array(frames, np.int64), np.array(subs, np.int64)


DEFAULT = Schedule()

__all__ = ["DEFAULT", "Schedule"]
