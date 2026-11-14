# Copyright 2026 Nathan. Apache-2.0.
"""
Learning inside a storm: which systems have this storm right so far?

Weights learned on last season say which systems are good on average. But a model's errors on one storm repeat from
run to run: if it has had this storm's steering wrong for two days, it very likely still does. The memory here keeps,
for the storm being forecast, how far each system's earlier runs were from where the storm was then observed to be,
and scales the systems' weights by it before the next consensus.

It is strictly causal. An earlier run enters only at leads that have already verified, and "observed" means the
position known in real time at the start of the new run (``CARQ``), not the best track written after the season.

Errors are compared at equal leads, as ratios to the mean over systems, so a system is not punished for the storm
being hard. The scale is ``exp(-gain * (ratio - 1))``, clipped: a system twice as wrong as the average counts for
about a third as much with ``gain = 1``; one that was half as wrong counts for about 1.6 times as much.
"""

from __future__ import annotations

import numpy as np

from .geodesy import distance_km


class StormMemory:
    def __init__(self, gain: float = 1.0, halflife_hours: float = 36.0, clip: tuple = (0.2, 3.0)) -> None:
        self.gain, self.halflife, self.clip = gain, halflife_hours, clip
        self.runs = []                                   # (init, {system: {lead: (lat, lon)}}, real-time fixes)

    def remember(self, init: int, points: dict, carq: dict) -> None:
        """Keep a run's (bias-corrected) system tracks and the storm's real-time fixes ``{hour: {lead: (lat, lon,
        wind)}}`` (the ``CARQ`` rows of its a-deck)."""
        self.runs.append((int(init), points, carq))

    @staticmethod
    def _observed(carq: dict, hour: int, now: int):
        """Where the storm was at ``hour`` as known at ``now``: real-time fixes only (``CARQ``), else nothing."""
        rows = carq.get(int(now), {})
        back = int(hour) - int(now)
        if back in rows:
            return rows[back][0], rows[back][1]
        rows = carq.get(int(hour), {})                   # the fix made at that hour itself
        if 0 in rows and hour <= now:
            return rows[0][0], rows[0][1]
        return None

    def scale(self, now: int, systems: list) -> dict:
        """``{system: factor}`` for a forecast starting at ``now`` (1 for every system when nothing has verified)."""
        ratios = {s: [] for s in systems}
        weights = {s: [] for s in systems}
        for init, points, carq in self.runs:
            age = now - init
            if age <= 0:
                continue
            decay = 0.5 ** ((age - 12) / self.halflife) if age > 12 else 1.0
            for lead in sorted({ld for p in points.values() for ld in p}):
                if lead > age or lead <= 0:
                    continue                                 # not verified yet
                obs = self._observed(carq, init + lead, now)
                if obs is None:
                    continue
                errs = {s: float(distance_km(p[lead][0], p[lead][1], obs[0], obs[1]))
                        for s, p in points.items() if lead in p and np.isfinite(p[lead][0])}
                if len(errs) < 2:
                    continue
                mean = max(np.mean(list(errs.values())), 10.0)      # 10 km floor: fixes are no better than that
                for s, e in errs.items():
                    if s in ratios:
                        ratios[s].append(max(e, 10.0) / mean)
                        weights[s].append(decay)
        out = {}
        for s in systems:
            if not ratios[s]:
                out[s] = 1.0
                continue
            r = float(np.average(ratios[s], weights=weights[s]))
            trust = min(1.0, sum(weights[s]) / 2.0)                 # one verified lead moves the weight half-way
            out[s] = float(np.clip(np.exp(-self.gain * trust * (r - 1.0)), *self.clip))
        return out


__all__ = ["StormMemory"]
