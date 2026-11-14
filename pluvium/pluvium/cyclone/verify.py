# Copyright 2026 Nathan. Apache-2.0.
"""
Scoring track forecasts the way the National Hurricane Center does.

* The error is the great-circle distance between the forecast and best-track positions at the same time.
* A forecast counts only when the storm is a tropical or subtropical cyclone in the best track at the start and at
  the verifying time.
* Forecasts are compared on a homogeneous sample: only the cases every compared technique forecast. Averages over
  different cases say nothing, because storms differ far more than techniques do.
* Skill is measured against climatology and persistence (CLIPER5, ``OCD5`` in the decks): the percentage by which
  the error is smaller.

Uncertainty comes from a bootstrap over storms, not forecasts: successive forecasts of one storm share its errors.
"""

from __future__ import annotations

import numpy as np

from .atcf import BestTrack
from .geodesy import KM_PER_NM, along_cross, distance_km

LEADS = (12, 24, 36, 48, 60, 72, 96, 120)


def verifiable(best: BestTrack, init: int, lead: int) -> bool:
    return best.tropical(int(init)) and best.tropical(int(init) + int(lead))


def case_errors(points: dict, best: BestTrack, init: int, leads=LEADS) -> dict:
    """``{lead: error km}`` for the leads that verify and that the forecast reaches (``points``: lead -> (lat, lon))."""
    out = {}
    for lead in leads:
        if lead not in points or not verifiable(best, init, lead):
            continue
        obs = best.position(int(init) + lead)
        lat, lon = points[lead][0], points[lead][1]
        if obs is None or not np.isfinite(lat) or not np.isfinite(lon):
            continue
        out[lead] = float(distance_km(lat, lon, obs[0], obs[1]))
    return out


def case_along_cross(points: dict, best: BestTrack, init: int, leads=LEADS) -> dict:
    """``{lead: (along km, cross km)}``: positive when the forecast is ahead of the storm, and to the right of it."""
    out = {}
    for lead in leads:
        if lead not in points or not verifiable(best, init, lead):
            continue
        hour = int(init) + lead
        obs, heading = best.position(hour), best.heading(hour)
        if obs is None or not np.isfinite(heading) or not np.isfinite(points[lead][0]):
            continue
        a, c = along_cross(points[lead][0], points[lead][1], obs[0], obs[1], heading)
        out[lead] = (float(a), float(c))
    return out


class Scoreboard:
    """Errors of many techniques over many cases: ``add`` them, then compare on homogeneous samples."""

    def __init__(self, leads=LEADS) -> None:
        self.leads = tuple(leads)
        self.errors = {}                                     # tech -> {(storm, init, lead): km}

    def add(self, tech: str, storm: str, init: int, errors: dict) -> None:
        table = self.errors.setdefault(tech, {})
        for lead, km in errors.items():
            table[(storm, int(init), int(lead))] = float(km)

    def common(self, techs, lead: int) -> list:
        keys = None
        for tech in techs:
            mine = {k for k in self.errors.get(tech, {}) if k[2] == lead}
            keys = mine if keys is None else keys & mine
        return sorted(keys or [])

    def mean_errors(self, techs, unit: str = "nm") -> dict:
        """``{lead: {"n": cases, tech: mean error}}`` on the sample every technique in ``techs`` forecast."""
        scale = KM_PER_NM if unit == "nm" else 1.0
        out = {}
        for lead in self.leads:
            keys = self.common(techs, lead)
            if not keys:
                continue
            out[lead] = {"n": len(keys), **{t: float(np.mean([self.errors[t][k] for k in keys])) / scale for t in techs}}
        return out

    def versus(self, tech: str, rival: str, lead: int, resamples: int = 2000, seed: int = 0) -> dict | None:
        """Mean error difference (``tech`` minus ``rival``, km) on their common cases, with a 95% interval from a
        bootstrap over storms."""
        keys = self.common([tech, rival], lead)
        if len(keys) < 5:
            return None
        diff = np.array([self.errors[tech][k] - self.errors[rival][k] for k in keys])
        storms = np.array([k[0] for k in keys])
        names = np.unique(storms)
        rng = np.random.default_rng(seed)
        sums = np.array([diff[storms == s].sum() for s in names])
        counts = np.array([(storms == s).sum() for s in names])
        pick = rng.integers(0, names.size, size=(resamples, names.size))
        means = sums[pick].sum(1) / counts[pick].sum(1)
        lo, hi = np.quantile(means, [0.025, 0.975])
        return {"n": len(keys), "storms": int(names.size), "delta_km": float(diff.mean()), "ci95": [float(lo), float(hi)],
                "relative": float(diff.mean() / np.mean([self.errors[rival][k] for k in keys]))}

    def skill(self, tech: str, baseline: str = "OCD5") -> dict:
        """``{lead: percent}`` by which ``tech`` beats the climatology-and-persistence baseline on common cases."""
        out = {}
        for lead in self.leads:
            keys = self.common([tech, baseline], lead)
            if len(keys) >= 5:
                mine = np.mean([self.errors[tech][k] for k in keys])
                base = np.mean([self.errors[baseline][k] for k in keys])
                out[lead] = float(100.0 * (1.0 - mine / base))
        return out


__all__ = ["LEADS", "Scoreboard", "case_along_cross", "case_errors", "verifiable"]
