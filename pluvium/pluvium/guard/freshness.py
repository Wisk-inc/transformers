# Copyright 2026 Nathan. Apache-2.0.
"""
Old data posing as new.

A forecast "from now" built on a driving run 30 hours old, a GOES scene from yesterday, or an airport report from
three hours ago fed as "the last hour" is wrong in a way no score will show until the rain is missed. Each input's
age is measured against the issue time and checked against limits: observations older than their limit are dropped
from the model's "rain now" inputs (their known-flags turn off, exactly as in training when they were missing), and
everything is reported in the forecast's notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

LIMITS_HOURS = {"driving_run": 18.0, "satellite": 1.5, "airports": 1.5, "lightning": 0.5}


@dataclass
class Freshness:
    issue_hour: int
    ages: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def add(self, name: str, observed_hour: float | None) -> bool:
        """Record an input's time; returns whether it is fresh enough to use."""
        if observed_hour is None:
            self.ages[name] = None
            self.notes.append(f"{name}: not available")
            return False
        age = float(self.issue_hour - observed_hour)
        self.ages[name] = age
        limit = LIMITS_HOURS.get(name, 6.0)
        if age < -0.5:
            self.notes.append(f"{name}: time {age:+.1f} h from the issue time (clock or feed error); not used")
            return False
        if age > limit:
            self.notes.append(f"{name}: {age:.1f} h old (limit {limit:.1f} h); not used as current")
            return False
        return True

    def ok(self, name: str) -> bool:
        age = self.ages.get(name)
        return age is not None and -0.5 <= age <= LIMITS_HOURS.get(name, 6.0)


__all__ = ["Freshness", "LIMITS_HOURS"]
