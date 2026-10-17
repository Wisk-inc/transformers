# Copyright 2026 Nathan. Apache-2.0.
"""
Track-shape analogs: storms that were moving *like this one*, turned to face the same way.

The idea: CLIPER is a regression -- it averages every storm into one linear rule. Analogs are the
non-parametric opposite: find the historical storms whose recent track looked most like this one's, and see
what they did next. Two refinements make it work far from the classic version:

* **rotation to the motion frame** -- each historical case is stored with its future displacements rotated
  so that its last 12 hours of motion point along the same axis. A storm heading north-west at 5 m/s is then
  compared with every storm that was turning and accelerating the same way, wherever it was heading, and its
  future is rotated back onto this storm's heading. Curvature and acceleration -- the shape of the track --
  are what is matched, not the compass direction;
* **context still counts** -- latitude, longitude within the basin, season and intensity enter the distance,
  so a turning storm at 28 N is matched with turning storms near the ridge, not in the deep tropics.

What is stored and averaged is each analog's *departure from persistence* (its future minus carrying on at its
own current speed and heading), added back onto this storm's own current motion -- so the first hours inherit
the storm's exact motion and the analogs contribute the turns and speed changes that follow.

Futures are combined with Gaussian kernel weights over the nearest analogs; the poleward frame mirrors the
southern hemisphere so both share one database. The database is built from best tracks
(:meth:`AnalogDatabase.from_cases`) -- for the benchmark, 1980-2014 only.
"""

from __future__ import annotations

import numpy as np

from .geometry import from_local, local_displacement_km, rotate, wrap_longitude

#: Feature scales (one unit of distance per scale).
SCALES = {"lat": 4.0, "lon": 15.0, "season": 0.5, "wind": 30.0, "speed": 6.0, "turn": 15.0, "accel": 4.0, "turn24": 20.0}


def _shape_features(case):
    """``(features, heading_deg)`` in the poleward frame, or None without a 12-hour history."""
    from .cliper import GROUPS
    from .data import HISTORY_HOURS

    h = list(HISTORY_HOURS)
    lat, lon = case.history["lat"], case.history["lon"]
    sign = -1.0 if case.lat < 0 else 1.0

    def leg(a, b):
        i, j = h.index(a), h.index(b)
        if not (np.isfinite(lat[i]) and np.isfinite(lat[j])):
            return None
        e, n = local_displacement_km(lat[i], lon[i], lat[j], lon[j])
        return np.array([e, sign * n]) / ((b - a))
    m0 = leg(-6, 0)
    m1 = leg(-12, -6)
    if m0 is None or m1 is None:
        return None
    m2 = leg(-24, -12)
    heading = np.degrees(np.arctan2(m0[0], m0[1]))
    def angle(a, b):
        return float((np.degrees(np.arctan2(a[0], a[1]) - np.arctan2(b[0], b[1])) + 180.0) % 360.0 - 180.0)
    turn = angle(m0, m1)
    turn24 = angle(m0, m2) if m2 is not None else turn
    speed = float(np.hypot(*m0)) / 3.6
    accel = speed - float(np.hypot(*m1)) / 3.6
    _, reference = GROUPS.get(case.basin, ("NA", 0.0))
    day = (np.datetime64(case.time, "D") - np.datetime64(str(case.time)[:4] + "-01-01", "D")).astype(int)
    phase = 2 * np.pi * ((day + (182 if sign < 0 else 0)) % 365) / 365.0
    wind = case.wind_kt if np.isfinite(case.wind_kt) else 50.0
    f = np.array([abs(case.lat) / SCALES["lat"], wrap_longitude(case.lon - reference) / SCALES["lon"],
                  np.cos(phase) / SCALES["season"], np.sin(phase) / SCALES["season"], wind / SCALES["wind"],
                  speed / SCALES["speed"], turn / SCALES["turn"], accel / SCALES["accel"], turn24 / SCALES["turn24"]])
    return f, float(heading)


class AnalogDatabase:
    """Historical cases' shape features and their futures in the motion-aligned poleward frame."""

    def __init__(self, features: np.ndarray, futures: np.ndarray, groups: np.ndarray, leads) -> None:
        self.features, self.futures, self.groups, self.leads = features, futures, groups, list(leads)

    @classmethod
    def from_cases(cls, cases, leads) -> "AnalogDatabase":
        from .cliper import group_of

        feats, futs, groups = [], [], []
        for c in cases:
            got = _shape_features(c)
            if got is None:
                continue
            f, heading = got
            sign = -1.0 if c.lat < 0 else 1.0
            speed_kmh = f[5] * SCALES["speed"] * 3.6
            fut = np.full((len(leads), 2), np.nan)
            for k in range(len(leads)):
                if c.truth["valid"][k]:
                    e, n = local_displacement_km(c.lat, c.lon, c.truth["lat"][k], c.truth["lon"][k])
                    # Rotate so the storm's current heading points north, then keep only what departs from
                    # carrying on straight at the current speed: the analog's *deviation* from persistence.
                    re, rn = rotate(e, sign * n, heading)
                    fut[k] = re, rn - speed_kmh * leads[k]
            feats.append(f)
            futs.append(fut)
            groups.append(group_of(c))
        return cls(np.asarray(feats), np.asarray(futs), np.asarray(groups), leads)

    def forecast(self, case, neighbours: int = 60, bandwidth: float = 1.0, same_group: bool = True) -> np.ndarray:
        """``(L, 2)`` lat/lon: the kernel-weighted mean future of the nearest analogs, rotated to this heading."""
        from .cliper import group_of

        out = np.full((len(self.leads), 2), np.nan)
        got = _shape_features(case)
        if got is None or len(self.features) == 0:
            return out
        f, heading = got
        speed_kmh = f[5] * SCALES["speed"] * 3.6
        pool = self.groups == group_of(case) if same_group else np.ones(len(self.groups), bool)
        if pool.sum() < neighbours:
            pool = np.ones(len(self.groups), bool)
        idx = np.nonzero(pool)[0]
        d2 = ((self.features[idx] - f) ** 2).sum(1)
        order = idx[np.argsort(d2)[:neighbours]]
        weights = np.exp(-0.5 * np.sort(d2)[:neighbours] / bandwidth ** 2)
        sign = -1.0 if case.lat < 0 else 1.0
        for k in range(len(self.leads)):
            fut = self.futures[order, k]
            ok = np.isfinite(fut[:, 0])
            if ok.sum() < 5:
                continue
            w = weights[ok] / weights[ok].sum()
            east, north = (fut[ok] * w[:, None]).sum(0)
            north = north + speed_kmh * self.leads[k]
            east, north = rotate(east, north, -heading)
            lat, lon = from_local(case.lat, case.lon, east, sign * north)
            out[k] = lat, lon
        return out


__all__ = ["AnalogDatabase", "SCALES"]
