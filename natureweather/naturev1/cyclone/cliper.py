# Copyright 2026 Nathan. Apache-2.0.
"""
CLIPER, rebuilt: climatology and persistence as a regression, per basin, in the poleward frame.

The idea (Neumann 1972, still NHC's skill baseline as CLIPER5): where storms in this place, at this time of
year, moving like this, went next -- from 35 years of best tracks, with no model fields at all. It beats pure
extrapolation from about a day on because climatology knows storms recurve at the subtropical ridge and
slow down near it; extrapolation knows only the last six hours.

This version improves on the classic in three ways:

* **Poleward frame**: southern-hemisphere storms are mirrored north-south, so the Southern Indian and South
  Pacific basins share one model with twice the data, and every "poleward" coefficient means the same thing
  in both hemispheres.
* **Three motion windows** (the last 6 h, the 6 h before, the 12 h before that): curvature and acceleration
  enter as differences of windows, which the regression weighs against climatology lead by lead.
* **Displacements in an exact local frame** (azimuthal equidistant), not degree differences, so the
  regression's target means the same at 10 and 40 degrees latitude.

Coefficients are fitted once on 1980-2014 (no overlap with the 2016-2022 seasons the benchmark uses) by
:mod:`naturev1.cyclone.fit` and stored as constants in :mod:`naturev1.cyclone.coefficients`.
"""

from __future__ import annotations

import numpy as np

from .geometry import from_local, local_displacement_km, wrap_longitude

#: Basin groups sharing one regression, and each group's reference longitude.
GROUPS = {"NA": ("NA", -60.0), "EP": ("EP", -120.0), "WP": ("WP", 140.0), "NI": ("NI", 85.0),
          "SI": ("SH", 80.0), "SP": ("SH", 170.0), "SA": ("SH", -30.0)}

PREDICTORS = ("one", "lat", "lat2", "lon", "u6", "v6", "u12", "v12", "u24", "v24", "wind", "dwind", "season_s",
              "season_c", "lat_u6", "lat_v6", "u6v6", "u6sq", "v6sq", "wind_v6", "lon_v6")


def _motion(case, start: int, stop: int):
    """(east, north) km/h between history offsets ``start`` and ``stop`` hours (negative), or None."""
    from .data import HISTORY_HOURS

    a, b = HISTORY_HOURS.index(start), HISTORY_HOURS.index(stop)
    la, loa = case.history["lat"][a], case.history["lon"][a]
    lb, lob = case.history["lat"][b], case.history["lon"][b]
    if not (np.isfinite(la) and np.isfinite(lb)):
        return None
    east, north = local_displacement_km(la, loa, lb, lob)
    hours = stop - start
    return east / hours, north / hours


def predictors(case) -> np.ndarray | None:
    """The CLIPER predictor vector in the poleward frame, or None without a 6- or 12-hour history."""
    sign = -1.0 if case.lat < 0 else 1.0
    m6 = _motion(case, -6, 0) or (_motion(case, -12, 0))
    if m6 is None:
        return None
    m12 = _motion(case, -12, -6) or m6
    m24 = _motion(case, -24, -12) or m12
    u6, v6 = m6[0], sign * m6[1]
    u12, v12 = m12[0], sign * m12[1]
    u24, v24 = m24[0], sign * m24[1]
    _, reference = GROUPS.get(case.basin, ("NA", -60.0))
    lat = abs(case.lat) / 30.0
    lon = wrap_longitude(case.lon - reference) / 60.0
    wind = (case.wind_kt if np.isfinite(case.wind_kt) else 50.0) / 100.0
    before = case.history["wind"][-3]                                       # 12 hours ago
    dwind = ((case.wind_kt - before) / 20.0) if np.isfinite(before) and np.isfinite(case.wind_kt) else 0.0
    day = (np.datetime64(case.time, "D") - np.datetime64(str(case.time)[:4] + "-01-01", "D")).astype(int)
    phase = 2 * np.pi * ((day + (182 if sign < 0 else 0)) % 365) / 365.0
    s = 1.0 / 20.0                                                          # km/h -> order one
    u6s, v6s = u6 * s, v6 * s
    return np.array([1.0, lat, lat * lat, lon, u6s, v6s, u12 * s, v12 * s, u24 * s, v24 * s, wind, dwind,
                     np.sin(phase), np.cos(phase), lat * u6s, lat * v6s, u6s * v6s, u6s * u6s, v6s * v6s,
                     wind * v6s, lon * v6s])


def group_of(case) -> str:
    return GROUPS.get(case.basin, ("NA", 0.0))[0]


def cliper_track(case, leads, table: dict | None = None) -> np.ndarray:
    """``(L, 2)`` lat/lon from the stored coefficients (NaN where the leads or the group are not covered)."""
    if table is None:
        from .coefficients import CLIPER as table
    out = np.full((len(leads), 2), np.nan)
    x = predictors(case)
    group = table.get(group_of(case))
    if x is None or group is None:
        return out
    sign = -1.0 if case.lat < 0 else 1.0
    stored_leads = list(table["leads"])
    weights = np.asarray(group, np.float64)                                  # (L, 2, P)
    for k, lead in enumerate(leads):
        if lead not in stored_leads:
            continue
        east, north = weights[stored_leads.index(lead)] @ x
        lat, lon = from_local(case.lat, case.lon, east, sign * north)
        out[k] = lat, lon
    return out


def fit_cliper(cases, leads, ridge: float = 1e-2) -> dict:
    """
    Least squares (with a small ridge) of the poleward-frame displacement at each lead on :func:`predictors`,
    one regression per basin group. Returns ``{"leads": [...], group: (L, 2, P) nested lists}``.
    """
    rows: dict[str, list] = {}
    for case in cases:
        x = predictors(case)
        if x is None:
            continue
        sign = -1.0 if case.lat < 0 else 1.0
        target = np.full((len(leads), 2), np.nan)
        for k in range(len(leads)):
            if case.truth["valid"][k]:
                east, north = local_displacement_km(case.lat, case.lon, case.truth["lat"][k], case.truth["lon"][k])
                target[k] = east, sign * north
        rows.setdefault(group_of(case), []).append((x, target))
    table = {"leads": [int(h) for h in leads]}
    for group, items in rows.items():
        x = np.stack([i[0] for i in items])
        y = np.stack([i[1] for i in items])
        weights = np.zeros((len(leads), 2, x.shape[1]))
        for k in range(len(leads)):
            ok = np.isfinite(y[:, k, 0])
            if ok.sum() < 5 * x.shape[1]:
                weights[k] = np.nan
                continue
            a = x[ok]
            gram = a.T @ a + ridge * ok.sum() * np.diag([0.0] + [1.0] * (x.shape[1] - 1))
            weights[k] = np.linalg.solve(gram, a.T @ y[ok, k]).T
        table[group] = weights.round(5).tolist()
    return table


__all__ = ["GROUPS", "PREDICTORS", "cliper_track", "fit_cliper", "group_of", "predictors"]
