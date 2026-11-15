# Copyright 2026 Nathan. Apache-2.0.
"""
When in the day it rains: the first three days of the forecast, split into six-hour periods.

The 14-day forecast says how likely rain is on each day. For the first days the question people ask is when. The
six systems publish their rain every six hours or oftener, so for each place and period this module asks their
members (the ECMWF and NOAA ensembles, and the four single forecasts) two things: on how many of them is the day
wet, and on how many of those is this period wet. The second over the first is the chance that the period is wet
given that the day is. Multiplied by the calibrated chance of a wet day from the blend, it gives the chance of rain
in the period:

    chance(rain in the period) = chance(rain that day, calibrated) x share of wet-day members wet in the period

So the daily number stays the calibrated one, and the periods only say how it is spread over the day. A period is
wet with at least 1 mm, as a day is, so a period's chance can never pass its day's.

The members' answer is mixed, half and half, with the usual time of day of rain at the place (from the satellite
record, :func:`usual_timing`). Checked for Jamaica, the members alone placed the rain of a wet day worse than what
is usual at the place, and the mixture placed it better than either. So a blend that has no record of what is usual
(``timing.npz``, :func:`fit_usual`) gets no periods at all.

Periods are six hours of UTC (00-06, 06-12, 12-18, 18-24) because that is how every system publishes. They are
labelled in the place's own clock (:func:`labels`): in Jamaica 7 pm to 1 am, 1 am to 7 am, 7 am to 1 pm and 1 pm to
7 pm.

This needs no GPU. The hour-by-hour forecast of the Pluvium-500 network is finer, and needs the network trained.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

HOURS = 6                     # length of a period
PERIODS = 24 // HOURS         # periods in a UTC day
DAYS = 3                      # days of the forecast that are split
WET_MM = 1.0                  # a wet period, as a wet day: at least this much
#: how much the members' answer counts against what is usual at the place, by day (1, 2, 3). Half and half scored
#: best of five mixtures for Jamaica on 61 runs of May 2025 to May 2026: 7.8% better than what is usual alone on wet
#: days, where the members alone were 4.9% worse (:mod:`pluvium.benchmarks.timing_bench`, docs/TIMING_BENCHMARK.md)
MODEL_WEIGHT = (0.5, 0.5, 0.5)
#: weight of the prior in the share of wet-day members wet in a period, in units of "a system that is sure the day
#: is wet". With few wet-day members the share is noisy, and is pulled towards the prior
PRIOR_WEIGHT = 0.1


def steps_for(init: int, day0: int, days: int = DAYS) -> list:
    """The lead hours of a run started at ``init`` that end each period of the ``days`` days from ``day0`` (00 UTC),
    and the one at ``day0`` itself when the run started earlier."""
    offset = int(day0) - int(init)
    if offset < 0 or offset % HOURS:
        raise ValueError("a run must start at or before the first day, a whole number of periods earlier")
    return [offset + HOURS * k for k in range(PERIODS * days + 1) if offset + HOURS * k > 0]


def period_amounts(z: dict, init: int, day0: int, days: int = DAYS) -> np.ndarray:
    """Rain in each period on a run's own grid, ``(members, days x 4, H, W)`` mm, from its cached accumulations
    (``z`` as :meth:`pluvium.sources.archive.ArchiveReader.load` returns it). Periods the run does not reach, or
    whose steps were not read, are NaN. Accumulated rain can dip by a hair between steps (the files are packed);
    a period is never negative."""
    values = np.asarray(z["values"], np.float32)
    at = {int(s): k for k, s in enumerate(z["steps"])}
    offset = int(day0) - int(init)
    out = np.full((values.shape[0], PERIODS * days) + values.shape[2:], np.nan, np.float32)
    for k in range(PERIODS * days):
        a, b = offset + HOURS * k, offset + HOURS * (k + 1)
        if b not in at or (a > 0 and a not in at):
            continue
        before = values[:, at[a]] if a > 0 else 0.0
        out[:, k] = np.maximum(values[:, at[b]] - before, 0.0)
    return out


def at_places(amounts: np.ndarray, z: dict, lat, lon) -> np.ndarray:
    """``(members, periods, H, W)`` on a run's grid -> ``(members, periods, places)``, as the daily forecast samples
    its fields. Members with nothing known in a period stay NaN there."""
    from .dataset import bilinear

    known = np.isfinite(amounts).reshape(amounts.shape[:2] + (-1,)).all(-1)            # (members, periods)
    out = bilinear(amounts, np.asarray(z["lat"], np.float64), np.asarray(z["lon"], np.float64), lat, lon)
    out[~known] = np.nan
    return out


def read_periods(reader, region: str, runs: dict, day0: int, lat, lon, days: int = DAYS) -> dict:
    """``{system: (members, days x 4, places)}`` for the systems whose six-hour fields can be read. ``runs`` maps
    each system to the hour its run started (the runs the daily forecast used), ``reader`` is an
    :class:`pluvium.sources.archive.ArchiveReader` for the area."""
    out = {}
    for system, init in runs.items():
        try:
            steps = steps_for(int(init), int(day0), days)
        except ValueError:
            continue
        if not reader.read_steps(system, int(init), steps):
            continue
        z = reader.load(region, f"{system}_6h", int(init))
        if z is None:
            continue
        got = at_places(period_amounts(z, int(init), int(day0), days), z, lat, lon)
        if np.isfinite(got).any():
            out[system] = got
    return out


def member_shares(periods: dict, days: int = DAYS, wet: float = WET_MM) -> dict:
    """What the systems' members say, each system counting the same: for every day, period and place,

    * ``wet_day``    ``(days, places)``     the share of members on which the day is wet;
    * ``wet_period`` ``(days, 4, places)``  the share on which the period is wet (and so the day);
    * ``amount``     ``(days, 4, places)``  the mean rain in the period, mm;
    * ``systems``    how many systems went in.

    A system with a period missing on a day is left out of that day.
    """
    n = next(iter(periods.values())).shape[-1] if periods else 0
    wet_day, wet_period = np.zeros((days, n)), np.zeros((days, PERIODS, n))
    amount, weight = np.zeros((days, PERIODS, n)), np.zeros(days)
    for values in periods.values():
        v = values.reshape(values.shape[0], days, PERIODS, n)
        for d in range(days):
            ok = np.isfinite(v[:, d]).all((1, 2))                               # members with the whole day
            if not ok.any():
                continue
            day = v[ok, d]                                                      # (members, 4, places)
            wet_day[d] += (day.sum(1) >= wet).mean(0)
            wet_period[d] += (day >= wet).mean(0)
            amount[d] += day.mean(0)
            weight[d] += 1.0
    scale = np.maximum(weight, 1.0)
    return {"wet_day": wet_day / scale[:, None], "wet_period": wet_period / scale[:, None, None],
            "amount": amount / scale[:, None, None], "systems": weight}


def conditional(shares: dict, prior=None, prior_weight: float = PRIOR_WEIGHT) -> np.ndarray:
    """The chance that each period is wet given that its day is, ``(days, 4, places)``: the share of wet-day
    members that are wet in the period, pulled towards ``prior`` where few members have a wet day.

    ``prior`` is ``(days, 4, places)`` or None. Without one, each day's own answer pooled over all places is used:
    when in the day the systems put the rain, over the whole area.
    """
    wet_day, wet_period = shares["wet_day"], shares["wet_period"]
    if prior is None:
        area = wet_period.sum(-1) / np.maximum(wet_day.sum(-1), 1e-9)[:, None]         # (days, 4)
        area = np.where(wet_day.sum(-1)[:, None] > 0, area, 1.0 / PERIODS)
        prior = np.broadcast_to(area[:, :, None], wet_period.shape)
    return np.clip((wet_period + prior_weight * prior) / (wet_day[:, None, :] + prior_weight), 0.0, 1.0)


def mixed(model: np.ndarray, usual: np.ndarray | None, weights=MODEL_WEIGHT) -> np.ndarray:
    """The members' answer mixed with what is usual at the place, day by day; ``model`` alone without ``usual``."""
    if usual is None:
        return model
    w = np.asarray([weights[min(d, len(weights) - 1)] for d in range(model.shape[0])])[:, None, None]
    return w * model + (1.0 - w) * usual


def usual_timing(rain: np.ndarray, months: np.ndarray, wet: float = WET_MM, pool: int = 1, least: int = 8) -> np.ndarray:
    """How rain is usually spread over the day at each cell, by calendar month: the chance that a period is wet on
    a wet day, ``(12, 4, cells)``, from six-hour truth ``rain`` ``(days, 4, cells)`` with each day's ``months``
    (1-12). A month is pooled with ``pool`` months either side. Where fewer than ``least`` wet days are known the
    cells' common answer is used, and 1/4 where there is none."""
    rain = np.asarray(rain, np.float64)
    known = np.isfinite(rain).all(1)                                            # (days, cells)
    wet_day = known & (np.nansum(rain, 1) >= wet)
    wet_period = wet_day[:, None, :] & (rain >= wet)
    out = np.full((12, PERIODS, rain.shape[-1]), 1.0 / PERIODS)
    for m in range(1, 13):
        near = np.isin(months, [((m - 1 + k) % 12) + 1 for k in range(-pool, pool + 1)])
        days = wet_day[near].sum(0)                                             # (cells,)
        hits = wet_period[near].sum(0)                                          # (4, cells)
        if days.sum() == 0:
            continue
        common = hits.sum(-1) / days.sum()                                      # (4,)
        own = hits / np.maximum(days, 1)
        out[m - 1] = np.where(days >= least, own, common[:, None])
    return out


def fit_usual(root, region: str, box, first: str, end: str, say=None) -> dict:
    """Learn when in the day it usually rains over an area: stage six-hour satellite rain from ``first`` to ``end``
    into ``root/truth6`` (about a minute a month; what is there is not read again) and return ``{"lat", "lon",
    "given"}`` with ``given`` ``(12, 4, H, W)``, the chance that each period is wet on a wet day, by month, on the
    area's 0.25-degree grid."""
    from pathlib import Path

    from .sources.cmorph_parts import stage_parts
    from .util import hour_of, hour_to_datetime64

    truth = stage_parts(Path(root) / "truth6", {region: tuple(box)}, first, end)
    lat, lon = truth.grid(region)
    a, b = hour_of(first), hour_of(end)
    days = np.arange(a - a % 24, b, 24)
    rain = truth.days(region, days)                                             # (days, 4, H, W)
    months = np.array([int(str(hour_to_datetime64(int(h)))[5:7]) for h in days])
    given = usual_timing(rain.reshape(days.size, PERIODS, -1), months).reshape(12, PERIODS, lat.size, lon.size)
    if say is not None:
        known = np.isfinite(rain).all(1).mean()
        say(f"usual time of day of rain: learned from {days.size} days ({known:.0%} of cell-days known)")
    return {"lat": lat, "lon": lon, "given": given.astype(np.float32)}


def chances(daily: np.ndarray, given_wet: np.ndarray) -> np.ndarray:
    """``(days, 4, places)`` chance of rain in each period: the calibrated chance of a wet day times the chance the
    period is wet on a wet day. ``daily`` is ``(days, places)``."""
    return np.clip(daily[:, None, :] * given_wet, 0.0, daily[:, None, :])


def labels(day0: int, zone: str = "UTC", days: int = DAYS) -> list:
    """What to call each period, in the clock of ``zone`` (an IANA name): ``["Sat 7 pm to 1 am", ...]``, one per
    period of the ``days`` days from ``day0`` (00 UTC)."""
    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(zone)
    except Exception:
        tz = timezone.utc

    def clock(t: datetime) -> str:
        hour = t.hour % 12 or 12
        return f"{hour}{'' if t.minute == 0 else f':{t.minute:02d}'} {'am' if t.hour < 12 else 'pm'}"

    start = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(hours=int(day0))
    out = []
    for k in range(PERIODS * days):
        a = (start + timedelta(hours=HOURS * k)).astimezone(tz)
        b = (start + timedelta(hours=HOURS * (k + 1))).astimezone(tz)
        out.append(f"{a:%a} {clock(a)} to {clock(b)}")
    return out


__all__ = ["DAYS", "HOURS", "MODEL_WEIGHT", "PERIODS", "PRIOR_WEIGHT", "WET_MM", "at_places", "chances", "conditional",
           "fit_usual", "labels", "member_shares", "mixed", "period_amounts", "read_periods", "steps_for", "usual_timing"]
