# Copyright 2026 Nathan. Apache-2.0.
"""
When the flood peaks.

Water takes time to reach a town: the time of concentration (Kirpich 1940) ``tc = 0.0078 L^0.77 S^-0.385`` minutes
for a flow length ``L`` in feet and slope ``S`` -- for Caribbean catchments of 10-500 km2, from under an hour to about
half a day. The SCS triangular unit hydrograph turns one millimetre of runoff into a flow pulse that rises for
``tp = 0.6 tc + dt/2`` and recedes for ``1.67 tp``; convolving it with the hourly runoff of the forecast rain gives
the flow at the town, hour by hour. Its peak hour is the flood's most likely timing; its peak relative to the
catchment's size is an index of how sharp the flood would be.
"""

from __future__ import annotations

import numpy as np

from .runoff import runoff_depth


def time_of_concentration(area_km2: np.ndarray, slope: np.ndarray) -> np.ndarray:
    """Hours, from catchment area (flow length from Hack's law, ``L = 1.4 A^0.6`` km) and mean slope (m/m)."""
    length_ft = 1.4 * np.maximum(np.asarray(area_km2, np.float64), 0.1) ** 0.6 * 3280.84
    s = np.clip(np.asarray(slope, np.float64), 0.002, 1.0)
    return np.clip(0.0078 * length_ft**0.77 * s**-0.385 / 60.0, 0.25, 48.0)


def unit_hydrograph(tc_hours: float, dt: float = 1.0) -> np.ndarray:
    """Triangular unit hydrograph ordinates per ``dt`` hour, summing to 1."""
    tp = 0.6 * tc_hours + dt / 2
    tb = 2.67 * tp
    t = np.arange(0, np.ceil(tb / dt) + 1) * dt + dt / 2
    q = np.where(t <= tp, t / tp, np.clip((tb - t) / (tb - tp), 0, None))
    return q / max(q.sum(), 1e-12)


def peak_timing(hourly_rain: np.ndarray, cn: np.ndarray, area_km2: np.ndarray, slope: np.ndarray) -> dict:
    """``hourly_rain (P, T)`` catchment-mean forecast rain -> peak hour, peak runoff rate (mm/h) and total runoff (mm)
    per town."""
    rain = np.clip(np.nan_to_num(hourly_rain), 0, None)
    cum = np.cumsum(rain, -1)
    runoff_cum = runoff_depth(cum, np.asarray(cn)[:, None])
    runoff = np.diff(np.concatenate([np.zeros((rain.shape[0], 1)), runoff_cum], -1), axis=-1)
    tc = time_of_concentration(area_km2, slope)
    peak_hour = np.full(rain.shape[0], np.nan)
    peak_rate = np.zeros(rain.shape[0])
    for k in range(rain.shape[0]):
        if runoff[k].sum() <= 0:
            continue
        flow = np.convolve(runoff[k], unit_hydrograph(float(tc[k])))[: rain.shape[1]]
        peak_hour[k] = float(np.argmax(flow))
        peak_rate[k] = float(flow.max())
    return {"peak_hour": peak_hour, "peak_runoff_mm_h": peak_rate, "runoff_mm": runoff_cum[:, -1],
            "time_of_concentration_h": tc}


__all__ = ["peak_timing", "time_of_concentration", "unit_hydrograph"]
