# Copyright 2026 Nathan. Apache-2.0.
"""
The convective environment: how ready the air is to make thunderstorms, from the driving model's temperature,
humidity and wind at 850, 700, 500 and 200 hPa (the forecaster's classic instability and moisture indices).

Global AI weather models are known to smear tropical rain towards a mean (the "double penalty" of training on mean
squared error) and to misplace its daily cycle; a forecaster in Kingston instead reads the instability, the mid-level
moisture and the upper-level support, which say whether the afternoon will produce storms whatever the coarse model's
rain field shows. These indices give the network the same reading:

* **K-index** ``(T850 - T500) + Td850 - (T700 - Td700)``: lapse rate plus low-level moisture minus mid-level dryness
  (above about 30 C thunderstorms are likely in the tropics);
* **total totals** ``T850 + Td850 - 2 T500``;
* **convective instability** ``theta-e(850) - theta-e(500)``;
* **lifted index** at 500 hPa for a parcel lifted from 850 hPa (negative: unstable);
* **700 hPa relative humidity**: dry mid-levels (the Saharan air layer in summer) choke Caribbean showers;
* **850 hPa moisture-flux convergence** (g/kg per day): air and water piling in at low levels;
* **850-200 hPa wind shear**: organisation of storms, and the shear that tears tropical systems apart;
* **200 hPa divergence**: upper-level outflow that sustains deep convection.
"""

from __future__ import annotations

import numpy as np

from .features import _grad, theta_e

CHANNELS = ("k_index", "total_totals", "theta_e_diff", "lifted_index", "rh700", "mfc850", "shear_850_200", "div200")
NEEDS = ("t850", "q850", "t700", "q700", "t500", "q500", "u850", "v850", "u200", "v200")


def vapour_pressure(q: np.ndarray, p_hpa: float) -> np.ndarray:
    return q * p_hpa / (0.622 + 0.378 * q)


def saturation_pressure(t_k: np.ndarray) -> np.ndarray:
    tc = t_k - 273.15
    return 6.112 * np.exp(17.67 * tc / (tc + 243.5))


def dewpoint_c(q: np.ndarray, p_hpa: float) -> np.ndarray:
    e = np.maximum(vapour_pressure(q, p_hpa), 1e-3)
    x = np.log(e / 6.112)
    return 243.5 * x / (17.67 - x)


def saturation_q(t_k: np.ndarray, p_hpa: float) -> np.ndarray:
    es = saturation_pressure(t_k)
    return 0.622 * es / np.maximum(p_hpa - 0.378 * es, 1.0)


def parcel_temperature(theta_e_parcel: np.ndarray, p_hpa: float, iterations: int = 24) -> np.ndarray:
    """Temperature (K) at ``p_hpa`` of a saturated parcel with the given theta-e (bisection; theta-e of a saturated
    parcel increases with temperature)."""
    lo = np.full(theta_e_parcel.shape, 200.0)
    hi = np.full(theta_e_parcel.shape, 330.0)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        th = theta_e(mid, saturation_q(mid, p_hpa), p_hpa)
        lo = np.where(th < theta_e_parcel, mid, lo)
        hi = np.where(th < theta_e_parcel, hi, mid)
    return 0.5 * (lo + hi)


def convective_channels(upper: dict, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """``(F, len(CHANNELS), H, W)`` normalised to order one, from upper-air fields ``(F, H, W)`` on the grid."""
    t850, q850 = upper["t850"], np.clip(upper["q850"], 1e-6, None)
    t700, q700 = upper["t700"], np.clip(upper["q700"], 1e-6, None)
    t500, q500 = upper["t500"], np.clip(upper["q500"], 1e-7, None)
    td850, td700 = dewpoint_c(q850, 850.0), dewpoint_c(q700, 700.0)
    c850, c700, c500 = t850 - 273.15, t700 - 273.15, t500 - 273.15
    k_index = (c850 - c500) + td850 - (c700 - td700)
    total_totals = c850 + td850 - 2 * c500
    the850, the500 = theta_e(t850, q850, 850.0), theta_e(t500, q500, 500.0)
    lifted = t500 - parcel_temperature(the850, 500.0)
    rh700 = np.clip(vapour_pressure(q700, 700.0) / saturation_pressure(t700), 0, 1.2)
    qu, qv = q850 * 1000.0 * upper["u850"], q850 * 1000.0 * upper["v850"]
    gx, _ = _grad(qu, lat, lon)
    _, gy = _grad(qv, lat, lon)
    mfc = -(gx + gy) * 0.864                                  # (g/kg)(m/s) per 100 km -> g/kg per day
    shear = np.hypot(upper["u200"] - upper["u850"], upper["v200"] - upper["v850"])
    ux, _ = _grad(upper["u200"], lat, lon)
    _, vy = _grad(upper["v200"], lat, lon)
    div200 = ux + vy                                          # 1e-5 per second
    out = [(k_index - 30.0) / 10.0, (total_totals - 44.0) / 5.0, (the850 - the500 - 15.0) / 10.0, lifted / 3.0,
           (rh700 - 0.6) / 0.25, np.clip(mfc / 20.0, -5, 5), (shear - 12.0) / 8.0, np.clip(div200 / 2.0, -5, 5)]
    return np.nan_to_num(np.stack(out, axis=1)).astype(np.float32)


__all__ = ["CHANNELS", "NEEDS", "convective_channels", "dewpoint_c", "parcel_temperature", "saturation_q"]
