# Copyright 2026 Nathan. Apache-2.0.
"""
Moist thermodynamics, done properly.

Version 2 used the textbook shortcut ``theta-e = (T + L q / c_p)(1000/p)^0.286`` and a lifted index from 850 hPa.
Tropical convection lives in differences of a few kelvin, where the shortcut's error (1-3 K) is as large as the
signal. Here:

* saturation vapour pressure over water (Bolton 1980, eq. 10), mixing ratio, virtual temperature;
* equivalent potential temperature by Bolton's eq. 39 (accurate to about 0.02 K);
* the lifting condensation level by Bolton's eq. 21;
* a pseudo-adiabatic parcel ascent integrated in log-pressure, and the convective available potential energy and
  convective inhibition of a parcel lifted from 850 hPa through the 850/700/500/200 hPa profile (log-pressure
  interpolation of the environment). Four levels cannot resolve every inversion; the numbers are indices of the
  column's readiness, not soundings, and are clipped to physically sensible ranges.
"""

from __future__ import annotations

import numpy as np

RD, RV, CP, G, LV = 287.04, 461.5, 1005.7, 9.80665, 2.501e6
EPS = RD / RV
LEVELS = (850.0, 700.0, 500.0, 200.0)


def saturation_vapour_pressure(t_k: np.ndarray) -> np.ndarray:
    """hPa, over liquid water (Bolton 1980, eq. 10)."""
    tc = np.asarray(t_k, np.float64) - 273.15
    return 6.112 * np.exp(17.67 * tc / (tc + 243.5))


def mixing_ratio(e_hpa: np.ndarray, p_hpa) -> np.ndarray:
    return EPS * e_hpa / np.maximum(np.asarray(p_hpa, np.float64) - e_hpa, 1e-3)


def vapour_pressure_from_q(q: np.ndarray, p_hpa) -> np.ndarray:
    q = np.clip(np.asarray(q, np.float64), 1e-7, 0.05)
    return q * p_hpa / (EPS + (1 - EPS) * q)


def dewpoint_k(e_hpa: np.ndarray) -> np.ndarray:
    x = np.log(np.maximum(e_hpa, 1e-4) / 6.112)
    return 243.5 * x / (17.67 - x) + 273.15


def lcl(t_k: np.ndarray, td_k: np.ndarray) -> np.ndarray:
    """Temperature (K) at the lifting condensation level (Bolton 1980, eq. 15)."""
    t, td = np.asarray(t_k, np.float64), np.asarray(td_k, np.float64)
    return 1.0 / (1.0 / (td - 56.0) + np.log(t / td) / 800.0) + 56.0


def theta_e_bolton(t_k: np.ndarray, q: np.ndarray, p_hpa) -> np.ndarray:
    """Equivalent potential temperature (K), Bolton 1980 eq. 39."""
    t = np.asarray(t_k, np.float64)
    e = vapour_pressure_from_q(q, p_hpa)
    r = mixing_ratio(e, p_hpa)                                       # kg/kg
    tl = lcl(t, dewpoint_k(e))
    theta_l = t * (1000.0 / (p_hpa - e)) ** 0.2854 * (t / tl) ** (0.28e-3 * r * 1000.0)
    return theta_l * np.exp((3.036 / tl - 0.00178) * r * 1000.0 * (1 + 0.448e-3 * r * 1000.0))


def _moist_lapse(t: np.ndarray, p_hpa: np.ndarray) -> np.ndarray:
    """dT/dlnp of a saturated parcel (pseudo-adiabatic), K per unit ln p."""
    es = saturation_vapour_pressure(t)
    rs = mixing_ratio(es, p_hpa)
    num = (RD * t + LV * rs) / (CP + (LV**2 * rs * EPS) / (RD * t**2))
    return num


def parcel_profile(t0: np.ndarray, q0: np.ndarray, p0: float, levels) -> np.ndarray:
    """Temperatures of a parcel lifted from ``p0`` (dry to its LCL, then moist), at ``levels`` (hPa, decreasing)."""
    t0 = np.asarray(t0, np.float64)
    e0 = vapour_pressure_from_q(q0, p0)
    t_lcl = lcl(t0, dewpoint_k(e0))
    p_lcl = p0 * (t_lcl / t0) ** (1 / 0.2854)
    out = []
    for p in levels:
        dry = t0 * (p / p0) ** 0.2854
        # moist ascent from the LCL to p in log-pressure steps
        t = t_lcl.copy()
        lp0, lp1 = np.log(np.minimum(p_lcl, p0)), np.log(p)
        n = 12
        for k in range(n):
            lp = lp0 + (lp1 - lp0) * (k + 0.5) / n
            t = t + _moist_lapse(t, np.exp(lp)) * (lp1 - lp0) / n
        out.append(np.where(p >= p_lcl, dry, t))
    return np.stack(out)


def cape_cin(t: dict, q: dict, origin: float = 850.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CAPE and CIN (J/kg) of a parcel from ``origin`` through the profile, and the LCL height (m), from temperatures
    ``t[p]`` and specific humidities ``q[p]`` at 850/700/500/200 hPa (arrays of any shape). The integral is a
    trapezoid in ln p on a 16-level grid between 850 and 200 hPa (environment interpolated linearly in ln p).
    """
    grid = np.exp(np.linspace(np.log(origin), np.log(200.0), 16))
    lev = np.log(np.array(LEVELS))
    env_t = np.stack([t[int(p)] for p in LEVELS]).astype(np.float64)
    env_q = np.stack([q[int(p)] for p in LEVELS]).astype(np.float64)

    def interp(field, lp):
        k = np.clip(np.searchsorted(-lev, -lp) - 1, 0, len(LEVELS) - 2)
        w = (lp - lev[k]) / (lev[k + 1] - lev[k])
        return field[k] * (1 - w) + field[k + 1] * w

    parcel = parcel_profile(t[int(origin)], q[int(origin)], origin, grid)
    env = np.stack([interp(env_t, np.log(p)) for p in grid])
    envq = np.stack([interp(env_q, np.log(p)) for p in grid])
    tv_env = env * (1 + 0.61 * envq)
    es = saturation_vapour_pressure(parcel)
    qs = EPS * es / np.maximum(grid.reshape((-1,) + (1,) * (parcel.ndim - 1)) - (1 - EPS) * es, 1e-3)
    tv_par = parcel * (1 + 0.61 * np.minimum(qs, q[int(origin)]))
    buoy = RD * (tv_par - tv_env) / tv_env * tv_env                 # R_d (Tv_p - Tv_e)
    dlnp = -np.diff(np.log(grid))
    seg = 0.5 * (buoy[1:] + buoy[:-1]) * dlnp.reshape((-1,) + (1,) * (buoy.ndim - 1))
    cape = np.clip(np.where(seg > 0, seg, 0).sum(0), 0, 6000)
    # CIN: negative area below the first positive layer
    positive = np.cumsum(seg > 0, axis=0) > 0
    cin = np.clip(np.where((seg < 0) & ~positive, seg, 0).sum(0), -500, 0)
    e0 = vapour_pressure_from_q(q[int(origin)], origin)
    t_lcl = lcl(t[int(origin)], dewpoint_k(e0))
    lcl_m = np.clip((t[int(origin)] - t_lcl) / 0.0098, 0, 4000)      # dry adiabatic metres above the origin level
    return cape.astype(np.float32), cin.astype(np.float32), lcl_m.astype(np.float32)


__all__ = ["cape_cin", "dewpoint_k", "lcl", "mixing_ratio", "parcel_profile", "saturation_vapour_pressure",
           "theta_e_bolton", "vapour_pressure_from_q"]
