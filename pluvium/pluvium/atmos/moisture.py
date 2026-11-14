# Copyright 2026 Nathan. Apache-2.0.
"""
The column's water, and the most robust law of tropical rain.

Over tropical oceans, rain rate rises roughly *exponentially* with column relative humidity -- the precipitable
water divided by its saturated value (Bretherton, Peters and Back 2004: ``P = exp(a (r - b))`` with ``a = 15.6``,
``b = 0.603``, in mm/day). A coarse driving model can put the right amount of water in the column and still rain
in the wrong place; giving the network this law as an input lets it place rain where the column is near saturation.

Also here: the vertically integrated moisture-flux convergence from the 850/700/500 hPa winds and humidities (where
water is being imported faster than it rains out), and the precipitation efficiency the driving model implies.
"""

from __future__ import annotations

import numpy as np

from .thermo import saturation_vapour_pressure

G = 9.80665
#: layer thicknesses (hPa) the three levels stand for in column integrals
WEIGHTS = {850: 225.0, 700: 175.0, 500: 250.0}


def saturated_q(t_k: np.ndarray, p_hpa: float) -> np.ndarray:
    es = saturation_vapour_pressure(t_k)
    return 0.622 * es / np.maximum(p_hpa - 0.378 * es, 1.0)


def column_relative_humidity(t: dict, q: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(precipitable water mm, saturated precipitable water mm, their ratio)`` from 850/700/500 hPa."""
    pw = sum(WEIGHTS[p] * 100.0 * np.clip(q[p], 0, 0.04) for p in WEIGHTS) / G
    pws = sum(WEIGHTS[p] * 100.0 * saturated_q(t[p], float(p)) for p in WEIGHTS) / G
    return pw, pws, np.clip(pw / np.maximum(pws, 1e-3), 0, 1.2)


def bretherton_rate(crh: np.ndarray) -> np.ndarray:
    """Expected rain (mm/h) at a column relative humidity (Bretherton et al. 2004, converted from mm/day)."""
    return np.exp(15.6 * (np.asarray(crh) - 0.603)) / 24.0


def vimfc(u: dict, v: dict, q: dict, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Vertically integrated moisture-flux convergence (mm/day) from the three lower levels."""
    from ..features import _grad

    total = 0.0
    for p, w in WEIGHTS.items():
        qu, qv = q[p] * u[p], q[p] * v[p]
        gx, _ = _grad(qu, lat, lon)
        _, gy = _grad(qv, lat, lon)
        total = total - (gx + gy) * 1e-5 * w * 100.0 / G             # per 100 km -> per metre; kg m-2 s-1
    return total * 86400.0


def moisture_channels(upper: dict, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """``(F, 3, H, W)``: column relative humidity, the Bretherton rain rate (log1p), and VIMFC (normalised)."""
    t = {p: upper[f"t{p}"] for p in WEIGHTS}
    q = {p: upper[f"q{p}"] for p in WEIGHTS}
    u = {p: upper[f"u{p}"] for p in WEIGHTS}
    v = {p: upper[f"v{p}"] for p in WEIGHTS}
    _, _, crh = column_relative_humidity(t, q)
    conv = vimfc(u, v, q, lat, lon)
    return np.stack([(crh - 0.7) / 0.15, np.log1p(bretherton_rate(crh)), np.clip(conv / 20.0, -5, 5)], 1).astype(np.float32)


__all__ = ["bretherton_rate", "column_relative_humidity", "moisture_channels", "saturated_q", "vimfc"]
