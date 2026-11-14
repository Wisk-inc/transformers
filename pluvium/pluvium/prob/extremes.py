# Copyright 2026 Nathan. Apache-2.0.
"""
The tail: cloudbursts and their return periods.

A lognormal fitted to everyday rain underestimates the rare extreme hour: real hourly rain has a heavier tail.
Extreme-value theory says exceedances of a high threshold ``u`` follow a generalised Pareto distribution (Pickands
1975): ``P(X > x | X > u) = (1 + xi (x - u) / sigma)^(-1/xi)``. Fitted by probability-weighted moments (Hosking and
Wallis 1987) to the satellite record of each region, it gives:

* the chance that a heavy hour turns torrential, from data instead of the lognormal's shape;
* the return period of a forecast amount -- how often, on average, this place sees an hour (or a day) this wet:
  ``ARI(x) = 1 / (lambda P(X > x | X > u))`` years, with ``lambda`` the yearly rate of exceeding ``u``. "A 1-in-10-year
  downpour" is a sentence emergency managers act on.
"""

from __future__ import annotations

import numpy as np


def fit_gpd(exceedances: np.ndarray) -> tuple[float, float]:
    """``(xi, sigma)`` of a generalised Pareto fitted to positive exceedances by probability-weighted moments."""
    y = np.sort(np.asarray(exceedances, np.float64))
    n = y.size
    if n < 10:
        return 0.1, float(max(y.mean(), 1e-3)) if n else 1.0
    b0 = y.mean()
    b1 = np.sum((np.arange(n) / (n - 1)) * y) / n                    # E[X F(X)]
    # Hosking & Wallis: a0 = b0, a1 = b0 - b1 (moments of 1 - F), xi = 2 - a0 / (a0 - 2 a1), sigma = 2 a0 a1 / (a0 - 2 a1)
    a0, a1 = b0, b0 - b1
    denom = a0 - 2 * a1
    if abs(denom) < 1e-9:
        return 0.0, float(a0)
    k = a0 / denom - 2.0                                              # Hosking's k = -xi
    sigma = 2 * a0 * a1 / denom
    return float(np.clip(-k, -0.4, 0.6)), float(max(sigma, 1e-3))


class GPDTail:
    def __init__(self, threshold: float, xi: float, sigma: float, rate_per_year: float) -> None:
        self.u, self.xi, self.sigma, self.rate = threshold, xi, sigma, rate_per_year

    def exceed(self, x) -> np.ndarray:
        """``P(X > x | X > u)``."""
        z = np.maximum(np.asarray(x, np.float64) - self.u, 0) / self.sigma
        if abs(self.xi) < 1e-6:
            return np.exp(-z)
        return np.clip(1 + self.xi * z, 0, None) ** (-1 / self.xi)

    def return_period_years(self, x) -> np.ndarray:
        return 1.0 / np.maximum(self.rate * self.exceed(x), 1e-9)

    def level(self, years: float) -> float:
        """The amount exceeded on average once in ``years``."""
        p = 1.0 / (years * self.rate)
        if abs(self.xi) < 1e-6:
            return float(self.u - self.sigma * np.log(p))
        return float(self.u + self.sigma / self.xi * (p ** (-self.xi) - 1))


class RainfallTails:
    """GPD tails of hourly rates and 24-hour totals per region, from a record ``(hours, points)`` in mm/h."""

    def __init__(self) -> None:
        self.tails: dict[tuple[str, str], GPDTail] = {}

    def fit(self, record: np.ndarray, regions: np.ndarray, years: float, hourly_u: float = 20.0,
            daily_u: float = 60.0) -> RainfallTails:
        record = np.asarray(record, np.float64)
        daily = np.add.reduceat(np.nan_to_num(record), np.arange(0, record.shape[0], 24), axis=0)
        for g in np.unique(regions):
            cols = regions == g
            for name, data, u in (("hour", record[:, cols], hourly_u), ("day", daily[:, cols], daily_u)):
                x = data[np.isfinite(data)]
                exc = x[x > u] - u
                rate = exc.size / max(years * cols.sum(), 1e-9)               # per point per year
                xi, sigma = fit_gpd(exc)
                self.tails[(str(g), name)] = GPDTail(u, xi, sigma, rate)
        return self

    def return_period(self, amount: np.ndarray, regions: np.ndarray, kind: str = "hour") -> np.ndarray:
        out = np.full(np.shape(amount), np.inf)
        for g in np.unique(regions):
            tail = self.tails.get((str(g), kind))
            if tail is not None:
                cols = regions == g
                out[cols] = np.where(amount[cols] > tail.u, tail.return_period_years(amount[cols]), np.inf)
        return out


__all__ = ["GPDTail", "RainfallTails", "fit_gpd"]
