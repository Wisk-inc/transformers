# Copyright 2026 Nathan. Apache-2.0.
"""
How rare is this amount, here? Return periods from the daily record.

"80 mm" means a wet day in one town and a disaster in another. A return period puts an amount in the place's own
terms: 80 mm is a day this place sees about once in six years.

The record at one place is short (a year or a few), so the fit is regional (the index-flood idea of Dalrymple, 1960):

* each place keeps its own **threshold** (a high quantile of its daily rain) and its own **mean excess** over it,
  shrunk towards the area's when it has few days above the threshold;
* the **shape** of the tail is one number for the whole area, fitted to every place's excesses after dividing by the
  place's mean excess (:func:`pluvium.tails.gpd.fit_pwm`).

A place's return period for an amount is then one over the yearly rate of days above it. Two limits are stated with
every answer and built into the wording: below the threshold the amount is ordinary and no period is given, and
beyond ``horizon`` years (four times the length of the record) the answer is "rarer than once in that many years"
and nothing more exact, because nothing in the record can check it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import gpd

DAYS_PER_YEAR = 365.25


@dataclass
class ReturnLevels:
    threshold: np.ndarray           # (n,) mm: above this a day counts as heavy at the place
    scale: np.ndarray               # (n,) scale of the excess at the place
    rate: np.ndarray                # (n,) days a year above the threshold
    shape: float                    # the tail's shape, one for the area
    years: float                    # length of the record

    @property
    def horizon(self) -> float:
        """Return periods beyond this many years are not told apart: four times the record."""
        return 4.0 * self.years

    @classmethod
    def fit(cls, daily: np.ndarray, quantile: float = 0.98, min_threshold: float = 5.0, pseudo: float = 5.0) -> ReturnLevels:
        """From daily rain ``(days, n)`` in mm (NaN where missing). ``quantile`` sets each place's threshold,
        ``min_threshold`` keeps it meaningful in a dry place, ``pseudo`` is how many excesses the area's mean excess
        is worth when a place's own is shrunk towards it."""
        daily = np.asarray(daily, np.float32)                         # a year of a large area: kept in single precision
        ok = np.isfinite(daily)
        counts = ok.sum(0)
        order = np.sort(np.where(ok, daily, np.float32(-1.0)), axis=0)   # missing days first, then ascending rain
        k = np.clip((daily.shape[0] - counts) + np.floor(quantile * counts).astype(int), 0, daily.shape[0] - 1)
        threshold = np.maximum(order[k, np.arange(daily.shape[1])], min_threshold).astype(np.float64)
        del order
        excess = np.where(ok & (daily > threshold), daily - threshold, np.nan).astype(np.float32)
        n_excess = np.isfinite(excess).sum(0)
        own = np.where(n_excess > 0, np.nansum(excess, 0, dtype=np.float64) / np.maximum(n_excess, 1), np.nan)
        usable = n_excess >= 3
        ratio = float(np.nanmedian(own[usable] / threshold[usable])) if usable.any() else 0.5
        regional = ratio * threshold                                  # what a place like this one usually has
        mean = (n_excess * np.where(n_excess > 0, own, 0.0) + pseudo * regional) / (n_excess + pseudo)
        pooled = (excess / mean)[:, usable]
        shape, _ = gpd.fit_pwm(pooled[np.isfinite(pooled)]) if usable.any() else (0.0, 1.0)
        years = float(np.median(counts)) / DAYS_PER_YEAR
        rate = np.maximum(n_excess, 0.5) / np.maximum(counts, 1) * DAYS_PER_YEAR
        return cls(threshold.astype(np.float32), (mean * (1.0 - shape)).astype(np.float32), rate.astype(np.float32),
                   float(shape), years)

    # ----------------------------------------------------------------------------------- answers --
    def period(self, amount: np.ndarray, index=None) -> np.ndarray:
        """Return period in years of ``amount`` mm at each place (``index`` picks places). Zero below the place's
        threshold: an ordinary amount. Not capped; see :meth:`wording` and :attr:`horizon`."""
        pick = slice(None) if index is None else index
        u, s, r = self.threshold[pick], self.scale[pick], self.rate[pick]
        amount = np.asarray(amount, np.float64)
        yearly = r * gpd.survival(amount - u, self.shape, s)
        return np.where(amount > u, 1.0 / np.maximum(yearly, 1e-12), 0.0)

    def level(self, years: float, index=None) -> np.ndarray:
        """The amount, mm, with a return period of ``years`` at each place."""
        pick = slice(None) if index is None else index
        u, s, r = self.threshold[pick], self.scale[pick], self.rate[pick]
        p = np.clip(1.0 - 1.0 / np.maximum(years * r, 1.0 + 1e-9), 0.0, 1.0)
        return u + gpd.quantile(p, self.shape, s)

    def wording(self, period: float) -> str:
        """A return period in words, with the record's limit built in."""
        period = float(period)
        if not np.isfinite(period) or period < 0.25:
            return ""
        if period < 0.75:
            return "a few times a year"
        if period < 1.5:
            return "about once a year"
        if period >= self.horizon:
            return f"rarer than once in {max(int(round(self.horizon)), 2)} years"
        return f"about once in {int(round(period))} years"

    # ---------------------------------------------------------------------------------------- io --
    def save(self, path) -> Path:
        path = Path(path)
        np.savez_compressed(path, threshold=self.threshold, scale=self.scale, rate=self.rate,
                            shape=np.float64(self.shape), years=np.float64(self.years))
        return path

    @classmethod
    def load(cls, path) -> ReturnLevels:
        z = np.load(path)
        return cls(z["threshold"], z["scale"], z["rate"], float(z["shape"]), float(z["years"]))


__all__ = ["DAYS_PER_YEAR", "ReturnLevels"]
