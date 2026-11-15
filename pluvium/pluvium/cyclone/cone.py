# Copyright 2026 Nathan. Apache-2.0.
"""
How far off the track may be: a cone that widens when the systems disagree.

A forecast cone is a circle per lead that should hold the storm's centre with a stated chance (two in three, as in the
National Hurricane Center's cone). The usual cone has one radius per lead, from past errors, whatever the situation. But
some forecasts are known to be harder: when the systems' own tracks fan out, the consensus is usually further off.

:class:`ConeLearner` learns both things from verified forecasts, causally:

* the size: the ``level`` quantile of the consensus error at each lead;
* the dependence on spread: the slope of log error on log spread, pooled over leads and shrunk towards zero until
  there are enough forecasts to trust it.

The radius for a new forecast is the learned quantile, scaled by today's spread relative to the usual spread at that
lead, raised to the learned slope. With slope zero it is the ordinary static cone.

Before anything has verified the radius is a prior: the typical error for the lead, times the usual ratio between the
two-thirds quantile and the mean. As forecasts verify, the prior itself is rescaled by how the errors at all leads
compare with it, so what the short leads show early is carried to the long leads, which verify last.
"""

from __future__ import annotations

import numpy as np

from .geodesy import distance_km
from .learner import typical_km

PRIOR_RATIO = 1.1                       # two-thirds quantile over mean error, about what long error records show


class ConeLearner:
    def __init__(self, level: float = 2.0 / 3.0, pseudo: float = 12.0, slope_pseudo: float = 60.0,
                 max_slope: float = 1.0, window: int = 400) -> None:
        self.level, self.pseudo, self.slope_pseudo, self.max_slope = level, pseudo, slope_pseudo, max_slope
        self.window = window
        self.rows = {}                              # lead -> list of (error km, spread km)
        self.pending = []                           # (verify hour, lead, (lat, lon), spread km, carq)

    # ----------------------------------------------------------------------------------- learning --
    def add(self, lead: int, error_km: float, spread_km: float) -> None:
        rows = self.rows.setdefault(int(lead), [])
        rows.append((float(error_km), float(spread_km)))
        if len(rows) > self.window:
            del rows[0]

    def queue(self, init: int, points: dict, spreads: dict, carq: dict) -> None:
        """Remember a forecast (consensus ``points`` and the systems' ``spreads`` by lead) until it verifies."""
        for lead, pos in points.items():
            self.pending.append((int(init) + int(lead), int(lead), pos, float(spreads.get(lead, np.nan)), carq))

    def learn_until(self, now: int) -> int:
        """Learn from queued forecasts valid by ``now``, against the position known in real time."""
        ready = [p for p in self.pending if p[0] <= now]
        self.pending = [p for p in self.pending if p[0] > now]
        used = 0
        for verify, lead, pos, spread, carq in ready:
            fix = carq.get(int(verify), {}).get(0)
            if fix is not None and np.isfinite(pos[0]):
                self.add(lead, float(distance_km(pos[0], pos[1], fix[0], fix[1])), spread)
                used += 1
        return used

    # ------------------------------------------------------------------------------------- radius --
    def slope(self) -> float:
        """Slope of log error on log spread, pooled over leads (each lead centred on its own means), shrunk to zero."""
        xs, ys = [], []
        for rows in self.rows.values():
            a = np.array([(e, s) for e, s in rows if np.isfinite(s) and s > 0 and e > 0])
            if a.shape[0] >= 8:
                x, y = np.log(a[:, 1]), np.log(a[:, 0])
                xs.append(x - x.mean())
                ys.append(y - y.mean())
        if not xs:
            return 0.0
        x, y = np.concatenate(xs), np.concatenate(ys)
        var = float((x * x).sum())
        if var <= 0:
            return 0.0
        raw = float((x * y).sum() / var)
        return float(np.clip(raw * x.size / (x.size + self.slope_pseudo), 0.0, self.max_slope))

    def skill_ratio(self) -> float:
        """How the errors so far compare with the prior, over all leads together: the mean of (two-thirds quantile /
        prior radius), weighted by the forecasts at each lead and pulled to 1 while they are few. Short leads verify
        first and most often, so this carries what they show to the long leads, which verify last."""
        num, den = self.pseudo, self.pseudo
        for lead, rows in self.rows.items():
            if len(rows) >= 8:
                q = float(np.quantile([e for e, _ in rows], self.level))
                num += len(rows) * q / (PRIOR_RATIO * typical_km(lead))
                den += len(rows)
        return float(num / den)

    def prior(self, lead: int) -> float:
        return PRIOR_RATIO * typical_km(lead) * self.skill_ratio()

    def usual_spread(self, lead: int) -> float:
        s = [s for _, s in self.rows.get(int(lead), []) if np.isfinite(s) and s > 0]
        return float(np.exp(np.mean(np.log(s)))) if len(s) >= 8 else float("nan")

    def factor(self, lead: int, spread_km, slope: float | None = None) -> np.ndarray:
        """How much wider (above 1) or narrower the cone is for a spread, against the usual spread at that lead."""
        usual = self.usual_spread(lead)
        spread = np.asarray(spread_km, np.float64)
        slope = self.slope() if slope is None else slope
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.clip(spread / usual, 0.33, 3.0) ** slope
        return np.where(np.isfinite(f) & (spread > 0), f, 1.0)

    def base(self, lead: int) -> float:
        """The static radius at a lead: the learned quantile, pulled to the prior while forecasts are few."""
        prior = self.prior(lead)
        rows = self.rows.get(int(lead), [])
        if not rows:
            return prior
        q = float(np.quantile([e for e, _ in rows], self.level))
        n = len(rows)
        return (n * q + self.pseudo * prior) / (n + self.pseudo)

    def radius(self, lead: int, spread_km: float | None = None) -> float:
        """The cone radius (km) at a lead, for a forecast whose systems are ``spread_km`` apart: the quantile of past
        errors, each divided by its own spread factor, times today's factor."""
        rows = self.rows.get(int(lead), [])
        if spread_km is None or not np.isfinite(spread_km) or spread_km <= 0 or len(rows) < 8:
            return self.base(lead)
        slope = self.slope()
        if slope <= 0:
            return self.base(lead)
        e, s = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
        q = float(np.quantile(e / self.factor(lead, s, slope), self.level))
        n = len(rows)
        return float((n * q + self.pseudo * self.prior(lead)) / (n + self.pseudo) * self.factor(lead, spread_km, slope))

    def rms(self, lead: int) -> float:
        """Root-mean-square consensus error at a lead so far (the prior's equivalent before any forecast)."""
        rows = self.rows.get(int(lead), [])
        prior = 1.15 * typical_km(lead) * self.skill_ratio()     # rms over mean, for errors that are roughly Rayleigh
        if not rows:
            return prior
        n = len(rows)
        return float(np.sqrt((sum(e * e for e, _ in rows) + self.pseudo * prior ** 2) / (n + self.pseudo)))


    def state(self) -> dict:
        """The verified forecasts so far, as plain lists (forecasts still waiting to verify are not included)."""
        return {"level": self.level, "rows": {str(lead): [[e, s if np.isfinite(s) else None] for e, s in rows]
                                              for lead, rows in self.rows.items()}}

    def load_state(self, state: dict) -> None:
        self.rows = {int(lead): [(float(e), float("nan") if s is None else float(s)) for e, s in rows]
                     for lead, rows in state["rows"].items()}


def coverage(radii: np.ndarray, errors: np.ndarray) -> float:
    """Share of verified forecasts whose error was inside the radius given for them."""
    ok = np.isfinite(radii) & np.isfinite(errors)
    return float(np.mean(errors[ok] <= radii[ok])) if ok.any() else float("nan")


__all__ = ["PRIOR_RATIO", "ConeLearner", "coverage"]
