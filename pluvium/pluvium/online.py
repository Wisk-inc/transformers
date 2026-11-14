# Copyright 2026 Nathan. Apache-2.0.
"""
Learning from its own mistakes, live.

Every forecast is kept (:class:`ForecastArchive`: the hourly chances of rain at every airport and the parish each one
belongs to). When the hours it forecast have passed, :func:`learn` reads what the airports reported in those hours
(:mod:`pluvium.sources.metar`) and updates a small **online corrector**: for each parish, lead-time band and part of
the day, a logistic recalibration ``p' = sigmoid(a * logit(p) + b)`` of the chance of rain, and a multiplier on the
expected amounts, trained by stochastic gradient descent on the log-loss of the verified hours. A pull back towards
"no correction" (``a = 1``, ``b = 0``) at every step means a parish needs repeated evidence before its forecasts move,
and old errors fade as new ones arrive -- so a model that has started to over-forecast afternoon showers in St. Ann
this month is corrected within days, and the correction relaxes when the pattern ends.

What it cannot do: correct a parish with no reporting airport (it borrows its country's correction), or fix errors
in situations that have not happened recently. The model's own weights are retrained by :mod:`pluvium.program`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .calibrate import LEAD_BANDS

DAY_PARTS = ((0, 6), (6, 12), (12, 18), (18, 24))      # local solar hours
WINDOW_KEYS = (6, 12, 24, 48, 72)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sig(x):
    return 1 / (1 + np.exp(-x))


def lead_band(hours: np.ndarray) -> np.ndarray:
    out = np.zeros(np.shape(hours), np.int64)
    for k, (a, b) in enumerate(LEAD_BANDS):
        out[(hours >= a) & (hours < b)] = k
    return out


def day_part(local_hour: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(local_hour) // 6).astype(np.int64), 0, 3)


class OnlineCorrector:
    """Per-group logistic recalibration, learned from verified forecasts. Groups are names (``"JM/St. Ann"``); a
    group never verified uses its country's correction, and a country never verified uses none."""

    def __init__(self, lr: float = 0.05, anchor: float = 0.02) -> None:
        self.lr, self.anchor = lr, anchor
        self.params: dict[str, np.ndarray] = {}          # group -> (bands, parts, 2): a, b
        self.windows: dict[str, np.ndarray] = {}         # group -> (len(WINDOW_KEYS), 2)
        self.amount: dict[str, float] = {}               # group -> log multiplier
        self.seen: dict[str, int] = {}
        self.updated = 0.0

    # ----------------------------------------------------------------------------------- applying --
    def _lookup(self, group: str, table: dict, default):
        if group in table:
            return table[group]
        country = group.split("/")[0]
        return table.get(country, default)

    def apply_hourly(self, p: np.ndarray, groups, lon: np.ndarray | None = None, t0: int = 0) -> np.ndarray:
        """``p (P, T)`` hourly chances for places in ``groups``; returns corrected chances."""
        p = np.asarray(p, np.float64)
        out = p.copy()
        if not self.params:
            return out
        t = p.shape[-1]
        bands = lead_band(np.arange(t))
        lst = ((t0 + np.arange(t) + 0.5)[None, :] + (np.zeros(p.shape[0]) if lon is None else np.asarray(lon))[:, None]
               / 15.0) % 24
        parts = day_part(lst)
        for k, g in enumerate(groups):
            ab = self._lookup(str(g), self.params, None)
            if ab is None:
                continue
            a, b = ab[bands, parts[k], 0], ab[bands, parts[k], 1]
            out[k] = _sig(a * _logit(p[k]) + b)
        return out

    def apply_window(self, p: np.ndarray, groups, hours: int) -> np.ndarray:
        if not self.windows or hours not in WINDOW_KEYS:
            return p
        i = WINDOW_KEYS.index(hours)
        out = np.asarray(p, np.float64).copy()
        for k, g in enumerate(groups):
            ab = self._lookup(str(g), self.windows, None)
            if ab is not None:
                out[k] = _sig(ab[i, 0] * _logit(out[k]) + ab[i, 1])
        return out

    def amount_factor(self, groups) -> np.ndarray:
        return np.array([np.exp(self._lookup(str(g), self.amount, 0.0)) for g in groups])

    # ----------------------------------------------------------------------------------- learning --
    def update(self, p: np.ndarray, observed: np.ndarray, groups, lon: np.ndarray, t0: int,
               observed_mm: np.ndarray | None = None, expected_mm: np.ndarray | None = None, steps: int = 3) -> int:
        """One verified forecast: ``p (P, T)`` the chances it gave (before this corrector), ``observed (P, T)`` 1/0 or
        NaN. Each group (and its country) takes a few gradient steps on its log-loss. Returns hours used."""
        p = np.asarray(p, np.float64)
        obs = np.asarray(observed, np.float64)
        t = p.shape[-1]
        bands = lead_band(np.arange(t))
        lst = ((t0 + np.arange(t) + 0.5)[None, :] + np.asarray(lon)[:, None] / 15.0) % 24
        parts = day_part(lst)
        used = 0
        for k, g in enumerate(groups):
            ok = np.isfinite(obs[k])
            if not ok.any():
                continue
            for name in {str(g), str(g).split("/")[0]}:
                ab = self.params.setdefault(name, np.stack([np.ones((len(LEAD_BANDS), 4)),
                                                            np.zeros((len(LEAD_BANDS), 4))], -1))
                x = _logit(p[k, ok])
                y = obs[k, ok]
                bi, pi = bands[ok], parts[k, ok]
                for _ in range(steps):
                    q = _sig(ab[bi, pi, 0] * x + ab[bi, pi, 1])
                    err = q - y
                    ga = np.zeros((len(LEAD_BANDS), 4))
                    gb = np.zeros((len(LEAD_BANDS), 4))
                    np.add.at(ga, (bi, pi), err * x)
                    np.add.at(gb, (bi, pi), err)
                    cnt = np.zeros((len(LEAD_BANDS), 4))
                    np.add.at(cnt, (bi, pi), 1.0)
                    cnt = np.maximum(cnt, 1.0)
                    ab[..., 0] -= self.lr * (ga / cnt + self.anchor * (ab[..., 0] - 1.0))
                    ab[..., 1] -= self.lr * (gb / cnt + self.anchor * ab[..., 1])
                ab[..., 0] = np.clip(ab[..., 0], 0.3, 3.0)
                ab[..., 1] = np.clip(ab[..., 1], -3.0, 3.0)
                # windows: any rain in the next h hours
                wab = self.windows.setdefault(name, np.stack([np.ones(len(WINDOW_KEYS)), np.zeros(len(WINDOW_KEYS))], -1))
                for i, h in enumerate(WINDOW_KEYS):
                    if h > t or not np.isfinite(obs[k, :h]).all():
                        continue
                    pw = 1 - np.prod(1 - p[k, :h])
                    yw = float(obs[k, :h].max())
                    for _ in range(steps):
                        q = _sig(wab[i, 0] * _logit(pw) + wab[i, 1])
                        wab[i, 0] -= self.lr * ((q - yw) * _logit(pw) + self.anchor * (wab[i, 0] - 1.0))
                        wab[i, 1] -= self.lr * ((q - yw) + self.anchor * wab[i, 1])
                if observed_mm is not None and expected_mm is not None:
                    o, e = np.nansum(observed_mm[k]), np.nansum(expected_mm[k])
                    if e > 0.5 or o > 0.5:
                        c = self.amount.get(name, 0.0)
                        self.amount[name] = float(np.clip(c + 0.05 * (np.log1p(o) - np.log1p(e)) - self.anchor * c, -1, 1))
                self.seen[name] = self.seen.get(name, 0) + int(ok.sum())
            used += int(ok.sum())
        self.updated = time.time()
        return used

    # ------------------------------------------------------------------------------------- saving --
    def save(self, path) -> None:
        Path(path).write_text(json.dumps({
            "lr": self.lr, "anchor": self.anchor, "updated": self.updated, "seen": self.seen, "amount": self.amount,
            "params": {k: v.tolist() for k, v in self.params.items()},
            "windows": {k: v.tolist() for k, v in self.windows.items()}}))

    @classmethod
    def load(cls, path) -> OnlineCorrector:
        path = Path(path)
        c = cls()
        if not path.exists():
            return c
        d = json.loads(path.read_text())
        c.lr, c.anchor, c.updated = d.get("lr", c.lr), d.get("anchor", c.anchor), d.get("updated", 0.0)
        c.seen, c.amount = d.get("seen", {}), d.get("amount", {})
        c.params = {k: np.array(v) for k, v in d.get("params", {}).items()}
        c.windows = {k: np.array(v) for k, v in d.get("windows", {}).items()}
        return c

    def describe(self) -> dict:
        out = {}
        for g, ab in self.params.items():
            out[g] = {"hours_verified": self.seen.get(g, 0), "mean_slope": float(ab[..., 0].mean()),
                      "mean_bias": float(ab[..., 1].mean()), "amount_factor": float(np.exp(self.amount.get(g, 0.0)))}
        return out


class ForecastArchive:
    """Each forecast's raw hourly chances at the airports, kept until verified."""

    def __init__(self, root) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, t0: int, p_wet: np.ndarray, expected_mm: np.ndarray, groups, lon: np.ndarray) -> Path:
        path = self.root / f"forecast_{int(t0)}.npz"
        np.savez_compressed(path, t0=int(t0), p_wet=p_wet.astype(np.float32), expected_mm=expected_mm.astype(np.float32),
                            groups=np.asarray(groups, dtype=str), lon=np.asarray(lon, np.float64), learned_until=int(t0))
        return path

    def pending(self, now_hour: int) -> list[Path]:
        return sorted(p for p in self.root.glob("forecast_*.npz") if int(np.load(p)["learned_until"]) < now_hour)


def learn(archive: ForecastArchive, corrector: OnlineCorrector, station_codes_for, now_hour: int,
          keep_days: float = 4.0) -> dict:
    """
    Verify every archived forecast against the airport reports of the hours that have passed since it was last
    checked, and update the corrector. ``station_codes_for(hours) -> (len(hours), S)`` packed codes (live METAR, or
    the staged archive in hindcasts).
    """
    from .weather import decode

    used, forecasts = 0, 0
    for path in archive.pending(now_hour):
        z = dict(np.load(path))
        t0, done = int(z["t0"]), int(z["learned_until"])
        t = z["p_wet"].shape[1]
        hours = np.arange(max(done, t0), min(now_hour, t0 + t))
        if hours.size == 0:
            continue
        d = decode(station_codes_for(hours))                                   # (H, S)
        obs = np.full(z["p_wet"].shape, np.nan)
        cols = hours - t0
        obs[:, cols] = np.where(d["known"].T, d["wet"].T.astype(float), np.nan)
        mask = np.zeros(t, bool)
        mask[cols] = True
        p = np.where(mask[None], z["p_wet"], np.nan)
        keep = np.isfinite(obs)
        if keep.any():
            used += corrector.update(np.nan_to_num(p, nan=0.5), np.where(keep, obs, np.nan), z["groups"], z["lon"], t0)
            forecasts += 1
        z["learned_until"] = int(hours[-1] + 1)
        if now_hour - t0 > keep_days * 24 or z["learned_until"] >= t0 + t:
            path.unlink()
        else:
            np.savez_compressed(path, **z)
    return {"forecasts": forecasts, "hours_verified": used}


__all__ = ["DAY_PARTS", "ForecastArchive", "OnlineCorrector", "learn"]
