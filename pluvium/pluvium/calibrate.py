# Copyright 2026 Nathan. Apache-2.0.
"""
Making the stated chances true: isotonic recalibration fitted on held-out forecasts.

A forecast that says 70% should see rain on 70% of such days. Neural nets trained by likelihood are close to that
but not exactly; the remaining error is fixed here, after training, with the pool-adjacent-violators algorithm
(monotone, so it can never reorder which hours are wetter), separately for each lead-time band and for the
"any rain in the next N hours" chances. The map is part of the checkpoint and is applied to every live forecast.
"""

from __future__ import annotations

import json

import numpy as np

LEAD_BANDS = ((0, 6), (6, 12), (12, 24), (24, 48), (48, 72))


def pava(x: np.ndarray, y: np.ndarray, bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Isotonic fit of ``y`` (0/1) on ``x`` (probabilities), as knots ``(xs, ys)`` for interpolation; binned first so
    the fit is cheap and smooth."""
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    xs, ys, ws = [], [], []
    for k in range(len(edges) - 1):
        m = which == k
        if m.any():
            xs.append(float(x[m].mean()))
            ys.append(float(y[m].mean()))
            ws.append(float(m.sum()))
    xs, ys, ws = np.array(xs), np.array(ys), np.array(ws)
    # pool adjacent violators
    blocks = [[ys[0], ws[0], [0]]]
    for i in range(1, len(ys)):
        blocks.append([ys[i], ws[i], [i]])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            y2, w2, i2 = blocks.pop()
            y1, w1, i1 = blocks.pop()
            blocks.append([(y1 * w1 + y2 * w2) / (w1 + w2), w1 + w2, i1 + i2])
    fitted = np.empty_like(ys)
    for value, _, idx in blocks:
        fitted[idx] = value
    return np.concatenate([[0.0], xs, [1.0]]), np.concatenate([[0.0], fitted, [max(1.0, fitted[-1])]]).clip(0, 1)


EXCEED = ("moderate", "heavy")       # chance of at least 2.5 mm/h and of at least 7.6 mm/h


class Calibrator:
    def __init__(self, hourly: dict | None = None, window: dict | None = None, exceed: dict | None = None) -> None:
        self.hourly = hourly or {}       # band index -> (xs, ys)
        self.window = window or {}       # window hours -> (xs, ys)
        self.exceed = exceed or {}       # "moderate"/"heavy" -> {band index -> (xs, ys)}

    @classmethod
    def fit(cls, p_hourly: np.ndarray, wet_hourly: np.ndarray, known: np.ndarray, p_window: dict | None = None,
            wet_window: dict | None = None, p_exceed: dict | None = None, y_exceed: dict | None = None) -> Calibrator:
        """``p_hourly``/``wet_hourly``/``known`` are ``(N, T)`` (points x hours) pooled over held-out forecasts;
        ``p_exceed``/``y_exceed`` the same for the chance of moderate-or-heavy and of heavy rain."""
        hourly = {}
        for k, (a, b) in enumerate(LEAD_BANDS):
            x = p_hourly[:, a:b][known[:, a:b]]
            y = wet_hourly[:, a:b][known[:, a:b]]
            if x.size >= 200:
                hourly[k] = pava(x, y)
        exceed = {}
        for name in EXCEED:
            if not p_exceed or name not in p_exceed:
                continue
            bands = {}
            for k, (a, b) in enumerate(LEAD_BANDS):
                x = p_exceed[name][:, a:b][known[:, a:b]]
                y = y_exceed[name][:, a:b][known[:, a:b]]
                if x.size >= 200 and y.sum() >= 20:
                    bands[k] = pava(x, y)
            exceed[name] = bands
        window = {}
        for h, p in (p_window or {}).items():
            y = wet_window[h]
            ok = np.isfinite(y)
            if ok.sum() >= 200:
                window[int(h)] = pava(p[ok], y[ok])
        return cls(hourly, window, exceed)

    def apply_hourly(self, p: np.ndarray) -> np.ndarray:
        out = p.copy()
        for k, (a, b) in enumerate(LEAD_BANDS):
            if k in self.hourly:
                xs, ys = self.hourly[k]
                out[..., a:b] = np.interp(p[..., a:b], xs, ys)
        return out

    def _bands(self, p: np.ndarray, table: dict) -> np.ndarray:
        out = p.copy()
        for k, (a, b) in enumerate(LEAD_BANDS):
            if k in table:
                xs, ys = table[k]
                out[..., a:b] = np.interp(p[..., a:b], xs, ys)
        return out

    def apply_classes(self, classes: np.ndarray, wet: np.ndarray) -> np.ndarray:
        """Calibrated dry/light/moderate/heavy chances ``(..., T, 4)`` from raw ones and the calibrated chance of
        rain ``wet (..., T)``: the chances of moderate-or-heavy and of heavy are recalibrated, then kept nested."""
        moderate = classes[..., 2] + classes[..., 3]
        heavy = classes[..., 3]
        raw_wet = np.maximum(1 - classes[..., 0], 1e-9)
        moderate = self._bands(moderate, self.exceed["moderate"]) if self.exceed.get("moderate") else moderate * wet / raw_wet
        heavy = self._bands(heavy, self.exceed["heavy"]) if self.exceed.get("heavy") else heavy * wet / raw_wet
        moderate = np.minimum(moderate, wet)
        heavy = np.minimum(heavy, moderate)
        return np.stack([1 - wet, wet - moderate, moderate - heavy, heavy], -1)

    def apply_window(self, p: np.ndarray, hours: int) -> np.ndarray:
        if hours in self.window:
            xs, ys = self.window[hours]
            return np.interp(p, xs, ys)
        return p

    def to_json(self) -> str:
        return json.dumps({"hourly": {str(k): [v[0].tolist(), v[1].tolist()] for k, v in self.hourly.items()},
                           "window": {str(k): [v[0].tolist(), v[1].tolist()] for k, v in self.window.items()},
                           "exceed": {n: {str(k): [v[0].tolist(), v[1].tolist()] for k, v in t.items()}
                                      for n, t in self.exceed.items()}})

    @classmethod
    def from_json(cls, text: str) -> Calibrator:
        d = json.loads(text)
        return cls({int(k): (np.array(v[0]), np.array(v[1])) for k, v in d.get("hourly", {}).items()},
                   {int(k): (np.array(v[0]), np.array(v[1])) for k, v in d.get("window", {}).items()},
                   {n: {int(k): (np.array(v[0]), np.array(v[1])) for k, v in t.items()}
                    for n, t in d.get("exceed", {}).items()})


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[tuple[float, float, int]]:
    """``[(mean forecast, observed frequency, count)]`` in equal-width probability bins."""
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for k in range(bins):
        m = (p >= edges[k]) & (p < edges[k + 1] if k < bins - 1 else p <= 1)
        if m.any():
            out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return out


__all__ = ["Calibrator", "EXCEED", "LEAD_BANDS", "pava", "reliability"]
