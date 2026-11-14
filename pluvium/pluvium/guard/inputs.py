# Copyright 2026 Nathan. Apache-2.0.
"""
Checking and repairing driving fields before the model sees them.

Every repair is recorded in the report so the forecast's notes say what was done. Checks:

* **units** -- 2 m temperature in Celsius (median below 100) becomes Kelvin; sea-level pressure in hPa (median below
  2,000) becomes Pa; cloud cover in percent (maximum above 1.5) becomes a fraction; water vapour in metres (median
  below 1) becomes kg/m2; specific humidity in g/kg (median above 0.1) becomes kg/kg;
* **range** -- values outside physical limits are clipped (negative rain, humidity, cloud);
* **gaps** -- frames that are entirely missing are filled by interpolation between their neighbours in time; a
  field missing everywhere gets its physical default and a warning;
* **frozen fields** -- a field identical in every future frame (a stuck feed) is flagged;
* **spikes** -- a frame differing from both neighbours by more than ``spike_sd`` standard deviations of the field's
  hour-to-hour change is replaced by their mean.
"""

from __future__ import annotations

import numpy as np

LIMITS = {"tp": (0, 300), "tcwv": (0, 90), "u10": (-80, 80), "v10": (-80, 80), "t2m": (250, 330),
          "msl": (85000, 106000), "tcc": (0, 1)}
UPPER_LIMITS = {"u": (-120, 120), "v": (-120, 120), "q": (0, 0.04), "t": (180, 330)}
DEFAULTS = {"tp": 0.0, "tcwv": 40.0, "u10": 0.0, "v10": 0.0, "t2m": 298.0, "msl": 101500.0, "tcc": 0.5}


def _fill_gaps(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Fill frames that are all NaN by linear interpolation in time."""
    flat = x.reshape(x.shape[0], -1)
    missing = ~np.isfinite(flat).any(1)
    if not missing.any() or missing.all():
        return x, int(missing.sum()) if not missing.all() else 0
    t = np.arange(x.shape[0])
    good = ~missing
    out = flat.copy()
    for i in np.flatnonzero(missing):
        a, b = t[good][t[good] < i], t[good][t[good] > i]
        if a.size and b.size:
            w = (i - a[-1]) / (b[0] - a[-1])
            out[i] = flat[a[-1]] * (1 - w) + flat[b[0]] * w
        else:
            out[i] = flat[a[-1]] if a.size else flat[b[0]]
    return out.reshape(x.shape), int(missing.sum())


def _despike(x: np.ndarray, spike_sd: float) -> tuple[np.ndarray, int]:
    if x.shape[0] < 3:
        return x, 0
    flat = x.reshape(x.shape[0], -1).astype(np.float64)
    diff = np.nanmean(np.abs(np.diff(flat, axis=0)), axis=1)
    scale = np.nanmedian(diff) + 1e-9
    fixed = 0
    for i in range(1, x.shape[0] - 1):
        a = np.nanmean(np.abs(flat[i] - flat[i - 1]))
        b = np.nanmean(np.abs(flat[i] - flat[i + 1]))
        c = np.nanmean(np.abs(flat[i + 1] - flat[i - 1]))
        if a > spike_sd * scale and b > spike_sd * scale and c < 0.5 * min(a, b):
            flat[i] = 0.5 * (flat[i - 1] + flat[i + 1])
            fixed += 1
    return flat.reshape(x.shape).astype(x.dtype), fixed


def check_driving(fields: dict, upper: dict, spike_sd: float = 8.0) -> tuple[dict, dict, list[str]]:
    """Repaired copies of ``fields`` and ``upper`` and a list of notes."""
    notes = []
    f = {k: np.array(v, np.float32, copy=True) for k, v in fields.items()}
    u = {k: np.array(v, np.float32, copy=True) for k, v in upper.items()}
    med = lambda a: float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan     # noqa: E731
    if "t2m" in f and med(f["t2m"]) < 100:
        f["t2m"] += 273.15
        notes.append("2 m temperature was in Celsius; converted to Kelvin")
    if "msl" in f and med(f["msl"]) < 2000:
        f["msl"] *= 100.0
        notes.append("sea-level pressure was in hPa; converted to Pa")
    if "tcc" in f and np.nanmax(f["tcc"]) > 1.5:
        f["tcc"] /= 100.0
        notes.append("cloud cover was in percent; converted to a fraction")
    if "tcwv" in f and med(f["tcwv"]) < 1.0:
        f["tcwv"] *= 1000.0
        notes.append("water vapour was in metres; converted to kg/m2")
    for k in list(u):
        if k.startswith("q") and med(u[k]) > 0.1:
            u[k] /= 1000.0
            notes.append(f"{k} was in g/kg; converted to kg/kg")
        if k.startswith("t") and med(u[k]) < 100:
            u[k] += 273.15
            notes.append(f"{k} was in Celsius; converted to Kelvin")
    for name, table, limits in (("surface", f, LIMITS), ("upper", u, None)):
        for k, x in table.items():
            lo, hi = (limits.get(k) if limits else UPPER_LIMITS.get(k[0])) or (-np.inf, np.inf)
            if not np.isfinite(x).any():
                table[k] = np.full_like(x, DEFAULTS.get(k, 0.0) if limits else np.nan)
                notes.append(f"{k} missing everywhere; set to its default")
                continue
            x, filled = _fill_gaps(x)
            if filled:
                notes.append(f"{k}: {filled} missing frames interpolated")
            bad = np.isfinite(x) & ((x < lo) | (x > hi))
            if bad.mean() > 0.001:
                notes.append(f"{k}: {bad.mean():.1%} of values outside physical limits clipped")
            x = np.clip(x, lo, hi)
            x, spikes = _despike(x, spike_sd)
            if spikes:
                notes.append(f"{k}: {spikes} spiked frames replaced")
            if x.shape[0] > 4 and np.allclose(x[4:], x[4:5], equal_nan=True) and k not in ("tp",):
                notes.append(f"{k}: identical in every future frame (feed may be stuck)")
            table[k] = x
    return f, u, notes


__all__ = ["check_driving"]
