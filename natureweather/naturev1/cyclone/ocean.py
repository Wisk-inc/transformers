# Copyright 2026 Nathan. Apache-2.0.
"""
The ocean fights back: the cold wake a storm stirs up, and how much of it the storm itself has to sit on.

The idea: a hurricane's winds mix cold thermocline water up to the surface. A storm moving at 5 m/s leaves the
cold water behind it; a storm that stalls -- Dorian over Grand Bahama, 1 mph for a day and a half -- sits on top
of its own wake, its fuel supply drops, and it weakens even with everything else favourable. Potential
intensity computed from the analysis's sea-surface temperature knows none of this. Here, after the
observational fits of Lloyd and Vecchi (2011) and Mei et al. (2012):

* **wake cooling** grows with intensity (wind stress goes as speed squared, mixing faster still) and falls
  with translation speed (less time over each patch of ocean), and is smaller over very warm water (a warmer
  surface usually means a deeper mixed layer, more water to stir before the cold arrives);
* **the share under the core**: a fast storm feels a small fraction of its wake, a stalled one nearly all;
* **residence time**: cooling under a stalled storm accumulates (e-folding ``RESIDENCE_H``), so the longer it
  sits, the colder its ocean.

:func:`effective_sst` gives the sea-surface temperature the core actually feels at each forecast hour, which
the intensity model passes to potential intensity (:mod:`naturev1.cyclone.intensity`).
"""

from __future__ import annotations

import numpy as np

#: Peak wake cooling for a 100 kt storm moving at 5 m/s over 28.5 C water, K.
COOLING_K = 1.4
#: Speed scale of the translation dependence, m/s, and its exponent.
SPEED_SCALE_MS = 5.0
SPEED_POWER = 0.8
#: E-folding time over which cooling under a slow storm accumulates, hours.
RESIDENCE_H = 18.0


def wake_cooling_k(vmax_kt, translation_ms, sst_k) -> np.ndarray:
    """Peak cooling of the wake, K: rises with intensity^1.5, falls with translation speed and warm water."""
    v = np.clip(np.asarray(vmax_kt, np.float64), 0.0, 200.0) / 100.0
    u = np.maximum(np.asarray(translation_ms, np.float64), 0.3)
    speed = np.clip((SPEED_SCALE_MS / u) ** SPEED_POWER, 0.4, 4.0)
    warm = np.clip(1.0 - 0.25 * (np.asarray(sst_k, np.float64) - 273.15 - 28.5), 0.5, 1.3)
    return COOLING_K * v ** 1.5 * speed * warm


def core_share(translation_ms) -> np.ndarray:
    """Fraction of the wake cooling felt under the core: ~1 for a stalled storm, ~0.25 at 7 m/s."""
    u = np.maximum(np.asarray(translation_ms, np.float64), 0.0)
    return 1.0 / (1.0 + u / 2.3)


def effective_sst(sst_k: np.ndarray, vmax_kt: np.ndarray, translation_ms: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """
    Sea-surface temperature under the core along a forecast, K.

    All arguments are ``(T,)`` along the forecast (``hours`` from now). Slow stretches accumulate cooling with
    :data:`RESIDENCE_H`; moving off resets the residence (the storm reaches new water).
    """
    sst_k = np.asarray(sst_k, np.float64)
    out = sst_k.copy()
    residence = 0.0
    hours = np.asarray(hours, np.float64)
    for k in range(sst_k.size):
        dt = hours[k] - hours[k - 1] if k else 0.0
        slow = translation_ms[k] < 3.0
        residence = residence + dt if slow else max(residence - 2.0 * dt, 0.0)
        build = 1.0 - np.exp(-(residence + 6.0) / RESIDENCE_H)
        cooling = wake_cooling_k(vmax_kt[k], translation_ms[k], sst_k[k]) * core_share(translation_ms[k]) * build
        out[k] = sst_k[k] - cooling
    return out


__all__ = ["COOLING_K", "RESIDENCE_H", "core_share", "effective_sst", "wake_cooling_k"]
