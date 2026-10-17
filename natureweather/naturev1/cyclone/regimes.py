# Copyright 2026 Nathan. Apache-2.0.
"""
What kind of track situation this is -- because what to trust depends on it.

The idea: the members of a consensus do not have fixed skill. Extrapolation is excellent for a straight-runner
in the trades and terrible for a storm about to stall at a col or recurve into the westerlies; a frozen
steering flow is fine for a day in a steady pattern and useless by day four in a changing one; climatology is
the safe bet exactly when the pattern is ordinary. Forecasters reason about the regime first. This module
does it with numbers the other modules compute:

* ``stall``: :func:`naturev1.cyclone.stall.stall_diagnostics` says the path runs into collapsed steering near a
  col (Dorian);
* ``recurve``: the storm is within a few degrees of the subtropical ridge axis, or a trough upstream is close
  and deep enough to reach it (:func:`naturev1.cyclone.ridge.ridge_diagnostics`);
* ``high``: poleward of 30 degrees -- already in or near the westerlies, extratropical transition territory;
* ``weak``: depressions and weak tropical storms (<= 40 kt), whose centres are poorly defined, jump, and are
  steered shallowly;
* ``straight``: everything else -- the steady trade-wind cases.

:func:`classify` returns the label and the scores; the consensus fits weights per regime where a regime has
enough past cases, falling back to the global weights where it does not.
"""

from __future__ import annotations

import numpy as np

REGIMES = ("stall", "recurve", "high", "weak", "straight")


def classify(case, stall: dict | None = None, ridge: dict | None = None) -> dict:
    """``{"regime": label, "scores": {...}}`` from the stall and ridge diagnostics and the case itself."""
    scores = {}
    scores["stall"] = float(stall["stall_probability"]) if stall else 0.0
    gap = ridge.get("ridge_gap_deg", np.nan) if ridge else np.nan
    trough_d = ridge.get("trough_distance_km", np.nan) if ridge else np.nan
    trough_a = ridge.get("trough_amplitude", np.nan) if ridge else np.nan
    recurve = 0.0
    if np.isfinite(gap):
        recurve = max(recurve, float(np.clip(1.0 - gap / 8.0, 0.0, 1.0)))
    if np.isfinite(trough_d) and np.isfinite(trough_a):
        recurve = max(recurve, float(np.clip((2500.0 - trough_d) / 2000.0, 0.0, 1.0) * np.clip(trough_a / 4e-5, 0.0, 1.0)))
    scores["recurve"] = recurve
    scores["high"] = float(np.clip((abs(case.lat) - 27.0) / 6.0, 0.0, 1.0))
    wind = case.wind_kt if np.isfinite(case.wind_kt) else 40.0
    scores["weak"] = float(np.clip((45.0 - wind) / 10.0, 0.0, 1.0))
    if scores["stall"] > 0.5:
        label = "stall"
    elif scores["high"] > 0.5:
        label = "high"
    elif scores["recurve"] > 0.5:
        label = "recurve"
    elif scores["weak"] > 0.5:
        label = "weak"
    else:
        label = "straight"
    return {"regime": label, "scores": scores}


__all__ = ["REGIMES", "classify"]
