# Copyright 2026 Nathan. Apache-2.0.
"""
Pluvium Frontier: the second generation of Pluvium Blend.

The first full-year benchmark showed where Pluvium Blend was weak: rain of 50 mm or more, a record hurricane that
calibration shrank, winter in the continental mid-latitudes, expected-rain maps that smooth out with lead, and anomaly
correlation beyond a week. Frontier is the answer to each, built so that every design choice is made inside the
training year and the test year is scored once:

* :mod:`.oof` and :mod:`.store` keep what every component says about days it never trained on, so any pool can be
  fitted and compared without refitting a component;
* :mod:`pluvium.pool` holds the pools: weights by threshold, season and weather regime, and an online aggregator that
  re-weighs the components as the season unfolds;
* :mod:`pluvium.tails` trains on every heavy-rain row and hands record storms back to the raw systems' consensus;
* :mod:`pluvium.scenarios` turns the calibrated chances into sharp, coherent scenario fields;
* :mod:`pluvium.anomaly` adds an expected-amount head built on multi-model anomalies;
* :mod:`pluvium.cyclone` tracks tropical cyclones in every member of every system.
"""

from .oof import components_for, fit_component, oof_rows, parity, row_arrays

__all__ = ["components_for", "fit_component", "oof_rows", "parity", "row_arrays"]
