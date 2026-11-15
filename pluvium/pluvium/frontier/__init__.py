# Copyright 2026 Nathan. Apache-2.0.
"""
Pluvium Frontier: the second generation of Pluvium Blend.

The first full-year benchmark showed where Pluvium Blend was weak: rain of 50 mm or more, a record hurricane that
calibration shrank, winter in the continental mid-latitudes, and expected-rain maps that smooth out with lead.
Frontier is the answer to each, built so that design choices are made inside the training year and written down
before the test year is scored:

* :mod:`.oof` and :mod:`.store` keep what every component says about days it never trained on, so any pool can be
  fitted and compared without refitting a component;
* :mod:`.evaluate` scores pool designs on a place's stored components, and :mod:`.report` turns the results into
  tables, losses included;
* :mod:`.model` is the design as a deployable model for any country (``pluvium blend train``);
* :mod:`pluvium.pool` holds the pools: weights by band of thresholds and lead, and an online aggregator that
  re-weighs the components as the season unfolds;
* :mod:`pluvium.tails` trains on every heavy-rain row, hands record storms back to the raw systems' consensus where
  a place's own staged storms say that helps, and puts amounts in the place's own terms (return periods);
* :mod:`pluvium.scenarios` turns the calibrated chances into sharp, coherent scenario fields.

Ideas that were built, tested and dropped (weights by season and regime, a separate expected-amount head, an audit
of ordinary days as the hand-over's switch) are in ``docs/WHAT_DID_NOT_WORK.md``.
"""

from .oof import components_for, fit_component, oof_rows, parity, row_arrays

__all__ = ["components_for", "fit_component", "oof_rows", "parity", "row_arrays"]
