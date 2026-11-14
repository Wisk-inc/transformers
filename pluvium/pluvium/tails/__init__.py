# Copyright 2026 Nathan. Apache-2.0.
"""
The tail of the rain distribution: heavy days, record storms, and what calibration does to them.

Calibration learned from ordinary weather pulls every forecast towards what usually happens. That is right on an
ordinary day and wrong on the day that matters most. This package holds what Pluvium does about it:

* :mod:`.stratify` trains on every heavy-rain row, with weights that keep the chances unbiased;
* :mod:`.twcrps` scores and weighs the tail with a proper score (threshold-weighted CRPS);
* :mod:`.consensus` is the uncalibrated agreement of the raw systems, the one forecast that does not shrink;
* :mod:`.support` measures how far a forecast lies outside anything seen in training;
* :mod:`.override` hands the forecast back to the raw consensus in proportion, so a record storm is not calibrated
  away;
* :mod:`.withheld` tunes that hand-over honestly: it trains with the largest storms withheld and tests on them;
* :mod:`.gpd` and :mod:`.returns` put amounts in context with extreme-value theory (return periods);
* :mod:`.alerts` turns all of it into alert levels.
"""

from .stratify import draw_plan, heavy_rows, signal_of, table_rows, tail_sample

__all__ = ["draw_plan", "heavy_rows", "signal_of", "table_rows", "tail_sample"]
