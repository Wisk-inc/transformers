# Copyright 2026 Nathan. Apache-2.0.
"""
From rain to flood, the hydrology the rule in :mod:`pluvium.flood` leaves out:

* :mod:`.antecedent` -- how wet the ground already is (an antecedent precipitation index, and the antecedent moisture
  condition it implies);
* :mod:`.runoff`     -- how much of the rain runs off (the SCS curve-number method, with the curve number adjusted
  for soil wetness, slope and how built-up a place is);
* :mod:`.hydrograph` -- when the water arrives (time of concentration from catchment length and slope, and the SCS
  triangular unit hydrograph convolved with the forecast rain), so a town is told *when* a flood would peak.
"""

from .antecedent import antecedent_index, moisture_condition
from .hydrograph import peak_timing, time_of_concentration, unit_hydrograph
from .runoff import curve_number, runoff_depth

__all__ = ["antecedent_index", "curve_number", "moisture_condition", "peak_timing", "runoff_depth",
           "time_of_concentration", "unit_hydrograph"]
