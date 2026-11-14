# Copyright 2026 Nathan. Apache-2.0.
"""
Probability done carefully, where point-by-point forecasts fall short:

* :mod:`.copula`      -- independent sampling at each town gives the wrong chance of "rain somewhere in the parish"
  and the wrong catchment totals; a Gaussian copula with a spatial correlation length makes samples coherent;
* :mod:`.conformal`   -- prediction intervals for totals with guaranteed coverage on held-out data (conformalised
  quantile regression);
* :mod:`.extremes`    -- lognormal tails underestimate cloudbursts; generalised Pareto tails fitted to the satellite
  record give torrential-rain chances and return periods ("a 1-in-10-year hour here");
* :mod:`.scores`      -- the scores that mean-error metrics hide: Brier decomposition, ROC area, the equitable threat
  score, SEDI for rare events, and the fractions skill score for displaced rain;
* :mod:`.diagnostics` -- PIT histograms, reliability with consistency bars, spread-skill.
"""

from .conformal import ConformalIntervals
from .copula import SpatialCopula, gaussian_field
from .diagnostics import pit_histogram, reliability_bands, spread_skill
from .extremes import GPDTail, RainfallTails, fit_gpd
from .scores import brier_decomposition, ets, fss_points, roc_auc, sedi

__all__ = ["ConformalIntervals", "GPDTail", "RainfallTails", "SpatialCopula", "brier_decomposition", "ets", "fit_gpd",
           "fss_points", "gaussian_field", "pit_histogram", "reliability_bands", "roc_auc", "sedi", "spread_skill"]
