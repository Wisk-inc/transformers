# Copyright 2026 Nathan. Apache-2.0.
"""
Scenarios: sharp, coherent rain fields from calibrated chances.

The expected-rain map of a calibrated forecast smooths out with lead. That is honest, and it is the wrong picture of a
storm. This package turns Pluvium Blend's calibrated chances into scenarios that stay sharp at every lead and are
still calibrated at every place:

* :mod:`.ecc` couples the calibrated distributions to a physical ensemble's ranks (ensemble copula coupling);
* :mod:`.schaake` does the same from past observed days when no ensemble is at hand;
* :mod:`.pmm` reduces scenarios to one sharp map (probability-matched mean, medoid);
* :mod:`.areal` gives the rain over an area (a parish, a basin) with honest spread;
* :mod:`.scores` scores what place-by-place scores cannot see: energy and variogram scores, the CRPS of areal
  totals, the sharpness of single scenarios;
* :mod:`.fss` and :mod:`.spectra` measure structure on grids: the fractions skill score by scale, and power spectra.
"""

from .ecc import ecc, independent, ranks
from .pmm import medoid, probability_matched_mean
from .schaake import schaake
from .scores import areal_crps, crps_ensemble, energy_score, variogram_score

__all__ = ["areal_crps", "crps_ensemble", "ecc", "energy_score", "independent", "medoid", "probability_matched_mean",
           "ranks", "schaake", "variogram_score"]
