# Copyright 2026 Nathan. Apache-2.0.
"""
Pluvium Cyclone: where a tropical cyclone will go, from every operational system at once.

Pluvium forecasts rain, and the heaviest rain a tropical country sees comes with cyclones. Rain cannot be placed
without the track, so this package tracks every storm in every member of six operational systems and combines them:

* :mod:`.fields` reads sea-level pressure for a basin from ECMWF ENS, HRES and AIFS and NOAA GEFS, GFS and GraphCast;
* :mod:`.detect` and :mod:`.tracker` follow the pressure minimum of each member from the position known in real time;
* :mod:`.bias` removes each system's along-track and cross-track bias, learned on an earlier season;
* :mod:`.weights` and :mod:`.consensus` weigh the systems by their error covariance and combine them;
* :mod:`.interp` shifts a consensus to the next advisory cycle, as forecasters do with "early" guidance;
* :mod:`.clusters`, :mod:`.cone` and :mod:`.strike` turn the members into scenarios, an error radius that follows
  the ensemble spread, and strike probabilities;
* :mod:`.rain_swath` moves each member's rain with its own storm onto the consensus track, so the rain core is not
  smeared by track spread;
* :mod:`.verify` scores everything against the best track and against the aids in the National Hurricane Center's
  decks (:mod:`.atcf`): the official forecast, its consensus aids and Google DeepMind's model.
"""

from .atcf import BestTrack, aids, best_track, fetch_decks
from .basins import BASINS, basin_of
from .geodesy import distance_km, mean_position
from .storms import Storm, active, first_guess, load_season, start_hours

__all__ = ["BASINS", "BestTrack", "Storm", "active", "aids", "basin_of", "best_track", "distance_km", "fetch_decks",
           "first_guess", "load_season", "mean_position", "start_hours"]
