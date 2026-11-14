# Copyright 2026 Nathan. Apache-2.0.
"""
Understanding the model: why it forecast what it did, and where it goes wrong.

* :mod:`.attribution` -- which inputs drove a town's forecast (occlusion of input groups: the rain already falling,
  the airports, the driving model's rain, fronts, convection, terrain, the daily clock);
* :mod:`.errors`      -- verification split by situation (rain regime, lead time, hour of day, elevation, distance
  from the coast, season) to find the conditions where skill collapses;
* :mod:`.cases`       -- replays of chosen days: forecast and truth side by side, hour by hour, for case studies.
"""

from .attribution import GROUPS, attribute
from .cases import replay
from .errors import stratified_scores

__all__ = ["GROUPS", "attribute", "replay", "stratified_scores"]
