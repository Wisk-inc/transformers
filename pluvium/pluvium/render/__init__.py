# Copyright 2026 Nathan. Apache-2.0.
"""
Forecasts in the forms people and agencies use:

* :mod:`.maps`     -- PNG maps of the chance of rain, thunder or flood risk over an island, coloured by place;
* :mod:`.bulletin` -- a national bulletin: headline, then each parish in plain language (English, or Jamaican
  Patois for Jamaica);
* :mod:`.alerts`   -- Common Alerting Protocol (CAP 1.2) messages for heavy rain, thunderstorms and flash-flood risk
  per parish, the format Caribbean meteorological services and disaster offices exchange.
"""

from .alerts import cap_alerts
from .bulletin import bulletin
from .maps import draw_map

__all__ = ["bulletin", "cap_alerts", "draw_map"]
