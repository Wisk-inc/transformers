# Copyright 2026 Nathan. Apache-2.0.
"""
What can go wrong when the model is running, and what Pluvium does about it.

* :mod:`.inputs`      -- a driving model changes units, drops a field, freezes, or publishes garbage: fields are
  checked and repaired (Kelvin vs Celsius, Pa vs hPa, percent vs fraction, negative rain, stuck frames, spikes);
* :mod:`.freshness`   -- old data posing as new: the age of every input is checked and stale observations are not
  fed to the model as "now";
* :mod:`.drift`       -- situations unlike anything in training (an unusual storm, an upgraded driving model): the
  inputs are compared with the training distribution and, when far outside it, the forecast is pulled towards
  climatology in proportion;
* :mod:`.consistency` -- outputs that contradict each other (any rain in 6 h more likely than in 12 h, heavy more
  likely than any rain, a start window ending before it begins) are found and repaired;
* :mod:`.fallback`    -- when the network cannot run at all, a simple statistical forecast from the driving models'
  rain and climatology is issued instead, clearly labelled, rather than nothing.
"""

from .consistency import check_summary
from .drift import InputStats
from .fallback import fallback_summary
from .freshness import Freshness
from .inputs import check_driving

__all__ = ["Freshness", "InputStats", "check_driving", "check_summary", "fallback_summary"]
