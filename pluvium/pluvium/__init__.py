# Copyright 2026 Nathan. Apache-2.0.
"""
Pluvium: rain forecasts for every town, in any country, on the NatureV1 / iHELIX engine.

``pluvium forecast`` makes the live forecast with a published model; ``pluvium program`` trains one on a single GPU.
``pluvium setup`` and ``pluvium blend`` set up any country and forecast its rain for 14 days from every operational
system (:mod:`pluvium.blend`, :mod:`pluvium.frontier`). ``pluvium cyclone`` forecasts the track of every tropical
cyclone alive now from the same systems (:mod:`pluvium.cyclone`).
See :mod:`pluvium.model` for the architecture and its experimental components.
"""

__version__ = "0.3.0"

from .domain import Gazetteer, Geography, PointSet  # noqa: E402
from .model import PluviumConfig, PluviumModel  # noqa: E402
from .schedule import DEFAULT, Schedule  # noqa: E402

__all__ = ["DEFAULT", "Gazetteer", "Geography", "PluviumConfig", "PluviumModel", "PointSet", "Schedule", "__version__"]
