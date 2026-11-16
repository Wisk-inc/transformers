# Copyright 2026 Nathan. Apache-2.0.
"""
Pluvium: rain forecasts for every town, in any country, on the NatureV1 / iHELIX engine.

``pluvium now --country JM`` goes from nothing to a 14-day rain forecast for a country in one command
(:mod:`pluvium.quick`). ``pluvium setup`` and ``pluvium blend`` are its steps, one at a time (:mod:`pluvium.blend`,
:mod:`pluvium.frontier`). ``pluvium cyclone`` forecasts the track of every tropical cyclone alive now from the same
six systems (:mod:`pluvium.cyclone`). ``pluvium forecast`` makes the hourly forecast with a published network, and
``pluvium program`` trains one on a single GPU. ``pluvium upstream`` copies the published weights of the AI models
whose forecasts Pluvium reads (:mod:`pluvium.upstream`).
See :mod:`pluvium.model` for the architecture and its experimental components.
"""

__version__ = "0.5.3"

from .domain import Gazetteer, Geography, PointSet  # noqa: E402
from .model import PluviumConfig, PluviumModel  # noqa: E402
from .schedule import DEFAULT, Schedule  # noqa: E402

__all__ = ["DEFAULT", "Gazetteer", "Geography", "PluviumConfig", "PluviumModel", "PointSet", "Schedule", "__version__"]
