# Copyright 2026 Nathan. Apache-2.0.
"""
Pluvium: hour-by-hour rain forecasts for every Caribbean town, on the NatureV1 / iHELIX engine.

``pluvium forecast`` makes the live forecast with a published model; ``pluvium program`` trains one on a single GPU.
See :mod:`pluvium.model` for the architecture and its experimental components.
"""

__version__ = "0.1.0"

from .domain import Gazetteer, Geography, PointSet  # noqa: E402
from .model import PluviumConfig, PluviumModel  # noqa: E402
from .schedule import DEFAULT, Schedule  # noqa: E402

__all__ = ["DEFAULT", "Gazetteer", "Geography", "PluviumConfig", "PluviumModel", "PointSet", "Schedule", "__version__"]
