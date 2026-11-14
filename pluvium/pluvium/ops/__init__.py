# Copyright 2026 Nathan. Apache-2.0.
"""
Running Pluvium as a service:

* :mod:`.loop` -- the hourly cycle: forecast, guard, archive, learn from the airports, publish the outputs, with
  retries and a fallback, and a ``latest`` folder that always holds the newest complete forecast;
* :mod:`.api`  -- a small HTTP server answering JSON queries for a town, a parish or a country from the latest
  forecast (standard library only).
"""

from .api import serve
from .loop import run_cycle, run_forever

__all__ = ["run_cycle", "run_forever", "serve"]
