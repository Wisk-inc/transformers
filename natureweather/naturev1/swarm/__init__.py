# Copyright 2026 Nathan. Apache-2.0.
"""
The swarm sandbox (idea: Nathan, CorX Labs): a particle filter over possible futures of a storm, fed by the satellite
path and painted onto the ground, working alongside NatureV1's physics and model.

==============  ======================================================================================================
core            the swarm: every world a 4D worldline (longitude, latitude, steering depth, time) with its own
                physics, integrated in micro-segments, re-weighted by every fix, resampled with genealogy
fixes           the satellite path: NHC best-track and fix decks (Dvorak, microwave, scatterometer, recon, radar),
                each fix with its own uncertainty, streamed into the swarm minute by minute
analysis        GFS analyses (live, or any time since 2021) cut into the storm-centred box of the hindcasts
geography       every Caribbean country and territory as a raster of its parishes / provinces / districts, with
                terrain
impact          each world's wind and rain painted onto that raster: per-parish probabilities of tropical-storm and
                hurricane-force wind and gusts, rain with orographic enhancement, flash-flood signal, arrival times
live            one call from an NHC storm id to an impact report (live, or a replay of what was known at the time)
plot            maps of the swarm, its worldlines and its impacts
==============  ======================================================================================================

It works with :func:`naturev1.cyclone.physics_forecast` (``swarm=N``): the physics consensus stays the best single
track, and the swarm, re-centred on it, supplies the probabilities around it. Given a dynamical model's forecast
steering (:class:`naturev1.cyclone.forecastflow.ForecastFlow`, read from GFS by :mod:`analysis`), the worlds are
carried through it, which is what makes their recurvature probabilities skilful.
"""

from .core import Swarm, SwarmFields, SwarmForecast, SwarmPriors, classify_tracks, swarm_products

__all__ = ["Swarm", "SwarmFields", "SwarmForecast", "SwarmPriors", "classify_tracks", "swarm_products"]
