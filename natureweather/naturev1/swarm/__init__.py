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
models          the live models -- GFS, ECMWF IFS and AIFS (open data), GraphCast-GFS, the GEFS mean -- each read by
                byte range into forecast steering and the model's own storm track; the swarm's worlds each follow one
geography       every Caribbean country and territory as a raster of its parishes / provinces / districts, with
                terrain
impact          each world's wind and rain painted onto that raster: per-parish probabilities of tropical-storm and
                hurricane-force wind and gusts, rain with orographic enhancement, flash-flood signal, arrival times
landfall        when the centre comes ashore (median and 10-90%), where first, how strong then and at peak, the chance
                the storm peaks at each category, the rapid-intensification branch, and each parish's peak wind, gust,
                rain (if hit, and its wettest cell) and arrival times, with dates
observations    barometer readings fitted to a vortex (a centre fix with uncertainty and a central pressure) and point
                winds blended into the painted map -- operators for feeds that do not yet exist here
live            one call from an NHC storm id to an impact report (live, or a replay of what was known at the time);
                every cycle saved, the previous one's error fed back
bulletin        the cycle record (JSON + parish CSV), the history of a storm's cycles, the previous cycle found again
infrared        GOES band-13 read quantitatively: cold-top share, coldest top, ring symmetry, eye, their trend (diagnostic)
adeck           the NHC's operational guidance: a-deck aids interpolated to the leads (blended in by cyclone.aids), the RI probabilities
plot            maps of the swarm, its worldlines and its impacts; the parish-by-parish landfall figure
naturev2        the trained NatureV2 on a live storm: GFS turned into its 89-channel training state, NHC's best track,
                GOES-East / West storm patches, and its storm heads and full-state rollout run beside the swarm
==============  ======================================================================================================

It works with :func:`naturev1.cyclone.physics_forecast` (``swarm=N``): the physics consensus stays the best single
track, and the swarm, re-centred on it, supplies the probabilities around it. Given a dynamical model's forecast
steering (:class:`naturev1.cyclone.forecastflow.ForecastFlow`, read from GFS by :mod:`analysis`), the worlds are
carried through it, which is what makes their recurvature probabilities skilful.
"""

from .core import Swarm, SwarmFields, SwarmForecast, SwarmPriors, classify_tracks, swarm_products

__all__ = ["Swarm", "SwarmFields", "SwarmForecast", "SwarmPriors", "classify_tracks", "swarm_products"]
