# Copyright 2026 Nathan. Apache-2.0.
"""
The whole chain for one forecast: track, cone, cloud, scenarios and passage chances.

:class:`Forecaster` holds everything that learns (the consensus weights of :mod:`.learner` and the cone of
:mod:`.cone`) and is stepped through cases in time order. Each step first learns from whatever has verified by the
case's start time, then issues :class:`Guidance` from the past alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .approach import closest_approach, passage_chance
from .cloud import Cloud, build_cloud, dress, mixture_cloud
from .clusters import scenarios
from .cone import ConeLearner
from .geodesy import distance_km, mean_position
from .interp import interpolate
from .learner import SeasonLearner
from .places import near


@dataclass
class Guidance:
    storm: str
    init: int
    start: tuple | None                 # where the storm was at the start time, as known then
    track: dict                         # the consensus {lead: (lat, lon)}
    spread: dict                        # {lead: km} between the systems' own tracks
    cone: dict                          # {lead: km} radius that should hold the centre two times in three
    cone_static: dict                   # the same without today's spread
    cloud: Cloud | None = None          # the mixture of systems, centred on the track and sized by verified errors
    raw_cloud: Cloud | None = None      # the members as they came
    scenarios: list = field(default_factory=list)
    #: {name: {"chance" (the cloud's), "mixture_chance", "sized_chance", "raw_chance", "track_km", "track_hour"}}
    places: dict = field(default_factory=dict)

    def early(self, observed_now: tuple, age_hours: int = 12) -> dict:
        """The track as early guidance ``age_hours`` later (:func:`pluvium.cyclone.interp.interpolate`)."""
        return interpolate(self.track, observed_now, age_hours, extend_hours=age_hours)


def system_spread(points: dict, lead: int) -> float:
    """Root-mean-square distance (km) of the systems' tracks from their mean at a lead (NaN under three systems)."""
    pts = [p[lead] for p in points.values() if lead in p and np.isfinite(p[lead][0])]
    if len(pts) < 3:
        return float("nan")
    lat, lon = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    mlat, mlon = mean_position(lat, lon)
    return float(np.sqrt(np.mean(distance_km(lat, lon, mlat, mlon) ** 2)))


class Forecaster:
    def __init__(self, systems: list, learner: dict | None = None, radius_km: float = 150.0, horizon: int = 120,
                 within_km: float = 2000.0, level: float = 2.0 / 3.0, sized: bool = True) -> None:
        self.learner = SeasonLearner(list(systems), **(learner or dict(method="minvar", memory_gain=1.0)))
        self.cone = ConeLearner(level)
        self.radius_km, self.horizon, self.within_km = radius_km, horizon, within_km
        #: scale the cloud to the consensus errors verified so far (off: the ensembles' own spread is kept). On gave
        #: the better passage chances in both development seasons.
        self.sized = sized

    def state(self) -> dict:
        """Everything learned so far (weights and cone), for a JSON file."""
        return {"learner": self.learner.state(), "cone": self.cone.state()}

    def load_state(self, state: dict) -> None:
        """Start from what an earlier forecaster learned (see :mod:`pluvium.cyclone.live` for the shipped prior)."""
        self.learner.load_state(state["learner"])
        self.cone.load_state(state["cone"])

    def step(self, case, basin: str | None = None, clouds: bool = True) -> Guidance:
        """Guidance for one case (:class:`pluvium.cyclone.season.Case`); cases must come in time order."""
        self.learner.learn_until(case.init)
        self.cone.learn_until(case.init)
        track = self.learner.predict(case.storm.id, case.init, case.points, case.carq)
        spread = {lead: system_spread(case.points, lead) for lead in track}
        static = {lead: self.cone.base(lead) for lead in track}
        cone = {lead: self.cone.radius(lead, spread[lead]) for lead in track}
        self.cone.queue(case.init, track, spread, case.carq)
        fix = case.carq.get(int(case.init), {}).get(0)
        start = None if fix is None else (float(fix[0]), float(fix[1]))
        out = Guidance(case.storm.id, int(case.init), start, track, spread, cone, static)
        if not clouds or not track:
            return out
        leads = sorted(track)
        out.raw_cloud = build_cloud(case.sets, None, leads)
        # size the cloud by what has verified, widened or narrowed like the cone by today's spread
        rms = {lead: self.cone.rms(lead) * cone[lead] / max(static[lead], 1e-9) for lead in leads}
        mixture = dress(mixture_cloud(case.sets, self.learner, leads), track)
        sized = dress(mixture, None, rms)
        out.cloud = sized if self.sized else mixture
        out.scenarios = scenarios(out.cloud)
        if basin is not None and start is not None:
            for name, (plat, plon) in near(basin, start[0], start[1], self.within_km).items():
                km, hour = closest_approach(track, plat, plon, start, self.horizon)
                chance = {key: passage_chance(c, plat, plon, self.radius_km, start, self.horizon)
                          for key, c in (("chance", out.cloud), ("mixture_chance", mixture), ("sized_chance", sized),
                                         ("raw_chance", out.raw_cloud))}
                out.places[name] = {**chance, "track_km": km, "track_hour": hour}
        return out


__all__ = ["Forecaster", "Guidance", "system_spread"]
