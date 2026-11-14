# Copyright 2026 Nathan. Apache-2.0.
"""
A season of storms: learn on one, forecast another.

``learn`` measures every system's bias and the covariance of their errors on a training season, against the best
track. ``forecast`` then tracks every storm of another season and builds the consensus in several forms, so the value
of each step can be read off the scores:

* ``equal``: the plain mean of the systems;
* ``weighted``: weights from the error covariance;
* ``corrected``: bias removed, then weighted (the full batch consensus);
* ``adaptive``: the same, re-weighted inside each storm by how the systems have done on it so far (:mod:`.online`).

Nothing from the forecast season is used to forecast it: the start position is the real-time one, the bias and weights
are last season's, and the in-storm re-weighting sees only positions already observed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bias import TrackBias
from .consensus import consensus
from .fields import SYSTEMS, FieldArchive
from .online import StormMemory
from .storms import Storm, active
from .tracker import TrackerConfig
from .tracks import TrackStore
from .verify import LEADS, verifiable
from .weights import ConsensusWeights, error_vectors


@dataclass
class Case:
    storm: Storm
    init: int
    sets: dict                                   # system -> TrackSet
    points: dict                                 # system -> {lead: (lat, lon)} (the system's mean track)
    consensus: dict = field(default_factory=dict)   # variant -> {lead: (lat, lon)}
    carq: dict = field(default_factory=dict)        # real-time fixes: {hour: {lead: (lat, lon, wind)}}

    @property
    def key(self) -> tuple:
        return (self.storm.id, int(self.init))


def observed(storm: Storm, init: int, leads=LEADS) -> dict:
    """Best-track positions at the leads that verify: ``{lead: (lat, lon)}``."""
    out = {}
    for lead in leads:
        if verifiable(storm.track, init, lead):
            pos = storm.track.position(int(init) + lead)
            if pos is not None:
                out[lead] = pos
    return out


def cases_of(archive: FieldArchive, storms: list, inits, store: TrackStore, systems=SYSTEMS,
             cfg: TrackerConfig | None = None, progress=None) -> list:
    """Track every storm at every start time (tracks are cached in ``store``)."""
    out = []
    carq = {s.id: s.aids({"CARQ"}).get("CARQ", {}) for s in storms}
    for k, init in enumerate(inits):
        for storm in active(storms, int(init)):
            sets = store.get(archive, storm, int(init), systems, cfg, carq[storm.id])
            sets = {s: t for s, t in sets.items() if s in systems}
            points = {s: t.points() for s, t in sets.items()}
            points = {s: p for s, p in points.items() if p}
            if points:
                out.append(Case(storm, int(init), sets, points, carq=carq[storm.id]))
        if progress and k % 20 == 0:
            progress(f"  tracked {k + 1} of {len(inits)} start times ({len(out)} cases)")
    return out


def learn(cases: list, systems=SYSTEMS, bias_shrink: float = 15.0, cov_shrink: float = 0.3,
          to_equal: float = 0.25) -> tuple[TrackBias, ConsensusWeights]:
    """Bias and weights from a season's cases, against the best track."""
    bias = TrackBias(bias_shrink)
    for case in cases:
        obs = observed(case.storm, case.init)
        for system, points in case.points.items():
            bias.learn(system, points, obs)
    weights = ConsensusWeights(list(systems), shrink=cov_shrink, to_equal=to_equal)
    for case in cases:
        obs = observed(case.storm, case.init)
        for system, points in case.points.items():
            weights.add(case.key, system, error_vectors(bias.correct(system, points), obs))
    return bias, weights.fit()


def forecast(cases: list, bias: TrackBias | None, weights: ConsensusWeights | None, adaptive: bool = True,
             memory_gain: float = 1.0) -> list:
    """Fill every case's consensus variants. Cases are taken in time order so the in-storm memory stays causal."""
    memory = {}
    for case in sorted(cases, key=lambda c: (c.init, c.storm.id)):
        case.consensus["equal"] = consensus(case.points)
        if weights is not None:
            case.consensus["weighted"] = consensus(case.points, weights)
        if bias is not None and weights is not None:
            case.consensus["corrected"] = consensus(case.points, weights, bias)
            if adaptive:
                mem = memory.setdefault(case.storm.id, StormMemory(gain=memory_gain))
                scale = mem.scale(case.init, list(case.points))
                case.consensus["adaptive"] = consensus(case.points, weights, bias, scale=scale)
                mem.remember(case.init, {s: bias.correct(s, p) for s, p in case.points.items()}, case.carq)
    return cases


def spread(case: Case, lead: int) -> float:
    """Spread of the systems' mean tracks around their mean at a lead, km (NaN with fewer than three systems)."""
    from .geodesy import distance_km, mean_position

    pts = [p[lead] for p in case.points.values() if lead in p]
    if len(pts) < 3:
        return float("nan")
    lat, lon = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    mlat, mlon = mean_position(lat, lon)
    return float(np.sqrt(np.mean(distance_km(lat, lon, mlat, mlon) ** 2)))


__all__ = ["Case", "cases_of", "forecast", "learn", "observed", "spread"]
