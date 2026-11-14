# Copyright 2026 Nathan. Apache-2.0.
"""
One track from every system: the consensus.

For each lead, the systems that still carry the storm are corrected for their own bias (:mod:`.bias`), weighed
(:mod:`.weights`) and averaged on the sphere. When a system drops out (it lost the storm, or its run is missing) the
remaining weights are rescaled, so the consensus degrades gracefully instead of jumping.

A consensus is only as good as its members are different. Three kinds of system feed it: physical ensembles (ECMWF ENS,
NOAA GEFS), physical high-resolution runs (ECMWF HRES, NOAA GFS) and AI models (ECMWF AIFS, NOAA GraphCast).
"""

from __future__ import annotations

import numpy as np

from .bias import TrackBias
from .geodesy import mean_position
from .weights import ConsensusWeights


def consensus(tracks: dict, weights: ConsensusWeights | None = None, bias: TrackBias | None = None,
              scale: dict | None = None, min_systems: int = 1) -> dict:
    """The consensus track ``{lead: (lat, lon)}`` from ``{system: {lead: (lat, lon)}}``.

    ``weights`` and ``bias`` come from an earlier season (equal weights and no correction without them). ``scale``
    multiplies each system's weight (``{system: factor}``), which is how the in-storm re-weighting of
    :mod:`.online` enters."""
    if bias is not None:
        tracks = {s: bias.correct(s, pts) for s, pts in tracks.items()}
    leads = sorted({lead for pts in tracks.values() for lead in pts})
    out = {}
    for lead in leads:
        have = [s for s, pts in tracks.items() if lead in pts and np.isfinite(pts[lead][0])]
        if len(have) < min_systems:
            continue
        base = weights.at(lead) if weights is not None else {}
        w = np.array([base.get(s, 1.0 / len(have) if not base else 0.0) * (scale or {}).get(s, 1.0) for s in have])
        if w.sum() <= 0:
            w = np.ones(len(have))
        lat, lon = mean_position(np.array([tracks[s][lead][0] for s in have]),
                                 np.array([tracks[s][lead][1] for s in have]), w / w.sum())
        out[lead] = (float(lat), float(lon))
    return out


def member_cloud(sets: dict, weights: ConsensusWeights | None = None, lead: int | None = None) -> tuple:
    """Every member of every system as one weighted cloud at a lead: ``(lat, lon, weight)`` arrays. A system's weight
    is shared equally among its members that still carry the storm, so a 31-member ensemble does not outvote a
    single high-resolution run by headcount."""
    lats, lons, ws = [], [], []
    for system, ts in sets.items():
        k = int(np.searchsorted(ts.steps, lead))
        if k >= ts.steps.size or ts.steps[k] != lead:
            continue
        ok = np.isfinite(ts.lat[:, k])
        if not ok.any():
            continue
        w_sys = (weights.at(lead).get(system, 0.0) if weights is not None else 1.0)
        lats.append(ts.lat[ok, k])
        lons.append(ts.lon[ok, k])
        ws.append(np.full(int(ok.sum()), w_sys / ok.sum()))
    if not lats:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    w = np.concatenate(ws)
    return np.concatenate(lats), np.concatenate(lons), w / max(w.sum(), 1e-12)


__all__ = ["consensus", "member_cloud"]
