# Copyright 2026 Nathan. Apache-2.0.
"""
Every member of every system as one cloud of tracks, centred and sized by what has verified.

An ensemble's tracks say what shape the uncertainty has today: stretched along the track when the timing is in doubt,
split in two when the storm may or may not recurve. They do not say where the centre of that uncertainty should be
(the consensus is better than any ensemble's own mean) or how large it should be (raw ensembles are usually too
narrow). So the cloud is **dressed**:

* **centred**: at every lead the cloud is moved so its weighted mean sits on the consensus track;
* **sized**: the members' offsets from that mean are scaled so the cloud's spread matches the consensus errors
  verified so far at that lead (:meth:`pluvium.cyclone.cone.ConeLearner.rms`).

The shape, and with it any split into scenarios, is kept. A system's weight is shared among its members, so a 31-member
ensemble does not outvote a single high-resolution run by headcount.

Two ways to build the cloud are offered. :func:`build_cloud` takes the members as they are. :func:`mixture_cloud`
gives every system's track the ensembles' perturbations, so a single run is a cloud too and not one line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geodesy import mean_position, offset_km, shift


@dataclass
class Cloud:
    leads: np.ndarray                   # (L,) hours
    lat: np.ndarray                     # (M, L), NaN where a member has lost the storm
    lon: np.ndarray
    weight: np.ndarray                  # (M, L), each column summing to one over the members with a fix
    system: list                        # (M,) the system each member belongs to

    @property
    def members(self) -> int:
        return self.lat.shape[0]

    def member_weight(self) -> np.ndarray:
        """One weight per member: its mean weight over the leads it has a fix for (summing to one)."""
        have = np.isfinite(self.lat)
        w = np.where(have, self.weight, 0.0).sum(1) / np.maximum(have.sum(1), 1)
        return w / max(w.sum(), 1e-12)

    def spread_km(self) -> np.ndarray:
        """Weighted root-mean-square distance of the members from the cloud's mean, by lead."""
        out = np.full(self.leads.size, np.nan)
        for k in range(self.leads.size):
            ok = np.isfinite(self.lat[:, k])
            if ok.sum() < 3:
                continue
            w = self.weight[ok, k] / self.weight[ok, k].sum()
            mlat, mlon = mean_position(self.lat[ok, k], self.lon[ok, k], w)
            east, north = offset_km(mlat, mlon, self.lat[ok, k], self.lon[ok, k])
            out[k] = float(np.sqrt((w * (east ** 2 + north ** 2)).sum()))
        return out


def build_cloud(sets: dict, weights=None, leads=None) -> Cloud:
    """The cloud from ``{system: TrackSet}``. ``weights`` is anything with ``at(lead) -> {system: weight}`` (equal
    weights without it)."""
    systems = [s for s, t in sets.items() if t.members > 0]
    if leads is None:
        leads = sorted({int(x) for s in systems for x in sets[s].steps})
    leads = np.asarray(leads, np.int64)
    lat, lon, weight, owner = [], [], [], []
    for s in systems:
        t = sets[s]
        col = {int(step): k for k, step in enumerate(t.steps)}
        la = np.full((t.members, leads.size), np.nan)
        lo = np.full((t.members, leads.size), np.nan)
        for j, lead in enumerate(leads):
            if int(lead) in col:
                la[:, j], lo[:, j] = t.lat[:, col[int(lead)]], t.lon[:, col[int(lead)]]
        w = np.zeros((t.members, leads.size))
        for j, lead in enumerate(leads):
            have = np.isfinite(la[:, j])
            if have.any():
                share = 1.0 if weights is None else float(weights.at(int(lead)).get(s, 0.0))
                w[have, j] = share / have.sum()
        lat.append(la)
        lon.append(lo)
        weight.append(w)
        owner += [s] * t.members
    if not lat:
        return Cloud(leads, np.zeros((0, leads.size)), np.zeros((0, leads.size)), np.zeros((0, leads.size)), [])
    w = np.concatenate(weight)
    w = w / np.maximum(w.sum(0, keepdims=True), 1e-12)
    return Cloud(leads, np.concatenate(lat), np.concatenate(lon), w, owner)


def perturbations(sets: dict, leads: np.ndarray, min_members: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Every ensemble member's offset from its own system's mean track: ``(east km, north km)``, each ``(P, L)``,
    NaN where the member has lost the storm. Only systems with at least ``min_members`` members contribute."""
    east, north = [], []
    for t in sets.values():
        if t.members < min_members:
            continue
        col = {int(step): k for k, step in enumerate(t.steps)}
        mlat, mlon = t.mean()
        e = np.full((t.members, leads.size), np.nan)
        n = np.full((t.members, leads.size), np.nan)
        for j, lead in enumerate(leads):
            k = col.get(int(lead))
            if k is not None and np.isfinite(mlat[k]):
                e[:, j], n[:, j] = offset_km(mlat[k], mlon[k], t.lat[:, k], t.lon[:, k])
        east.append(e)
        north.append(n)
    if not east:
        return np.zeros((0, leads.size)), np.zeros((0, leads.size))
    return np.concatenate(east), np.concatenate(north)


def mixture_cloud(sets: dict, weights=None, leads=None, min_members: int = 5) -> Cloud:
    """The cloud as a mixture: every system's own track, each carrying all the ensembles' perturbations.

    A single high-resolution or AI run is one track, but it is not certain: counted as a point it would put a third
    of the chance on one line. Here every system's mean track is given the pooled offsets of the ensemble members
    from their own means (:func:`perturbations`), so each system becomes a full cloud with the ensembles' shape, and
    the clouds are mixed with the consensus weights. The mixture's mean is the weighted mean of the systems' tracks,
    which is the consensus. Falls back to :func:`build_cloud` when no ensemble is present."""
    systems = [s for s, t in sets.items() if t.members > 0]
    if leads is None:
        leads = sorted({int(x) for s in systems for x in sets[s].steps})
    leads = np.asarray(leads, np.int64)
    east, north = perturbations(sets, leads, min_members)
    if east.shape[0] == 0:
        return build_cloud(sets, weights, leads)
    lat, lon, weight, owner = [], [], [], []
    for s in systems:
        t = sets[s]
        col = {int(step): k for k, step in enumerate(t.steps)}
        mlat, mlon = t.mean()
        la = np.full(east.shape, np.nan)
        lo = np.full(east.shape, np.nan)
        w = np.zeros(east.shape)
        for j, lead in enumerate(leads):
            k = col.get(int(lead))
            if k is None or not np.isfinite(mlat[k]):
                continue
            ok = np.isfinite(east[:, j])
            if not ok.any():
                continue
            la[ok, j], lo[ok, j] = shift(np.full(int(ok.sum()), mlat[k]), np.full(int(ok.sum()), mlon[k]),
                                         east[ok, j], north[ok, j])
            share = 1.0 if weights is None else float(weights.at(int(lead)).get(s, 0.0))
            w[ok, j] = share / ok.sum()
        lat.append(la)
        lon.append(lo)
        weight.append(w)
        owner += [s] * east.shape[0]
    w = np.concatenate(weight)
    w = w / np.maximum(w.sum(0, keepdims=True), 1e-12)
    return Cloud(leads, np.concatenate(lat), np.concatenate(lon), w, owner)


def dress(cloud: Cloud, centre: dict | None = None, rms_km: dict | None = None, clip: tuple = (0.5, 4.0),
          min_members: int = 5) -> Cloud:
    """The cloud moved onto ``centre`` (``{lead: (lat, lon)}``) and scaled to ``rms_km`` (``{lead: km}``), lead by
    lead. Either may be left out. Leads with fewer than ``min_members`` fixes are only moved, never scaled."""
    lat, lon = cloud.lat.copy(), cloud.lon.copy()
    for k, lead in enumerate(cloud.leads):
        ok = np.isfinite(lat[:, k])
        if not ok.any():
            continue
        w = cloud.weight[ok, k] / max(cloud.weight[ok, k].sum(), 1e-12)
        mlat, mlon = mean_position(lat[ok, k], lon[ok, k], w)
        east, north = offset_km(mlat, mlon, lat[ok, k], lon[ok, k])
        scale = 1.0
        if rms_km is not None and int(lead) in rms_km and ok.sum() >= min_members:
            have = float(np.sqrt((w * (east ** 2 + north ** 2)).sum()))
            if have > 1.0 and np.isfinite(rms_km[int(lead)]):
                scale = float(np.clip(rms_km[int(lead)] / have, *clip))
        clat, clon = (mlat, mlon) if centre is None or int(lead) not in centre else centre[int(lead)]
        lat[ok, k], lon[ok, k] = shift(np.full(int(ok.sum()), clat), np.full(int(ok.sum()), clon), east * scale,
                                       north * scale)
    return Cloud(cloud.leads, lat, lon, cloud.weight, cloud.system)


__all__ = ["Cloud", "build_cloud", "dress", "mixture_cloud", "perturbations"]
