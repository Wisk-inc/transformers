# Copyright 2026 Nathan. Apache-2.0.
"""
The operational guidance blended in.

The NHC's a-deck carries every aid's track and intensity for a cycle minutes after it. Two tables fitted on the
tuning seasons (``coefficients.ADECK_TRACK``, ``coefficients.ADECK_INTENSITY``) say how much of each aid to take
beside our own forecast, lead by lead. The weights are renormalised over the members present, so a missing aid never
breaks the blend, and a retired model's slot can be taken by its successor (``substitutes``: HAFS-A for HWRF, HAFS-B
for HMON, TVCN for HCCA) -- a stand-in by lineage, reported as such, not a verified member.

Reading the decks is :mod:`naturev1.swarm.adeck`; this module is the arithmetic, with no network.
"""

from __future__ import annotations

import numpy as np

from . import geometry as geo


def resolve(available: dict, members, substitutes: dict | None = None) -> tuple[dict, dict]:
    """The members present in ``available`` (``{tech: array}``), a substitute standing in for an absent one.
    Returns ``(chosen {member: array}, substituted {member: stand-in})``; ``"ours"`` is never looked up."""
    chosen, subs = {}, {}
    for m in members:
        if m == "ours":
            continue
        v = available.get(m)
        if v is not None and np.isfinite(np.asarray(v, np.float64)).any():
            chosen[m] = np.asarray(v, np.float64)
            continue
        for s in (substitutes or {}).get(m, ()):
            v = available.get(s)
            if v is not None and np.isfinite(np.asarray(v, np.float64)).any():
                chosen[m] = np.asarray(v, np.float64)
                subs[m] = s
                break
    return chosen, subs


def _weights(table: dict, leads) -> np.ndarray:
    """(L, M) weights at the caller's leads: the nearest lead of the table."""
    tl = np.asarray(table["leads"], np.float64)
    W = np.asarray(table["weights"], np.float64)
    return W[[int(np.argmin(np.abs(tl - float(h)))) for h in leads]]


def _share(members, w: np.ndarray, s: np.ndarray) -> dict:
    share = w / np.maximum(s, 1e-9)[:, None]
    return {m: float(np.mean(share[:, j])) for j, m in enumerate(members) if share[:, j].any()}


def blend_track(case, track, aids: dict, leads, table: dict) -> tuple[np.ndarray, dict | None]:
    """
    Our track blended with the aids' tracks in local displacement space (km east and north of the analysis
    position), weights per lead renormalised over the members present at that lead.

    Args:
        track: our (L, 2) latitude/longitude at ``leads``.
        aids: ``{tech: (L, 2)}`` at the same leads (:func:`naturev1.swarm.adeck.aids_at`).
        table: ``{"members": ["ours", ...], "leads", "weights" (L, M), "substitutes": {member: [stand-ins]}}``.

    Returns:
        ``(track, used)``: the blended track and ``{"members", "substituted", "weight_share"}``, or
        ``(track unchanged, None)`` when none of the table's aids is present.
    """
    members = list(table["members"])
    chosen, subs = resolve(aids or {}, members, table.get("substitutes"))
    track = np.asarray(track, np.float64)
    if not chosen:
        return track, None
    L, M = len(leads), len(members)
    la0, lo0 = float(case.lat), float(case.lon)
    X = np.full((L, M, 2), np.nan)
    for j, m in enumerate(members):
        t = track if m == "ours" else chosen.get(m)
        if t is None or np.asarray(t).shape != (L, 2):
            continue
        e, n = geo.local_displacement_km(la0, lo0, np.asarray(t)[:, 0], np.asarray(t)[:, 1])
        X[:, j, 0], X[:, j, 1] = e, n
    W = _weights(table, leads)
    ok = np.isfinite(X).all(-1)
    w = np.where(ok, W, 0.0)
    s = w.sum(-1)
    D = (np.nan_to_num(X) * w[..., None]).sum(1) / np.maximum(s, 1e-9)[:, None]
    lat, lon = geo.move(la0, lo0, D[:, 0], D[:, 1])
    out = np.where((s > 0)[:, None], np.stack([np.asarray(lat, np.float64), np.asarray(lon, np.float64)], -1), track)
    return out, {"members": [m for m in members if m in chosen], "substituted": subs, "weight_share": _share(members, w, s)}


def blend_intensity(intensity, aids: dict, leads, table: dict) -> tuple[np.ndarray, dict | None]:
    """Our intensity (kt at ``leads``) blended with the aids' intensities, as :func:`blend_track`; clipped to 15-200 kt."""
    members = list(table["members"])
    chosen, subs = resolve(aids or {}, members, table.get("substitutes"))
    intensity = np.asarray(intensity, np.float64)
    if not chosen:
        return intensity, None
    L, M = len(leads), len(members)
    X = np.full((L, M), np.nan)
    for j, m in enumerate(members):
        v = intensity if m == "ours" else chosen.get(m)
        if v is not None and np.asarray(v).shape == (L,):
            X[:, j] = v
    W = _weights(table, leads)
    ok = np.isfinite(X)
    w = np.where(ok, W, 0.0)
    s = w.sum(-1)
    out = np.where(s > 0, (np.nan_to_num(X) * w).sum(-1) / np.maximum(s, 1e-9), intensity)
    out = np.where(np.isfinite(out), np.clip(out, 15.0, 200.0), out)
    return out, {"members": [m for m in members if m in chosen], "substituted": subs, "weight_share": _share(members, w, s)}


__all__ = ["blend_intensity", "blend_track", "resolve"]
