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


def previous_errors(case, previous_time, previous_tracks: dict, leads) -> dict:
    """
    How far each member's previous-cycle forecast was from the position now analysed: ``{member: km}`` for the
    members in ``previous_tracks`` (``{member: (L, 2) track issued at previous_time}``), the previous track
    interpolated to the gap between the cycles (6-24 h; nothing outside that). This is what the adaptive weights of
    :func:`blend_track` are earned from -- the member that caught the last turn keeps more of its vote.
    """
    out = {}
    try:
        gap = float((np.datetime64(case.time, "m") - np.datetime64(previous_time, "m")) / np.timedelta64(60, "m"))
    except Exception:
        return out
    if not (5.5 <= gap <= 24.5):
        return out
    hours = np.asarray(leads, np.float64)
    for member, t in previous_tracks.items():
        t = np.asarray(t, np.float64) if t is not None else None
        if t is None or t.ndim != 2 or t.shape[0] != len(hours):
            continue
        ok = np.isfinite(t[:, 0]) & np.isfinite(t[:, 1])
        if ok.sum() < 1 or hours[ok].max() < gap:
            continue
        lat = np.interp(gap, hours[ok], t[ok, 0])
        lon = np.interp(gap, hours[ok], t[ok, 1])
        out[member] = float(geo.great_circle_km(lat, lon, float(case.lat), float(case.lon)))
    return out


def adaptive_weights(table: dict, leads, previous: dict | None) -> np.ndarray | None:
    """
    The weight of ours against the table's first aid, per lead, from the previous cycle's verified errors:
    ``w_ours = clip(sigmoid(b * log((e_aid + eps) / (e_ours + eps))), 1 - cap, cap)``, with ``b`` per lead from
    ``table["adaptive"]`` (zero where the tuning seasons found nothing to earn). None without the table or the errors.
    """
    adaptive = table.get("adaptive")
    if not adaptive or not previous:
        return None
    members = list(table["members"])
    aid = members[1] if len(members) > 1 else None
    e_ours, e_aid = previous.get("ours"), previous.get(aid)
    if e_ours is None or e_aid is None:
        return None
    eps = float(adaptive.get("eps_km", 5.0))
    cap = float(adaptive.get("cap", 0.75))
    tl = np.asarray(table["leads"], np.float64)
    b = np.asarray(adaptive["b"], np.float64)[[int(np.argmin(np.abs(tl - float(h)))) for h in leads]]
    r = np.log((float(e_aid) + eps) / (float(e_ours) + eps))
    return np.clip(1.0 / (1.0 + np.exp(-b * r)), 1.0 - cap, cap)


def blend_track(case, track, aids: dict, leads, table: dict, previous_errors: dict | None = None) -> tuple[np.ndarray, dict | None]:
    """
    Our track blended with the aids' tracks in local displacement space (km east and north of the analysis
    position), weights per lead renormalised over the members present at that lead.

    Args:
        track: our (L, 2) latitude/longitude at ``leads``.
        aids: ``{tech: (L, 2)}`` at the same leads (:func:`naturev1.swarm.adeck.aids_at`).
        table: ``{"members": ["ours", ...], "leads", "weights" (L, M), "substitutes": {member: [stand-ins]},
            "adaptive": {"b": per lead, "cap", "eps_km"}}``.
        previous_errors: ``{member: km}`` from :func:`previous_errors` -- with the table's ``adaptive`` entry the
            split between ours and the first aid is earned from them (the other aids keep their table weights).

    Returns:
        ``(track, used)``: the blended track and ``{"members", "substituted", "weight_share", "disagreement_km"
        (ours against the first aid, per lead), "adaptive_w_ours" (when earned)}``, or ``(track unchanged, None)``
        when none of the table's aids is present.
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
    W = _weights(table, leads).copy()
    used: dict = {"members": [m for m in members if m in chosen], "substituted": subs}
    w_ours = adaptive_weights(table, leads, previous_errors)
    if w_ours is not None and M > 1:
        pair = W[:, 0] + W[:, 1]                               # the ours / first-aid share is split by what was earned
        W[:, 0], W[:, 1] = pair * w_ours, pair * (1.0 - w_ours)
        used["adaptive_w_ours"] = w_ours
        used["previous_errors_km"] = dict(previous_errors)
    ok = np.isfinite(X).all(-1)
    w = np.where(ok, W, 0.0)
    s = w.sum(-1)
    D = (np.nan_to_num(X) * w[..., None]).sum(1) / np.maximum(s, 1e-9)[:, None]
    lat, lon = geo.move(la0, lo0, D[:, 0], D[:, 1])
    out = np.where((s > 0)[:, None], np.stack([np.asarray(lat, np.float64), np.asarray(lon, np.float64)], -1), track)
    used["weight_share"] = _share(members, w, s)
    if M > 1 and ok[:, 1].any():
        used["disagreement_km"] = np.where(ok[:, 0] & ok[:, 1], np.hypot(X[:, 0, 0] - X[:, 1, 0], X[:, 0, 1] - X[:, 1, 1]), np.nan)
    return out, used


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


__all__ = ["adaptive_weights", "blend_intensity", "blend_track", "previous_errors", "resolve"]
