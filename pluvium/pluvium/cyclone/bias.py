# Copyright 2026 Nathan. Apache-2.0.
"""
Each system's systematic track error, measured on an earlier season and removed.

A model's track errors are not random. Most global models move storms too slowly, some turn them too early, and the
bias grows with lead. The bias is measured in the storm's own frame: along its direction of motion (fast or slow) and
across it (left or right). In real time the direction of motion at the verifying time is not known, so both learning
and correction use the forecast's own heading at that lead.

Biases are shrunk towards zero with the number of cases behind them: with ``n`` cases a bias counts for
``n / (n + shrink)`` of its measured value. A bias seen in 10 forecasts is mostly noise; one seen in 80 is not.
"""

from __future__ import annotations

import numpy as np

from .geodesy import along_cross, bearing_deg, from_along_cross


def headings(points: dict) -> dict:
    """The forecast's own heading at each lead (degrees), from its neighbouring positions."""
    leads = sorted(points)
    out = {}
    for k, lead in enumerate(leads):
        a = points[leads[max(k - 1, 0)]]
        b = points[leads[min(k + 1, len(leads) - 1)]]
        if a == b or not np.isfinite(a[0]) or not np.isfinite(b[0]):
            continue
        out[lead] = float(bearing_deg(a[0], a[1], b[0], b[1]))
    return out


class TrackBias:
    """Along-track and cross-track bias per system and lead."""

    def __init__(self, shrink: float = 15.0) -> None:
        self.shrink = shrink
        self.sums = {}                               # (system, lead) -> [n, sum along, sum cross]

    def learn(self, system: str, points: dict, observed: dict) -> None:
        """Add one forecast: ``points`` and ``observed`` are ``{lead: (lat, lon)}`` (the observed where it verifies)."""
        head = headings(points)
        for lead, obs in observed.items():
            if lead not in points or lead not in head or not np.isfinite(points[lead][0]):
                continue
            a, c = along_cross(points[lead][0], points[lead][1], obs[0], obs[1], head[lead])
            s = self.sums.setdefault((system, int(lead)), [0, 0.0, 0.0])
            s[0] += 1
            s[1] += float(a)
            s[2] += float(c)

    def bias(self, system: str, lead: int) -> tuple[float, float]:
        """The shrunk ``(along, cross)`` bias in km (zero when the system was never seen at that lead)."""
        n, sa, sc = self.sums.get((system, int(lead)), (0, 0.0, 0.0))
        if n == 0:
            return 0.0, 0.0
        k = 1.0 / (n + self.shrink)
        return sa * k, sc * k

    def correct(self, system: str, points: dict) -> dict:
        """The forecast with its system's bias removed, lead by lead."""
        head = headings(points)
        out = {}
        for lead, (lat, lon) in points.items():
            if lead not in head or not np.isfinite(lat):
                out[lead] = (lat, lon)
                continue
            a, c = self.bias(system, lead)
            nlat, nlon = from_along_cross(lat, lon, -a, -c, head[lead])
            out[lead] = (float(nlat), float(nlon))
        return out

    def table(self) -> dict:
        """``{system: {lead: (along km, cross km, cases)}}`` for reports."""
        out = {}
        for (system, lead), (n, _, _) in sorted(self.sums.items()):
            a, c = self.bias(system, lead)
            out.setdefault(system, {})[lead] = (a, c, n)
        return out


__all__ = ["TrackBias", "headings"]
