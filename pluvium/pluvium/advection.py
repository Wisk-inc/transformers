# Copyright 2026 Nathan. Apache-2.0.
"""
Rain already falling, carried forward (experimental as an input to a neural forecaster of this kind).

For the first hours of a forecast nothing beats extrapolation: the shower over the sea east of Port Antonio at
2 pm is over Port Antonio at 3 pm if the steering wind is 25 km/h from the east. Radar nowcasting systems (pySTEPS,
DGMR, NowcastNet's evolution network) are built on this. The Caribbean has little open radar, so here the rain field
of the last hour (CMORPH in training, GOES-East live) is advected semi-Lagrangian-wise along the steering wind (the
mean of the driving model's 850 and 700 hPa winds at the issue time): each point's value for forecast hour L is the
observed rain at the point the wind will carry to it in L + 1 hours. The network gets that as one more input per
point and hour for the first 12 hours, with a flag for whether the rain field was observed, and learns how fast
such rain decays or grows.
"""

from __future__ import annotations

import numpy as np

from .domain import KM_PER_DEG

HOURS = 12            # advected rain is given for forecast hours 0-11


def _xyz(lat, lon) -> np.ndarray:
    la, lo = np.radians(np.asarray(lat, np.float64)), np.radians(np.asarray(lon, np.float64))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


class Advector:
    """Observed rain at scattered points (``obs_lat``, ``obs_lon``) read at upstream positions."""

    def __init__(self, obs_lat: np.ndarray, obs_lon: np.ndarray, neighbours: int = 4, max_km: float = 45.0) -> None:
        self.xyz = _xyz(obs_lat, obs_lon)
        self.k, self.max_km = neighbours, max_km

    def read(self, values: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inverse-distance mean of the ``k`` nearest observed values within ``max_km`` of each point, and whether
        any was found."""
        from scipy.spatial import cKDTree

        ok = np.isfinite(values)
        if not ok.any():
            return np.zeros(np.shape(lat), np.float32), np.zeros(np.shape(lat), bool)
        tree = cKDTree(self.xyz[ok])
        dist, idx = tree.query(_xyz(lat, lon).reshape(-1, 3), k=min(self.k, int(ok.sum())))
        dist, idx = np.atleast_2d(dist.T).T if dist.ndim == 1 else dist, np.atleast_2d(idx.T).T if idx.ndim == 1 else idx
        km = dist * 6371.0
        w = np.where(km <= self.max_km, 1.0 / (km + 2.0), 0.0)
        vals = values[ok][idx]
        found = w.sum(-1) > 0
        out = (w * vals).sum(-1) / np.maximum(w.sum(-1), 1e-12)
        return out.reshape(np.shape(lat)).astype(np.float32), found.reshape(np.shape(lat))

    def extrapolate(self, values: np.ndarray, lat: np.ndarray, lon: np.ndarray, u: np.ndarray, v: np.ndarray,
                    leads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(len(leads), N)`` advected rain (mm/h) and found-flags for forecast hours ``leads`` at points, with the
        steering wind ``u``/``v`` (m/s, ``(N,)``)."""
        lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        out = np.zeros((len(leads), lat.size), np.float32)
        found = np.zeros((len(leads), lat.size), bool)
        coslat = np.cos(np.radians(lat))
        for i, lead in enumerate(leads):
            dt = float(lead) + 1.0                                # last observed hour's middle -> this hour's middle
            up_lat = lat - v * 3.6 * dt / KM_PER_DEG
            up_lon = lon - u * 3.6 * dt / (KM_PER_DEG * np.maximum(coslat, 0.2))
            out[i], found[i] = self.read(values, up_lat, up_lon)
        return out, found


__all__ = ["Advector", "HOURS"]
