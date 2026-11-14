# Copyright 2026 Nathan. Apache-2.0.
"""
Spatially coherent samples.

The chain gives each town its own distribution of rain paths, and version 2 sampled every town independently. That
is right for one town and wrong for a group: two neighbouring villages under the same shower get it together, so the
chance of "rain somewhere in the parish" is lower, and the spread of catchment totals higher, than independent
sampling says. A Gaussian copula fixes the joint behaviour without touching any single town's distribution: the
uniform numbers that drive each town's state transitions and amounts are made spatially correlated, by sampling a
Gaussian random field with correlation length ``L`` (about 15-30 km for showers, 100 km or more for fronts) on a
grid by FFT filtering of white noise, interpolating it to the towns, and mapping it through the normal CDF.
"""

from __future__ import annotations

import numpy as np


def gaussian_field(shape: tuple, spacing_km: float, length_km: float, rng, count: int = 1) -> np.ndarray:
    """``(count, H, W)`` standard-normal fields with a squared-exponential correlation of length ``length_km``."""
    h, w = shape
    ky = np.fft.fftfreq(h, d=spacing_km)[:, None]
    kx = np.fft.fftfreq(w, d=spacing_km)[None, :]
    spectrum = np.exp(-0.5 * (2 * np.pi) ** 2 * (kx**2 + ky**2) * (length_km**2) / 2.0)
    noise = rng.standard_normal((count, h, w))
    field = np.fft.ifft2(np.fft.fft2(noise) * np.sqrt(spectrum)).real
    return field / np.maximum(field.std(axis=(1, 2), keepdims=True), 1e-9)


class SpatialCopula:
    """Correlated uniform numbers at fixed points."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray, spacing_km: float = 4.0) -> None:
        self.lat, self.lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        self.spacing = spacing_km
        pad = 0.3
        self.s, self.w = self.lat.min() - pad, self.lon.min() - pad
        dlat = spacing_km / 111.195
        dlon = spacing_km / (111.195 * np.cos(np.radians(self.lat.mean())))
        self.h = int(np.ceil((self.lat.max() + pad - self.s) / dlat)) + 1
        self.wid = int(np.ceil((self.lon.max() + pad - self.w) / dlon)) + 1
        self.i = (self.lat - self.s) / dlat
        self.j = (self.lon - self.w) / dlon

    def normals(self, count: int, length_km: float, rng) -> np.ndarray:
        """``(count, P)`` correlated standard normals."""
        from scipy import ndimage

        fields = gaussian_field((self.h, self.wid), self.spacing, length_km, rng, count)
        return np.stack([ndimage.map_coordinates(f, [self.i, self.j], order=1, mode="nearest") for f in fields])

    def uniforms(self, count: int, hours: int, length_km: float, rng, persistence: float = 0.7) -> np.ndarray:
        """``(count, P, hours)`` correlated uniforms, also correlated hour to hour (AR(1) with ``persistence``)."""
        from scipy.special import ndtr

        z = self.normals(count, length_km, rng)
        out = np.empty((count, self.lat.size, hours))
        out[..., 0] = ndtr(z)
        for t in range(1, hours):
            z = persistence * z + np.sqrt(1 - persistence**2) * self.normals(count, length_km, rng)
            out[..., t] = ndtr(z)
        return out


def area_probability(wet_any: np.ndarray, groups: np.ndarray) -> dict:
    """``wet_any (S, P)`` whether each sampled path at each place has rain in a window -> ``{group: P(somewhere)}``."""
    out = {}
    for g in np.unique(groups):
        cols = groups == g
        out[str(g)] = float(wet_any[:, cols].any(1).mean())
    return out


__all__ = ["SpatialCopula", "area_probability", "gaussian_field"]
