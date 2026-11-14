# Copyright 2026 Nathan. Apache-2.0.
"""
Reading the ground around every town.

Thirteen numbers (mean and maximum height, slopes, land fractions, coast distance) described each place in version 2.
Whether a town sits in a valley open to the trades, on a plateau, or on the lee side of a ridge is a pattern, not a
number. Here a 16 x 16 patch of elevation and land cover around each point (2.2 km spacing, about 35 km across) is
read by a small convolutional encoder into the point's query. Patches are cut once from the terrain raster for the
fixed point sets (8 km cells, airports) and on the fly for towns.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

SIZE, SPACING_KM = 16, 2.2


def terrain_patches(terrain, lat: np.ndarray, lon: np.ndarray, size: int = SIZE, spacing_km: float = SPACING_KM,
                    chunk: int = 4096) -> np.ndarray:
    """``(N, 2, size, size)`` float16: elevation (km) and land fraction around each point, north up."""
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    k = (np.arange(size) - (size - 1) / 2.0) * spacing_km
    out = np.zeros((lat.size, 2, size, size), np.float16)
    land = terrain.land.astype(np.float32)
    for start in range(0, lat.size, chunk):
        la = lat[start: start + chunk, None, None] + k[None, :, None] / 111.195
        lo = lon[start: start + chunk, None, None] + k[None, None, :] / (111.195 * np.cos(np.radians(lat[start: start + chunk, None, None])))
        shape = la.shape[:1] + (size, size)
        la = np.broadcast_to(la, shape).ravel()
        lo = np.broadcast_to(lo, shape).ravel()
        out[start: start + chunk, 0] = (terrain.sample(terrain.elevation, la, lo).reshape(shape) / 1000.0).astype(np.float16)
        out[start: start + chunk, 1] = terrain.sample(land, la, lo).reshape(shape).astype(np.float16)
    return out


class TerrainPatchEncoder(nn.Module):
    def __init__(self, hidden: int, channels: int = 24) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, channels, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(channels, 2 * channels, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(2 * channels, 2 * channels, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2 * channels, hidden))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """``patches (P, 2, S, S)`` -> ``(P, hidden)``."""
        return self.net(patches.to(self.net[0].weight.dtype))


__all__ = ["SIZE", "SPACING_KM", "TerrainPatchEncoder", "terrain_patches"]
