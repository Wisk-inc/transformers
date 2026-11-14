# Copyright 2026 Nathan. Apache-2.0.
"""
The daily clock of tropical rain, written down instead of learned.

Global AI models are known to put the tropical daily rain maximum at the wrong hour (GraphCast concentrates it too
close to local noon). Two physical clocks set it:

* over **land**, rain follows heating: storms need several hours of sunshine to build the boundary layer, so the
  maximum is mid to late afternoon. The integral of insolation since sunrise (normalised by its daily total) is the
  "fuel gauge";
* over the **sea**, rain peaks before dawn: the cloud tops cool by radiation overnight while the sea stays warm, so
  the column destabilises at night (the nocturnal maritime maximum), strongest near 05-07 local time.

Inputs per point and frame: the insolation fraction since sunrise, hours to sunset, and the two clocks weighted by
how much land surrounds the point.
"""

from __future__ import annotations

import numpy as np

from ..features import cos_zenith, local_solar_hour


def diurnal_features(lat: np.ndarray, lon: np.ndarray, hours: np.ndarray, land_fraction: np.ndarray) -> np.ndarray:
    """``(F, N, 3)``: insolation since sunrise (0-1), the land clock and the sea clock (each -1..1, land-weighted)."""
    hours = np.asarray(hours, np.float64)
    lst = local_solar_hour(lon[None, :], hours[:, None])
    # insolation since local sunrise, integrated over the preceding hours (cos zenith is clipped at night)
    steps = np.arange(0, 13)
    past = hours[:, None] - steps[None, :]                                                  # (F, 13)
    cz = cos_zenith(lat[None, None, :], lon[None, None, :], past[..., None])                # (F, 13, N)
    so_far = cz.sum(1) / 7.6                                     # about 1 by sunset in the Caribbean (sum of cos z)
    land = np.clip(np.asarray(land_fraction, np.float64), 0, 1)[None]
    land_clock = np.cos(2 * np.pi * (lst - 15.5) / 24.0)
    sea_clock = np.cos(2 * np.pi * (lst - 5.5) / 24.0)
    return np.stack([np.clip(so_far, 0, 1.5), land_clock * land, sea_clock * (1 - land)], -1).astype(np.float32)


__all__ = ["diurnal_features"]
