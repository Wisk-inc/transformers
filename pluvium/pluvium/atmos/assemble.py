# Copyright 2026 Nathan. Apache-2.0.
"""
Putting the atmospheric physics together: the extra coarse channels and the extra per-point inputs of version 3.

Coarse channels (on the 1-degree grid, per frame): CAPE, CIN and LCL height of an 850 hPa parcel; column relative
humidity and the Bretherton rain rate it implies; vertically integrated moisture-flux convergence; the Saharan air
layer index; tropical-wave trough strength and arrival time; parametric cyclone rain and closeness to a cyclone.

Per-point inputs (per frame): forced ascent at two terrain scales and the Froude number; the sea-breeze pulse and the
hours since the front passed; insolation since sunrise and the land and sea rain clocks.
"""

from __future__ import annotations

import numpy as np

from .cyclones import cyclone_features
from .diurnal import diurnal_features
from .moisture import moisture_channels
from .saharan import saharan_index
from .sea_breeze import sea_breeze_arrival
from .thermo import cape_cin
from .upslope import upslope_features
from .waves import wave_channels

CHANNELS = ("cape", "cin", "lcl_height", "crh", "bretherton_rate", "vimfc", "saharan", "wave_trough", "wave_eta",
            "tc_rain", "tc_distance")
POINT_FEATURES = ("upslope_5km", "upslope_20km", "froude", "sea_breeze_pulse", "sea_breeze_since", "insolation",
                  "land_clock", "sea_clock")
NEEDS = tuple(f"{v}{p}" for v in ("t", "q", "u", "v") for p in (850, 700, 500, 200))


def atmos_channels(fields: dict, upper: dict, lat: np.ndarray, lon: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """``(F, len(CHANNELS), H, W)`` from raw fields and upper-air fields already on the hourly grid."""
    t = {p: upper[f"t{p}"] for p in (850, 700, 500, 200)}
    q = {p: upper[f"q{p}"] for p in (850, 700, 500, 200)}
    cape, cin, lcl_m = cape_cin(t, q)
    moist = moisture_channels(upper, lat, lon)
    sal = saharan_index(upper, hours)
    waves = wave_channels(upper, lat, lon, hours)
    tc = cyclone_features(fields, upper, lat, lon)
    out = np.concatenate([np.stack([np.log1p(cape) / 7.0, cin / 100.0, lcl_m / 1000.0], 1), moist, sal[:, None],
                          waves, tc], 1)
    return np.nan_to_num(out).astype(np.float32)


def point_physics(terrain, statics: np.ndarray, lat: np.ndarray, lon: np.ndarray, u10: np.ndarray, v10: np.ndarray,
                  hours: np.ndarray, kind: np.ndarray | None = None) -> np.ndarray:
    """``(F, N, len(POINT_FEATURES))`` from the points' static descriptors and the frame winds at them."""
    from ..domain import STATIC_FEATURES

    col = {name: k for k, name in enumerate(STATIC_FEATURES)}
    relief = statics[:, col["elev_max"]] * 1000.0
    up = upslope_features(terrain, lat, lon, u10, v10, relief)
    land = statics[:, col["land_5km"]] > 0.5 if kind is None else np.asarray(kind) == 0
    coast_km = statics[:, col["coast_km"]] * 100.0
    pulse, since = sea_breeze_arrival(coast_km, statics[:, col["sea_dir_x"]], statics[:, col["sea_dir_y"]], u10, v10,
                                      lon, hours, land)
    di = diurnal_features(lat, lon, hours, statics[:, col["land_15km"]])
    return np.concatenate([up, pulse[..., None], (since / 12.0)[..., None], di], -1).astype(np.float32)


__all__ = ["CHANNELS", "NEEDS", "POINT_FEATURES", "atmos_channels", "point_physics"]
