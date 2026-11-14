# Copyright 2026 Nathan. Apache-2.0.
"""
Theoretical atmospheric physics computed from the driving fields and given to the network as inputs.

Each module targets a known weakness of data-driven rain forecasting in the tropics:

* :mod:`.thermo`      -- exact moist thermodynamics (Bolton 1980) and parcel CAPE/CIN, instead of crude indices;
* :mod:`.moisture`    -- column relative humidity and its exponential control of tropical rain (Bretherton et al.
  2004), and the vertically integrated moisture-flux convergence;
* :mod:`.saharan`     -- the Saharan air layer, which suppresses Caribbean summer showers;
* :mod:`.waves`       -- tropical waves: trough axes at 700 hPa, their westward motion and arrival times;
* :mod:`.cyclones`    -- tropical cyclones: vortex detection and a parametric rain field (R-CLIPER-style), because
  cyclones are too rare in any training period for a network to learn their rain from data alone;
* :mod:`.sea_breeze`  -- the sea-breeze front: when it reaches each inland point, from gravity-current theory;
* :mod:`.diurnal`     -- the heating clock: insolation since sunrise, and the nocturnal maritime rain maximum;
* :mod:`.upslope`     -- wind meeting terrain at every point: forced ascent at two scales, and blocking (Froude);
* :mod:`.assemble`    -- all of it as the extra coarse channels and per-point inputs of version 3.
"""

from .assemble import CHANNELS as ATMOS_CHANNELS
from .assemble import POINT_FEATURES, atmos_channels, point_physics
from .cyclones import cyclone_features, find_cyclones, parametric_rain
from .diurnal import diurnal_features
from .moisture import column_relative_humidity, moisture_channels
from .saharan import saharan_index
from .sea_breeze import sea_breeze_arrival
from .thermo import cape_cin, lcl, mixing_ratio, saturation_vapour_pressure, theta_e_bolton
from .upslope import upslope_features
from .waves import wave_channels

__all__ = ["ATMOS_CHANNELS", "POINT_FEATURES", "atmos_channels", "cape_cin", "point_physics", "upslope_features", "column_relative_humidity", "cyclone_features", "diurnal_features",
           "find_cyclones", "lcl", "mixing_ratio", "moisture_channels", "parametric_rain", "saharan_index",
           "saturation_vapour_pressure", "sea_breeze_arrival", "theta_e_bolton", "wave_channels"]
