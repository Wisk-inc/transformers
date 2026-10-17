# Copyright 2026 Nathan. Apache-2.0.
"""
The top of the storm: upper-tropospheric outflow, and the troughs that feed it or tear it.

The idea: a hurricane is a heat engine whose exhaust is the 200 hPa outflow. How easily it can vent decides
whether it intensifies, and the upper-level environment acts in two ways the low-level predictors miss:

* **divergence aloft** -- strong 200 hPa divergence over the storm (an outflow jet, a trough's right entrance
  region nearby) evacuates mass and favours deepening;
* **eddy flux convergence of angular momentum** (EFC; Molinari and Vollaro 1989, a SHIPS predictor) -- the
  convergence of the eddy flux of relative angular momentum by the environment's asymmetric flow at 200 hPa,
  300-600 km out: ``EFC = -(1 / r^2) d(r^2 <u_L' v_T'>) / dr``. Large positive EFC from an approaching trough
  spins up the outflow anticyclone and can trigger intensification -- *if* shear stays moderate; with strong
  shear the same trough destroys the storm. The classification below encodes that distinction.
* **outflow channels** -- the number of azimuthal sectors (of eight) with strong radial outflow at 500-1000 km:
  one channel is ordinary, two (poleward into a jet and equatorward) is the signature of the most intense storms;
* **outflow temperature** comes from the potential intensity calculation (:mod:`naturev1.thermo`).
"""

from __future__ import annotations

import numpy as np

from .fields import Environment
from .geometry import destination


def _ring_winds(env: Environment, lat0: float, lon0: float, radius_km: float, level: int, azimuths: int = 32):
    bearing = np.arange(azimuths) * 360.0 / azimuths
    plat, plon = destination(lat0, lon0, bearing, radius_km)
    u = env.sample(env.level("u", level), plat, plon)
    v = env.sample(env.level("v", level), plat, plon)
    b = np.radians(bearing)
    radial = u * np.sin(b) + v * np.cos(b)
    tangential = -u * np.cos(b) + v * np.sin(b)
    return radial, tangential


def eddy_flux_convergence(env: Environment, lat0: float, lon0: float, level: int = 200,
                          radii_km=(300.0, 400.0, 500.0, 600.0)) -> float:
    """EFC in m/s per day, averaged over 300-600 km (cyclonic sign per hemisphere)."""
    sign = 1.0 if lat0 >= 0 else -1.0
    radii = np.asarray(radii_km) * 1000.0
    flux = []
    for r in radii_km:
        radial, tangential = _ring_winds(env, lat0, lon0, r, level)
        flux.append(np.mean((radial - radial.mean()) * (sign * tangential - np.mean(sign * tangential))))
    flux = np.asarray(flux)
    d = np.gradient(radii ** 2 * flux, radii)
    efc = -(d / radii ** 2)
    return float(np.mean(efc) * 86400.0)


def outflow_diagnostics(env: Environment, env_clean: Environment, lat0: float, lon0: float, shear_ms: float) -> dict:
    """Divergence aloft, EFC, outflow channels and a trough-interaction class (favourable / hostile / none)."""
    div = env.divergence(env.level("u", 200), env.level("v", 200))
    r = env.distance_km(lat0, lon0)
    disc = r < 1000.0
    divergence = float(div[disc].mean()) * 1e5 if disc.any() else 0.0
    efc = eddy_flux_convergence(env_clean, lat0, lon0)
    channels = 0
    for radius in (500.0, 750.0, 1000.0):
        radial, _ = _ring_winds(env, lat0, lon0, radius, 200, azimuths=32)
        sectors = radial.reshape(8, 4).mean(1)
        channels = max(channels, int((sectors > 6.0).sum()))
    if efc > 10.0 and shear_ms < 10.0:
        interaction = "favourable"
    elif efc > 10.0:
        interaction = "hostile"
    else:
        interaction = "none"
    return {"divergence_200": divergence, "efc_ms_day": efc, "outflow_channels": channels, "trough_interaction": interaction}


__all__ = ["eddy_flux_convergence", "outflow_diagnostics"]
