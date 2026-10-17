# Copyright 2026 Nathan. Apache-2.0.
"""
How big the storm is, measured from the analysis and carried forward with its climatology.

The idea: size is not decoration. A storm's outer circulation decides how fast it drifts (beta drift scales
with it), how its bogus is built (:mod:`naturev1.cyclone.bogus`) and how far its winds reach (the hazard
maps). The best track records gale radii only in some basins and years; the analysis always has the storm's
footprint in its pressure field. So:

* :func:`outer_closed_isobar` -- the radius of the outermost closed isobar (ROCI): the largest ring around the
  centre whose every sample is higher than the centre by at least a contour interval (2 hPa), searched outward
  in 50 km steps, the classic operational size measure, from the analysis's sea-level pressure;
* :func:`gale_radius_estimate` -- the 34 kt radius from ROCI and intensity when the best track has none (a
  regression form after Knaff et al. 2016: R34 grows with intensity and latitude, and with ROCI);
* :func:`size_evolution` -- how the gale radius changes over a forecast: storms grow as they age and move
  poleward (and expand rapidly in extratropical transition), shrink over land.
"""

from __future__ import annotations

import numpy as np

from .fields import Environment
from .geometry import destination


def outer_closed_isobar(env: Environment, lat0: float, lon0: float, interval_hpa: float = 2.0, max_km: float = 1500.0,
                        step_km: float = 25.0, azimuths: int = 24) -> dict:
    """
    ``{"roci_km", "poci_hpa", "centre_hpa"}`` from the analysis sea-level pressure (Pa).

    Rays leave the centre in ``azimuths`` directions; an isobar (a multiple of ``interval_hpa`` above the centre) is
    closed when every ray reaches it within ``max_km``. The outermost closed isobar is the highest such one; its
    radius is the mean distance at which the rays first reach it.
    """
    msl = env.fields["msl"] / 100.0
    bearing = np.arange(azimuths) * 360.0 / azimuths
    radii = np.arange(0.0, max_km + 1e-9, step_km)
    plat, plon = destination(lat0, lon0, bearing[:, None], radii[None, :])
    profiles = env.sample(msl, plat, plon)                                    # (azimuths, radii)
    centre = float(profiles[:, :4].min())
    roci, poci = 0.0, centre
    level = np.floor(centre / interval_hpa) * interval_hpa + interval_hpa
    while True:
        reached = profiles >= level
        if not reached.any(1).all():
            break
        first = reached.argmax(1)
        roci, poci = float(radii[first].mean()), float(level)
        level += interval_hpa
    return {"roci_km": roci, "poci_hpa": poci, "centre_hpa": centre}


def gale_radius_estimate(wind_kt: float, lat: float, roci_km: float) -> float:
    """34 kt radius (nmi) from intensity, latitude and ROCI (0 below gale force)."""
    if not np.isfinite(wind_kt) or wind_kt < 34:
        return 0.0
    base = 40.0 + 0.75 * (wind_kt - 34.0) + 1.5 * max(abs(lat) - 15.0, 0.0)
    if np.isfinite(roci_km) and roci_km > 0:
        base = 0.6 * base + 0.4 * (0.35 * roci_km / 1.852)
    return float(np.clip(base, 20.0, 400.0))


def size_evolution(r34_nmi: float, lat: np.ndarray, land: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """Gale radius along a forecast: +1.5%/6 h of growth, +4%/6 h poleward of 30 degrees, -6%/6 h over land."""
    out = np.empty_like(np.asarray(hours, np.float64))
    r = r34_nmi
    for k in range(len(out)):
        dt = hours[k] - hours[k - 1] if k else 0.0
        rate = 0.015 + 0.04 * float(abs(lat[k]) > 30.0) - 0.075 * float(land[k] > 0.5)
        r = max(r * (1.0 + rate * dt / 6.0), 0.0)
        out[k] = r
    return out


__all__ = ["gale_radius_estimate", "outer_closed_isobar", "size_evolution"]
