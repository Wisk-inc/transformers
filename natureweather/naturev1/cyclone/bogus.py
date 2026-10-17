# Copyright 2026 Nathan. Apache-2.0.
"""
The storm a 1.5-degree analysis cannot hold, put back in the shape that matters for its motion.

The idea: at 1.5 degrees a hurricane's core is a smudge, but its *motion* does not depend on the core -- it
depends on the vortex's outer circulation (300-1000 km), which drives the beta gyres (Fiorino and Elsberry
1989), and on how deep it is. So the bogus vortex is specified by its outer strength, not its maximum wind:

* a modified-Rankine tangential profile ``V(r) = Vm (r / Rm)`` inside ``Rm`` and ``Vm (Rm / r)^alpha`` outside,
  tapered to zero beyond ``R_taper`` (a vortex with unbounded circulation would interact with the whole
  domain);
* ``Vm`` chosen so the profile's mean tangential wind between 300 and 800 km equals the outer strength
  measured from the analysis's own vortex (:func:`naturev1.cyclone.beta.outer_strength`), so the bogus has
  exactly the outer circulation the reanalysis saw, with a core the model grid can resolve;
* ``Rm`` widened from the best-track radius of maximum wind to at least two grid lengths;
* the size taken from the best-track gale radius when it exists (larger storms decay more slowly outside).

Returns winds and vorticity on any (x, y) km grid around the centre.
"""

from __future__ import annotations

import numpy as np

#: Profile shape outside the radius of maximum wind, and the taper radius, km.
ALPHA = 0.55
TAPER_KM = 1400.0


def rankine_profile(r_km: np.ndarray, vmax: float, rmax_km: float, alpha: float = ALPHA,
                    taper_km: float = TAPER_KM) -> np.ndarray:
    r = np.asarray(r_km, np.float64)
    inner = vmax * r / rmax_km
    outer = vmax * (rmax_km / np.maximum(r, 1e-6)) ** alpha
    profile = np.where(r < rmax_km, inner, outer)
    return profile * np.exp(-(r / taper_km) ** 4)


def vmax_for_outer_strength(outer_ms: float, rmax_km: float, alpha: float = ALPHA,
                            radii=(300.0, 400.0, 500.0, 600.0, 700.0, 800.0)) -> float:
    """The peak wind that gives the profile a mean tangential wind of ``outer_ms`` over ``radii``."""
    unit = rankine_profile(np.asarray(radii), 1.0, rmax_km, alpha).mean()
    return float(outer_ms / max(unit, 1e-9))


def alpha_for_size(r34_nmi: float) -> float:
    """Outer decay exponent from the gale radius: big storms decay slowly (0.4), small ones fast (0.7)."""
    if not np.isfinite(r34_nmi) or r34_nmi <= 0:
        return ALPHA
    return float(np.clip(0.75 - 0.0025 * r34_nmi, 0.4, 0.7))


def bogus_winds(x_km: np.ndarray, y_km: np.ndarray, outer_ms: float, rmax_km: float = 150.0, alpha: float = ALPHA,
                cyclonic_sign: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) m/s of the bogus vortex centred at the origin; ``cyclonic_sign`` -1 in the southern hemisphere."""
    r = np.hypot(x_km, y_km)
    vm = vmax_for_outer_strength(outer_ms, rmax_km, alpha)
    vt = cyclonic_sign * rankine_profile(r, vm, rmax_km, alpha)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(r > 0, -vt * y_km / r, 0.0)
        v = np.where(r > 0, vt * x_km / r, 0.0)
    return u, v


def bogus_vorticity(r_km: np.ndarray, outer_ms: float, rmax_km: float = 150.0, alpha: float = ALPHA,
                    cyclonic_sign: float = 1.0) -> np.ndarray:
    """Relative vorticity (s^-1) of the bogus vortex, ``(1/r) d(r V)/dr``, evaluated by central differences."""
    r = np.asarray(r_km, np.float64)
    vm = vmax_for_outer_strength(outer_ms, rmax_km, alpha)
    h = 1.0
    rv = lambda rr: rr * rankine_profile(rr, vm, rmax_km, alpha)
    rr = np.maximum(r, 2 * h)
    zeta = (rv(rr + h) - rv(rr - h)) / (2 * h) / rr / 1000.0
    return cyclonic_sign * zeta


__all__ = ["ALPHA", "TAPER_KM", "alpha_for_size", "bogus_vorticity", "bogus_winds", "rankine_profile",
           "vmax_for_outer_strength"]
