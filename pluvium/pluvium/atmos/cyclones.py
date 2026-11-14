# Copyright 2026 Nathan. Apache-2.0.
"""
Tropical cyclones: too rare to learn from data, too important to miss.

Even six training years hold only a handful of hurricanes passing near any one island, so a network cannot learn
from data alone how rain is arranged around a storm. Theory and climatology give the shape:

* **finding the storm** -- a minimum of sea-level pressure below 1006 hPa with strong cyclonic 850 hPa vorticity;
  its strength from the pressure deficit (the Atkinson-Holliday wind-pressure relation, ``V = 6.7 (1010 - p)^0.644``
  knots);
* **its rain** -- an R-CLIPER-shaped radial profile (Marks and DeMaria 2003; Tuleya et al. 2007): rain rises
  linearly from the centre to a maximum at the radius of maximum rain, then decays exponentially, the maximum and the
  radii growing with intensity (azimuthal-mean peaks of about 5-15 mm/h, Lonfat et al. 2004);
* **its asymmetry** -- with vertical wind shear the rain concentrates down-shear and to the left (Chen et al. 2006),
  a cosine weighting by the angle to the down-shear-left direction.

The parametric field and the distance to the nearest storm are inputs; the network learns how much to trust them.
The coefficients here are chosen to match published magnitudes; they are a prior, not a fitted model.
"""

from __future__ import annotations

import numpy as np

from ..features import _grad
from ..util import great_circle_km


def find_cyclones(msl: np.ndarray, u850: np.ndarray, v850: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                  p_max_hpa: float = 1006.0, vort_min: float = 4.0) -> list[list[tuple]]:
    """Per frame, ``[(lat, lon, p_min hPa, v_max m/s), ...]`` of cyclone centres on the grid."""
    from scipy import ndimage

    vx, _ = _grad(v850, lat, lon)
    _, uy = _grad(u850, lat, lon)
    vort = vx - uy                                                    # 1e-5 per second
    out = []
    for f in range(msl.shape[0]):
        p = msl[f] / 100.0
        minima = (p == ndimage.minimum_filter(p, size=5, mode="nearest")) & (p < p_max_hpa)
        found = []
        for i, j in zip(*np.nonzero(minima)):
            if ndimage.maximum_filter(vort[f], size=3)[i, j] < vort_min:
                continue
            v_kt = 6.7 * max(1010.0 - p[i, j], 0.0) ** 0.644
            found.append((float(lat[i]), float(lon[j]), float(p[i, j]), v_kt * 0.5144))
        out.append(found)
    return out


def parametric_rain(dist_km: np.ndarray, vmax: float, shear_u: float = 0.0, shear_v: float = 0.0,
                    bearing: np.ndarray | None = None) -> np.ndarray:
    """Rain rate (mm/h) at distances from a storm of maximum wind ``vmax`` (m/s), with shear asymmetry."""
    u = np.clip(vmax / 33.0, 0.3, 2.5)                       # 1 = a minimal hurricane
    t_max = 4.0 + 5.0 * u                                     # mm/h at the radius of maximum rain
    r_m = np.clip(70.0 - 15.0 * u, 25.0, 70.0)                # km
    r_e = 120.0 + 60.0 * u                                    # e-folding km
    d = np.asarray(dist_km, np.float64)
    inner = 0.3 * t_max + 0.7 * t_max * d / r_m
    rate = np.where(d < r_m, inner, t_max * np.exp(-(d - r_m) / r_e))
    if bearing is not None:
        shear = np.hypot(shear_u, shear_v)
        if shear > 1.0:
            down = np.degrees(np.arctan2(shear_u, shear_v))           # direction the shear points to
            left = (down - 45.0) % 360.0                              # down-shear-left
            a = np.clip(shear / 10.0, 0, 1) * 0.6
            rate = rate * (1 + a * np.cos(np.radians(bearing - left)))
    return np.clip(rate, 0, 60).astype(np.float32)


def _bearing(lat0, lon0, lat, lon):
    la0, la = np.radians(lat0), np.radians(lat)
    dlon = np.radians(lon - lon0)
    x = np.sin(dlon) * np.cos(la)
    y = np.cos(la0) * np.sin(la) - np.sin(la0) * np.cos(la) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def cyclone_features(fields: dict, upper: dict, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """``(F, 2, H, W)``: parametric storm rain (log1p mm/h) and closeness to the nearest storm (exp(-d / 300 km))."""
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    storms = find_cyclones(fields["msl"], upper["u850"], upper["v850"], lat, lon)
    f = fields["msl"].shape[0]
    rain = np.zeros((f,) + la.shape, np.float32)
    close = np.zeros((f,) + la.shape, np.float32)
    for k, found in enumerate(storms):
        for s_lat, s_lon, _, vmax in found:
            d = great_circle_km(la, lo, s_lat, s_lon)
            su = float(np.nanmean(upper["u200"][k] - upper["u850"][k]))
            sv = float(np.nanmean(upper["v200"][k] - upper["v850"][k]))
            rain[k] = np.maximum(rain[k], parametric_rain(d, vmax, su, sv, _bearing(s_lat, s_lon, la, lo)))
            close[k] = np.maximum(close[k], np.exp(-d / 300.0))
    return np.stack([np.log1p(rain), close], 1)


__all__ = ["cyclone_features", "find_cyclones", "parametric_rain"]
