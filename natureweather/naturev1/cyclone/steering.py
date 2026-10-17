# Copyright 2026 Nathan. Apache-2.0.
"""
The flow that carries the storm: layer means of the environmental wind, averaged over a disc that moves
with it, for a family of layers.

The idea: no single layer steers every storm. A shallow, sheared depression is carried by the lower
troposphere; a deep major hurricane by nearly the whole column (Velden and Leslie 1991); NHC has run three
beta-advection models for decades for exactly this reason (BAMS 850-700, BAMM 850-400, BAMD 850-200). Here:

* :data:`LAYERS`: eight layers from 925-700 up to 925-150, all pressure-weighted from the 13 analysis levels
  (log-pressure interpolation at the boundaries) -- finer than the three BAMs.
* :func:`intensity_layer`: the layer an intensity calls for, interpolated continuously in wind speed rather
  than in three steps.
* :func:`optimal_layer` (Galarneau and Davis 2013's "optimal steering" idea, run forwards): the layer and disc
  radius whose environmental flow best matches the storm's *observed* recent motion. A storm tells you which
  layer is steering it by how it is moving; that choice then carries the forecast.

The disc average uses an equal-area polar stencil around the (moving) position, sampled bilinearly from
the vortex-removed layer mean, so it is a true geodesic disc at any latitude.
"""

from __future__ import annotations

import numpy as np

from .fields import Environment
from .geometry import destination

#: (name, top hPa, bottom hPa).
LAYERS = (("925-700", 700, 925), ("850-500", 500, 850), ("850-400", 400, 850), ("850-300", 300, 850),
          ("850-250", 250, 850), ("850-200", 200, 850), ("925-200", 200, 925), ("925-150", 150, 925))
#: Default steering disc radius, km.
STEERING_RADIUS_KM = 500.0
#: Intensity (kt) -> layer top (hPa): the continuous Velden-Leslie relation.
INTENSITY_TOPS = ((30.0, 600.0), (50.0, 500.0), (70.0, 400.0), (90.0, 300.0), (120.0, 250.0), (150.0, 200.0))


def polar_stencil(radius_km: float, rings: int = 4, azimuths: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Equal-area (distance, bearing) sample points filling a disc, centre included."""
    distances, bearings = [0.0], [0.0]
    for i in range(rings):
        r = radius_km * np.sqrt((i + 0.5) / rings)
        for j in range(azimuths):
            distances.append(r)
            bearings.append((360.0 * (j + 0.5 * (i % 2)) / azimuths) % 360.0)
    return np.asarray(distances), np.asarray(bearings)


def disc_mean(env: Environment, field: np.ndarray, lat, lon, radius_km: float = STEERING_RADIUS_KM) -> np.ndarray:
    """Mean of a ``(H, W)`` field over a geodesic disc around each (lat, lon); broadcasting over points."""
    lat = np.atleast_1d(np.asarray(lat, np.float64))
    lon = np.atleast_1d(np.asarray(lon, np.float64))
    distances, bearings = polar_stencil(radius_km)
    plat, plon = destination(lat[:, None], lon[:, None], bearings[None, :], distances[None, :])
    return env.sample(field, plat, plon).mean(-1)


def _grid_stencil(env: Environment, radius_km: float):
    """Bilinear weights of every grid point's disc stencil, cached on the environment per radius."""
    cache = env.__dict__.setdefault("_stencils", {})
    if radius_km not in cache:
        lat_g, lon_g = np.meshgrid(env.lat, env.lon, indexing="ij")
        distances, bearings = polar_stencil(radius_km)
        plat, plon = destination(lat_g.ravel()[:, None], lon_g.ravel()[:, None], bearings[None, :], distances[None, :])
        fi, fj = env._fractional_index(plat, plon)
        i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
        cache[radius_km] = (i0, j0, fi - i0, fj - j0, lat_g.shape)
    return cache[radius_km]


def disc_smooth(env: Environment, field: np.ndarray, radius_km: float = STEERING_RADIUS_KM) -> np.ndarray:
    """The geodesic disc mean of ``field`` evaluated at every grid point of the box, ``(H, W)``."""
    i0, j0, di, dj, shape = _grid_stencil(env, radius_km)
    values = ((1 - di) * (1 - dj) * field[i0, j0] + (1 - di) * dj * field[i0, j0 + 1]
              + di * (1 - dj) * field[i0 + 1, j0] + di * dj * field[i0 + 1, j0 + 1])
    return values.mean(-1).reshape(shape)


class SteeringFlow:
    """
    Layer-mean environmental winds of one analysis, ready to sample anywhere: ``velocity(lat, lon)`` returns
    ``(u, v)`` m/s averaged over the steering disc. The disc mean is evaluated once at every grid point and then
    interpolated, so a trajectory step costs one bilinear sample instead of a stencil.
    """

    def __init__(self, env: Environment, top: float, bottom: float, radius_km: float = STEERING_RADIUS_KM,
                 previous: bool = False) -> None:
        self.env, self.radius_km = env, radius_km
        self.top, self.bottom = top, bottom
        self.u = disc_smooth(env, env.layer_mean("u", top, bottom, previous=previous), radius_km)
        self.v = disc_smooth(env, env.layer_mean("v", top, bottom, previous=previous), radius_km)

    def velocity(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        lat = np.atleast_1d(np.asarray(lat, np.float64))
        lon = np.atleast_1d(np.asarray(lon, np.float64))
        return self.env.sample(self.u, lat, lon), self.env.sample(self.v, lat, lon)


def intensity_layer(wind_kt: float) -> tuple[float, float]:
    """``(top, bottom)`` hPa for a storm of ``wind_kt``, continuous in wind speed."""
    winds = [w for w, _ in INTENSITY_TOPS]
    tops = [t for _, t in INTENSITY_TOPS]
    top = float(np.interp(wind_kt if np.isfinite(wind_kt) else 50.0, winds, tops))
    return top, 850.0


def layer_velocities(env: Environment, lat0: float, lon0: float, radii=(300.0, 500.0, 700.0)) -> dict:
    """``{(layer name, radius): (u, v)}`` at the storm for every layer and disc radius."""
    out = {}
    for name, top, bottom in LAYERS:
        u = env.layer_mean("u", top, bottom)
        v = env.layer_mean("v", top, bottom)
        for radius in radii:
            out[(name, radius)] = (float(disc_mean(env, u, lat0, lon0, radius)[0]),
                                   float(disc_mean(env, v, lat0, lon0, radius)[0]))
    return out


def optimal_layer(env: Environment, lat0: float, lon0: float, motion_ms: tuple[float, float],
                  radii=(300.0, 500.0, 700.0), prior: tuple[float, float] | None = None,
                  prior_weight: float = 1.0) -> dict:
    """
    The layer and radius whose environmental flow at the storm best matches its observed motion.

    The mismatch is the vector difference, in m/s; with ``prior`` (an intensity layer's ``(top, bottom)``) a
    penalty of ``prior_weight`` m/s per 100 hPa of distance between layer tops keeps the choice from jumping to
    an implausible layer on a coincidence. Returns ``{"layer", "top", "bottom", "radius", "mismatch_ms", "u", "v"}``.
    """
    candidates = layer_velocities(env, lat0, lon0, radii)
    tops = {name: (top, bottom) for name, top, bottom in LAYERS}
    best = None
    for (name, radius), (u, v) in candidates.items():
        miss = float(np.hypot(u - motion_ms[0], v - motion_ms[1]))
        score = miss + (prior_weight * abs(tops[name][0] - prior[0]) / 100.0 if prior is not None else 0.0)
        if best is None or score < best[0]:
            best = (score, name, radius, miss, u, v)
    _, name, radius, miss, u, v = best
    return {"layer": name, "top": tops[name][0], "bottom": tops[name][1], "radius": radius, "mismatch_ms": miss,
            "u": u, "v": v}


__all__ = ["INTENSITY_TOPS", "LAYERS", "STEERING_RADIUS_KM", "SteeringFlow", "disc_mean", "disc_smooth", "intensity_layer",
           "layer_velocities", "optimal_layer", "polar_stencil"]
