# Copyright 2026 Nathan. Apache-2.0.
"""
Ground truth from the ground: observations that keep coming when the storm is on top of the island.

Two operators, ready for feeds that do not yet exist in this package (no telecom or crowd-sourced data is read
here; nothing was trained on any):

* :func:`centre_from_pressures` -- a swarm of barometers (phones, watches, home weather stations: any set of
  surface-pressure readings with positions) turned into a **centre fix with an uncertainty and a central pressure**
  by fitting the Holland pressure profile to the readings. Pressure is the quantity that survives a landfall best:
  a barometer indoors reads the same storm as one outdoors. The fix goes into the swarm like any satellite fix
  (:class:`naturev1.swarm.fixes.Fix`), and the central pressure gives an intensity through a pressure-wind relation.
* :func:`nudge_wind_map` -- point wind observations (anemometers, and one day tower accelerometers or link
  attenuation once someone calibrates those into wind) blended into the painted wind map by optimal
  interpolation: each observation corrects the map around it over a correlation length, by as much as its own
  error allows.

Both are checked on constructed storms (:mod:`tests.test_observations`). What they are not: a trained inference
from signal loss to wind. That needs the data first.
"""

from __future__ import annotations

import numpy as np

from ..cyclone.geometry import great_circle_km

#: Holland profile shape used when the readings cannot pin it down.
DEFAULT_B = 1.3


def holland_pressure(r_km, pc_hpa, penv_hpa, rmw_km, b=DEFAULT_B):
    """Surface pressure at radius ``r_km`` of a Holland (1980) vortex."""
    r = np.maximum(np.asarray(r_km, np.float64), 1.0)
    return pc_hpa + (penv_hpa - pc_hpa) * np.exp(-((rmw_km / r) ** b))


def pressure_to_wind_kt(pc_hpa, penv_hpa=1010.0) -> float:
    """Atkinson-Holliday (1977) pressure-wind relation, kt: ``6.7 (penv - pc)^0.644`` (3.44 in m/s)."""
    return float(6.7 * max(penv_hpa - pc_hpa, 0.0) ** 0.644)


def centre_from_pressures(lat, lon, pressure_hpa, guess=None, sigma_hpa: float = 1.0, penv_hpa: float | None = None,
                          search_deg: float = 1.5, step_deg: float = 0.05, rmw_grid_km=(15.0, 25.0, 35.0, 50.0, 70.0, 100.0),
                          b: float = DEFAULT_B) -> dict:
    """
    Fit a Holland vortex to point pressure readings.

    Args:
        lat, lon, pressure_hpa: ``(P,)`` readings (10 or more; indoor readings are fine -- pressure goes through walls).
        guess: ``(lat, lon)`` to search around (the readings' pressure-weighted centre when None).
        sigma_hpa: the error of one reading (phones: about 1 hPa after their own bias is removed).
        penv_hpa: environmental pressure; the 90th percentile of the readings when None.
        search_deg, step_deg: the search box and its grid; the best cell is refined with a local parabola.

    Returns ``{"lat", "lon", "sigma_km", "pc_hpa", "rmw_km", "wind_kt", "rmse_hpa", "n"}``.
    """
    la, lo, p = (np.asarray(x, np.float64) for x in (lat, lon, pressure_hpa))
    ok = np.isfinite(la) & np.isfinite(lo) & np.isfinite(p)
    la, lo, p = la[ok], lo[ok], p[ok]
    if la.size < 5:
        raise ValueError("at least five readings are needed to place a centre")
    penv = float(np.quantile(p, 0.9)) if penv_hpa is None else float(penv_hpa)
    if guess is None:
        w = np.maximum(penv - p, 0.0) ** 2 + 1e-6
        guess = (float(np.sum(w * la) / w.sum()), float(np.sum(w * lo) / w.sum()))
    lats = np.arange(guess[0] - search_deg, guess[0] + search_deg + 1e-9, step_deg)
    lons = np.arange(guess[1] - search_deg, guess[1] + search_deg + 1e-9, step_deg)
    best = None
    cost = np.full((lats.size, lons.size), np.inf)
    for i, cla in enumerate(lats):
        d = great_circle_km(cla, lons[:, None], la[None, :], lo[None, :])                  # (lons, P)
        for j in range(lons.size):
            for rmw in rmw_grid_km:
                f = np.exp(-((rmw / np.maximum(d[j], 1.0)) ** b))                             # p = pc + dp f: linear in (pc, dp)
                a = np.stack([np.ones_like(f), f], 1)
                coef, *_ = np.linalg.lstsq(a, p, rcond=None)
                pc, dp = coef
                if dp < 0 or pc < 850:
                    continue
                resid = p - (pc + dp * f)
                c = float(np.sum(resid ** 2))
                if c < cost[i, j]:
                    cost[i, j] = c
                    if best is None or c < best[0]:
                        best = (c, cla, lons[j], pc, pc + dp, rmw)
    if best is None:
        raise ValueError("no vortex fits these readings")
    c0, cla, clo, pc, penv_fit, rmw = best
    i, j = np.unravel_index(int(np.argmin(cost)), cost.shape)
    # the radius of maximum wind refined around the best grid value (it sets how deep the fitted centre is)
    d = great_circle_km(cla, clo, la, lo)
    for r_try in np.arange(max(rmw * 0.6, 8.0), rmw * 1.5 + 1e-9, max(rmw * 0.05, 1.0)):
        f = np.exp(-((r_try / np.maximum(d, 1.0)) ** b))
        a = np.stack([np.ones_like(f), f], 1)
        coef, *_ = np.linalg.lstsq(a, p, rcond=None)
        c = float(np.sum((p - a @ coef) ** 2))
        if c < c0 and coef[1] > 0:
            c0, pc, penv_fit, rmw = c, float(coef[0]), float(coef[0] + coef[1]), float(r_try)
    # sub-grid refinement: a parabola through the three cells in each direction
    def refine(axis_vals, k, along):
        if 0 < k < along.size - 1 and np.isfinite(along[k - 1]) and np.isfinite(along[k + 1]):
            denom = along[k - 1] - 2 * along[k] + along[k + 1]
            if denom > 0:
                return float(axis_vals[k] + 0.5 * (along[k - 1] - along[k + 1]) / denom * (axis_vals[1] - axis_vals[0]))
        return float(axis_vals[k])
    lat_c, lon_c = refine(lats, i, cost[:, j]), refine(lons, j, cost[i, :])
    rmse = float(np.sqrt(c0 / la.size))
    # the uncertainty of the centre: where the cost rises by one reading-variance above the minimum
    threshold = c0 + sigma_hpa ** 2 * max(1.0, la.size / 10.0)
    within = cost <= threshold
    ii, jj = np.nonzero(within)
    sigma_km = max(float(np.mean(great_circle_km(lats[ii], lons[jj], lat_c, lon_c))), 0.5 * step_deg * 111.0) if ii.size > 1 else step_deg * 111.0
    return {"lat": lat_c, "lon": lon_c, "sigma_km": sigma_km, "pc_hpa": float(pc), "penv_hpa": float(penv_fit), "rmw_km": float(rmw),
            "wind_kt": pressure_to_wind_kt(pc, penv_fit), "rmse_hpa": rmse, "n": int(la.size)}


def fix_from_pressures(time, lat, lon, pressure_hpa, source: str = "BARO", **kwargs):
    """A :class:`naturev1.swarm.fixes.Fix` from barometer readings, for the swarm's filter."""
    from .fixes import Fix

    fit = centre_from_pressures(lat, lon, pressure_hpa, **kwargs)
    return Fix(np.datetime64(time, "m"), fit["lat"], fit["lon"], fit["sigma_km"], source, fit["wind_kt"]), fit


def nudge_wind_map(grid: np.ndarray, grid_lat: np.ndarray, grid_lon: np.ndarray, obs_lat, obs_lon, obs_kt,
                   length_km: float = 30.0, obs_error_kt: float = 5.0, background_error_kt: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Optimal interpolation of point wind observations into a gridded wind map (kt): the map is corrected by the
    observations' innovations, spread with a Gaussian correlation of scale ``length_km`` and weighted by the ratio of
    background to observation error. Returns ``(corrected grid, innovations (O,))``.
    """
    g = np.asarray(grid, np.float64)
    ola, olo, okt = (np.asarray(x, np.float64) for x in (obs_lat, obs_lon, obs_kt))
    ok = np.isfinite(ola) & np.isfinite(olo) & np.isfinite(okt)
    ola, olo, okt = ola[ok], olo[ok], okt[ok]
    if ola.size == 0:
        return g.copy(), np.zeros(0)
    fi = np.clip((ola - grid_lat[0]) / (grid_lat[1] - grid_lat[0]), 0, grid_lat.size - 1.001)
    fj = np.clip((olo - grid_lon[0]) / (grid_lon[1] - grid_lon[0]), 0, grid_lon.size - 1.001)
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    di, dj = fi - i0, fj - j0
    gg = np.nan_to_num(g, nan=0.0)
    at_obs = ((1 - di) * (1 - dj) * gg[i0, j0] + (1 - di) * dj * gg[i0, j0 + 1] + di * (1 - dj) * gg[i0 + 1, j0] + di * dj * gg[i0 + 1, j0 + 1])
    innovation = okt - at_obs
    d_oo = great_circle_km(ola[:, None], olo[:, None], ola[None, :], olo[None, :])
    b_oo = background_error_kt ** 2 * np.exp(-0.5 * (d_oo / length_km) ** 2)
    weights = np.linalg.solve(b_oo + obs_error_kt ** 2 * np.eye(ola.size), innovation)
    lon2, lat2 = np.meshgrid(grid_lon, grid_lat)
    correction = np.zeros_like(g)
    for k in range(ola.size):
        d = great_circle_km(lat2, lon2, ola[k], olo[k])
        correction += weights[k] * background_error_kt ** 2 * np.exp(-0.5 * (d / length_km) ** 2)
    out = np.where(np.isfinite(g), np.maximum(g + correction, 0.0), g)
    return out, innovation


__all__ = ["DEFAULT_B", "centre_from_pressures", "fix_from_pressures", "holland_pressure", "nudge_wind_map", "pressure_to_wind_kt"]
