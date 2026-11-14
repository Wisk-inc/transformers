# Copyright 2026 Nathan. Apache-2.0.
"""
Diagnostics computed from the driving fields before the network sees them: where the fronts and shear lines are,
when each one (and each band of rain the driving model shows) will reach every point, where island heating will
pull the afternoon storms, where the sun is, and which kind of weather is making the rain.

Everything here is NumPy on the small coarse grids (1 degree hourly, 1.5 degrees six-hourly), so it runs in the
data workers for free. Two pieces are experimental:

* **arrival-time features** -- for every point and frame, the hours until the driving model brings a front or
  shear line within 75 km, and the hours until it brings rain. In training the "driving model" is the reanalysis
  itself (perfect prognosis), live it is the GFS or ECMWF forecast. The network gets a direct timing prior instead
  of having to infer motion from frames.
* **forecast-error emulation** -- during training on reanalysis, the predictor frames for the future are displaced,
  blurred and rescaled by errors that grow with lead time (about 25 km at the start, 170 km at three days), so a
  model trained on a perfect atmosphere still learns to hedge as a real forecast would require.
"""

from __future__ import annotations

import math

import numpy as np

from .util import calendar

HOURLY_VARS = ("total_precipitation", "total_column_water_vapour", "10m_u_component_of_wind",
               "10m_v_component_of_wind", "2m_temperature", "mean_sea_level_pressure", "total_cloud_cover")
HOURLY_SHORT = ("tp", "tcwv", "u10", "v10", "t2m", "msl", "tcc")
UPPER_VARS = ("u_component_of_wind", "v_component_of_wind", "specific_humidity", "temperature")
UPPER_SHORT = ("u", "v", "q", "t")
LEVELS = (850, 700, 500, 200)
#: Channels the coarse hourly encoder reads per frame (after :func:`hourly_channels`): version 1's fourteen, then the
#: convective environment of :mod:`pluvium.convection` (version 2).
HOURLY_BASE = ("tp", "tcwv", "u10", "v10", "t2m", "msl", "tcc", "theta_e850", "theta_e_grad", "frontness",
               "shear_line", "front_eta", "rain_eta", "cos_zenith")
CONVECTIVE = ("k_index", "total_totals", "theta_e_diff", "lifted_index", "rh700", "mfc850", "shear_850_200", "div200")
HOURLY_CHANNELS = HOURLY_BASE + CONVECTIVE
UPPER_CHANNELS = tuple(f"{v}{p}" for v in UPPER_SHORT for p in LEVELS)
REGIMES = ("cold_front", "tropical_wave", "tropical_cyclone", "afternoon_storms", "trade_wind_showers", "other")
REGIME_TEXT = {"cold_front": "cold front / shear line", "tropical_wave": "tropical wave",
               "tropical_cyclone": "tropical cyclone", "afternoon_storms": "afternoon thunderstorms",
               "trade_wind_showers": "trade-wind showers", "other": "other"}
KM_PER_DEG = 111.195
REF_T = {850: 290.0, 700: 282.0, 500: 267.0, 200: 220.0}


# ------------------------------------------------------------------------------------------------ sun --

def cos_zenith(lat, lon, hours) -> np.ndarray:
    """Cosine of the solar zenith angle at the middle of each hour (clipped at 0 at night). Broadcasts."""
    lat = np.radians(np.asarray(lat, np.float64))
    lon = np.asarray(lon, np.float64)
    hours = np.asarray(hours, np.float64) + 0.5
    day = hours / 24.0
    doy = (day - 10957.0) % 365.2425                                      # days since 2000-01-01, wrapped
    decl = np.radians(23.44) * np.sin(2 * np.pi * (doy - 80.0) / 365.2425)
    solar_hour = (hours % 24.0) + lon / 15.0
    angle = np.radians(15.0 * (solar_hour - 12.0))
    c = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(angle)
    return np.clip(c, 0.0, None)


def local_solar_hour(lon, hours) -> np.ndarray:
    return (np.asarray(hours, np.float64) + 0.5 + np.asarray(lon, np.float64) / 15.0) % 24.0


def time_features(lat, lon, hours) -> np.ndarray:
    """``(..., 6)``: cos zenith, sin/cos local solar hour, sin/cos day of year, and the season of fronts (Nov-Apr)."""
    lst = local_solar_hour(lon, hours)
    doy = calendar(np.asarray(hours).ravel())["doy"].reshape(np.shape(hours)).astype(np.float64)
    month = calendar(np.asarray(hours).ravel())["month"].reshape(np.shape(hours))
    cold = np.isin(month, (11, 12, 1, 2, 3, 4)).astype(np.float64)
    shape = np.broadcast(np.asarray(lat), np.asarray(lon), np.asarray(hours)).shape
    cols = [cos_zenith(lat, lon, hours), np.sin(2 * np.pi * lst / 24), np.cos(2 * np.pi * lst / 24),
            np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), cold]
    return np.stack([np.broadcast_to(c, shape) for c in cols], axis=-1).astype(np.float32)


# ------------------------------------------------------------------------------------------- gradients --

def _grad(field: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(d/dx, d/dy)`` per 100 km of ``(..., H, W)`` fields on a regular lat/lon grid (rows = lat, ascending)."""
    dy = (lat[1] - lat[0]) * KM_PER_DEG / 100.0
    dx = (lon[1] - lon[0]) * KM_PER_DEG / 100.0 * np.cos(np.radians(lat))[:, None]
    gy, gx = np.gradient(field, axis=(-2, -1))
    return gx / dx, gy / dy


def theta_e(t: np.ndarray, q: np.ndarray, pressure_hpa: float = 850.0) -> np.ndarray:
    """Equivalent potential temperature (K), the simple form ``(T + L q / c_p)(1000 / p)^0.286``."""
    return (t + 2.5e6 * q / 1004.0) * (1000.0 / pressure_hpa) ** 0.286


def frontal_diagnostics(theta: np.ndarray, u10: np.ndarray, v10: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> dict:
    """
    ``theta`` (850 hPa theta-e), ``u10``/``v10`` as ``(F, H, W)``:

    * ``theta_e_grad`` -- |grad theta-e| in K per 100 km;
    * ``frontness`` -- the thermal front parameter ``-grad|grad theta| . n`` (positive on the cold-air edge of a
      frontal zone) times the gradient, scaled to about 1 on a sharp front;
    * ``shear_line`` -- cyclonic shear times convergence of the 10 m wind, the signature of a Caribbean shear line
      (a front's remnant, its thermal contrast gone, its wind shift and its rain still there), about 1 when strong.
    """
    gx, gy = _grad(theta, lat, lon)
    mag = np.hypot(gx, gy)
    mx, my = _grad(mag, lat, lon)
    tfp = -(mx * gx + my * gy) / (mag + 1e-6)
    frontness = np.clip(tfp, 0, None) * mag / 0.5
    ux, uy = _grad(u10, lat, lon)
    vx, vy = _grad(v10, lat, lon)
    vort = vx - uy
    conv = -(ux + vy)
    shear = np.clip(vort, 0, None) * np.clip(conv, 0, None) / 4.0
    return {"theta_e_grad": mag.astype(np.float32), "frontness": frontness.astype(np.float32),
            "shear_line": shear.astype(np.float32), "vorticity": vort.astype(np.float32)}


def distance_km(mask: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Distance (km) from every cell to the nearest True cell of ``(F, H, W)`` masks (1e4 where a frame has none)."""
    from scipy import ndimage

    dy = (lat[1] - lat[0]) * KM_PER_DEG
    dx = (lon[1] - lon[0]) * KM_PER_DEG * math.cos(math.radians(float(np.mean(lat))))
    out = np.full(mask.shape, 1e4, np.float32)
    for k in range(mask.shape[0]):
        if mask[k].any():
            out[k] = ndimage.distance_transform_edt(~mask[k], sampling=(dy, dx))
    return out


def arrival_hours(present: np.ndarray, hours: np.ndarray, horizon: float = 72.0) -> np.ndarray:
    """For ``present (F, H, W)`` booleans at frame ``hours (F,)``: the hours from each frame until the first frame at
    or after it where the cell is present (``horizon`` when never within the window)."""
    eta = np.full(present.shape, horizon, np.float32)
    nxt = np.full(present.shape[1:], np.inf)
    for k in range(present.shape[0] - 1, -1, -1):
        nxt = np.where(present[k], hours[k], nxt)
        eta[k] = np.minimum(nxt - hours[k], horizon)
    return eta


# ----------------------------------------------------------------------------------------- the frames --

def hourly_channels(fields: dict, upper: dict, lat: np.ndarray, lon: np.ndarray, hours: np.ndarray,
                    front_km: float = 75.0) -> np.ndarray:
    """
    The coarse encoder's input ``(F, len(HOURLY_CHANNELS), H, W)``, normalised to order one, from raw fields
    (``fields[short] (F, H, W)`` in SI units, tp in mm/h) and 850 hPa temperature and humidity already on this grid.
    """
    tp = np.clip(fields["tp"], 0, None)
    th = theta_e(upper["t850"], upper["q850"], 850.0)
    diag = frontal_diagnostics(th, fields["u10"], fields["v10"], lat, lon)
    front = (diag["frontness"] > 0.5) | (diag["shear_line"] > 0.5)
    near_front = distance_km(front, lat, lon) < front_km
    rain = tp >= 0.3
    eta_front = arrival_hours(near_front, hours)
    eta_rain = arrival_hours(rain, hours)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    cz = cos_zenith(la[None], lo[None], hours[:, None, None])
    out = [np.log1p(tp), fields["tcwv"] / 50.0 - 1.0, fields["u10"] / 10.0, fields["v10"] / 10.0,
           (fields["t2m"] - 298.0) / 5.0, (fields["msl"] - 101500.0) / 500.0, fields["tcc"], (th - 340.0) / 10.0,
           diag["theta_e_grad"] / 2.0, np.clip(diag["frontness"], 0, 5), np.clip(diag["shear_line"], 0, 5),
           eta_front / 24.0, eta_rain / 24.0, cz]
    base = np.stack(out, axis=1).astype(np.float32)
    from .convection import NEEDS, convective_channels

    if all(k in upper for k in NEEDS):
        return np.concatenate([base, convective_channels(upper, lat, lon)], axis=1)
    return np.concatenate([base, np.zeros((base.shape[0], len(CONVECTIVE)) + base.shape[2:], np.float32)], axis=1)


def upper_channels(upper: dict) -> np.ndarray:
    """``(F, 16, H, W)`` normalised u, v, q, t at 850/700/500/200 hPa."""
    out = []
    for name in UPPER_SHORT:
        for p in LEVELS:
            x = upper[f"{name}{p}"]
            if name in ("u", "v"):
                out.append(x / 10.0)
            elif name == "q":
                out.append(x * 1000.0 / 10.0)
            else:
                out.append((x - REF_T[p]) / 5.0)
    return np.stack(out, axis=1).astype(np.float32)


def regime_flags(fields: dict, upper: dict, lat: np.ndarray, lon: np.ndarray, hours: np.ndarray,
                 storms: list | None = None) -> np.ndarray:
    """
    Weak labels of the weather making rain, ``(F, 4, H, W)`` booleans: tropical cyclone, cold front / shear line,
    tropical wave, trade-wind flow. They are rules, not truth: a cyclone within 500 km (best-track positions when
    given, else a deep low with strong vorticity); a front or shear line within 200 km in the season of fronts
    (October-May); a tropical-wave trough at 700 hPa (southerly turning to northerly westward, in easterlies, June-
    November) within 300 km; easterly 10 m wind. :func:`regime_labels` adds land and the hour for afternoon storms.
    """
    th = theta_e(upper["t850"], upper["q850"], 850.0)
    diag = frontal_diagnostics(th, fields["u10"], fields["v10"], lat, lon)
    month = calendar(hours)["month"]
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    tc = np.zeros(fields["u10"].shape, bool)
    if storms:
        from .util import great_circle_km

        for k, h in enumerate(hours):
            for s_lat, s_lon in [(a, b) for (t, a, b) in storms if abs(t - h) <= 3]:
                tc[k] |= great_circle_km(la, lo, s_lat, s_lon) < 500.0
    else:
        deep = (fields["msl"] < 100400.0) & (diag["vorticity"] > 3.0)
        tc = distance_km(deep, lat, lon) < 300.0
    front = (diag["frontness"] > 0.5) | (diag["shear_line"] > 0.5)
    front = (distance_km(front, lat, lon) < 200.0) & np.isin(month, (10, 11, 12, 1, 2, 3, 4, 5))[:, None, None]
    vx, _ = _grad(upper["v700"], lat, lon)
    trough = (vx > 0.5) & (np.abs(upper["v700"]) < 4.0) & (upper["u700"] < -2.0)
    wave = (distance_km(trough, lat, lon) < 300.0) & np.isin(month, (6, 7, 8, 9, 10, 11))[:, None, None]
    trade = fields["u10"] < -2.0
    return np.stack([tc, front, wave, trade], axis=1)


def regime_labels(flags: np.ndarray, land: np.ndarray, lon: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """Per point and frame: ``flags (F, 4, N)`` sampled at points, ``land (N,)`` -> ``(F, N)`` indices into REGIMES,
    in priority order cyclone > front > wave > afternoon storms (land, 11:00-19:00 solar) > trade showers > other."""
    f, _, n = flags.shape
    out = np.full((f, n), REGIMES.index("other"), np.int64)
    lst = local_solar_hour(lon[None, :], hours[:, None])
    afternoon = land[None, :] & (lst >= 11.0) & (lst <= 19.0)
    order = [(flags[:, 3], "trade_wind_showers"), (afternoon, "afternoon_storms"), (flags[:, 2], "tropical_wave"),
             (flags[:, 1], "cold_front"), (flags[:, 0], "tropical_cyclone")]
    for mask, name in order:  # later entries win
        out = np.where(mask, REGIMES.index(name), out)
    return out


# --------------------------------------------------------------------------------- island heating --

def heating_potential(terrain, lat: np.ndarray, lon: np.ndarray, u: np.ndarray, v: np.ndarray,
                      taus_h=(1.0, 3.0)) -> np.ndarray:
    """
    Island-heating convergence potential (experimental): the difference of land fraction smoothed at 15 and 40 km
    (positive over the heart of an island, where daytime heating pulls the sea breezes together) read upwind by the
    distance the background wind carries air in ``taus_h`` hours -- the convergence line, and the afternoon storms
    with it, drift downwind of the island's centre. ``u``/``v`` (m/s) are ``(F, N)`` at the points; returns ``(F, N,
    len(taus_h))``.
    """
    dog = terrain.smooth["land_15km"] - terrain.smooth["land_40km"]
    out = np.zeros(u.shape + (len(taus_h),), np.float32)
    for k, tau in enumerate(taus_h):
        dist = tau * 3.6                                                   # km per (m/s) in tau hours
        shifted_lat = lat[None, :] - v * dist / KM_PER_DEG
        shifted_lon = lon[None, :] - u * dist / (KM_PER_DEG * np.cos(np.radians(lat))[None, :])
        out[..., k] = terrain.sample(dog, shifted_lat.ravel(), shifted_lon.ravel()).reshape(u.shape) * 4.0
    return out


# ----------------------------------------------------------------------------- error emulation --

def emulate_forecast_error(hourly: np.ndarray, upper: np.ndarray, leads: np.ndarray, rng: np.random.Generator,
                           cell_km: float, upper_cell_km: float, scale: float | None = None) -> tuple:
    """
    Make perfect predictors look like a forecast (experimental). Frames with ``leads >= 0`` get a coherent random
    displacement whose spread grows from ``25 km`` to ``25 + 2 * lead`` km (an AR(1) walk, correlation time one day),
    a lognormal error on the rain channel and a blur that both grow with lead. ``scale`` (drawn from U(0, 1.5) when
    None) sets how wrong this forecast is; 0 leaves it perfect. Returns the perturbed copies and the scale used.
    """
    from scipy import ndimage

    scale = float(rng.uniform(0.0, 1.5)) if scale is None else float(scale)
    hourly, upper = hourly.copy(), upper.copy()
    if scale <= 0:
        return hourly, upper, scale
    x = rng.standard_normal(2)
    previous = None
    for k, lead in enumerate(leads):
        if lead < 0:
            continue
        if previous is not None:
            rho = math.exp(-(lead - previous) / 24.0)
            x = rho * x + math.sqrt(max(0.0, 1 - rho**2)) * rng.standard_normal(2)
        previous = lead
        sigma_km = scale * (25.0 + 2.0 * lead)
        shift = x * sigma_km
        for frame, km in ((hourly[k], cell_km), (upper[k], upper_cell_km)):
            for c in range(frame.shape[0]):
                frame[c] = ndimage.shift(frame[c], (shift[1] / km, shift[0] / km), order=1, mode="nearest")
        blur = scale * (0.2 + 0.01 * lead)
        if blur > 0.05:
            hourly[k, 0] = ndimage.gaussian_filter(hourly[k, 0], blur)
        hourly[k, 0] = np.log1p(np.expm1(np.clip(hourly[k, 0], 0, None)) *
                                math.exp(rng.normal(0.0, scale * (0.1 + 0.005 * lead))))
    return hourly, upper, scale


__all__ = ["CONVECTIVE", "HOURLY_BASE", "HOURLY_CHANNELS", "HOURLY_SHORT", "HOURLY_VARS", "LEVELS", "REGIMES", "REGIME_TEXT", "UPPER_CHANNELS",
           "UPPER_SHORT", "UPPER_VARS", "arrival_hours", "cos_zenith", "distance_km", "emulate_forecast_error",
           "frontal_diagnostics", "heating_potential", "hourly_channels", "local_solar_hour", "regime_flags",
           "regime_labels", "theta_e", "time_features", "upper_channels"]
