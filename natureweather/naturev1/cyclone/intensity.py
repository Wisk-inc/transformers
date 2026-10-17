# Copyright 2026 Nathan. Apache-2.0.
"""
Intensity without a neural network: three statistical-dynamical models that read the atmosphere along the
*forecast track*, not just where the storm is now.

The idea: an intensity forecast is a forecast of the conditions the storm will meet -- warmer or cooler water,
more or less shear, drier air, land -- and the track forecast says where it will meet them. So every
environmental predictor is sampled along the physics consensus track, hour by hour, from the analysis
(:func:`environment_along_track`), with two corrections no reanalysis carries: the **cold wake** under a slow
storm (:mod:`naturev1.cyclone.ocean`) lowers the sea-surface temperature the core feels, and **land and
terrain** (:mod:`naturev1.cyclone.land`) are applied as decay after the fact.

* :func:`shifor_intensity` -- SHIFOR (Jarvinen and Neumann 1979) rebuilt like CLIPER: climatology and
  persistence (current wind, its 12- and 24-hour trends, position, season, motion), per basin, 1980-2014.
* :func:`ships_intensity` -- SHIPS-style (DeMaria and Kaplan 1994): the change at each lead regressed on
  predictors averaged along the track up to that lead -- potential intensity minus the current wind (the
  room to grow), shear in two layers, mid-level humidity, effective ocean temperature, land, persistence.
* :func:`lgem_intensity` -- LGEM-style (DeMaria 2009): ``dV/dt = kappa V - beta (V / V_pi)^n V`` integrated
  hourly along the track, the growth rate ``kappa`` a linear function of the environment, the potential
  intensity a hard ceiling the storm approaches logistically.

Coefficients: SHIFOR on 1980-2014 best tracks; SHIPS and LGEM on the 2016/2017/2019 hindcast seasons -- never
on the 2018/2020-2022 seasons the benchmark reports.
"""

from __future__ import annotations

import numpy as np

from .geometry import great_circle_km, wrap_longitude
from .land import apply_land_decay, land_along_track
from .ocean import effective_sst

KT_PER_MS = 1.0 / 0.514444
SHIFOR_PREDICTORS = ("one", "wind", "dwind12", "dwind24", "lat", "lon", "season_s", "season_c", "u6", "v6", "wind2",
                     "wind_lat")


def _season_phase(case) -> float:
    day = (np.datetime64(case.time, "D") - np.datetime64(str(case.time)[:4] + "-01-01", "D")).astype(int)
    shift = 182 if case.lat < 0 else 0
    return 2 * np.pi * ((day + shift) % 365) / 365.0


def shifor_predictors(case) -> np.ndarray | None:
    from .cliper import GROUPS, _motion

    if not np.isfinite(case.wind_kt):
        return None
    w = case.history["wind"]
    w12, w24 = w[-3], w[-5]
    d12 = (case.wind_kt - w12) / 10.0 if np.isfinite(w12) else 0.0
    d24 = (case.wind_kt - w24) / 10.0 if np.isfinite(w24) else d12
    sign = -1.0 if case.lat < 0 else 1.0
    m = _motion(case, -6, 0) or _motion(case, -12, 0) or (0.0, 0.0)
    _, reference = GROUPS.get(case.basin, ("NA", -60.0))
    phase = _season_phase(case)
    wind = case.wind_kt / 100.0
    lat = abs(case.lat) / 30.0
    return np.array([1.0, wind, d12, d24, lat, wrap_longitude(case.lon - reference) / 60.0, np.sin(phase), np.cos(phase),
                     m[0] / 20.0, sign * m[1] / 20.0, wind * wind, wind * lat])


def fit_shifor(cases, leads, ridge: float = 1e-2) -> dict:
    from .cliper import group_of

    rows: dict[str, list] = {}
    for c in cases:
        x = shifor_predictors(c)
        if x is None:
            continue
        target = np.where(c.truth["valid"] & np.isfinite(c.truth["wind"]), c.truth["wind"] - c.wind_kt, np.nan)
        rows.setdefault(group_of(c), []).append((x, target))
    table = {"leads": [int(h) for h in leads]}
    for group, items in rows.items():
        x = np.stack([i[0] for i in items])
        y = np.stack([i[1] for i in items])
        weights = np.full((len(leads), x.shape[1]), np.nan)
        for k in range(len(leads)):
            ok = np.isfinite(y[:, k])
            if ok.sum() < 5 * x.shape[1]:
                continue
            a = x[ok]
            gram = a.T @ a + ridge * ok.sum() * np.diag([0.0] + [1.0] * (x.shape[1] - 1))
            weights[k] = np.linalg.solve(gram, a.T @ y[ok, k])
        table[group] = weights.round(5).tolist()
    return table


def shifor_intensity(case, leads, table: dict | None = None) -> np.ndarray:
    from .cliper import group_of

    if table is None:
        from .coefficients import SHIFOR as table
    x = shifor_predictors(case)
    out = np.full(len(leads), np.nan)
    group = table.get(group_of(case))
    if x is None or group is None:
        return out
    stored = list(table["leads"])
    w = np.asarray(group, np.float64)
    for k, lead in enumerate(leads):
        if lead in stored:
            out[k] = max(case.wind_kt + float(w[stored.index(lead)] @ x), 15.0)
    return out


# ------------------------------------------------------------------------------- along the track --

def _relative_humidity(t_k: np.ndarray, q: np.ndarray, p_hpa: float) -> np.ndarray:
    es = 6.112 * np.exp(17.67 * (t_k - 273.15) / (t_k - 29.65))
    e = q * p_hpa / (0.622 + 0.378 * q)
    return np.clip(e / es, 0.0, 1.2)


def potential_intensity_at(env, lat, lon, sst_k) -> np.ndarray:
    """Bister-Emanuel potential intensity (m/s, surface) at points, from the analysis column there."""
    import torch

    from ..thermo import potential_intensity

    levels = [p for p in env.levels if p >= 100]
    idx = [list(env.levels).index(p) for p in levels]
    t = np.stack([env.sample(env.fields["t"][k], lat, lon) for k in idx], -1)
    q = np.stack([env.sample(env.fields["q"][k], lat, lon) for k in idx], -1)
    msl = env.sample(env.fields["msl"], lat, lon) / 100.0
    out = potential_intensity(torch.tensor(np.asarray(sst_k, np.float64)), torch.tensor(msl),
                              torch.tensor(np.asarray(levels, np.float64)), torch.tensor(t), torch.tensor(q))
    return np.nan_to_num(out["vmax_ms"].numpy(), nan=0.0)


def environment_along_track(env, env_clean, case, track: np.ndarray, leads, wind_guess_kt: np.ndarray | None = None) -> dict:
    """
    Hourly environment along the forecast track: effective SST (cold wake), potential intensity, shear (850-200
    and 850-500, annulus 200-800 km), 700-500 hPa humidity, land and terrain, translation speed.
    """
    from .shear import annulus_profile

    land = land_along_track(env, case.lat, case.lon, track, leads)
    hours, la, lo = land["hours"], land["lat"], land["lon"]
    step = np.gradient(hours)
    dist = np.concatenate([[0.0], great_circle_km(la[:-1], lo[:-1], la[1:], lo[1:])])
    speed = np.convolve(dist, np.ones(6) / 6.0, mode="same") / np.maximum(step, 1e-6) / 3.6
    sst = env.sample(np.nan_to_num(env.fields["sst"], nan=float(np.nanmean(env.fields["sst"]))), la, lo)
    wind = np.interp(hours, np.concatenate([[0.0], np.asarray(leads, float)]),
                     np.concatenate([[case.wind_kt], wind_guess_kt if wind_guess_kt is not None else np.full(len(leads), case.wind_kt)]))
    sst_eff = effective_sst(sst, wind, speed, hours)
    sample_hours = np.arange(0.0, hours[-1] + 1e-9, 6.0)
    pick = np.searchsorted(hours, sample_hours).clip(0, len(hours) - 1)
    pi = potential_intensity_at(env, la[pick], lo[pick], sst_eff[pick]) * KT_PER_MS
    shear_l, shear_m, rh = [], [], []
    for j in pick:
        prof = annulus_profile(env_clean, la[j], lo[j])
        shear_l.append(np.hypot(prof[200][0] - prof[850][0], prof[200][1] - prof[850][1]))
        shear_m.append(np.hypot(prof[500][0] - prof[850][0], prof[500][1] - prof[850][1]))
        rhs = []
        for p in (700, 600, 500):
            k = list(env.levels).index(p)
            rhs.append(float(env.sample(_relative_humidity(env.fields["t"][k], env.fields["q"][k], p), la[j], lo[j])))
        rh.append(np.mean(rhs))
    return {"hours": hours, "lat": la, "lon": lo, "land": land["land"], "height_m": land["height_m"], "speed_ms": speed,
            "sst_k": sst, "sst_eff_k": sst_eff, "six_hours": sample_hours, "pi_kt": pi, "shear_ms": np.asarray(shear_l),
            "shear_mid_ms": np.asarray(shear_m), "rh_mid": np.asarray(rh), "landfall_hour": land["landfall_hour"]}


SHIPS_PREDICTORS = ("one", "wind", "dwind12", "potential", "potential2", "shear", "shear_mid", "rh", "sst", "land",
                    "wind_shear", "lat", "speed")


def ships_predictors(case, along: dict, lead: float) -> np.ndarray:
    """Predictors averaged along the track from now to ``lead``."""
    six = along["six_hours"]
    use = six <= lead + 1e-9
    hourly = along["hours"] <= lead + 1e-9
    wind = case.wind_kt
    w12 = case.history["wind"][-3]
    pi = float(np.mean(along["pi_kt"][use]))
    room = (pi - wind) / 50.0
    shear = float(np.mean(along["shear_ms"][use])) / 10.0
    return np.array([1.0, wind / 100.0, ((wind - w12) / 10.0) if np.isfinite(w12) else 0.0, room, room * abs(room),
                     shear, float(np.mean(along["shear_mid_ms"][use])) / 10.0, float(np.mean(along["rh_mid"][use])),
                     (float(np.mean(along["sst_eff_k"][hourly])) - 300.15) / 2.0, float(np.mean(along["land"][hourly])),
                     wind / 100.0 * shear, abs(case.lat) / 30.0, float(np.mean(along["speed_ms"][hourly])) / 5.0])


def fit_ships(cases, alongs: list, leads, ridge: float = 3e-2) -> dict:
    """Ridge regression of the wind change at each lead on :func:`ships_predictors`; one model for all basins."""
    table = {"leads": [int(h) for h in leads], "weights": []}
    for k, lead in enumerate(leads):
        xs, ys = [], []
        for c, a in zip(cases, alongs):
            if a is None or not c.truth["valid"][k] or not np.isfinite(c.truth["wind"][k]) or not np.isfinite(c.wind_kt):
                continue
            xs.append(ships_predictors(c, a, lead))
            ys.append(c.truth["wind"][k] - c.wind_kt)
        x, y = np.asarray(xs), np.asarray(ys)
        if len(y) < 10 * x.shape[1]:
            table["weights"].append([float("nan")] * len(SHIPS_PREDICTORS))
            continue
        gram = x.T @ x + ridge * len(y) * np.diag([0.0] + [1.0] * (x.shape[1] - 1))
        table["weights"].append(np.linalg.solve(gram, x.T @ y).round(5).tolist())
    return table


def ships_intensity(case, along: dict, leads, table: dict | None = None) -> np.ndarray:
    if table is None:
        from .coefficients import SHIPS as table
    stored = list(table["leads"])
    out = np.full(len(leads), np.nan)
    for k, lead in enumerate(leads):
        if lead in stored and along is not None:
            w = np.asarray(table["weights"][stored.index(lead)], np.float64)
            out[k] = max(case.wind_kt + float(w @ ships_predictors(case, along, lead)), 15.0)
    return out


#: LGEM: the exponent of the logistic ceiling, the damping rate (per hour) and the growth-rate coefficients
#: (per hour) on [1, shear/10, rh, (sst_eff - 28.5 C)/2, (pi - v)/50].
LGEM_N = 2.5
LGEM_BETA = 1.0 / 24.0
LGEM_KAPPA = (0.010, -0.012, 0.035, 0.006, 0.015)


def lgem_table() -> tuple[np.ndarray, float, float]:
    """The shipped LGEM fit (``coefficients.LGEM``) if there is one, else the hand-set defaults above."""
    from . import coefficients

    table = getattr(coefficients, "LGEM", None)
    if table is None:
        return np.asarray(LGEM_KAPPA, np.float64), LGEM_N, LGEM_BETA
    return np.asarray(table["kappa"], np.float64), float(table["n"]), float(table["beta_per_h"])


def lgem_intensity(case, along: dict, leads, kappa=None, n: float | None = None, beta: float | None = None) -> np.ndarray:
    """Logistic growth to potential intensity, integrated hourly along the track; land decay applied after."""
    fitted, fitted_n, fitted_beta = lgem_table()
    kappa = fitted if kappa is None else np.asarray(kappa, np.float64)
    n = fitted_n if n is None else n
    beta = fitted_beta if beta is None else beta
    hours = along["hours"]
    six = along["six_hours"]
    pi = np.maximum(np.interp(hours, six, along["pi_kt"]), 20.0)
    shear = np.interp(hours, six, along["shear_ms"]) / 10.0
    rh = np.interp(hours, six, along["rh_mid"])
    sst = (along["sst_eff_k"] - 301.65) / 2.0
    v = np.empty_like(hours)
    v[0] = max(case.wind_kt, 15.0)
    for k in range(1, len(hours)):
        dt = hours[k] - hours[k - 1]
        rate = kappa @ np.array([1.0, shear[k], rh[k], sst[k], (pi[k] - v[k - 1]) / 50.0])
        growth = rate * v[k - 1] - beta * (v[k - 1] / pi[k]) ** n * v[k - 1]
        v[k] = float(np.clip(v[k - 1] + dt * growth, 15.0, 200.0))
    v = apply_land_decay(hours, v, along["land"], along["height_m"])
    return np.interp(np.asarray(leads, float), hours, v, right=np.nan)


def lgem_batch(start_kt: np.ndarray, alongs: list, leads, kappa=None, n: float = LGEM_N, beta: float = LGEM_BETA,
               land: bool = True) -> np.ndarray:
    """
    :func:`lgem_intensity` for many storms at once (``(N, L)``): the environment arrays are stacked over cases on
    the common hourly axis, so fitting the growth-rate coefficients over thousands of hindcasts takes seconds.
    """
    kappa = np.asarray(LGEM_KAPPA if kappa is None else kappa, np.float64)
    hours = alongs[0]["hours"]
    t = len(hours)
    six = alongs[0]["six_hours"]

    def stack(key, six_hourly):
        rows = []
        for a in alongs:
            x = np.asarray(a[key], np.float64)
            x = np.interp(hours, six, x) if six_hourly else x
            rows.append(x[:t] if x.size >= t else np.pad(x, (0, t - x.size), mode="edge"))
        return np.stack(rows)

    pi = np.maximum(stack("pi_kt", True), 20.0)
    shear = stack("shear_ms", True) / 10.0
    rh = stack("rh_mid", True)
    sst = (stack("sst_eff_k", False) - 301.65) / 2.0
    v = np.empty((len(alongs), t))
    v[:, 0] = np.maximum(np.asarray(start_kt, np.float64), 15.0)
    for k in range(1, t):
        dt = hours[k] - hours[k - 1]
        rate = kappa[0] + kappa[1] * shear[:, k] + kappa[2] * rh[:, k] + kappa[3] * sst[:, k] + kappa[4] * (pi[:, k] - v[:, k - 1]) / 50.0
        growth = rate * v[:, k - 1] - beta * (v[:, k - 1] / pi[:, k]) ** n * v[:, k - 1]
        v[:, k] = np.clip(v[:, k - 1] + dt * growth, 15.0, 200.0)
    if land:
        lands, heights = stack("land", False), stack("height_m", False)
        v = np.stack([apply_land_decay(hours, v[i], lands[i], heights[i]) for i in range(len(alongs))])
    idx = [int(np.argmin(np.abs(hours - h))) for h in leads]
    return v[:, idx]


def fit_lgem(cases, alongs: list, leads, start=None, iterations: int = 300) -> np.ndarray:
    """Growth-rate coefficients minimising the mean absolute wind error over all leads (Nelder-Mead)."""
    keep = [i for i, a in enumerate(alongs) if a is not None and np.isfinite(cases[i].wind_kt)]
    al = [alongs[i] for i in keep]
    start_kt = np.array([cases[i].wind_kt for i in keep])
    truth = np.stack([cases[i].truth["wind"] for i in keep])
    valid = np.stack([cases[i].truth["valid"] for i in keep]) & np.isfinite(truth)

    def loss(k):
        pred = lgem_batch(start_kt, al, leads, k, land=False)
        return float(np.mean(np.abs(pred - truth)[valid]))

    x0 = np.asarray(LGEM_KAPPA if start is None else start, np.float64)
    simplex = [x0] + [x0 + np.eye(len(x0))[j] * (0.5 * abs(x0[j]) + 0.005) for j in range(len(x0))]
    values = [loss(x) for x in simplex]
    for _ in range(iterations):
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        centroid = np.mean(simplex[:-1], axis=0)
        reflected = centroid + (centroid - simplex[-1])
        fr = loss(reflected)
        if fr < values[0]:
            expanded = centroid + 2.0 * (centroid - simplex[-1])
            fe = loss(expanded)
            simplex[-1], values[-1] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < values[-2]:
            simplex[-1], values[-1] = reflected, fr
        else:
            contracted = centroid + 0.5 * (simplex[-1] - centroid)
            fc = loss(contracted)
            if fc < values[-1]:
                simplex[-1], values[-1] = contracted, fc
            else:
                simplex = [simplex[0] + 0.5 * (x - simplex[0]) for x in simplex]
                values = [loss(x) for x in simplex]
    return simplex[int(np.argmin(values))]


def decay_along(case, along: dict, leads, water_kt: np.ndarray) -> np.ndarray:
    """Apply inland and terrain decay to an over-water forecast given at ``leads``."""
    hours = along["hours"]
    water = np.interp(hours, np.concatenate([[0.0], np.asarray(leads, float)]), np.concatenate([[case.wind_kt], water_kt]))
    decayed = apply_land_decay(hours, water, along["land"], along["height_m"])
    return np.interp(np.asarray(leads, float), hours, decayed, right=np.nan)


__all__ = ["LGEM_KAPPA", "SHIFOR_PREDICTORS", "SHIPS_PREDICTORS", "decay_along", "environment_along_track",
           "fit_lgem", "fit_shifor", "fit_ships", "lgem_batch", "lgem_intensity", "lgem_table", "potential_intensity_at", "shifor_intensity",
           "shifor_predictors", "ships_intensity", "ships_predictors"]
