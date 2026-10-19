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


def environment_along_track(env, env_clean, case, track: np.ndarray, leads, wind_guess_kt: np.ndarray | None = None,
                            ocean=None) -> dict:
    """
    Hourly environment along the forecast track: effective SST (cold wake), potential intensity, shear (850-200
    and 850-500, annulus 200-800 km), 700-500 hPa humidity, land and terrain, translation speed -- and, with an
    :class:`naturev1.cyclone.ocean.OceanClimatology` (``ocean``), the depth of the 26 C isotherm and the heat potential
    along the track, with the cold wake sized by that depth.
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
    d26 = tchp = np.full(hours.size, np.nan)
    power = None
    if ocean is not None:
        from . import coefficients

        from .ocean import anomaly_adjusted_depth

        sampled = ocean.sample(la, lo, case.time)
        d26, tchp = sampled["d26_m"], sampled["tchp_kj_cm2"]
        table = getattr(coefficients, "OCEAN", None) or {}
        power = table.get("d26_power")
        sst_clim = sampled.get("sst_c", np.full(hours.size, np.nan))
        d26_used = anomaly_adjusted_depth(d26, sst, sst_clim, table.get("sst_anomaly_per_c", 0.0))
    sst_eff = effective_sst(sst, wind, speed, hours, d26_used if ocean is not None else None, power)
    sample_hours = np.arange(0.0, hours[-1] + 1e-9, 6.0)
    pick = np.searchsorted(hours, sample_hours).clip(0, len(hours) - 1)
    pi = potential_intensity_at(env, la[pick], lo[pick], sst_eff[pick]) * KT_PER_MS
    # the potential intensity over the uncooled sea as well, so a different wake can be priced without the profiles
    pi_warm = potential_intensity_at(env, la[pick], lo[pick], sst[pick]) * KT_PER_MS if ocean is not None else pi
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
            "shear_mid_ms": np.asarray(shear_m), "rh_mid": np.asarray(rh), "landfall_hour": land["landfall_hour"],
            "d26_m": d26, "tchp_kj_cm2": tchp, "pi_warm_kt": pi_warm,
            "sst_clim_c": sampled.get("sst_c", np.full(hours.size, np.nan)) if ocean is not None else np.full(hours.size, np.nan)}


SHIPS_PREDICTORS = ("one", "wind", "dwind12", "potential", "potential2", "shear", "shear_mid", "rh", "sst", "land",
                    "wind_shear", "lat", "speed", "heat_potential", "size", "has_size")


def _size_features(case) -> list:
    """The storm's observed 34-kt radius (mean quadrant, in units of 150 km) and whether it was reported: small storms
    intensify faster, and the radius is in the best track for 2004 on (``Case.r34_nmi``, from IBTrACS or the b-deck)."""
    r = getattr(case, "r34_nmi", float("nan"))
    return [float(r) * 1.852 / 150.0 if np.isfinite(r) and r > 0 else 0.0, 1.0 if np.isfinite(r) and r > 0 else 0.0]


def _heat_potential(along: dict, mask) -> float:
    """Mean tropical-cyclone heat potential along the track (kJ/cm2) scaled to ~1 at 50, NaN -> 0 (unknown)."""
    tchp = np.asarray(along.get("tchp_kj_cm2", np.full(np.count_nonzero(mask), np.nan)), np.float64)
    if tchp.size != np.size(mask):
        return 0.0
    v = tchp[mask]
    return float(np.nanmean(v)) / 50.0 if np.isfinite(v).any() else 0.0


def ships_predictors(case, along: dict, lead: float) -> np.ndarray:
    """Predictors averaged along the track from now to ``lead`` (the heat potential last: a table fitted without the
    ocean climatology simply ignores it)."""
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
                     wind / 100.0 * shear, abs(case.lat) / 30.0, float(np.mean(along["speed_ms"][hourly])) / 5.0,
                     _heat_potential(along, hourly), *_size_features(case)])


def _fit_length(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Features cut or zero-padded to a table's length: newer predictors are always appended last."""
    if features.size == weights.size:
        return features
    if features.size > weights.size:
        return features[:weights.size]
    return np.concatenate([features, np.zeros(weights.size - features.size)])


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
            out[k] = max(case.wind_kt + float(w @ _fit_length(ships_predictors(case, along, lead), w)), 15.0)
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


#: The trend member's memory (hours): how long a storm keeps intensifying (or weakening) the way it has been.
TREND_TAU_H = 24.0


def recent_change(case, hours: int = 24) -> float:
    """Intensity change (kt) over the last ``hours`` from the best-track history (NaN when unknown)."""
    from .data import HISTORY_HOURS

    if -hours not in HISTORY_HOURS:
        return float("nan")
    past = case.history["wind"][HISTORY_HOURS.index(-hours)]
    return float(case.wind_kt - past) if np.isfinite(past) else float("nan")


def trend_intensity(case, along: dict, leads, tau_h: float | None = None) -> np.ndarray:
    """
    The storm keeps doing what it has been doing: the last 24 hours' change continued with an e-folding memory of
    ``tau_h`` (``V0 + r tau (1 - exp(-t / tau))``), never above the potential intensity along the track, then land and
    terrain decay. This is the member that can follow a storm through rapid intensification, which the regressions
    (built to minimise the average error) pull back toward the mean.
    """
    from . import coefficients

    tau = tau_h if tau_h is not None else getattr(coefficients, "TREND", {}).get("tau_h", TREND_TAU_H)
    change = recent_change(case, 24)
    if not np.isfinite(change):
        change = 2.0 * recent_change(case, 12) if np.isfinite(recent_change(case, 12)) else 0.0
    rate = change / 24.0
    hours = along["hours"]
    path = case.wind_kt + rate * tau * (1.0 - np.exp(-hours / tau))
    pi = np.interp(hours, along["six_hours"], along["pi_kt"])
    path = np.where(rate > 0, np.minimum(path, np.maximum(pi, case.wind_kt)), path)
    path = apply_land_decay(hours, np.maximum(path, 15.0), along["land"], along["height_m"])
    return np.interp(np.asarray(leads, float), hours, path, right=np.nan)


#: Rapid intensification (+30 kt in 24 h): predictor names, in order.
RI_FEATURES = ("one", "dv12", "dv24", "wind", "headroom", "shear", "rh", "sst", "lat", "land24")


#: With a dynamical model's forecast: its central-pressure deficit change by 24 and 48 h, and whether it was there.
RI_MODEL_FEATURES = ("model_dp24", "model_dp48", "has_model")
#: With the ocean climatology: the heat potential over the first 24 hours (kJ/cm2 / 50) and the 26 C depth (m / 50).
RI_OCEAN_FEATURES = ("heat_potential", "warm_depth")
#: The storm's observed 34-kt radius (units of 150 km) and whether it was reported.
RI_SIZE_FEATURES = ("size", "has_size")


def ri_features(case, along: dict, model_dp=None, leads=None) -> np.ndarray:
    """
    SHIPS-RII-style predictors over the first 24 hours along the forecast track; with ``model_dp`` (the models' mean
    change in central-pressure deficit at the standard leads, hPa, negative = deepening, from
    :func:`model_deficit_change`) three more: the deepening by 24 h, by 48 h, and that a model was there.
    """
    h = along["hours"]
    six = along["six_hours"]
    first = six <= 24.0 + 1e-9
    pi = float(np.nanmean(np.asarray(along["pi_kt"])[first]))
    dv12, dv24 = recent_change(case, 12), recent_change(case, 24)
    base = np.array([1.0, np.nan_to_num(dv12) / 10.0, np.nan_to_num(dv24) / 10.0, case.wind_kt / 50.0,
                     (pi - case.wind_kt) / 50.0, float(np.nanmean(np.asarray(along["shear_ms"])[first])) / 10.0,
                     float(np.nanmean(np.asarray(along["rh_mid"])[first])),
                     (float(np.nanmean(np.asarray(along["sst_eff_k"])[h <= 24.0])) - 301.65) / 2.0,
                     abs(case.lat) / 20.0, float(np.mean(np.asarray(along["land"])[h <= 24.0]))])
    from .data import LEADS

    if model_dp is None:
        model = np.zeros(3)
    else:
        dp = np.asarray(model_dp, np.float64)
        L = np.asarray(leads if leads is not None else LEADS, np.float64)
        k24, k48 = int(np.argmin(np.abs(L - 24.0))), int(np.argmin(np.abs(L - 48.0)))
        have = np.isfinite(dp[k24])
        model = np.array([np.nan_to_num(-dp[k24]) / 10.0 if have else 0.0, np.nan_to_num(-dp[k48]) / 20.0 if have else 0.0, float(have)])
    first_day = h <= 24.0
    tchp = np.asarray(along.get("tchp_kj_cm2", np.full(h.size, np.nan)), np.float64)
    d26 = np.asarray(along.get("d26_m", np.full(h.size, np.nan)), np.float64)
    ocean = np.array([float(np.nanmean(tchp[first_day])) / 50.0 if np.isfinite(tchp[first_day]).any() else 0.0,
                      float(np.nanmean(d26[first_day])) / 50.0 if np.isfinite(d26[first_day]).any() else 0.0])
    return np.concatenate([base, model, ocean, _size_features(case)])


def model_deficit_change(extras: dict | list | None, leads) -> np.ndarray | None:
    """
    ``(L,)`` the models' mean change in central-pressure deficit (central minus environmental pressure, hPa) from
    the analysis to each lead -- negative when the model deepens its storm -- from the ``extras`` of
    :func:`naturev1.swarm.models.fetch_models` (each with ``hours``, ``msl_min_hpa``, ``msl_env_hpa``). None without any.
    """
    if not extras:
        return None
    rows = []
    for e in (extras.values() if isinstance(extras, dict) else extras):
        e = e.get("extras", e)
        hours, pmin, penv = np.asarray(e["hours"], float), np.asarray(e["msl_min_hpa"], float), np.asarray(e["msl_env_hpa"], float)
        deficit = pmin - penv
        if hours.size == 0 or not np.isfinite(deficit[0]):
            continue
        at = np.array([deficit[np.argmin(np.abs(hours - h))] if np.any(np.abs(hours - h) < 1e-6) else np.nan for h in leads])
        rows.append(at - deficit[0])
    if not rows:
        return None
    return np.nanmean(np.stack(rows), axis=0) if len(rows) > 1 else rows[0]


def model_trend_intensity(case, along: dict, leads, model_dp, table: dict | None = None) -> np.ndarray:
    """
    The dynamical models' own intensification, as an intensity member: ``V0 + k_lead * deepening`` (kt per hPa of
    central-pressure-deficit change fitted per lead on the tuning seasons, ``coefficients.MODEL_INTENSITY``), capped
    at the potential intensity, then land and terrain decay. A model that deepens its storm by 30 hPa is saying
    something the statistical members cannot.
    """
    if table is None:
        from . import coefficients

        table = getattr(coefficients, "MODEL_INTENSITY", None) or {"leads": list(leads), "kt_per_hpa": [1.0] * len(leads)}
    hours = along["hours"]
    k = np.interp(np.asarray(leads, float), table["leads"], table["kt_per_hpa"])
    dp = np.asarray(model_dp, np.float64)
    water = case.wind_kt + k * (-np.nan_to_num(dp))
    pi = np.interp(np.asarray(leads, float), along["six_hours"], along["pi_kt"])
    water = np.where(np.isfinite(dp), np.minimum(water, np.maximum(pi, case.wind_kt)), np.nan)
    knots = np.concatenate([[0.0], np.asarray(leads, float)])
    path = np.interp(hours, knots, np.concatenate([[case.wind_kt], np.nan_to_num(water, nan=case.wind_kt)]))
    path = apply_land_decay(hours, np.maximum(path, 15.0), along["land"], along["height_m"])
    return np.where(np.isfinite(water), np.interp(np.asarray(leads, float), hours, path), np.nan)


def fit_ri_path(cases, alongs: list, leads, threshold_kt: float = 30.0) -> dict:
    """
    The intensity path of the storms that *did* intensify rapidly: the mean and 75th percentile of their change from
    the analysis at each lead (kt), over the tuning seasons' cases whose 24-hour change reached ``threshold_kt``.
    """
    from .data import LEADS

    k24 = list(LEADS).index(24)
    rows = []
    for c, a in zip(cases, alongs):
        if a is None or not c.truth["valid"][k24] or not np.isfinite(c.truth["wind"][k24]) or not np.isfinite(c.wind_kt):
            continue
        if c.truth["wind"][k24] - c.wind_kt >= threshold_kt:
            rows.append(np.where(c.truth["valid"], c.truth["wind"], np.nan) - c.wind_kt)
    rows = np.asarray(rows)
    return {"leads": [int(h) for h in leads], "threshold_kt": threshold_kt, "cases": int(rows.shape[0]),
            "mean_change_kt": [round(float(np.nanmean(rows[:, k])), 2) for k in range(len(leads))],
            "p75_change_kt": [round(float(np.nanquantile(rows[:, k], 0.75)), 2) for k in range(len(leads))]}


def ri_scenario_intensity(case, along: dict, leads, table: dict | None = None, which: str = "mean_change_kt") -> np.ndarray:
    """
    The intensity *if the storm intensifies rapidly*: the analysis plus the rapid intensifiers' mean change at each
    lead (``coefficients.RI_PATH``), capped at the potential intensity along the track, then land and terrain decay.
    Issued beside the consensus with its probability (``ri_probability``): the consensus is the expected value, this
    is the branch a parish cannot afford to ignore when the probability is real.
    """
    if table is None:
        from . import coefficients

        table = getattr(coefficients, "RI_PATH", None)
        if table is None:
            return np.full(len(leads), np.nan)
    hours = along["hours"]
    change = np.interp(np.asarray(leads, float), table["leads"], table[which])
    water = case.wind_kt + change
    pi = np.interp(np.asarray(leads, float), along["six_hours"], along["pi_kt"])
    water = np.minimum(water, np.maximum(pi, case.wind_kt))
    knots = np.concatenate([[0.0], np.asarray(leads, float)])
    path = np.interp(hours, knots, np.concatenate([[case.wind_kt], water]))
    path = apply_land_decay(hours, np.maximum(path, 15.0), along["land"], along["height_m"])
    return np.interp(np.asarray(leads, float), hours, path)


def category_probabilities(intensity: np.ndarray, regime: str, table: dict | None = None, ri_probability: float | None = None,
                           ri_intensity: np.ndarray | None = None) -> dict:
    """
    ``{"thresholds_kt": [...], "probability": (L, T)}`` -- P(intensity >= each threshold) at each lead, from the
    forecast plus the empirical distribution of the consensus's out-of-sample errors on the tuning seasons
    (``coefficients.INTENSITY_SPREAD``: 25 quantiles per lead, separately for the normal and the rapid regime).
    """
    if table is None:
        from . import coefficients

        table = getattr(coefficients, "INTENSITY_SPREAD", None)
    if table is None:
        return {}
    q = np.asarray(table["quantiles"].get(regime, table["quantiles"]["normal"]), np.float64)        # (L, 25)
    thresholds = np.asarray(table.get("thresholds_kt", (64, 96, 113, 137)), np.float64)
    v = np.asarray(intensity, np.float64)
    samples = v[:, None] + q                                                                        # (L, 25)
    prob = np.stack([(samples >= t).mean(1) for t in thresholds], -1)
    mixed = table.get("ri_mixture") and ri_probability is not None and ri_intensity is not None and np.isfinite(ri_intensity).any()
    if mixed:
        # the rapid-intensification branch: with probability P(RI) the storm follows the rapid intensifiers' path,
        # spread by the rapid regime's own error quantiles (switched on by the table only where it scored better)
        qr = np.asarray(table["quantiles"]["rapid"], np.float64)
        branch = np.asarray(ri_intensity, np.float64)[:, None] + qr
        prob_ri = np.stack([(branch >= t).mean(1) for t in thresholds], -1)
        p = float(np.clip(ri_probability, 0.0, 1.0))
        prob = (1.0 - p) * prob + p * np.where(np.isfinite(prob_ri), prob_ri, prob)
    prob[~np.isfinite(v)] = np.nan
    return {"thresholds_kt": thresholds.tolist(), "probability": prob, "ri_mixture": bool(mixed)}


def fit_ri(cases, alongs: list, threshold_kt: float = 30.0, l2: float = 1e-2, iterations: int = 30,
           model_dp: dict | None = None, n_features: int | None = None) -> dict:
    """Logistic regression for P(intensity rises by >= ``threshold_kt`` in 24 h), by Newton's method, on
    :func:`ri_features` (``model_dp``: ``{case.key: (L,) deficit change}`` when the models' deepening is a predictor;
    ``n_features`` cuts the predictor list, e.g. 10 for the environment alone)."""
    from .data import LEADS

    k24 = list(LEADS).index(24)
    x, y = [], []
    for c, a in zip(cases, alongs):
        if a is None or not c.truth["valid"][k24] or not np.isfinite(c.truth["wind"][k24]):
            continue
        f = ri_features(c, a, None if model_dp is None else model_dp.get(c.key))
        if n_features is not None:
            f = _fit_length(f, np.zeros(n_features))
        if np.isfinite(f).all():
            x.append(f)
            y.append(float(c.truth["wind"][k24] - c.wind_kt >= threshold_kt))
    x, y = np.asarray(x), np.asarray(y)
    w = np.zeros(x.shape[1])
    reg = l2 * len(y) * np.diag([0.0] + [1.0] * (x.shape[1] - 1))
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-x @ w))
        grad = x.T @ (p - y) + reg @ w
        hess = (x * (p * (1 - p))[:, None]).T @ x + reg
        w = w - np.linalg.solve(hess, grad)
    names = list(RI_FEATURES) + list(RI_MODEL_FEATURES) + list(RI_OCEAN_FEATURES) + list(RI_SIZE_FEATURES)
    return {"features": names[:x.shape[1]], "threshold_kt": threshold_kt, "weights": w.round(5).tolist(),
            "base_rate": float(y.mean()), "cases": int(len(y))}


def ri_probability(case, along: dict, table: dict | None = None, model_dp=None, leads=None) -> float:
    """P(+30 kt in 24 h) for a storm on its forecast track (``coefficients.RI`` by default; a table fitted with the
    model-deepening predictors uses them when ``model_dp`` is given, and counts the model as absent otherwise)."""
    if table is None:
        from .coefficients import RI as table
    w = np.asarray(table["weights"], np.float64)
    f = _fit_length(ri_features(case, along, model_dp, leads), w)    # a table fitted with fewer predictors ignores the rest
    if not np.isfinite(f).all():
        return float(table.get("base_rate", 0.05))
    return float(1.0 / (1.0 + np.exp(-w @ f)))



#: Predictors of the peak-intensity logistic (``coefficients.PEAK``), after the intercept: the analysis intensity, the
#: consensus's rise and maximum, the warm-ocean potential-intensity headroom over 72 h, mean ocean heat content, shear
#: and mid-level humidity over 48 h, P(RI) and its logit, the model's 24 h deepening and whether there is one, the
#: share of the first 72 h over land, latitude.
PEAK_FEATURES = ("one", "v0", "cons_rise", "cons_max", "headroom72", "tchp48", "shear48", "rh48", "p_ri", "logit_p", "dp24", "has_dp",
                 "land72", "abslat")


def _series_upto(along: dict, key: str, upto_h: float) -> np.ndarray:
    v = np.asarray(along[key], np.float64)
    t = np.asarray(along["six_hours"] if v.size == len(along["six_hours"]) else along["hours"], np.float64)
    return v[t <= upto_h]


def peak_features(case, along: dict, consensus_kt, ri_probability: float, model_dp=None, leads=None) -> np.ndarray:
    """The :data:`PEAK_FEATURES` row for one case (raw, before the table's standardisation)."""
    from .data import LEADS

    cv = np.asarray(consensus_kt, np.float64)
    p = float(np.clip(ri_probability if ri_probability is not None and np.isfinite(ri_probability) else 0.05, 1e-3, 1 - 1e-3))
    pi = _series_upto(along, "pi_warm_kt" if "pi_warm_kt" in along else "pi_kt", 72.0)
    L = np.asarray(leads if leads is not None else LEADS, np.float64)
    k24 = int(np.argmin(np.abs(L - 24.0)))
    dp = np.asarray(model_dp, np.float64) if model_dp is not None else np.full(L.size, np.nan)
    has_dp = bool(dp.size > k24 and np.isfinite(dp[k24]))
    tchp = _series_upto(along, "tchp_kj_cm2", 48.0) if "tchp_kj_cm2" in along else np.array([np.nan])
    with np.errstate(all="ignore"):
        row = [1.0, case.wind_kt, np.nanmax(cv) - case.wind_kt, np.nanmax(cv), np.nanmax(pi) - case.wind_kt, np.nanmean(tchp),
               np.nanmean(_series_upto(along, "shear_ms", 48.0)), np.nanmean(_series_upto(along, "rh_mid", 48.0)), p, np.log(p / (1 - p)),
               dp[k24] if has_dp else 0.0, float(has_dp), np.nanmean(_series_upto(along, "land", 72.0)), abs(case.lat)]
    return np.nan_to_num(np.asarray(row, np.float64))


def peak_probability(case, along: dict, consensus_kt, ri_probability: float, model_dp=None, table: dict | None = None, leads=None) -> dict:
    """
    ``{"thresholds_kt": [64, 96, 113, 137], "probability": [...]}`` -- the probability that the storm's **peak**
    intensity within the forecast window reaches each threshold (hurricane, major, category 4, category 5): a logistic
    on :data:`PEAK_FEATURES` fitted on the tuning seasons (``coefficients.PEAK``). This is the question a point
    forecast cannot answer: the consensus is the expected intensity, and for the storms that do explode it sits
    30 kt under their peak; this says how likely the explosion is. 1 where the storm is already at the threshold.
    """
    if table is None:
        from . import coefficients

        table = getattr(coefficients, "PEAK", None)
    if not table:
        return {}
    x = peak_features(case, along, consensus_kt, ri_probability, model_dp, leads)
    mu, sd = np.asarray(table["mean"], np.float64), np.asarray(table["std"], np.float64)
    z = np.concatenate([x[:1], (x[1:] - mu) / np.where(sd > 1e-9, sd, 1.0)])
    out = []
    for thr in table["thresholds_kt"]:
        w = np.asarray(table["weights"][str(int(thr))], np.float64)
        out.append(1.0 if case.wind_kt >= thr else float(1.0 / (1.0 + np.exp(-float(z @ w)))))
    return {"thresholds_kt": [float(t) for t in table["thresholds_kt"]], "probability": out}


def _peak_row(case, along: dict, consensus_kt, ri_probability, model_dp, intensity_if_ri, table: dict, leads=None) -> np.ndarray:
    """The standardised PEAK features plus the rapid-intensification path's rise, as the peak-quantile tables were fitted."""
    x = peak_features(case, along, consensus_kt, ri_probability, model_dp, leads)
    mu, sd = np.asarray(table["mean"], np.float64), np.asarray(table["std"], np.float64)
    n = x.size - 1
    z = np.concatenate([x[:1], (x[1:] - mu[:n]) / np.where(sd[:n] > 1e-9, sd[:n], 1.0)])
    scen = np.asarray(intensity_if_ri, np.float64) if intensity_if_ri is not None else np.array([np.nan])
    rise = (np.nanmax(scen) - case.wind_kt) if np.isfinite(scen).any() else 0.0
    return np.append(z, (rise - mu[n]) / (sd[n] if sd[n] > 1e-9 else 1.0))


def peak_quantiles(case, along: dict, consensus_kt, ri_probability, model_dp=None, intensity_if_ri=None, table: dict | None = None,
                   leads=None) -> dict:
    """
    ``{"levels": [0.5, 0.75, 0.9, 0.95], "peak_kt": [...]}`` -- calibrated quantiles of the storm's **peak** intensity
    within the forecast window (``coefficients.PEAK_QUANTILES``: linear pinball-loss fits on the tuning seasons). The
    point forecast cannot reach an explosive storm's top end and should not try; the 90% quantile can: on the held-out
    storms that reached 137 kt it sits 4 kt above their true peak where the consensus maximum sits 37 kt below, with
    91% of all held-out peaks under it. Quantiles are made monotone at prediction time.
    """
    if table is None:
        from . import coefficients

        table = getattr(coefficients, "PEAK_QUANTILES", None)
    if not table:
        return {}
    z = _peak_row(case, along, consensus_kt, ri_probability, model_dp, intensity_if_ri, table, leads)
    levels = sorted(float(q) for q in table["quantiles"])
    rise = np.maximum.accumulate(np.array([float(z @ np.asarray(table["quantiles"][str(q)], np.float64)) for q in levels]))
    return {"levels": levels, "peak_kt": [float(np.clip(case.wind_kt + r, 15.0, 200.0)) for r in rise]}


def peak_if_ri(case, along: dict, consensus_kt, ri_probability, model_dp=None, intensity_if_ri=None, table: dict | None = None,
               leads=None) -> float:
    """
    E[peak within the window | the storm intensifies rapidly], kt (``coefficients.PEAK_IF_RI``, a ridge fit on the
    tuning seasons' rapid intensifiers): unbiased on the held-out storms that did go rapid (+0.6 kt, MAE 13) where the
    rapid-intensification path's peak sat 17 kt low. To be read with ``ri_probability``, never as the forecast: on the
    storms that do not go rapid it runs 35 kt high.
    """
    if table is None:
        from . import coefficients

        table = getattr(coefficients, "PEAK_IF_RI", None)
    if not table:
        return float("nan")
    z = _peak_row(case, along, consensus_kt, ri_probability, model_dp, intensity_if_ri, table, leads)
    return float(np.clip(case.wind_kt + float(z @ np.asarray(table["weights"], np.float64)), 15.0, 200.0))

__all__ = ["LGEM_KAPPA", "PEAK_FEATURES", "RI_FEATURES", "RI_MODEL_FEATURES", "RI_OCEAN_FEATURES", "RI_SIZE_FEATURES", "SHIFOR_PREDICTORS",
           "SHIPS_PREDICTORS",
           "TREND_TAU_H",
           "category_probabilities", "decay_along", "environment_along_track", "fit_ri", "fit_ri_path", "model_deficit_change",
           "model_trend_intensity", "peak_features", "peak_if_ri", "peak_probability", "peak_quantiles", "recent_change", "ri_features", "ri_probability", "ri_scenario_intensity", "trend_intensity",
           "fit_lgem", "fit_shifor", "fit_ships", "lgem_batch", "lgem_intensity", "lgem_table", "potential_intensity_at", "shifor_intensity",
           "shifor_predictors", "ships_intensity", "ships_predictors"]
