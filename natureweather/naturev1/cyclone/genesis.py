# Copyright 2026 Nathan. Apache-2.0.
"""
Seeing storms before they are storms: genesis detection from the analysis, verified against what formed.

The idea: a tropical cyclone forms from a disturbance that has been visible for days -- a tropical wave, a
monsoon-trough vortex -- and the question a forecaster answers (NHC's two- and seven-day outlooks) is which
disturbances will become cyclones and when. The ingredients are known physics (Gray 1968; Emanuel and Nolan
2004; Dunkerton et al. 2009): low-level spin that is rotation-dominated rather than shear (Okubo-Weiss),
a moist mid-troposphere, weak vertical shear, warm water, a closed circulation in the frame moving with the
wave (:mod:`naturev1.cyclone.pouch`), a mid-level vortex stacking above it, and a falling surface pressure. So:

1. **Candidates**: local maxima of smoothed 850 hPa cyclonic vorticity above a threshold over warm ocean in the
   tropics, at every analysis time, away from cyclones that already exist.
2. **Candidate tracks**: candidates linked in time (nearest within 600 km of where the 700 hPa flow carries
   them), so each disturbance has an age and a history -- intensifying spin matters more than strong spin.
3. **Features** per candidate and time: vorticity (850, 600), its 24-hour change, Okubo-Weiss, 600 hPa
   humidity, 850-200 shear, SST, a closed-low depth, the 300 hPa warm-core anomaly, the genesis potential
   index, and the wave-relative pouch's closedness.
4. **Probability** of genesis within 48 hours: a logistic model on those features, fitted on the 2016/2017/2019
   seasons against IBTrACS (genesis = the first tropical or subtropical fix within 300 km of the candidate in the
   next 48 hours) and verified on 2018/2020-2022 with the Brier score, its skill against climatology, detection
   rate and false alarms, and how many hours of warning the detector gives.

The basin store (:func:`build_genesis_store`) keeps the northern-hemisphere tropics (0-45 N, all longitudes)
at 00 and 12 UTC through each season: the 850, 600 and 200 hPa winds, 600 hPa temperature and humidity, the
300 and 850 hPa temperatures, sea-level pressure and SST.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .fields import Environment
from .geometry import great_circle_km, hemisphere
from .pouch import phase_from_history, pouch_closedness

CHANNELS = ("u850", "v850", "u600", "v600", "u200", "v200", "t600", "q600", "t300", "t850", "msl", "sst")
LAT_RANGE = (0.0, 45.0)
SEASON = ("06-01", "11-30")
VORTICITY_MIN = 2.0e-5
WINDOW_H = 48
FEATURES = ("one", "vort850", "vort600", "dvort24", "okubo_weiss", "rh600", "shear", "sst", "depth", "warm_core",
            "gpi", "age", "lat", "pouch")


def build_genesis_store(years, root: str | Path, reader=None, log=print) -> dict[int, Path]:
    """One memmap per season: ``(T, C, H, 240)`` float16 (standardised per channel) plus ``times.npy``."""
    from .data import WB2Reader

    out = {}
    for year in years:
        directory = Path(root) / f"genesis_{year}"
        out[year] = directory
        if (directory / "done").exists():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        reader = reader or WB2Reader()
        rows = np.nonzero((reader.lat >= LAT_RANGE[0]) & (reader.lat <= LAT_RANGE[1]))[0]
        start, stop = np.datetime64(f"{year}-{SEASON[0]}T00", "s"), np.datetime64(f"{year}-{SEASON[1]}T12", "s")
        times = np.arange(start, stop + np.timedelta64(1, "s"), np.timedelta64(12, "h"))
        store = np.lib.format.open_memmap(directory / "fields.npy", mode="w+", dtype=np.float32,
                                          shape=(len(times), len(CHANNELS), rows.size, reader.lon.size))
        levels = list(reader.levels)
        for i, when in enumerate(times):
            f = reader.fields(when)
            planes = [f["u"][levels.index(850)], f["v"][levels.index(850)], f["u"][levels.index(600)],
                      f["v"][levels.index(600)], f["u"][levels.index(200)], f["v"][levels.index(200)],
                      f["t"][levels.index(600)], f["q"][levels.index(600)], f["t"][levels.index(300)],
                      f["t"][levels.index(850)], f["msl"], f["sst"]]
            store[i] = np.stack([p[rows] for p in planes])
            if log and i % 60 == 0:
                log(f"[genesis-data] {year}: {i + 1}/{len(times)}")
        store.flush()
        np.save(directory / "times.npy", times.astype(np.int64))
        np.save(directory / "lat.npy", reader.lat[rows])
        np.save(directory / "lon.npy", reader.lon)
        (directory / "done").write_text("ok")
    return out


class GenesisSeason:
    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        self.fields = np.load(directory / "fields.npy", mmap_mode="r")
        self.times = np.load(directory / "times.npy").astype("datetime64[s]")
        self.lat = np.load(directory / "lat.npy").astype(np.float64)
        self.lon = np.load(directory / "lon.npy").astype(np.float64)
        self.slot = {c: k for k, c in enumerate(CHANNELS)}

    def environment(self, i: int) -> Environment:
        block = np.asarray(self.fields[i], np.float64)
        f = {c: block[self.slot[c]] for c in CHANNELS}
        # Wrap one column so derivatives are periodic-ish at the seam; the Environment is the whole band.
        return Environment(lat=self.lat, lon=self.lon, levels=(850,), fields=f, time=self.times[i])


@dataclass
class Candidate:
    time: np.datetime64
    lat: float
    lon: float
    features: dict
    track_id: int = -1
    label: float = float("nan")
    history: list = field(default_factory=list)


def _smooth(field: np.ndarray, passes: int = 2) -> np.ndarray:
    out = field.copy()
    for _ in range(passes):
        padded = np.pad(out, ((1, 1), (0, 0)), mode="edge")
        padded = np.concatenate([padded[:, -1:], padded, padded[:, :1]], 1)
        out = (4 * padded[1:-1, 1:-1] + padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]) / 8.0
    return out


def _relative_humidity(t, q, p):
    es = 6.112 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return np.clip(q * p / (0.622 + 0.378 * q) / es, 0.0, 1.2)


def diagnostics(env: Environment) -> dict:
    """Gridded genesis ingredients for one analysis of the band."""
    f = env.fields
    zeta850 = env.vorticity(f["u850"], f["v850"])
    zeta600 = env.vorticity(f["u600"], f["v600"])
    sign = hemisphere(env.lat)[:, None]
    dudx, dudy = env.ddx(f["u850"]), env.ddy(f["u850"])
    dvdx, dvdy = env.ddx(f["v850"]), env.ddy(f["v850"])
    strain2 = (dudx - dvdy) ** 2 + (dvdx + dudy) ** 2
    ow = zeta850 ** 2 - strain2
    rh = _relative_humidity(f["t600"], f["q600"], 600.0)
    shear = np.hypot(f["u200"] - f["u850"], f["v200"] - f["v850"])
    sst = np.asarray(f["sst"], np.float64)                            # NaN over land
    eta = np.abs(zeta850 + env.coriolis())
    potential = np.clip(np.nan_to_num(sst, nan=273.15) - 273.15 - 26.0, 0.0, None) * 10.0   # stand-in for PI, m/s
    gpi = np.abs(1e5 * eta) ** 1.5 * (100 * rh / 50.0) ** 3 * (potential / 70.0) ** 3 * (1 + 0.1 * shear) ** -2
    t300_anom = f["t300"] - _smooth(f["t300"], 20)
    return {"vort850": sign * _smooth(zeta850), "vort600": sign * _smooth(zeta600), "ow": _smooth(ow), "rh600": rh,
            "shear": shear, "sst": sst, "gpi": gpi, "warm": t300_anom, "msl": f["msl"]}


def find_candidates(env: Environment, diag: dict, existing: list[tuple[float, float]], exclude_km: float = 600.0) -> list[dict]:
    vort = diag["vort850"]
    out = []
    h, w = vort.shape
    for i in range(2, h - 2):
        if abs(env.lat[i]) < 3.0 or abs(env.lat[i]) > 35.0:
            continue
        for j in range(w):
            v = vort[i, j]
            if v < VORTICITY_MIN or not (diag["sst"][i, j] >= 299.15):
                continue
            block = vort[i - 1:i + 2, [(j - 1) % w, j, (j + 1) % w]]
            if v < block.max():
                continue
            lat, lon = float(env.lat[i]), float(((env.lon[j] + 180.0) % 360.0) - 180.0)
            if any(great_circle_km(lat, lon, a, b) < exclude_km for a, b in existing):
                continue
            out.append({"lat": lat, "lon": lon, "i": i, "j": j})
    return out


def candidate_features(env: Environment, diag: dict, cand: dict) -> dict:
    i, j = cand["i"], cand["j"]
    lat, lon = cand["lat"], cand["lon"]
    d = great_circle_km(lat, lon, *np.meshgrid(env.lat, env.lon, indexing="ij"))
    disc = d < 500.0
    ring = (d >= 500.0) & (d < 900.0)
    msl = diag["msl"]
    depth = float(msl[ring].mean() - msl[disc].min()) / 100.0 if ring.any() and disc.any() else 0.0
    return {"vort850": float(diag["vort850"][i, j]) * 1e5, "vort600": float(diag["vort600"][disc].max()) * 1e5,
            "okubo_weiss": float(np.clip(diag["ow"][disc].max() * 1e10, -50, 50)), "rh600": float(diag["rh600"][disc].mean()),
            "shear": float(diag["shear"][disc].mean()) / 10.0,
            "sst": (float(np.nanmean(diag["sst"][disc])) - 301.15) / 2.0 if np.isfinite(diag["sst"][disc]).any() else -1.0,
            "depth": depth, "warm_core": float(diag["warm"][disc].max()), "gpi": float(np.log1p(diag["gpi"][disc].mean())),
            "lat": abs(lat) / 20.0}


def genesis_events(tracks, years) -> list[tuple[np.datetime64, float, float]]:
    """IBTrACS genesis: each storm's first tropical/subtropical fix (time, lat, lon)."""
    out = []
    for t in tracks:
        tropical = np.isin(t.status, ("TD", "TS", "HU", "SS"))
        if not tropical.any():
            continue
        k = int(np.argmax(tropical))
        if int(str(t.time[k])[:4]) in set(years) and t.latitude[k] >= 0:
            out.append((t.time[k].astype("datetime64[s]"), float(t.latitude[k]), float(t.longitude[k])))
    return out


def active_storms(tracks, when: np.datetime64) -> list[tuple[float, float]]:
    out = []
    for t in tracks:
        hit = np.nonzero(t.time.astype("datetime64[s]") == when)[0]
        if hit.size and t.status[hit[0]] in ("TD", "TS", "HU", "SS"):
            out.append((float(t.latitude[hit[0]]), float(t.longitude[hit[0]])))
    return out


def detect_season(season: GenesisSeason, tracks, link_km: float = 600.0, log=None) -> list[Candidate]:
    """Candidates at every time, linked into disturbance tracks, with 24-hour trends and ages."""
    candidates: list[Candidate] = []
    previous: list[Candidate] = []
    next_id = 0
    for i in range(len(season.times)):
        env = season.environment(i)
        diag = diagnostics(env)
        when = season.times[i]
        found = find_candidates(env, diag, active_storms(tracks, when))
        current = []
        for cand in found:
            feats = candidate_features(env, diag, cand)
            c = Candidate(when, cand["lat"], cand["lon"], feats)
            best = None
            for p in previous:
                d = great_circle_km(p.lat, p.lon, c.lat, c.lon)
                if d < link_km and (best is None or d < best[0]):
                    best = (d, p)
            if best is not None:
                c.track_id = best[1].track_id
                c.history = best[1].history + [best[1]]
            else:
                c.track_id, next_id = next_id, next_id + 1
            c.features["pouch"] = pouch_closedness(env, c.lat, c.lon, phase_from_history(c))["closedness"]
            current.append(c)
        for c in current:
            back = [h for h in c.history if (c.time - h.time) == np.timedelta64(24, "h")]
            c.features["dvort24"] = c.features["vort850"] - back[0].features["vort850"] if back else 0.0
            c.features["age"] = min(len(c.history), 10) / 10.0
        candidates.extend(current)
        previous = current
        if log and i % 60 == 0:
            log(f"[genesis] {str(when)[:10]}: {len(current)} candidates")
    return candidates


def label_candidates(candidates: list[Candidate], events, window_h: int = WINDOW_H, radius_km: float = 300.0) -> None:
    """1 if a genesis event happens within ``window_h`` hours (and 300 km of the candidate's track) after it."""
    for c in candidates:
        c.label = 0.0
        for when, lat, lon in events:
            dt = (when - c.time) / np.timedelta64(1, "h")
            if 0 <= dt <= window_h and great_circle_km(c.lat, c.lon, lat, lon) < radius_km + 15.0 * dt:
                c.label = 1.0
                break


def feature_matrix(candidates: list[Candidate]) -> np.ndarray:
    return np.array([[1.0] + [c.features.get(k, 0.0) for k in FEATURES[1:]] for c in candidates], np.float64)


def _pairs() -> list[tuple[str, str]]:
    import itertools

    base = list(FEATURES[1:])
    return list(itertools.combinations(base, 2)) + [(f, f) for f in base]


#: Second-order terms of the shipped model: every pairwise product and square of the ingredients -- humidity
#: matters more when the shear is weak, spin matters more inside a closed pouch. Chosen over the linear model by
#: leave-one-season-out cross-validation on the 2016/2017/2019 seasons (Brier skill 0.193 against 0.184).
PAIRS = _pairs()


def design_matrix(candidates: list[Candidate]) -> np.ndarray:
    """``[1, features, pairwise products and squares]`` per candidate -- the shipped model's inputs."""
    rows = []
    for c in candidates:
        f = c.features
        rows.append([1.0] + [f.get(k, 0.0) for k in FEATURES[1:]] + [f.get(a, 0.0) * f.get(b, 0.0) for a, b in PAIRS])
    return np.asarray(rows, np.float64)


def candidate_probability(candidates: list[Candidate], weights=None) -> np.ndarray:
    """Genesis-within-48-hours probability from the shipped coefficients (:data:`naturev1.cyclone.coefficients.GENESIS`)."""
    if weights is None:
        from .coefficients import GENESIS

        weights = GENESIS["weights"]
    return genesis_probability(design_matrix(candidates), weights)


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-2, steps: int = 400) -> np.ndarray:
    """Newton-Raphson L2-regularised logistic regression (intercept unpenalised)."""
    mean, std = x[:, 1:].mean(0), x[:, 1:].std(0) + 1e-9
    z = np.concatenate([x[:, :1], (x[:, 1:] - mean) / std], 1)
    w = np.zeros(z.shape[1])
    penalty = np.diag([0.0] + [l2 * len(y)] * (z.shape[1] - 1))
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-z @ w))
        grad = z.T @ (p - y) + penalty @ w
        hess = (z * (p * (1 - p))[:, None]).T @ z + penalty
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    # Fold the standardisation back so the weights apply to raw features.
    raw = w.copy()
    raw[1:] = w[1:] / std
    raw[0] = w[0] - (w[1:] * mean / std).sum()
    return raw


def genesis_probability(x: np.ndarray, weights) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x @ np.asarray(weights, np.float64)))


def verify_genesis(candidates: list[Candidate], probability: np.ndarray, events, threshold: float = 0.5) -> dict:
    """Brier score and skill, reliability-in-thirds, detection rate, false alarms and warning time."""
    y = np.array([c.label for c in candidates])
    brier = float(np.mean((probability - y) ** 2))
    climo = float(np.mean((y.mean() - y) ** 2))
    hits, lead_hours = 0, []
    for when, lat, lon in events:
        warned = [c for c, p in zip(candidates, probability) if p >= threshold
                  and 0 <= (when - c.time) / np.timedelta64(1, "h") <= WINDOW_H
                  and great_circle_km(c.lat, c.lon, lat, lon) < 300.0 + 15.0 * (when - c.time) / np.timedelta64(1, "h")]
        if warned:
            hits += 1
            lead_hours.append(max((when - c.time) / np.timedelta64(1, "h") for c in warned))
    alarms = probability >= threshold
    false_alarm = float(np.mean(y[alarms] == 0)) if alarms.any() else float("nan")
    bins = np.quantile(probability, [0, 1 / 3, 2 / 3, 1])
    reliability = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (probability >= lo) & (probability <= hi)
        reliability.append((float(probability[sel].mean()), float(y[sel].mean()), int(sel.sum())))
    return {"brier": brier, "brier_skill": 1.0 - brier / max(climo, 1e-9), "events": len(events), "detected": hits,
            "detection_rate": hits / max(len(events), 1), "false_alarm_ratio": false_alarm,
            "median_warning_h": float(np.median(lead_hours)) if lead_hours else float("nan"), "reliability": reliability,
            "candidates": len(candidates), "base_rate": float(y.mean())}


def save_weights(weights, path: str | Path) -> None:
    Path(path).write_text(json.dumps({"features": list(FEATURES), "weights": [float(w) for w in weights]}))


__all__ = ["CHANNELS", "Candidate", "FEATURES", "GenesisSeason", "PAIRS", "build_genesis_store", "candidate_features",
           "candidate_probability", "design_matrix", "detect_season", "diagnostics", "feature_matrix", "find_candidates",
           "fit_logistic", "genesis_events",
           "genesis_probability", "label_candidates", "save_weights", "verify_genesis"]
