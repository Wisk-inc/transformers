# Copyright 2026 Nathan. Apache-2.0.
"""
A full year, 14 days ahead, around the world: Pluvium Blend against the operational systems.

* **Where**: six regions on the 0.25-degree grid the systems publish (the Caribbean; the Maritime Continent; the Alps;
  the US Pacific Northwest; the US Great Plains; southern Chile and the Andes), and countries at 8 km where Pluvium's
  network runs. Land cells only.
* **When**: every season. Blends are trained on April 2024 to March 2025 and scored on April 2025 to March 2026.
  They are refitted every quarter on everything verified by then (expanding window, never a future day).
* **How far**: days 1 to 3 from runs every second day, days 1 to 14 from runs every fourth day.
* **Who**: ECMWF ENS, NOAA GEFS, ECMWF HRES, ECMWF AIFS, NOAA GraphCast (to day 10) and NOAA GFS. Each is scored
  raw and post-processed with the same small network trained on its own inputs alone. Pluvium Blend pools those
  post-processed forecasts and the same network trained on all six at once (and, where it runs, Pluvium's own
  network), with weights fitted on weeks the components were not trained on (:class:`pluvium.blend.StackPool`).
* **Scores**: Brier score and skill against the training year's climatology, critical success index (the chance cut
  chosen on training data), CRPS, RMSE and anomaly correlation of the daily amount, ROC area, the spatial sharpness of
  the expected field (does it blur with lead?), by season, with a block bootstrap over runs for Pluvium Blend against
  the best post-processed system.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import blend as B
from ..sources.archive import ARCHIVE_SYSTEMS, LIMIT, REGIONS, ArchiveReader, run_set
from ..sources.cmorph_daily import DailyTruth
from ..util import hour_of, hour_to_datetime64, log

ENSEMBLES = {"ens", "gefs", "neural"}
LABELS = {"ens": "ECMWF ENS", "gefs": "NOAA GEFS", "ifs": "ECMWF HRES", "aifs": "ECMWF AIFS",
          "graphcast": "NOAA GraphCast", "gfs": "NOAA GFS", "neural": "Pluvium network", "blend": "Pluvium Blend"}
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
LEAD_GROUPS = {"days 1-3": (1, 2, 3), "days 4-7": (4, 5, 6, 7), "days 8-10": (8, 9, 10), "days 11-14": (11, 12, 13, 14)}


@dataclass
class FullYearConfig:
    archive: str
    truth: str
    out: str
    regions: tuple = tuple(REGIONS)
    train_start: str = "2024-04-01T00"
    test_start: str = "2025-04-01T00"
    test_end: str = "2026-04-01T00"
    refit_months: int = 3
    max_cells: int = 1500
    max_rows: int = 250_000
    #: nets per model, each on its own subsample of ``members * max_rows`` training rows, chances averaged
    members: int = 1
    #: lead ranges (days, inclusive) that get their own nets; one range means one net for every lead
    lead_ranges: tuple = ((1, 14),)
    #: chance of hiding each system from a training row (blends only: the blend cannot lean on one combination)
    system_drop: float = 0.0
    #: Pluvium Blend as the pool of every model (each system post-processed, and the networks over several)
    stack: bool = True
    #: training rows for the pool's weights
    stack_rows: int = 200_000
    thresholds: tuple = B.SCORED
    seed: int = 0
    #: days whose forecasts are kept in full for case studies (Hurricane Melissa struck Jamaica on 28 October 2025)
    case_days: tuple = ("2025-10-27", "2025-10-28", "2025-10-29")
    extra: dict = field(default_factory=dict)


def _month(hour: int) -> int:
    return int(str(hour_to_datetime64(int(hour)))[5:7])


def _day(hour: int) -> str:
    return str(hour_to_datetime64(int(hour)))[:10]


# ------------------------------------------------------------------------------------- the data --

def land_mask(reader: ArchiveReader, region: str, lat: np.ndarray, lon: np.ndarray, cache: Path) -> np.ndarray:
    """Land fraction >= 0.5 at the truth cells, from ECMWF's land-sea mask (one message, cached)."""
    path = cache / f"lsm_{region}.npy"
    if path.exists():
        return np.load(path)
    from ..dataset import bilinear
    from ..sources.baselines import ECMWF, ECMWF_ARCHIVES, _ecmwf_rows, _first
    from ..sources.nwp import decode

    urls = [ECMWF.format(base=b, day="20250401", hour=0, path="ifs/0p25/oper", step=0, kind="oper-fc")
            for b in ECMWF_ARCHIVES]
    rows = _ecmwf_rows(_first(urls, ".index").decode(), param="lsm")
    if not rows:
        log(f"{region}: no land-sea mask in the archive; scoring every cell")
        return np.ones((lat.size, lon.size), bool)
    _, a, b = rows[0]
    values, la, lo = decode(_first(urls, ".grib2", a, b))
    i = (la >= lat.min() - 1) & (la <= lat.max() + 1)
    j = (lo >= lon.min() - 1) & (lo <= lon.max() + 1)
    g = np.meshgrid(lat, lon, indexing="ij")
    frac = bilinear(values[np.ix_(i, j)], la[i], lo[j], g[0].ravel(), g[1].ravel()).reshape(g[0].shape)
    mask = frac >= 0.5
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mask)
    return mask


class CellSet:
    """The cells scored in a region: their coordinates, how to read truth there, and the climatology. ``grid_fn``
    (optional) reads the whole grid and ``index`` picks the cells from it, so the climatology can be pooled over
    neighbouring cells (one year of daily rain per cell is noisy)."""

    def __init__(self, lat, lon, truth_fn, name: str, grid_fn=None, index=None) -> None:
        self.lat, self.lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        self.truth_fn, self.name = truth_fn, name
        self.grid_fn, self.index = grid_fn, index
        self.clim = None

    def __len__(self) -> int:
        return self.lat.size

    def climatology(self, day_hours: np.ndarray, thresholds=(1.0, 10.0, 25.0, 50.0)) -> None:
        """By calendar month, from ``day_hours`` (the training year): chance of each threshold and mean, pooled over
        the month before and after."""
        from scipy.ndimage import uniform_filter

        pooled = self.grid_fn is not None
        rain = np.stack([(self.grid_fn if pooled else self.truth_fn)(int(h)) for h in day_hours])   # (D, ...)
        months = np.array([_month(h) for h in day_hours])
        clim = {}
        for m in range(1, 13):
            near = np.isin(months, [(m - 2) % 12 + 1, m, m % 12 + 1])
            r = rain[near]
            ok = np.isfinite(r)
            sums = {f"clim_p{int(t)}": ((r >= t) & ok).sum(0).astype(np.float64) for t in thresholds}
            sums["mean_mm"] = np.where(ok, r, 0).sum(0).astype(np.float64)
            count = ok.sum(0).astype(np.float64)
            if pooled:                                                  # pool each cell with its 8 neighbours
                sums = {k: uniform_filter(v, size=3, mode="nearest") for k, v in sums.items()}
                count = uniform_filter(count, size=3, mode="nearest")
            entry = {k: v / np.maximum(count, 1e-9) for k, v in sums.items()}
            entry["clim_mean"] = np.log1p(entry["mean_mm"])
            if pooled:
                entry = {k: v[self.index] for k, v in entry.items()}
            clim[m] = {k: v.astype(np.float32) for k, v in entry.items()}
        self.clim = clim

    def clim_at(self, valid_hour: int) -> dict:
        return self.clim[_month(valid_hour)]


def daily_members(z: dict, lead: int) -> np.ndarray | None:
    """Rain of day ``lead`` (1-based) from cumulative values ``(M, S, H, W)``; None when the run stops earlier."""
    v = z["values"]
    if v.shape[1] < lead:
        return None
    day = v[:, lead - 1] - (v[:, lead - 2] if lead > 1 else 0.0)
    return np.maximum(day, 0.0)


def system_at_cells(z: dict, lead: int, cells: CellSet) -> dict | None:
    """One system's features at the cells for one lead, with its neighbourhood (half a degree) features."""
    from scipy.ndimage import uniform_filter

    from ..dataset import bilinear

    day = daily_members(z, lead)
    if day is None:
        return None
    finite = np.isfinite(day).reshape(day.shape[0], -1).all(1)
    if not finite.any():
        return None
    day = day[finite]
    lat, lon = np.asarray(z["lat"], np.float64), np.asarray(z["lon"], np.float64)
    step = abs(lon[1] - lon[0])
    k = 2 * int(round(0.5 / step)) + 1                                              # cells within half a degree
    at = bilinear(day, lat, lon, cells.lat, cells.lon)                              # (M, N)
    nb = [bilinear(uniform_filter(day, size=(1, k, k), mode="nearest"), lat, lon, cells.lat, cells.lon)]
    feats = B.system_features(at)
    feats["nb_mean"] = np.log1p(np.maximum(nb[0], 0).mean(0)).astype(np.float32)
    for t in (1, 10):
        frac = uniform_filter((day >= t).astype(np.float32), size=(1, k, k), mode="nearest")
        feats[f"nb_p{t}"] = bilinear(frac, lat, lon, cells.lat, cells.lon).mean(0).astype(np.float32)
    feats["_members"] = at.astype(np.float32)
    return feats


# ---------------------------------------------------------------------------------- the dataset --

class Table:
    """Rows of (run, lead, cell): features of every system, truth, and raw forecasts for scoring."""

    def __init__(self) -> None:
        self.parts = []

    def add(self, run: int, lead: int, feats: dict, clim: dict, y: np.ndarray, n: int) -> None:
        self.parts.append((run, lead, feats, clim, y, n))


def build_table(cells: CellSet, reader: ArchiveReader, region: str, runs: np.ndarray, long_runs: set,
                systems=ARCHIVE_SYSTEMS, neural=None, scores=None, test_window=None, keep_raw: bool = False) -> dict:
    """Everything needed to train and score in one region: per (run, lead) the features of every system, the
    climatology and the truth. Raw systems are scored into ``scores`` here for runs in ``test_window``.

    With ``keep_raw`` the table also keeps the raw systems' consensus for every block (see
    :func:`pluvium.tails.consensus.raw_consensus`): ``raw_p`` the chance of every threshold, ``raw_mean`` the mean
    amount and ``raw_n`` how many systems were there."""
    t0 = time.time()
    out = {"run": [], "lead": [], "y": [], "clim": [], "feats": {s: [] for s in systems}}
    if neural is not None:
        out["feats"]["neural"] = []
    if keep_raw:
        from ..tails.consensus import raw_consensus

        out.update({"raw_p": [], "raw_mean": [], "raw_n": []})
    for k, run in enumerate(runs):
        loaded = {s: reader.load(region, s, int(run)) for s in systems}
        leads = range(1, 15) if int(run) in long_runs else range(1, 4)
        for lead in leads:
            valid = int(run) + 24 * (lead - 1)
            y = cells.truth_fn(valid)
            if not np.isfinite(y).any():
                continue
            clim = cells.clim_at(valid)
            out["run"].append(int(run))
            out["lead"].append(lead)
            out["y"].append(y.astype(np.float32))
            out["clim"].append(clim)
            testing = scores is not None and test_window[0] <= int(run) < test_window[1]
            raw = {}
            for s in systems:
                f = system_at_cells(loaded[s], lead, cells) if loaded[s] is not None and lead <= LIMIT.get(s, 14) else None
                members = None if f is None else f.pop("_members")
                if testing and members is not None:
                    _score_raw(scores, s, int(run), lead, y, clim, members)
                if keep_raw and members is not None:
                    raw[s] = members
                out["feats"][s].append(f)
            if keep_raw:
                p_raw, mean_raw = raw_consensus(raw, len(cells))
                out["raw_p"].append(p_raw.astype(np.float16))
                out["raw_mean"].append(mean_raw.astype(np.float32))
                out["raw_n"].append(len(raw))
            if neural is not None:
                f = neural(int(run), lead)
                members = None if f is None else f.pop("_members")
                if testing and members is not None:
                    _score_raw(scores, "neural", int(run), lead, y, clim, members)
                out["feats"]["neural"].append(f)
        if k % 40 == 0:
            log(f"  {cells.name}: {k + 1} of {len(runs)} runs read ({time.time() - t0:.0f}s)")
    out["run"] = np.asarray(out["run"], np.int64)
    out["lead"] = np.asarray(out["lead"], np.int64)
    return out


def rows_for(table: dict, idx: np.ndarray, systems: list[str], n_cells: int, max_rows: int | None = None,
             seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feature rows, truth and block index for blocks ``idx`` (a random subset of cells per block when the rows
    would exceed ``max_rows``)."""
    rng = np.random.default_rng(seed)
    take = n_cells if max_rows is None else max(1, min(n_cells, int(np.ceil(max_rows / max(len(idx), 1)))))
    xs, ys, blocks = [], [], []
    for i in idx:
        per = {s: table["feats"][s][i] for s in systems}
        x = B.assemble(per, systems, ENSEMBLES, table["clim"][i], float(table["lead"][i]), n_cells)
        sel = np.arange(n_cells) if take >= n_cells else rng.choice(n_cells, take, replace=False)
        xs.append(x[sel])
        ys.append(table["y"][i][sel])
        blocks.append(np.full(sel.size, i))
    if not xs:
        return np.zeros((0, 1), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(blocks)


# -------------------------------------------------------------------------------------- scoring --

def crps_quantiles(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS of quantile forecasts ``(N, L)`` (levels (i + 0.5) / L) as an equally weighted ensemble."""
    from ..verify import crps_samples

    return crps_samples(q.T.astype(np.float64), y.astype(np.float64))


def csi(p: np.ndarray, y: np.ndarray, cut: float) -> float:
    yes = p >= cut
    hits = float((yes & y).sum())
    return hits / max(hits + float((yes & ~y).sum()) + float((~yes & y).sum()), 1.0)


def best_cut(p: np.ndarray, y: np.ndarray) -> float:
    cuts = np.linspace(0.05, 0.95, 19)
    return float(cuts[int(np.argmax([csi(p, y, c) for c in cuts]))])


def acc_per_run(fc: np.ndarray, ob: np.ndarray, clim: np.ndarray, blocks: np.ndarray) -> float:
    """Mean over runs of the spatial anomaly correlation (centred), runs with any variance only."""
    vals = []
    for b in np.unique(blocks):
        m = (blocks == b) & np.isfinite(ob)
        if m.sum() < 10:
            continue
        a, o = fc[m] - clim[m], ob[m] - clim[m]
        a, o = a - a.mean(), o - o.mean()
        den = math.sqrt(float((a * a).sum()) * float((o * o).sum()))
        if den > 0:
            vals.append(float((a * o).sum()) / den)
    return float(np.mean(vals)) if vals else float("nan")


def sharpness(fc: np.ndarray, ob: np.ndarray, blocks: np.ndarray) -> float:
    """How much spatial structure the forecast field has next to the observed one: the root of the forecast's spatial
    variance over the observed, each summed over every day (run and lead). 1 is as varied as reality; an expected
    value falls below 1 as uncertainty grows. Summing before dividing keeps dry days, when the observed field is
    almost flat, from dominating (a mean of daily ratios explodes on them)."""
    var_fc = var_ob = 0.0
    for b in np.unique(blocks):
        m = (blocks == b) & np.isfinite(ob)
        if m.sum() < 10:
            continue
        var_fc += float(np.var(fc[m])) * int(m.sum())
        var_ob += float(np.var(ob[m])) * int(m.sum())
    return float(np.sqrt(var_fc / var_ob)) if var_ob > 0 else float("nan")


def block_bootstrap(diff_by_run: np.ndarray, resamples: int = 1000, block: int = 5, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    n = diff_by_run.size
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    starts = rng.integers(0, max(1, n - block + 1), size=(resamples, math.ceil(n / block)))
    idx = (starts[..., None] + np.arange(block)).reshape(resamples, -1)[:, :n] % n
    means = diff_by_run[idx].mean(1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(diff_by_run.mean()), float(lo), float(hi)


class Scores:
    """Per model: per lead the pieces of every score (pooled over runs and cells), and the forecasts of case days."""

    def __init__(self, thresholds, case_days=()) -> None:
        self.thr = tuple(thresholds)
        self.data = {}
        self.case_days = set(case_days)
        self.cases = []

    def case(self, model: str, run: int, lead: int, y: np.ndarray, mean: np.ndarray, p50: np.ndarray, p100) -> None:
        valid = _day(run + 24 * (lead - 1))
        if valid in self.case_days:
            ok = np.isfinite(y)
            self.cases.append({"model": model, "run": _day(run), "lead": lead, "valid": valid,
                               "forecast_mean_mm": float(mean[ok].mean()), "forecast_max_mm": float(mean[ok].max()),
                               "p50": float(np.mean(p50[ok])), "p100": None if p100 is None else float(np.mean(p100[ok])),
                               "observed_mean_mm": float(y[ok].mean()), "observed_max_mm": float(y[ok].max()),
                               "observed_share_50": float(np.mean(y[ok] >= 50)), "observed_share_100": float(np.mean(y[ok] >= 100))})

    def add(self, model: str, lead: int, season: str, run_ids: np.ndarray, y: np.ndarray, p: dict, mean: np.ndarray,
            crps: np.ndarray, clim_p: dict, clim_mean: np.ndarray, cuts: dict | None) -> None:
        d = self.data.setdefault(model, {}).setdefault(lead, [])
        d.append({"season": season, "run": run_ids, "block": run_ids * 100 + lead, "y": y, "p": p, "mean": mean,
                  "crps": crps, "clim_p": clim_p, "clim_mean": clim_mean, "cuts": cuts})

    def summary(self, model: str, leads, season: str | None = None) -> dict:
        parts = [x for lead in leads for x in self.data.get(model, {}).get(lead, []) if season in (None, x["season"])]
        if not parts:
            return {}
        cat = lambda k: np.concatenate([x[k] for x in parts])                       # noqa: E731
        y, mean, crps, blocks = cat("y"), cat("mean"), cat("crps"), cat("block")
        clim_mean = cat("clim_mean")
        ok = np.isfinite(y)
        out = {"n": int(ok.sum()), "crps": float(np.nanmean(crps[ok])), "rmse": float(np.sqrt(np.nanmean((mean[ok] - y[ok]) ** 2))),
               "acc": acc_per_run(mean[ok], y[ok], np.expm1(clim_mean[ok]), blocks[ok]),
               "sharpness": sharpness(mean[ok], y[ok], blocks[ok])}
        for t in self.thr:
            key = f"{t:g}"
            p = np.concatenate([x["p"][key] for x in parts])[ok]
            pc = np.concatenate([x["clim_p"][key] for x in parts])[ok]
            yy = y[ok] >= t
            bs, bs_ref = float(np.mean((p - yy) ** 2)), float(np.mean((pc - yy) ** 2))
            entry = {"bs": bs, "bss": 1 - bs / max(bs_ref, 1e-12), "base_rate": float(yy.mean()),
                     "never": 1 - float(yy.mean()) / max(bs_ref, 1e-12)}
            cuts = [x["cuts"][key] for x in parts if x["cuts"] is not None]
            cut = float(np.median(cuts)) if cuts else 0.5
            entry["csi"] = csi(p, yy, cut)
            from .operational import roc_area

            entry["auc"] = roc_area(p, yy)
            out[key] = entry
        return out

    def by_run(self, model: str, leads, t: float) -> dict:
        """Brier score per run (for the bootstrap)."""
        acc = {}
        for lead in leads:
            for x in self.data.get(model, {}).get(lead, []):
                ok = np.isfinite(x["y"])
                err = (x["p"][f"{t:g}"][ok] - (x["y"][ok] >= t)) ** 2
                for r in np.unique(x["run"][ok]):
                    acc.setdefault(int(r), []).append(err[x["run"][ok] == r])
        return {r: float(np.concatenate(v).mean()) for r, v in acc.items()}


# ------------------------------------------------------------------------------------ one region --

def _fit_model(table: dict, idx: np.ndarray, members: list, n: int, cfg: FullYearConfig):
    """One model (a single system, or a network over several) on blocks ``idx``: the net and its CSI cuts."""
    x, y, _ = rows_for(table, idx, members, n, max_rows=cfg.max_rows * cfg.members, seed=cfg.seed)
    if x.shape[0] == 0:
        return None
    groups = B.system_groups(members, ENSEMBLES) if len(members) > 1 else None
    net = B.fit(x, y, seed=cfg.seed, max_rows=cfg.max_rows, members=cfg.members, groups=groups, drop=cfg.system_drop)
    pick = np.random.default_rng(cfg.seed).choice(x.shape[0], min(200_000, x.shape[0]), replace=False)
    p = B.predict(net, x[pick])
    return net, {f"{t:g}": best_cut(B.exceed(p, t), y[pick] >= t) for t in cfg.thresholds}


def _usable(table: dict, blocks, name: str, members: list, systems) -> np.ndarray:
    """Blocks a model can forecast: single systems only where they exist, networks over several where any does."""
    if name in systems or name == "neural_pp":
        return np.asarray([i for i in blocks if table["feats"][members[0]][i] is not None], np.int64)
    return np.asarray([i for i in blocks if any(table["feats"][s][i] is not None for s in members)], np.int64)


def _rows_x(table: dict, i: int, members: list, n: int) -> np.ndarray:
    return B.assemble({s: table["feats"][s][i] for s in members}, members, ENSEMBLES, table["clim"][i],
                      float(table["lead"][i]), n)


def torch_lead(days: np.ndarray):
    """Leads in days as the pool's input (days / 14)."""
    import torch

    return torch.from_numpy(np.asarray(days, np.float32) / 14.0)


def _fit_stack(table: dict, train: np.ndarray, models: dict, systems, n: int, cfg: FullYearConfig):
    """Pluvium Blend's pool, fitted on forecasts each component made for weeks it was not trained on (two folds
    of alternate weeks, by the day forecast, so no fold model ever saw the truth it is judged on)."""
    runs, leads = table["run"], table["lead"]
    week = ((runs + 24 * (leads - 1)) // (24 * 7)) % 2
    names = list(models)
    rows = _usable(table, train, "stack", sorted({s for m in models.values() for s in m}), systems)
    rng = np.random.default_rng(cfg.seed + 7)
    take = max(1, min(n, int(np.ceil(cfg.stack_rows / max(rows.size, 1)))))
    cells = {int(i): (np.arange(n) if take >= n else np.sort(rng.choice(n, take, replace=False))) for i in rows}
    order = [(int(i), c) for i in rows for c in [cells[int(i)]]]
    offsets = np.cumsum([0] + [c.size for _, c in order])
    total = int(offsets[-1])
    z = np.zeros((total, len(names), len(B.THRESHOLDS)), np.float32)
    present = np.zeros((total, len(names)), np.float32)
    for k, name in enumerate(names):
        members = models[name]
        usable = _usable(table, train, name, members, systems)
        for fold in (0, 1):
            fitted = _fit_model(table, usable[week[usable] != fold], members, n, cfg)
            if fitted is None:
                continue
            net, _ = fitted
            for r, (i, c) in enumerate(order):
                if week[i] != fold or (name in systems or name == "neural_pp") and table["feats"][members[0]][i] is None:
                    continue
                if not any(table["feats"][s][i] is not None for s in members):
                    continue
                z[offsets[r]:offsets[r + 1], k] = B.to_logits(B.predict(net, _rows_x(table, i, members, n)[c]))
                present[offsets[r]:offsets[r + 1], k] = 1.0
    y = np.concatenate([table["y"][i][c] for i, c in order])
    lead = np.concatenate([np.full(c.size, float(leads[i])) for i, c in order])
    pool = B.fit_stack(z, present, lead, y)
    p = B.predict_stack(pool, z, present, lead)
    ok = np.isfinite(y)
    cuts = {f"{t:g}": best_cut(B.exceed(p[ok], t), y[ok] >= t) for t in cfg.thresholds}
    w = pool.weights(torch_lead(np.array([1.0, 7.0, 14.0]))).detach().numpy()[:, [2, 7]]   # 1 and 10 mm
    log("  pool weights at 1 mm and 10 mm (days 1, 7, 14): " + ", ".join(
        f"{m} {w[0, 0, j]:.2f}/{w[1, 0, j]:.2f}/{w[2, 0, j]:.2f} | {w[0, 1, j]:.2f}/{w[1, 1, j]:.2f}/{w[2, 1, j]:.2f}"
        for j, m in enumerate(names)))
    return pool, names, cuts


def evaluate_cells(cells: CellSet, table: dict, cfg: FullYearConfig, systems, with_neural: bool = False,
                   scores=None) -> dict:
    """Train (quarterly, expanding) and score every model on one set of cells (``scores`` already holds the raw
    systems, scored while the table was built). With ``cfg.stack``, Pluvium Blend is the pool of every model."""
    n = len(cells)
    runs, leads = table["run"], table["lead"]
    valid_end = runs + 24 * leads                                                 # when each block's truth is known
    test_start, test_end = hour_of(cfg.test_start), hour_of(cfg.test_end)
    refits = []
    t = test_start
    while t < test_end:
        refits.append(t)
        y_, m_ = int(_day(t)[:4]), int(_day(t)[5:7]) - 1 + cfg.refit_months
        t = hour_of(f"{y_ + m_ // 12}-{m_ % 12 + 1:02d}-01T00")
    refits.append(test_end)
    models = {s: [s] for s in systems}
    models["blend_ops"] = list(systems)
    if with_neural:
        models["neural_pp"] = ["neural"]
        models["blend"] = list(systems) + ["neural"]
    stacking = cfg.stack and len(cfg.lead_ranges) == 1
    scores = scores if scores is not None else Scores(cfg.thresholds)
    test_blocks = np.flatnonzero((runs >= test_start) & (runs < test_end))
    for a, b in zip(refits[:-1], refits[1:]):
        train = np.flatnonzero(valid_end <= a)
        test = test_blocks[(runs[test_blocks] >= a) & (runs[test_blocks] < b)]
        if test.size == 0 or train.size == 0:
            continue
        log(f"  {cells.name}: fitting for {_day(a)} to {_day(b)} on {train.size} blocks ({train.size * n:,} rows)")
        fitted = {}
        for name, members in models.items():
            usable = _usable(table, train, name, members, systems)
            fitted[name] = [(lo, hi, _fit_model(table, usable[(leads[usable] >= lo) & (leads[usable] <= hi)], members,
                                                n, cfg)) for lo, hi in cfg.lead_ranges]
        pool = _fit_stack(table, train, models, systems, n, cfg) if stacking else None
        for i in test:
            z = np.zeros((n, len(models), len(B.THRESHOLDS)), np.float32)
            present = np.zeros((n, len(models)), np.float32)
            for k, (name, members) in enumerate(models.items()):
                if (name in systems or name == "neural_pp") and table["feats"][members[0]][i] is None:
                    continue
                if not any(table["feats"][s][i] is not None for s in members):
                    continue
                fit_ = next((f for lo, hi, f in fitted[name] if lo <= leads[i] <= hi), None)
                if fit_ is None:
                    continue
                net, cuts = fit_
                p = B.predict(net, _rows_x(table, i, members, n))
                _score(scores, name, i, table, p=p, q=B.quantiles(p), cuts=cuts)
                z[:, k], present[:, k] = B.to_logits(p), 1.0
            if pool is not None and present.any():
                p = B.predict_stack(pool[0], z, present, float(leads[i]))
                _score(scores, "stack", i, table, p=p, q=B.quantiles(p), cuts=pool[2])
    if stacking:
        models["stack"] = list(models)
    return _report(scores, cells, models, systems, with_neural, cfg)


def _season(hour: int) -> str:
    m = _month(hour)
    return next(k for k, v in SEASONS.items() if m in v)


def _score(scores: Scores, name: str, i: int, table: dict, p: np.ndarray, q: np.ndarray, cuts: dict) -> None:
    y = table["y"][i]
    clim = table["clim"][i]
    lead = int(table["lead"][i])
    valid = int(table["run"][i]) + 24 * (lead - 1)
    probs = {f"{t:g}": B.exceed(p, t) for t in scores.thr}
    clim_p = {f"{t:g}": clim[f"clim_p{int(t)}"] for t in scores.thr}
    scores.add(name, lead, _season(valid), np.full(y.size, int(table["run"][i])), y, probs, q.mean(1),
               crps_quantiles(q, np.nan_to_num(y)), clim_p, clim["clim_mean"], cuts)
    scores.case(name, int(table["run"][i]), lead, y, q.mean(1), probs["50"] if "50" in probs else B.exceed(p, 50.0),
                B.exceed(p, 100.0))


def _score_raw(scores: Scores, s: str, run: int, lead: int, y: np.ndarray, clim: dict, members: np.ndarray) -> None:
    from ..verify import crps_samples

    valid = run + 24 * (lead - 1)
    m = members[np.isfinite(members).all(1)]
    if m.shape[0] == 0:
        return
    probs = {f"{t:g}": (m >= t).mean(0) for t in scores.thr}
    clim_p = {f"{t:g}": clim[f"clim_p{int(t)}"] for t in scores.thr}
    cuts = {f"{t:g}": 0.5 for t in scores.thr}
    scores.add(f"{s}_raw", lead, _season(valid), np.full(y.size, run), y, probs, m.mean(0),
               crps_samples(m.astype(np.float64), np.nan_to_num(y).astype(np.float64)), clim_p, clim["clim_mean"], cuts)
    scores.case(f"{s}_raw", run, lead, y, m.mean(0), (m >= 50).mean(0), (m >= 100).mean(0))


def _report(scores: Scores, cells: CellSet, models: dict, systems, with_neural: bool, cfg: FullYearConfig) -> dict:
    out = {"cells": len(cells), "leads": {}, "groups": {}, "seasons": {}, "versus": {}, "cases": scores.cases}
    names = [f"{s}_raw" for s in systems] + list(models) + (["neural_raw"] if with_neural else [])
    for lead in range(1, 15):
        out["leads"][str(lead)] = {m: scores.summary(m, [lead]) for m in names if scores.data.get(m, {}).get(lead)}
    for g, leads in LEAD_GROUPS.items():
        out["groups"][g] = {m: scores.summary(m, leads) for m in names}
        out["seasons"][g] = {season: {m: scores.summary(m, leads, season) for m in names} for season in SEASONS}
    top = "stack" if "stack" in models else ("blend" if with_neural else "blend_ops")
    rivals = list(systems) + [m for m in ("blend_ops", "neural_pp", "blend") if m in models and m != top]
    for g, leads in LEAD_GROUPS.items():
        for t in cfg.thresholds[:2]:
            mine = scores.by_run(top, leads, t)
            for rival in rivals:
                theirs = scores.by_run(rival, leads, t)
                common = sorted(set(mine) & set(theirs))
                if len(common) < 3:
                    continue
                diff = np.array([mine[r] - theirs[r] for r in common])
                mean, lo, hi = block_bootstrap(diff, seed=cfg.seed)
                out["versus"].setdefault(g, {}).setdefault(f"{t:g}", {})[rival] = {"delta_bs": mean, "ci95": [lo, hi],
                                                                                    "runs": len(common)}
    return out


# ------------------------------------------------------------------------------------- the run --

def region_cells(cfg: FullYearConfig, region: str, reader: ArchiveReader) -> CellSet:
    truth = DailyTruth(cfg.truth)
    lat, lon = truth.grid(region)
    mask = land_mask(reader, region, lat, lon, Path(cfg.out) / "masks")
    ii, jj = np.nonzero(mask)
    if ii.size > cfg.max_cells:                                                  # a regular subset of land cells
        keep = np.linspace(0, ii.size - 1, cfg.max_cells).astype(int)
        ii, jj = ii[keep], jj[keep]

    def truth_fn(day_hour: int) -> np.ndarray:
        return truth.days(region, [day_hour])[0][ii, jj]

    def grid_fn(day_hour: int) -> np.ndarray:
        return truth.days(region, [day_hour])[0]

    return CellSet(lat[ii], lon[jj], truth_fn, region, grid_fn=grid_fn, index=(ii, jj))


def run_region(cfg: FullYearConfig, region: str) -> dict:
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{region}.json"
    if path.exists():
        return json.loads(path.read_text())
    reader = ArchiveReader(cfg.archive)
    cells = region_cells(cfg, region, reader)
    train_days = np.arange(hour_of(cfg.train_start), hour_of(cfg.test_start), 24)
    cells.climatology(train_days)
    runs, long = run_set(cfg.train_start, cfg.test_end, every=2, long_every=4)
    log(f"{region}: {len(cells)} land cells, {len(runs)} runs ({len(long)} to 14 days)")
    scores = Scores(cfg.thresholds, cfg.case_days)
    table = build_table(cells, reader, region, runs, set(int(r) for r in long), scores=scores,
                        test_window=(hour_of(cfg.test_start), hour_of(cfg.test_end)))
    res = evaluate_cells(cells, table, cfg, list(ARCHIVE_SYSTEMS), scores=scores)
    res["region"] = region
    path.write_text(json.dumps(res, indent=1, default=float))
    return res


def country_cells(name: str, geography: str, cmorph_root: str) -> CellSet:
    """A country's land cells at 8 km (a Pluvium geography), with CMORPH truth summed over each UTC day."""
    from ..domain import Geography
    from ..sources.cmorph import CMORPHCache

    geo = Geography.load(geography)
    land = np.flatnonzero(geo.targets.kind <= 1)
    cm = CMORPHCache(cmorph_root)

    def truth_fn(day_hour: int) -> np.ndarray:
        r = cm.targets(int(day_hour) + np.arange(24), land)
        return r.sum(0)                                                        # NaN when an hour is missing

    return CellSet(geo.targets.lat[land], geo.targets.lon[land], truth_fn, name)


def neural_reader(forecasts: str):
    """Pluvium's network as a system: its 400 sampled daily totals at the cells are its members."""
    folder = Path(forecasts)

    def read(run: int, lead: int) -> dict | None:
        f = folder / f"{_day(run)}.npz"
        if lead > 3 or not f.exists():
            return None
        samples = np.load(f)["samples"][:, lead - 1].astype(np.float32)            # (S, P)
        feats = B.system_features(samples)
        feats["_members"] = samples
        return feats

    return read


def run_country(cfg: FullYearConfig, name: str, geography: str, cmorph_root: str, forecasts: str,
                clim_span: tuple, region: str = "caribbean") -> dict:
    """Pluvium Blend with Pluvium's network, at a country's 8 km cells, days 1-14."""
    out_dir = Path(cfg.out)
    path = out_dir / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    cells = country_cells(name, geography, cmorph_root)
    cells.climatology(np.arange(hour_of(clim_span[0]), hour_of(clim_span[1]), 24))
    reader = ArchiveReader(cfg.archive)
    runs, long = run_set(cfg.train_start, cfg.test_end, every=2, long_every=4)
    log(f"{name}: {len(cells)} land cells at 8 km, {len(runs)} runs; climatology {clim_span[0][:10]} to {clim_span[1][:10]}")
    scores = Scores(cfg.thresholds, cfg.case_days)
    table = build_table(cells, reader, region, runs, set(int(r) for r in long), neural=neural_reader(forecasts),
                        scores=scores, test_window=(hour_of(cfg.test_start), hour_of(cfg.test_end)))
    res = evaluate_cells(cells, table, cfg, list(ARCHIVE_SYSTEMS), with_neural=True, scores=scores)
    res["region"] = name
    path.write_text(json.dumps(res, indent=1, default=float))
    return res


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="pluvium fullyear", description="Pluvium Blend against the operational systems: "
                                "a full year, 14 days, six regions.")
    p.add_argument("--archive", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--regions", default=",".join(REGIONS))
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--members", type=int, default=FullYearConfig.members, help="nets per model (a bag)")
    p.add_argument("--max-rows", type=int, default=FullYearConfig.max_rows, help="training rows per net")
    p.add_argument("--lead-ranges", default=None, help="lead ranges with their own nets, e.g. 1-3,4-14")
    p.add_argument("--system-drop", type=float, default=FullYearConfig.system_drop,
                   help="chance of hiding each system from a blend's training row")
    a = p.parse_args(argv)
    if a.threads:
        import torch

        torch.set_num_threads(a.threads)
    ranges = FullYearConfig.lead_ranges if not a.lead_ranges else \
        tuple(tuple(int(v) for v in r.split("-")) for r in a.lead_ranges.split(","))
    cfg = FullYearConfig(archive=a.archive, truth=a.truth, out=a.out, regions=tuple(a.regions.split(",")),
                         members=a.members, max_rows=a.max_rows, lead_ranges=ranges, system_drop=a.system_drop)
    (Path(cfg.out)).mkdir(parents=True, exist_ok=True)
    (Path(cfg.out) / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    for region in cfg.regions:
        run_region(cfg, region)


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------------------------------ report --

REGION_TITLES = {"caribbean": "Caribbean (tropical islands and coasts)", "maritime": "Maritime Continent (tropical, deep convection)",
                 "alps": "Alps (mid-latitude mountains)", "pacific_nw": "US Pacific Northwest (mid-latitude coast and mountains)",
                 "great_plains": "US Great Plains (mid-latitude storms)", "andes_south": "Southern Andes and Chile (southern mid-latitudes)",
                 "jamaica": "Jamaica at 8 km (with Pluvium's network)"}


#: short names of the benchmark's places, for table rows and sentences
SHORT_TITLES = {"jamaica": "Jamaica", "caribbean": "Caribbean", "maritime": "Maritime Continent", "alps": "Alps",
                "pacific_nw": "US Pacific Northwest", "great_plains": "US Great Plains", "andes_south": "Southern Andes"}


def place_name(region: str) -> str:
    """A place's short name for a table row: its own if it is one of the benchmark's, else its key made readable."""
    return SHORT_TITLES.get(region, region.replace("_", " ").title())


def _name(m: str, with_network: bool = False, stacked: bool = True) -> str:
    if m.endswith("_raw"):
        return f"{LABELS.get(m[:-4], m[:-4])} (raw)"
    if m == "stack":
        return "Pluvium Blend (with Pluvium's network)" if with_network else "Pluvium Blend"
    if m == "blend_ops":
        if stacked:
            return "One network over the six (a Blend component)"
        return "Blend of the six (no Pluvium network)" if with_network else "Pluvium Blend"
    if m == "neural_pp":
        return "Pluvium network (post-processed)"
    if m == "blend":
        return "One network over all seven (a Blend component)" if stacked else "Pluvium Blend (with Pluvium's network)"
    return f"{LABELS.get(m, m)} (post-processed)"


def leaders(res: dict, group: str, key: str, t: str | None = None, higher: bool = True) -> list:
    """Models ranked on one score in one lead group."""
    rows = []
    for m, v in res["groups"][group].items():
        if not v:
            continue
        val = v[t][key] if t else v[key]
        if val == val:
            rows.append((val, m))
    return sorted(rows, reverse=higher)


#: (title, key, threshold, higher is better) for the scoreboard
BOARD = (("Brier skill, 1 mm", "bss", "1", True), ("Brier skill, 10 mm", "bss", "10", True),
         ("Brier skill, 25 mm", "bss", "25", True), ("Brier skill, 50 mm", "bss", "50", True),
         ("CSI, 1 mm", "csi", "1", True), ("CSI, 10 mm", "csi", "10", True), ("ROC area, 1 mm", "auc", "1", True),
         ("CRPS", "crps", None, False), ("RMSE", "rmse", None, False), ("ACC", "acc", None, True))


def _value(v: dict, key: str, t: str | None) -> float:
    if not v:
        return float("nan")
    x = v[t][key] if t else v[key]
    return float(x) if x is not None else float("nan")


def _top(res: dict) -> str:
    g = res["groups"]["days 1-3"]
    return "stack" if g.get("stack") else ("blend" if g.get("blend") else "blend_ops")


def _ctx(res: dict) -> tuple[bool, bool]:
    """(Pluvium's network present, Blend stacked) for naming."""
    g = res["groups"]["days 1-3"]
    return bool(g.get("neural_pp") or g.get("blend")), bool(g.get("stack"))


def scoreboard(results: dict, group_of=LEAD_GROUPS, season: str | None = None) -> list[str]:
    """For every score and lead group: in how many places Pluvium's top product beat every other system, raw and
    post-processed (Pluvium's own variants are not counted as rivals)."""
    ours = {"stack", "blend", "blend_ops", "neural_pp", "neural_raw"}
    lines = ["| Score | " + " | ".join(group_of) + " |", "|---|" + "---|" * len(group_of)]
    for title, key, t, higher in BOARD:
        cells = []
        for g in group_of:
            won = total = 0
            for res in results.values():
                rows = res["groups"][g] if season is None else res["seasons"][g][season]
                mine = _value(rows.get(_top(res)), key, t)
                theirs = [_value(v, key, t) for m, v in rows.items() if m not in ours and v]
                theirs = [x for x in theirs if x == x]
                if mine != mine or not theirs:
                    continue
                total += 1
                won += (mine > max(theirs)) if higher else (mine < min(theirs))
            cells.append(f"{won} of {total}" if total else "-")
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    for t in ("25", "50"):                                  # the reference every system struggles with at rare amounts
        cells = []
        for g in group_of:
            won = total = 0
            for res in results.values():
                rows = res["groups"][g] if season is None else res["seasons"][g][season]
                v = rows.get(_top(res))
                if not v:
                    continue
                total += 1
                won += v[t]["bss"] > v[t]["never"]
            cells.append(f"{won} of {total}" if total else "-")
        lines.append(f"| Better than never saying rain, {t} mm | " + " | ".join(cells) + " |")
    return lines


def melissa(res: dict, day: str = "2025-10-28") -> list[str]:
    """Hurricane Melissa's landfall day: every system's chance of 50 mm or more and expected rain, by how far ahead."""
    rows = [c for c in res.get("cases", []) if c["valid"] == day]
    if not rows:
        return []
    obs = rows[0]
    leads = sorted({c["lead"] for c in rows})
    models = []
    for c in rows:
        if c["model"] not in models:
            models.append(c["model"])
    net, stacked = _ctx(res)
    lines = [f"Observed on {day} (CMORPH): {obs['observed_mean_mm']:.0f} mm on average over the cells, up to "
             f"{obs['observed_max_mm']:.0f} mm; {100 * obs['observed_share_50']:.0f}% of cells had 50 mm or more and "
             f"{100 * obs['observed_share_100']:.0f}% had 100 mm or more.", "",
             "Each cell: the chance of 50 mm or more, averaged over the cells, then the expected rain (mm).", "",
             "| System | " + " | ".join(f"{lead} days ahead" for lead in leads) + " |", "|---|" + "---|" * len(leads)]
    for m in models:
        cells = []
        for lead in leads:
            c = next((c for c in rows if c["model"] == m and c["lead"] == lead), None)
            cells.append("-" if c is None else f"{100 * c['p50']:.0f}%, {c['forecast_mean_mm']:.0f} mm")
        lines.append(f"| {_name(m, net, stacked)} | " + " | ".join(cells) + " |")
    return lines


def markdown(results: dict, cfg: dict) -> str:
    """The full-year benchmark as Markdown: one section per region, the leader of every score marked."""
    lines = ["# Pluvium Blend against the operational systems: a full year, 14 days, around the world", "",
             f"Blends trained on {cfg['train_start'][:10]} to {cfg['test_start'][:10]}, refitted every "
             f"{cfg['refit_months']} months on everything verified by then, and scored on {cfg['test_start'][:10]} to "
             f"{cfg['test_end'][:10]}. Truth: CMORPH daily totals, land cells only. Every system is scored raw and after "
             "the same post-processing; **bold** marks the best of all.", ""]
    for region, res in results.items():
        lines += [f"## {REGION_TITLES.get(region, region)}", "", f"{res['cells']:,} land cells.", ""]
        net, stacked = _ctx(res)
        for t in ("1", "10", "25"):
            lines += [f"**Brier skill score, {t} mm or more** (higher is better; 0 is climatology)", "",
                      "| System | " + " | ".join(LEAD_GROUPS) + " |", "|---|" + "---|" * len(LEAD_GROUPS)]
            models = [m for m in res["groups"]["days 1-3"] if res["groups"]["days 1-3"][m]]
            best = {g: (leaders(res, g, "bss", t) or [(None, None)])[0][1] for g in LEAD_GROUPS}
            for m in models:
                cells = []
                for g in LEAD_GROUPS:
                    v = res["groups"][g].get(m) or {}
                    if not v:
                        cells.append("-")
                        continue
                    txt = f"{v[t]['bss']:+.3f}"
                    cells.append(f"**{txt}**" if best[g] == m else txt)
                lines.append(f"| {_name(m, net, stacked)} | " + " | ".join(cells) + " |")
            never = [res["groups"][g][models[0]][t]["never"] if res["groups"][g].get(models[0]) else float("nan")
                     for g in LEAD_GROUPS]
            lines += ["| *never says rain* | " + " | ".join(f"*{v:+.3f}*" for v in never) + " |", ""]
        lines += ["**Other scores, days 1-3** (CSI at 1 and 10 mm, higher is better; CRPS and RMSE in mm, lower is better; "
                  "ACC higher is better; sharpness = spatial spread of the expected field over the observed one)", "",
                  "| System | CSI 1 mm | CSI 10 mm | CRPS | RMSE | ACC | Sharpness |", "|---|---|---|---|---|---|---|"]
        g = "days 1-3"
        marks = {"csi1": leaders(res, g, "csi", "1"), "csi10": leaders(res, g, "csi", "10"),
                 "crps": leaders(res, g, "crps", higher=False), "rmse": leaders(res, g, "rmse", higher=False),
                 "acc": leaders(res, g, "acc")}
        for m, v in res["groups"][g].items():
            if not v:
                continue
            def fmt(key, val, digits=3):
                s_ = f"{val:.{digits}f}"
                return f"**{s_}**" if marks[key] and marks[key][0][1] == m else s_
            lines.append(f"| {_name(m, net, stacked)} | {fmt('csi1', v['1']['csi'])} | {fmt('csi10', v['10']['csi'])} | "
                         f"{fmt('crps', v['crps'], 2)} | {fmt('rmse', v['rmse'], 2)} | {fmt('acc', v['acc'])} | "
                         f"{v['sharpness']:.2f} |")
        lines.append("")
        top = _top(res)
        singles = [m for m in res["leads"].get("1", {}) if m in ARCHIVE_SYSTEMS]
        lines += ["**Lead by lead, 1 mm or more** (Brier skill score; the best post-processed system named)", "",
                  "| Day | Pluvium Blend | Best other (post-processed) | ECMWF ENS (raw) | Pluvium lead |",
                  "|---|---|---|---|---|"]
        for lead in range(1, 15):
            row = res["leads"].get(str(lead), {})
            if not row.get(top):
                continue
            mine = row[top]["1"]["bss"]
            rivals = [(row[m]["1"]["bss"], m) for m in singles if row.get(m)]
            best_v, best_m = max(rivals) if rivals else (float("nan"), "-")
            ens = row.get("ens_raw", {}).get("1", {}).get("bss", float("nan"))
            lines.append(f"| {lead} | {mine:+.3f} | {best_v:+.3f} ({_name(best_m, net, stacked)}) | {ens:+.3f} | {mine - best_v:+.3f} |")
        lines += ["", "**By season, days 1-3, 1 mm or more** (Brier skill score)", "",
                  "| Season | Pluvium Blend | Best other (post-processed) |", "|---|---|---|"]
        for season in SEASONS:
            row = res["seasons"]["days 1-3"][season]
            if not row.get(top):
                continue
            rivals = [(row[m]["1"]["bss"], m) for m in singles if row.get(m)]
            best_v, best_m = max(rivals) if rivals else (float("nan"), "-")
            lines.append(f"| {season} | {row[top]['1']['bss']:+.3f} | {best_v:+.3f} ({_name(best_m, net, stacked)}) |")
        lines.append("")
        vs = res.get("versus", {})
        if vs:
            top = _name(_top(res), net, stacked)
            lines += [f"**{top} against each post-processed system**: Brier score difference at 1 mm (negative: "
                      "Pluvium better), 95% block-bootstrap interval over runs. Bold when Pluvium is better with 95% "
                      "confidence.", "", "| Against | " + " | ".join(LEAD_GROUPS) + " |", "|---|" + "---|" * len(LEAD_GROUPS)]
            rivals = sorted({r for gg in vs.values() for r in gg.get("1", {})})
            for r in rivals:
                cells = []
                for gg in LEAD_GROUPS:
                    c = vs.get(gg, {}).get("1", {}).get(r)
                    if not c:
                        cells.append("-")
                        continue
                    txt = f"{c['delta_bs']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
                    cells.append(f"**{txt}**" if c["ci95"][1] < 0 else txt)
                lines.append(f"| {_name(r, net, stacked)} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines) + "\n"
