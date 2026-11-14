# Copyright 2026 Nathan. Apache-2.0.
"""
Pluvium against the operational forecasting systems, on real forecasts only.

The question: given the forecasts that existed on each morning, does Pluvium give better chances of rain for each of
the next three days, at every 8 km land cell, than the systems forecasters use? Every input is out-of-sample and
operational:

* **Pluvium** is driven by that morning's real GFS run (0.5 degrees, NOAA's archive), never by a reanalysis. Its
  weights are fine-tuned (stage B) on GFS runs that all end before the first calibration day; nothing later is seen
  in training. Its other inputs are the satellite rain of the last three hours and the airport reports, both
  available before the GFS run is.
* **The baselines** are the operational products as published (:mod:`pluvium.sources.baselines`): ECMWF ENS
  (51 members), NOAA GEFS (31 members), ECMWF HRES, ECMWF AIFS, NOAA GraphCast and NOAA GFS, all from the same
  00 UTC cycle and verified on the same windows.
* **Calibration** is the same for every system, Pluvium included, and causal: on each test day an isotonic
  regression per day ahead and threshold is fitted on the 30 previous runs whose outcomes were known by then.
  Each system's predictor is the better (on that window) of its point probability and its neighbourhood probability
  (members and grid points within 0.5 degrees), so the baselines get their strongest standard treatment.
* **Blends**: a logistic regression on all six operational systems (the strongest reference: what a forecaster
  could build from everything published), and the same with Pluvium added, which measures what Pluvium adds to all
  of them.
* **Truth** is CMORPH (satellite rain adjusted to gauges) at every land cell. The reference for skill is a CMORPH
  climatology of the same weeks of the year from earlier years.

Scores: Brier score and Brier skill score for 24-hour rain of at least 1, 10, 25 and 50 mm on days 1, 2 and 3; ROC
area; the CRPS of the 24-hour totals; and, for Pluvium against every other system, the difference in Brier score
with a 95% interval from a moving-block bootstrap over days (blocks of 5 days keep the correlation between
neighbouring days and cells honest).

Run it in steps (each resumes) or all at once::

    python -m pluvium.benchmarks.operational all --root RUN --geography GEO --checkpoint CKPT --stations STATIONS
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..sources.baselines import SYSTEMS, BaselineReader, at_points, daily
from ..util import hour_of, hour_to_datetime64, log, parallel_map

THRESHOLDS = (1.0, 10.0, 25.0, 50.0)
DAYS = 3
PLUVIUM = "pluvium"
BLEND_OPS = "blend_ops"
BLEND_ALL = "blend_all"
LABELS = {PLUVIUM: "Pluvium (GFS-driven)", BLEND_OPS: "Blend of the 6 operational systems",
          BLEND_ALL: "Blend of the 6 + Pluvium", **{k: v[0] for k, v in SYSTEMS.items()}}


@dataclass
class OperationalConfig:
    root: str                                        # data, checkpoints and results
    geography: str                                   # a saved Geography
    checkpoint: str                                  # the stage A model to fine-tune on GFS runs
    stations: str | None = None                      # the model's airports (a StationCache folder holding the list)
    train_period: tuple = ("2024-05-01T00", "2025-03-29T00")
    calibration_start: str = "2025-04-01T00"
    test_period: tuple = ("2025-05-01T00", "2025-07-01T00")
    window_days: int = 30
    lag_days: int = 3
    calibration: str = "logistic"                    # "logistic" (frozen after May-June 2025) or "isotonic"
    reference_years: tuple = tuple(range(2017, 2024))  # plus the training year, for the climatology of skill
    thresholds: tuple = THRESHOLDS
    systems: tuple = tuple(SYSTEMS)
    finetune_steps: int = 900
    finetune_hours: float = 4.0                      # a ceiling; the step count ends the schedule
    lr: float = 7e-4
    batch_size: int = 2
    samples: int = 400
    group_size: int = 450                            # land cells decoded together, as in training
    ocean_cells: int = 100
    neighbourhood_deg: float = 0.5
    workers: int = 6
    seed: int = 0
    extra: dict = field(default_factory=dict)


def issue_times(period) -> np.ndarray:
    """The 00 UTC cycles in ``[start, end)``."""
    a, b = hour_of(period[0]), hour_of(period[1])
    return np.arange(a + (-a) % 24, b, 24, dtype=np.int64)


def _day(hour: int) -> str:
    return str(hour_to_datetime64(int(hour)))[:10]


# ------------------------------------------------------------------------------------- statistics --

def isotonic(x: np.ndarray, y: np.ndarray, blocks: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Increasing least-squares fit of ``y`` on ``x`` (pool-adjacent-violators on at most ``blocks`` quantile bins);
    returns knots ``(xs, ys)`` for :func:`numpy.interp`."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    ux, inv = np.unique(x, return_inverse=True)
    if ux.size > blocks:                                      # bin by quantiles of x (ties stay together)
        edges = np.unique(np.quantile(x, np.linspace(0, 1, blocks + 1)[1:-1]))
        inv = np.searchsorted(edges, x, side="right")
        ux = None
    n = np.bincount(inv).astype(np.float64)
    keep = n > 0
    sx = np.bincount(inv, x)[keep] / n[keep]
    sy = np.bincount(inv, y)[keep] / n[keep]
    w = n[keep]
    vals, wts, xs = [], [], []
    for xi, yi, wi in zip(sx, sy, w):
        vals.append(yi)
        wts.append(wi)
        xs.append([xi])
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wsum = wts[-2] + wts[-1]
            merged = xs[-2] + xs[-1]
            vals[-2:], wts[-2:], xs[-2:] = [v], [wsum], [merged]
    knots_x, knots_y = [], []
    for v, group in zip(vals, xs):                           # a flat step from the block's first to last x
        knots_x += [group[0], group[-1]]
        knots_y += [v, v]
    kx, ky = np.array(knots_x), np.array(knots_y)
    kx = kx + np.arange(kx.size) * 1e-12                     # strictly increasing for interp
    return kx, ky


def logistic_fit(features: np.ndarray, y: np.ndarray, l2: float = 1e-3, iterations: int = 25) -> np.ndarray:
    """Logistic regression with an intercept by Newton's method; returns the coefficients (intercept last)."""
    X = np.concatenate([features, np.ones((features.shape[0], 1))], 1).astype(np.float64)
    beta = np.zeros(X.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
        g = X.T @ (p - y) + l2 * np.r_[beta[:-1], 0.0] * len(y)
        h = (X * (p * (1 - p))[:, None]).T @ X + l2 * len(y) * np.diag(np.r_[np.ones(X.shape[1] - 1), 0.0])
        step = np.linalg.solve(h + 1e-9 * np.eye(X.shape[1]), g)
        beta -= step
        if np.abs(step).max() < 1e-7:
            break
    return beta


def _transform(x: np.ndarray, kind: str) -> np.ndarray:
    """A calibration predictor on a scale where a logistic regression is natural: log(1 + mm) for amounts, the logit
    for chances."""
    x = np.asarray(x, np.float64)
    return np.log1p(np.maximum(x, 0.0)) if kind == "amount" else _logit(x)


def calibrator(x: np.ndarray, y: np.ndarray, kind: str, method: str = "logistic"):
    """Fit ``P(y | x)`` and return it as a function. ``logistic``: two parameters on :func:`_transform` (robust on a
    30-day window). ``isotonic``: a monotone step function (flexible, and found to overfit a 30-day window in which
    one wet day dominates the highest chances)."""
    if method == "isotonic":
        kx, ky = isotonic(x, y)
        return lambda v: np.interp(np.asarray(v, np.float64), kx, ky)
    if method != "logistic":
        raise ValueError(f"unknown calibration {method!r}")
    beta = logistic_fit(_transform(x, kind)[:, None], y)
    return lambda v: logistic_apply(beta, _transform(v, kind)[..., None])


def logistic_apply(beta: np.ndarray, features: np.ndarray) -> np.ndarray:
    z = features @ beta[:-1] + beta[-1]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _logit(p: np.ndarray, eps: float = 0.01) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def roc_area(p: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney, ties counted half)."""
    from scipy.stats import rankdata

    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list:
    edges = np.linspace(0, 1, bins + 1)
    k = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = k == b
        if m.any():
            out.append([float(p[m].mean()), float(y[m].mean()), int(m.sum())])
    return out


def crps_ensemble(members: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS of an ensemble ``(M, N)`` for outcomes ``(N,)``; for one member it is the absolute error."""
    from ..verify import crps_samples

    return crps_samples(members.astype(np.float64), y.astype(np.float64))


def block_bootstrap(per_day: np.ndarray, resamples: int = 2000, block: int = 5, seed: int = 0) -> tuple:
    """Mean of ``per_day`` (days,) with a 95% moving-block bootstrap interval and the share of resamples below 0."""
    rng = np.random.default_rng(seed)
    n = per_day.size
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    starts = rng.integers(0, max(1, n - block + 1), size=(resamples, math.ceil(n / block)))
    idx = (starts[..., None] + np.arange(block)).reshape(resamples, -1)[:, :n] % n
    means = per_day[idx].mean(1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(per_day.mean()), float(lo), float(hi), float((means < 0).mean())


# ------------------------------------------------------------------------------------------ the run --

class Operational:
    def __init__(self, cfg: OperationalConfig) -> None:
        from ..domain import Geography
        from ..train import read_meta

        self.cfg = cfg
        self.root = Path(cfg.root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.geo = Geography.load(cfg.geography)
        t = self.geo.targets
        self.land = np.flatnonzero(t.kind <= 1)
        self.sea = np.flatnonzero(t.kind == 2)
        self.lat, self.lon = t.lat[self.land], t.lon[self.land]
        meta = read_meta(cfg.checkpoint)
        if not meta:
            raise FileNotFoundError(f"no model in {cfg.checkpoint}")
        self.grids = {k: np.asarray(meta["grids"][k], np.float64) for k in ("lat1", "lon1", "lat2", "lon2")}
        self.train_inits = issue_times(cfg.train_period)
        self.eval_inits = issue_times((cfg.calibration_start, cfg.test_period[1]))
        self.test_inits = issue_times(cfg.test_period)
        self.thr = tuple(float(x) for x in cfg.thresholds)
        (self.root / "config.json").write_text(json.dumps(asdict(cfg), indent=1, default=str))

    @property
    def box(self) -> tuple:
        return (float(self.lat.min()), float(self.lat.max()), float(self.lon.min()), float(self.lon.max()))

    # ------------------------------------------------------------------------------------ staging --
    def stage_truth(self, workers: int = 4) -> None:
        """CMORPH from a month before training to three days after the last test run, and the airports' reports."""
        from ..sources.cmorph import stage_cmorph
        from ..sources.stations import stage_stations

        a = hour_of(self.cfg.train_period[0]) - 24 * 30 - 6
        b = int(self.eval_inits[-1]) + 96
        log(f"operational: CMORPH {_day(a)} to {_day(b)}")
        stage_cmorph(self.root / "cmorph", self.geo.targets, self.geo.mesh, period=(a, b), workers=workers)
        if self.cfg.stations:
            listed = Path(self.cfg.stations).expanduser() / "stations" / "stations.json"
            dest = self.root / "stations" / "stations" / "stations.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(listed, dest)
            years = list(range(int(_day(a)[:4]), int(_day(b)[:4]) + 1))
            stage_stations(self.root / "stations", years, workers=16)

    def stage_reference(self, workers: int = 4) -> None:
        """CMORPH for the same weeks of the reference years (the climatology that skill is measured against)."""
        from ..sources.cmorph import stage_cmorph

        a = np.datetime64(self.cfg.test_period[0][:10], "D") - np.timedelta64(20, "D")
        b = np.datetime64(self.cfg.test_period[1][:10], "D") + np.timedelta64(23, "D")
        for year in self.cfg.reference_years:
            later = year + int(str(b)[:4]) - int(str(a)[:4])            # a period may cross into the next year
            p = (hour_of(f"{year}{str(a)[4:]}T00"), hour_of(f"{later}{str(b)[4:]}T00"))
            stage_cmorph(self.root / "cmorph_reference", self.geo.targets, self.geo.mesh, period=p, workers=workers)

    def reader(self):
        from ..sources.nwp import NWPReader

        g = self.grids
        return NWPReader("gfs05", g["lat1"], g["lon1"], g["lat2"], g["lon2"], self.root / "nwp", workers=12)

    def stage_gfs(self, which: str = "all", parallel: int = 3) -> int:
        """The GFS runs Pluvium is driven by (stage B training runs, then the calibration and test runs)."""
        inits = {"train": self.train_inits, "eval": self.eval_inits,
                 "all": np.concatenate([self.train_inits, self.eval_inits])}[which]
        reader = self.reader()

        def one(init):
            try:
                reader.frames(int(init), offset=0)
                return True
            except Exception as error:
                log(f"GFS {_day(init)}: skipped ({type(error).__name__}: {error})")
                return False

        done = sum(parallel_map(one, inits, workers=parallel, progress="GFS runs"))
        log(f"operational: {done} of {len(inits)} GFS runs staged")
        return done

    def stage_baselines(self, systems=None) -> None:
        """Every system's calibration and test runs (ECMWF's three in turn, NOAA's alongside)."""
        systems = tuple(systems or self.cfg.systems)
        reader = BaselineReader(self.box, self.root / "baselines", workers=self.cfg.workers)

        def run(group):
            for system in group:
                got = sum(reader.read(system, int(i)) is not None for i in self.eval_inits)
                log(f"operational: {SYSTEMS[system][0]}: {got} of {len(self.eval_inits)} runs")

        ecmwf = [s for s in systems if s in ("ens", "ifs", "aifs")]
        noaa = [[s] for s in systems if s not in ("ens", "ifs", "aifs")]
        parallel_map(run, [ecmwf] + noaa, workers=1 + len(noaa))

    # ---------------------------------------------------------------------------------- Pluvium --
    def climatology(self):
        """The model's climatology input: the 12 months of CMORPH before the calibration period."""
        from ..dataset import Climatology, compute_climatology
        from ..sources.cmorph import CMORPHCache

        path = self.root / "climatology.npz"
        if not path.exists():
            b = hour_of(self.cfg.calibration_start)
            hours = np.arange(b - 365 * 24, b)
            cm = CMORPHCache(self.root / "cmorph")
            clim = compute_climatology(lambda h: cm.targets(h), self.geo.targets.lon, hours)
            clim.save(path)
        return Climatology.load(path)

    def builder(self, model, augment: bool):
        from ..dataset import SampleBuilder
        from ..sources.cmorph import CMORPHCache
        from ..sources.stations import StationCache
        from ..train import borrowed_climatology, nearest_target

        reader = self.reader()

        def driver(t0):
            frames = reader.frames(int(t0), offset=0)
            return frames["fields"], frames["upper"]

        clim = self.climatology()
        stations = st_clim = None
        if model.stations is not None:
            stations = StationCache(self.root / "stations")
            if len(stations) != model.stations.lat.size:
                raise RuntimeError(f"the model knows {model.stations.lat.size} airports, the cache {len(stations)}")
            st_clim = borrowed_climatology(clim, nearest_target(self.geo, stations.lat, stations.lon))
        holder = SimpleNamespace(**self.grids)
        return SampleBuilder(self.geo, holder, CMORPHCache(self.root / "cmorph"), stations, clim, st_clim,
                             ocean_targets=self.cfg.ocean_cells, max_land_targets=self.cfg.group_size,
                             seed=self.cfg.seed, driver=driver, augment=augment)

    def usable(self, inits) -> np.ndarray:
        """Runs whose GFS frames are staged and whose satellite truth covers the history and all 72 hours."""
        from ..sources.cmorph import CMORPHCache

        cm = CMORPHCache(self.root / "cmorph")
        ok = []
        for t0 in inits:
            stamp = str(hour_to_datetime64(int(t0)))[:13].replace(":", "")
            if not (self.root / "nwp" / f"gfs05_{stamp}.npz").exists():
                continue
            hours = int(t0) + np.arange(-3, 72)
            truth = cm.targets(hours, self.land[:: max(1, self.land.size // 40)])
            if np.isfinite(truth).mean() > 0.97:
                ok.append(int(t0))
        return np.array(ok, np.int64)

    def finetune(self):
        """Stage B: continue the stage A model on real GFS runs (resumes; returns the fine-tuned model)."""
        from ..train import TrainConfig, Trainer, load_model, save_checkpoint

        cfg = self.cfg
        final = self.root / "model"
        model, loss, meta, payload = load_model(cfg.checkpoint, self.geo)
        if (final / "model.pt").exists():
            model, loss, _, _ = load_model(final, self.geo)
            log("operational: the fine-tuned model is ready")
            return model
        if "ema" in payload:                                 # start from the released (averaged) weights
            model.load_state_dict({**model.state_dict(), **{
                k: v.to(model.state_dict()[k].dtype) for k, v in payload["ema"]["shadow"].items()}})
        v3 = model.config.version >= 3
        starts = self.usable(self.train_inits)
        if starts.size < 40:
            raise RuntimeError(f"only {starts.size} usable GFS runs for stage B")
        val = starts[-8:]
        train = starts[:-8]
        tcfg = TrainConfig(size=model.config.name, batch_size=cfg.batch_size, lr=cfg.lr, warmup_steps=50,
                           max_steps=cfg.finetune_steps, hours=cfg.finetune_hours, workers=1,
                           eval_every_minutes=1e9, save_every_minutes=4.0, eval_forecasts=8, log_every=50,
                           seed=cfg.seed, ema_decay=0.995 if v3 else 0.0, curriculum=v3)
        log(f"operational: stage B on {train.size} GFS runs ({_day(train[0])} to {_day(train[-1])}), "
            f"{cfg.finetune_steps} steps")
        trainer = Trainer(model, loss, self.builder(model, augment=v3), train, val, self.root / "stage_b", tcfg,
                          stage="B")
        result = trainer.run()
        info = {"stage": "B", "steps": int(result["step"]), "minutes": float(result["elapsed"]) / 60,
                "runs": int(train.size), "first": _day(train[0]), "last": _day(train[-1]),
                "parameters": int(model.num_parameters()), "model": model.config.name, "version": model.config.version}
        save_checkpoint(final, trainer.model, trainer.loss_fn, state=info)
        (self.root / "finetune.json").write_text(json.dumps(info, indent=1))
        return trainer.model

    def forecast(self, model, inits=None, out_dir=None) -> None:
        """Pluvium's daily-total samples for every calibration and test run (or ``inits``), cached per run."""
        import torch

        from ..chain import sample_paths
        from ..dataset import collate
        from ..products import component

        out_dir = Path(out_dir) if out_dir is not None else self.root / "forecasts"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.eval()
        builder = self.builder(model, augment=False)
        sea = builder.choose_targets(train=False)
        sea = sea[np.isin(sea, self.sea)]
        groups = [np.arange(k, self.land.size, math.ceil(self.land.size / self.cfg.group_size))
                  for k in range(math.ceil(self.land.size / self.cfg.group_size))]
        inits = self.eval_inits if inits is None else np.asarray(inits, np.int64)
        todo = [int(i) for i in inits if not (out_dir / f"{_day(i)}.npz").exists()]
        usable = set(self.usable(np.array(todo, np.int64)).tolist()) if todo else set()
        t_start = time.time()
        for n, t0 in enumerate(todo):
            if t0 not in usable:
                log(f"Pluvium {_day(t0)}: no GFS run or truth; skipped")
                continue
            S = self.cfg.samples
            samples = np.zeros((S, DAYS, self.land.size), np.float32)
            for g in groups:
                targets = np.concatenate([self.land[g], sea])
                ex = builder.example(t0, train=False, targets=targets, emulate=False)
                with torch.no_grad():
                    out = model(collate([ex]))
                comp = component(out, "", 0)
                paths = sample_paths(comp, comp["states"], S, seed=(t0 // 24) % (2**31))     # (S, P, T)
                if paths.shape[-1] < 24 * DAYS:
                    raise RuntimeError(f"the model forecasts {paths.shape[-1]} hours; {24 * DAYS} are needed")
                day_tot = paths[:, : g.size, : 24 * DAYS].reshape(S, g.size, DAYS, 24).sum(-1)
                samples[:, :, g] = day_tot.transpose(0, 2, 1)
            np.savez_compressed(out_dir / f"{_day(t0)}.npz", samples=samples.astype(np.float16))
            if n % 10 == 0:
                rate = (n + 1) / (time.time() - t_start)
                log(f"Pluvium forecasts: {n + 1} of {len(todo)} ({rate * 60:.1f} per minute)")

    # --------------------------------------------------------------------------------- evaluation --
    def truth(self, inits) -> np.ndarray:
        """Observed rain of days 1-3 ``(runs, 3, cells)``; NaN where an hour is missing."""
        from ..sources.cmorph import CMORPHCache

        cm = CMORPHCache(self.root / "cmorph")
        out = np.full((len(inits), DAYS, self.land.size), np.nan, np.float32)
        for i, t0 in enumerate(inits):
            r = cm.targets(int(t0) + np.arange(24 * DAYS), self.land)                  # (72, P) mm/h
            out[i] = r.reshape(DAYS, 24, -1).sum(1)                                    # NaN propagates
        return out

    def reference(self, inits) -> np.ndarray:
        """Climatological chance of each threshold ``(runs, 3, thresholds, cells)`` for the valid day: CMORPH of the
        reference years (and the training year), days within 15 days of the same date, cells within 25 km pooled."""
        from scipy import sparse
        from scipy.spatial import cKDTree

        from ..sources.cmorph import CMORPHCache

        years = list(self.cfg.reference_years) + [int(_day(self.train_inits[0])[:4])]
        caches = {y: CMORPHCache(self.root / ("cmorph" if y == years[-1] else "cmorph_reference")) for y in years}
        doys = [int(np.datetime64(_day(int(t) + 24 * d), "D").astype(object).timetuple().tm_yday)
                for t in inits for d in range(DAYS)]
        first, last = min(doys) - 17, max(doys) + 17
        totals = {}                                          # (year, day of year) -> daily rain at the cells
        for y in years:
            for d in range(first, last):
                start = hour_of(f"{y}-01-01T00") + (d - 1) * 24
                r = caches[y].targets(start + np.arange(24), self.land)
                if np.isfinite(r).all(0).mean() > 0.9:
                    totals[(y, d)] = r.sum(0)
        xyz = np.stack([np.cos(np.radians(self.lat)) * np.cos(np.radians(self.lon)),
                        np.cos(np.radians(self.lat)) * np.sin(np.radians(self.lon)), np.sin(np.radians(self.lat))], 1)
        pairs = cKDTree(xyz).query_pairs(25.0 / 6371.0, output_type="ndarray")
        n = self.land.size
        W = sparse.coo_matrix((np.ones(2 * len(pairs) + n), (np.r_[pairs[:, 0], pairs[:, 1], np.arange(n)],
                                                             np.r_[pairs[:, 1], pairs[:, 0], np.arange(n)])),
                              shape=(n, n)).tocsr()
        out = np.zeros((len(inits), DAYS, len(self.thr), n), np.float32)
        cache = {}
        for i, t0 in enumerate(inits):
            for d in range(DAYS):
                day = np.datetime64(_day(int(t0) + 24 * d), "D")
                doy = int(day.astype(object).timetuple().tm_yday)
                if doy not in cache:
                    keys = [k for k in totals if abs(k[1] - doy) <= 15]
                    stack = np.stack([totals[k] for k in keys])                          # (days, P)
                    ok = np.isfinite(stack)
                    hit = np.stack([((stack >= t) & ok).sum(0) for t in self.thr]).astype(np.float64)
                    num = np.asarray(W @ hit.T).T
                    den = np.asarray(W @ ok.sum(0).astype(np.float64))
                    cache[doy] = (num / np.maximum(den, 1)[None]).astype(np.float32)
                out[i, d] = cache[doy]
        log(f"operational: reference climatology from {len(totals):,} days of {len(years)} years")
        return out

    def _neighbourhood(self, lat_axis, lon_axis):
        """Sparse ``(cells, grid points)`` averaging over grid points within ``neighbourhood_deg`` of each cell."""
        from scipy import sparse

        la, lo = np.meshgrid(lat_axis, lon_axis, indexing="ij")
        r = self.cfg.neighbourhood_deg
        rows, cols = [], []
        flat_la, flat_lo = la.ravel(), lo.ravel()
        for i, (a, b) in enumerate(zip(self.lat, self.lon)):
            near = np.flatnonzero((flat_la - a) ** 2 + ((flat_lo - b) * np.cos(np.radians(a))) ** 2 <= r * r)
            rows += [i] * near.size
            cols += near.tolist()
        W = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.lat.size, flat_la.size)).tocsr()
        return sparse.diags(1.0 / np.maximum(np.asarray(W.sum(1)).ravel(), 1)) @ W

    def predictors(self, system: str) -> dict | None:
        """For every calibration and test run (NaN where a run is missing): ``point`` chances (the share of members,
        or of Pluvium's samples, at or above each threshold) and ``nbhd`` chances (the same over every grid point
        within ``neighbourhood_deg``), both ``(runs, 3, thresholds, cells)``; the ``amount`` (mean of the members,
        ``(runs, 3, cells)``); and the ``members``' daily rain at the cells ``(runs, members, 3, cells)``."""
        path = self.root / "predictors" / f"{system}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        sources = [self.root / "forecasts" / f"{_day(t)}.npz" if system == PLUVIUM else
                   self.root / "baselines" / f"{system}_{_day(t).replace('-', '')}00.npz" for t in self.eval_inits]
        present = np.array([f.exists() for f in sources])
        if path.exists():                                   # reuse only if built from exactly these runs
            z = np.load(path)
            if z["inits"].tolist() == self.eval_inits.tolist() and z["present"].tolist() == present.tolist():
                return {k: z[k] for k in z.files}
        R, K, P = len(self.eval_inits), len(self.thr), self.land.size
        point = np.full((R, DAYS, K, P), np.nan, np.float32)
        amount = np.full((R, DAYS, P), np.nan, np.float32)
        members = nbhd = None
        if system == PLUVIUM:
            for i, t0 in enumerate(self.eval_inits):
                f = self.root / "forecasts" / f"{_day(t0)}.npz"
                if not f.exists():
                    continue
                s = np.load(f)["samples"].astype(np.float32)                              # (S, 3, P)
                if members is None:
                    members = np.full((R, s.shape[0], DAYS, P), np.nan, np.float16)
                members[i] = s
                point[i] = np.stack([(s >= t).mean(0) for t in self.thr], 1)
                amount[i] = s.mean(0)
            if members is None:
                return None
            nbhd = self._smooth_cells(point)
        else:
            reader = BaselineReader(self.box, self.root / "baselines")
            W = None
            for i, t0 in enumerate(self.eval_inits):
                if not (reader.cache / f"{system}_{_day(t0).replace('-', '')}00.npz").exists():
                    continue
                f = reader.read(system, int(t0))
                M = f["values"].shape[0]
                if members is None:
                    members = np.full((R, M, DAYS, P), np.nan, np.float32)
                    nbhd = np.full((R, DAYS, K, P), np.nan, np.float32)
                    W = self._neighbourhood(f["lat"], f["lon"])
                at = daily(at_points(f, self.lat, self.lon))                                # (M, 3, P)
                g = f["values"].reshape(M, DAYS, -1)                                        # (M, 3, G) to 24/48/72 h
                g = np.maximum(np.diff(g, axis=1, prepend=np.zeros_like(g[:, :1])), 0.0)  # each day, per grid point
                members[i] = at
                amount[i] = at.mean(0)
                for k, t in enumerate(self.thr):
                    point[i, :, k] = (at >= t).mean(0)
                    nbhd[i, :, k] = np.asarray(W @ (g >= t).mean(0).T).T
            if members is None:
                return None
        got = {"point": point, "nbhd": nbhd, "amount": amount, "members": members, "inits": self.eval_inits,
               "present": present}
        np.savez_compressed(path, **got)
        return got

    @staticmethod
    def _x(pred: dict, kind: str, rows, d: int, k: int) -> np.ndarray:
        """A calibration predictor: a chance for ``point`` and ``nbhd``, the mean amount for ``amount``."""
        return pred["amount"][rows, d] if kind == "amount" else pred[kind][rows, d, k]

    def _smooth_cells(self, p: np.ndarray) -> np.ndarray:
        """Mean of a cell field over the cells within ``neighbourhood_deg``."""
        W = self._neighbourhood_cells()
        flat = p.reshape(-1, p.shape[-1])
        out = np.where(np.isfinite(flat).all(1, keepdims=True), np.asarray(W @ np.nan_to_num(flat).T).T, np.nan)
        return out.reshape(p.shape).astype(np.float32)

    def _neighbourhood_cells(self):
        from scipy import sparse
        from scipy.spatial import cKDTree

        r = self.cfg.neighbourhood_deg
        xy = np.stack([self.lat, self.lon * np.cos(np.radians(self.lat.mean()))], 1)
        pairs = cKDTree(xy).query_pairs(r, output_type="ndarray")
        n = self.lat.size
        W = sparse.coo_matrix((np.ones(2 * len(pairs) + n), (np.r_[pairs[:, 0], pairs[:, 1], np.arange(n)],
                                                             np.r_[pairs[:, 1], pairs[:, 0], np.arange(n)])),
                              shape=(n, n)).tocsr()
        return sparse.diags(1.0 / np.asarray(W.sum(1)).ravel()) @ W

    def evaluate(self, resamples: int = 2000) -> dict:
        """Calibrate every system causally, score all of them on the test runs, compare Pluvium with each."""
        cfg = self.cfg
        inits = self.eval_inits
        obs = self.truth(inits)                                                             # (R, 3, P)
        names = [PLUVIUM] + [s for s in cfg.systems]
        preds = {}
        for name in names:
            got = self.predictors(name)
            if got is not None:
                preds[name] = got
            else:
                log(f"operational: {LABELS[name]} has no runs; left out")
        available = [n for n in names if n in preds]
        test_idx = np.flatnonzero(np.isin(inits, self.test_inits))
        # a run takes part when every system has it and the truth is known
        have = np.ones(len(inits), bool)
        for n in available:
            have &= np.isfinite(preds[n]["point"]).all((1, 2, 3))
        test_idx = test_idx[have[test_idx]]

        def window(t0):
            return np.flatnonzero((inits >= t0 - 24 * (cfg.lag_days + cfg.window_days)) &
                                  (inits <= t0 - 24 * cfg.lag_days) & have)

        test_idx = np.array([i for i in test_idx if window(inits[i]).size >= 10], np.int64)
        log(f"operational: {test_idx.size} test runs with every system; calibrating on the {cfg.window_days} runs "
            f"before each (outcomes known {cfg.lag_days} days earlier)")
        K = len(self.thr)
        calibrated = {n: np.full((len(inits), DAYS, K, self.land.size), np.nan, np.float32)
                      for n in available + [BLEND_OPS, BLEND_ALL]}
        kinds = ("point", "nbhd", "amount")
        chosen = {n: dict.fromkeys(kinds, 0) for n in available}
        ops = [n for n in available if n != PLUVIUM]
        for i in test_idx:
            win = window(inits[i])
            for d in range(DAYS):
                y_win = obs[win, d]                                                         # (W, P)
                ok = np.isfinite(y_win)
                for k, t in enumerate(self.thr):
                    yy = (y_win[ok] >= t).astype(np.float64)
                    fitted = {}
                    for n in available:
                        best = None
                        for kind in kinds:
                            x = self._x(preds[n], kind, win, d, k)[ok]
                            fit = calibrator(x, yy, kind, cfg.calibration)
                            bs = float(np.mean((fit(x) - yy) ** 2))
                            if best is None or bs < best[0]:
                                best = (bs, kind, fit)
                        _, kind, fit = best
                        chosen[n][kind] += 1
                        calibrated[n][i, d, k] = fit(self._x(preds[n], kind, i, d, k))
                        fitted[n] = (kind, fit)
                    for blend, members in ((BLEND_OPS, ops), (BLEND_ALL, available)):
                        if len(members) < 2:
                            continue
                        feats_w = np.stack([_logit(fitted[n][1](self._x(preds[n], fitted[n][0], win, d, k)[ok]))
                                            for n in members], 1)
                        beta = logistic_fit(feats_w, yy)
                        feats_t = np.stack([_logit(calibrated[n][i, d, k]) for n in members], 1)
                        calibrated[blend][i, d, k] = logistic_apply(beta, feats_t)
        tuned = self.root / "finetune.json"
        results = {"config": asdict(cfg), "thresholds": list(self.thr), "test_runs": [_day(inits[i]) for i in test_idx],
                   "finetune": json.loads(tuned.read_text()) if tuned.exists() else {},
                   "cells": int(self.land.size), "systems": {}, "chosen_predictor": chosen, "versus": {}}
        y_test = obs[test_idx]                                                              # (T, 3, P)
        ok_test = np.isfinite(y_test)
        ref = self.reference(inits[test_idx])                                               # (T, 3, K, P)
        for n in available + ([BLEND_OPS, BLEND_ALL] if len(ops) > 1 else []):
            row = {"label": LABELS[n], "days": {}}
            for d in range(DAYS):
                day_row = {}
                for k, t in enumerate(self.thr):
                    y = (y_test[:, d] >= t)[ok_test[:, d]].astype(np.float64)
                    pr = ref[:, d, k][ok_test[:, d]]
                    bs_ref = float(np.mean((pr - y) ** 2))
                    entry = {"base_rate": float(y.mean()), "bs_reference": bs_ref}
                    if n in preds:
                        p_raw = preds[n]["point"][test_idx, d, k][ok_test[:, d]]
                        bs_raw = float(np.mean((p_raw - y) ** 2))
                        entry.update(bs_raw=bs_raw, bss_raw=1 - bs_raw / bs_ref, auc_raw=roc_area(p_raw, y))
                    pc = calibrated[n][test_idx, d, k][ok_test[:, d]]
                    if np.isfinite(pc).all():
                        bs = float(np.mean((pc - y) ** 2))
                        entry.update(bs=bs, bss=1 - bs / bs_ref, auc=roc_area(pc, y), reliability=reliability(pc, y))
                    day_row[f"{t:g}"] = entry
                if n in preds:
                    mem = preds[n]["members"][test_idx, :, d].astype(np.float32)            # (T, M, P)
                    yy = y_test[:, d]
                    mm = np.moveaxis(mem, 1, 0)[:, ok_test[:, d]]
                    day_row["crps_mm"] = float(crps_ensemble(mm, yy[ok_test[:, d]]).mean())
                    day_row["mean_mm"] = float(mm.mean())
                    day_row["observed_mean_mm"] = float(yy[ok_test[:, d]].mean())
                row["days"][str(d + 1)] = day_row
            results["systems"][n] = row
        # Pluvium against every other system: per-day Brier differences, moving-block bootstrap over days
        rivals = [n for n in available + [BLEND_OPS, BLEND_ALL] if n != PLUVIUM] if PLUVIUM in available else []
        for other in rivals:
            comp = {}
            for d in range(DAYS):
                for k, t in enumerate(self.thr):
                    yv = (y_test[:, d] >= t).astype(np.float64)
                    a = (calibrated[PLUVIUM][test_idx, d, k] - yv) ** 2
                    b = (calibrated[other][test_idx, d, k] - yv) ** 2
                    m = ok_test[:, d]
                    per_day = np.array([(a[j][m[j]] - b[j][m[j]]).mean() if m[j].any() else np.nan
                                        for j in range(len(test_idx))])
                    per_day = per_day[np.isfinite(per_day)]
                    mean, lo, hi, p_better = block_bootstrap(per_day, resamples, seed=cfg.seed + 17 * d + k)
                    comp[f"day{d + 1}_{t:g}mm"] = {"delta_bs": mean, "ci95": [lo, hi], "share_pluvium_better": p_better}
            results["versus"][other] = comp
        (self.root / f"results_{self.tag}.json").write_text(json.dumps(results, indent=1, default=float))
        return results

    @property
    def tag(self) -> str:
        """``2025-05_logistic``: the first test month and the calibration method."""
        return f"{self.cfg.test_period[0][:7]}_{self.cfg.calibration}"


# ----------------------------------------------------------------------------------------- report --

METHODS = {"isotonic": "isotonic regression", "logistic": "logistic regression"}


def markdown(results: dict) -> str:
    """The benchmark as a Markdown report."""
    cfg = results["config"]
    sys = results["systems"]
    order = [n for n in [PLUVIUM, "ens", "gefs", "ifs", "aifs", "graphcast", "gfs", BLEND_OPS, BLEND_ALL] if n in sys]
    thr = [f"{float(t):g}" for t in results.get("thresholds", cfg["thresholds"])]
    lines = [
        "# Pluvium against the operational systems",
        "",
        f"Test: {len(results['test_runs'])} daily 00 UTC runs, {results['test_runs'][0]} to {results['test_runs'][-1]}, "
        f"at {results['cells']:,} land cells of 8 km (Jamaica, the Cayman Islands, eastern Cuba and western Haiti). "
        "Truth: CMORPH. Every forecast is a real, archived operational run; none uses a reanalysis.",
        "",
        "Brier skill score against the climatology of the same weeks (higher is better; 0 is climatology). "
        f"Each system is calibrated the same causal way ({METHODS[cfg.get('calibration', 'isotonic')]} on the "
        f"{cfg['window_days']} previous runs). The raw score is in brackets.",
        "",
    ]
    for t in thr:
        lines += [f"## 24-hour rain of at least {t} mm", "", "| System | Day 1 | Day 2 | Day 3 |", "|---|---|---|---|"]
        for n in order:
            cells = []
            for d in ("1", "2", "3"):
                e = sys[n]["days"][d][t]
                txt = f"{e['bss']:+.3f}" if "bss" in e else "-"
                if "bss_raw" in e:
                    txt += f" ({e['bss_raw']:+.3f})"
                cells.append(txt)
            label = sys[n]["label"] + (" **(Pluvium)**" if n == PLUVIUM else "")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        base = sys[order[0]]["days"]["1"][t]["base_rate"]
        lines += ["", f"Share of cell-days with at least {t} mm on day 1: {base:.1%}.", ""]
    lines += ["## Pluvium against each system", "",
              "Difference in Brier score, Pluvium minus the other system, after calibration (negative: Pluvium "
              "better), with a 95% moving-block bootstrap interval over days.", ""]
    head = "| Against | " + " | ".join(f"Day {d}, {t} mm" for d in (1, 2, 3) for t in thr[:3]) + " |"
    lines += [head, "|---|" + "---|" * (3 * len(thr[:3]))]
    for other, comp in results["versus"].items():
        cells = []
        for d in (1, 2, 3):
            for t in thr[:3]:
                c = comp[f"day{d}_{t}mm"]
                mark = "**" if c["ci95"][1] < 0 else ""
                cells.append(f"{mark}{c['delta_bs']:+.4f}{mark} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
        lines.append(f"| {LABELS.get(other, other)} | " + " | ".join(cells) + " |")
    lines += ["", "Bold: Pluvium better with 95% confidence.", "",
              "## CRPS of the 24-hour totals (mm, lower is better; raw forecasts)", "",
              "| System | Day 1 | Day 2 | Day 3 |", "|---|---|---|---|"]
    for n in order:
        if "crps_mm" in sys[n]["days"]["1"]:
            lines.append(f"| {sys[n]['label']} | " + " | ".join(f"{sys[n]['days'][d]['crps_mm']:.2f}"
                                                                for d in ("1", "2", "3")) + " |")
    lines += ["", "## ROC area after calibration (day 1)", "", "| System | " + " | ".join(f"{t} mm" for t in thr) + " |",
              "|---|" + "---|" * len(thr)]
    for n in order:
        lines.append(f"| {sys[n]['label']} | " + " | ".join(
            f"{sys[n]['days']['1'][t].get('auc', float('nan')):.3f}" for t in thr) + " |")
    ft = results.get("finetune", {})
    lines += ["", "## How it was run", "",
              f"* **Pluvium**: {ft.get('model', 'pluvium-small')} (version {ft.get('version', 3)}, "
              f"{ft.get('parameters', 0):,} parameters). Stage A: 32 minutes on ERA5 for June to September 2021 "
              f"(the version benchmark's run). Stage B: {ft.get('steps', 0):,} steps ({ft.get('minutes', 0):.0f} minutes "
              f"on a 4-core CPU) on {ft.get('runs', 0)} real 00 UTC GFS runs from {ft.get('first', '?')} to "
              f"{ft.get('last', '?')}, all ending before the first calibration day. At issue time it reads that "
              "morning's GFS run (0.5 degrees), the satellite rain of the previous 3 hours, the airports' reports of "
              "the previous 3 hours, terrain, and a CMORPH climatology of the 12 months before calibration starts.",
              "* **Baselines**: each system's own 00 UTC run, read from its producer's archive, interpolated "
              "bilinearly to the cell centres. Ensembles give the share of members at or above each threshold.",
              f"* **Calibration** (the same for all, Pluvium included): for each test run, a "
              f"{METHODS[cfg.get('calibration', 'isotonic')]} per day and threshold fitted on the "
              f"{cfg['window_days']} previous runs whose outcomes were known "
              f"{cfg['lag_days']} days before. Each system uses whichever predictor fits its window best: its point "
              f"chance, its chance over everything within {cfg['neighbourhood_deg']:g} degrees, or its mean amount.",
              "* **Blends**: logistic regression on the calibrated chances, fitted on the same window.",
              "* **Truth**: CMORPH 8 km rain (satellite estimates adjusted to rain gauges) summed over 00-24 UTC.",
              f"* **Reference**: the CMORPH climatology of {min(cfg['reference_years'])}-"
              f"{int(str(ft.get('first', '2024'))[:4])}: days within 15 days of the same date, cells within 25 km.",
              "* **Intervals**: moving-block bootstrap over test days (blocks of 5 days, 2,000 resamples).",
              "", "## Limits", "",
              "* One island and two months (the early rainy season). Skill elsewhere or in other seasons is untested.",
              "* Truth is CMORPH, which Pluvium was also trained on. Calibration on the same truth puts every system "
              "on the same footing, but the raw scores favour Pluvium.",
              "* Pluvium's \"rain now\" input comes from the CMORPH record, which is adjusted to gauges after the fact. "
              "Live, Pluvium reads GOES-East and radar instead, which are cruder.",
              "* Pluvium ran at its small size, trained for about two hours on a CPU in total. Pluvium-500 is 60 "
              "times larger and is meant to train for a day on a GPU.",
              ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="pluvium operational", description="Pluvium against the operational forecasting systems.")
    p.add_argument("step", choices=["truth", "reference", "gfs", "baselines", "finetune", "forecast", "evaluate", "all"])
    p.add_argument("--root", required=True)
    p.add_argument("--geography", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stations", default=None)
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--report", default=None, help="also write the Markdown report here")
    p.add_argument("--which", default="all", help="gfs: train, eval or all")
    p.add_argument("--test-period", default="2025-05-01T00,2025-07-01T00", help="first and end issue time, UTC")
    p.add_argument("--calibration", default="logistic", choices=["logistic", "isotonic"])
    a = p.parse_args(argv)
    if a.threads:
        import torch

        torch.set_num_threads(a.threads)
    op = Operational(OperationalConfig(root=a.root, geography=a.geography, checkpoint=a.checkpoint,
                                       stations=a.stations, finetune_steps=a.steps,
                                       test_period=tuple(a.test_period.split(",")), calibration=a.calibration))
    steps = (["truth", "gfs", "baselines", "reference", "finetune", "forecast", "evaluate"] if a.step == "all"
             else [a.step])
    model = None
    for s in steps:
        if s == "truth":
            op.stage_truth()
        elif s == "reference":
            op.stage_reference()
        elif s == "gfs":
            op.stage_gfs(a.which)
        elif s == "baselines":
            op.stage_baselines()
        elif s == "finetune":
            model = op.finetune()
        elif s == "forecast":
            op.forecast(model or op.finetune())
        elif s == "evaluate":
            results = op.evaluate()
            text = markdown(results)
            (op.root / f"OPERATIONAL_BENCHMARK_{op.tag}.md").write_text(text)
            if a.report:
                Path(a.report).write_text(text)
            print(text)


if __name__ == "__main__":
    main()
