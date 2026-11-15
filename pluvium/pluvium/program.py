# Copyright 2026 Nathan. Apache-2.0.
"""
Pluvium-500 from nothing on one GPU, in stages that each finish or resume:

====================  =====================================================================================
geography             Caribbean terrain, coastlines and parishes (geoBoundaries + Terrarium via NatureV1), the
                      model's mesh, the 8 km target cells, the orographic tiles, and every town (GeoNames)
data                  ERA5 (1 degree hourly, 1.5 degree aloft), CMORPH 8 km rain, airport reports -- for the
                      training years and one held-out year, staged in parallel
climatology           the chance of rain by month and local hour at every target cell (training years only)
train A               perfect prognosis on ERA5 with forecast-error emulation (most of the budget)
train B               model output statistics on real GFS runs (when the budget allows; fetched in the
                      background while stage A trains)
calibrate + verify    isotonic calibration on half of the held-out data, scores on the other half
publish               weights, calibration, geography, towns and an honest model card to the Hub
forecast              the live forecast for every town, if the driving models are reachable
====================  =====================================================================================

Everything expensive is mirrored to the Hugging Face Hub as soon as it exists (``HF_REPO``, ``HF_TOKEN``), so a
second session -- or a machine that was stopped -- pulls it back in minutes and continues where the last one ended.
The time budget (``GPU_HOURS``) is wall clock from the program's start: training gets whatever the other stages leave.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .util import hour_of, hour_to_iso, log, year_hours

DEFAULT_REPO = "CorxLabs/Pluvium-500"


@dataclass
class ProgramConfig:
    gpu_hours: float = 24.0
    root: str = "~/pluvium"
    hub_repo: str | None = DEFAULT_REPO
    hf_token: str | None = None
    private: bool = False
    size: str = "pluvium-500"
    train_years: tuple | None = None     # None: chosen from the budget
    holdout_year: int = 2021             # the last whole year in the hourly ERA5 archive (checked when a run starts)
    stage_b: bool | None = None          # None: on when the budget allows
    stage_b_period: tuple = ("2025-01-01T00", "2025-07-01T00")
    stage_b_cycles: int = 240
    batch_size: int = 0                  # 0: measure
    vram: str = "fill"                   # "fill": size each step to the GPU's memory; "fastest": the fastest batch
    vram_share: float = 0.95             # how much of the GPU's memory a training step may need when filling
    workers: int = -1                    # -1: every core but two
    checkpointing: bool = True
    lr: float = 3e-4
    forecast: bool = True
    report_countries: tuple | None = None
    countries: tuple | None = None       # ISO codes of the area to serve (None: the Caribbean)
    box: tuple | None = None             # (south, north, west, east) instead of the countries' own extent
    overrides: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, **kw) -> ProgramConfig:
        env = os.environ
        cfg = cls(**kw)
        if env.get("GPU_HOURS"):
            cfg.gpu_hours = float(env["GPU_HOURS"])
        if env.get("PLUVIUM_ROOT"):
            cfg.root = env["PLUVIUM_ROOT"]
        if "HF_REPO" in env:
            cfg.hub_repo = env["HF_REPO"] or None
        cfg.hf_token = cfg.hf_token or env.get("HF_TOKEN") or None
        if env.get("PLUVIUM_SIZE"):
            cfg.size = env["PLUVIUM_SIZE"]
        if env.get("PLUVIUM_COUNTRIES"):
            cfg.countries = tuple(c.strip() for c in env["PLUVIUM_COUNTRIES"].split(",") if c.strip())
        if env.get("PLUVIUM_BOX"):
            cfg.box = tuple(float(x) for x in env["PLUVIUM_BOX"].split(","))
        if env.get("PLUVIUM_VRAM"):
            cfg.vram = env["PLUVIUM_VRAM"]
        if env.get("PLUVIUM_VRAM_SHARE"):
            cfg.vram_share = float(env["PLUVIUM_VRAM_SHARE"])
        if env.get("PLUVIUM_BATCH"):
            cfg.batch_size = int(env["PLUVIUM_BATCH"])
        return cfg


def choose_years(gpu_hours: float, holdout: int = 2021) -> tuple:
    """More training years when the budget can use them (each year is ~8,760 issue times)."""
    n = 6 if gpu_hours >= 20 else 4 if gpu_hours >= 10 else 2
    return tuple(range(holdout - n, holdout))


class Program:
    def __init__(self, cfg: ProgramConfig) -> None:
        self.cfg = cfg
        self.started = time.time()
        self.deadline = self.started + cfg.gpu_hours * 3600.0
        self.root = Path(cfg.root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "program.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.years = tuple(cfg.train_years or choose_years(cfg.gpu_hours, cfg.holdout_year))
        self.stage_b = cfg.stage_b if cfg.stage_b is not None else cfg.gpu_hours >= 12
        from naturev1.mirror import HubMirror

        self.mirror = HubMirror(cfg.hub_repo, cfg.hf_token, private=cfg.private) if cfg.hub_repo else None
        self._prefetch = None

    # ---------------------------------------------------------------------------------- helpers --
    def left(self) -> float:
        """Hours left in the budget."""
        return (self.deadline - time.time()) / 3600.0

    def mark(self, stage: str, **info) -> None:
        self.state[stage] = {"done": time.strftime("%Y-%m-%d %H:%M"), **info}
        self.state_path.write_text(json.dumps(self.state, indent=1))

    def check_years(self) -> None:
        """Make sure the years asked for are in the hourly ERA5 archive, whose first and last year are read from the
        archive itself. Before 0.4.3 the program held out 2022 by default, a year the archive does not have, and
        stopped while staging it, after every training year had been downloaded."""
        from .sources.era5 import hourly_years

        first, last = hourly_years()
        cfg = self.cfg
        if not first <= cfg.holdout_year <= last:
            log(f"the hourly ERA5 archive holds {first} to {last}: holding out {last}, not {cfg.holdout_year}")
            cfg.holdout_year = last
            if cfg.train_years is None:
                self.years = choose_years(cfg.gpu_hours, last)
        kept = tuple(y for y in self.years if first <= y <= last and y != cfg.holdout_year)
        if kept != tuple(self.years):
            log(f"training years {sorted(set(self.years) - set(kept))} are held out or not in the archive "
                f"({first} to {last}); training on {list(kept)}")
            self.years = kept
        if not self.years:
            raise ValueError(f"no training year is left: the hourly ERA5 archive holds {first} to {last} and "
                             f"{cfg.holdout_year} is held out. Use --years with years before {cfg.holdout_year}.")

    def pull(self, local: Path, repo_path: str, what: str | None = None) -> bool:
        return bool(self.mirror and self.mirror.pull(local, repo_path, what))

    def push(self, local: Path, repo_path: str, what: str | None = None) -> None:
        if self.mirror is not None:
            self.mirror.push(local, repo_path, what=what)

    # ---------------------------------------------------------------------------------- geography --
    def area(self):
        """The countries this run serves (:class:`pluvium.area.Area`), or None for the Caribbean."""
        if not self.cfg.countries and not self.cfg.box:
            return None
        from .area import Area

        if self.cfg.countries:
            return Area.for_countries(self.cfg.countries, box=self.cfg.box, cache=self.root / "cache" / "areas")
        from .domain import CARIBBEAN_ISO2

        raise ValueError(f"--box needs --country too (which countries' towns and borders to use, e.g. {CARIBBEAN_ISO2[0]})")

    def geography(self):
        from .domain import Gazetteer, Geography, PointSet, build_gazetteer

        d = self.root / "geography"
        files = ("terrain.npz", "mesh.npz", "targets.npz", "tiles.npz", "geography.json")
        for name in files:
            self.pull(d / name, f"geography/{name}")
        if not all((d / n).exists() for n in files):
            area = self.area()
            log(f"geography: building {'the Caribbean' if area is None else area.name} (terrain, coasts, parishes) ...")
            geo = Geography.build(cache=self.root / "cache" / "geography", area=area)
            geo.save(d)
            for name in files:
                self.push(d / name, f"geography/{name}")
        geo = Geography.load(d)
        towns_path = self.root / "towns.npz"
        self.pull(towns_path, "towns.npz", "town list")
        if not towns_path.exists():
            log("towns: reading every populated place from GeoNames ...")
            area = geo.area
            gaz = (build_gazetteer(geo.terrain, cache=self.root / "cache" / "geonames") if area is None else
                   build_gazetteer(geo.terrain, countries=area.iso2, cache=self.root / "cache" / "geonames",
                                   box=area.target_box))
            gaz.points.save(towns_path)
            self.push(towns_path, "towns.npz")
        towns = Gazetteer(PointSet.load(towns_path))
        log(f"geography: {len(geo.mesh):,} mesh points, {len(geo.targets):,} target cells, {len(geo.tiles)} "
            f"orographic tiles, {len(towns.points):,} places")
        self.mark("geography", mesh=len(geo.mesh), targets=len(geo.targets), towns=len(towns.points))
        return geo, towns

    # --------------------------------------------------------------------------------------- data --
    def data(self, geo, years) -> tuple:
        """Stage ERA5, CMORPH and airport reports for ``years`` (pulled from the Hub where it holds them)."""
        from .features import HOURLY_SHORT, LEVELS, UPPER_SHORT

        d = self.root / "data"
        era5_dir, cm_dir, st_dir = d / "era5", d / "cmorph", d / "stations2"
        # pull what the Hub holds
        self.pull(era5_dir / "era5.json", "data/era5/era5.json", "ERA5 metadata")
        for y in years:
            for s in HOURLY_SHORT:
                self.pull(era5_dir / "era5_1h" / f"{s}_{y}.npy", f"data/era5/era5_1h/{s}_{y}.npy")
            for s in UPPER_SHORT:
                for p in LEVELS:
                    self.pull(era5_dir / "era5_6h" / f"{s}{p}_{y}.npy", f"data/era5/era5_6h/{s}{p}_{y}.npy")
            for k in ("targets", "mesh"):
                self.pull(cm_dir / "cmorph" / f"{k}_{y}.npy", f"data/cmorph/{k}_{y}.npy")
            self.pull(st_dir / "stations" / f"{y}.npy", f"data/stations2/{y}.npy")
        self.pull(cm_dir / "cmorph" / "cmorph.json", "data/cmorph/cmorph.json", "CMORPH metadata")
        self.pull(st_dir / "stations" / "stations.json", "data/stations2/stations.json", "airport list")
        _merge_years(era5_dir / "era5.json", years, cm_dir / "cmorph" / "cmorph.json")
        # stage the rest, the three sources at once (all network-bound)
        began = time.time()
        jobs = {
            "era5": (_stage_era5_job, (str(era5_dir), list(years), list(geo.mesh_box))),
            "cmorph": (_stage_cmorph_job, (str(cm_dir), str(self.root / "geography"), list(years))),
            "stations": (_stage_stations_job, (str(st_dir), list(years), list(geo.target_box))),
        }
        ctx = multiprocessing.get_context("spawn")
        procs = {name: ctx.Process(target=fn, args=args, name=f"stage-{name}") for name, (fn, args) in jobs.items()}
        for p in procs.values():
            p.start()
        for name, p in procs.items():
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"staging {name} failed (exit code {p.exitcode}); see the lines above")
        log(f"data: {len(years)} years staged in {(time.time() - began) / 60:.1f} min")
        # mirror what was staged
        for y in years:
            for s in HOURLY_SHORT:
                self.push(era5_dir / "era5_1h" / f"{s}_{y}.npy", f"data/era5/era5_1h/{s}_{y}.npy")
            for s in UPPER_SHORT:
                for p in LEVELS:
                    self.push(era5_dir / "era5_6h" / f"{s}{p}_{y}.npy", f"data/era5/era5_6h/{s}{p}_{y}.npy")
            for k in ("targets", "mesh"):
                self.push(cm_dir / "cmorph" / f"{k}_{y}.npy", f"data/cmorph/{k}_{y}.npy")
            self.push(st_dir / "stations" / f"{y}.npy", f"data/stations2/{y}.npy")
        self.push(era5_dir / "era5.json", "data/era5/era5.json")
        self.push(cm_dir / "cmorph" / "cmorph.json", "data/cmorph/cmorph.json")
        self.push(st_dir / "stations" / "stations.json", "data/stations2/stations.json")
        from .sources.cmorph import CMORPHCache
        from .sources.era5 import ERA5Cache
        from .sources.stations import StationCache

        self.mark("data", years=list(years))
        return ERA5Cache(era5_dir), CMORPHCache(cm_dir), StationCache(st_dir)

    # -------------------------------------------------------------------------------- climatology --
    def climatology(self, geo, cmorph):
        from .dataset import Climatology, compute_climatology

        path = self.root / "climatology.npz"
        self.pull(path, "climatology.npz", "climatology")
        if not path.exists():
            hours = np.concatenate([np.arange(*year_hours(y)) for y in self.years])
            log(f"climatology: {hours.size:,} hours at {len(geo.targets):,} cells ...")
            clim = compute_climatology(lambda h: cmorph.targets(h), geo.targets.lon, hours)
            clim.save(path)
            self.push(path, "climatology.npz")
        clim = Climatology.load(path)
        self.mark("climatology", wet_share=float(clim.wet.mean()))
        return clim

    # ------------------------------------------------------------------------------------ stage B --
    def stage_b_inits(self) -> list[int]:
        a, b = hour_of(self.cfg.stage_b_period[0]), hour_of(self.cfg.stage_b_period[1])
        inits = np.arange(a - a % 12, b - 72 - 6, 12)
        cap = min(self.cfg.stage_b_cycles, int(10 * self.cfg.gpu_hours))
        if inits.size > cap:
            inits = inits[np.linspace(0, inits.size - 1, cap).astype(int)]
        return [int(x) for x in inits]

    def start_prefetch(self, grids: dict) -> None:
        """Fetch the GFS runs (and CMORPH, airports) for stage B in a background process while stage A trains."""
        if not self.stage_b or self._prefetch is not None:
            return
        ctx = multiprocessing.get_context("spawn")
        self._prefetch = ctx.Process(target=_prefetch_job, name="stage-b-prefetch",
                                     args=(str(self.root), grids, self.stage_b_inits(), list(self.cfg.stage_b_period)))
        self._prefetch.start()
        log(f"stage B: fetching {len(self.stage_b_inits())} GFS runs in the background")

    # -------------------------------------------------------------------------------------- train --
    def train_cfg(self, hours: float, **kw):
        from .train import TrainConfig

        v3 = not self.cfg.size.endswith(("-v1", "-v2"))
        kw = {"ema_decay": 0.999 if v3 else 0.0, "curriculum": v3, **kw}
        if self.cfg.vram not in ("fill", "fastest"):
            raise ValueError(f"vram must be 'fill' or 'fastest', not {self.cfg.vram!r}")
        if not 0.3 <= self.cfg.vram_share <= 0.99:
            raise ValueError(f"vram_share must be between 0.3 and 0.99, not {self.cfg.vram_share}: the last per cent "
                             "of a GPU's memory cannot be promised to a training step")
        return TrainConfig(size=self.cfg.size, batch_size=self.cfg.batch_size, workers=self.cfg.workers,
                           checkpointing=self.cfg.checkpointing, lr=self.cfg.lr, hours=hours,
                           overrides=self.cfg.overrides, memory=self.cfg.vram, memory_share=self.cfg.vram_share, **kw)

    def run(self) -> dict:
        import torch

        from .dataset import SampleBuilder
        from .losses import PluviumLoss
        from .train import (
            Trainer,
            borrowed_climatology,
            build_model,
            fit_calibration,
            grids_of,
            hindcast,
            load_model,
            nearest_target,
            save_checkpoint,
            station_points,
        )
        from .verify import format_report

        cfg = self.cfg
        self.check_years()
        log(f"Pluvium program: {cfg.size}, budget {cfg.gpu_hours:.1f} h, training years {self.years[0]}-"
            f"{self.years[-1]}, held out {cfg.holdout_year}, stage B {'on' if self.stage_b else 'off'}; "
            + (self.mirror.describe() if self.mirror else "mirror: off"))
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            how = (f"each training step is sized to {cfg.vram_share:.0%} of it" if cfg.vram == "fill" and not cfg.batch_size
                   else f"batch {cfg.batch_size} as given" if cfg.batch_size else "the fastest batch is kept")
            log(f"GPU: {props.name}, {props.total_memory / 2**30:.0f} GB ({how}); {os.cpu_count()} CPU cores")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        geo, towns = self.geography()
        era5, cmorph, stations = self.data(geo, self.years + (cfg.holdout_year,))
        clim = self.climatology(geo, cmorph)
        grids = grids_of(era5)
        self.start_prefetch(grids)
        st_points = station_points(geo, stations)
        st_clim = borrowed_climatology(clim, nearest_target(geo, stations.lat, stations.lon)) if len(stations) else None
        builder = SampleBuilder(geo, era5, cmorph, stations if len(stations) else None, clim, st_clim,
                                augment=not cfg.size.endswith(("-v1", "-v2")))
        train_hours = np.concatenate([np.arange(*year_hours(y)) for y in self.years])
        hold = np.arange(*year_hours(cfg.holdout_year))
        log("issue times: checking which have complete data ...")
        train_starts = builder.valid_starts(train_hours)
        hold_starts = builder.valid_starts(hold[::6])
        months = (hold_starts // 24 // 30) % 2
        cal_starts, test_starts = hold_starts[months == 0][::4], hold_starts[months == 1][::4]
        val_starts = hold_starts[:: max(1, hold_starts.size // 32)]
        log(f"issue times: {train_starts.size:,} for training, {cal_starts.size} for calibration, "
            f"{test_starts.size} for scoring")
        model = build_model(geo, grids, cfg.size, st_points, **cfg.overrides)
        loss = PluviumLoss(states=model.config.states)
        reserve = 0.9 + 0.1 * cfg.gpu_hours / 24          # calibration, scoring, publishing, forecast
        train_hours_total = max(0.25, self.left() - reserve)
        share_a = train_hours_total * (0.85 if self.stage_b else 1.0)
        trainer = Trainer(model, loss, builder, train_starts, val_starts, self.root / "stage_a",
                          self.train_cfg(share_a), self.mirror, "checkpoint/stage_a", stage="A")
        if "train_a" in self.state and not self.state.get("extend_a"):
            log(f"stage A: already trained ({self.state['train_a'].get('hours', 0):.2f} h); using its best weights")
        else:
            # a session that finds stage A unfinished adds its own time to what was already trained
            trainer.cfg.hours = trainer.elapsed / 3600 + share_a
            result_a = trainer.run()
            self.mark("train_a", steps=result_a["step"], hours=result_a["elapsed"] / 3600)
        best = self.root / "stage_a" / "best"
        if not (best / "model.pt").exists():
            self.pull(best / "model.pt", "checkpoint/stage_a/best/model.pt", "best stage A weights")
            self.pull(best / "pluvium.json", "checkpoint/stage_a/best/pluvium.json", "best stage A metadata")
        final_dir = best if (best / "model.pt").exists() else self.root / "stage_a"
        model_device = trainer.device
        model, loss, _, _ = load_model(final_dir, geo, model_device)
        batch_a = trainer.cfg.batch_size
        del trainer
        live_builder, live_starts = None, None
        if self.stage_b:
            try:
                got = self.finish_stage_b(model, loss, geo, grids, clim, stations, st_points, st_clim, batch_a)
            except Exception as error:                 # stage B is an extra: stage A's model must still be released
                log(f"stage B stopped ({type(error).__name__}: {error}); the model trained in stage A is kept")
                model, loss, _, _ = load_model(final_dir, geo, model_device)
                got = None
            if got is not None:
                model, loss, live_builder, live_starts = got
        # calibration and scores
        log("calibration: hindcasts on the held-out year (with forecast-like errors) ...")
        cal_board, cal_data = hindcast(model, builder, cal_starts, batch_size=2, emulate=True)
        calibrator = fit_calibration(cal_data)
        scores = {}
        log("scores: held-out year, driving fields = reanalysis (an upper bound on live skill)")
        board, _ = hindcast(model, builder, test_starts, batch_size=2, emulate=False, calibrator=calibrator)
        scores["reanalysis_driven"] = board.report()
        log(format_report(scores["reanalysis_driven"]))
        log("scores: held-out year, reanalysis with emulated forecast errors (closer to live)")
        board, _ = hindcast(model, builder, test_starts, batch_size=2, emulate=True, calibrator=calibrator)
        scores["emulated_errors"] = board.report()
        log(format_report(scores["emulated_errors"]))
        if live_builder is not None and live_starts is not None and live_starts.size:
            half = live_starts.size // 2
            _, live_cal = hindcast(model, live_builder, live_starts[:half], batch_size=2)
            calibrator = fit_calibration(live_cal)
            board, _ = hindcast(model, live_builder, live_starts[half:], batch_size=2, calibrator=calibrator)
            scores["gfs_driven"] = board.report()
            log("scores: held-out GFS runs (the live situation)")
            log(format_report(scores["gfs_driven"]))
        # package
        out = self.root / "release"
        extras = self.release_extras(builder, train_starts, cal_data, cmorph, geo, towns)
        save_checkpoint(out, model, loss, None, {"stage": "release"}, calibrator,
                        extra={"scores": scores, "data": {"train_years": list(self.years),
                                                          "holdout_year": cfg.holdout_year,
                                                          "stage_b": bool(live_builder is not None)}, **extras})
        for name in ("terrain.npz", "mesh.npz", "targets.npz", "tiles.npz", "geography.json"):
            (out / "geography").mkdir(exist_ok=True)
            shutil.copy2(self.root / "geography" / name, out / "geography" / name)
        shutil.copy2(self.root / "climatology.npz", out / "climatology.npz")
        shutil.copy2(self.root / "towns.npz", out / "towns.npz")
        from .card import model_card

        (out / "README.md").write_text(model_card(cfg.hub_repo or "Pluvium-500", model, scores, self.years,
                                                  cfg.holdout_year, live_builder is not None), encoding="utf-8")
        self.mark("release", scores={k: {m: v.get(m) for m in ("bss_vs_clim", "brier_p", "csi_50")}
                                     for k, v in scores.items()})
        if self.mirror is not None and self.mirror.can_push:
            self.publish(out)
        if cfg.forecast:
            try:
                from .forecast import Forecaster

                fc = Forecaster(out, geography=geo, towns=towns, climatology=clim, cache=self.root / "cache")
                result = fc.run(out_dir=self.root / "forecast", countries=cfg.report_countries)
                log((self.root / "forecast" / "report.txt").read_text()[:4000])
                self.mark("forecast", t0=result["info"]["t0"])
            except Exception as error:
                log(f"live forecast not made ({type(error).__name__}: {error})")
        if self.mirror is not None:
            self.mirror.wait()
        log(f"done in {(time.time() - self.started) / 3600:.2f} h; the release is in {out}")
        return {"scores": scores, "release": str(out)}

    def release_extras(self, builder, train_starts, cal_data: dict, cmorph, geo, towns) -> dict:
        """What the live forecast needs besides the weights: the training distribution of the inputs (to notice
        unusual situations), conformal margins for totals, extreme-value tails per country (return periods), and
        the fallback's coefficients."""
        from .features import HOURLY_CHANNELS
        from .guard.drift import InputStats
        from .guard.fallback import fit_fallback
        from .prob.extremes import RainfallTails

        rng = np.random.default_rng(0)
        extras = {}
        try:
            pick = rng.choice(train_starts, min(120, train_starts.size), replace=False)
            sample = np.stack([builder.example(int(t), False, np.arange(1), emulate=False)["hourly"] for t in pick])
            extras["input_stats"] = InputStats(list(HOURLY_CHANNELS)).fit(sample.reshape((-1,) + sample.shape[2:])).to_json()
        except Exception as error:
            log(f"input statistics skipped ({type(error).__name__}: {error})")
        try:
            from .forecast import nearest_target_town

            country = towns.points.extra["country"].astype(str)[nearest_target_town(towns, geo.targets.lat, geo.targets.lon)]
            cols = np.arange(0, len(geo.targets), 4)
            hours = np.concatenate([np.arange(*year_hours(y)) for y in self.years])[::1]
            record = np.concatenate([cmorph.targets(hours[a: a + 2000], cols) for a in range(0, hours.size, 2000)])
            tails = RainfallTails().fit(record, country[cols], years=hours.size / 8766.0)
            extras["tails"] = {f"{g}|{k}": [t.u, t.xi, t.sigma, t.rate] for (g, k), t in tails.tails.items()}
        except Exception as error:
            log(f"rain tails skipped ({type(error).__name__}: {error})")
        try:
            obs = np.where(cal_data["known"], cal_data["wet"], np.nan).ravel()
            pick = rng.choice(obs.size, min(500_000, obs.size), replace=False)
            extras["fallback"] = fit_fallback(cal_data["driving"].ravel()[pick], cal_data["clim_wet"].ravel()[pick],
                                              obs[pick])
        except Exception as error:
            log(f"fallback fit skipped ({type(error).__name__}: {error})")
        return extras

    def finish_stage_b(self, model, loss, geo, grids, clim, stations, st_points, st_clim, batch_size: int):
        """Stage B on the GFS runs fetched so far; returns ``(model, loss, builder, held-out starts)`` or None."""
        from .dataset import SampleBuilder
        from .sources.cmorph import CMORPHCache
        from .sources.nwp import NWPReader
        from .sources.stations import StationCache
        from .train import Trainer

        if self._prefetch is not None:
            self._prefetch.join(timeout=max(0.0, min(1800.0, (self.left() - 2.5) * 3600)))
            if self._prefetch.is_alive():
                log("stage B: the GFS download is still running; training on the runs fetched so far")
        b_root = self.root / "stage_b_data"
        try:
            cmorph_b = CMORPHCache(b_root / "cmorph")
            stations_b = StationCache(b_root / "stations") if (b_root / "stations" / "stations" / "stations.json").exists() else None
        except Exception as error:
            log(f"stage B: skipped (truth not staged: {error})")
            return None
        lat1, lon1, lat2, lon2 = (np.asarray(grids[k]) for k in ("lat1", "lon1", "lat2", "lon2"))
        reader = NWPReader("gfs", lat1, lon1, lat2, lon2, b_root / "nwp")
        cached = {}
        for init in self.stage_b_inits():
            stamp = str(np.datetime64(init, "h"))[:13].replace(":", "")
            if (b_root / "nwp" / f"gfs_{stamp}+6.npz").exists():
                cached[init + 6] = init

        def driver(t0):
            frames = reader.frames(cached[int(t0)], offset=6)
            return frames["fields"], frames["upper"]

        holder = SimpleNamespace(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2)
        st = stations_b if stations_b is not None and len(stations_b) == len(stations) else None
        builder = SampleBuilder(geo, holder, cmorph_b, st, clim, st_clim if st is not None else None, driver=driver)
        starts = []
        for t0 in sorted(cached):
            truth = cmorph_b.targets(np.array([t0, t0 + 71]), np.array([0]))
            if np.isfinite(truth).all():
                starts.append(t0)
        starts = np.array(starts, np.int64)
        if starts.size < 24:
            log(f"stage B: only {starts.size} usable GFS runs; skipped")
            return None
        cut = int(starts.size * 0.75)
        train_b, held_b = starts[:cut], starts[cut:]
        hours_b = max(0.2, self.left() - (0.9 + 0.1 * self.cfg.gpu_hours / 24))
        cfg_b = self.train_cfg(hours_b, warmup_steps=50)
        cfg_b.lr = self.cfg.lr / 3
        # stage B's forecasts are built differently, so its step is sized afresh, unless a batch was given by hand
        # or the fastest batch is wanted, in which case stage A's is kept
        cfg_b.batch_size = 0 if (self.cfg.vram == "fill" and not self.cfg.batch_size) else batch_size
        if st is None:
            model.link_stations = None                 # airports of stage B differ: train without them
        trainer = Trainer(model, loss, builder, train_b, held_b[:16], self.root / "stage_b", cfg_b, self.mirror,
                          "checkpoint/stage_b", stage="B")
        trainer.cfg.hours = trainer.elapsed / 3600 + hours_b
        result = trainer.run()
        self.mark("train_b", steps=result["step"], hours=result["elapsed"] / 3600, runs=int(starts.size))
        return trainer.model, trainer.loss_fn, builder, held_b

    def publish(self, out: Path) -> None:
        """Upload the release folder to the Hub repository (one commit). The network's own card, with the scores
        measured in this run, always goes up as ``NETWORK_CARD.md``. It also becomes the repository's ``README.md``
        unless the repository already has one (the full card that ``hub/upload_to_hub.py`` puts there)."""
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self.mirror.token)
            api.create_repo(self.cfg.hub_repo, repo_type="model", exist_ok=True, private=self.cfg.private)
            if (out / "README.md").exists():
                shutil.copy2(out / "README.md", out / "NETWORK_CARD.md")
            try:
                has_card = bool(api.file_exists(self.cfg.hub_repo, "README.md"))
            except Exception:
                has_card = False
            api.upload_folder(folder_path=str(out), repo_id=self.cfg.hub_repo, repo_type="model",
                              commit_message="Pluvium-500 release", ignore_patterns=["README.md"] if has_card else None)
            log(f"published to https://huggingface.co/{self.cfg.hub_repo}")
        except Exception as error:
            log(f"publish skipped ({type(error).__name__}: {error})")


# ------------------------------------------------------------------------------- background jobs --

def _merge_years(era5_json: Path, years, cmorph_json: Path) -> None:
    """After pulling per-year files from the Hub, make sure the local metadata lists every year present."""
    for path, folder, pattern in ((era5_json, era5_json.parent / "era5_1h", "tp_{y}.npy"),
                                  (cmorph_json, cmorph_json.parent, "targets_{y}.npy")):
        if not path.exists():
            continue
        meta = json.loads(path.read_text())
        have = set(meta.get("years", []))
        have |= {y for y in years if (folder / pattern.format(y=y)).exists()}
        meta["years"] = sorted(have)
        path.write_text(json.dumps(meta))


def _stage_era5_job(root: str, years: list, box: list | None = None) -> None:
    from .domain import DOMAIN
    from .sources.era5 import stage_era5

    stage_era5(root, years=years, box=tuple(box) if box else DOMAIN, workers=48)


def _stage_cmorph_job(root: str, geo_dir: str, years: list) -> None:
    from .domain import Geography
    from .sources.cmorph import stage_cmorph

    geo = Geography.load(geo_dir)
    stage_cmorph(root, geo.targets, geo.mesh, years=years, workers=max(8, min(32, (os.cpu_count() or 8))),
                 cmorph_box=geo.cmorph_box)


def _stage_stations_job(root: str, years: list, box: list | None = None) -> None:
    from .domain import TARGET_BOX
    from .sources.stations import stage_stations

    stage_stations(root, years, box=tuple(box) if box else TARGET_BOX, workers=32)


def _prefetch_job(root: str, grids: dict, inits: list, period: list) -> None:
    """Stage B's data: CMORPH and airports for the period, then each GFS run (cached as it arrives)."""
    try:
        from .domain import Geography
        from .sources.cmorph import stage_cmorph
        from .sources.nwp import NWPReader
        from .sources.stations import stage_stations

        r = Path(root)
        geo = Geography.load(r / "geography")
        listed = r / "data" / "stations2" / "stations" / "stations.json"
        if listed.exists():                         # the same airports as stage A, in the same order
            (r / "stage_b_data" / "stations" / "stations").mkdir(parents=True, exist_ok=True)
            shutil.copy2(listed, r / "stage_b_data" / "stations" / "stations" / "stations.json")
        a, b = hour_of(period[0]), hour_of(period[1])
        stage_cmorph(r / "stage_b_data" / "cmorph", geo.targets, geo.mesh, period=(a, b), workers=8, verbose=False,
                     cmorph_box=geo.cmorph_box)
        years = sorted({int(str(np.datetime64(h, "h"))[:4]) for h in (a, b - 1)})
        try:
            stage_stations(r / "stage_b_data" / "stations", years, box=geo.target_box, workers=16, verbose=False)
        except Exception as error:
            log(f"stage B airports not staged ({error})")
        lat1, lon1, lat2, lon2 = (np.asarray(grids[k]) for k in ("lat1", "lon1", "lat2", "lon2"))
        reader = NWPReader("gfs", lat1, lon1, lat2, lon2, r / "stage_b_data" / "nwp", workers=16)
        done = 0
        for init in inits:
            try:
                reader.frames(int(init), offset=6)
                done += 1
            except Exception as error:
                log(f"stage B: GFS {hour_to_iso(int(init))} skipped ({type(error).__name__}: {error})")
        log(f"stage B: {done} of {len(inits)} GFS runs fetched")
    except Exception as error:
        log(f"stage B prefetch stopped ({type(error).__name__}: {error})")


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train, calibrate, score and publish Pluvium on one GPU.")
    parser.add_argument("--hours", type=float, default=None, help="time budget (default: GPU_HOURS or 24)")
    parser.add_argument("--root", default=None, help="working folder (default: PLUVIUM_ROOT or ~/pluvium)")
    parser.add_argument("--repo", default=None, help=f"Hub repository (default: HF_REPO or {DEFAULT_REPO}); '' for none")
    parser.add_argument("--size", default=None, choices=("pluvium-tiny", "pluvium-base", "pluvium-500"))
    parser.add_argument("--years", default=None, help="training years, e.g. 2016-2021")
    parser.add_argument("--no-stage-b", action="store_true")
    parser.add_argument("--no-forecast", action="store_true")
    parser.add_argument("--batch", type=int, default=None, help="forecasts per training step (default: found by measuring)")
    parser.add_argument("--vram", default=None, choices=("fill", "fastest"),
                        help="fill: size each training step to the GPU's memory (default). fastest: keep the fastest "
                             "batch per forecast, which leaves most of a large GPU's memory unused")
    parser.add_argument("--vram-share", type=float, default=None,
                        help="how much of the GPU's memory a training step may need when filling (default 0.95)")
    parser.add_argument("--country", default=None, help="ISO codes of the area to serve, e.g. TT or DO,HT (default: the Caribbean)")
    parser.add_argument("--box", default=None, help="south,north,west,east instead of the countries' own extent")
    args = parser.parse_args(argv)
    # segments that can grow keep the allocator's footprint close to what the tensors need, which matters when a
    # step is sized to fill the card. Left alone when the allocator has been configured by hand.
    if not any(os.environ.get(k) for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cfg = ProgramConfig.from_env()
    if args.hours is not None:
        cfg.gpu_hours = args.hours
    if args.root:
        cfg.root = args.root
    if args.repo is not None:
        cfg.hub_repo = args.repo or None
    if args.size:
        cfg.size = args.size
    if args.years:
        a, _, b = args.years.partition("-")
        cfg.train_years = tuple(range(int(a), int(b or a) + 1))
    if args.no_stage_b:
        cfg.stage_b = False
    cfg.forecast = not args.no_forecast
    if args.batch is not None:
        cfg.batch_size = args.batch
    if args.vram:
        cfg.vram = args.vram
    if args.vram_share is not None:
        cfg.vram_share = args.vram_share
    if args.country:
        cfg.countries = tuple(c.strip() for c in args.country.split(",") if c.strip())
    if args.box:
        cfg.box = tuple(float(x) for x in args.box.split(","))
    Program(cfg).run()


__all__ = ["DEFAULT_REPO", "Program", "ProgramConfig", "choose_years", "main"]

# ``python -m pluvium.program`` is how the notebook cells start a training run. Until 0.4.3 these two lines were
# missing: the command imported this module and ended at once with exit code 0, having trained nothing.
if __name__ == "__main__":
    main()
