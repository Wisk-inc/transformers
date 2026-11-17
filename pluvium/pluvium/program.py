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
On a machine that stops a session after a fixed time (molab: 12 hours) the budget ends before that, so the run is
scored and published while the session is still up.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .util import atomic_save, hour_of, hour_to_iso, log, year_hours

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


# ------------------------------------------------------------------------------------- sessions --
#: molab, where these runs are made, stops a notebook 12 hours after it starts ("Notebooks can run for as long as 12
#: hours before molab shuts them down", docs.marimo.io/guides/molab). The first full run there trained for 11.6 of
#: its 24 hours and was stopped before it was scored and published.
MOLAB_HOURS = 12.0
#: how long before a session's end the run means to be done
SESSION_MARGIN_HOURS = 0.5
#: the least time kept for calibration, scoring, publishing and the forecast when the session has an end
SESSION_RESERVE_HOURS = 1.5
#: where molab keeps a notebook's packages (the tracebacks of the runs there show it)
MOLAB_PACKAGES = "/tmp/uv-venv"
#: cells each held-out forecast is scored at in calibration and the scores. Scoring one forecast took 12.2 s at 7,220
#: cells on a CPU and grows in step with them: the network's 7,500 would have needed about 2.2 hours for the four
#: passes, more than a molab session has left for them
SCORE_POINTS = 2000
#: hours kept after the scores for the release itself: its extras, the card and the upload
PUBLISH_HOURS = 0.3


def _proc_stat(pid: int) -> list:
    """The fields of ``/proc/<pid>/stat`` after the command name, so that field 3 (the state) is the first."""
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()


def _ancestors(most: int = 16) -> list:
    """This process and the ones above it, nearest first: the program, its supervisor, the notebook, ..."""
    out, pid = [], os.getpid()
    while pid > 0 and len(out) < most:
        out.append(pid)
        try:
            pid = int(_proc_stat(pid)[1])
        except (OSError, ValueError, IndexError):
            break
    return out


def _began(pid: int) -> float | None:
    """When a process started, in seconds since 1970, or None where there is no ``/proc``."""
    try:
        boot = next(float(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                    if line.startswith("btime"))
        return boot + int(_proc_stat(pid)[19]) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def _command(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def on_molab(env=None, places=None, commands=None) -> bool:
    """Whether this runs in a molab notebook: the packages are where molab keeps them, a process above this one is
    marimo, or marimo's settings are in the environment."""
    env = os.environ if env is None else env
    places = [sys.prefix, __file__] if places is None else places
    commands = [_command(pid) for pid in _ancestors()[1:]] if commands is None else commands
    return (any(str(p).startswith(MOLAB_PACKAGES) for p in places) or any("marimo" in c for c in commands)
            or any(k.startswith("MARIMO") or "MOLAB" in k for k in env))


def session_limit(env=None) -> tuple:
    """``(hours, why)``: how long one session on this machine may run, or ``(None, why)`` when nothing says it ends.
    ``PLUVIUM_SESSION_HOURS`` says so (0 for no end); otherwise a molab notebook has 12 hours."""
    env = os.environ if env is None else env
    given = str(env.get("PLUVIUM_SESSION_HOURS", "")).strip()
    if given:
        hours, why = float(given), env.get("PLUVIUM_SESSION_WHY") or "PLUVIUM_SESSION_HOURS"
        return (hours, why) if hours > 0 else (None, why)
    if on_molab(env):
        return MOLAB_HOURS, "molab stops a notebook after 12 hours"
    return None, "nothing says this machine stops its sessions"


def session_start(hours: float, env=None, now: float | None = None) -> float:
    """When this session began: ``PLUVIUM_SESSION_START`` when the supervisor passes it on, otherwise the earliest
    start among this process and the ones above it (the notebook, its server, the container) that is less than
    ``hours`` ago. A process older than that was there before the session."""
    env = os.environ if env is None else env
    now = time.time() if now is None else now
    if env.get("PLUVIUM_SESSION_START"):
        return float(env["PLUVIUM_SESSION_START"])
    times = [t for t in map(_began, _ancestors()) if t is not None and now - hours * 3600.0 < t <= now]
    return min(times, default=now)


def _read_state(path: Path) -> dict:
    """What the runs before this one in the folder finished. A file torn by a stop counts as nothing finished: every
    stage checks its own files, so the run goes on from them."""
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        log(f"{path.name} could not be read, so it is written again as the stages finish")
        return {}


def _clock(seconds: float) -> str:
    return time.strftime("%H:%M", time.gmtime(seconds))


class Program:
    def __init__(self, cfg: ProgramConfig) -> None:
        self.cfg = cfg
        self.started = time.time()
        self.deadline = self.started + cfg.gpu_hours * 3600.0
        #: ``(began, hours, why)`` when this machine stops its sessions after a fixed time. The run is planned to be
        #: published half an hour before then; a run stopped before that publishes nothing
        self.session = None
        hours, why = session_limit()
        if hours:
            began = session_start(hours)
            end = began + (hours - SESSION_MARGIN_HOURS) * 3600.0
            if end < self.deadline:
                self.deadline, self.session = end, (began, hours, why)
        self.root = Path(cfg.root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "program.json"
        self.state = _read_state(self.state_path)
        self.years = tuple(cfg.train_years or choose_years(cfg.gpu_hours, cfg.holdout_year))
        self.stage_b = cfg.stage_b if cfg.stage_b is not None else cfg.gpu_hours >= 12
        from naturev1.mirror import HubMirror

        self.mirror = HubMirror(cfg.hub_repo, cfg.hf_token, private=cfg.private) if cfg.hub_repo else None
        self._prefetch = None

    # ---------------------------------------------------------------------------------- helpers --
    def left(self) -> float:
        """Hours left in the budget."""
        return (self.deadline - time.time()) / 3600.0

    def reserve(self) -> float:
        """Hours kept for calibration, scoring, publishing and the forecast. More when the session has an end: none
        of these has run at full size on a GPU yet, and a session stopped in the middle of them publishes nothing."""
        base = 0.9 + 0.1 * self.cfg.gpu_hours / 24
        return max(base, SESSION_RESERVE_HOURS) if self.session else base

    def mark(self, stage: str, **info) -> None:
        self.state[stage] = {"done": time.strftime("%Y-%m-%d %H:%M"), **info}
        atomic_save(self.state_path, lambda p: Path(p).write_text(json.dumps(self.state, indent=1)))

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

    def push_many(self, files: list, what: str) -> None:
        """Mirror many files in a few commits, in the background (:func:`mirror_files`)."""
        if self.mirror is None or not self.mirror.can_push:
            return
        thread = threading.Thread(target=mirror_files, args=(self.mirror, files, what), name=f"mirror-{what}", daemon=True)
        thread.start()
        self.mirror._threads.append(thread)                 # waited for with the mirror's own uploads

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
        # mirror what was staged, in a few commits (one commit per file ran into the Hub's limit, HTTP 429)
        files = []
        for y in years:
            for s in HOURLY_SHORT:
                files.append((era5_dir / "era5_1h" / f"{s}_{y}.npy", f"data/era5/era5_1h/{s}_{y}.npy"))
            for s in UPPER_SHORT:
                for p in LEVELS:
                    files.append((era5_dir / "era5_6h" / f"{s}{p}_{y}.npy", f"data/era5/era5_6h/{s}{p}_{y}.npy"))
            for k in ("targets", "mesh"):
                files.append((cm_dir / "cmorph" / f"{k}_{y}.npy", f"data/cmorph/{k}_{y}.npy"))
            files.append((st_dir / "stations" / f"{y}.npy", f"data/stations2/{y}.npy"))
        files += [(era5_dir / "era5.json", "data/era5/era5.json"),
                  (cm_dir / "cmorph" / "cmorph.json", "data/cmorph/cmorph.json"),
                  (st_dir / "stations" / "stations.json", "data/stations2/stations.json")]
        gone = [str(local) for local, _ in files if not Path(local).exists()]
        if gone:
            raise RuntimeError(f"{len(gone)} data files are missing after staging, the first is {gone[0]}. Running the "
                               "program again stages them; if it comes back, delete that source's folder under "
                               f"{d} and run again")
        self.push_many(files, "data")
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

    def stop_prefetch(self) -> None:
        """End the background fetch if it is still going. A run that had failed used to wait for it at exit, an hour
        and a half the first time it happened."""
        fetch = self._prefetch
        if fetch is not None and fetch.is_alive():
            fetch.terminate()
            fetch.join(timeout=30)
            if fetch.is_alive():
                fetch.kill()

    # -------------------------------------------------------------------------------------- train --
    def train_cfg(self, hours: float, **kw):
        from .train import TrainConfig

        v3 = not self.cfg.size.endswith(("-v1", "-v2"))
        kw = {"ema_decay": 0.999 if v3 else 0.0, "curriculum": v3, **kw}
        # a run that was started again (see supervise) draws its batches differently, so that a batch which
        # stopped the run before it is not met again at the same step
        kw.setdefault("seed", max(0, int(os.environ.get("PLUVIUM_ATTEMPT") or 1) - 1))
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
        if self.session:
            began, hours, why = self.session
            log(f"this session began at {_clock(began)} UTC and ends {hours:g} hours later ({why}): the network is "
                f"trained, scored and published by {_clock(self.deadline)} UTC")
            if self.left() < self.reserve() + 1.0:
                log(f"only {self.left():.1f} hours of this session are left, so training gets little of them; a "
                    f"session started fresh has all {hours:g}")
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
        reserve = self.reserve()                           # calibration, scoring, publishing, forecast
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
        # calibration and scores. Each pass may use a share of the time left before the release has to start and
        # stops early when that is spent, so that the run is published within its time whatever the scoring costs
        publish_by = self.deadline - PUBLISH_HOURS * 3600.0

        def until(share: float) -> float:
            return time.time() + share * max(0.0, publish_by - time.time())

        log("calibration: hindcasts on the held-out year (with forecast-like errors) ...")
        cal_board, cal_data = hindcast(model, builder, cal_starts, batch_size=2, emulate=True, max_points=SCORE_POINTS,
                                       until=until(0.4))
        calibrator = fit_calibration(cal_data)
        scores = {}
        log("scores: held-out year, driving fields = reanalysis (an upper bound on live skill)")
        board, _ = hindcast(model, builder, test_starts, batch_size=2, emulate=False, calibrator=calibrator,
                            max_points=SCORE_POINTS, until=until(0.4))
        scores["reanalysis_driven"] = board.report()
        log(format_report(scores["reanalysis_driven"]))
        log("scores: held-out year, reanalysis with emulated forecast errors (closer to live)")
        board, _ = hindcast(model, builder, test_starts, batch_size=2, emulate=True, calibrator=calibrator,
                            max_points=SCORE_POINTS, until=until(0.6))
        scores["emulated_errors"] = board.report()
        log(format_report(scores["emulated_errors"]))
        if live_builder is not None and live_starts is not None and live_starts.size:
            half = live_starts.size // 2
            _, live_cal = hindcast(model, live_builder, live_starts[:half], batch_size=2, max_points=SCORE_POINTS,
                                   until=until(0.5))
            calibrator = fit_calibration(live_cal)
            board, _ = hindcast(model, live_builder, live_starts[half:], batch_size=2, calibrator=calibrator,
                                max_points=SCORE_POINTS, until=until(1.0))
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
        if cfg.forecast and self.left() <= 0:
            log("the live forecast is left out: the run's time is up")
        elif cfg.forecast:
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
        hours_b = max(0.2, self.left() - self.reserve())
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
    """Stage B's data: CMORPH and airports for the period, then each GFS run (cached as it arrives). It stops when the
    program that started it is gone: a program killed outright left it running, and the one started after it began
    a second download of the same files."""
    parent = os.getppid()
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
            if os.getppid() != parent:
                log("stage B: the program that started this download is gone; stopping it")
                return
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
    if _supervised(cfg):
        raise SystemExit(supervise(sys.argv[1:] if argv is None else list(argv), cfg.vram_share, vram=cfg.vram,
                                   root=cfg.root, repo=cfg.hub_repo, token=cfg.hf_token))
    program = Program(cfg)
    try:
        program.run()
    finally:
        program.stop_prefetch()


#: files mirrored per commit. The Hub takes at most 128 commits an hour in a repository. Until 0.5.2 the staged
#: data went up one commit per file, about 150 of them at once: most were refused (HTTP 429), and the checkpoints of
#: the following hour could be refused too
FILES_PER_COMMIT = 60


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_id(path: Path) -> str:
    """What the Hub reports for a small (non-LFS) file: git's blob id."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def mirror_files(mirror, files: list, what: str, api=None) -> int:
    """Upload the files that the repository does not hold yet, or holds in another version, ``FILES_PER_COMMIT`` to
    a commit. ``files`` is a list of ``(local path, path in the repository)``. Returns how many went up. A refusal is
    reported and never raised: a mirror that fails must not stop a run."""
    if mirror is None or not mirror.can_push:
        return 0
    files = [(Path(local), str(path)) for local, path in files if Path(local).exists()]
    if not files:
        return 0
    try:
        from huggingface_hub import CommitOperationAdd, HfApi

        api = api or HfApi(token=mirror.token)
        held = {entry.path: entry for entry in api.get_paths_info(mirror.repo_id, [p for _, p in files], repo_type="model")}

        def same(local: Path, path: str) -> bool:
            entry = held.get(path)
            if entry is None:
                return False
            lfs = getattr(entry, "lfs", None)
            if lfs is not None:
                sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
                return sha == _sha256(local)
            return getattr(entry, "blob_id", None) == _git_blob_id(local)

        todo = [(local, path) for local, path in files if not same(local, path)]
        for k in range(0, len(todo), FILES_PER_COMMIT):
            part = todo[k: k + FILES_PER_COMMIT]
            api.create_commit(mirror.repo_id, repo_type="model", commit_message=f"mirror {what} ({len(part)} files)",
                              operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(f)) for f, p in part])
        if todo:
            log(f"[mirror] {what}: {len(todo)} files uploaded in {-(-len(todo) // FILES_PER_COMMIT)} commit(s)")
        return len(todo)
    except Exception as error:
        log(f"[mirror] {what}: not mirrored ({type(error).__name__}: {str(error)[:200]}); the run goes on without it")
        return 0


#: how much less of the GPU's memory a training step is sized to when a run is started again after it stopped
RETRY_SHARE_DROP = 0.10


def _supervised(cfg: ProgramConfig) -> bool:
    """A run on a GPU is watched by a parent process (:func:`supervise`), unless it is that parent's child, a batch
    was given by hand, or ``PLUVIUM_RETRIES=0``."""
    if cfg.batch_size > 0 or os.environ.get("PLUVIUM_ATTEMPT") or os.environ.get("PLUVIUM_RETRIES") == "0":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def retry_plan(vram: str, share: float) -> list:
    """The settings a run is tried with, in order, as ``(vram, share, batch)``: a run that fills the card, then one
    that fills ``RETRY_SHARE_DROP`` less of it, then the batch rule of 0.4.2, then one forecast per step. A run started
    with the old rule begins at the third."""
    tries = [("fill", share, 0), ("fill", round(max(0.5, share - RETRY_SHARE_DROP), 2), 0)] if vram == "fill" else []
    return tries + [("fastest", share, 0), ("fastest", share, 1)]


#: a run that prints nothing for this many minutes is taken to have hung; it is stopped and started again
STALL_MINUTES = 120.0
#: words in a stopped run's last lines that say the step was too large for the GPU, so the next try is smaller
SIZE_SIGNS = ("out of memory", "OutOfMemoryError", "CUBLAS_STATUS", "CUDNN_STATUS", "CUDA error", "cudaError")


def about_size(tail) -> bool:
    """Whether a stopped run's last lines say its step was too large for the GPU."""
    return any(sign in line for line in list(tail)[-60:] for sign in SIZE_SIGNS)
#: lines of a stopped run's output that are kept, on disk and on the Hub, so that its error can be read afterwards
TAIL_LINES = 400
_SECRET = re.compile(r"hf_[A-Za-z0-9]{20,}|pypi-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}")


def save_attempt(root, repo, token, number: int, how: str, tail: list, api=None) -> Path | None:
    """Write the end of a stopped run's output to ``<root>/logs/attempt-<number>.txt`` and, with a repository and a
    token, put the file on the Hub under ``logs/``, where it can be read from any machine. Anything that looks like
    a token is blanked first. Returns the local file; never raises."""
    try:
        folder = Path(root).expanduser() / "logs"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"attempt-{number}.txt"
        path.write_text(_SECRET.sub("[token]", f"{how}\n\n" + "".join(tail)), encoding="utf-8")
    except Exception:
        return None
    try:
        if repo and (token or api is not None):
            if api is None:
                from huggingface_hub import HfApi

                api = HfApi(token=token)
            api.upload_file(path_or_fileobj=str(path), path_in_repo=f"logs/{path.name}", repo_id=repo, repo_type="model",
                            commit_message=f"log of a run that stopped ({path.name})")
    except Exception as error:
        log(f"the stopped run's log was not put on the Hub ({type(error).__name__}: {str(error)[:120]}); it is in {path}")
    return path


def supervise(argv: list, share: float, command: list | None = None, vram: str = "fill", root=None, repo=None,
              token=None, stall_minutes: float | None = None, api=None) -> int:
    """Run the program in a child process and keep it going. If the child stops with an error, is killed (the
    machine ran out of RAM, say), or prints nothing for ``stall_minutes``, it is started again from its last
    checkpoint, with a smaller step (:func:`retry_plan`) when the stop could have come from the step's size. Any other
    error is tried again at the same size: 0.5.5 stopped on a bug that had nothing to do with memory, and every try
    after it had less of the card. A fresh process starts with a fresh CUDA context,
    which some GPU errors leave unusable; the data and the checkpoints are on disk, so the next try is training
    again within minutes.

    The end of each stopped run's output is kept in ``<root>/logs`` and put on the Hub (:func:`save_attempt`): the
    first GPU runs stopped with an error nobody saw the last line of. Returns 0, or the last child's exit code.
    Stopping this process (the notebook's stop button) stops the child and nothing is started again."""
    import signal
    import subprocess
    from collections import deque

    command = command or [sys.executable, "-u", "-m", "pluvium.program"]
    tries = retry_plan(vram, share)
    limit = 60.0 * float(stall_minutes if stall_minutes is not None else os.environ.get("PLUVIUM_STALL_MINUTES") or STALL_MINUTES)
    state = {"child": None}
    hours, why = session_limit()                              # found once, so that every try plans the same end
    session = {"PLUVIUM_SESSION_HOURS": f"{hours or 0:g}", "PLUVIUM_SESSION_WHY": why,
               "PLUVIUM_SESSION_START": f"{session_start(hours) if hours else time.time():.0f}"}

    def end_child() -> None:
        child = state["child"]
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()

    def stop(signum, frame):                                  # stopping this process stops the training too
        end_child()
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, stop)
    code, level = 1, 0
    try:
        for k in range(len(tries)):
            rule, cut, batch = tries[level]
            env = dict(os.environ, **session, PLUVIUM_ATTEMPT=str(k + 1), PLUVIUM_VRAM=rule,
                       PLUVIUM_VRAM_SHARE=str(cut), PYTHONUNBUFFERED="1")
            extra = ["--vram", rule, "--vram-share", str(cut)]
            if batch:
                env["PLUVIUM_BATCH"] = str(batch)
                extra += ["--batch", str(batch)]
            tail, seen = deque(maxlen=TAIL_LINES), [time.time()]
            child = state["child"] = subprocess.Popen([*command, *argv, *extra], env=env, stdout=subprocess.PIPE,
                                                      stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")

            def relay(stream=child.stdout, tail=tail, seen=seen) -> None:
                for line in stream:                           # shown as it comes, and the end of it kept
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    tail.append(line)
                    seen[0] = time.time()

            reader = threading.Thread(target=relay, daemon=True)
            reader.start()
            hung = False
            while child.poll() is None:
                time.sleep(min(2.0, max(0.05, limit / 20)))
                if time.time() - seen[0] > limit:
                    hung = True
                    end_child()
            reader.join(timeout=15)
            code = child.returncode
            if code == 0 and not hung:
                return 0
            what = (f"printed nothing for {limit / 60:.0f} minutes and was stopped" if hung else
                    f"was killed (signal {-code})" if code < 0 else f"stopped with exit code {code}")
            settings = f"{rule}, share {cut}" + (f", batch {batch}" if batch else "")
            saved = save_attempt(root, repo, token, k + 1, f"attempt {k + 1} of {len(tries)} ({settings}): the program {what}",
                                 list(tail), api=api) if root else None
            smaller = hung or code < 0 or about_size(tail)
            code = code if code > 0 else 1
            if k + 1 < len(tries):
                level = min(level + 1, len(tries) - 1) if smaller else level
                then, then_cut, then_batch = tries[level]
                how = ("with the same settings, since the error was not about memory" if not smaller else
                       f"with one forecast per step (--batch {then_batch})" if then_batch else
                       f"with each training step sized to {then_cut:.0%} of the GPU's memory" if then == "fill" else
                       "with the batch rule of 0.4.2 (--vram fastest)")
                log(f"the program {what}" + (f"; the end of its output is in {saved}" if saved else "")
                    + f". Starting it again from its last checkpoint, {how}")
        return code
    except KeyboardInterrupt:
        end_child()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


__all__ = ["DEFAULT_REPO", "FILES_PER_COMMIT", "MOLAB_HOURS", "Program", "ProgramConfig", "RETRY_SHARE_DROP",
           "SESSION_MARGIN_HOURS", "SESSION_RESERVE_HOURS", "STALL_MINUTES", "about_size", "choose_years", "main",
           "mirror_files", "on_molab", "retry_plan", "save_attempt", "session_limit", "session_start", "supervise"]

# ``python -m pluvium.program`` is how the notebook cells start a training run. Until 0.4.3 these two lines were
# missing: the command imported this module and ended at once with exit code 0, having trained nothing.
if __name__ == "__main__":
    main()
