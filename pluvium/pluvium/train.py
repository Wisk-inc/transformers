# Copyright 2026 Nathan. Apache-2.0.
"""
Training on one GPU: batches built on every CPU core, bf16 on the GPU, a step sized by measuring to fill the GPU's
memory (:mod:`pluvium.training.fit`), a learning-rate schedule that ends when the time budget does, checkpoints that
resume exactly, and (optionally) a mirror of every checkpoint on the Hugging Face Hub so a rented machine that stops
loses nothing.

Stage A trains on ERA5 (perfect prognosis, with forecast-error emulation); stage B continues on real GFS runs
(model output statistics). Both use :class:`Trainer`; only the examples' driving fields differ.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .calibrate import Calibrator
from .dataset import Climatology, SampleBuilder, collate
from .domain import PointSet
from .losses import PluviumLoss
from .model import PluviumConfig, PluviumModel
from .products import WINDOWS, summarize
from .schedule import DEFAULT, Schedule
from .util import atomic_save, log
from .verify import Scoreboard

CHECKPOINT = "model.pt"
META = "pluvium.json"


@dataclass
class TrainConfig:
    size: str = "pluvium-500"
    batch_size: int = 0                  # 0: measure and pick the fastest that fits
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.05
    warmup_steps: int = 300
    max_steps: int = 1_000_000
    hours: float = 10.0                  # this stage's time budget; the schedule ends with it
    grad_clip: float = 1.0
    bf16: bool = True
    checkpointing: bool = False          # gradient checkpointing (memory for speed)
    workers: int = -1                    # -1: every core but two
    eval_every_minutes: float = 30.0
    save_every_minutes: float = 10.0
    eval_forecasts: int = 32
    log_every: int = 25
    seed: int = 0
    ema_decay: float = 0.0               # > 0: keep an exponential moving average of the weights (released)
    curriculum: bool = False             # draw issue times favouring heavy rain and thunder (pluvium.training)
    expert_balance: float = 0.01         # weight of the mixture-of-experts load-balancing term
    overrides: dict = field(default_factory=dict)   # PluviumConfig overrides (ablations)
    memory: str = "fill"                 # "fill": size the step to the GPU's memory; "fastest": the fastest batch
    memory_share: float = 0.95           # how much of the GPU's memory a step may need when filling
    lr_reference_batch: int = 8          # the learning rate is as given up to this batch and grows above it
    lr_scale_cap: float = 2.0            # ... with the square root of the batch, to at most this many times
    probe_steps: int = 2                 # timed steps per measurement when sizing


class CudaMeter:
    """How much of a CUDA device's memory training needs, read from PyTorch's allocator. The trainer sizes its
    steps with one of these; the tests pass a made-up one, so the sizing is checked without a GPU."""

    def __init__(self, device) -> None:
        self.device = device

    def usable(self) -> float:
        """Bytes this process can have: what is free on the card now, and what it already holds."""
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(self.device)
        return float(free + torch.cuda.memory_reserved(self.device))

    def start(self, batch: int = 0, scale: float = 1.0, checkpointing: bool = False) -> None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)

    def sync(self) -> None:
        torch.cuda.synchronize(self.device)

    def peak(self) -> float:
        """The most the allocator held since :meth:`start`: what ``nvidia-smi`` shows for this process, less the
        CUDA context."""
        torch.cuda.synchronize(self.device)
        return float(torch.cuda.max_memory_reserved(self.device))

    def used(self) -> tuple:
        """``(bytes in use on the card by everything, the card's size)``, as ``nvidia-smi`` reports them."""
        free, total = torch.cuda.mem_get_info(self.device)
        return float(total - free), float(total)

    def release(self) -> None:
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------- batches --

class BatchSet(torch.utils.data.Dataset):
    """Item ``i`` is a whole batch: ``batch_size`` issue times drawn for step ``i`` and one shared target subset.
    Deterministic in ``(seed, i)``, so a resumed run sees the batches it would have seen."""

    def __init__(self, builder: SampleBuilder, starts: np.ndarray, batch_size: int, seed: int = 0,
                 sampler=None) -> None:
        self.builder, self.starts, self.batch_size, self.seed = builder, np.asarray(starts), batch_size, seed
        self.sampler = sampler

    def __len__(self) -> int:
        return 1 << 40

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng((self.seed, int(i)))
        if self.sampler is not None:
            t0s, weights = self.sampler.draw(rng, self.batch_size)
        else:
            t0s = rng.choice(self.starts, self.batch_size, replace=self.starts.size < self.batch_size)
            weights = np.ones(self.batch_size)
        self.builder.rng = np.random.default_rng((self.seed, int(i), 1))
        targets = self.builder.choose_targets(train=True)
        examples = [self.builder.example(int(t), True, targets) for t in t0s]
        for e, w in zip(examples, weights):
            e["weight"] = np.float32(w)
        return collate(examples)


def _workers(requested: int) -> int:
    if requested >= 0:
        return requested
    return max(0, (os.cpu_count() or 2) - 2)


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# -------------------------------------------------------------------------------------------- set-up --

def station_points(geo, stations) -> PointSet | None:
    """The airports as decoder points, with their ICAO codes and names when the cache has them (live reports are
    matched by code)."""
    if stations is None or len(stations) == 0:
        return None
    lat, lon = np.asarray(stations.lat), np.asarray(stations.lon)
    listed = getattr(stations, "meta", {}).get("stations", []) if hasattr(stations, "meta") else []
    extra = {}
    if len(listed) == lat.size:
        extra = {"icao": np.array([str(s.get("icao", "")) for s in listed]),
                 "name": np.array([str(s.get("name", "")) for s in listed])}
    return PointSet(lat, lon, geo.terrain.descriptors(lat, lon), np.zeros(lat.size, np.int8), None, extra)


def nearest_target(geo, lat, lon) -> np.ndarray:
    """Index of the CMORPH target cell nearest each point (towns and airports borrow its climatology)."""
    from scipy.spatial import cKDTree

    t = geo.targets
    xyz = lambda la, lo: np.stack([np.cos(np.radians(la)) * np.cos(np.radians(lo)),     # noqa: E731
                                   np.cos(np.radians(la)) * np.sin(np.radians(lo)), np.sin(np.radians(la))], -1)
    return cKDTree(xyz(t.lat, t.lon)).query(xyz(np.asarray(lat), np.asarray(lon)))[1]


def borrowed_climatology(clim: Climatology, index: np.ndarray) -> Climatology:
    return Climatology(clim.wet[:, :, index], clim.rate[:, :, index])


def build_model(geo, grids: dict, size: str = "pluvium-500", stations: PointSet | None = None,
                schedule: Schedule = DEFAULT, **overrides) -> PluviumModel:
    config = PluviumConfig.named(size, **overrides)
    return PluviumModel(config, geo, grids, schedule, stations)


def grids_of(source) -> dict:
    return {k: np.asarray(getattr(source, k), np.float64).tolist() for k in ("lat1", "lon1", "lat2", "lon2")}


# ----------------------------------------------------------------------------------------- checkpoints --

def save_checkpoint(directory, model: PluviumModel, loss: PluviumLoss, optimizer=None, state: dict | None = None,
                    calibrator: Calibrator | None = None, extra: dict | None = None,
                    extra_payload: dict | None = None) -> Path:
    """``model.pt`` (weights, truth-source model, optimizer) and ``pluvium.json`` (everything needed to rebuild the
    model around those weights)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "loss": loss.state_dict(), "state": state or {}, **(extra_payload or {})}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    atomic_save(d / CHECKPOINT, lambda p: torch.save(payload, p))
    meta = {"config": model.config.to_dict(), "grids": model.grids, "schedule": asdict(model.schedule),
            "stations": None if model.stations is None else {
                "lat": model.stations.lat.tolist(), "lon": model.stations.lon.tolist(),
                "list": [{"icao": str(model.stations.extra.get("icao", [""] * len(model.stations))[k]),
                          "name": str(model.stations.extra.get("name", [""] * len(model.stations))[k]),
                          "lat": float(model.stations.lat[k]), "lon": float(model.stations.lon[k])}
                         for k in range(len(model.stations))]},
            "parameters": model.num_parameters(), "truth_sources": loss.sources.describe(), "state": state or {}}
    previous = read_meta(d)
    if calibrator is None and previous.get("calibration"):
        meta["calibration"] = previous["calibration"]
    elif calibrator is not None:
        meta["calibration"] = calibrator.to_json()
    for key in ("scores", "data"):
        if key in previous:
            meta[key] = previous[key]
    meta.update(extra or {})
    atomic_save(d / META, lambda p: Path(p).write_text(json.dumps(meta, indent=1)))
    return d


def read_meta(directory) -> dict:
    path = Path(directory) / META
    return json.loads(path.read_text()) if path.exists() else {}


def load_model(directory, geo, device="cpu", with_optimizer: bool = False):
    """Rebuild a saved model around ``geo``; returns ``(model, loss, meta, payload)``."""
    d = Path(directory)
    meta = read_meta(d)
    config = PluviumConfig.from_dict(meta["config"])
    stations = None
    if meta.get("stations"):
        lat, lon = np.array(meta["stations"]["lat"]), np.array(meta["stations"]["lon"])
        listed = meta["stations"].get("list") or []
        extra = {"icao": np.array([s.get("icao", "") for s in listed]),
                 "name": np.array([s.get("name", "") for s in listed])} if len(listed) == lat.size else {}
        stations = PointSet(lat, lon, geo.terrain.descriptors(lat, lon), np.zeros(lat.size, np.int8), None, extra)
    model = PluviumModel(config, geo, meta["grids"], Schedule(**meta.get("schedule", {})), stations)
    payload = torch.load(d / CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    loss = PluviumLoss(states=config.states)
    loss.load_state_dict(payload["loss"])
    model.to(device)
    loss.to(device)
    if not with_optimizer:
        payload.pop("optimizer", None)
    return model, loss, meta, payload


# ----------------------------------------------------------------------------------------------- loop --

def _param_groups(modules, weight_decay: float):
    decay, no_decay = [], []
    for module in modules:
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.dim() < 2 or "norm" in name or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_factor(step: int, elapsed: float, cfg: TrainConfig) -> float:
    """Linear warm-up, then cosine to ``min_lr_ratio`` as either the steps or the time budget run out."""
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    progress = min(1.0, max(step / max(cfg.max_steps, 1), elapsed / max(cfg.hours * 3600.0, 1.0)))
    return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


class Trainer:
    """
    ``Trainer(model, loss, builder, train_starts, val_starts, out_dir, cfg).run()``.

    ``out_dir`` holds the checkpoint; a run started on a directory that already has one continues it (same step,
    same optimizer moments, same position in the batch sequence, the remaining time budget).
    """

    def __init__(self, model: PluviumModel, loss: PluviumLoss, builder: SampleBuilder, train_starts, val_starts,
                 out_dir, cfg: TrainConfig, mirror=None, mirror_prefix: str = "checkpoint", stage: str = "A",
                 device=None, meter=None) -> None:
        self.model, self.loss_fn, self.builder = model, loss, builder
        self.train_starts, self.val_starts = np.asarray(train_starts), np.asarray(val_starts)
        self.out, self.cfg, self.mirror, self.mirror_prefix, self.stage = Path(out_dir), cfg, mirror, mirror_prefix, stage
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.loss_fn.to(self.device)
        if cfg.checkpointing:
            self.model.gradient_checkpointing_enable()
        #: reads the GPU's memory (``meter`` is for tests: an object with :class:`CudaMeter`'s methods)
        self.meter = meter if meter is not None else (CudaMeter(self.device) if self.device.type == "cuda" else None)
        #: target cells per forecast as the builder was given them; sizing may raise them, evaluation never uses more
        self._usual_cells = (int(builder.max_land), int(builder.ocean_targets))
        self.scale, self.lr_scale, self.fit = 1.0, 1.0, None
        self._probe_examples: dict = {}
        self.elev = torch.tensor(model.geo.targets.statics[:, 0], dtype=torch.float32, device=self.device)
        self.optimizer = torch.optim.AdamW(_param_groups([self.model, self.loss_fn], cfg.weight_decay), lr=cfg.lr,
                                           betas=(0.9, 0.95), fused=self.device.type == "cuda")
        self.step, self.elapsed, self.best = 0, 0.0, float("inf")
        self.history: list[dict] = []
        self.ema = None
        if cfg.ema_decay > 0:
            from .training import EMA

            self.ema = EMA(self.model, cfg.ema_decay)
        self._resume()

    # ------------------------------------------------------------------------------------ resume --
    def _resume(self) -> None:
        path = self.out / CHECKPOINT
        if not path.exists() and self.mirror is not None:
            self.mirror.pull(path, f"{self.mirror_prefix}/{CHECKPOINT}", what=f"stage {self.stage} checkpoint")
            self.mirror.pull(self.out / META, f"{self.mirror_prefix}/{META}", what="checkpoint metadata")
        if not path.exists():
            return
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.loss_fn.load_state_dict(payload["loss"])
        if self.ema is not None and "ema" in payload:
            self.ema.load_state_dict(payload["ema"])
        state = payload.get("state", {})
        if state.get("stage") == self.stage and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
            self.step = int(state.get("step", 0))
            self.elapsed = float(state.get("elapsed", 0.0))
            self.best = float(state.get("best", float("inf")))
            self.history = list(state.get("history", []))
            self._resume_size(state)
            log(f"stage {self.stage}: continuing from step {self.step:,} ({self.elapsed / 3600:.2f} h trained)")
        else:
            log(f"stage {self.stage}: starting from the stage {state.get('stage', '?')} weights")

    def _resume_size(self, state: dict) -> None:
        """Take the step's size from a checkpoint when it still applies. A size found by filling is kept only on a
        card of the same size with the same share asked for; a batch from before sizing-to-memory existed is not
        kept when this run fills, since it left most of a large card empty. Otherwise :meth:`run` measures again."""
        if self.cfg.batch_size > 0:                               # a batch given by hand overrules everything
            return
        fit = state.get("fit")
        filling = self.cfg.memory == "fill" and self.meter is not None
        if fit and filling:
            usable = self.meter.usable()
            same_card = abs(float(fit.get("usable", 0.0)) - usable) <= 0.03 * usable
            if same_card and abs(float(fit.get("share", 0.0)) - self.cfg.memory_share) < 0.005:
                self._adopt(int(fit["batch"]), float(fit["scale"]), bool(fit["checkpointing"]), fit)
                log(f"stage {self.stage}: keeping the step size found before (batch {fit['batch']}, "
                    f"{fit['scale']:.2f} times the usual target cells)")
        elif state.get("batch_size") and not filling:
            self.cfg.batch_size = int(state["batch_size"])

    def _state(self) -> dict:
        return {"stage": self.stage, "step": self.step, "elapsed": self.elapsed, "best": self.best,
                "batch_size": self.cfg.batch_size, "fit": self.fit, "history": self.history[-200:], "saved": time.time()}

    def save(self, push: bool = True) -> None:
        save_checkpoint(self.out, self.model, self.loss_fn, self.optimizer, self._state(),
                        extra_payload={"ema": self.ema.state_dict()} if self.ema is not None else None)
        if push and self.mirror is not None:
            self.mirror.push(self.out / CHECKPOINT, f"{self.mirror_prefix}/{CHECKPOINT}", what="checkpoint")
            self.mirror.push(self.out / META, f"{self.mirror_prefix}/{META}", what="checkpoint metadata")

    # ------------------------------------------------------------------------------ batch size --
    def pick_batch_size(self, candidates=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32)) -> int:
        """The batch for this run: the one given, or one found by measuring. With ``cfg.memory == "fill"`` the step
        is sized to the GPU's memory (:meth:`_fill_memory`); with ``"fastest"`` the fastest batch per forecast is
        kept, which is what every run did before 0.4.3 and leaves most of a large card empty."""
        if self.cfg.batch_size > 0:
            return self.cfg.batch_size
        if self.meter is None:
            self.cfg.batch_size = 2
            return 2
        if self.cfg.memory == "fill":
            try:
                return self._fill_memory()
            except Exception as error:                        # sizing must never be what stops a run
                log(f"sizing to the GPU's memory did not work ({type(error).__name__}: {error}); "
                    "keeping the fastest batch instead")
                self._apply(1.0, self.cfg.checkpointing)
        if self.device.type != "cuda":
            self.cfg.batch_size = 2
            return 2
        return self._fastest_batch(candidates)

    def _apply(self, scale: float, checkpointing: bool) -> None:
        """Set how many target cells each training forecast is scored at (``scale`` times the usual number, as far
        as the area has them) and whether activations are recomputed to save memory."""
        land, sea = self._usual_cells
        if scale > 1.0:
            land = int(min(max(self.builder.land_idx.size, land), round(land * scale)))
            sea = int(round(sea * scale))
        self.builder.max_land, self.builder.ocean_targets = land, sea
        self.model.gradient_checkpointing_enable(bool(checkpointing))
        self.scale = float(scale)

    def _adopt(self, batch: int, scale: float, checkpointing: bool, fit: dict | None = None) -> None:
        """Train at this size from now on. The learning rate follows the batch (:func:`pluvium.training.fit.lr_scale`)."""
        from .training.fit import lr_scale

        self._apply(scale, checkpointing)
        self.cfg.batch_size = int(batch)
        self.lr_scale = lr_scale(int(batch), self.cfg.lr_reference_batch, self.cfg.lr_scale_cap)
        self.fit = None if fit is None else {**fit, "batch": int(batch), "scale": float(scale),
                                             "checkpointing": bool(checkpointing)}

    def _optimizer_bytes(self) -> float:
        """Memory AdamW will take at its first step and has not taken yet: two moments for every parameter."""
        if len(self.optimizer.state) > 0:
            return 0.0
        return 2.0 * sum(p.numel() * p.element_size() for g in self.optimizer.param_groups for p in g["params"])

    def _measure(self, batch: int, scale: float, checkpointing: bool):
        """A few training steps at one setting, on one forecast repeated: the memory they needed and the forecasts
        per second (:class:`pluvium.training.fit.Probe`). Running out of memory is an answer, not an error."""
        from .training.fit import Probe

        self._apply(scale, checkpointing)
        key = round(float(scale), 3)
        if key not in self._probe_examples:
            targets = self.builder.choose_targets(True)
            self._probe_examples[key] = self.builder.example(int(self.train_starts[0]), True, targets)
        example = self._probe_examples[key]
        moved = None
        try:
            self.meter.start(batch, scale, checkpointing)
            moved = to_device(collate([example] * batch), self.device)
            began = time.time()
            for k in range(self.cfg.probe_steps + 1):
                if k == 1:                                    # the first step warms up and is not timed
                    self.meter.sync()
                    began = time.time()
                self._forward_backward(moved)
                self.optimizer.zero_grad(set_to_none=True)
            self.meter.sync()
            rate = self.cfg.probe_steps * batch / max(time.time() - began, 1e-9)
            return Probe(batch, scale, checkpointing, self.meter.peak() + self._optimizer_bytes(), rate)
        except torch.OutOfMemoryError:
            return Probe(batch, scale, checkpointing, None)
        finally:
            del moved
            self.optimizer.zero_grad(set_to_none=True)
            self.meter.release()

    def _fill_memory(self) -> int:
        """Size the step to ``cfg.memory_share`` of the GPU's memory: gradient checkpointing on or off, the largest
        batch, and more target cells per forecast with what is left (:func:`pluvium.training.fit.fill`)."""
        from .training.fit import fill

        usable = self.meter.usable()
        budget = self.cfg.memory_share * usable
        max_scale = max(1.0, self.builder.land_idx.size / max(self._usual_cells[0], 1))
        log(f"sizing the training step to {self.cfg.memory_share:.0%} of the GPU's {usable / 2**30:.0f} GB "
            "(a few steps at each setting) ...")
        weights = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        try:
            fit = fill(self._measure, budget, max_scale=max_scale, say=log)
        finally:
            self.model.load_state_dict(weights)
            self._probe_examples.clear()
            self.optimizer.zero_grad(set_to_none=True)
            self.meter.release()
        self._adopt(fit.batch, fit.scale, fit.checkpointing,
                    {**fit.to_json(), "usable": usable, "share": self.cfg.memory_share, "policy": "fill"})
        log(f"step size: {fit.describe()}; learning rate x{self.lr_scale:.2f}")
        return fit.batch

    def _fastest_batch(self, candidates=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32)) -> int:
        """Run two training steps at each size on one repeated example; keep the fastest per forecast that fits
        in 90% of the GPU's memory."""
        example = self.builder.example(int(self.train_starts[0]), True, self.builder.choose_targets(True))
        total = torch.cuda.get_device_properties(self.device).total_memory
        best, best_rate = 1, 0.0
        state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        for b in candidates:
            batch = to_device(collate([example] * b), self.device)
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.synchronize()
                for k in range(3):
                    if k == 1:
                        torch.cuda.synchronize()
                        began = time.time()
                    self._forward_backward(batch)
                    self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                rate = 2 * b / (time.time() - began)
                peak = torch.cuda.max_memory_allocated(self.device)
                log(f"  batch {b:>2}: {rate:.2f} forecasts/s, peak memory {peak / 2**30:.1f} GB")
                if peak > 0.9 * total:
                    break
                if rate > best_rate * 1.03:
                    best, best_rate = b, rate
            except torch.OutOfMemoryError:
                log(f"  batch {b:>2}: out of memory")
                break
            finally:
                del batch
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
        self.model.load_state_dict(state)
        self.cfg.batch_size = best
        log(f"batch size {best} ({best_rate:.2f} forecasts/s)")
        return best

    # ---------------------------------------------------------------------------------- one step --
    def _forward_backward(self, batch: dict) -> dict:
        self.model.train()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.cfg.bf16 and self.device.type == "cuda"):
            out = self.model(batch)
        rows = batch["targets"][0]
        loss, metrics = self.loss_fn(out, batch, self.elev[rows])
        if "weight" in batch:                                 # importance weights of the rare-event curriculum
            loss = loss * batch["weight"].float().mean()
        experts = getattr(self.model, "experts", None)
        if experts is not None and torch.is_tensor(experts.balance) and experts.balance.requires_grad:
            loss = loss + self.cfg.expert_balance * experts.balance
            metrics["expert_balance"] = float(experts.balance.detach())
        loss.backward()
        return metrics

    def train_step(self, batch: dict) -> dict:
        factor = lr_factor(self.step, self.elapsed, self.cfg) * self.lr_scale
        for group in self.optimizer.param_groups:
            group["lr"] = self.cfg.lr * factor
        metrics = self._forward_backward(batch)
        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        norm = torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
        if torch.isfinite(norm):
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update(self.model)
        else:
            metrics["skipped"] = 1.0
        self.optimizer.zero_grad(set_to_none=True)
        metrics["grad_norm"] = float(norm)
        metrics["lr"] = self.cfg.lr * factor
        return metrics

    # ---------------------------------------------------------------------------------- evaluate --
    def _yardstick_targets(self) -> np.ndarray:
        """The cells held-out forecasts are scored at: the usual number, whatever training is sized to, so that
        scores from runs of different sizes can be compared."""
        land, sea = self.builder.max_land, self.builder.ocean_targets
        self.builder.max_land, self.builder.ocean_targets = self._usual_cells
        try:
            return self.builder.choose_targets(train=False)
        finally:
            self.builder.max_land, self.builder.ocean_targets = land, sea

    @torch.no_grad()
    def evaluate(self, starts=None, batch_size: int | None = None) -> dict:
        """Mean loss terms and Brier score over held-out forecasts (no emulated errors: a fixed yardstick)."""
        starts = self.val_starts[: self.cfg.eval_forecasts] if starts is None else np.asarray(starts)
        if starts.size == 0:
            return {}
        self.model.eval()
        b = batch_size or max(1, self.cfg.batch_size)
        targets = self._yardstick_targets()
        while True:
            try:
                return self._evaluate(starts, targets, b)
            except torch.OutOfMemoryError:
                if b == 1:
                    raise
                b = max(1, b // 2)
                if self.meter is not None:
                    self.meter.release()
                log(f"held out: out of memory; scoring {b} forecasts at a time")

    def _evaluate(self, starts: np.ndarray, targets: np.ndarray, b: int) -> dict:
        sums, n = {}, 0
        for k in range(0, starts.size, b):
            chunk = starts[k: k + b]
            batch = to_device(collate([self.builder.example(int(t), False, targets, emulate=False) for t in chunk]),
                              self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                enabled=self.cfg.bf16 and self.device.type == "cuda"):
                out = self.model(batch)
            _, metrics = self.loss_fn(out, batch, self.elev[batch["targets"][0]])
            for key, value in metrics.items():                # weighed by forecasts, so the mean does not depend
                sums[key] = sums.get(key, 0.0) + value * chunk.size        # on how many are scored at a time
            n += chunk.size
        return {k: v / n for k, v in sums.items()}

    # ---------------------------------------------------------------------------------------- run --
    def _batch_bytes(self) -> float:
        """Bytes of one batch as the loader hands it over."""
        example = self.builder.example(int(self.train_starts[0]), True, self.builder.choose_targets(True))
        return float(self.cfg.batch_size * sum(np.asarray(v).nbytes for v in example.values()))

    def _loader(self, sampler) -> tuple:
        """The batches, built on the CPU cores: ``(loader, workers, batches each worker keeps ready)``. How many
        batches wait in memory is kept within the machine's RAM (:func:`pluvium.training.fit.loader_plan`): a batch
        sized to a large GPU is large in RAM too."""
        from .training.fit import loader_plan, ram_bytes

        cfg = self.cfg
        workers, ready = _workers(cfg.workers), 4
        if workers:
            workers, ready = loader_plan(workers, self._batch_bytes(), ram_bytes())
        data = BatchSet(self.builder, self.train_starts, cfg.batch_size, cfg.seed, sampler)
        loader = torch.utils.data.DataLoader(
            data, batch_size=None, sampler=range(self.step, cfg.max_steps), num_workers=workers,
            pin_memory=self.device.type == "cuda", prefetch_factor=ready if workers else None,
            persistent_workers=False, multiprocessing_context="fork" if workers else None)
        return loader, workers, ready

    def _memory_text(self) -> str:
        """``VRAM used/total`` for the log, as ``nvidia-smi`` would show it; empty without a GPU."""
        if self.meter is None:
            return ""
        try:
            used, total = self.meter.used()
        except Exception:
            return ""
        return f"  VRAM {used / 2**30:.1f}/{total / 2**30:.1f} GB"

    def _shrink(self, times: int) -> bool:
        """A training step ran out of memory: make the step smaller, so that the run carries on. The extra target
        cells go first, then forecasts from the batch. False when there is nothing left to give up."""
        self.optimizer.zero_grad(set_to_none=True)
        if self.meter is not None:
            self.meter.release()
        batch, scale = self.cfg.batch_size, self.scale
        if times > 8 or (batch <= 1 and scale <= 1.0):
            return False
        if scale > 1.0:
            scale = 1.0 + 0.6 * (scale - 1.0) if scale > 1.1 else 1.0
        else:
            batch = max(1, min(batch - 1, int(batch * 0.9)))
        log(f"stage {self.stage}: a step ran out of memory at batch {self.cfg.batch_size} and {self.scale:.2f} times "
            f"the usual target cells; carrying on at batch {batch} and {scale:.2f} times")
        self._adopt(batch, scale, bool(self.model.gradient_checkpointing), self.fit)
        return True

    def run(self) -> dict:
        cfg = self.cfg
        budget = cfg.hours * 3600.0
        if self.elapsed >= budget or self.step >= cfg.max_steps:
            log(f"stage {self.stage}: budget already spent ({self.elapsed / 3600:.2f} of {cfg.hours:.2f} h)")
            if self.ema is not None and self.ema.updates:
                self.model.load_state_dict({**self.model.state_dict(), **{
                    k: v.to(self.model.state_dict()[k].dtype) for k, v in self.ema.shadow.items()}})
            return {"step": self.step, "elapsed": self.elapsed, "history": self.history}
        self.pick_batch_size()
        sampler = None
        if cfg.curriculum:
            from .training import RareEventSampler

            sampler = RareEventSampler.from_truth(self.train_starts, lambda h: self.builder.cmorph.targets(h))
        ran_out = 0
        try:
            while True:
                try:
                    self._train(sampler, budget)
                    break
                except torch.OutOfMemoryError:
                    ran_out += 1
                    if not self._shrink(ran_out):
                        raise
        finally:                                       # whoever uses the builder next gets its usual target cells
            self.builder.max_land, self.builder.ocean_targets = self._usual_cells
        self._eval_and_keep()
        self.save()
        if self.ema is not None:                       # the averaged weights are the ones released and scored
            self.model.load_state_dict({**self.model.state_dict(),
                                        **{k: v.to(self.model.state_dict()[k].dtype) for k, v in self.ema.shadow.items()}})
        if self.mirror is not None:
            self.mirror.wait()
        return {"step": self.step, "elapsed": self.elapsed, "best": self.best, "history": self.history}

    def _train(self, sampler, budget: float) -> None:
        """Train until the time budget or the step limit is reached. A step that runs out of memory raises, and
        :meth:`run` starts this again at a smaller size."""
        cfg = self.cfg
        if self.meter is not None:                     # after a step ran out of memory, what it held is free by now
            self.meter.release()
        loader, workers, ready = self._loader(sampler)
        log(f"stage {self.stage}: {self.model.config.name} ({self.model.num_parameters() / 1e6:.1f}M parameters), "
            f"batch {cfg.batch_size}, {self.builder.max_land:,} land and {self.builder.ocean_targets:,} sea cells a "
            f"forecast at most, gradient checkpointing {'on' if self.model.gradient_checkpointing else 'off'}, "
            f"learning rate x{self.lr_scale:.2f}, {workers} data workers each keeping {ready} batches ready, "
            f"{len(self.train_starts):,} issue times, {cfg.hours - self.elapsed / 3600:.2f} h left of {cfg.hours:.2f} h")
        last_save = last_eval = last_log = time.time()
        window, waited, began = [], 0.0, time.time()
        t_prev, first = time.time(), True
        for batch in loader:
            got = time.time()
            waited += got - t_prev
            batch = to_device(batch, self.device)
            metrics = self.train_step(batch)
            now = time.time()
            self.elapsed += now - t_prev
            t_prev = now
            self.step += 1
            window.append(metrics)
            # every ``log_every`` steps, and also after the first step and at least every five minutes: a step sized
            # to a large GPU can take a while, and a silent log looks like a program that has stopped
            if self.step % cfg.log_every == 0 or first or now - last_log > 300:
                first, last_log = False, now
                mean = {k: float(np.mean([m.get(k, np.nan) for m in window])) for k in window[-1]}
                rate = len(window) * cfg.batch_size / max(now - began, 1e-6)
                log(f"step {self.step:,}  loss {mean['loss']:.4f}  timing {mean['timing_nll']:.4f}  "
                    f"amount {mean.get('amount_nll', float('nan')):.3f}  brier {mean['brier']:.4f}  "
                    f"lr {mean['lr']:.2e}  {rate:.2f} forecasts/s  waiting for data {waited / max(now - began, 1e-6):.0%}"
                    f"{self._memory_text()}  {self.elapsed / 3600:.2f}/{cfg.hours:.2f} h")
                self.history.append({"step": self.step, "elapsed": self.elapsed, **{k: mean[k] for k in
                                     ("loss", "timing_nll", "brier") if k in mean}})
                window, waited, began = [], 0.0, time.time()
            if now - last_eval > cfg.eval_every_minutes * 60:      # (evaluation and saving count against the budget)
                self._eval_and_keep()
                last_eval = time.time()
            if now - last_save > cfg.save_every_minutes * 60:
                self.save()
                last_save = time.time()
            if self.elapsed >= budget or self.step >= cfg.max_steps:
                break

    def _eval_and_keep(self) -> None:
        if self.val_starts.size == 0:
            return
        metrics = self.evaluate()
        if not metrics:
            return
        log(f"held out: loss {metrics['loss']:.4f}  timing {metrics['timing_nll']:.4f}  brier {metrics['brier']:.4f}"
            f"  (wet {metrics['wet_rate']:.1%})")
        self.history.append({"step": self.step, "elapsed": self.elapsed, "val_loss": metrics["loss"],
                             "val_brier": metrics["brier"]})
        if metrics["loss"] < self.best:
            self.best = metrics["loss"]
            best_dir = self.out / "best"
            save_checkpoint(best_dir, self.model, self.loss_fn, None, self._state())
            if self.mirror is not None:
                self.mirror.push(best_dir / CHECKPOINT, f"{self.mirror_prefix}/best/{CHECKPOINT}", what="best weights")
                self.mirror.push(best_dir / META, f"{self.mirror_prefix}/best/{META}", what="best metadata")


# -------------------------------------------------------------------------------------- hindcasting --

@torch.no_grad()
def hindcast(model: PluviumModel, builder: SampleBuilder, starts, batch_size: int = 2, device=None,
             emulate: bool = False, calibrator: Calibrator | None = None, samples: int = 100,
             max_points: int | None = None, stations: bool = False) -> tuple[Scoreboard, dict]:
    """
    Run the model on held-out issue times and score it (:class:`pluvium.verify.Scoreboard`). Also returns the pooled
    raw chances and outcomes that :meth:`pluvium.calibrate.Calibrator.fit` needs, and (``stations=True``) the
    airports' thunder chances and reports.

    ``emulate=True`` gives the driving fields forecast-like errors (a fairer stand-in for live skill than the
    reanalysis itself, which knows where the rain went).
    """
    from .products import _window_any, component

    device = torch.device(device or next(model.parameters()).device)
    model.eval()
    sch = builder.schedule
    hour_frame, _ = sch.hour_frame()
    targets = builder.choose_targets(train=False)
    if max_points is not None and targets.size > max_points:
        targets = targets[np.linspace(0, targets.size - 1, max_points).astype(int)]
    lon = builder.geo.targets.lon[targets]
    clim = borrowed_climatology(builder.clim, targets)
    board = Scoreboard()
    pool = {"p": [], "wet": [], "known": [], "win_p": {h: [] for h in WINDOWS}, "win_y": {h: [] for h in WINDOWS},
            "p_moderate": [], "y_moderate": [], "p_heavy": [], "y_heavy": [], "driving": [], "clim_wet": [],
            "thunder_p": [], "thunder_obs": [], "thunder_known": [], "station_hours": []}
    starts = np.asarray(starts)
    for k in range(0, starts.size, batch_size):
        chunk = starts[k: k + batch_size]
        exs = [builder.example(int(t), False, targets, emulate=emulate) for t in chunk]
        batch = to_device(collate(exs), device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(batch)
        for j, t0 in enumerate(chunk):
            s = summarize(out, int(t0), batch_index=j, samples=samples, calibrator=calibrator, keep_samples=True)
            truth = exs[j]["truth"].T                                           # (P, T)
            hours = int(t0) + np.arange(sch.num_hours)
            clim_w, clim_r = clim.at(hours, lon)                                # (T, P)
            clim_cls = np.moveaxis(clim.classes_at(hours, lon), 0, 1)           # (P, T, 4)
            last = builder.cmorph.targets(np.array([int(t0) - 1]), targets)[0]
            driving = np.expm1(exs[j]["target_ctx"][hour_frame, :, 8]).T        # (P, T) driving rain mm/h
            board.add(s["p_wet"], truth, clim_w.T, last, driving, s["any_rain"], s["start_median_24"],
                      s.get("samples24"), clim_r.T, s["classes"], clim_cls)
            known = np.isfinite(truth)
            comp = component(out, "", j)
            k_states = comp["states"]
            wet_target = np.arange(k_states) > 0
            from .chain import class_probs, marginals_np

            ps = marginals_np(comp["a"], comp["pi"])
            raw = class_probs(comp, k_states, ps)                                # (P, T, 4)
            pool["p"].append((1 - ps[..., 0]).astype(np.float32))
            pool["wet"].append((np.nan_to_num(truth) >= 0.2).astype(np.float32))
            pool["p_moderate"].append((raw[..., 2] + raw[..., 3]).astype(np.float32))
            pool["y_moderate"].append((np.nan_to_num(truth) >= 2.5).astype(np.float32))
            pool["p_heavy"].append(raw[..., 3].astype(np.float32))
            pool["driving"].append(driving.astype(np.float32))
            pool["clim_wet"].append(clim_w.T.astype(np.float32))
            pool["y_heavy"].append((np.nan_to_num(truth) >= 7.6).astype(np.float32))
            pool["known"].append(known)
            for h in WINDOWS:
                ok = known[:, :h].all(1)
                pool["win_p"][h].append(_window_any(comp["a"], comp["pi"], wet_target, h).astype(np.float32))
                pool["win_y"][h].append(np.where(ok, (np.nan_to_num(truth[:, :h]) >= 0.2).any(1), np.nan))
            if stations and "station_thunder" in out:
                from .weather import decode

                d = decode(exs[j]["station_truth"])                              # (T, S)
                pool["thunder_p"].append(torch.sigmoid(out["station_thunder"][j]).float().cpu().numpy())
                pool["thunder_obs"].append(d["thunder"])
                pool["thunder_known"].append(d["known"])
                pool["station_hours"].append(hours)
    data = {"p": np.concatenate(pool["p"]), "wet": np.concatenate(pool["wet"]), "known": np.concatenate(pool["known"]),
            **{k: np.concatenate(pool[k]) for k in ("p_moderate", "y_moderate", "p_heavy", "y_heavy", "driving",
                                                    "clim_wet")},
            "win_p": {h: np.concatenate(v) for h, v in pool["win_p"].items()},
            "win_y": {h: np.concatenate(v) for h, v in pool["win_y"].items()}}
    if pool["thunder_p"]:
        data.update(thunder_p=np.stack(pool["thunder_p"]), thunder_obs=np.stack(pool["thunder_obs"]),
                    thunder_known=np.stack(pool["thunder_known"]), station_hours=np.stack(pool["station_hours"]))
    return board, data


def fit_calibration(data: dict) -> Calibrator:
    exceed_p = {"moderate": data["p_moderate"], "heavy": data["p_heavy"]} if "p_heavy" in data else None
    exceed_y = {"moderate": data["y_moderate"], "heavy": data["y_heavy"]} if "p_heavy" in data else None
    return Calibrator.fit(data["p"], data["wet"], data["known"], data["win_p"], data["win_y"], exceed_p, exceed_y)


__all__ = ["BatchSet", "CudaMeter", "TrainConfig", "Trainer", "borrowed_climatology", "build_model", "fit_calibration",
           "grids_of", "hindcast", "load_model", "lr_factor", "nearest_target", "read_meta", "save_checkpoint",
           "station_points", "to_device"]
