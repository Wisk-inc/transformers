# Copyright 2026 Nathan. Apache-2.0.
"""
Training on one GPU: batches built on every CPU core, bf16 on the GPU, the fastest batch size found by measuring,
a learning-rate schedule that ends when the time budget does, checkpoints that resume exactly, and (optionally) a
mirror of every checkpoint on the Hugging Face Hub so a rented machine that stops loses nothing.

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
                 device=None) -> None:
        self.model, self.loss_fn, self.builder = model, loss, builder
        self.train_starts, self.val_starts = np.asarray(train_starts), np.asarray(val_starts)
        self.out, self.cfg, self.mirror, self.mirror_prefix, self.stage = Path(out_dir), cfg, mirror, mirror_prefix, stage
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.loss_fn.to(self.device)
        if cfg.checkpointing:
            self.model.gradient_checkpointing_enable()
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
            if state.get("batch_size"):
                self.cfg.batch_size = int(state["batch_size"])
            log(f"stage {self.stage}: continuing from step {self.step:,} ({self.elapsed / 3600:.2f} h trained)")
        else:
            log(f"stage {self.stage}: starting from the stage {state.get('stage', '?')} weights")

    def _state(self) -> dict:
        return {"stage": self.stage, "step": self.step, "elapsed": self.elapsed, "best": self.best,
                "batch_size": self.cfg.batch_size, "history": self.history[-200:], "saved": time.time()}

    def save(self, push: bool = True) -> None:
        save_checkpoint(self.out, self.model, self.loss_fn, self.optimizer, self._state(),
                        extra_payload={"ema": self.ema.state_dict()} if self.ema is not None else None)
        if push and self.mirror is not None:
            self.mirror.push(self.out / CHECKPOINT, f"{self.mirror_prefix}/{CHECKPOINT}", what="checkpoint")
            self.mirror.push(self.out / META, f"{self.mirror_prefix}/{META}", what="checkpoint metadata")

    # ------------------------------------------------------------------------------ batch size --
    def pick_batch_size(self, candidates=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32)) -> int:
        """Run two training steps at each size on one repeated example; keep the fastest per forecast that fits
        in 90% of the GPU's memory."""
        if self.cfg.batch_size > 0:
            return self.cfg.batch_size
        if self.device.type != "cuda":
            self.cfg.batch_size = 2
            return 2
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
        factor = lr_factor(self.step, self.elapsed, self.cfg)
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
    @torch.no_grad()
    def evaluate(self, starts=None, batch_size: int | None = None) -> dict:
        """Mean loss terms and Brier score over held-out forecasts (no emulated errors: a fixed yardstick)."""
        starts = self.val_starts[: self.cfg.eval_forecasts] if starts is None else np.asarray(starts)
        if starts.size == 0:
            return {}
        self.model.eval()
        b = batch_size or max(1, self.cfg.batch_size)
        targets = self.builder.choose_targets(train=False)
        sums, n = {}, 0
        for k in range(0, starts.size, b):
            batch = to_device(collate([self.builder.example(int(t), False, targets, emulate=False)
                                       for t in starts[k: k + b]]), self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                enabled=self.cfg.bf16 and self.device.type == "cuda"):
                out = self.model(batch)
            _, metrics = self.loss_fn(out, batch, self.elev[batch["targets"][0]])
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + value
            n += 1
        return {k: v / n for k, v in sums.items()}

    # ---------------------------------------------------------------------------------------- run --
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
        workers = _workers(cfg.workers)
        sampler = None
        if cfg.curriculum:
            from .training import RareEventSampler

            sampler = RareEventSampler.from_truth(self.train_starts, lambda h: self.builder.cmorph.targets(h))
        data = BatchSet(self.builder, self.train_starts, cfg.batch_size, cfg.seed, sampler)
        loader = torch.utils.data.DataLoader(
            data, batch_size=None, sampler=range(self.step, cfg.max_steps), num_workers=workers,
            pin_memory=self.device.type == "cuda", prefetch_factor=4 if workers else None,
            persistent_workers=False, multiprocessing_context="fork" if workers else None)
        log(f"stage {self.stage}: {self.model.config.name} ({self.model.num_parameters() / 1e6:.1f}M parameters), "
            f"batch {cfg.batch_size}, {workers} data workers, {len(self.train_starts):,} issue times, "
            f"{cfg.hours - self.elapsed / 3600:.2f} h left of {cfg.hours:.2f} h")
        last_save = last_eval = time.time()
        window, waited, began = [], 0.0, time.time()
        t_prev = time.time()
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
            if self.step % cfg.log_every == 0:
                mean = {k: float(np.mean([m.get(k, np.nan) for m in window])) for k in window[-1]}
                rate = len(window) * cfg.batch_size / max(now - began, 1e-6)
                log(f"step {self.step:,}  loss {mean['loss']:.4f}  timing {mean['timing_nll']:.4f}  "
                    f"amount {mean.get('amount_nll', float('nan')):.3f}  brier {mean['brier']:.4f}  "
                    f"lr {mean['lr']:.2e}  {rate:.2f} forecasts/s  waiting for data {waited / max(now - began, 1e-6):.0%}"
                    f"  {self.elapsed / 3600:.2f}/{cfg.hours:.2f} h")
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
        self._eval_and_keep()
        self.save()
        if self.ema is not None:                       # the averaged weights are the ones released and scored
            self.model.load_state_dict({**self.model.state_dict(),
                                        **{k: v.to(self.model.state_dict()[k].dtype) for k, v in self.ema.shadow.items()}})
        if self.mirror is not None:
            self.mirror.wait()
        return {"step": self.step, "elapsed": self.elapsed, "best": self.best, "history": self.history}

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
            "p_moderate": [], "y_moderate": [], "p_heavy": [], "y_heavy": [],
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
            **{k: np.concatenate(pool[k]) for k in ("p_moderate", "y_moderate", "p_heavy", "y_heavy")},
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


__all__ = ["BatchSet", "TrainConfig", "Trainer", "borrowed_climatology", "build_model", "fit_calibration",
           "grids_of", "hindcast", "load_model", "lr_factor", "nearest_target", "read_meta", "save_checkpoint",
           "station_points", "to_device"]
