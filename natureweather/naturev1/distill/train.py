# Copyright 2026 Nathan. Apache-2.0.
"""
Distillation: NatureV1 rolls forward on its own forecasts and is scored, step by step, against the teachers.

The loss at every 6-hour step has three parts:

* **teacher term** -- the expected Gaussian negative log-likelihood of the teacher mixture under NatureV1's
  predicted distribution: with mixture mean ``mu_T`` and variance ``s_T^2`` (:mod:`naturev1.distill.targets`),
  ``0.5 [log(s^2 + s_T^2) + (mu_T - mu)^2 / (s^2 + s_T^2)]`` -- a soft target that is sharp where the teachers
  agree and forgiving where they do not; computed in NatureV1's residual space (each channel's tendency spread),
  so it is the same quantity the model already predicts;
* **truth anchor** -- the ordinary rollout likelihood against ERA5 (when the valid time is in the staged
  record), weight ``alpha``: teachers are good, not perfect, and the record keeps NatureV1 from inheriting their
  shared biases (all ML teachers were trained on ERA5 too, but each drifts its own way);
* **hurricane focus** -- grid cells within ``storm_radius_km`` of an active tropical cyclone (from the best track
  at the initial time and along its known track) are up-weighted by ``1 + storm_boost``, the rest scaled so the
  mean weight stays one: the teachers' best skill is in large-scale steering, and that is exactly what the
  storm-relevant cells carry.

Between steps the trajectory is detached (truncated back-propagation), so a long horizon costs memory linearly
and a curriculum (:func:`horizon_for`) grows it from one step to the full horizon over the run.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .stage import StagedTeacher
from .targets import combine


def storm_weight_map(grid_lat: np.ndarray, grid_lon: np.ndarray, centres, radius_km: float = 800.0,
                     boost: float = 3.0) -> np.ndarray:
    """``(P,)`` weights: ``1 + boost`` near any of ``centres`` (lat, lon), renormalised to mean one."""
    from ..cyclone.geometry import great_circle_km

    w = np.ones(grid_lat.size)
    for lat, lon in centres:
        near = great_circle_km(lat, lon, grid_lat, grid_lon) < radius_km
        w = np.where(near, 1.0 + boost, w)
    return w / w.mean()


def horizon_for(step: int, total_steps: int, max_horizon: int) -> int:
    """Curriculum: 1 step for the first 10% of training, linear to ``max_horizon`` by 60%, then constant."""
    frac = step / max(total_steps, 1)
    if frac < 0.1:
        return 1
    return int(min(max_horizon, 1 + math.floor((frac - 0.1) / 0.5 * (max_horizon - 1) + 1e-9)))


class DistillDataset(torch.utils.data.Dataset):
    """
    One initial time: the ERA5 analysis window, ERA5 truth at each rollout step (when staged), and the teacher
    mixture at each step a teacher covers. ``teachers``: list of staged teacher directories.
    """

    def __init__(self, era5: str | Path, teachers: list, inits, channels: int, horizon: int = 12, history: int = 2,
                 step_hours: int = 6, weights: dict | None = None, storms=None, storm_radius_km: float = 800.0,
                 storm_boost: float = 3.0) -> None:
        from ..era5 import Normalizer

        self.path = Path(era5)
        meta = json.loads(self.path.with_suffix(".json").read_text())
        self.normalizer = Normalizer.from_dict(meta["statistics"])
        self.times = np.asarray(meta["time"], np.int64)
        self.index = {int(t): i for i, t in enumerate(self.times)}
        self.channels, self.horizon, self.history, self.step_hours = channels, horizon, history, step_hours
        self.teachers = [StagedTeacher(d) for d in teachers]
        self.names = [t.meta["teacher"] for t in self.teachers]
        self.weights = weights
        self.inits = [np.datetime64(t, "s") for t in inits
                      if int(np.datetime64(t, "s").astype(np.int64)) in self.index
                      and self.index[int(np.datetime64(t, "s").astype(np.int64))] >= history - 1]
        lat = np.asarray(meta.get("latitude") or np.linspace(-90, 90, 121), np.float64)
        lon = np.asarray(meta.get("longitude") or np.arange(240) * 1.5, np.float64)
        lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
        self.grid_lat, self.grid_lon = lat_g.ravel(), lon_g.ravel()
        self.storms = storms or {}
        self.storm_radius_km, self.storm_boost = storm_radius_km, storm_boost
        self._values = None

    def __len__(self) -> int:
        return len(self.inits)

    def _memmap(self):
        if self._values is None:
            self._values = np.load(self.path, mmap_mode="r")
        return self._values

    def __getitem__(self, item: int) -> dict:
        from ..model import calendar_features
        from ..satdata import prepared_to_state

        when = self.inits[item]
        t0 = int(when.astype(np.int64))
        i0 = self.index[t0]
        values = self._memmap()
        window = [prepared_to_state(values[i0 - self.history + 1 + k], self.normalizer, self.channels)
                  for k in range(self.history)]
        stamps = torch.tensor(self.times[i0 - self.history + 1: i0 + 1], dtype=torch.float64)
        points, width = window[0].shape
        cvars = len(self.normalizer.variables)
        truth = torch.zeros(self.horizon, points, width)
        truth_ok = torch.zeros(self.horizon, dtype=torch.bool)
        for s in range(self.horizon):
            j = self.index.get(t0 + (s + 1) * self.step_hours * 3600)
            if j is not None:
                truth[s] = prepared_to_state(values[j], self.normalizer, self.channels)
                truth_ok[s] = True
        means, variances = {}, {}
        for name, teacher in zip(self.names, self.teachers):
            got = teacher.get(when)
            if got is None:
                continue
            m, v = got
            leads = teacher.leads
            mm = np.full((self.horizon, points, cvars), np.nan, np.float32)
            vv = np.full((self.horizon, points, cvars), np.nan, np.float32)
            for s in range(self.horizon):
                lead = (s + 1) * self.step_hours
                if lead in leads:
                    k = leads.index(lead)
                    mm[s] = m[k].reshape(points, cvars)
                    if v is not None:
                        vv[s] = v[k].reshape(points, cvars)
            means[name] = mm
            if v is not None:
                variances[name] = vv
        if means:
            w = None
            if self.weights is not None:
                w = {n: np.asarray(self.weights[n])[: self.horizon] for n in means if n in self.weights}
                w = w if len(w) == len(means) else None
            mean, var, covered = combine(means, variances, w)
        else:
            mean = np.full((self.horizon, points, cvars), np.nan, np.float32)
            var, covered = np.zeros_like(mean), np.zeros(mean.shape, bool)
        centres = self.storms.get(t0, [])
        weight = storm_weight_map(self.grid_lat, self.grid_lon, centres, self.storm_radius_km, self.storm_boost) if centres else np.ones(points)
        return {"analysis": torch.stack(window), "calendar": calendar_features(stamps), "valid_time": stamps[-1].clone(),
                "state_target": truth, "truth_ok": truth_ok,
                "teacher_mean": torch.from_numpy(np.nan_to_num(mean).astype(np.float32)),
                "teacher_var": torch.from_numpy(np.nan_to_num(var).astype(np.float32)),
                "teacher_covered": torch.from_numpy(covered), "storm_weight": torch.from_numpy(weight.astype(np.float32))}


def distill_rollout_loss(model, batch: dict, stepper, horizon: int, alpha: float = 0.3,
                         channel_weights: torch.Tensor | None = None, precision: str = "bf16") -> tuple[torch.Tensor, dict]:
    """The teacher + truth rollout loss of one batch (see the module docstring)."""
    from ..state import _advance_calendar, _autocast

    history = batch["analysis"]
    calendar, valid_time = batch["calendar"], batch["valid_time"]
    prognostic = stepper.roles.prognostic
    cvars = batch["teacher_mean"].shape[-1]
    prog_vars = [c for c in prognostic if c < cvars]
    slot = [prognostic.index(c) for c in prog_vars]
    scale = stepper.tendency[prog_vars].to(history.device, torch.float32)
    area = stepper.grid.weights.to(history.device, torch.float32)
    area = (area / area.mean()).view(1, -1, 1)
    weights = area * batch["storm_weight"].to(history.device, torch.float32).view(history.shape[0], -1, 1)
    if channel_weights is not None:
        weights = weights * channel_weights[prog_vars].to(history.device, torch.float32).view(1, 1, -1)
    total = torch.zeros((), device=history.device)
    parts: dict[str, float] = {}
    horizon = max(1, min(horizon, batch["teacher_mean"].shape[1]))
    for step in range(horizon):
        current = history[:, -1].float()
        with _autocast(history.device, precision):
            out = model(analysis=history, analysis_grid=stepper.grid, calendar=calendar, output_grid=stepper.grid)
        mean = out["state_delta"].float()[..., slot]
        log_var = out["state_delta_log_var"].float()[..., slot]
        var = log_var.exp()
        target = (batch["teacher_mean"][:, step][..., prog_vars].float() - current[..., prog_vars]) / scale
        t_var = batch["teacher_var"][:, step][..., prog_vars].float() / scale ** 2
        covered = batch["teacher_covered"][:, step][..., prog_vars].float()
        total_var = var + t_var
        teacher_nll = 0.5 * (total_var.log() + (target - mean) ** 2 / total_var)
        teacher_term = (teacher_nll * weights * covered).sum() / (weights * covered).sum().clamp_min(1.0)
        loss = (1.0 - alpha) * teacher_term
        parts[f"teacher_{step + 1}"] = float(teacher_term.detach())
        ok = batch["truth_ok"][:, step].float().view(-1, 1, 1)
        if alpha > 0 and bool(ok.any()):
            truth = stepper.to_state_delta(history[:, -1], batch["state_target"][:, step])[..., slot].float()
            truth_nll = 0.5 * (log_var + (truth - mean) ** 2 / var)
            truth_term = (truth_nll * weights * ok).sum() / (weights * ok).expand_as(truth_nll).sum().clamp_min(1.0)
            loss = loss + alpha * truth_term
            parts[f"truth_{step + 1}"] = float(truth_term.detach())
        total = total + loss / horizon
        if step + 1 < horizon:
            full = out["state_delta"].float().detach()
            history, valid_time = stepper.advance(history, full, valid_time)
            calendar = _advance_calendar(calendar, stepper.step_hours)
    parts["horizon"] = float(horizon)
    return total, parts


class DistillStep(nn.Module):
    """One distillation forward pass as a module (so data parallelism sees one model call)."""

    def __init__(self, model, stepper, max_horizon: int = 8, alpha: float = 0.3, total_steps: int = 10000,
                 channel_weights: torch.Tensor | None = None, precision: str = "bf16") -> None:
        super().__init__()
        self.model, self.stepper = model, stepper
        self.max_horizon, self.alpha, self.total_steps, self.precision = max_horizon, alpha, total_steps, precision
        self.register_buffer("step_count", torch.zeros((), dtype=torch.long), persistent=False)
        self.channel_weights = channel_weights

    def forward(self, batch: dict):
        horizon = horizon_for(int(self.step_count), self.total_steps, self.max_horizon) if self.training else self.max_horizon
        if self.training:
            self.step_count += 1
        loss, parts = distill_rollout_loss(self.model, batch, self.stepper, horizon, self.alpha, self.channel_weights,
                                           self.precision)
        return loss, parts, {}


def _collate(samples: list[dict]) -> dict:
    return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}


def train_distillation(model, settings, era5, teacher_dirs: list, stepper, train_inits, val_inits, max_horizon: int = 8,
                       alpha: float = 0.3, weights: dict | None = None, storms: dict | None = None) -> dict:
    """
    Distil the teachers into NatureV1 with the shared engine (one GPU or many): every checkpoint pushed, the
    weight average used for evaluation. Returns the last held-out losses.
    """
    from ..engine import Engine, is_main

    channels = model.config.analysis_channels
    train = DistillDataset(era5, teacher_dirs, train_inits, channels, max_horizon, weights=weights, storms=storms)
    held = DistillDataset(era5, teacher_dirs, val_inits, channels, max_horizon, weights=weights, storms=storms)
    estimate = max(1, int(settings.hours * 3600 / 2.0))           # a step every ~2 s on one large GPU
    step = DistillStep(model, stepper, max_horizon, alpha, total_steps=settings.max_steps or estimate,
                       precision=settings.precision)
    params = [p for p in model.parameters() if p.requires_grad]
    engine = Engine(settings, model, step, parameters=params, config=model.config.to_dict())
    loader = engine.loader(train, collate_fn=_collate)
    val_loader = engine.loader(held, shuffle=False, drop_last=False, collate_fn=_collate)
    engine.resume()
    if is_main():
        print(f"[distill] {len(train):,} initial times to train on, {len(held):,} held out, teachers "
              f"{sorted(set(train.names))}, horizon up to {max_horizon} steps, truth weight {alpha}", flush=True)

    def held_out(max_batches=None) -> dict:
        step.eval()
        totals, count = {}, 0
        with torch.no_grad(), engine.averaged():
            for index, batch in enumerate(val_loader):
                if max_batches is not None and index >= max_batches:
                    break
                _, parts, _ = step(engine.to_device(batch))
                for k, v in parts.items():
                    totals[k] = totals.get(k, 0.0) + v
                count += 1
        step.train()
        return {k: v / max(count, 1) for k, v in totals.items()}

    epoch = engine.epoch
    while not engine.should_stop():
        engine.set_epoch(loader, epoch)
        for batch in loader:
            batch = engine.to_device(batch)
            with engine.autocast():
                loss, parts, _ = engine.model(batch)
            engine.backward_step(loss, parts)
            stop, save, evaluate = engine.poll()
            if evaluate:
                report = held_out(50)
                if is_main():
                    print("[distill] held out: " + "  ".join(f"{k} {v:.4f}" for k, v in sorted(report.items())), flush=True)
                engine.last_eval = time.time()
            if save:
                engine.checkpoint()
            if stop:
                break
        epoch += 1
    report = held_out()
    engine.finish(report)
    from ..sattrain import _keep_average

    _keep_average(engine)
    return report


__all__ = ["DistillDataset", "DistillStep", "distill_rollout_loss", "horizon_for", "storm_weight_map",
           "train_distillation"]
