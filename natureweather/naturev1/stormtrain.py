# Copyright 2026 Nathan. Apache-2.0.
"""
Stage B: the storm heads and the backbone's last blocks, on every basin, with satellites in the loop.

Stage two trained 0.96M parameters of heads on 584 Atlantic storms over a frozen backbone. The red team
found what that leaves on the table: southern-hemisphere tracks worse than physics alone, poleward turns
missed, major hurricanes 10-18 kt too weak, rapid intensification nearly invisible. This stage changes
four things at once:

* **Every basin.** About 4,500 storms since 1980 from IBTrACS instead of 584 -- eight times the labels --
  in the poleward frame (:attr:`NatureConfig.hemisphere_frame`), so both hemispheres teach one model.
* **The backbone's last blocks learn too.** The last ``tail`` blocks (default 4 of 17) and the final
  norm are trained with the heads, so the features the heads read can bend toward what storms need.
  Two things keep the backbone the forecaster it was: the next-step state loss it was trained with
  stays in the objective, and every tail weight is pulled back toward where it started (L2-SP).
* **Satellites in the loop.** With probability ``retrieval_share`` a sample's newest frame is replaced by
  the satellite retrieval's reconstruction from a 6-12 h old state -- the model learns to forecast from
  what it will actually be given in real time, not only from a perfect reanalysis.
* **The satellite storm network fused in.** Its embedding (zero-initialised fusion) and its RI and
  intensity-change estimates (inputs to RI 2.0) reach the heads for every storm.

Afterwards, on held-out seasons: feature banks with both rollout members, the consensus weights, the
probability calibration (landfall, RI, RI 2.0), the intensity bias correction and the input monitor --
then a scorecard on the test seasons, which nothing was ever fitted on.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from .engine import Engine, EngineSettings, barrier, is_main
from .model import SURFACE_FIELDS
from .satdata import prepared_to_state

#: Storms per initial time, padded. The busiest global hours carry eight or nine.
MAX_STORMS = 12
#: Keys of the storm network's outputs kept per storm.
SATNET_KEYS = ("embedding", "wind_kt", "wind_log_var", "pressure_hpa", "centre_offset_km", "wind_change_kt",
               "ri_logits")


def storm_times(tracks, store_times: np.ndarray, lead_offsets, history: int = 6) -> list[tuple[int, list]]:
    """
    Pair every synoptic best-track fix with the reanalysis and group the fixes by initial time.

    Returns ``[(window_start, [StormTargets, ...]), ...]`` sorted by time: one backbone pass per entry
    serves every storm active at that hour.
    """
    from .storms import pair_tracks_with_reanalysis

    starts, targets, _ = pair_tracks_with_reanalysis(tracks, store_times, lead_offsets, history=history)
    grouped: dict[int, list] = {}
    for start, target in zip(starts.tolist(), targets):
        grouped.setdefault(int(start), []).append(target)
    return sorted(grouped.items())


def _table_lookup(table: dict) -> dict:
    return {(sid, int(t)): i for i, (sid, t) in enumerate(zip(table["sid"], table["time"]))}


class StormTimeDataset(torch.utils.data.Dataset):
    """
    One initial time: the analysis window, the next state, and every storm active then with its labels.

    Args:
        era5: a staged ERA5 memmap (:func:`naturev1.materialise`) the windows are read from.
        items: from :func:`storm_times`.
        satnet: the storm network's outputs per label-table row (:func:`satnet_outputs_for_table`), or None.
        table: the label table, to find each storm's row.
        frames, retrieval_share, stale_hours: when ``frames`` is given, samples carry what the retrieval
            needs to replace their newest frame, for a ``retrieval_share`` of samples.
        patches: the staged storm patches (:class:`naturev1.satdata.StormPatches`, needs ``table``): every storm then
            carries its storm-centred infrared now and six hours earlier as ``satellite/ir`` ``(MAX_STORMS, 2, 256,
            256)`` and ``satellite/available`` -- what the in-model satellite encoder and NatureV2's infrared
            structure read.
    """

    def __init__(self, era5: str | Path, items, channels: int, history: int = 6, satnet: dict | None = None,
                 table: dict | None = None, frames=None, retrieval_share: float = 0.0, stale_hours=(6, 12),
                 seed: int = 0, train: bool = True, patches=None) -> None:
        from .era5 import Normalizer
        from .satdata import _Memmaps

        self.path = Path(era5)
        meta = json.loads(self.path.with_suffix(".json").read_text())
        self.normalizer = Normalizer.from_dict(meta["statistics"])
        self.times = np.asarray(meta["time"], dtype=np.int64)
        self.items, self.channels, self.history = items, channels, history
        self.satnet, self.frames = satnet, frames
        self.lookup = _table_lookup(table) if table is not None else {}
        self.patches = patches if table is not None else None
        self.table_position = (np.asarray(table["lat"], np.float64), np.asarray(table["lon"], np.float64)) \
            if self.patches is not None else None
        self.retrieval_share, self.stale_hours = retrieval_share, tuple(stale_hours)
        self.seed, self.train, self.epoch = seed, train, 0
        self.maps = _Memmaps()

    def __len__(self) -> int:
        return len(self.items)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, item: int) -> dict:
        from .model import calendar_features

        start, targets = self.items[item]
        values = self.maps.get(self.path)
        window = [prepared_to_state(values[start + k], self.normalizer, self.channels) for k in range(self.history + 1)]
        analysis = torch.stack(window[: self.history])
        stamps = torch.tensor(self.times[start : start + self.history], dtype=torch.float64)
        now = int(self.times[start + self.history - 1])
        built = [t.build() for t in targets[:MAX_STORMS]]
        count = len(built)
        storms = {key: torch.stack([b[key] for b in built]) for key in built[0]}
        padded = {}
        for key, value in storms.items():
            # Padding slots are never read: every use goes through ``storm_mask``.
            fill = torch.zeros((MAX_STORMS, *value.shape[1:]), dtype=value.dtype)
            fill[:count] = value
            padded[key] = fill
        sample = {"analysis": analysis, "next_state": window[self.history], "calendar": calendar_features(stamps),
                  "valid_time": stamps[-1].clone(), "storm_mask": torch.arange(MAX_STORMS) < count,
                  **{f"storm/{k}": v for k, v in padded.items()}}
        if self.satnet is not None:
            rows = [self.lookup.get((t.track.storm_id, int(t.track.time[t.point].astype("datetime64[s]").astype(np.int64))), -1)
                    for t in targets[:MAX_STORMS]]
            present = torch.zeros(MAX_STORMS, dtype=torch.bool)
            for key in SATNET_KEYS:
                stored = self.satnet[key]
                block = torch.zeros((MAX_STORMS, *stored.shape[1:]), dtype=stored.dtype)
                for slot, row in enumerate(rows):
                    if row >= 0 and bool(self.satnet["present"][row]):
                        block[slot] = stored[row]
                        present[slot] = True
                sample[f"satnet/{key}"] = block
            sample["satnet_present"] = present
        if self.patches is not None:
            sample.update(self._satellite(targets[:MAX_STORMS]))
        use_retrieval = False
        if self.frames is not None and self.retrieval_share > 0:
            rng = np.random.default_rng((self.seed, self.epoch, item))
            use_retrieval = bool(rng.uniform() < self.retrieval_share) if self.train else (item % 2 == 0)
            stale = int(rng.choice(self.stale_hours)) if self.train else self.stale_hours[item % len(self.stale_hours)]
            row = int(np.searchsorted(self.times, now - stale * 3600))
            ok = row < len(self.times) and int(self.times[row]) == now - stale * 3600
            sat = np.zeros((3, 8, *self.frames.shape), np.uint8)
            seen = np.zeros(3, bool)
            for k, hours in enumerate((0, 3, 6)):
                frame = self.frames.frame(now - hours * 3600)
                if frame is not None:
                    sat[k], seen[k] = frame, True
            use_retrieval = use_retrieval and ok and bool(seen[0])
            sample["retrieval/satellite"] = torch.from_numpy(sat)
            sample["retrieval/available"] = torch.from_numpy(seen)
            sample["retrieval/background"] = (prepared_to_state(values[row], self.normalizer, self.channels) if ok
                                              else window[self.history - 1].clone())
            sample["retrieval/stale_hours"] = torch.tensor(float(stale))
        sample["use_retrieval"] = torch.tensor(use_retrieval)
        return sample


    def _satellite(self, targets) -> dict:
        """Storm-centred infrared now and six hours earlier for every storm, cut around its best-track centre."""
        from .satdata import recentre
        from .stormsat import PATCH_PIXELS

        codes = np.zeros((MAX_STORMS, 2, PATCH_PIXELS, PATCH_PIXELS), np.uint8)
        seen = np.zeros((MAX_STORMS, 2), bool)
        lats, lons = self.table_position
        for slot, target in enumerate(targets):
            track, point = target.track, target.point
            stamp = int(track.time[point].astype("datetime64[s]").astype(np.int64))
            centre = (float(track.latitude[point]), float(track.longitude[point]))
            for k, hours in enumerate((0, 6)):
                row = self.lookup.get((track.storm_id, stamp - hours * 3600), -1)
                patch = None if row < 0 else self.patches.patch(row)
                if patch is None:
                    continue
                crop, inside = recentre(patch, (float(lats[row]), float(lons[row])), centre, PATCH_PIXELS)
                codes[slot, k], seen[slot, k] = crop[0], inside > 0.5
        return {"satellite/ir": torch.from_numpy(codes), "satellite/available": torch.from_numpy(seen)}


class StormStep(nn.Module):
    """
    One stage-B forward pass as a module, so data parallelism sees it as one model call.

    Returns the loss and its parts. The frozen part of the backbone runs under ``no_grad``; the tail, the
    heads and the fusion learn; the decoder and state head are frozen but carry gradient back into the
    tail, which is what keeps the tail a weather model.
    """

    def __init__(self, model, stepper, grid, variables, tail: int = 4, state_weight: float = 1.0,
                 anchor_weight: float = 1e-3, checkpointing: bool = True,
                 class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        from .state import channel_loss_weights

        self.model, self.stepper, self.grid, self.variables = model, stepper, grid, list(variables)
        self.tail, self.state_weight, self.anchor_weight = tail, state_weight, anchor_weight
        self.checkpointing = checkpointing
        weights = channel_loss_weights(stepper.roles.channels, model.config.analysis_channels)[stepper.roles.prognostic]
        self.register_buffer("channel_weights", weights / weights.mean(), persistent=False)
        # Intensity-loss weight per class (below hurricane, category 1-2, major), from :func:`class_weights`.
        self.register_buffer("class_weights", torch.ones(3) if class_weights is None
                             else torch.as_tensor(class_weights, dtype=torch.float32), persistent=False)
        start = len(model.blocks) - tail
        # Where the tail started, as buffers so they follow the module to the GPU (L2-SP pulls back to them).
        self.anchor_names = []
        for name, parameter in model.blocks[start:].named_parameters():
            key = "anchor_" + name.replace(".", "_")
            self.register_buffer(key, parameter.detach().clone(), persistent=False)
            self.anchor_names.append((name, key))

    def tail_parameters(self):
        return self.model.blocks[len(self.model.blocks) - self.tail:].parameters()

    def _retrieved(self, batch: dict) -> torch.Tensor:
        """The analysis window with the newest frame replaced by the retrieval where the batch asks for it."""
        from .satnet import apply_retrieval

        analysis = batch["analysis"]
        flags = batch.get("use_retrieval")
        model = self.model
        if flags is None or model.retrieval is None or not bool(flags.any()):
            return analysis
        chosen = flags.nonzero().squeeze(-1)
        with torch.no_grad():
            out = model.retrieval(batch["retrieval/satellite"][chosen], batch["retrieval/available"][chosen],
                                  batch["retrieval/background"][chosen], batch["retrieval/stale_hours"][chosen],
                                  batch["calendar"][chosen, -1], self.grid, model.latent_grid, model.link)
            state = apply_retrieval(self.stepper, batch["retrieval/background"][chosen], out,
                                    batch["retrieval/stale_hours"][chosen], batch["valid_time"][chosen])
        analysis = analysis.clone()
        analysis[chosen, -1] = state.to(analysis.dtype)
        return analysis

    def forward(self, batch: dict):
        from .losses import total_loss
        from .steering import steering_track, storm_environment
        from .storms import ri2_targets
        from .thermo import storm_physics

        model = self.model
        analysis = self._retrieved(batch)
        latent_grid = model.latent_grid
        start = len(model.blocks) - self.tail
        with torch.no_grad():
            x = model.encode(None, None, analysis, self.grid, batch["calendar"])
            for block in model.blocks[:start]:
                x = block(x, latent_grid)
        x = x.detach()
        for block in model.blocks[start:]:
            if self.checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, latent_grid, use_reentrant=False)
            else:
                x = block(x, latent_grid)
        current = model.norm(x)[:, -1]                                           # (B, P, H)

        mask = batch["storm_mask"]
        b_index, k_index = mask.nonzero(as_tuple=True)
        take = (lambda key: batch[f"storm/{key}"][b_index, k_index])
        centres = take("storm_center")
        rows = current[b_index]
        near = model.storm_pool(rows, centres)
        far = model.storm_pool(rows, centres, 2.0 * model.config.storm_pool_radius_km)
        with torch.no_grad():
            now = analysis[b_index, -1].float()
            environment = storm_environment(now, self.variables, self.stepper.normalizer, self.grid, centres)
            prior = steering_track(now, self.variables, self.stepper.normalizer, self.grid, centres,
                                   model.config.lead_times_hours)
        satnet = None
        if "satnet_present" in batch:
            present = batch["satnet_present"][b_index, k_index]
            satnet = {key: batch[f"satnet/{key}"][b_index, k_index] for key in SATNET_KEYS}
            satnet["embedding"] = satnet["embedding"] * present.view(-1, 1).to(satnet["embedding"].dtype)
            satnet["present"] = present
        targets = {key[len("storm/"):]: value[b_index, k_index] for key, value in batch.items()
                   if key.startswith("storm/")}
        if model.ri2_head is not None:
            targets["ri2_target"], targets["ri2_mask"] = ri2_targets(targets, model.config.lead_times_hours)
        wind_class = torch.bucketize(torch.nan_to_num(targets["current_wind_kt"].float(), nan=40.0),
                                     torch.tensor(CLASS_EDGES_KT, device=centres.device), right=True)
        targets["sample_weight"] = self.class_weights[wind_class]
        with torch.no_grad():
            physics = storm_physics(now, self.variables, self.stepper.normalizer, self.grid, centres, take("storm_state"))
        satellite = None
        if "satellite/ir" in batch:
            codes = batch["satellite/ir"][b_index, k_index]
            if self.training:
                from .storms import augment_codes

                codes = augment_codes(codes)
            satellite = (codes, batch["satellite/available"][b_index, k_index])
        ocean = None
        if getattr(model, "storm_ocean_embed", None) is not None:
            from .thermo import storm_ocean

            ocean = storm_ocean(centres, batch["valid_time"][b_index], environment, targets.get("current_wind_radii_nmi"))
        outputs = model.storm_heads(near, centres, take("storm_state"), environment,
                                    prior if model.uses_track_prior() else None, far, satellite, satnet, physics,
                                    storm_ocean=ocean)
        loss, parts = total_loss({k: v.float() for k, v in outputs.items()}, targets, SURFACE_FIELDS)

        if self.state_weight and model.state_head is not None:
            seed = model.mesh_seed.expand(current.shape[0], self.grid.num_points, -1)
            decoded = model.decode_norm(model.decoder(seed, current, model.link(self.grid, latent_grid, 24)))
            raw = model.state_head(decoded).view(current.shape[0], self.grid.num_points, -1, 2).float()
            truth = self.stepper.to_state_delta(analysis[:, -1].float(), batch["next_state"].float())
            area = self.grid.weights.to(raw.device, torch.float32)
            area = (area / area.mean()).view(1, -1, 1)
            log_var = raw[..., 1].clamp(-10.0, 10.0)
            nll = 0.5 * (log_var + (truth - raw[..., 0]) ** 2 / log_var.exp())
            state_term = (nll * area * self.channel_weights.view(1, 1, -1)).mean()
            loss = loss + self.state_weight * state_term
            parts["state"] = float(state_term.detach())
        if self.anchor_weight:
            tail = dict(model.blocks[start:].named_parameters())
            drift = sum(((tail[name] - getattr(self, key)) ** 2).sum() for name, key in self.anchor_names)
            loss = loss + self.anchor_weight * drift
            parts["anchor"] = float(drift.detach())
        parts["storms"] = float(len(b_index))
        extra = {}
        if satnet is not None:
            extra["storm_satnet"] = satnet
        if satellite is not None:
            extra["storm_satellite"] = satellite
        if ocean is not None:
            extra["storm_ocean"] = ocean
        return loss, parts, extra


CLASS_EDGES_KT = (63.5, 95.5)


def class_weights(items, power: float = 0.5, cap: float = 4.0) -> torch.Tensor:
    """
    Intensity-loss weights for the three classes (below hurricane, category 1-2, major hurricane).

    Inverse class frequency to ``power`` (0.5: the square root, halfway between plain and fully balanced),
    capped at ``cap`` times the smallest, then scaled so a training fix weighs one on average: the loss
    keeps its size, but the rare strong storms the heads otherwise regress toward the middle count more.
    """
    counts = np.zeros(3)
    for _, targets in items:
        for target in targets[:MAX_STORMS]:
            wind = float(target.track.max_wind_kt[target.point])
            if np.isfinite(wind):
                counts[int(np.searchsorted(CLASS_EDGES_KT, wind))] += 1
    share = np.maximum(counts, 1.0) / max(counts.sum(), 1.0)
    weights = share ** -power
    weights = np.minimum(weights, cap * weights.min())
    weights = weights / max(float((weights * counts).sum() / max(counts.sum(), 1.0)), 1e-9)
    return torch.tensor(weights, dtype=torch.float32)


def _collate(samples: list[dict]) -> dict:
    return {key: torch.stack([s[key] for s in samples]) for key in samples[0]}


def train_storms(model, settings: EngineSettings, era5: str | Path, stepper, grid, variables, train_items, val_items,
                 satnet: dict | None = None, table: dict | None = None, frames=None, tail: int = 4,
                 retrieval_share: float = 0.3, tail_lr_scale: float = 0.1, rebalance: bool = True,
                 patches=None) -> dict:
    """Stage B. Returns the last held-out losses; see :func:`finalise_storms` for the fitting afterwards."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    weights = class_weights(train_items) if rebalance else None
    step = StormStep(model, stepper, grid, variables, tail=tail, class_weights=weights)
    heads = list(model.storm_head_parameters())
    tail_params = list(step.tail_parameters()) + list(model.norm.parameters())
    for parameter in heads + tail_params:
        parameter.requires_grad = True
    model.config.hemisphere_frame = True
    groups = [{"params": heads, "lr": settings.learning_rate},
              {"params": tail_params, "lr": settings.learning_rate * tail_lr_scale}]
    settings.find_unused_parameters = True
    engine = Engine(settings, model, step, parameters=groups, config=model.config.to_dict())
    grid = model._aligned(grid)
    channels = model.config.analysis_channels
    train = StormTimeDataset(era5, train_items, channels, satnet=satnet, table=table, frames=frames,
                             retrieval_share=retrieval_share if model.retrieval is not None else 0.0,
                             seed=settings.seed, train=True, patches=patches)
    held = StormTimeDataset(era5, val_items, channels, satnet=satnet, table=table, frames=frames,
                            retrieval_share=retrieval_share if model.retrieval is not None else 0.0,
                            seed=settings.seed, train=False, patches=patches)
    loader = engine.loader(train, collate_fn=_collate)
    val_loader = engine.loader(held, shuffle=False, drop_last=False, collate_fn=_collate)
    engine.resume()
    if is_main():
        storms = sum(len(t) for _, t in train_items)
        print(f"[storms-b] {len(train):,} initial times ({storms:,} storm fixes) to train on, {len(held):,} held out; "
              f"tail of {tail} blocks + heads: {sum(p.numel() for p in heads + tail_params) / 1e6:.1f}M trainable; "
              f"retrieval in the loop for {100 * (retrieval_share if model.retrieval is not None else 0):.0f}% of samples",
              flush=True)
        if weights is not None:
            print("[storms-b] intensity weights by class (TS / cat 1-2 / major): "
                  + " / ".join(f"{float(w):.2f}" for w in weights), flush=True)

    def held_out(max_batches=None) -> dict:
        step.eval()
        totals: dict[str, float] = {}
        count = 0
        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                if max_batches is not None and index >= max_batches:
                    break
                batch = engine.to_device(batch)
                with engine.autocast():
                    loss, parts, _ = engine.model(batch)
                for key, value in parts.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                totals["loss"] = totals.get("loss", 0.0) + float(loss)
                count += 1
        from .engine import all_mean

        step.train()
        return all_mean({k: v / max(count, 1) for k, v in totals.items()}, engine.device)

    report, stop = {}, False
    while not stop:
        engine.set_epoch(loader, engine.epoch)
        for batch in loader:
            batch = engine.to_device(batch)
            with engine.autocast():
                loss, parts, extra = engine.model(batch)
            engine.backward_step(loss, parts)
            marks = {key: 1.0 for key in ("track", "landfall", "intensity", "ri", "ri_delta", "eyewall_peak",
                                          "eyewall_rmw", "wind_radii", "ri2", "backbone_tail")}
            model.mark_trained(marks, {"storm_center": True, "storm_state": True, "storm_environment": True,
                                       "pooled_far": True, "track_prior": True if model.uses_track_prior() else None,
                                       "storm_satnet": extra.get("storm_satnet"), "storm_physics": True,
                                       "storm_satellite": extra.get("storm_satellite"),
                                       "storm_ocean": extra.get("storm_ocean")})
            stop, save, evaluate = engine.poll()
            if evaluate:
                with engine.averaged():
                    report = held_out(max_batches=100)
                engine.last_eval = time.time()
                engine.history.append({"step": engine.step, **report})
                if report.get("loss", math.inf) < engine.best:
                    engine.best = report["loss"]
                if is_main():
                    print(f"[storms-b] held-out at step {engine.step:,}: "
                          + "  ".join(f"{k} {v:.4f}" for k, v in sorted(report.items())), flush=True)
            if save:
                engine.checkpoint(report)
            if stop:
                break
        engine.epoch += 1
    with engine.averaged():
        report = held_out()
    engine.history.append({"step": engine.step, "final": True, **report})
    engine.finish(report)
    from .sattrain import _keep_average

    _keep_average(engine)
    return report


# ---------------------------------------------------------------------------------------- banks --

@torch.no_grad()
def memmap_feature_bank(model, era5: str | Path, items, stepper, grid, variables, satnet: dict | None = None,
                        table: dict | None = None, rollout: bool = True, precision: str = "bf16",
                        part: Path | None = None, analog=None, ensemble_members: int = 1, rotations: int = 1,
                        patches=None) -> dict:
    """
    The storm feature bank (:func:`naturev1.storm_feature_bank`'s format) straight from a staged memmap,
    one backbone pass per initial time for all its storms -- plus the storm physics
    (:func:`naturev1.thermo.storm_physics`), the analog member when an :class:`naturev1.tracker.AnalogForecaster`
    is given, and, with ``rollout``, every rollout member from one full-state rollout per time (an ensemble of
    ``ensemble_members`` when more than one). With ``rotations`` above one it also keeps the storm pools of
    that many members of the test-time rotation ensemble (:meth:`naturev1.model.NatureV1.rotated_pools`) as
    ``features_tta`` / ``features_far_tta``, so :func:`select_rotations` can decide on held-out seasons
    whether the ensemble earns its cost. Distributed: each rank does every ``world``-th time and rank 0
    assembles; pass ``part`` (a path prefix) for the per-rank pieces.
    """
    from .model import rotation_angles
    from .thermo import storm_physics
    from .state import _autocast
    from .steering import steering_track, storm_environment
    from .storms import BANK_VERSION, _backbone_fingerprint, _stack_targets
    from .tracker import rollout_member_tracks

    device = next(model.parameters()).device
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    dataset = StormTimeDataset(era5, items, model.config.analysis_channels, satnet=satnet, table=table, train=False,
                               patches=patches)
    model.eval()
    leads = model.config.lead_times_hours
    pieces = []
    started = time.time()
    mine = list(range(rank, len(items), world))
    for done, item in enumerate(mine, start=1):
        sample = _collate([dataset[item]])
        sample = {k: v.to(device) for k, v in sample.items()}
        count = int(sample["storm_mask"].sum())
        centres = sample["storm/storm_center"][0, :count]
        with _autocast(device, precision):
            latent = model.encode(None, None, sample["analysis"], grid, sample["calendar"])
            current = model.norm(model._run_blocks(latent, model.latent_grid))[:, -1].float()
        rows = current.expand(count, -1, -1)
        piece = {"index": item, "features": model.storm_pool(rows, centres).float().cpu(),
                 "features_far": model.storm_pool(rows, centres, 2.0 * model.config.storm_pool_radius_km).float().cpu()}
        if rotations > 1:
            near, wide = [], []
            for degrees in rotation_angles(rotations):
                with _autocast(device, precision):
                    a, b = model.rotated_pools({"analysis": (sample["analysis"], grid)}, sample["calendar"],
                                               centres, degrees)
                near.append(a.float().cpu())
                wide.append(b.float().cpu())
            piece["features_tta"], piece["features_far_tta"] = torch.stack(near, 1), torch.stack(wide, 1)
        now = sample["analysis"][:, -1].float().expand(count, -1, -1)
        piece["storm_environment"] = storm_environment(now, variables, stepper.normalizer, grid, centres).cpu()
        piece["track_prior"] = steering_track(now, variables, stepper.normalizer, grid, centres, leads).cpu()
        state_vec = sample["storm/storm_state"][0, :count]
        piece["storm_physics"] = storm_physics(now, variables, stepper.normalizer, grid, centres, state_vec).cpu()
        if getattr(model, "storm_ocean_embed", None) is not None:
            from .thermo import storm_ocean

            radii_now = sample["storm/current_wind_radii_nmi"][0, :count] if "storm/current_wind_radii_nmi" in sample else None
            piece["storm_ocean"] = storm_ocean(centres, sample["valid_time"].expand(count), piece["storm_environment"].to(device),
                                               radii_now).cpu()
        if "satellite/ir" in sample:
            piece["satellite_ir"] = sample["satellite/ir"][0, :count].cpu()
            piece["satellite_available"] = sample["satellite/available"][0, :count].cpu()
        wind_kt = sample["storm/current_wind_kt"][0, :count].float()
        radii = sample["storm/wind_radii_target"][0, :count, 0, 0, :].float() if "storm/wind_radii_target" in sample else None
        size_km = None
        if radii is not None:
            # The gale radius at the first lead stands in for now's (the label table's own is not in the bank).
            size_km = torch.nan_to_num(radii.nanmean(-1), nan=200.0) * 1.852
        if analog is not None:
            day = (sample["valid_time"].double() % 31557600.0) / 31557600.0
            predicted = analog.predict(centres.cpu(), state_vec.cpu(), torch.nan_to_num(wind_kt.cpu(), nan=50.0),
                                       day.cpu().expand(count))
            piece["analog_track"] = predicted["displacement"]
            piece["analog_change_kt"] = predicted["change_kt"]
        if rollout and model.state_head is not None:
            members = rollout_member_tracks(model, {k: sample[k] for k in ("analysis", "calendar", "valid_time")},
                                            stepper, centres, leads, variables, precision, members=ensemble_members,
                                            wind_kt=torch.nan_to_num(wind_kt, nan=50.0), size_km=size_km)
            piece.update({k: v.cpu() for k, v in members.items()})
        if satnet is not None:
            piece["satnet"] = {key: sample[f"satnet/{key}"][0, :count].cpu() for key in SATNET_KEYS}
            piece["satnet_present"] = sample["satnet_present"][0, :count].cpu()
        pieces.append(piece)
        if is_main() and (done % 50 == 0 or done == len(mine)):
            rate = done / max(time.time() - started, 1e-9)
            print(f"\r[bank] {done:,}/{len(mine):,} times on this GPU ({(len(mine) - done) / max(rate, 1e-9) / 60:.0f} min left)",
                  end="", flush=True)
    if is_main():
        print(flush=True)
    if world > 1:
        if part is None:
            raise ValueError("a distributed bank build needs a part path")
        torch.save(pieces, f"{part}.{rank}")
        barrier()
        if not is_main():
            return {}
        pieces = [p for r in range(world) for p in torch.load(f"{part}.{r}", weights_only=False)]
        for r in range(world):
            Path(f"{part}.{r}").unlink(missing_ok=True)
    pieces.sort(key=lambda p: p["index"])
    targets = [t for _, group in items for t in group[:MAX_STORMS]]
    stacked = _stack_targets(targets)
    from .ibtracs import BASINS

    stacked["basin"] = torch.tensor([BASINS.index(t.track.basin) if t.track.basin in BASINS else -1 for t in targets])
    stacked["storm_environment"] = torch.cat([p["storm_environment"] for p in pieces])
    stacked["track_prior"] = torch.cat([p["track_prior"] for p in pieces])
    stacked["storm_physics"] = torch.cat([p["storm_physics"] for p in pieces])
    if pieces and all("storm_ocean" in p for p in pieces):
        stacked["storm_ocean"] = torch.cat([p["storm_ocean"] for p in pieces])
    for key in ("analog_track", "analog_change_kt", "rollout_steering", "rollout_vortex", "rollout_ensemble",
                "rollout_ensemble_spread_km", "rollout_pressure_hpa", "rollout_wind_kt", "rollout_phase"):
        if pieces and all(key in p for p in pieces):
            stacked[key] = torch.cat([p[key] for p in pieces])
    bank = {"features": torch.cat([p["features"] for p in pieces]),
            "features_far": torch.cat([p["features_far"] for p in pieces]),
            "targets": stacked, "fingerprint": _backbone_fingerprint(model), "version": BANK_VERSION}
    if rotations > 1 and pieces and all("features_tta" in p for p in pieces):
        bank["features_tta"] = torch.cat([p["features_tta"] for p in pieces])
        bank["features_far_tta"] = torch.cat([p["features_far_tta"] for p in pieces])
    if pieces and all("satellite_ir" in p for p in pieces):
        # The storm-centred infrared the in-model encoder reads, in the bank format :func:`naturev1.storms._satellite_batch` uses.
        bank["satellite"] = {"ir": torch.cat([p["satellite_ir"] for p in pieces]),
                             "available": torch.cat([p["satellite_available"] for p in pieces])}
    if satnet is not None:
        present = torch.cat([p["satnet_present"] for p in pieces])
        bank["satnet"] = {key: torch.cat([p["satnet"][key] for p in pieces]) for key in SATNET_KEYS}
        bank["satnet"]["embedding"] = bank["satnet"]["embedding"] * present.view(-1, 1).float()
        bank["satnet"]["present"] = present
    return bank


@torch.no_grad()
def satnet_outputs_for_table(model, table: dict, patches, rows=None, batch_size: int = 256,
                             part: Path | None = None) -> dict:
    """
    Run the trained storm network once for every label-table row that has a staged loop, centred on the
    best-track position (the centre a forecast is issued from). Returns tensors aligned to the table, with
    ``present`` saying which rows have them. Distributed like :func:`memmap_feature_bank`.
    """
    from .satdata import StormSequenceDataset

    device = next(model.parameters()).device
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    data = StormSequenceDataset(table, patches, rows, jitter_km=0.0, train=False,
                                pixels=model.storm_satnet.config.pixels)
    mine = torch.utils.data.Subset(data, list(range(rank, len(data), world)))
    loader = torch.utils.data.DataLoader(mine, batch_size=batch_size, num_workers=8 if device.type == "cuda" else 0)
    model.storm_satnet.eval()
    size = len(table["time"])
    out = {key: None for key in SATNET_KEYS}
    found_rows, found = [], {key: [] for key in SATNET_KEYS}
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            result = model.storm_satnet(batch["codes"].to(device), batch["available"].to(device),
                                        batch["latitude"].to(device), batch["context"].to(device))
        found_rows.append(batch["row"])
        for key in SATNET_KEYS:
            found[key].append(result[key].float().cpu())
    piece = {"rows": torch.cat(found_rows) if found_rows else torch.zeros(0, dtype=torch.long),
             **{key: torch.cat(v) if v else None for key, v in found.items()}}
    if world > 1:
        torch.save(piece, f"{part}.{rank}")
        barrier()
        pieces = [torch.load(f"{part}.{r}", weights_only=False) for r in range(world)]
        barrier()
        if is_main():
            for r in range(world):
                Path(f"{part}.{r}").unlink(missing_ok=True)
    else:
        pieces = [piece]
    present = torch.zeros(size, dtype=torch.bool)
    for p in pieces:
        if p["rows"].numel() == 0:
            continue
        present[p["rows"]] = True
        for key in SATNET_KEYS:
            if out[key] is None:
                out[key] = torch.zeros((size, *p[key].shape[1:]), dtype=torch.float32)
            out[key][p["rows"]] = p[key]
    for key in SATNET_KEYS:
        if out[key] is None:
            out[key] = torch.zeros(size, 1)
    out["present"] = present
    return out


def _head_skill(model, bank: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-lead error of the heads alone: the track mixture's mean (km) and the wind (kt), NaN where unverified."""
    from .storms import KT_TO_MS, _great_circle_km, _head_outputs

    out, t = _head_outputs(model, bank), bank["targets"]
    size = out["displacement"].shape[0]
    probabilities = out["mode_logits"].softmax(-1)
    mean = (out["displacement"] * probabilities.view(size, -1, 1, 1)).sum(1)
    centre, truth, valid = t["storm_center"].float(), t["track_target"].float(), t["track_valid"] > 0
    leads = mean.shape[1]
    track, wind = torch.full((leads,), float("nan")), torch.full((leads,), float("nan"))
    for lead in range(leads):
        ok = valid[:, lead]
        if ok.any():
            lat0, lon0 = centre[ok, 0], centre[ok, 1]
            track[lead] = _great_circle_km(lat0 + truth[ok, lead, 0], lon0 + truth[ok, lead, 1],
                                           lat0 + mean[ok, lead, 0], lon0 + mean[ok, lead, 1]).mean()
        seen = t["intensity_mask"][:, lead, 0] > 0
        if seen.any():
            wind[lead] = ((out["intensity_mean"][seen, lead, 0] - t["intensity_target"][seen, lead, 0]).abs()
                          / KT_TO_MS).mean()
    return track, wind


def select_rotations(model, val_bank: dict, margin: float = 0.002) -> str:
    """
    Decide on the held-out seasons how many members of the test-time rotation ensemble the storm outputs use.

    Every size the bank supports is scored against a single pass, on the heads' own track error and wind
    error at every lead; the score is the mean relative change over both. The smallest size within
    ``margin`` of the best wins, and a single pass unless an ensemble beats it by more than ``margin`` --
    a rotation ensemble costs a backbone pass per member at forecast time, so it has to pay for itself.
    """
    if "features_tta" not in val_bank:
        model.config.storm_rotations = 1
        return "rotation ensemble: not in the bank, single pass"
    available = int(val_bank["features_tta"].shape[1]) + 1
    scores, lines = {}, []
    model.config.storm_rotations = 1
    base_track, base_wind = _head_skill(model, val_bank)
    for size in range(1, available + 1):
        model.config.storm_rotations = size
        track, wind = (base_track, base_wind) if size == 1 else _head_skill(model, val_bank)
        change = torch.cat([track / base_track, wind / base_wind]) - 1.0
        scores[size] = float(change[torch.isfinite(change)].mean()) if torch.isfinite(change).any() else 0.0
        lines.append(f"  {size} member{'s' if size > 1 else ' '}: track {' '.join(f'{v:5.0f}' for v in track.tolist())} km, "
                     f"wind {' '.join(f'{v:4.1f}' for v in wind.tolist())} kt ({100 * scores[size]:+.2f}%)")
    best = min(scores.values())
    chosen = min(size for size, score in scores.items() if score <= best + margin)
    if scores[chosen] > -margin:
        chosen = 1
    model.config.storm_rotations = chosen
    return ("rotation ensemble on held-out seasons (heads alone, per lead):\n" + "\n".join(lines)
            + f"\n  -> {chosen} member{'s' if chosen > 1 else ''}"
            + (" (a single pass: no ensemble beat it)" if chosen == 1 else ""))


def finalise_storms(model, val_bank: dict, test_bank: dict, train_bank: dict | None = None) -> str:
    """
    Everything fitted after the heads, on the held-out seasons, then scored on the test seasons:
    consensus weights (with the rollout members), probability calibration, intensity bias correction,
    the input monitor. Returns the report.
    """
    from .storms import (
        _consensus_inputs, calibration_report, describe_consensus, fit_feature_monitor, fit_intensity_correction,
        fit_probability_calibration, fit_track_consensus, intensity_by_category, ri2_report, storm_scorecard,
    )

    from .statistical import (
        fit_climatology, fit_consensus_error, fit_intensity_consensus, fit_logistic, fit_regression, fit_rollout_map,
        fit_track_bias,
    )
    from .storms import consensus_scorecard

    leads = model.config.lead_times_hours
    # First, because every fit after it reads the head outputs the choice produces.
    sections = [select_rotations(model, val_bank)]
    members, truth, valid, latitude = _consensus_inputs(model, val_bank, leads)
    weights = fit_track_consensus(members, truth, valid, latitude)
    model.track_consensus.copy_(weights.to(model.track_consensus.device))
    sections.append("consensus weights (fitted on held-out seasons): " + describe_consensus(model.track_consensus, leads))
    sections.append(fit_track_bias(model, members, truth, valid, latitude))
    sections.append(fit_consensus_error(model, members, truth, valid, latitude))
    sections.append("probability calibration: " + fit_probability_calibration(model, val_bank))
    sections.append("intensity bias correction (fitted on held-out seasons):\n" + fit_intensity_correction(model, val_bank))
    # The statistical-dynamical intensity models learn from training storms; their consensus weights from
    # held-out ones -- the same separation the heads and the track consensus keep.
    fitting = train_bank if train_bank is not None else val_bank
    for fit in (fit_regression, fit_logistic, fit_climatology, fit_rollout_map):
        sections.append(fit(model, fitting))
    sections.append(fit_intensity_consensus(model, val_bank))
    from .statistical import fit_ri_consensus

    sections.append(fit_ri_consensus(model, val_bank))
    threshold = fit_feature_monitor(model, fitting)
    sections.append(f"input monitor threshold {threshold:.3f}")
    sections.append("TEST SEASONS -- never fitted on:")
    sections.append(storm_scorecard(model, test_bank, leads))
    sections.append(consensus_scorecard(model, test_bank))
    sections.append(intensity_by_category(model, test_bank))
    sections.append(calibration_report(model, test_bank))
    sections.append(ri2_report(model, test_bank))
    return "\n\n".join(sections)


__all__ = ["MAX_STORMS", "StormStep", "StormTimeDataset", "class_weights", "finalise_storms", "memmap_feature_bank",
           "satnet_outputs_for_table", "select_rotations", "storm_times", "train_storms"]


@torch.no_grad()
def compare_models(new_model, new_bank: dict, old_model, old_bank: dict) -> dict:
    """
    The new model against the one it started from, on the same test storms: consensus track error (km)
    and wind MAE (kt) per lead, globally and for the Atlantic alone (where the old heads were trained), and
    a verdict -- the share of (lead, metric) pairs where the new model is better.
    """
    from .ibtracs import BASINS
    from .storms import KT_TO_MS, _great_circle_km, _head_outputs, consensus_track, track_members

    leads = new_model.config.lead_times_hours
    result = {"leads": list(leads)}
    scored = []
    for name, model, bank in (("new", new_model, new_bank), ("old", old_model, old_bank)):
        out = _head_outputs(model, bank)
        t = bank["targets"]
        members = track_members(out, t, leads)
        weights = model.track_consensus.float()
        if weights.shape[-1] < members.shape[2]:
            weights = torch.cat([weights, weights.new_zeros(weights.shape[0], members.shape[2] - weights.shape[-1])], -1)
        if float(weights.abs().sum()) == 0:
            weights = torch.zeros_like(weights)
            weights[:, :4] = 0.25
        blended = consensus_track(members, weights)
        centre, truth, valid = t["storm_center"].float(), t["track_target"].float(), t["track_valid"] > 0
        basin = t.get("basin", torch.full((centre.shape[0],), -1))
        for region, keep in (("global", torch.ones_like(valid[:, 0])), ("atlantic", basin == BASINS.index("NA"))):
            track, wind = [], []
            for lead in range(len(leads)):
                ok = valid[:, lead] & keep
                if not ok.any():
                    track.append(float("nan"))
                    wind.append(float("nan"))
                    continue
                lat0, lon0 = centre[ok, 0], centre[ok, 1]
                d = blended[ok, lead]
                track.append(float(_great_circle_km(lat0 + truth[ok, lead, 0], lon0 + truth[ok, lead, 1],
                                                    lat0 + d[:, 0], lon0 + d[:, 1]).mean()))
                seen = ok & (t["intensity_mask"][:, lead, 0] > 0)
                wind.append(float(((out["intensity_mean"][seen, lead, 0] - t["intensity_target"][seen, lead, 0]).abs()
                                   / KT_TO_MS).mean()) if seen.any() else float("nan"))
            result[f"{name}_{region}_track_km"] = track
            result[f"{name}_{region}_wind_mae_kt"] = wind
    for region in ("global", "atlantic"):
        for metric in ("track_km", "wind_mae_kt"):
            new, old = result[f"new_{region}_{metric}"], result[f"old_{region}_{metric}"]
            scored += [(region, a < b) for a, b in zip(new, old) if math.isfinite(a) and math.isfinite(b)]
    result["better_share_global"] = (sum(ok for r, ok in scored if r == "global")
                                     / max(sum(1 for r, _ in scored if r == "global"), 1))
    result["better_share_atlantic"] = (sum(ok for r, ok in scored if r == "atlantic")
                                       / max(sum(1 for r, _ in scored if r == "atlantic"), 1))
    return result


def describe_comparison(result: dict) -> str:
    lines = [f"{'lead':>6} | {'track km new/old (global)':>27} | {'(Atlantic)':>14} | {'wind MAE new/old (global)':>26} | {'(Atlantic)':>12}"]
    for i, hours in enumerate(result["leads"]):
        lines.append(f"{hours:>5}h | {result['new_global_track_km'][i]:>12.0f} / {result['old_global_track_km'][i]:<12.0f} | "
                     f"{result['new_atlantic_track_km'][i]:>6.0f} / {result['old_atlantic_track_km'][i]:<6.0f} | "
                     f"{result['new_global_wind_mae_kt'][i]:>11.1f} / {result['old_global_wind_mae_kt'][i]:<11.1f} | "
                     f"{result['new_atlantic_wind_mae_kt'][i]:>5.1f} / {result['old_atlantic_wind_mae_kt'][i]:<5.1f}")
    lines.append(f"better than the starting model on {100 * result['better_share_global']:.0f}% of global and "
                 f"{100 * result['better_share_atlantic']:.0f}% of Atlantic (lead, metric) pairs")
    return "\n".join(lines)


def save_release(model, directory: str | Path, run: str, hub_repo: str | None, report: str | None = None) -> Path:
    """Write the finished stage as ``<directory>/<run>/latest.pt`` (the checkpoint format every loader reads)
    and push it, with its report, to ``checkpoints/<run>/`` on the Hub -- waiting until the upload lands."""
    from .checkpoint import TrainingState
    from .engine import CheckpointPusher
    from .safeio import atomic_torch_save, atomic_write_text

    folder = Path(directory) / run
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "latest.pt"
    steps = int(model.head_steps.sum())
    atomic_torch_save({"model": model.state_dict(), "optimizer": None, "scheduler": None, "scaler": None,
                       "state": TrainingState(step=steps).to_dict(), "config": model.config.to_dict(), "rng": {},
                       "saved_at": time.time(), "format": 2, "tag": run}, path)
    uploads = [(path, f"checkpoints/{run}/latest.pt")]
    if report is not None:
        atomic_write_text(folder / "report.txt", report)
        uploads.append((folder / "report.txt", f"checkpoints/{run}/report.txt"))
    pusher = CheckpointPusher(hub_repo)
    pusher.submit(uploads, f"{run} (final)")
    pusher.wait(3600)
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="NatureV1 stage B: storm heads and backbone tail, every basin")
    parser.add_argument("--base", required=True, help="released checkpoint to start from (stage two or three)")
    parser.add_argument("--satnet", help="stage A storm-network checkpoint (stage-a-satnet/latest.pt)")
    parser.add_argument("--retrieval", help="stage A retrieval checkpoint (stage-a-retrieval/latest.pt)")
    parser.add_argument("--era5", required=True, help="staged ERA5 memmap, 1980 onward")
    parser.add_argument("--tendency", help="tendency.pt; default: the one recorded in the checkpoint")
    parser.add_argument("--ibtracs", required=True, help="IBTrACS CSV (since 1980)")
    parser.add_argument("--table", help="label table .npz, for the storm network's outputs")
    parser.add_argument("--root", help="staged satellite root (storm patches and global frames)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--tail", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--retrieval-share", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--hub-repo", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-rollout-members", action="store_true")
    parser.add_argument("--ensemble-members", type=int, default=4,
                        help="rollout ensemble size for the ensemble-mean track member and its spread")
    parser.add_argument("--rotations", type=int, default=4,
                        help="test-time rotation ensemble size kept in the banks (1: none); used only if it "
                             "beats a single pass on the held-out seasons")
    parser.add_argument("--monitor-share", type=float, default=0.2,
                        help="share of training times whose features fit the input monitor")
    parser.add_argument("--patches-in-heads", action="store_true",
                        help="feed every storm's centred infrared (now and 6 h earlier) to the in-model satellite encoder "
                             "and, for a NatureV2 base, its infrared structure (needs --root and --table)")
    parser.add_argument("--compare-to",
                        help="the released model the release decision is measured against (default: --base with the "
                             "new paths off -- right for a base whose storm heads were trained, not for a stage-one base)")
    args = parser.parse_args(argv)

    from .engine import setup_distributed
    from .era5 import Normalizer, lead_offsets
    from .ibtracs import parse_ibtracs
    from .sattrain import TEST_SEASONS, VALIDATION_SEASONS, merged_model, staged_grid
    from .satdata import GlobalFrames, StormPatches
    from .state import StateStepper, channel_roles

    rank, _, world, device = setup_distributed()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # No trained retrieval, no retrieval: an untrained one in the loop would hand the heads noise for a share of samples.
    model = merged_model(args.base, args.satnet, args.retrieval, **({} if args.retrieval else {"retrieval_layers": 0})).to(device)
    meta = json.loads(Path(args.era5).with_suffix(".json").read_text())
    variables = list(meta["variables"])
    grid = model._aligned(staged_grid(meta))
    tendency = torch.load(args.tendency) if args.tendency else model.state_tendency.detach().cpu().clone()
    if float(tendency.abs().sum()) == 0:
        raise ValueError("no tendency scale: pass --tendency, or start from a checkpoint that records one")
    stepper = StateStepper(channel_roles(variables, model.config.analysis_channels), tendency,
                           Normalizer.from_dict(meta["statistics"]), grid,
                           variables.index("surface_pressure") if "surface_pressure" in variables else None)
    table = dict(np.load(args.table, allow_pickle=False)) if args.table else None
    frames = GlobalFrames(args.root) if args.root and model.retrieval is not None and args.retrieval_share > 0 else None
    patches = None
    if args.patches_in_heads:
        if not (args.root and args.table):
            raise ValueError("--patches-in-heads needs --root (the staged satellite record) and --table (the label table)")
        patches = StormPatches(args.root)
    satnet = None
    if table is not None and args.root and model.trained_heads()["satnet"] > 0:
        cache = output / "satnet_outputs.pt"
        if not cache.exists():
            outputs = satnet_outputs_for_table(model, table, StormPatches(args.root), part=output / "satnet_part")
            if is_main():
                torch.save(outputs, cache)
            barrier()
        satnet = torch.load(cache, weights_only=False)
        if is_main():
            print(f"[storms-b] storm-network outputs for {int(satnet['present'].sum()):,} fixes", flush=True)

    tracks = parse_ibtracs(args.ibtracs, since=1980)
    times = np.asarray(meta["time"], dtype=np.int64)
    items = storm_times(tracks, times, lead_offsets(model.config.lead_times_hours), model.config.history_frames)
    year = (lambda start: int(np.datetime64(int(times[start]), "s").astype("datetime64[Y]").astype(int) + 1970))
    train_items = [it for it in items if year(it[0]) not in VALIDATION_SEASONS + TEST_SEASONS]
    val_items = [it for it in items if year(it[0]) in VALIDATION_SEASONS]
    test_items = [it for it in items if year(it[0]) in TEST_SEASONS]
    settings = EngineSettings(run="stage-b-storms", output=str(output), hours=args.hours, batch_size=args.batch_size,
                              learning_rate=args.lr, workers=args.workers, hub_repo=args.hub_repo,
                              max_steps=args.max_steps, warmup_steps=200, checkpoint_minutes=20, eval_minutes=30)
    if is_main():
        extras = ((["storm-centred infrared into the heads"] if patches is not None else [])
                  + (["ocean heat content and storm size"] if model.config.storm_ocean else [])
                  + (["infrared structure"] if model.config.storm_ir and patches is not None else []))
        print(f"[storms-b] inputs beyond V1: {', '.join(extras) or 'none'}; satellite network outputs "
              f"{'fused' if satnet is not None else 'absent'}; retrieval {'in the loop' if frames is not None else 'off'}", flush=True)
    train_storms(model, settings, args.era5, stepper, grid, variables, train_items, val_items, satnet=satnet,
                 table=table, frames=frames, tail=args.tail, retrieval_share=args.retrieval_share, patches=patches)

    rollout = not args.no_rollout_members
    # Historical analogs from the training seasons only: a held-out storm never finds itself.
    from .tracker import AnalogForecaster

    analog_tracks = [t for t in tracks if t.year not in VALIDATION_SEASONS + TEST_SEASONS]
    analog = AnalogForecaster.from_tracks(analog_tracks, model.config.lead_times_hours)
    if is_main():
        print(f"[storms-b] analog database: {analog.features.shape[0]:,} fixes from {len(analog_tracks):,} storms",
              flush=True)
    rng = np.random.default_rng(0)
    monitor_items = [train_items[i] for i in sorted(rng.choice(len(train_items), max(1, int(args.monitor_share * len(train_items))),
                                                                replace=False))]
    banks = {}
    for name, chosen in (("val", val_items), ("test", test_items), ("monitor", monitor_items)):
        banks[name] = memmap_feature_bank(model, args.era5, chosen, stepper, grid, variables, satnet=satnet, table=table,
                                          rollout=rollout, part=output / f"bank_{name}", analog=analog,
                                          ensemble_members=args.ensemble_members, rotations=args.rotations,
                                          patches=patches)
    # The model it started from, on the same test storms, for the comparison the release decision uses.
    from .sattrain import build_model

    old = build_model(args.compare_to or args.base, satnet_embed=0, retrieval_layers=0, ri2=False,
                      hemisphere_frame=False).to(device)
    old_bank = memmap_feature_bank(old, args.era5, test_items, stepper, grid, variables, rollout=False,
                                   part=output / "bank_old")
    if is_main():
        report = finalise_storms(model, banks["val"], banks["test"], banks["monitor"])
        comparison = compare_models(model, banks["test"], old, old_bank)
        (output / "comparison.json").write_text(json.dumps(comparison, indent=2))
        report += "\n\nAGAINST THE MODEL IT STARTED FROM (same test storms):\n" + describe_comparison(comparison)
        print(report, flush=True)
        for name in ("val", "test"):
            torch.save(banks[name], output / f"bank_{name}.pt")
        save_release(model, output, "stage-b", args.hub_repo, report)
        print(f"[storms-b] finished: {output / 'stage-b' / 'latest.pt'}", flush=True)
    barrier()


if __name__ == "__main__":
    main()
