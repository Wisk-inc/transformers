# Copyright 2026 Nathan. Apache-2.0.
"""
Stage A: train the satellite networks on 8 GPUs inside a fixed time budget.

Two runs, meant to go side by side on one 8-GPU machine (four GPUs each) for the same three hours:

* ``satnet`` -- :class:`naturev1.satnet.SatNetStorm` on every storm fix since 1980 with a staged patch
  loop: intensity, centre, structure, intensity change, RI, motion.
* ``retrieval`` -- :class:`naturev1.satnet.SatelliteRetrieval` on every six-hourly ERA5 state since 1980
  with a staged global frame: bring a 6-48 h old state up to now.

Seasons 2017, 2019 and 2021 are held out for evaluation during training (the same seasons stage two held
out), and 2018, 2020 and 2022 are never touched -- they are the test set the final report uses.

Run either with ``torchrun --nproc_per_node N -m naturev1.sattrain satnet|retrieval --root ... --era5 ...``;
the B300 runbook (``b300_naturev1.py``) launches both with the right arguments.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .engine import Engine, EngineSettings, is_main
from .satnet import ri_threshold_list, satnet_loss, satnet_targets_ri

#: Held out during training; scored in the logs.
VALIDATION_SEASONS = (2017, 2019, 2021)
#: Never used until the final report.
TEST_SEASONS = (2018, 2020, 2022)
#: Variables the retrieval report shows, in physical units.
REPORT_VARIABLES = ("geopotential@500", "temperature@850", "u_component_of_wind@850", "v_component_of_wind@850",
                    "u_component_of_wind@200", "v_component_of_wind@200", "specific_humidity@700",
                    "mean_sea_level_pressure", "2m_temperature", "total_column_water_vapour")


def _sum_all(values: dict, device) -> dict:
    """Sum every scalar over ranks, over the union of their keys."""
    from .engine import all_sum

    return all_sum(values, device)


def _add(totals: dict, key: str, value) -> None:
    totals[key] = totals.get(key, 0.0) + float(value)


# ---------------------------------------------------------------------------------------- SatNet --

@torch.no_grad()
def evaluate_satnet(engine: Engine, network, loader, max_batches: int | None = None) -> dict:
    """
    Held-out scores of the storm network, with the baseline each one must beat.

    * wind: MAE, RMSE and bias against IBTrACS, and bias by intensity class (the red team's finding was
      a 10-18 kt low bias for major hurricanes);
    * centre: the fix error against the first-guess error it started from;
    * RI (24 h, 30 kt): Brier skill against the base rate;
    * 24 h intensity change: MAE against "no change";
    * 12 h motion: error against extrapolating the last six hours.
    """
    network.eval()
    totals: dict[str, float] = {}
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = engine.to_device(batch)
        with engine.autocast():
            out = network(batch["codes"], batch["available"], batch["latitude"], batch["context"])
        out = {k: v.float() for k, v in out.items()}
        wind = batch["wind_kt"].float()
        ok = torch.isfinite(wind)
        error = out["wind_kt"][ok] - wind[ok]
        _add(totals, "wind_n", ok.sum())
        _add(totals, "wind_abs", error.abs().sum())
        _add(totals, "wind_sq", (error ** 2).sum())
        _add(totals, "wind_bias", error.sum())
        for name, low, high in (("ts", 34, 64), ("cat12", 64, 96), ("cat35", 96, 400)):
            band = ok & (wind >= low) & (wind < high)
            _add(totals, f"{name}_n", band.sum())
            _add(totals, f"{name}_bias", (out["wind_kt"][band] - wind[band]).sum())
            _add(totals, f"{name}_abs", (out["wind_kt"][band] - wind[band]).abs().sum())
        pressure = batch["pressure_hpa"].float()
        good = torch.isfinite(pressure)
        _add(totals, "pres_n", good.sum())
        _add(totals, "pres_abs", (out["pressure_hpa"][good] - pressure[good]).abs().sum())
        truth = batch["centre_offset_km"].float()
        _add(totals, "centre_n", truth.shape[0])
        _add(totals, "centre_err", (out["centre_offset_km"] - truth).norm(dim=-1).sum())
        _add(totals, "guess_err", truth.norm(dim=-1).sum())
        targets, masks = satnet_targets_ri(batch["wind_change_kt"].float(), batch["tropical"])
        column = ri_threshold_list().index((24, 30.0))
        probability = torch.sigmoid(out["ri_logits"][:, column])
        valid = masks[:, column] > 0
        _add(totals, "ri_n", valid.sum())
        _add(totals, "ri_pos", targets[valid, column].sum())
        _add(totals, "ri_brier", ((probability[valid] - targets[valid, column]) ** 2).sum())
        change = batch["wind_change_kt"][:, 1].float()
        live = torch.isfinite(change) & batch["tropical"].bool()
        _add(totals, "chg_n", live.sum())
        _add(totals, "chg_abs", (out["wind_change_kt"][live, 1] - change[live]).abs().sum())
        _add(totals, "chg_persist", change[live].abs().sum())
        future, past = batch["future_motion_km"].float(), batch["past_motion_km"].float()
        moving = torch.isfinite(future).all(-1) & torch.isfinite(past).all(-1)
        _add(totals, "mot_n", moving.sum())
        _add(totals, "mot_err", (out["future_motion_km"][moving] - future[moving]).norm(dim=-1).sum())
        _add(totals, "mot_extrap", (2.0 * past[moving] - future[moving]).norm(dim=-1).sum())
    totals = _sum_all(totals, engine.device)
    network.train()
    n = (lambda key: max(totals.get(key, 0.0), 1.0))
    base = totals.get("ri_pos", 0.0) / n("ri_n")
    climatology = base * (1 - base)
    report = {
        "wind_mae_kt": totals.get("wind_abs", 0.0) / n("wind_n"),
        "wind_rmse_kt": math.sqrt(totals.get("wind_sq", 0.0) / n("wind_n")),
        "wind_bias_kt": totals.get("wind_bias", 0.0) / n("wind_n"),
        "pressure_mae_hpa": totals.get("pres_abs", 0.0) / n("pres_n"),
        "centre_error_km": totals.get("centre_err", 0.0) / n("centre_n"),
        "first_guess_error_km": totals.get("guess_err", 0.0) / n("centre_n"),
        "ri30_brier": totals.get("ri_brier", 0.0) / n("ri_n"),
        "ri30_brier_skill": 1.0 - (totals.get("ri_brier", 0.0) / n("ri_n")) / max(climatology, 1e-9),
        "ri30_base_rate": base,
        "change24_mae_kt": totals.get("chg_abs", 0.0) / n("chg_n"),
        "change24_persistence_mae_kt": totals.get("chg_persist", 0.0) / n("chg_n"),
        "motion12_error_km": totals.get("mot_err", 0.0) / n("mot_n"),
        "motion12_extrapolation_km": totals.get("mot_extrap", 0.0) / n("mot_n"),
    }
    for name in ("ts", "cat12", "cat35"):
        report[f"{name}_bias_kt"] = totals.get(f"{name}_bias", 0.0) / n(f"{name}_n")
        report[f"{name}_mae_kt"] = totals.get(f"{name}_abs", 0.0) / n(f"{name}_n")
    report["loss"] = report["wind_mae_kt"]
    return report


def describe_satnet(report: dict) -> str:
    r = report
    return "\n".join([
        f"  wind      MAE {r['wind_mae_kt']:.1f} kt  RMSE {r['wind_rmse_kt']:.1f}  bias {r['wind_bias_kt']:+.1f}"
        f"  | bias TS {r['ts_bias_kt']:+.1f}  Cat1-2 {r['cat12_bias_kt']:+.1f}  Cat3-5 {r['cat35_bias_kt']:+.1f} kt",
        f"  pressure  MAE {r['pressure_mae_hpa']:.1f} hPa",
        f"  centre    fix error {r['centre_error_km']:.0f} km  (first guess was {r['first_guess_error_km']:.0f} km off)",
        f"  RI 30 kt  Brier {r['ri30_brier']:.4f}  skill vs base rate {r['ri30_brier_skill']:+.3f}"
        f"  (base rate {100 * r['ri30_base_rate']:.1f}%)",
        f"  24 h change MAE {r['change24_mae_kt']:.1f} kt  (no-change {r['change24_persistence_mae_kt']:.1f})",
        f"  12 h motion error {r['motion12_error_km']:.0f} km  (extrapolation {r['motion12_extrapolation_km']:.0f})",
    ])


def train_satnet(model, settings: EngineSettings, table: dict, patches, train_rows, val_rows) -> dict:
    """Stage A, storm network. ``model`` is a :class:`naturev1.NatureV1` built with ``satnet_embed``."""
    from .satdata import StormSequenceDataset

    if model.storm_satnet is None:
        raise ValueError("the model has no storm network; build it with NatureConfig(satnet_embed=512)")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.storm_satnet.parameters():
        parameter.requires_grad = True
    engine = Engine(settings, model, model.storm_satnet, config=model.config.to_dict())
    train = StormSequenceDataset(table, patches, train_rows, train=True, seed=settings.seed)
    held = StormSequenceDataset(table, patches, val_rows, train=False, seed=settings.seed)
    loader = engine.loader(train)
    val_loader = engine.loader(held, shuffle=False, drop_last=False)
    engine.resume()
    if is_main():
        print(f"[satnet] {len(train):,} training sequences, {len(held):,} held out; "
              f"{model.storm_satnet.num_parameters() / 1e6:.1f}M parameters; {engine.world} GPU(s), "
              f"batch {settings.batch_size} per GPU", flush=True)
    report = {}
    stop = False
    while not stop:
        engine.set_epoch(loader, engine.epoch)
        for batch in loader:
            batch = engine.to_device(batch)
            with engine.autocast():
                out = engine.model(batch["codes"], batch["available"], batch["latitude"], batch["context"])
            loss, parts = satnet_loss({k: v.float() for k, v in out.items()}, batch)
            engine.backward_step(loss, parts)
            model.mark_trained({"satnet": 1.0})
            stop, save, evaluate = engine.poll()
            if evaluate:
                report = evaluate_satnet(engine, engine.model, val_loader, max_batches=200)
                engine.last_eval = time.time()
                engine.history.append({"step": engine.step, **report})
                if is_main():
                    print(f"[satnet] held-out at step {engine.step:,}:\n{describe_satnet(report)}", flush=True)
            if save:
                engine.checkpoint(report)
            if stop:
                break
        engine.epoch += 1
    report = evaluate_satnet(engine, engine.model, val_loader)
    if is_main():
        print(f"[satnet] final held-out scores ({len(held):,} sequences):\n{describe_satnet(report)}", flush=True)
    engine.history.append({"step": engine.step, "final": True, **report})
    engine.finish(report)
    return report


# ------------------------------------------------------------------------------------- retrieval --

@torch.no_grad()
def evaluate_retrieval(engine: Engine, model, loader, stepper, grid, max_batches: int | None = None) -> dict:
    """
    Held-out retrieval scores: skill (share of the change since the background that was recovered) per
    background age, and RMSE in physical units for the variables a forecaster reads, retrieved vs stale.
    """
    from .satnet import apply_retrieval, retrieval_targets

    model.retrieval.eval()
    names = stepper.roles.channels
    report_channels = [(name, names.index(name)) for name in REPORT_VARIABLES if name in names]
    area = grid.weights.to(engine.device, torch.float64)
    area = area / area.sum()
    totals: dict[str, float] = {}
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = engine.to_device(batch)
        with engine.autocast():
            out = engine.model(batch["satellite"], batch["satellite_available"], batch["background"],
                               batch["stale_hours"], batch["calendar"], grid, model.latent_grid, model.link)
        out = {k: v.float() for k, v in out.items()}
        truth = retrieval_targets(stepper, batch["background"], batch["target"], batch["stale_hours"])
        retrieved = apply_retrieval(stepper, batch["background"], out, batch["stale_hours"], batch["valid_time"])
        for b in range(truth.shape[0]):
            stale = int(batch["stale_hours"][b])
            err = ((truth[b] - out["increment"][b]) ** 2 * area.view(-1, 1)).sum()
            base = ((truth[b] ** 2) * area.view(-1, 1)).sum()
            _add(totals, f"err_{stale}", err)
            _add(totals, f"base_{stale}", base)
            _add(totals, f"n_{stale}", 1)
            for name, channel in report_channels:
                decode = (lambda x: torch.as_tensor(stepper.normalizer.decode(x.double(), name)))
                actual = decode(batch["target"][b, :, channel])
                mine = decode(retrieved[b, :, channel])
                old = decode(batch["background"][b, :, channel])
                _add(totals, f"{name}|ret|{stale}", (((mine - actual) ** 2) * area).sum())
                _add(totals, f"{name}|stale|{stale}", (((old - actual) ** 2) * area).sum())
    totals = _sum_all(totals, engine.device)
    model.retrieval.train()
    stales = sorted({int(k.split("_")[1]) for k in totals if k.startswith("n_")})
    report = {"by_age": {}}
    all_err = all_base = 0.0
    for stale in stales:
        count = max(totals[f"n_{stale}"], 1.0)
        entry = {"samples": int(totals[f"n_{stale}"]),
                 "skill": 1.0 - totals[f"err_{stale}"] / max(totals[f"base_{stale}"], 1e-12), "rmse": {}}
        for name, _ in report_channels:
            entry["rmse"][name] = (math.sqrt(totals[f"{name}|ret|{stale}"] / count),
                                   math.sqrt(totals[f"{name}|stale|{stale}"] / count))
        report["by_age"][stale] = entry
        all_err += totals[f"err_{stale}"]
        all_base += totals[f"base_{stale}"]
    report["skill"] = 1.0 - all_err / max(all_base, 1e-12)
    report["loss"] = -report["skill"]
    return report


def describe_retrieval(report: dict) -> str:
    lines = [f"  overall skill {report['skill']:+.3f}  (share of the change since the stale state recovered)"]
    for stale, entry in sorted(report["by_age"].items()):
        parts = "  ".join(f"{name.split('@')[0][:4]}{('@' + name.split('@')[1]) if '@' in name else ''} "
                          f"{mine:.3g}/{old:.3g}" for name, (mine, old) in list(entry["rmse"].items())[:6])
        lines.append(f"  {stale:>2} h old  skill {entry['skill']:+.3f}  RMSE retrieved/stale: {parts}")
    return "\n".join(lines)


def train_retrieval(model, settings: EngineSettings, frames, era5_path, stepper, grid, train_times, val_times,
                    stale_hours=(6, 12, 24, 48), warm_start: bool = True) -> dict:
    """Stage A, global retrieval. ``model`` is a :class:`naturev1.NatureV1` built with ``retrieval_layers``."""
    from .satdata import RetrievalDataset
    from .satnet import retrieval_loss
    from .state import channel_loss_weights

    if model.retrieval is None:
        raise ValueError("the model has no retrieval; build it with NatureConfig(retrieval_layers=6)")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.retrieval.named_parameters():
        # The retrieval reads one instant, so its blocks' temporal strands never run; left trainable they
        # would receive no gradient, which data parallelism treats as an error.
        parameter.requires_grad = ".temporal." not in name
    if warm_start and int(model.head_steps[list(model.trained_heads()).index("retrieval")]) == 0:
        copied = model.warm_start_retrieval()
        if is_main():
            print(f"[retrieval] warm start from the backbone: {len(copied)} modules copied "
                  f"({sum(c == 'IHelixBlock' for c in copied)} blocks)", flush=True)
    engine = Engine(settings, model, model.retrieval, config=model.config.to_dict())
    grid = model._aligned(grid)
    channels = model.config.analysis_channels
    train = RetrievalDataset(frames, era5_path, train_times, stale_hours, channels, train=True, seed=settings.seed)
    held = RetrievalDataset(frames, era5_path, val_times, stale_hours, channels, train=False, seed=settings.seed)
    loader = engine.loader(train)
    val_loader = engine.loader(held, shuffle=False, drop_last=False)
    weights = channel_loss_weights(stepper.roles.channels, model.config.analysis_channels)[stepper.roles.prognostic]
    weights = (weights / weights.mean()).to(engine.device)
    area = grid.weights.to(engine.device, torch.float32)
    area = area / area.mean()
    engine.resume()
    if is_main():
        print(f"[retrieval] {len(train):,} training states, {len(held):,} held out; "
              f"{sum(p.numel() for p in model.retrieval.parameters()) / 1e6:.1f}M parameters; "
              f"{engine.world} GPU(s), batch {settings.batch_size} per GPU", flush=True)
    report = {}
    stop = False
    while not stop:
        engine.set_epoch(loader, engine.epoch)
        for batch in loader:
            batch = engine.to_device(batch)
            with engine.autocast():
                out = engine.model(batch["satellite"], batch["satellite_available"], batch["background"],
                                   batch["stale_hours"], batch["calendar"], grid, model.latent_grid, model.link)
            loss, parts = retrieval_loss({k: v.float() for k, v in out.items()}, stepper, batch["background"],
                                         batch["target"], batch["stale_hours"], area, weights)
            engine.backward_step(loss, parts)
            model.mark_trained({"retrieval": 1.0})
            stop, save, evaluate = engine.poll()
            if evaluate:
                report = evaluate_retrieval(engine, model, val_loader, stepper, grid, max_batches=50)
                engine.last_eval = time.time()
                engine.history.append({"step": engine.step, "skill": report["skill"]})
                if is_main():
                    print(f"[retrieval] held-out at step {engine.step:,}:\n{describe_retrieval(report)}", flush=True)
            if save:
                engine.checkpoint({"skill": report.get("skill")})
            if stop:
                break
        engine.epoch += 1
    report = evaluate_retrieval(engine, model, val_loader, stepper, grid)
    if is_main():
        print(f"[retrieval] final held-out scores ({len(held):,} states):\n{describe_retrieval(report)}", flush=True)
    engine.history.append({"step": engine.step, "final": True, "skill": report["skill"]})
    engine.finish({"skill": report["skill"], "by_age": {str(k): v["skill"] for k, v in report["by_age"].items()}})
    return report


# ------------------------------------------------------------------------------------------- CLI --

def _seasons(times: np.ndarray) -> np.ndarray:
    return times.astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970


def build_model(checkpoint: str, satnet_embed: int = 512, retrieval_layers: int = 6, ri2: bool = True,
                hemisphere_frame: bool = True):
    """The 0.12 model from a released checkpoint (stage two or three), with the new paths switched on."""
    from ihelix import fibonacci_sphere

    from .model import NatureConfig, NatureV1

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = NatureConfig.from_dict({**payload["config"], "satnet_embed": satnet_embed,
                                     "retrieval_layers": retrieval_layers, "ri2": ri2,
                                     "hemisphere_frame": hemisphere_frame})
    grid = fibonacci_sphere(config.latent_points, num_neighbours=config.latent_neighbours,
                            cluster_size=config.latent_cluster)
    model = NatureV1(config, grid)
    model.load_state_dict(payload["model"])
    return model


def merged_model(base: str, satnet: str | None = None, retrieval: str | None = None, **switches):
    """
    The model after stage A: the released weights, with the storm network from the ``satnet`` run and the
    retrieval from the ``retrieval`` run (the two run side by side on separate GPUs, so each checkpoint
    holds only its own network trained). Training counts are carried over for what each run trained.
    """
    from .model import TRACKED_HEADS

    model = build_model(base, **switches)
    state = model.state_dict()
    for path, prefix, head in ((satnet, "storm_satnet.", "satnet"), (retrieval, "retrieval.", "retrieval")):
        if not path:
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)["model"]
        taken = {k: v for k, v in payload.items() if k.startswith(prefix)}
        if not taken:
            raise ValueError(f"{path} holds no {prefix[:-1]} weights")
        state.update(taken)
        steps = payload.get("head_steps")
        if steps is not None and steps.numel() > TRACKED_HEADS.index(head):
            state["head_steps"][TRACKED_HEADS.index(head)] = steps[TRACKED_HEADS.index(head)]
    model.load_state_dict(state)
    return model


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="NatureV1 stage A: the satellite networks")
    parser.add_argument("what", choices=("satnet", "retrieval"))
    parser.add_argument("--checkpoint", required=True, help="the model to start from (stage two or later)")
    parser.add_argument("--root", required=True, help="staged satellite root (naturev1.satstage)")
    parser.add_argument("--table", required=True, help="label table .npz (naturev1.ibtracs.storm_label_table)")
    parser.add_argument("--era5", help="staged ERA5 memmap covering 1980-2022 (retrieval)")
    parser.add_argument("--tendency", help="tendency.pt (six-hour spread per channel) for the retrieval")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--hub-repo", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--checkpoint-minutes", type=float, default=20.0)
    parser.add_argument("--eval-minutes", type=float, default=30.0)
    args = parser.parse_args(argv)

    from .satdata import GlobalFrames, StormPatches, season_rows

    model = build_model(args.checkpoint)
    table = dict(np.load(args.table, allow_pickle=False))
    train_seasons = [y for y in range(1980, 2025) if y not in VALIDATION_SEASONS + TEST_SEASONS]
    if args.what == "satnet":
        settings = EngineSettings(run="stage-a-satnet", output=args.output, hours=args.hours,
                                  batch_size=args.batch_size or 128, learning_rate=args.lr or 4e-4,
                                  workers=args.workers, hub_repo=args.hub_repo, max_steps=args.max_steps,
                                  compile=args.compile, checkpoint_minutes=args.checkpoint_minutes,
                                  eval_minutes=args.eval_minutes, warmup_steps=1000)
        patches = StormPatches(args.root)
        train_satnet(model, settings, table, patches, season_rows(table, train_seasons),
                     season_rows(table, VALIDATION_SEASONS))
        return
    from .state import StateStepper, channel_roles

    meta = json.loads(Path(args.era5).with_suffix(".json").read_text())
    from .era5 import Normalizer

    normalizer = Normalizer.from_dict(meta["statistics"])
    variables = list(meta["variables"])
    grid = staged_grid(meta)
    tendency = torch.load(args.tendency) if args.tendency else model.state_tendency.clone()
    if float(tendency.abs().sum()) == 0:
        raise ValueError("no tendency scale: pass --tendency, or start from a checkpoint that records one")
    roles = channel_roles(variables, model.config.analysis_channels)
    mass = variables.index("surface_pressure") if "surface_pressure" in variables else None
    stepper = StateStepper(roles, tendency, normalizer, grid, mass)
    stamps = np.asarray(meta["time"], dtype=np.int64)
    years = _seasons(stamps)
    frames = GlobalFrames(args.root)
    settings = EngineSettings(run="stage-a-retrieval", output=args.output, hours=args.hours,
                              batch_size=args.batch_size or 8, learning_rate=args.lr or 2e-4, workers=args.workers,
                              hub_repo=args.hub_repo, max_steps=args.max_steps, compile=args.compile,
                              checkpoint_minutes=args.checkpoint_minutes, eval_minutes=args.eval_minutes,
                              warmup_steps=300)
    train_retrieval(model, settings, frames, args.era5, stepper, grid,
                    stamps[np.isin(years, train_seasons)], stamps[np.isin(years, VALIDATION_SEASONS)])


def staged_grid(meta: dict):
    """The analysis grid of a staged ERA5 memmap, from its sidecar (equiangular, with poles)."""
    from ihelix import FieldGrid, Geometry

    from .era5 import equiangular_weights

    shape = meta["shape"]
    rows, cols = int(shape[1]), int(shape[2])
    # WeatherBench2's equiangular stores run south to north from the pole, 0 to 360 east; stagings since
    # 0.12 record their axes, older ones are those of the store they came from.
    latitudes = meta.get("latitude") or np.linspace(-90.0, 90.0, rows).tolist()
    longitudes = meta.get("longitude") or (np.arange(cols) * 360.0 / cols).tolist()
    # Exactly era5_source_grid's construction, so the grid -- and its fingerprint -- is the one the backbone
    # was trained on.
    from .corpora import latlon_to_coords

    coords = latlon_to_coords(np.asarray(latitudes, np.float64), np.asarray(longitudes, np.float64))
    weights = torch.tensor(equiangular_weights(np.asarray(latitudes, np.float64), len(longitudes)))
    return FieldGrid.source(coords, Geometry.globe(), weights=weights)


if __name__ == "__main__":
    main()
