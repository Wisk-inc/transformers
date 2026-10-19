# Copyright 2026 Nathan. Apache-2.0.
"""
Command line for distillation: stage the teachers, then train NatureV1 against them (one GPU or many).

    python -m naturev1.distill.run --base base.pt --era5 era5.npy --root distill --output runs \
        --teachers pangu,hres,graphcast,aurora,fgn --years 2018-2022 --hours 8 --hub-repo owner/repo

The teacher years overlap the hindcast test seasons on purpose only in *staging*: the split below keeps every
2018, 2020, 2021 and 2022 initial time for validation and testing, and trains on the rest of what the teachers
cover (2016, 2017, 2019 for HRES; 2019 and parts of others) -- so the benchmark seasons stay unseen. Pass
``--train-years`` to change that.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _years(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in text.split(",")]


def storm_centres(ibtracs: str | None, times) -> dict:
    """``{unix seconds: [(lat, lon), ...]}`` of tropical cyclones active at each initial time."""
    if not ibtracs:
        return {}
    from ..ibtracs import parse_ibtracs

    wanted = set(int(t) for t in times)
    out: dict = {}
    years = sorted({int(str(np.datetime64(int(t), "s"))[:4]) for t in wanted})
    for track in parse_ibtracs(ibtracs, since=years[0] - 1, until=years[-1] + 1):
        seconds = track.time.astype("datetime64[s]").astype(np.int64)
        for k, s in enumerate(seconds):
            if int(s) in wanted and track.status[k] in ("TD", "TS", "HU", "SS"):
                out.setdefault(int(s), []).append((float(track.latitude[k]), float(track.longitude[k])))
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--era5", required=True, help="staged ERA5 memmap (naturev1.b300.stage_era5_record)")
    parser.add_argument("--root", required=True, help="where staged teacher forecasts live")
    parser.add_argument("--output", required=True)
    parser.add_argument("--teachers", default="pangu,hres,graphcast,aurora,fgn")
    parser.add_argument("--years", default="2016-2022")
    parser.add_argument("--train-years", default="2016,2017,2019")
    parser.add_argument("--val-years", default="2021")
    parser.add_argument("--leads", default="6,12,18,24,36,48,72")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.3, help="weight of the ERA5 truth anchor")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hub-repo", default=None)
    parser.add_argument("--ibtracs", default=None, help="IBTrACS CSV, to up-weight cells near active cyclones")
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--init-hours", default="0,12",
                        help="teacher initial times to stage and train on; '0' halves the disk (~13 GB a teacher-year)")
    args = parser.parse_args(argv)
    init_hours = [int(h) for h in str(args.init_hours).split(",") if h.strip()]

    from ..era5 import Normalizer
    from .catalog import TEACHERS
    from .stage import stage_teacher

    meta = json.loads(Path(args.era5).with_suffix(".json").read_text())
    normalizer = Normalizer.from_dict(meta["statistics"])
    channels = list(normalizer.variables)
    leads = tuple(int(h) for h in args.leads.split(","))
    names = [n.strip() for n in args.teachers.split(",") if n.strip()]
    years = _years(args.years)
    dirs = []
    if not args.train_only:
        for name in names:
            for year in years:
                if year not in TEACHERS[name].archive_1p5:
                    continue
                dirs.append(stage_teacher(name, year, channels, normalizer, args.root, leads, init_hours=init_hours))
    else:
        dirs = sorted(p for p in Path(args.root).iterdir() if (p / "meta.json").exists()
                      and p.name.split("_")[0] in names)
    if args.stage_only:
        print("[distill] staged:", [str(d) for d in dirs], flush=True)
        return

    from ..engine import EngineSettings, setup_distributed
    from ..sattrain import build_model, staged_grid
    from ..state import StateStepper, channel_roles
    from .train import train_distillation

    _, _, _, device = setup_distributed()
    model = build_model(args.base, satnet_embed=0, retrieval_layers=0, ri2=False, hemisphere_frame=False).to(device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not name.startswith(("storm", "track", "landfall", "intensity", "ri_", "ri2", "eyewall",
                                                       "satnet", "retrieval", "enso"))
    grid = model._aligned(staged_grid(meta))
    tendency = model.state_tendency.detach().cpu().clone()
    if float(tendency.abs().sum()) == 0:
        raise ValueError("the base checkpoint records no tendency scale")
    variables = list(meta["variables"])
    stepper = StateStepper(channel_roles(variables, model.config.analysis_channels), tendency, normalizer, grid,
                           variables.index("surface_pressure") if "surface_pressure" in variables else None)
    times = np.asarray(meta["time"], np.int64)
    year_of = lambda t: int(str(np.datetime64(int(t), "s"))[:4])
    synoptic = [int(t) for t in times if (int(t) // 3600) % 24 in init_hours]
    train_years, val_years = set(_years(args.train_years)), set(_years(args.val_years))
    train_inits = [np.datetime64(t, "s") for t in synoptic if year_of(t) in train_years]
    val_inits = [np.datetime64(t, "s") for t in synoptic if year_of(t) in val_years]
    storms = storm_centres(args.ibtracs, [t for t in synoptic if year_of(t) in train_years | val_years])
    settings = EngineSettings(run="distill", output=args.output, hours=args.hours, batch_size=args.batch_size,
                              learning_rate=args.lr, workers=args.workers, hub_repo=args.hub_repo, max_steps=args.max_steps,
                              warmup_steps=200, checkpoint_minutes=20, eval_minutes=30)
    report = train_distillation(model, settings, args.era5, [str(d) for d in dirs], stepper, train_inits, val_inits,
                                max_horizon=args.horizon, alpha=args.alpha, storms=storms)
    from ..engine import is_main
    from ..safeio import atomic_torch_save

    if is_main():                       # under torchrun every rank holds the same weights; one writes them
        atomic_torch_save({"model": model.state_dict(), "config": model.config.to_dict(), "report": report},
                          Path(args.output) / "distill" / "distilled.pt")
        print("[distill] final held-out:", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
