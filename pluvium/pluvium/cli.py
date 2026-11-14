# Copyright 2026 Nathan. Apache-2.0.
"""``pluvium forecast | place | program | info``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_REPO = "CorxLabs/Pluvium-500"


def fetch_model(repo: str, cache: str = "~/.cache/pluvium/models") -> Path:
    """Download (once) a published Pluvium from the Hub; returns its folder."""
    from huggingface_hub import snapshot_download

    target = Path(cache).expanduser() / repo.replace("/", "--")
    snapshot_download(repo, local_dir=str(target), repo_type="model",
                      allow_patterns=["model.pt", "pluvium.json", "climatology.npz", "towns.npz", "geography/*",
                                      "README.md"])
    return target


def _forecaster(args):
    from .forecast import Forecaster

    folder = Path(args.model).expanduser() if args.model else fetch_model(args.repo)
    return Forecaster(folder, device=args.device)


def _models(args) -> tuple:
    from .sources.nwp import ensemble

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    return models + (tuple(ensemble(args.ensemble)) if getattr(args, "ensemble", 0) else ())


def cmd_forecast(args) -> None:
    fc = _forecaster(args)
    models = _models(args)
    result = fc.run(now=args.time, models=models, use_memory=not args.no_memory, out_dir=args.out,
                    countries=tuple(args.country.upper().split(",")) if args.country else None)
    print((Path(args.out).expanduser() / "report.txt").read_text())
    del result


def cmd_place(args) -> None:
    from .forecast import place_timeline

    fc = _forecaster(args)
    idx = fc.towns.find(args.name, args.country)
    if idx.size == 0:
        sys.exit(f"no place matching {args.name!r}")
    from .domain import Gazetteer

    one = Gazetteer(fc.towns.points.subset(idx[:1]))
    fc.towns = one
    fc._town_setup()
    models = _models(args)
    result = fc.run(now=args.time, models=models, use_memory=not args.no_memory, flood=False, archive=False)
    print(place_timeline(one, result["summary"], args.name, hours=args.hours))


def cmd_learn(args) -> None:
    """Check archived forecasts against what the airports reported since, and update the online corrections."""
    import time

    from .online import ForecastArchive, OnlineCorrector, learn
    from .sources.metar import fetch_metars, report_hour, station_codes
    from .train import read_meta

    folder = Path(args.model).expanduser() if args.model else fetch_model(args.repo)
    state = Path(args.state).expanduser()
    meta = read_meta(folder)
    stations = (meta.get("stations") or {}).get("list") or []
    if not stations:
        sys.exit("this model has no airport list to verify against")
    now = report_hour(time.time() + 300) + 1
    metars = fetch_metars(hours=72)
    corrector = OnlineCorrector.load(state / "corrector.json")
    got = learn(ForecastArchive(state / "archive"), corrector, lambda hours: station_codes(stations, metars, hours), now)
    corrector.save(state / "corrector.json")
    print(f"verified {got['hours_verified']:,} airport-hours from {got['forecasts']} forecasts; "
          f"{len(corrector.params)} places corrected")


def cmd_info(args) -> None:
    import json

    from .train import read_meta

    folder = Path(args.model).expanduser() if args.model else fetch_model(args.repo)
    meta = read_meta(folder)
    print(json.dumps({k: meta.get(k) for k in ("config", "parameters", "data", "scores", "truth_sources")}, indent=1))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="pluvium", description="Hour-by-hour rain forecasts for Caribbean towns.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--repo", default=DEFAULT_REPO, help="published model on the Hugging Face Hub")
        p.add_argument("--model", default=None, help="a local model folder instead of --repo")
        p.add_argument("--device", default=None)
        p.add_argument("--models", default="gfs,ifs,aifs", help="driving models to mix")
        p.add_argument("--ensemble", type=int, default=0, help="also mix this many GFS ensemble members (up to 31)")
        p.add_argument("--time", default=None, help="issue time (UTC, e.g. 2026-11-14T06); default: now")
        p.add_argument("--no-memory", action="store_true", help="do not read the rain already falling (GOES)")

    p = sub.add_parser("forecast", help="forecast every town; tables and a report")
    common(p)
    p.add_argument("--out", default="pluvium-forecast")
    p.add_argument("--country", default=None, help="limit the report to these country codes, e.g. JM,TT")
    p.set_defaults(func=cmd_forecast)
    p = sub.add_parser("place", help="hour by hour for one town")
    common(p)
    p.add_argument("name")
    p.add_argument("--country", default=None)
    p.add_argument("--hours", type=int, default=48)
    p.set_defaults(func=cmd_place)
    p = sub.add_parser("info", help="what a published model is and how it scored")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--model", default=None)
    p.set_defaults(func=cmd_info)
    p = sub.add_parser("learn", help="check past forecasts against airport reports and update the corrections")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--model", default=None)
    p.add_argument("--state", default="~/.cache/pluvium/live")
    p.set_defaults(func=cmd_learn)
    p = sub.add_parser("program", help="train, calibrate, score and publish on one GPU (see pluvium.program)")
    p.set_defaults(func=None)
    p = sub.add_parser("bench", help="train and score architectures on held-out data (see pluvium.bench)")
    p.set_defaults(func=None)
    args, rest = parser.parse_known_args(argv)
    if args.command == "program":
        from .program import main as program_main

        program_main(rest)
        return
    if args.command == "bench":
        from .bench import main as bench_main

        bench_main(rest)
        return
    args.func(args)


if __name__ == "__main__":
    main()
