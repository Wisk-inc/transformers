# Copyright 2026 Nathan. Apache-2.0.
"""``pluvium forecast | place | program | info``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

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


def cmd_setup(args) -> None:
    """Prepare Pluvium for any country: its area, terrain, coasts, provinces and every town, ready to train."""
    from .area import Area
    from .domain import Geography, build_gazetteer

    root = Path(args.root).expanduser()
    codes = [c.strip() for c in args.country.split(",") if c.strip()]
    box = tuple(float(x) for x in args.box.split(",")) if args.box else None
    area = Area.for_countries(codes, box=box, cache=root / "cache" / "areas")
    root.mkdir(parents=True, exist_ok=True)
    area.save(root / "area.json")
    geo = Geography.build(cache=root / "cache" / "geography", area=area)
    geo.save(root / "geography")
    towns = build_gazetteer(geo.terrain, countries=area.iso2, cache=root / "cache" / "geonames", box=area.target_box)
    towns.points.save(root / "towns.npz")
    s, n, w, e = area.target_box
    print(f"{area.name}: {', '.join(area.countries.values())}")
    print(f"  forecasts for {s:.2f} to {n:.2f} N, {w:.2f} to {e:.2f} E; weather read over {area.domain}")
    print(f"  mesh {len(geo.mesh):,} points, {len(geo.targets):,} truth cells, {len(towns.points):,} places")
    print(f"  live rain now from {area.satellite}; bulletins in "
          + "; ".join(f"{cc}: {', '.join(area.bulletin_languages(cc))}" for cc in area.iso2))
    print(f"next: pluvium program --root {root} --country {','.join(codes)}")


def cmd_blend(args) -> None:
    """Pluvium Blend for any country: train it on a year of archived runs, or forecast 14 days for every town."""
    from .area import Area
    from .blendlive import BlendModel, write_table
    from .domain import Gazetteer, PointSet

    root = Path(args.root).expanduser()
    if args.action == "train":
        codes = [c.strip() for c in (args.country or "").split(",") if c.strip()]
        area = Area.load(root / "area.json") if (root / "area.json").exists() and not codes else \
            Area.for_countries(codes, cache=root / "cache" / "areas")
        root.mkdir(parents=True, exist_ok=True)
        area.save(root / "area.json")
        model = BlendModel.train(area, root, start=args.start, end=args.end)
        print(f"blend for {area.name} trained on {model.meta['rows']:,} rows from {model.meta['runs']} runs; saved in {root / 'blend'}")
        return
    model = BlendModel.load(root / "blend")
    towns_path = root / "towns.npz"
    if towns_path.exists():
        towns = Gazetteer(PointSet.load(towns_path))
        lat, lon, names = towns.points.lat, towns.points.lon, [str(x) for x in towns.names]
    else:                                                     # no town list: the area's grid
        lat_g, lon_g = model.grid
        la, lo = np.meshgrid(lat_g, lon_g, indexing="ij")
        lat, lon = la.ravel(), lo.ravel()
        names = [f"{a:.2f},{b:.2f}" for a, b in zip(lat, lon)]
    now = None
    if args.time:
        from .util import hour_of

        now = hour_of(args.time)
    result = model.forecast(lat, lon, root / "live_archive", now=now)
    out = write_table(result, names, root / f"blend_{result['day0']}.csv")
    print(f"14-day blended forecast from {result['day0']} for {len(names):,} places -> {out}")
    print("runs used: " + ", ".join(f"{k} {v}Z" for k, v in result["runs"].items()))


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
    p = sub.add_parser("blend", help="Pluvium Blend for any country: train on a year of archived runs, then forecast 14 days")
    p.add_argument("action", choices=["train", "forecast"])
    p.add_argument("--country", default=None, help="ISO codes (train; default: the area saved in --root)")
    p.add_argument("--root", default="pluvium-area")
    p.add_argument("--start", default="2024-04-01T00", help="training runs from (train)")
    p.add_argument("--end", default="2025-04-01T00", help="training runs until (train)")
    p.add_argument("--time", default=None, help="forecast as of this UTC hour, e.g. 2026-11-14T12 (default: now)")
    p.set_defaults(func=cmd_blend)
    p = sub.add_parser("setup", help="prepare Pluvium for any country: area, terrain, provinces, towns")
    p.add_argument("--country", required=True, help="ISO codes, e.g. TT or DO,HT")
    p.add_argument("--box", default=None, help="south,north,west,east instead of the countries' own extent")
    p.add_argument("--root", default="pluvium-area", help="folder for the area (then: pluvium program --root ...)")
    p.set_defaults(func=cmd_setup)
    p = sub.add_parser("program", add_help=False, help="train, calibrate, score and publish on one GPU (see pluvium.program)")
    p.set_defaults(func=None)
    p = sub.add_parser("bench", add_help=False, help="train and score architectures on held-out data (see pluvium.bench)")
    p.set_defaults(func=None)
    p = sub.add_parser("operational", add_help=False, help="score Pluvium against ECMWF, NOAA and AI forecasts on real runs "
                                           "(see pluvium.benchmarks.operational)")
    p.set_defaults(func=None)
    args, rest = parser.parse_known_args(argv)
    if args.command == "operational":
        from .benchmarks.operational import main as operational_main

        operational_main(rest)
        return
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
