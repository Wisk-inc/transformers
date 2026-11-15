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
    from .quick import setup

    root = Path(args.root).expanduser()
    codes = [c.strip() for c in args.country.split(",") if c.strip()]
    box = tuple(float(x) for x in args.box.split(",")) if args.box else None
    area, geo, towns = setup(codes, root, box)
    s, n, w, e = area.target_box
    print(f"{area.name}: {', '.join(area.countries.values())}")
    print(f"  forecasts for {s:.2f} to {n:.2f} N, {w:.2f} to {e:.2f} E; weather read over {area.domain}")
    print(f"  mesh {len(geo.mesh):,} points, {len(geo.targets):,} truth cells, {len(towns.points):,} places")
    print(f"  live rain now from {area.satellite}; bulletins in "
          + "; ".join(f"{cc}: {', '.join(area.bulletin_languages(cc))}" for cc in area.iso2))
    print(f"next: pluvium now --country {','.join(codes)} --root {root}   (14-day rain, on a CPU)")
    print(f"  or: pluvium program --root {root} --country {','.join(codes)}   (the hourly network, on a GPU)")


def cmd_now(args) -> None:
    """From nothing to the 14-day rain forecast for a country, in one command (:mod:`pluvium.quick`)."""
    from .blendlive import alert_summary
    from .frontier.model import FrontierBlend
    from .quick import brief, forecast

    got = forecast(args.country, args.root, args.repo or None, time=args.time, learn=args.learn,
                   train=not args.no_train, handover=args.handover)
    model, result = got["model"], got["result"]
    source = {"local": "the blend already in this folder", "downloaded": f"the blend published in {args.repo}",
              "trained": "a blend trained just now"}[got["how"]]
    print(f"\n{model.area_json['name']}: 14-day rain forecast from {result['day0']} for {len(got['names']):,} places, "
          f"with {source}")
    print("runs read: " + ", ".join(f"{k} {v}Z" for k, v in result["runs"].items()))
    missing = [s for s in model.systems if s not in result["runs"]]
    if missing:
        print("not available right now, and left out: " + ", ".join(missing))
    if isinstance(model, FrontierBlend):
        print(handover_line(model))
    print()
    print(brief(result, got["names"], got["population"], top=args.top, days=args.days, place=args.place))
    print()
    alerts = alert_summary(result)
    if alerts:
        print(alerts)
    print(f"every place and all 14 days: {got['table']}")
    if args.storms:
        args.root, args.basins, args.offline, args.replay_days = str(got["root"]), args.basins, False, 10
        cmd_cyclone(args)


def cmd_upstream(args) -> None:
    """The published weights of the AI models whose forecasts Pluvium reads: list them, copy them, mirror them."""
    from . import upstream

    if args.action == "list":
        print(f"{'name':26s} {'MB':>6s}  {'licence':22s} {'feeds':10s} what")
        for m in upstream.MODELS:
            print(f"{m.name:26s} {m.size_mb:6d}  {m.licence:22s} {m.feeds:10s} {m.title}" + ("  (in use today)" if m.current else ""))
        print(f"{'':26s} {sum(m.size_mb for m in upstream.MODELS):6d}  in all. Pluvium reads these models' forecasts; "
              "it does not run them.")
        return
    if args.action == "fetch":
        for m in upstream.select(args.names):
            folder = upstream.fetch_model(m, args.cache)
            print(f"{m.name}: {folder}")
        return
    if not args.repo:
        raise SystemExit("mirror needs --repo, e.g. --repo CorxLabs/Pluvium-500")
    results = upstream.mirror(args.repo, names=args.names, cache=args.cache, private=args.private, keep=not args.no_keep)
    for name, outcome in results.items():
        print(f"{name}: {outcome}")
    print(f"upstream weights in https://huggingface.co/{args.repo}/tree/main/upstream: {upstream.summary(results)}")
    if any(r.startswith("failed") for r in results.values()):
        raise SystemExit(1)


def cmd_cyclone(args) -> None:
    """Pluvium Cyclone: where every tropical cyclone alive now is expected to go, from six systems."""
    from .cyclone import live
    from .cyclone.report import bulletin

    now = None
    if args.time:
        from .util import hour_of

        now = hour_of(args.time)
    basins = tuple(b.strip() for b in args.basins.split(",") if b.strip())
    found = live.run(args.root, now=now, basins=basins, download=not args.offline, replay_days=args.replay_days)
    if not found:
        print("no tropical cyclone at the newest run in " + ", ".join(basins))
        return
    for storm, guidance in found:
        print(bulletin(guidance, storm.name.title()))
    print(f"{len(found)} storm(s); bulletins and data in {Path(args.root).expanduser() / 'cyclone'}")


def handover_line(model) -> str:
    """One line on what a frontier blend does with a forecast beyond its training record, and why."""
    verdict = model.meta.get("handover", {})
    mode = verdict.get("mode", "auto")
    found = "were forecast better with it" if verdict.get("keep", True) else "were not forecast better with it"
    test = f"the storms staged in the training period {found}"
    if model.hands_over:
        why = test if mode == "auto" else f"set to always; {test}"
        return f"record storms: a forecast beyond the training record is handed, in part, to the raw systems ({why})"
    why = test if mode == "auto" else f"set to never; {test}"
    return (f"record storms: forecast by the calibration alone ({why}); the table still flags a forecast beyond the "
            "record. --handover always overrules this")


def cmd_blend(args) -> None:
    """Pluvium Blend for any country: train it on a year of archived runs, or forecast 14 days for every town."""
    from .area import Area
    from .blendlive import BlendModel, alert_summary, write_table
    from .domain import Gazetteer, PointSet
    from .frontier.model import FrontierBlend, load_blend

    root = Path(args.root).expanduser()
    if args.action == "publish":
        from .quick import publish_blend

        if not args.repo:
            raise SystemExit("publish needs --repo, e.g. --repo CorxLabs/Pluvium-500")
        publish_blend(args.repo, root, private=args.private)
        return
    if args.action == "train":
        codes = [c.strip() for c in (args.country or "").split(",") if c.strip()]
        area = Area.load(root / "area.json") if (root / "area.json").exists() and not codes else \
            Area.for_countries(codes, cache=root / "cache" / "areas")
        root.mkdir(parents=True, exist_ok=True)
        area.save(root / "area.json")
        start, end = args.start, args.end
        if not (start and end):
            from .quick import recent_year

            start, end = recent_year()
            print(f"training period: {start[:10]} to {end[:10]}, the most recent year NOAA's rain record covers")
        if args.classic:
            model = BlendModel.train(area, root, start=start, end=end)
        else:
            model = FrontierBlend.train(area, root, start=start, end=end, handover=args.handover or "auto")
        print(f"blend for {area.name} trained on {model.meta['runs']} runs at {model.meta['cells']:,} places "
              f"({len(model.meta['components'])} components pooled); saved in {root / 'blend'}")
        if isinstance(model, FrontierBlend):
            print(handover_line(model))
        return
    model = load_blend(root / "blend")
    if args.handover and isinstance(model, FrontierBlend):    # overrule the staged verdict for this run (not saved)
        model.set_handover(args.handover)
    now = None
    if args.time:
        from .util import hour_of

        now = hour_of(args.time)
    if args.action == "learn":
        if not isinstance(model, FrontierBlend):
            raise SystemExit("this blend was trained with --classic and has no online weights to update")
        from .quick import catch_up

        learned = catch_up(model, root, now)
        weights = ", ".join(f"{k} {v:.2f}" for k, v in model.weights().items())
        print(f"learned from {len(learned)} day(s); weights for light rain on days 1-3: {weights}")
        return
    towns_path = root / "towns.npz"
    if towns_path.exists():
        towns = Gazetteer(PointSet.load(towns_path))
        lat, lon, names = towns.points.lat, towns.points.lon, [str(x) for x in towns.names]
    else:                                                     # no town list: the area's grid
        lat_g, lon_g = model.grid
        la, lo = np.meshgrid(lat_g, lon_g, indexing="ij")
        lat, lon = la.ravel(), lo.ravel()
        names = [f"{a:.2f},{b:.2f}" for a, b in zip(lat, lon)]
    result = model.forecast(lat, lon, root / "live_archive", now=now)
    out = write_table(result, names, root / f"blend_{result['day0']}.csv")
    print(f"14-day blended forecast from {result['day0']} for {len(names):,} places -> {out}")
    print("runs used: " + ", ".join(f"{k} {v}Z" for k, v in result["runs"].items()))
    alerts = alert_summary(result)
    if alerts:
        print(alerts)


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
    p.add_argument("action", choices=["train", "forecast", "learn", "publish"],
                   help="train: fit the blend; forecast: 14 days for every town; learn: update the online weights "
                        "from the days whose rain is now known (run it daily, before forecast); publish: upload the "
                        "trained blend to a Hub repository so that others can skip training")
    p.add_argument("--repo", default=None, help="Hub model repository (publish)")
    p.add_argument("--private", action="store_true", help="create the repository as private (publish)")
    p.add_argument("--country", default=None, help="ISO codes (train; default: the area saved in --root)")
    p.add_argument("--root", default="pluvium-area")
    p.add_argument("--start", default=None, help="training runs from, e.g. 2024-04-01T00 (train; default with --end "
                                                 "left out too: the most recent year the rain record covers)")
    p.add_argument("--end", default=None, help="training runs until (train)")
    p.add_argument("--time", default=None, help="forecast as of this UTC hour, e.g. 2026-11-14T12 (default: now)")
    p.add_argument("--classic", action="store_true",
                   help="train the first Blend (no tail weighting, online weights or record-storm hand-over)")
    p.add_argument("--handover", choices=["auto", "always", "never"], default=None,
                   help="what to do with a forecast beyond the training record: follow the verdict of the storms "
                        "staged at training (the default), always hand it to the raw systems, or never. With train "
                        "it is saved in the model; with forecast it applies to that run only")
    p.set_defaults(func=cmd_blend)
    p = sub.add_parser("now", help="the 14-day rain forecast for a country in one command: sets it up, downloads a "
                                   "trained blend or trains one, reads the newest runs")
    p.add_argument("--country", required=True, help="ISO codes, e.g. JM or DO,HT")
    p.add_argument("--root", default=None, help="folder for the country's files (default: ~/pluvium-<codes>)")
    p.add_argument("--repo", default=DEFAULT_REPO, help="Hub repository to look in for a trained blend ('' for none)")
    p.add_argument("--place", default=None, help="show the places whose name contains this, not the largest towns")
    p.add_argument("--top", type=int, default=5, help="how many places to print")
    p.add_argument("--days", type=int, default=5, help="how many days to print (the table always has 14)")
    p.add_argument("--learn", action="store_true",
                   help="first update the online weights from the days whose rain is now known (daily use)")
    p.add_argument("--no-train", action="store_true", help="stop if no trained blend is found, do not train one")
    p.add_argument("--time", default=None, help="forecast as of this UTC hour, e.g. 2026-11-14T12 (default: now)")
    p.add_argument("--handover", choices=["auto", "always", "never"], default=None,
                   help="overrule the staged storms' verdict on record storms for this run")
    p.add_argument("--storms", action="store_true", help="also track every tropical cyclone alive now")
    p.add_argument("--basins", default="al,ep", help="basins for --storms")
    p.set_defaults(func=cmd_now)
    p = sub.add_parser("upstream", help="the published weights of the AI models whose forecasts Pluvium reads: "
                                        "list them, copy them, mirror them into a Hub repository")
    p.add_argument("action", choices=["list", "fetch", "mirror"])
    p.add_argument("names", nargs="*", help="which models (default: all; see `pluvium upstream list`)")
    p.add_argument("--repo", default=None, help="Hub model repository to mirror into (mirror)")
    p.add_argument("--cache", default="~/.cache/pluvium/upstream", help="where the local copies are kept")
    p.add_argument("--private", action="store_true", help="create the repository as private (mirror)")
    p.add_argument("--no-keep", action="store_true", help="delete each local copy once it is uploaded (mirror)")
    p.set_defaults(func=cmd_upstream)
    p = sub.add_parser("cyclone", help="Pluvium Cyclone: track, cone and passage chances for every storm alive now")
    p.add_argument("--root", default="pluvium-cyclone", help="folder for decks, pressure fields, tracks and bulletins")
    p.add_argument("--basins", default="al,ep", help="deck prefixes: al (North Atlantic), ep (East Pacific)")
    p.add_argument("--time", default=None, help="as of this UTC hour, e.g. 2025-10-25T22 (default: now)")
    p.add_argument("--replay-days", type=int, default=10, help="days of earlier runs replayed to learn from")
    p.add_argument("--offline", action="store_true", help="use only decks and fields already in --root")
    p.set_defaults(func=cmd_cyclone)
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
    from .util import PluviumError

    try:
        args.func(args)
    except PluviumError as error:                               # something the person can act on: no traceback
        raise SystemExit(f"pluvium: {error}") from None


if __name__ == "__main__":
    main()
