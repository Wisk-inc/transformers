# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""``helix tristream bench run|compare|leaderboard``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="helix tristream bench", description="TriStream benchmarks")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run a suite on systems")
    run.add_argument("--suite", default="quick", choices=["quick", "standard", "full"])
    run.add_argument("--systems", default="identity,harmonic-noise,tristream")
    run.add_argument("--root", default=None, help="a molab root with a trained TriStream (codec + runs/<size>)")
    run.add_argument("--size", default="ts-765m")
    run.add_argument("--out", default="bench-out")
    run.add_argument("--sample-rate", type=int, default=24000)
    run.add_argument("--data", nargs="*", default=[], help="set=folder, e.g. vocalset=/data/VocalSet")
    compare = sub.add_parser("compare", help="what got worse from an old run to a new one")
    compare.add_argument("new")
    compare.add_argument("old")
    compare.add_argument("--system", default="tristream")
    board = sub.add_parser("leaderboard", help="rank the systems of one or more runs")
    board.add_argument("results", nargs="+")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        from .runner import run_suite

        data = dict(kv.split("=", 1) for kv in args.data)
        result = run_suite(args.suite, tuple(s for s in args.systems.split(",") if s), args.out, args.sample_rate,
                           root=args.root, size=args.size, data_dirs=data)
        print((Path(args.out) / "report.md").read_text())
        for s in result["skipped"]:
            print("skipped:", s)
        return 0
    if args.cmd == "compare":
        from .regression import compare as diff
        from .regression import describe

        new, old = (json.loads(Path(p).read_text()) for p in (args.new, args.old))
        found = diff(new, old, args.system)
        print(describe(found))
        return 1 if any(f["significant"] for f in found) else 0
    from .leaderboard import load, merge, table

    print(table(merge(load(args.results))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
