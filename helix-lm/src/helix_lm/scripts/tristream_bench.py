# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The benchmark on molab: the trained TriStream of a root against the baselines, the change from the last run, the
composite score, and the results on the Hub. The training program runs it at the end of its eval stage; it can also
be run by hand:

    python -m helix_lm.scripts.tristream_bench --root /workspace/tristream --suite standard

Suites ``quick`` (synthetic voices only, minutes) and ``standard`` (adds VocalSet's test singers when the data
stage fetched VocalSet). The research sets are measured only when their folders are given with ``--data``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def run(root: str | Path, size: str = "ts-765m", suite: str = "quick", hf_repo: str | None = None,
        token: str | None = None, data: dict | None = None, log=print) -> dict:
    from ..bench.hub import pull_latest, push_results
    from ..bench.regression import compare, describe
    from ..bench.runner import run_suite
    from ..bench.systems import checkpoint_tag
    from ..music.composite import from_summary

    root = Path(root)
    out = root / "bench"
    data = dict(data or {})
    vocalset = root / "raw" / "vocalset"
    if vocalset.exists():
        data.setdefault("vocalset", str(vocalset))
    previous = None
    if (out / "latest.json").exists():
        previous = json.loads((out / "latest.json").read_text())
    elif hf_repo and token:
        previous = pull_latest(hf_repo, token, root / "bench-previous")
    # outputs of older checkpoints are never scored again: only the current checkpoint's are kept on disk
    keep = "tristream-" + checkpoint_tag(root / "codec" / "codec.pt", root / "runs" / size / "latest.pt")
    for stale in (out / "cache").glob("tristream-*") if (out / "cache").exists() else []:
        if stale.name != keep:
            shutil.rmtree(stale, ignore_errors=True)
    result = run_suite(suite, ("tristream", "harmonic-noise", "identity"), out, root=root, size=size, data_dirs=data,
                       log=log)
    if "tristream" not in result["summary"]:
        log(f"[bench] TriStream not measured: no trained model under {root} yet (codec/codec.pt and "
            f"runs/{size}/latest.pt); the baselines were")
    else:
        log(f"[bench] TriStream composite {from_summary(result['summary'], 'tristream'):.3f} (harmonic-noise "
            f"{from_summary(result['summary'], 'harmonic-noise'):.3f}; 1 is the target on every metric)")
    if previous and previous.get("suite") == result["suite"]:
        log("[bench] against the last run: " + describe(compare(result, previous)))
    (out / "latest.json").write_text(json.dumps(result, default=float))
    if hf_repo and token:
        push_results(out, hf_repo, token, log=log)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TriStream benchmark on a molab root")
    parser.add_argument("--root", default="/workspace/tristream")
    parser.add_argument("--size", default="ts-765m")
    parser.add_argument("--suite", default="quick", choices=["quick", "standard", "full"])
    parser.add_argument("--hf-repo", default="CorxLabs/Tristream-V2-SVS")
    parser.add_argument("--data", nargs="*", default=[])
    args = parser.parse_args(argv)
    token = os.environ.get("HF_TOKEN") or (Path("/root/hf_token").read_text().strip()
                                           if Path("/root/hf_token").exists() else None)
    run(args.root, args.size, args.suite, args.hf_repo or None, token, dict(kv.split("=", 1) for kv in args.data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
