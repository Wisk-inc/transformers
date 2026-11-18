# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Benchmark results on the Hub, beside the checkpoints they measure: ``bench/<time>/results.json`` and
``report.md`` for each run, ``bench/latest.json`` and ``bench/LATEST.md`` for the newest, in one commit. Never
raises: a failed upload is said and the training goes on."""

from __future__ import annotations

import json
import time
from pathlib import Path


def push_results(out_dir: str | Path, repo: str, token: str | None, api=None, prefix: str = "bench",
                 log=print) -> bool:
    out_dir = Path(out_dir)
    results, report = out_dir / "results.json", out_dir / "report.md"
    if not results.exists() or not repo or not (token or api):
        return False
    try:
        from huggingface_hub import CommitOperationAdd, HfApi

        api = api or HfApi(token=token)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        ops = [CommitOperationAdd(f"{prefix}/{stamp}/results.json", str(results)),
               CommitOperationAdd(f"{prefix}/latest.json", str(results))]
        if report.exists():
            ops += [CommitOperationAdd(f"{prefix}/{stamp}/report.md", str(report)),
                    CommitOperationAdd(f"{prefix}/LATEST.md", str(report))]
        api.create_commit(repo_id=repo, repo_type="model", operations=ops,
                          commit_message=f"benchmark {json.loads(results.read_text()).get('suite', '')} {stamp}")
        log(f"[bench] results on https://huggingface.co/{repo}/tree/main/{prefix}")
        return True
    except Exception as error:
        log(f"[bench] results not uploaded ({type(error).__name__}: {str(error)[:160]})")
        return False


def pull_latest(repo: str, token: str | None, dest: str | Path, prefix: str = "bench") -> dict | None:
    """The newest results on the Hub, or None."""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo, f"{prefix}/latest.json", token=token, local_dir=str(dest))
        return json.loads(Path(path).read_text())
    except Exception:
        return None


__all__ = ["pull_latest", "push_results"]
