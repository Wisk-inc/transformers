# Copyright 2026 Nathan. Apache-2.0.
"""
Keep a copy of the upstream weights inside a Hugging Face model repository, under ``upstream/<name>/``.

``mirror`` asks each publisher what it has, skips a model whose files are already in the repository at the same
sizes, and otherwise downloads the model (:mod:`pluvium.upstream.fetch`) and uploads it in one commit. Every folder
carries the publisher's licence and a ``SOURCE.md``; ``upstream/README.md`` lists what is there and under which
terms. A model that fails (a publisher that is down, a full disk) is reported and the others still go ahead.

It needs a write token for the repository, passed in or read from ``HF_TOKEN``.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..util import hub_refusal, log
from .fetch import DEFAULT_CACHE, fetch_model, plan
from .registry import Upstream, select

FOLDER = "upstream"


def repo_files(api, repo: str) -> dict:
    """``{path: bytes}`` for every file in a model repository."""
    info = api.model_info(repo, files_metadata=True)
    return {s.rfilename: s.size for s in (info.siblings or [])}


def is_mirrored(model: Upstream, files: list, have: dict) -> bool:
    """Whether every file of ``files`` (a plan) is in the repository at the size the publisher reports."""
    prefix = f"{FOLDER}/{model.name}/"
    return bool(files) and prefix + "SOURCE.md" in have and all(have.get(prefix + f["path"]) == f["size"] for f in files)


def index_text(models: list, results: dict) -> str:
    """``upstream/README.md``: what is kept here and under which terms."""
    now = datetime.now(timezone.utc)
    rows = ["| Folder | What it is | Made by | Licence | Stands behind |", "|---|---|---|---|---|"]
    for m in models:
        if results.get(m.name, "").startswith("failed"):
            continue
        rows.append(f"| [`{m.name}`]({m.name}/SOURCE.md) | {m.title}. {m.note} | {m.producer} | {m.licence} | "
                    f"Pluvium's {'GraphCast' if m.feeds == 'graphcast' else 'AIFS'} input |")
    return "\n".join([
        "# Upstream weights", "",
        "Two of the six forecast systems Pluvium reads are AI models whose weights are public: NOAA's run of "
        "GraphCast (now called AIGFS) and ECMWF's AIFS. The folders here are unchanged copies of those weights, kept "
        "so that this repository holds everything the forecasts depend on.", "",
        "Pluvium does not run these models. It reads the forecasts NOAA and ECMWF publish from them, a few megabytes "
        "for one country. Running them needs a large GPU and that day's global analysis from NOAA or ECMWF, which no "
        "copy of the weights can replace.", "",
        "The licences below are the publishers', not Pluvium's. Each folder has the full text in `LICENSE` and the "
        "notice to keep in `SOURCE.md`. Apache 2.0, which covers the rest of this repository, does not cover them.", "",
        *rows, "",
        f"Last checked against the publishers on {now.day} {now:%B %Y}. To copy them again or add a newer version: "
        "`pluvium upstream mirror --repo <this repository>`.", ""])


def mirror(repo: str, token: str | None = None, names=None, cache=DEFAULT_CACHE, private: bool = False,
           keep: bool = True, say=log, api=None) -> dict:
    """Copy the upstream models into ``repo``. Returns ``{name: "copied" | "already there" | "failed: why"}``.

    ``names`` limits the copy to some models (see ``pluvium upstream list``). ``keep=False`` deletes each local copy
    once it is uploaded. ``api`` is for tests: an object with the methods of ``huggingface_hub.HfApi`` used here.
    """
    if api is None:
        token = token or os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("mirroring needs a Hugging Face write token: set HF_TOKEN or pass one")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    try:
        api.create_repo(repo, repo_type="model", exist_ok=True, private=private)
        have = repo_files(api, repo)
    except Exception as error:
        refusal = hub_refusal(error, repo)
        if refusal is None:
            raise
        raise refusal from None
    models = select(names)
    results = {}
    for model in models:
        try:
            files = plan(model)
            if is_mirrored(model, files, have):
                results[model.name] = "already there"
                say(f"upstream: {model.title}: already in {repo}")
                continue
            folder = fetch_model(model, cache, files, say)
            say(f"upstream: {model.title}: uploading to {repo}/{FOLDER}/{model.name} ...")
            api.upload_folder(folder_path=str(folder), path_in_repo=f"{FOLDER}/{model.name}", repo_id=repo,
                              repo_type="model", ignore_patterns=["*.part"],
                              commit_message=f"Mirror {model.title}, unchanged ({model.licence})")
            results[model.name] = "copied"
            if not keep:
                shutil.rmtree(folder, ignore_errors=True)
        except Exception as error:                              # one publisher down must not stop the rest
            results[model.name] = f"failed: {type(error).__name__}: {str(error)[:200]}"
            say(f"upstream: {model.title}: not mirrored ({results[model.name][8:]})")
    if any(not r.startswith("failed") for r in results.values()):
        api.upload_file(path_or_fileobj=index_text(models, results).encode("utf-8"),
                        path_in_repo=f"{FOLDER}/README.md", repo_id=repo, repo_type="model",
                        commit_message="Upstream weights: what is kept here and under which licences")
    return results


def summary(results: dict) -> str:
    """One line for a log: how many were copied, were there already, or failed."""
    count = lambda word: sum(r.startswith(word) for r in results.values())      # noqa: E731
    parts = [f"{count('copied')} copied", f"{count('already')} already there"]
    if count("failed"):
        parts.append(f"{count('failed')} failed (" + ", ".join(k for k, r in results.items() if r.startswith("failed")) + ")")
    return ", ".join(parts)


def local_size(cache=DEFAULT_CACHE) -> float:
    """Megabytes the local copies take."""
    root = Path(cache).expanduser()
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1e6 if root.exists() else 0.0


__all__ = ["FOLDER", "index_text", "is_mirrored", "local_size", "mirror", "repo_files", "summary"]
