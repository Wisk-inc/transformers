# Copyright 2026 Nathan. Apache-2.0.
"""
Orchestration for one 8-GPU machine: stage the record, launch the training runs, follow them.

Every long job runs as its own process group, detached from the notebook or terminal that started it,
writing to a log file and a PID file -- so closing the notebook, losing the connection or restarting the
kernel never stops a run, and running the same cell again reattaches to it instead of starting a second
copy. The runbook (``b300_naturev1.py``) is a thin layer of settings over these functions.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ------------------------------------------------------------------------------------------- data --

def stage_era5_record(directory: str | Path, first_year: int, last_year: int, stats_path: str | Path,
                      workers: int = 32) -> Path:
    """
    Stage every six-hourly ERA5 state of ``first_year``-``last_year`` (WeatherBench 2, 1.5 degrees, the
    backbone's 89 channels) to one memmap, normalised with the *training* statistics -- so the retrieval
    and stage B see exactly what the backbone was trained on. About 7.7 GB a year. Resumable.
    """
    from .era5 import Normalizer, expand_variables, materialise
    from .upper import open_weatherbench_levels

    stats = json.loads(Path(stats_path).read_text())
    names = list(stats["variables"])
    levels = sorted({int(v.split("@")[1]) for v in names if "@" in v})
    base = tuple(dict.fromkeys(v.split("@")[0] for v in names))
    dataset, _, _ = open_weatherbench_levels()
    expanded = expand_variables(dataset, base, levels)
    if expanded != names:
        raise ValueError("the store's channels do not line up with the training statistics: "
                         f"{len(expanded)} vs {len(names)}")
    years = dataset.time.values.astype("datetime64[Y]").astype(int) + 1970
    indices = np.where((years >= first_year) & (years <= last_year))[0]
    path = Path(directory) / f"era5_{first_year}_{last_year}.npy"
    print(f"[b300] ERA5 {first_year}-{last_year}: {len(indices):,} states, about "
          f"{len(indices) * 121 * 240 * len(names) * 2 / 1e9:.0f} GB", flush=True)
    return materialise(dataset, indices, path, variables=base, normalizer=Normalizer.from_dict(stats), levels=levels,
                       workers=workers, chunk=16)


def memmap_tendency(path: str | Path, channels: int, samples: int = 512, seed: int = 0) -> torch.Tensor:
    """
    Per-channel spread of the six-hour change, in the model's channel layout, from a staged memmap: the
    scale the full-state rollout and the retrieval predict in. Only for checkpoints that do not record it.
    """
    from .era5 import Normalizer
    from .satdata import prepared_to_state

    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    values = np.load(path, mmap_mode="r")
    normalizer = Normalizer.from_dict(meta["statistics"])
    times = np.asarray(meta["time"], dtype=np.int64)
    consecutive = np.where(np.diff(times) == 21600)[0]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(consecutive, min(samples, len(consecutive)), replace=False)
    total = squares = None
    count = 0
    for row in chosen:
        delta = (prepared_to_state(values[row + 1], normalizer, channels)
                 - prepared_to_state(values[row], normalizer, channels)).double()
        total = delta.sum(0) if total is None else total + delta.sum(0)
        squares = (delta ** 2).sum(0) if squares is None else squares + (delta ** 2).sum(0)
        count += delta.shape[0]
    mean = total / count
    return (squares / count - mean ** 2).clamp_min(0).sqrt().clamp_min(1e-3).float()


# ------------------------------------------------------------------------------------------- jobs --

def _pid_file(logs: Path, name: str) -> Path:
    return logs / f"{name}.pid"


def running(logs: str | Path, name: str) -> int | None:
    """The PID of job ``name`` if it is still alive."""
    path = _pid_file(Path(logs), name)
    if not path.exists():
        return None
    try:
        pid = int(path.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


def start_job(name: str, args: list[str], logs: str | Path, gpus: list[int] | None = None,
              env: dict | None = None) -> int:
    """
    Start ``args`` detached (its own session), logging to ``<logs>/<name>.log``. With ``gpus``, it runs
    under ``torchrun`` on exactly those GPUs. Returns the PID; if the job is already running, returns
    that PID and starts nothing.
    """
    logs = Path(logs)
    logs.mkdir(parents=True, exist_ok=True)
    alive = running(logs, name)
    if alive:
        print(f"[b300] {name} is already running (pid {alive}); following it", flush=True)
        return alive
    environment = {**os.environ, **(env or {})}
    command = list(args)
    if gpus is not None:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        import zlib

        port = 29500 + zlib.crc32(name.encode()) % 400                 # stable per job, distinct between jobs
        command = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={len(gpus)}",
                   f"--master_port={port}", *command]
    environment.setdefault("OMP_NUM_THREADS", "4")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    log = open(logs / f"{name}.log", "ab")
    log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} start: {' '.join(command)}\n".encode())
    log.flush()
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                               start_new_session=True)
    _pid_file(logs, name).write_text(str(process.pid))
    print(f"[b300] started {name} (pid {process.pid}) -> {logs / (name + '.log')}", flush=True)
    return process.pid


def stop_job(logs: str | Path, name: str) -> None:
    pid = running(logs, name)
    if pid:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        print(f"[b300] stopped {name}", flush=True)


def follow(logs: str | Path, names: list[str], every: float = 2.0, quiet_minutes: float = 5.0,
           seconds: float | None = None) -> dict:
    """
    Print new log lines of every job as they come, a heartbeat when they are quiet, until all have ended
    (or ``seconds`` pass). Interrupting this stops only the watching. Returns each job's last line.
    """
    logs = Path(logs)
    offsets = {name: (logs / f"{name}.log").stat().st_size if (logs / f"{name}.log").exists() else 0
               for name in names}
    # Show the recent past once, so a re-attached cell has context.
    for name in names:
        path = logs / f"{name}.log"
        if path.exists():
            tail = path.read_bytes()[-4000:].decode(errors="replace").splitlines()[-15:]
            print("\n".join(f"[{name}] {line}" for line in tail), flush=True)
    began, last_output, last = time.time(), time.time(), {}
    try:
        while True:
            for name in names:
                path = logs / f"{name}.log"
                if not path.exists():
                    continue
                with open(path, "rb") as handle:
                    handle.seek(offsets[name])
                    chunk = handle.read()
                    offsets[name] = handle.tell()
                if chunk:
                    text = chunk.decode(errors="replace").rstrip("\n")
                    for line in text.splitlines():
                        if line.strip():
                            print(f"[{name}] {line}", flush=True)
                            last[name] = line
                    last_output = time.time()
            alive = [n for n in names if running(logs, n)]
            if not alive:
                print("[b300] all jobs have ended", flush=True)
                return last
            if time.time() - last_output > quiet_minutes * 60:
                print(f"[b300] {time.strftime('%H:%M')} still running: {', '.join(alive)}", flush=True)
                last_output = time.time()
            if seconds is not None and time.time() - began > seconds:
                return last
            time.sleep(every)
    except KeyboardInterrupt:
        print("[b300] stopped watching; the jobs keep running. Run the cell again to reattach.", flush=True)
        return last


# ------------------------------------------------------------------------------------------ release --

def publish_release(checkpoint: str | Path, stats_path: str | Path, hub_repo: str, directory: str | Path,
                    comparison: str | Path | None = None, report: str | Path | None = None,
                    minimum_better: float = 0.5, force: bool = False) -> str:
    """
    Publish the stage-B model as the release (``model.safetensors``, ``config.json``, ``normalizer.json``
    at the top of the repository, what :func:`naturev1.from_pretrained` reads) -- but only if it earned
    it: when ``comparison`` (from stage B) shows it better than the model it started from on fewer than
    ``minimum_better`` of the (lead, metric) pairs on the test storms, it goes to ``candidates/0.12/``
    instead and the current release is left alone. ``force`` publishes regardless. Returns where it went.
    """
    from dataclasses import asdict
    import hashlib

    from huggingface_hub import HfApi
    from safetensors.torch import save_file

    from .hub import from_pretrained
    from .sattrain import build_model

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = build_model(str(checkpoint), satnet_embed=config.get("satnet_embed", 0),
                        retrieval_layers=config.get("retrieval_layers", 0), ri2=config.get("ri2", False),
                        hemisphere_frame=config.get("hemisphere_frame", False))
    verdict, share = "release", None
    if comparison is not None and Path(comparison).exists():
        share = json.loads(Path(comparison).read_text()).get("better_share_global")
        if share is not None and share < minimum_better and not force:
            verdict = "candidate"
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].numpy().tobytes() if state[name].dtype != torch.bfloat16
                      else state[name].float().numpy().tobytes())
    save_file(state, out / "model.safetensors", metadata={"format": "pt", "tensor_sha256": digest.hexdigest()})
    torch.save(model.state_dict(), out / "model.safetensors.pt")
    (out / "config.json").write_text(json.dumps(asdict(model.config), indent=2, default=list))
    (out / "normalizer.json").write_text(Path(stats_path).read_text())
    (out / "training_record.json").write_text(json.dumps(
        {"trained_steps_per_output": model.trained_heads(), "released_from": str(checkpoint),
         "better_share_on_test_storms": share, "parameters": model.num_parameters()}, indent=2))
    for extra in (report, comparison):
        if extra is not None and Path(extra).exists():
            (out / Path(extra).name).write_text(Path(extra).read_text())
    prefix = "" if verdict == "release" else "candidates/0.12/"
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    api.create_repo(hub_repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=hub_repo, repo_type="model", path_in_repo=prefix or None,
                      commit_message=f"NatureV1 0.12 {'release' if verdict == 'release' else 'candidate'}")
    if verdict == "release":
        loaded, _, _ = from_pretrained(hub_repo, device="cpu")
        same = all(torch.equal(loaded.state_dict()[k].cpu(), v) for k, v in state.items())
        print(f"[b300] released to {hub_repo}; reloaded with from_pretrained: "
              f"{'identical' if same else 'DIFFERENT -- check the upload'}", flush=True)
    else:
        print(f"[b300] better than the starting model on only {100 * (share or 0):.0f}% of test comparisons: "
              f"uploaded to {hub_repo}/candidates/0.12/ and left the current release in place "
              "(pass force=True to override)", flush=True)
    return prefix or "/"


__all__ = ["follow", "publish_release", "memmap_tendency", "running", "stage_era5_record", "start_job", "stop_job"]
