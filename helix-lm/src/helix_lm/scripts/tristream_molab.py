# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
TriStream-2 on one RTX PRO 6000 Blackwell (96 GB): the whole program as one script for a notebook.

In a notebook cell:

    !pip install -q "helix-lm[music]"
    !python -m helix_lm.scripts.tristream_molab --root /workspace/tristream --stage all \
        --hf-repo CorxLabs/TriStream-2-765M

or stage by stage (``setup``, ``data``, ``prepare``, ``codec``, ``train``, ``eval``, ``push``, ``swap``). Every
stage is resumable and writes a ``done`` marker; the token is read from ``$HF_TOKEN`` or ``/root/hf_token``
(never pasted into a notebook that gets shared). Stop with Ctrl-C or the stop file ``<root>/stop``; start again
to continue.

What the stages do and how long they take on this GPU (estimates; the log prints measured rates):

| stage   | what                                                                                   | time        |
| ---     | ---                                                                                    | ---         |
| setup   | packages, GPU check, folders                                                           | minutes     |
| data    | the commercially clean sources (VocalSet, Dagstuhl ChoirSet, ESMUC, Emilia-YODAS       | hours       |
|         | subset, Slakh instrumentals, FMA CC-BY filter), plus your uploads; research-only sets  | (network)   |
|         | only with ``--include-research`` and only for evaluation                              |             |
| prepare | separation (HT-Demucs, MIT) for mixtures, F0, beats/key/sections, breath, dryness,     | ~1 day      |
|         | tags, singer-disjoint split, codec frames                                              | per 5k h    |
| codec   | the 25 Hz split-RVQ codec on the audio pool (or ``--codec-hours 0`` to skip and use a  | ~12 days    |
|         | pretrained open codec where its licence allows)                                       |             |
| train   | TriStream-2 765M with Muon, bf16, speech first then singing, singer-disjoint validation| ~7-10 days  |
|         | per 15B frames; checkpoints every 30 min, pushed to the Hub every 6 h                  |             |
| eval    | F0 RMSE/FFE, speaker similarity, breath/technique agreement, glitch rate, bleed, the   | hours       |
|         | critic's score of plans; SingMOS-Pro when installed                                   |             |
| swap    | the demo: a song + a reference voice -> the song in that voice, with the report        | minutes     |

The honest part: nothing here has been trained yet; the compute numbers come from the GPU's datasheet (about
500 TFLOPS dense bf16 inferred) and published codec/LM throughputs (docs/TRISTREAM.md). The clean-licence
singing data in the open is under a hundred hours; the script says what it found and what it did not.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import zipfile
from pathlib import Path

DEFAULT_REPO = "CorxLabs/Tristream-V2-SVS"
LAST_PUSH = 0.0

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

STAGES = ("setup", "data", "prepare", "codec", "refiner", "air", "train", "eval", "push", "swap")
PREPARE_FOLDERS = ("uploads", "raw/vocalset", "raw/dagstuhl-choirset", "raw/esmuc", "raw/emilia-yodas")


def log(*parts) -> None:
    print(time.strftime("[%H:%M:%S]"), *parts, flush=True)


def token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if not tok and Path("/root/hf_token").exists():
        tok = Path("/root/hf_token").read_text().strip()
    return tok or None


def done(root: Path, stage: str) -> bool:
    return (root / "markers" / stage).exists()


def has_audio(folder: Path) -> bool:
    return folder.exists() and any(p.suffix.lower() in AUDIO_SUFFIXES for p in folder.rglob("*"))


def source_on_disk(folder: Path) -> bool:
    """A source counts as present when an archive of it was extracted whole, or it ships plain audio files."""
    return any((folder / "extracted").rglob(".extracted")) or any(
        p.suffix.lower() in AUDIO_SUFFIXES for p in folder.glob("*")
    )


def wanted_sources(root: Path, args) -> list[str]:
    wanted = [n for n in CLEAN_SOURCES if n not in args.skip and n != "slakh2100"]  # Slakh is opt-in (104 GB)
    if getattr(args, "emilia_hours", 0) > 0 and token():
        wanted.append("emilia-yodas")
    return wanted


def missing_sources(root: Path, args) -> list[str]:
    found_file = root / "data_found.json"
    found = json.loads(found_file.read_text()) if found_file.exists() else {}
    return [n for n in wanted_sources(root, args) if found.get(n, 0) <= 0 or not source_on_disk(root / "raw" / n)]


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def take_lock(root: Path) -> bool:
    """One run per root. A notebook cell stopped and started again leaves the first process alive: two runs on
    one folder race on the same downloads and checkpoints, so the second refuses to start while the first lives."""
    lock = root / "run.lock"
    if lock.exists():
        try:
            holder = json.loads(lock.read_text())
            pid = int(holder.get("pid", 0))
        except Exception:
            pid = 0
        if pid and pid != os.getpid() and pid_alive(pid):
            log(f"!! another run (pid {pid}, started {holder.get('started', '?')}) holds {root}")
            log("   stop it first (restart the notebook kernel, or kill that pid), or delete run.lock")
            return False
    lock.write_text(json.dumps({"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}))

    def release() -> None:
        try:
            if lock.exists() and json.loads(lock.read_text()).get("pid") == os.getpid():
                lock.unlink()
        except Exception:
            pass

    atexit.register(release)
    return True


def complete(root: Path, stage: str, args) -> bool:
    """A stage counts as done when its marker AND its artifact exist: a marker pulled from the Hub without the
    file behind it, or a prepared folder that was never copied to this machine, is not done."""
    if not done(root, stage):
        return False
    if stage == "prepare":
        if not any((root / "prepared").rglob("manifest.json")):
            return False
        # a source that arrived after the last prepare (a download retried on a later run) is prepared now
        for folder in PREPARE_FOLDERS:
            manifest = root / "prepared" / folder.replace("/", "_") / "manifest.json"
            if has_audio(root / folder) and not manifest.exists():
                return False
        return True
    if stage == "data":
        # a source that failed (a cut-short download, a corrupt archive) is fetched again on the next run
        return not missing_sources(root, args)
    if stage in ("codec", "refiner", "air", "train") and getattr(args, f"{stage}_hours", 0) <= 0:
        return True
    artifact = {
        "codec": root / "codec" / "codec.pt",
        "refiner": root / "refiner" / "refiner.pt",
        "air": root / "air" / "air.pt",
        "train": root / "runs" / args.size / "latest.pt",
        "eval": root / "eval" / "summary.json",
    }.get(stage)
    if artifact is None:
        return True
    if not artifact.exists():
        return False
    budget = getattr(args, f"{stage}_hours", None)
    if budget is not None and checkpoint_meta(artifact).get("hours", budget) + 0.02 < budget:
        return False  # more hours were asked for than this stage has trained: it continues where it stopped
    if stage == "eval":
        # the scores belong to a model: after more training they are measured again
        trained = root / "runs" / args.size / "latest.pt"
        return not trained.exists() or artifact.stat().st_mtime >= trained.stat().st_mtime
    return True


class Budget:
    """Hours of a stage that carry across restarts: a stage stopped at hour 30 of 48 resumes with 18 left,
    not 48 again."""

    def __init__(self, hours: float, elapsed: float = 0.0) -> None:
        self.hours, self.before, self.start = hours, elapsed, time.time()

    @property
    def elapsed(self) -> float:
        return self.before + (time.time() - self.start) / 3600

    @property
    def remaining(self) -> float:
        return max(0.0, self.hours - self.elapsed)

    def over(self) -> bool:
        return self.remaining <= 0


def save_checkpoint(obj: dict, path: Path) -> None:
    """Atomic save (a crash mid-write never leaves a torn file) plus a small sidecar with the step and the hours,
    so a pull can compare checkpoints without loading them."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    side = {
        "step": int(obj.get("step", 0)),
        "hours": float(obj.get("hours", 0.0)),
        "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.with_suffix(".json").write_text(json.dumps(side))


def checkpoint_meta(path: Path) -> dict:
    side = path.with_suffix(".json")
    if side.exists():
        try:
            return json.loads(side.read_text())
        except Exception:
            pass
    return {"step": 0 if path.exists() else -1, "hours": 0.0}


def warn_if_codec_moved(root: Path, ckpt: Path, what: str) -> None:
    """The refiner and the singing model learn the codec's frames: when the codec trained again after them, they
    must train again too, or they speak an older dialect of it."""
    codec = root / "codec" / "codec.pt"
    if ckpt.exists() and codec.exists() and codec.stat().st_mtime > ckpt.stat().st_mtime + 60:
        log(
            f"!! the codec trained after the {what}: give the {what} more hours too so it adapts to the new codec "
            "(raising GPU_HOURS in the cell does this for every stage)"
        )


def loader_workers(requested: int = 0) -> int:
    """CPU threads that build batches: every core but two (one feeds the GPU, one is for the system)."""
    if requested > 0:
        return requested
    return max(2, min(32, (os.cpu_count() or 4) - 2))


def _pin(data):
    import torch

    if isinstance(data, torch.Tensor):
        return data.pin_memory()
    if isinstance(data, tuple):
        return tuple(_pin(x) for x in data)
    if isinstance(data, list):
        return [_pin(x) for x in data]
    return data


class _Chunk:
    """A chunk as it crosses from a worker process: torch's loader turns tuples into lists (and would scramble
    the fields of a batch); an object of its own passes through untouched, and pins itself."""

    __slots__ = ("data",)

    def __init__(self, data) -> None:
        self.data = data

    def pin_memory(self, *_):
        return _Chunk(_pin(self.data))


def _identity(item):
    return item


def _combine(items: list, device=None):
    """Join chunks into one batch: tensors are concatenated (on the GPU when a device is given, so pinned chunks
    copy asynchronously), lists are joined, counts are added; a tuple is joined field by field."""
    import torch

    first = items[0]
    if isinstance(first, torch.Tensor):
        if device is not None and device.type == "cuda":
            return torch.cat([x.to(device, non_blocking=True) for x in items])
        return torch.cat(items)
    if isinstance(first, tuple):
        return tuple(_combine([item[k] for item in items], device) for k in range(len(first)))
    if isinstance(first, list):
        return [x for item in items for x in item]
    if isinstance(first, int):
        return sum(items)
    return first


def _take(item, n: int):
    """The first ``n`` clips of a batch (the last chunk may overshoot the batch size)."""
    import torch

    if isinstance(item, (torch.Tensor, list)):
        return item[:n]
    if isinstance(item, tuple):
        return tuple(_take(x, n) for x in item)
    if isinstance(item, int):
        return min(item, n)
    return item


def shared_memory_gb() -> float:
    try:
        return shutil.disk_usage("/dev/shm").free / 1e9
    except OSError:
        return 0.0


class Prefetcher:
    """
    Builds the next batches on the CPU while the GPU trains on this one. File reads, augmentation and pitch
    tracks are CPU work, and one thread cannot keep a Blackwell card busy, so:

    * **worker processes** (one per core but two) each build small chunks of clips, ``make(gen, n)`` with
      their own random generator and one CPU thread each; the main process joins the chunks into a batch of
      ``size()`` clips on the GPU. Chunks keep memory in flight small whatever the batch size, and a resize
      takes effect on the very next batch;
    * worker processes pass chunks through shared memory (``/dev/shm``); when it is small (a container
      started without ``--shm-size``) or a worker dies, the feeder falls back to threads by itself;
    * on a CPU-only machine one thread builds the batches, as before;
    * ``wait_share()`` says how much of the time the GPU waited for data: near zero means the GPU is the limit.
    """

    def __init__(self, make, size, chunk: int = 16, workers: int = 1, seed: int = 0, device=None,
                 mode: str = "auto", log=log) -> None:  # fmt: skip
        self.make, self.size, self.chunk, self.seed, self.device, self.log = make, size, chunk, seed, device, log
        self.workers = max(1, workers)
        self.waited, self.started = 0.0, time.time()
        self.it, self.threads, self.stop, self.queue = None, [], threading.Event(), None
        self.saved_threads = None
        on_gpu = device is not None and device.type == "cuda"
        if mode == "auto":
            if not on_gpu or self.workers == 1:
                mode = "thread"
            elif shared_memory_gb() < 4.0:
                self.log(
                    f"  shared memory /dev/shm has {shared_memory_gb():.1f} GB free: batch workers run as threads "
                    "(start the container with --shm-size=32g for full speed)"
                )
                mode = "thread"
            else:
                mode = "process"
        self.mode = mode
        if mode == "process":
            self._start_processes()
        else:
            self._start_threads(self.workers if on_gpu or mode == "thread-forced" else 1)

    # -- processes
    def _start_processes(self) -> None:
        import torch

        make, chunk, seed = self.make, self.chunk, self.seed

        class Chunks(torch.utils.data.IterableDataset):
            def __iter__(self):
                info = torch.utils.data.get_worker_info()
                gen = torch.Generator().manual_seed(seed * 1000 + (info.id if info else 0))
                while True:
                    yield _Chunk(make(gen, chunk))

        loader = torch.utils.data.DataLoader(
            Chunks(),
            batch_size=None,
            collate_fn=_identity,
            num_workers=self.workers,
            multiprocessing_context="fork",
            pin_memory=self.device is not None and self.device.type == "cuda",
            prefetch_factor=4,
            persistent_workers=False,
        )
        self.it = iter(loader)

    # -- threads
    def _start_threads(self, n: int) -> None:
        import torch

        if n > 1:
            # torch's own thread pool fights the workers for the cores: one compute thread per worker instead
            self.saved_threads = torch.get_num_threads()
            torch.set_num_threads(1)
        self.mode = "thread"
        self.stop = threading.Event()
        self.queue = queue.Queue(maxsize=max(4, 4 * n))
        self.threads = []
        for i in range(n):
            gen = torch.Generator().manual_seed(self.seed * 1000 + i)
            thread = threading.Thread(target=self._run, args=(gen,), daemon=True)
            thread.start()
            self.threads.append(thread)

    def _run(self, gen) -> None:
        while not self.stop.is_set():
            try:
                item = self.make(gen, self.chunk)
            except Exception as error:  # surfaces in get()
                item = error
            while not self.stop.is_set():
                try:
                    self.queue.put(item, timeout=0.5)
                    break
                except queue.Full:
                    continue

    def _next_chunk(self):
        if self.mode == "process":
            try:
                return next(self.it).data
            except RuntimeError as error:
                if "DataLoader worker" not in str(error):
                    raise
                self.log(f"  a batch worker process died ({str(error)[:100]}); switching to threads")
                self._stop_processes()
                self._start_threads(self.workers)
        item = self.queue.get()
        if isinstance(item, Exception):
            raise item
        return item

    def get(self):
        begin = time.time()
        n = max(1, int(self.size()))
        items, have = [], 0
        while have < n:
            item = self._next_chunk()
            items.append(item)
            have += self._count(item)
        batch = _take(_combine(items, self.device), n)
        self.waited += time.time() - begin
        return batch

    @property
    def kind(self) -> str:
        return "worker processes" if self.mode == "process" else ("worker threads" if self.threads[1:] else "thread")

    @staticmethod
    def _count(item) -> int:
        import torch

        first = item[0] if isinstance(item, tuple) else item
        return int(first.shape[0]) if isinstance(first, torch.Tensor) else len(first)

    def wait_share(self, reset: bool = True) -> float:
        share = self.waited / max(1e-6, time.time() - self.started)
        if reset:
            self.waited, self.started = 0.0, time.time()
        return share

    def drain(self) -> None:
        if self.queue is not None:
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break

    def _stop_processes(self) -> None:
        it, self.it = self.it, None
        if it is not None and hasattr(it, "_shutdown_workers"):
            it._shutdown_workers()

    def close(self) -> None:
        """Stop every worker and wait for it: a thread killed mid-computation at exit aborts the program."""
        import torch

        self._stop_processes()
        self.stop.set()
        self.drain()
        for thread in self.threads:
            thread.join(timeout=60)
        self.threads = []
        if self.saved_threads is not None:
            torch.set_num_threads(self.saved_threads)
            self.saved_threads = None


class BatchSizer:
    """Find the batch that trains fastest on this card, instead of a fixed one. Starting from the stage's batch,
    it measures clips per second, doubles the batch while that keeps rising by more than 8 %, and keeps the
    fastest. Past that point a bigger batch only fills memory: the card is already busy and fewer, larger steps
    learn less per clip. On out-of-memory it goes back to the last batch that fit. ``--batch N`` fixes it."""

    def __init__(self, start: int, name: str, device, fixed: bool = False, cap: int = 2048, log=log) -> None:
        self.size, self.name, self.cap, self.log = start, name, cap, log
        self.done = fixed or device.type != "cuda"
        self.best: tuple[int, float] | None = None
        self._reset()

    def _reset(self) -> None:
        self.warm, self.steps, self.clips, self.start = 0, 0, 0, None

    def tick(self, clips: int) -> None:
        """After every successful step, with the clips that step trained on."""
        if self.done or clips != self.size:
            return  # a batch built before the last resize says nothing about the size being measured
        if self.start is None:  # the first steps at a new size are warm-up (cuDNN autotuning, allocator)
            self.warm += 1
            if self.warm >= 5:
                self.start = time.time()
            return
        self.steps += 1
        self.clips += clips
        if self.steps < 25:
            return
        rate = self.clips / max(1e-6, time.time() - self.start)
        if self.best is None or rate > 1.08 * self.best[1]:
            self.best = (self.size, rate)
            if self.size * 2 > self.cap:
                self._finish(f"batch {self.size} ({rate:.1f} clips/s), the cap")
            else:
                self.size *= 2
                self.log(f"{self.name}: {rate:.1f} clips/s at batch {self.size // 2}; trying {self.size}")
        else:
            self.size = self.best[0]
            self._finish(f"batch {self.size} is the fastest ({self.best[1]:.1f} clips/s); larger was not faster")
        self._reset()

    def _finish(self, why: str) -> None:
        import torch

        self.done = True
        free, total = torch.cuda.mem_get_info()
        self.log(f"{self.name}: {why}; GPU memory in use {(total - free) / 1e9:.0f} of {total / 1e9:.0f} GB")

    def out_of_memory(self) -> int:
        if self.best is not None and self.size > self.best[0]:
            self.size = self.best[0]  # the doubling did not fit: back to the last size that did
            self._finish(f"batch {self.size} (the next size did not fit)")
        else:
            self.size = max(1, int(self.size * 0.75))
            self.done = True
        self._reset()
        return self.size


def gpu_max() -> None:
    """The settings that let a Blackwell card run flat out: TF32 matmuls, cuDNN autotuning, high precision mode."""
    import torch

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def extract_archives(folder: Path, delete: bool = False) -> int:
    """Unpack every zip / tar under ``folder`` into ``folder/extracted/<name>`` once (a marker per archive)."""
    count = 0
    for archive in sorted(folder.rglob("*")):
        if archive.is_dir() or "extracted" in archive.parts:
            continue
        lower = archive.name.lower()
        if not lower.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
            continue
        target = folder / "extracted" / re.sub(ARCHIVE_SUFFIX, "", archive.name)
        marker = target / ".extracted"
        if marker.exists():
            if delete:
                archive.unlink(missing_ok=True)  # already unpacked: the archive is only disk space
            continue
        size = archive.stat().st_size
        valid = zipfile.is_zipfile(archive) if lower.endswith(".zip") else tarfile.is_tarfile(archive)
        if not valid:
            log(f"{archive.name} is not a whole archive ({size / 1e6:.1f} MB); deleted, to be downloaded again")
            archive.unlink()
            continue
        target.mkdir(parents=True, exist_ok=True)
        log(f"extracting {archive.name} ({size / 1e9:.2f} GB)")
        try:
            if lower.endswith(".zip"):
                with zipfile.ZipFile(archive) as handle:
                    bad = handle.testzip()
                    if bad is not None:
                        raise zipfile.BadZipFile(f"corrupt member {bad}")
                    handle.extractall(target)
            else:
                with tarfile.open(archive) as handle:
                    handle.extractall(target, filter="data")
            marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            count += 1
            if delete:
                archive.unlink()
        except Exception as error:
            log(f"  could not extract {archive.name}: {type(error).__name__}: {str(error)[:80]}; deleted")
            shutil.rmtree(target, ignore_errors=True)
            archive.unlink(missing_ok=True)
    return count


# -------------------------------------------------------------------------------------------- hub --
def prepared_records(root: Path) -> list[dict]:
    records = []
    for manifest in (root / "prepared").rglob("manifest.json"):
        data = json.loads(manifest.read_text())
        records += [{**clip, "dir": str(manifest.parent)} for clip in data["clips"] if clip.get("accepted")]
    return records


def hub_api():
    tok = token()
    if not tok:
        return None
    from huggingface_hub import HfApi

    return HfApi(token=tok)


def checkpoint_files(root: Path, size: str) -> list[tuple[Path, str]]:
    return [
        (root / "codec" / "codec.pt", "checkpoints/codec/codec.pt"),
        (root / "refiner" / "refiner.pt", "checkpoints/refiner/refiner.pt"),
        (root / "air" / "air.pt", "checkpoints/air/air.pt"),
        (root / "runs" / size / "latest.pt", f"checkpoints/runs/{size}/latest.pt"),
    ]


def hub_pull(args) -> None:
    """Resume from the Hub: every checkpoint the repo holds that is newer (more steps) than the local one is
    downloaded, with its marker and the eval summary, so a fresh machine continues where the last one stopped."""
    api = hub_api()
    if api is None or not args.hf_repo or args.no_pull:
        return
    from huggingface_hub import hf_hub_download

    root = Path(args.root)
    try:
        files = set(api.list_repo_files(args.hf_repo))
    except Exception as error:
        log(f"hub: nothing to pull ({type(error).__name__}: {str(error)[:80]})")
        return
    cache = root / "hf" / "pull"
    pulled = 0
    for local, remote in checkpoint_files(root, args.size):
        if remote not in files:
            continue
        remote_meta = {"step": 0, "hours": 0.0}
        side = remote[:-3] + ".json"
        if side in files:
            remote_meta = json.loads(
                Path(hf_hub_download(args.hf_repo, side, token=token(), local_dir=str(cache))).read_text()
            )
        local_meta = checkpoint_meta(local)
        if local.exists() and local_meta.get("step", 0) >= remote_meta.get("step", 0):
            continue
        log(f"hub: pulling {remote} (step {remote_meta.get('step')}, {remote_meta.get('hours', 0):.1f} h)")
        got = Path(hf_hub_download(args.hf_repo, remote, token=token(), local_dir=str(cache)))
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(got), str(local))
        local.with_suffix(".json").write_text(json.dumps(remote_meta))
        pulled += 1
    for name in ("codec", "refiner", "air", "train", "eval"):
        if f"markers/{name}" in files and not (root / "markers" / name).exists():
            got = Path(hf_hub_download(args.hf_repo, f"markers/{name}", token=token(), local_dir=str(cache)))
            (root / "markers").mkdir(parents=True, exist_ok=True)
            shutil.move(str(got), str(root / "markers" / name))
    if "eval/summary.json" in files and not (root / "eval" / "summary.json").exists():
        got = Path(hf_hub_download(args.hf_repo, "eval/summary.json", token=token(), local_dir=str(cache)))
        (root / "eval").mkdir(parents=True, exist_ok=True)
        shutil.move(str(got), str(root / "eval" / "summary.json"))
    log(f"hub: {pulled} checkpoint(s) pulled from https://huggingface.co/{args.hf_repo}")


def model_card(root: Path, args) -> str:
    rows = []
    for stage in STAGES:
        marker = root / "markers" / stage
        rows.append(f"| {stage} | {'done ' + marker.read_text().strip() if marker.exists() else 'pending'} |")
    checkpoints = []
    for local, remote in checkpoint_files(root, args.size):
        if local.exists():
            meta = checkpoint_meta(local)
            size_gb = local.stat().st_size / 1e9
            checkpoints.append(
                f"| `{remote}` | {meta.get('step', 0):,} | {meta.get('hours', 0.0):.1f} h | {size_gb:.2f} GB |"
            )
    summary = root / "prepared" / "summary.json"
    data = json.loads(summary.read_text()) if summary.exists() else {}
    evals = root / "eval" / "summary.json"
    results = json.loads(evals.read_text()) if evals.exists() else {}
    import helix_lm

    return "\n".join(
        [
            "---",
            "license: apache-2.0",
            "library_name: helix-lm",
            "pipeline_tag: audio-to-audio",
            "tags:",
            "- singing-voice-synthesis",
            "- svs",
            "- voice-conversion",
            "- zero-shot-voice-cloning",
            "- tristream",
            "- anansi",
            "- helix-lm",
            "---",
            "",
            "# TriStream-V2-SVS",
            "",
            f"TriStream-2 `{args.size}` singing voice synthesis and conversion, trained by the helix-lm",
            f"{helix_lm.__version__} molab program (`python -m helix_lm.scripts.tristream_molab`).",
            "This repository is the live state: pushed on a schedule while training runs and pulled back on",
            "restart, so a stopped run",
            "continues from here. Nothing here is a finished model until the `eval` row says done.",
            "",
            "## Status",
            "",
            "| stage | state |",
            "|---|---|",
            *rows,
            "",
            "## Checkpoints",
            "",
            "| file | steps | hours trained | size |",
            "|---|---|---|---|",
            *(checkpoints or ["| none yet | | | |"]),
            "",
            "## Data",
            "",
            f"Prepared: {data.get('hours', 0):.1f} h accepted over {data.get('accepted', 0)} clips",
            f"from {data.get('sources', [])}. Singing and speech are sampled by domain share (speech share annealed",
            f"{args.speech_share_start:.0%} -> {args.speech_share_end:.0%}), never by corpus size.",
            "Sources are the CC BY 4.0 sets in `helix music sources --commercial` (VocalSet, Dagstuhl ChoirSet,",
            "Emilia-YODAS speech, Slakh2100) plus the uploads folder; non-commercial research sets are kept out",
            "unless `--include-research` was given, and then only for evaluation.",
            "",
            "## Evaluation",
            "",
            "```json",
            json.dumps(results, indent=1) if results else "{}",
            "```",
            "",
            "## Use",
            "",
            "```bash",
            "pip install 'helix-lm[music]'",
            f"helix music swap song.wav reference.wav --out out.wav --root <folder with codec/ and runs/{args.size}/>",
            "```",
            "",
            "Clone only voices you own or have written permission for. Outputs pass through the fail-safe loop",
            "(docs/TRISTREAM.md section 12) and carry a report naming what was done.",
        ]
    )


def hub_push(args, force: bool = False, reason: str = "checkpoint") -> bool:
    """Push the checkpoints, markers, manifests, eval and the model card; throttled to --push-every-hours unless
    forced; never raises (a failed push must not stop a run); squashes the repo history so the Hub holds one
    copy of each file instead of every checkpoint ever pushed."""
    global LAST_PUSH
    if not args.hf_repo or not token():
        return False
    if not force and time.time() - LAST_PUSH < args.push_every_hours * 3600:
        return False
    try:
        from huggingface_hub import CommitOperationAdd

        api = hub_api()
        root = Path(args.root)
        try:
            api.create_repo(args.hf_repo, exist_ok=True, private=not args.public)
        except Exception as error:  # a token that may write but not create: the repo exists, carry on
            log(f"hub: create_repo skipped ({type(error).__name__})")
        ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=model_card(root, args).encode())]
        small = list((root / "markers").glob("*")) + list((root / "eval").glob("*.json"))
        small += list((root / "prepared").rglob("manifest.json")) + list((root / "prepared").glob("summary.json"))
        small += [p for p in (root / "data_found.json", root / "codec" / "audit.json") if p.exists()]
        for path in small:
            ops.append(CommitOperationAdd(path_in_repo=str(path.relative_to(root)), path_or_fileobj=str(path)))
        for local, remote in checkpoint_files(root, args.size):
            if local.exists():
                ops.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)))
                side = local.with_suffix(".json")
                if side.exists():
                    ops.append(CommitOperationAdd(path_in_repo=remote[:-3] + ".json", path_or_fileobj=str(side)))
        for attempt in range(3):
            try:
                api.create_commit(
                    args.hf_repo, operations=ops, commit_message=f"{reason} {time.strftime('%Y-%m-%d %H:%M')}"
                )
                break
            except Exception as error:
                log(f"hub: push attempt {attempt + 1} failed ({type(error).__name__}: {str(error)[:80]})")
                time.sleep(30 * (attempt + 1))
        else:
            return False
        if not args.no_squash:
            try:
                api.super_squash_history(args.hf_repo)
            except Exception as error:
                log(f"hub: history not squashed ({type(error).__name__})")
        LAST_PUSH = time.time()
        log(f"hub: pushed ({reason}) to https://huggingface.co/{args.hf_repo}")
        return True
    except Exception as error:
        log(f"hub: push failed ({type(error).__name__}: {str(error)[:100]}); the run continues")
        return False


def mark(root: Path, stage: str) -> None:
    (root / "markers").mkdir(parents=True, exist_ok=True)
    (root / "markers" / stage).write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


# ------------------------------------------------------------------------------------------ setup --
def stage_setup(args) -> None:
    import torch

    root = Path(args.root)
    for sub in ("raw", "uploads", "prepared", "codec", "runs", "eval", "markers", "hf"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root / "hf"))
    if not torch.cuda.is_available():
        log("!! no GPU visible; the data stages run, training will not")
    else:
        props = torch.cuda.get_device_properties(0)
        log(f"GPU: {props.name}, {props.total_memory / 1e9:.0f} GB")
    for package in ("demucs", "soundfile", "huggingface_hub"):
        try:
            __import__(package)
        except ImportError:
            log(f"installing {package}")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=False)
    log(
        "token:",
        "present"
        if token()
        else "NOT SET (set HF_TOKEN or write /root/hf_token; downloads of gated sets and the push need it)",
    )
    mark(root, "setup")


# ------------------------------------------------------------------------------------------- data --
CLEAN_SOURCES = {
    "vocalset": {"zenodo": "1193957", "note": "20 pro singers, techniques, CC BY 4.0"},
    "dagstuhl-choirset": {"zenodo": "4607766", "note": "choir close mics with F0, CC BY 4.0"},
    "esmuc": {"zenodo": "5848990", "note": "12 singers, F0 + notes, CC BY 4.0"},
    "slakh2100": {"zenodo": "4599666", "note": "synthesized instrumentals for augmentation, CC BY 4.0"},
}
RESEARCH_SOURCES = {
    "gtsinger": "AaronZ345/GTSinger",  # CC BY-NC-SA: evaluation and research only
}


def zenodo_files(record: str) -> list[dict]:
    import urllib.request

    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=60) as response:
        payload = json.load(response)
    return [
        {
            "name": f["key"],
            "url": f["links"]["self"],
            "size": int(f.get("size", 0) or 0),
            "checksum": f.get("checksum", ""),
            "record": record,
        }
        for f in payload.get("files", [])
    ]


def alternate_url(info: dict) -> str | None:
    """Zenodo's second download route for the same file (the web one); used when the API route is cut short."""
    if info.get("record") and info.get("name"):
        return f"https://zenodo.org/records/{info['record']}/files/{info['name']}?download=1"
    return None


def file_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - integrity check against the publisher's own md5, not security
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


ARCHIVE_SUFFIX = r"\.(zip|tar|tar\.gz|tgz)$"


def archive_extracted(folder: Path, name: str) -> bool:
    """True when the archive ``name`` was already unpacked whole under ``folder/extracted`` (its marker exists),
    whether or not the archive itself is still on disk."""
    if not re.search(ARCHIVE_SUFFIX, name.lower()):
        return False
    return (folder / "extracted" / re.sub(ARCHIVE_SUFFIX, "", name) / ".extracted").exists()


def _stamp(path: Path) -> Path:
    return path.with_name(path.name + ".ok")


def verified_before(path: Path, size: int, checksum: str) -> bool:
    """True when ``path`` passed its check on an earlier run and has not changed since (same size and time)."""
    try:
        seen = json.loads(_stamp(path).read_text())
        stat = path.stat()
    except Exception:
        return False
    return seen == {"size": stat.st_size, "mtime": int(stat.st_mtime), "checksum": checksum} and (
        not size or stat.st_size == size
    )


def mark_verified(path: Path, checksum: str) -> None:
    stat = path.stat()
    _stamp(path).write_text(json.dumps({"size": stat.st_size, "mtime": int(stat.st_mtime), "checksum": checksum}))


def bytes_on_disk(tmp: Path) -> int:
    """What an interrupted download left: the single-connection file and the pieces of a parallel one."""
    total = tmp.stat().st_size if tmp.exists() else 0
    return total + sum(p.stat().st_size for p in tmp.parent.glob(tmp.name + ".[0-9][0-9]"))


def verify_file(path: Path, size: int = 0, checksum: str = "") -> str | None:
    """``None`` when the file matches what the publisher lists, else the reason it does not."""
    if not path.exists():
        return "missing"
    actual = path.stat().st_size
    if size and actual != size:
        return f"size {actual} of {size} bytes"
    if actual < 1024:
        return f"only {actual} bytes"
    if checksum and ":" in checksum:
        algo, _, want = checksum.partition(":")
        if algo.lower() == "md5" and file_md5(path) != want.lower():
            return "md5 mismatch"
    return None


def _fetch(url: str, tmp: Path, size: int, log=log) -> int:
    """One pass of a download into ``tmp``, resumed from its current length with a Range request. Returns the
    bytes on disk afterwards. ``urllib`` returns a body cut short by the server or a proxy as if it were complete
    (it never raises IncompleteRead), so the caller compares the length with the published size."""
    import urllib.request

    have = tmp.stat().st_size if tmp.exists() else 0
    if size and have >= size:
        return have
    headers = {"User-Agent": "helix-lm tristream_molab"}
    if have:
        headers["Range"] = f"bytes={have}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", 200)
        if have and status != 206:
            have = 0  # the server ignored the range: start over
        kind = (response.headers.get("Content-Type") or "").lower()
        if "text/html" in kind or "application/json" in kind:
            raise OSError(f"the server answered with {kind.split(';')[0]} instead of the file")
        with open(tmp, "ab" if have else "wb") as handle:
            written, last = have, time.time()
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                handle.write(block)
                written += len(block)
                if size and time.time() - last > 30:
                    log(f"  {tmp.stem}: {written / 1e6:,.0f} of {size / 1e6:,.0f} MB")
                    last = time.time()
    return tmp.stat().st_size if tmp.exists() else 0


def _fetch_parallel(url: str, tmp: Path, size: int, log=log, connections: int = 16, attempts: int = 30) -> int:
    """Fetch ``size`` bytes with ``connections`` ranged requests at once. Zenodo (and many proxies) throttle each
    connection; sixteen of them fill the link. Each piece lives in its own file and resumes from its own length,
    so a dropped connection costs only its unfinished piece; the pieces are joined into ``tmp`` at the end.
    Progress (MB, MB/s, minutes left) is logged every 30 s. Raises when the server refuses ranges."""
    import urllib.request

    step = -(-size // connections)
    spans = [(a, min(size, a + step) - 1) for a in range(0, size, step)]
    pieces = [tmp.with_name(f"{tmp.name}.{i:02d}") for i in range(len(spans))]
    errors: list[str] = []
    if tmp.exists() and not any(piece.exists() for piece in pieces):
        # a one-connection download was interrupted: cut what it got into the pieces instead of starting over
        have = tmp.stat().st_size
        if have >= size:
            return have  # already whole: the caller checks it
        with open(tmp, "rb") as source:
            for (start, end), piece in zip(spans, pieces, strict=True):
                if have <= start:
                    break
                source.seek(start)
                with open(piece, "wb") as out:
                    out.write(source.read(min(have, end + 1) - start))
        tmp.unlink()

    def piece_size(i: int) -> int:
        return pieces[i].stat().st_size if pieces[i].exists() else 0

    def worker(i: int) -> None:
        start, end = spans[i]
        want, stalls = end - start + 1, 0
        while stalls < attempts:
            have = piece_size(i)
            if have >= want:
                return
            request = urllib.request.Request(
                url, headers={"User-Agent": "helix-lm tristream_molab", "Range": f"bytes={start + have}-{end}"}
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    if getattr(response, "status", 200) != 206:
                        raise OSError("ranges refused")
                    with open(pieces[i], "ab") as handle:
                        while True:
                            block = response.read(1 << 18)
                            if not block:
                                break
                            handle.write(block)
            except OSError as error:
                if "ranges refused" in str(error):
                    errors.append("ranges refused")
                    return
            stalls = 0 if piece_size(i) > have else stalls + 1
            if piece_size(i) < want:
                time.sleep(min(30, 2 * stalls))
        errors.append(f"piece {i} stalled")

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(len(spans))]
    for thread in threads:
        thread.start()
    begin = time.time()
    first = sum(piece_size(i) for i in range(len(spans)))
    last_log = begin
    while any(thread.is_alive() for thread in threads):
        time.sleep(1)
        if time.time() - last_log >= 30:
            got = sum(piece_size(i) for i in range(len(spans)))
            rate = (got - first) / max(1e-6, time.time() - begin)
            left = (size - got) / rate / 60 if rate > 0 else float("inf")
            log(
                f"  {tmp.stem}: {got / 1e6:,.0f} of {size / 1e6:,.0f} MB, "
                f"{rate / 1e6:.1f} MB/s, about {left:.0f} min left"
            )
            last_log = time.time()
    if "ranges refused" in errors:
        for piece in pieces:
            piece.unlink(missing_ok=True)
        raise OSError("ranges refused")
    if any(piece_size(i) < spans[i][1] - spans[i][0] + 1 for i in range(len(spans))):
        return sum(piece_size(i) for i in range(len(spans)))  # the caller tries again; the pieces stay
    with open(tmp, "wb") as out:
        for piece in pieces:
            with open(piece, "rb") as handle:
                shutil.copyfileobj(handle, out, length=1 << 22)
    for piece in pieces:
        piece.unlink(missing_ok=True)
    return tmp.stat().st_size


def download(
    url: str,
    target: Path,
    log=log,
    size: int = 0,
    checksum: str = "",
    alternate: str | None = None,
    attempts: int = 6,
    parallel_from: int = 64 << 20,
) -> Path:
    """Fetch ``url`` into ``target`` and keep only a file that is whole: the size (and md5 when the publisher
    lists one) is checked before the file gets its final name; a short body is resumed with a Range request,
    a bad one is deleted and fetched again, alternating between the two routes; a file already on disk that
    does not match is thrown away. Only passes that gain nothing count against ``attempts``: a link that keeps
    cutting out but moves forward is followed until the file is whole."""
    if target.exists():
        if verified_before(target, size, checksum):
            log(f"  {target.name}: already on disk and checked, not downloaded again")
            return target
        log(f"  {target.name}: already on disk; checking it is whole ({target.stat().st_size / 1e9:.2f} GB)")
        problem = verify_file(target, size, checksum)
        if problem is None:
            mark_verified(target, checksum)
            log(f"  {target.name}: whole, not downloaded again")
            return target
        log(f"  {target.name} on disk is bad ({problem}); downloading it again")
        target.unlink()
        _stamp(target).unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    routes = [url] + ([alternate] if alternate else [])
    partial = bytes_on_disk(tmp)
    if partial:
        log(f"  {target.name}: resuming, {partial / 1e6:,.0f} of {size / 1e6:,.0f} MB already on disk")
    else:
        log(f"downloading {url} -> {target.name}" + (f" ({size / 1e9:.2f} GB)" if size else ""))
    problem, stalls, attempt, ranges_refused = "not started", 0, 0, False
    while stalls < attempts:
        route = routes[attempt % len(routes)]
        before = tmp.stat().st_size if tmp.exists() else 0
        attempt += 1
        try:
            if size >= parallel_from and not ranges_refused:
                try:
                    have = _fetch_parallel(route, tmp, size, log)
                except OSError as error:
                    if "ranges refused" not in str(error):
                        raise
                    ranges_refused = True
                    log("  the server refuses ranged requests; one connection instead")
                    have = _fetch(route, tmp, size, log)
            else:
                have = _fetch(route, tmp, size, log)
        except Exception as error:
            have = tmp.stat().st_size if tmp.exists() else 0
            log(f"  pass {attempt}: {type(error).__name__}: {str(error)[:100]} ({have / 1e6:.0f} MB on disk)")
            stalls = 0 if have > before else stalls + 1
            time.sleep(min(60, 5 * stalls))
            continue
        problem = verify_file(tmp, size, checksum)
        if problem is None:
            tmp.rename(target)
            mark_verified(target, checksum)
            return target
        if size and have < size:
            log(f"  pass {attempt}: cut short at {have / 1e6:.0f} of {size / 1e6:.0f} MB; resuming")
            stalls = 0 if have > before else stalls + 1
            continue
        log(f"  pass {attempt}: {problem}; starting over")
        tmp.unlink(missing_ok=True)
        stalls += 1
        time.sleep(min(60, 5 * stalls))
    tmp.unlink(missing_ok=True)
    raise OSError(f"{target.name}: {problem} after {attempt} passes")


def stage_data(args) -> None:
    root = Path(args.root)
    raw = root / "raw"
    found = {}
    for name, info in CLEAN_SOURCES.items():
        if name in args.skip:
            continue
        try:
            files = zenodo_files(info["zenodo"])
            kept = [
                f
                for f in files
                if f["name"].lower().endswith((".zip", ".tar.gz", ".tgz", ".wav", ".flac", ".csv", ".json"))
            ]
            gb = sum(f["size"] for f in kept) / 1e9
            log(f"{name}: {len(kept)} files, {gb:.1f} GB ({info['note']})")
            if gb > args.max_gb:
                log(f"  skipped: over --max-gb {args.max_gb}")
                continue
            fetched = 0
            for f in kept:
                if archive_extracted(raw / name, f["name"]):
                    log(f"  {f['name']}: already unpacked on disk, not downloaded again")
                    fetched += 1
                    continue
                try:
                    download(f["url"], raw / name / f["name"], size=f["size"], checksum=f["checksum"],
                             alternate=alternate_url(f))  # fmt: skip
                    fetched += 1
                except Exception as error:
                    log(f"  {f['name']}: not fetched ({type(error).__name__}: {str(error)[:80]})")
            extracted = extract_archives(raw / name)
            found[name] = fetched
            if fetched < len(kept):
                log(f"  {name}: {fetched} of {len(kept)} files; the rest is retried on the next run")
            elif extracted == 0 and not any((raw / name / "extracted").rglob(".extracted")):
                log(f"  {name}: nothing extracted; the archives are retried on the next run")
        except Exception as error:
            log(f"{name}: not fetched ({type(error).__name__}: {str(error)[:80]}); continue without it")
    if args.emilia_hours > 0 and token():
        try:
            from huggingface_hub import snapshot_download

            # Emilia-YODAS is the CC BY 4.0 half (1,362 EN archives named EN-B000000.tar ...; about 50 h each):
            # fetch exactly the first N, by name, and only those whose clips are not already unpacked (the tar is
            # deleted once its clips are out, so the tar being gone means nothing)
            emilia = raw / "emilia-yodas"
            shards = [f"EN-B{i:06d}" for i in range(max(1, int(args.emilia_hours / 50)))]
            todo = [s for s in shards if not archive_extracted(emilia, s + ".tar")]
            if todo:
                log(f"emilia-yodas: {len(shards) - len(todo)} of {len(shards)} shards already on disk; "
                    f"downloading {len(todo)}")  # fmt: skip
                snapshot_download(
                    "amphion/Emilia-Dataset",
                    repo_type="dataset",
                    allow_patterns=[f"Emilia-YODAS/EN/{s}.tar" for s in todo],
                    local_dir=str(emilia),
                    token=token(),
                )
            else:
                log(f"emilia-yodas: all {len(shards)} shards already on disk, not downloaded again")
            extract_archives(emilia, delete=True)  # the tar is gone once its clips are out
            found["emilia-yodas"] = sum(archive_extracted(emilia, s + ".tar") for s in shards)
            log(f"emilia-yodas: {found['emilia-yodas']} of {len(shards)} shards unpacked (CC BY 4.0 half only)")
        except Exception as error:
            log(f"emilia-yodas: not fetched ({type(error).__name__}: {str(error)[:80]})")
    if args.include_research and token():
        for name, repo in RESEARCH_SOURCES.items():
            try:
                from huggingface_hub import snapshot_download

                snapshot_download(repo, repo_type="dataset", local_dir=str(raw / name), token=token(), max_workers=4)
                found[name] = 1
                log(f"{name}: fetched for EVALUATION ONLY (non-commercial licence)")
            except Exception as error:
                log(f"{name}: not fetched ({type(error).__name__})")
    uploads = list((root / "uploads").rglob("*"))
    log(f"uploads folder: {sum(p.suffix.lower() in AUDIO_SUFFIXES for p in uploads)} audio files")
    (root / "markers").mkdir(exist_ok=True)
    (root / "data_found.json").write_text(json.dumps(found, indent=1))
    missing = missing_sources(root, args)
    if missing:
        counter = root / "markers" / "data.attempts"
        tries = (int(counter.read_text()) if counter.exists() else 0) + 1
        counter.write_text(str(tries))
        if tries <= args.data_attempts:
            raise RuntimeError(
                f"sources not whole yet: {', '.join(missing)}; data attempt {tries} of {args.data_attempts}"
            )
        log(f"!! continuing without {', '.join(missing)} after {tries - 1} attempts; they are retried on each run")
    mark(root, "data")


# ---------------------------------------------------------------------------------------- prepare --
def emilia_speaker(path: Path) -> str:
    """The speaker id of an Emilia clip: from the JSON beside it, else the ``EN_B000000_S00000`` prefix."""
    side = path.with_suffix(".json")
    if side.exists():
        try:
            data = json.loads(side.read_text())
            if isinstance(data, dict) and data.get("speaker"):
                return str(data["speaker"])
        except Exception:
            pass
    match = re.match(r"([A-Z]{2}_B\d+_S\d+)", path.stem)
    return match.group(1) if match else path.parent.name


def stage_prepare(args) -> None:
    import torch

    from helix_lm.music.data import find_audio, prepare_uploads
    from helix_lm.music.separation import try_pretrained

    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    separate = try_pretrained("htdemucs")
    log("separator:", "HT-Demucs (MIT)" if separate else "none (install demucs); full songs will be rejected")
    log(f"measurements on {device}, {loader_workers(args.prepare_workers)} workers")
    manifests = {}
    sources = [
        ("uploads", args.uploads_are_mixtures, args.upload_accent, True, None, "singing"),
        ("raw/vocalset", False, "unknown", True, lambda p: "vocalset/" + p.stem.split("_")[0], "singing"),
        ("raw/dagstuhl-choirset", False, "unknown", True, None, "choir"),
        ("raw/esmuc", False, "unknown", True, None, "choir"),
        ("raw/emilia-yodas", False, "unknown", False, emilia_speaker, "speech"),  # speech: no song analysis
    ]
    for folder, is_mixture, accent, analysis, singer_of, domain in sources:
        path = root / folder
        if not path.exists() or not find_audio([path]):
            continue
        out = root / "prepared" / folder.replace("/", "_")
        if (out / "manifest.json").exists() and not args.force:
            manifests[folder] = json.loads((out / "manifest.json").read_text())
            continue
        log(f"preparing {folder}: {len(find_audio([path]))} files")
        manifests[folder] = prepare_uploads(
            [path],
            out,
            sample_rate=args.sample_rate,
            singer=folder,
            accent=accent,
            mixtures=is_mixture,
            separate=separate,
            log=log if folder == "uploads" else (lambda *_: None),
            device=device,
            analysis=analysis,
            workers=loader_workers(args.prepare_workers),
            singer_of=singer_of,
            domain=domain,
        )
        log(f"  {folder}: {manifests[folder]['accepted']} clips accepted, {manifests[folder]['hours']:.1f} h")
    hours = sum(m.get("hours", 0) for m in manifests.values())
    accepted = sum(m.get("accepted", 0) for m in manifests.values())
    all_gaps = [g for m in manifests.values() for g in m.get("gaps", [])]
    log(f"prepared: {accepted} clips accepted, {hours:.1f} h; gaps: {all_gaps[:6]}")
    (root / "prepared" / "summary.json").write_text(
        json.dumps({"hours": hours, "accepted": accepted, "sources": list(manifests)}, indent=1)
    )
    mark(root, "prepare")


# ------------------------------------------------------------------------------------------ codec --
def stage_codec(args) -> None:
    import torch

    from helix_lm.audio import AudioCodec, CodecConfig, codec_losses
    from helix_lm.audio.codec import MultiPeriodDiscriminator, discriminator_loss
    from helix_lm.audio.data import read_wave
    from helix_lm.music.data import find_audio

    root = Path(args.root)
    if args.codec_hours <= 0:
        log("codec stage skipped (--codec-hours 0)")
        mark(root, "codec")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = CodecConfig.at_frame_rate(args.codec_rate, sample_rate=args.sample_rate)
    codec, disc = AudioCodec(config).to(device), MultiPeriodDiscriminator().to(device)
    files = find_audio([root / "prepared"])
    files = [f for f in files if f.name.endswith(".vocal.wav") or f.name.endswith(".instrumental.wav")]
    if not files:
        log("!! no prepared audio; run the prepare stage")
        return
    (root / "codec").mkdir(parents=True, exist_ok=True)  # every stage makes its own folder: no order dependence
    ckpt = root / "codec" / "codec.pt"
    step = 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        codec.load_state_dict(state["codec"])
        disc.load_state_dict(state["disc"])
        step = state["step"]
    opt_g = torch.optim.AdamW(codec.parameters(), lr=1e-4, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-4, betas=(0.8, 0.99))
    budget = Budget(args.codec_hours, checkpoint_meta(ckpt).get("hours", 0.0) if ckpt.exists() else 0.0)
    crop = int(1.5 * args.sample_rate)
    g = torch.Generator().manual_seed(step)
    sizer = BatchSizer(args.batch if args.batch > 0 else (96 if device.type == "cuda" else 4), "codec", device,
                       fixed=args.batch > 0)  # fmt: skip
    # the schedule counts clips, not steps, so a bigger batch does not delay the discriminator
    clips_seen = int(state.get("clips", step * 96)) if ckpt.exists() else 0
    speech_files = [f for f in files if "emilia" in str(f)]
    sung_files = [f for f in files if "emilia" not in str(f)] or files
    log(
        f"codec: {len(sung_files)} sung + {len(speech_files)} spoken files (drawn half and half), "
        f"{budget.remaining:.1f} h left of {args.codec_hours} h, resuming at step {step}"
    )

    # the high register must not starve: clips whose F0 reaches 500 Hz get a fixed share of every batch
    high_files = [
        Path(r["dir"]) / f"{Path(r['path']).stem}.vocal.wav"
        for r in prepared_records(root)
        if r["tags"].get("f0_high", 0) >= 500
    ]
    high_files = [f for f in high_files if f.exists()]
    low_files = [f for f in sung_files if f not in set(high_files)] or sung_files
    log(f"codec: {len(high_files)} high-register files get {args.high_register_share:.0%} of each batch")
    from helix_lm.audio.codec import codebook_usage_by_register
    from helix_lm.audio.refiner import high_band_loss
    from helix_lm.music.regularize import augment

    def pick_file(gen) -> Path:
        draw = float(torch.rand((), generator=gen))
        if high_files and draw < args.high_register_share:
            pool = high_files
        elif speech_files and draw < args.high_register_share + 0.5 * (1 - args.high_register_share):
            pool = speech_files
        else:
            pool = sung_files
        return pool[int(torch.randint(0, len(pool), (), generator=gen))]

    def crop_of(path: Path, length: int, gen) -> torch.Tensor:
        wave = read_wave(path, args.sample_rate)
        if wave.shape[-1] < length:
            wave = torch.nn.functional.pad(wave, (0, length - wave.shape[-1]))
        start = int(torch.randint(0, wave.shape[-1] - length + 1, (), generator=gen))
        return wave[start : start + length]

    def make_batch(gen, n: int) -> torch.Tensor:
        batch = []
        for _ in range(n):
            wave = crop_of(pick_file(gen), crop, gen)
            if float(torch.rand((), generator=gen)) < args.augment_prob:  # rooms, EQ, noise: "clean" is not a domain
                wave, _ = augment(wave, args.sample_rate, seed=int(torch.randint(0, 1 << 30, (), generator=gen)))
            batch.append(wave)
        return torch.stack(batch)[:, None]

    audit_high = torch.stack([crop_of(f, crop, g) for f in high_files[:16]]) if high_files else None
    audit_low = torch.stack([crop_of(f, crop, g) for f in low_files[:16]]) if high_files else None
    prefetch = Prefetcher(make_batch, lambda: sizer.size, chunk=16, workers=loader_workers(args.loader_workers),
                          seed=step, device=device, mode=args.loader_mode)  # fmt: skip
    log(f"codec: {prefetch.workers} {prefetch.kind} build the batches")
    last_save = time.time()
    while not budget.over() and not (root / "stop").exists():
        wave = prefetch.get().to(device, non_blocking=True)
        disc_on = clips_seen > 5000 * 96
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = codec(wave)
                terms = codec_losses(
                    out,
                    wave,
                    config,
                    discriminator=disc if disc_on else None,
                    time_weight=1.0 if clips_seen < 20000 * 96 else 0.1,
                )
                terms["high_band"] = 2.0 * high_band_loss(out["recon"].float(), wave.float(), args.sample_rate)
                terms["total"] = terms["total"] + terms["high_band"]
            opt_g.zero_grad()
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 10.0)
            opt_g.step()
            if disc_on:
                d_loss = discriminator_loss(disc, out["recon"].float(), wave)
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
        except torch.cuda.OutOfMemoryError:
            opt_g.zero_grad(set_to_none=True)
            opt_d.zero_grad(set_to_none=True)
            del wave
            torch.cuda.empty_cache()
            prefetch.drain()
            log(f"codec out of memory: batch now {sizer.out_of_memory()}")
            continue
        step += 1
        clips_seen += wave.shape[0]
        sizer.tick(wave.shape[0])
        if step % 100 == 0:
            log(
                f"codec step {step}: mel {terms['mel'].item():.3f} high-band {terms['high_band'].item():.3f} "
                f"commit {terms['commitment'].item():.4f} usage {codec.quantizer.semantic.usage():.0%} "
                f"batch {wave.shape[0]}, GPU waited for data {prefetch.wait_share():.0%}"
            )
        if step % 2000 == 0 and audit_high is not None:
            codec.eval()
            audit = codebook_usage_by_register(codec, audit_high.to(device), audit_low.to(device))
            codec.train()
            (root / "codec" / "audit.json").write_text(json.dumps({"step": step, **audit}, indent=1))
            log(
                "codebook audit: high-register "
                + " ".join(f"{u:.0%}" for u in audit["high"])
                + " | low-register "
                + " ".join(f"{u:.0%}" for u in audit["low"])
                + (" !! upper tiers starved on the high register" if audit["starved"] else " (ok)")
            )
        if step % 2000 == 0 or time.time() - last_save > 1800:
            save_checkpoint(
                {
                    "codec": codec.state_dict(),
                    "disc": disc.state_dict(),
                    "step": step,
                    "clips": clips_seen,
                    "config": config.__dict__,
                    "hours": budget.elapsed,
                },
                ckpt,
            )
            last_save = time.time()
            hub_push(args, reason=f"codec step {step}")
    prefetch.close()
    save_checkpoint(
        {
            "codec": codec.state_dict(),
            "disc": disc.state_dict(),
            "step": step,
            "clips": clips_seen,
            "config": config.__dict__,
            "hours": budget.elapsed,
        },
        ckpt,
    )
    log(f"codec saved at step {step}")
    mark(root, "codec")


# ------------------------------------------------------------------------------------------ train --
def load_codec(root: Path, device):
    import torch

    from helix_lm.audio import AudioCodec, CodecConfig

    ckpt = root / "codec" / "codec.pt"
    if not ckpt.exists():
        return None
    state = torch.load(ckpt, map_location=device)
    codec = AudioCodec(CodecConfig(**state["config"])).to(device).eval()
    codec.load_state_dict(state["codec"])
    return codec


# ----------------------------------------------------------------------------------------- refiner --
def load_refiner(root: Path, device):
    import torch

    from helix_lm.audio.refiner import HarmonicNoiseRefiner, RefinerConfig

    ckpt = root / "refiner" / "refiner.pt"
    if not ckpt.exists():
        return None
    state = torch.load(ckpt, map_location=device)
    refiner = HarmonicNoiseRefiner(RefinerConfig(**state["config"])).to(device).eval()
    refiner.load_state_dict(state["refiner"])
    return refiner


def stage_refiner(args) -> None:
    """The refiner learns to put back what the codec averages away: trained on (codec reconstruction -> original)
    pairs of the prepared dry vocals, with the multi-period discriminator after the first 2,000 steps."""
    import torch

    from helix_lm.audio.codec import MultiPeriodDiscriminator, discriminator_loss
    from helix_lm.audio.data import read_wave
    from helix_lm.audio.refiner import HarmonicNoiseRefiner, RefinerConfig, refiner_losses
    from helix_lm.audio.singing import estimate_f0

    root = Path(args.root)
    if args.refiner_hours <= 0:
        log("refiner stage skipped (--refiner-hours 0)")
        mark(root, "refiner")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = load_codec(root, device)
    if codec is None:
        log("!! no codec (run the codec stage first)")
        return
    files = sorted((root / "prepared").rglob("*.vocal.wav"))
    if not files:
        log("!! no prepared vocals")
        return
    (root / "refiner").mkdir(parents=True, exist_ok=True)
    ckpt = root / "refiner" / "refiner.pt"
    config = RefinerConfig(
        sample_rate=args.sample_rate, frame_dim=codec.config.dim, frame_rate=codec.config.frame_rate
    )
    refiner = HarmonicNoiseRefiner(config).to(device)
    disc = MultiPeriodDiscriminator().to(device)
    opt = torch.optim.AdamW(refiner.parameters(), lr=2e-4, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=2e-4, betas=(0.8, 0.99))
    step = 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        refiner.load_state_dict(state["refiner"])
        disc.load_state_dict(state["disc"])
        step = state["step"]
        log(f"refiner resumed at step {step}")
        warn_if_codec_moved(root, ckpt, "refiner")
    log(f"refiner: {refiner.num_parameters() / 1e6:.1f}M parameters, {len(files)} clips, {args.refiner_hours:.0f} h")
    budget = Budget(args.refiner_hours, checkpoint_meta(ckpt).get("hours", 0.0) if ckpt.exists() else 0.0)
    sizer = BatchSizer(max(4, args.batch if args.batch > 0 else (48 if device.type == "cuda" else 4)), "refiner",
                       device, fixed=args.batch > 0)  # fmt: skip
    clips_seen = int(state.get("clips", step * 48)) if ckpt.exists() else 0
    need = int(2.0 * args.sample_rate)
    hop_fine = config.hop
    tf = need // hop_fine
    need = tf * hop_fine
    last_save = time.time()

    def save() -> None:
        save_checkpoint(
            {
                "refiner": refiner.state_dict(),
                "disc": disc.state_dict(),
                "step": step,
                "clips": clips_seen,
                "config": config.__dict__,
                "hours": budget.elapsed,
            },
            ckpt,
        )

    from helix_lm.music.regularize import augment

    def make_batch(gen, n: int):
        waves = []
        for _ in range(n):
            wave = read_wave(files[int(torch.randint(0, len(files), (), generator=gen))], args.sample_rate)
            if wave.shape[-1] < need:
                wave = torch.nn.functional.pad(wave, (0, need - wave.shape[-1]))
            start = int(torch.randint(0, wave.shape[-1] - need + 1, (), generator=gen))
            wave = wave[start : start + need]
            if float(torch.rand((), generator=gen)) < args.augment_prob:
                wave, _ = augment(wave, args.sample_rate, seed=int(torch.randint(0, 1 << 30, (), generator=gen)))
            waves.append(wave)
        waves = torch.stack(waves)
        # the pitch track is CPU work: done here, in the workers, so the GPU never waits on it
        f0_fine = estimate_f0(waves, args.sample_rate, frame_rate=config.fine_rate)[:, :tf]
        return waves, torch.nn.functional.pad(f0_fine, (0, tf - f0_fine.shape[-1]))

    prefetch = Prefetcher(make_batch, lambda: sizer.size, chunk=8, workers=loader_workers(args.loader_workers),
                          seed=step, device=device, mode=args.loader_mode)  # fmt: skip
    log(f"refiner: {prefetch.workers} {prefetch.kind} build the batches and their pitch tracks")
    while not budget.over() and not (root / "stop").exists():
        wave, f0_fine = prefetch.get()
        wave = wave.to(device, non_blocking=True)
        f0_fine = f0_fine.to(device, non_blocking=True)
        disc_on = clips_seen > 2000 * 48
        try:
            with torch.no_grad():
                frames = codec.listen(wave)
                recon = codec.decode_frames(frames)[:, :, :need]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = refiner(recon, frames, f0_fine)
                terms = refiner_losses(out, wave, args.sample_rate, disc if disc_on else None)
            opt.zero_grad()
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            opt.step()
            if disc_on:
                opt_d.zero_grad()
                discriminator_loss(disc, out["wave"].detach(), wave[:, None]).backward()
                opt_d.step()
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            prefetch.drain()
            log(f"refiner out of memory: batch now {sizer.out_of_memory()}")
            continue
        step += 1
        clips_seen += wave.shape[0]
        sizer.tick(wave.shape[0])
        if step % 50 == 0:
            log(
                f"refiner step {step}: "
                + " ".join(f"{k} {float(v.detach()):.3f}" for k, v in terms.items())
                + f" batch {wave.shape[0]}, GPU waited for data {prefetch.wait_share():.0%}"
            )
        if time.time() - last_save > 1800:
            save()
            last_save = time.time()
            hub_push(args, reason=f"refiner step {step}")
    prefetch.close()
    save()
    log(f"refiner saved at step {step}")
    mark(root, "refiner")


# --------------------------------------------------------------------------------------------- air --
def stage_air(args) -> None:
    """The air band above the codec's Nyquist, learned from your own recordings at 44.1/48 kHz (the uploads
    folder, read at 48 kHz). Opt in with --air-hours; without 48 kHz sources there is nothing to learn from."""
    import torch

    from helix_lm.audio.data import read_wave
    from helix_lm.audio.refiner import AirBandExtension, AirConfig, air_losses
    from helix_lm.music.data import find_audio

    root = Path(args.root)
    if args.air_hours <= 0:
        log("air stage skipped (--air-hours 0)")
        mark(root, "air")
        return
    files = find_audio([root / "uploads"])
    if not files:
        log("!! no uploads to learn the air band from")
        mark(root, "air")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (root / "air").mkdir(parents=True, exist_ok=True)
    ckpt = root / "air" / "air.pt"
    config = AirConfig(in_rate=args.sample_rate, out_rate=48000)
    air = AirBandExtension(config).to(device)
    opt = torch.optim.AdamW(air.parameters(), lr=3e-4)
    step = 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        air.load_state_dict(state["air"])
        step = state["step"]
    budget = Budget(args.air_hours, checkpoint_meta(ckpt).get("hours", 0.0) if ckpt.exists() else 0.0)
    g = torch.Generator().manual_seed(step)
    need = int(2.0 * 48000)
    ratio = 48000 // args.sample_rate
    batch_size = max(2, (args.batch if args.batch > 0 else 16) // 2)
    while not budget.over() and not (root / "stop").exists():
        waves = []
        for _ in range(batch_size):
            wave = read_wave(files[int(torch.randint(0, len(files), (), generator=g))], 48000)
            if wave.shape[-1] < need:
                wave = torch.nn.functional.pad(wave, (0, need - wave.shape[-1]))
            start = int(torch.randint(0, wave.shape[-1] - need + 1, (), generator=g))
            waves.append(wave[start : start + need])
        target = torch.stack(waves).to(device)
        body = torch.nn.functional.avg_pool1d(target[:, None], ratio, stride=ratio)  # the 24 kHz body of the take
        out = air(body)
        terms = air_losses(out, target, 48000, cutoff=args.sample_rate / 2 * 0.95)
        opt.zero_grad()
        terms["total"].backward()
        opt.step()
        step += 1
        if step % 50 == 0:
            log(f"air step {step}: " + " ".join(f"{k} {float(v.detach()):.3f}" for k, v in terms.items()))
        if step % 500 == 0:
            save_checkpoint(
                {"air": air.state_dict(), "step": step, "config": config.__dict__, "hours": budget.elapsed}, ckpt
            )
            hub_push(args, reason=f"air step {step}")
    save_checkpoint({"air": air.state_dict(), "step": step, "config": config.__dict__, "hours": budget.elapsed}, ckpt)
    log(f"air band saved at step {step}")
    mark(root, "air")


def stage_train(args) -> None:
    import torch

    from helix_lm.audio import SpeakerEncoder
    from helix_lm.audio.data import read_wave
    from helix_lm.music.regularize import ACCENTS, ClipTags, augment, mixture_weights, speech_share_at, split_singers
    from helix_lm.music.tristream import TriStreamConfig, TriStreamForSinging, noise_features, source_features
    from helix_lm.optim import build_optimizer

    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = load_codec(root, device)
    if codec is None:
        log("!! no codec (run the codec stage, or point --codec at a saved one)")
        return
    sizer = BatchSizer(args.batch if args.batch > 0 else 64, "train", device, fixed=args.batch > 0)
    if args.batch <= 0:
        log("auto batch size: starting at 64 clips, then the fastest size for this card is measured")
    records = []
    for manifest in (root / "prepared").rglob("manifest.json"):
        data = json.loads(manifest.read_text())
        records += [{**clip, "dir": str(manifest.parent)} for clip in data["clips"] if clip["accepted"]]
    if not records:
        log("!! no accepted clips")
        return
    tags = [ClipTags(**r["tags"]) for r in records]
    split = split_singers([t.singer for t in tags])
    train_idx = [i for i, t in enumerate(tags) if split[t.singer] == "train"]
    train_tags = [tags[i] for i in train_idx]
    sung = sum(t.domain != "speech" for t in train_tags)
    log(
        f"domains: {sung} sung clips, {len(train_tags) - sung} spoken clips; speech share annealed "
        f"{args.speech_share_start:.0%} -> {args.speech_share_end:.0%} of each batch, spoken loss weight "
        f"{args.speech_loss_weight}"
    )
    # shared memory: the curriculum updates it in place and the batch workers (other processes) read it
    weights = mixture_weights(
        train_tags, speech_share_at(0.0, args.speech_share_start, args.speech_share_end)
    ).share_memory_()
    config = TriStreamConfig.named(
        args.size,
        frame_dim=codec.config.dim,
        sample_rate=args.sample_rate,
        semantic_codebook=codec.config.codebook_size,  # the content codes come from this codec
        frame_rate=codec.config.frame_rate,
    )
    model = TriStreamForSinging(config).to(device)
    speaker_encoder = SpeakerEncoder(frame_dim=codec.config.dim, embedding_dim=config.speaker_dim).to(device)
    params = list(model.parameters()) + list(speaker_encoder.parameters())
    optimizer = build_optimizer(torch.nn.ModuleList([model, speaker_encoder]), "muon", lr=args.lr, weight_decay=0.05)
    run = root / "runs" / args.size
    run.mkdir(parents=True, exist_ok=True)
    step = 0
    ckpt = run / "latest.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        speaker_encoder.load_state_dict(state["speaker"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        log(f"resumed at step {step}")
        warn_if_codec_moved(root, ckpt, "singing model")
    log(
        f"TriStream-2 {args.size}: {model.num_parameters() / 1e6:.0f}M parameters, {len(train_idx)} training clips, "
        f"{len(records) - len(train_idx)} validation clips (singer-disjoint)"
    )
    budget = Budget(args.train_hours, checkpoint_meta(ckpt).get("hours", 0.0) if ckpt.exists() else 0.0)
    log(f"train: {budget.remaining:.1f} h left of {args.train_hours} h")
    frames_per_clip = int(args.clip_seconds * config.frame_rate)
    singer_ids = {t.singer: i % config.num_singers for i, t in enumerate(tags)}
    last_save = time.time()

    def make_batch(gen, n: int):
        chosen = torch.multinomial(weights, n, replacement=True, generator=gen).tolist()
        batch_rows = [records[train_idx[i]] for i in chosen]
        waves, f0s, accents, singers = [], [], [], []
        loss_weights = [
            1.0 if r["tags"].get("domain", "singing") != "speech" else args.speech_loss_weight for r in batch_rows
        ]
        for r in batch_rows:
            stem = Path(r["path"]).stem
            wave = read_wave(Path(r["dir"]) / f"{stem}.vocal.wav", args.sample_rate)
            f0 = torch.load(Path(r["dir"]) / f"{stem}.f0.pt")
            need = int(args.clip_seconds * args.sample_rate)
            if wave.shape[-1] < need:
                wave = torch.nn.functional.pad(wave, (0, need - wave.shape[-1]))
                f0 = torch.nn.functional.pad(f0, (0, max(0, frames_per_clip - f0.shape[-1])))
            start = int(torch.randint(0, wave.shape[-1] - need + 1, (), generator=gen))
            f_start = int(start / args.sample_rate * config.frame_rate)
            waves.append(wave[start : start + need])
            f0s.append(f0[f_start : f_start + frames_per_clip])
            accents.append(ACCENTS.index(r["tags"]["accent"]) if r["tags"]["accent"] in ACCENTS else 0)
            singers.append(singer_ids[r["tags"]["singer"]])
        clean = torch.stack(waves)
        dirty = (
            clean.clone()
        )  # what the model hears: the dry take through rooms, EQ, noise, holes; the target stays dry
        for i in range(clean.shape[0]):
            if float(torch.rand((), generator=gen)) < args.augment_prob:
                dirty[i], _ = augment(
                    clean[i], args.sample_rate, seed=int(torch.randint(0, 1 << 30, (), generator=gen))
                )
        return clean, dirty, f0s, accents, singers, len(batch_rows), torch.tensor(loss_weights)

    prefetch = Prefetcher(make_batch, lambda: sizer.size, chunk=8, workers=loader_workers(args.loader_workers),
                          seed=step, device=device, mode=args.loader_mode)  # fmt: skip
    log(f"train: {prefetch.workers} {prefetch.kind} build the batches")
    last_share = -1.0
    while not budget.over() and not (root / "stop").exists():
        share = speech_share_at(
            budget.elapsed / max(budget.hours, 1e-6), args.speech_share_start, args.speech_share_end
        )
        if abs(share - last_share) >= 0.05:  # re-derive the sampler as the curriculum moves (in place: shared)
            weights.copy_(mixture_weights(train_tags, share))
            last_share = share
            log(f"curriculum: speech share now {share:.0%}")
        wave, dirty, f0s, accents, singers, rows, loss_weights = prefetch.get()
        wave = wave.to(device, non_blocking=True)
        dirty = dirty.to(device, non_blocking=True)
        with torch.no_grad():
            frames = codec.listen(wave)[:, :frames_per_clip]
            frames_in = codec.listen(dirty)[:, :frames_per_clip]
        t = frames.shape[1]
        f0 = torch.stack([torch.nn.functional.pad(f, (0, max(0, t - f.shape[0])))[:t] for f in f0s]).to(device)
        energy = frames.pow(2).mean(-1)
        speaker = speaker_encoder(frames)  # the clip's own voice: the model learns to copy it (in-context cloning)
        batch = {
            "source": source_features(f0, energy), "noise": noise_features(frames_in), "frames": frames_in,
            "target_frames": frames,
            "phonemes": torch.full((rows, 1), -1, dtype=torch.long, device=device),
            "speaker": speaker, "singer": torch.tensor(singers, device=device),
            "accent": torch.tensor(accents, device=device), "f0_hz": f0, "weights": loss_weights.to(device),
        }  # fmt: skip
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                losses = model(batch)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        except torch.cuda.OutOfMemoryError:
            # pushing the 96 GB to the edge: start big, halve on the first out-of-memory, keep going
            optimizer.zero_grad(set_to_none=True)
            del batch, frames
            torch.cuda.empty_cache()
            prefetch.drain()
            log(f"out of memory: batch now {sizer.out_of_memory()} clips of {args.clip_seconds:.0f} s")
            continue
        step += 1
        sizer.tick(rows)
        if step % 20 == 0:
            log(
                f"step {step}: "
                + " ".join(f"{k} {float(v.detach()):.3f}" for k, v in losses.items() if k != "mask_fraction")
                + f" batch {rows}, GPU waited for data {prefetch.wait_share():.0%}"
            )
        if time.time() - last_save > 1800:
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "speaker": speaker_encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": config.__dict__,
                    "hours": budget.elapsed,
                },
                ckpt,
            )
            last_save = time.time()
            log(f"checkpoint at step {step} ({budget.elapsed:.1f} h trained)")
            hub_push(args, reason=f"train step {step}")
    prefetch.close()
    save_checkpoint(
        {
            "model": model.state_dict(),
            "speaker": speaker_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config.__dict__,
            "hours": budget.elapsed,
        },
        ckpt,
    )
    log(f"saved at step {step}")
    mark(root, "train")


# ------------------------------------------------------------------------------------------- eval --
def stage_eval(args) -> None:
    import torch

    from helix_lm.audio import SpeakerEncoder
    from helix_lm.audio.data import read_wave
    from helix_lm.audio.singing import estimate_f0
    from helix_lm.audio.speaker import speaker_similarity
    from helix_lm.music.failsafe import Gates, synthesize_with_gates
    from helix_lm.music.register import register_break_report
    from helix_lm.music.regularize import ClipTags, split_singers
    from helix_lm.music.tristream import TriStreamConfig, TriStreamForSinging, noise_features, source_features

    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = load_codec(root, device)
    refiner = load_refiner(root, device)
    ckpt = root / "runs" / args.size / "latest.pt"
    if codec is None or not ckpt.exists():
        log("!! nothing to evaluate yet")
        return
    state = torch.load(ckpt, map_location=device)
    config = TriStreamConfig(**state["config"])
    model = TriStreamForSinging(config).to(device).eval()
    model.load_state_dict(state["model"])
    speaker_encoder = SpeakerEncoder(frame_dim=codec.config.dim, embedding_dim=config.speaker_dim).to(device).eval()
    speaker_encoder.load_state_dict(state["speaker"])
    records = []
    for manifest in (root / "prepared").rglob("manifest.json"):
        data = json.loads(manifest.read_text())
        records += [{**clip, "dir": str(manifest.parent)} for clip in data["clips"] if clip["accepted"]]
    tags = [ClipTags(**r["tags"]) for r in records]
    split = split_singers([t.singer for t in tags])
    paired = list(zip(records, tags, strict=True))
    sung = [(r, t) for r, t in paired if t.domain != "speech"] or paired
    held = [r for r, t in sung if split[t.singer] == "validation"][: args.eval_clips]
    if not held and sung:
        log("no validation singers yet (too few singers for the split): evaluating on training clips, optimistic")
        held = [r for r, _ in sung][: args.eval_clips]
    results = {
        "clips": len(held), "f0_rmse_cents": [], "speaker_similarity": [], "glitch_rate": [], "buzz_db": [],
        "hnr_db": [], "sync_lag_ms": [], "robotic": [], "register_pops_per_minute": [], "rounds": [], "fallbacks": [],
        "passed": [],
    }  # fmt: skip
    gates = Gates(steps=config.flow_steps, min_similarity=None, frame_rate=config.frame_rate)
    for r in held:
        stem = Path(r["path"]).stem
        wave = read_wave(Path(r["dir"]) / f"{stem}.vocal.wav", args.sample_rate)[: int(10 * args.sample_rate)].to(
            device
        )
        with torch.no_grad():
            frames = codec.listen(wave[None])
            f0 = estimate_f0(wave[None].cpu(), args.sample_rate, config.frame_rate)[:, : frames.shape[1]].to(device)
            batch = {
                "source": source_features(f0, frames.pow(2).mean(-1)),
                "noise": noise_features(frames),
                "frames": frames,
                "phonemes": torch.full((1, 1), -1, dtype=torch.long, device=device),
                "speaker": speaker_encoder(frames),
            }
            audio, report = synthesize_with_gates(
                model, codec, batch, args.sample_rate, f0[0], refiner=refiner, source_wave=wave, gates=gates
            )
            audio = audio.cpu()
        final = report.final
        results["f0_rmse_cents"].append(final["f0_rmse_cents"])
        with torch.no_grad():
            sim = speaker_similarity(speaker_encoder(codec.listen(audio[None].to(device))), speaker_encoder(frames))
        results["speaker_similarity"].append(float(sim))
        glitches = final["glitches"]
        results["glitch_rate"].append(
            float(glitches["clicks"].numel() + glitches["dropouts"].numel())
            / max(audio.shape[-1] / args.sample_rate, 1e-6)
        )
        results["buzz_db"].append(final["buzz"]["buzz_db"])
        results["hnr_db"].append(final.get("coherence", {}).get("hnr_db", 0.0))
        results["sync_lag_ms"].append(abs(final.get("sync", {}).get("lag_ms", 0.0)))
        results["robotic"].append(final.get("robotic", {}).get("score", 0.0))
        top = float(r["tags"].get("f0_high", 0.0)) * 0.85  # the clip's own upper register
        if top > 0:
            pops = register_break_report(audio, f0[0].cpu(), args.sample_rate, top, config.frame_rate)
            results["register_pops_per_minute"].append(pops["pops_per_minute"])
        results["rounds"].append(len(report.rounds))
        results["fallbacks"].append(float(any("sketch" in a or "transplant" in a for a in report.applied)))
        results["passed"].append(float(report.ok))
    summary = {k: (sum(v) / len(v) if isinstance(v, list) and v else v) for k, v in results.items()}
    (root / "eval").mkdir(parents=True, exist_ok=True)
    (root / "eval" / "summary.json").write_text(json.dumps(summary, indent=1))
    log("eval:", summary)
    mark(root, "eval")


# ------------------------------------------------------------------------------------------- push --
def stage_push(args, quiet: bool = False) -> None:
    if not args.hf_repo or not token():
        log("push skipped: --hf-repo and a token are needed")
        return
    if hub_push(args, force=True, reason="push stage") and not quiet:
        mark(Path(args.root), "push")


# -------------------------------------------------------------------------------------------- eta --
def stage_eta(args) -> None:
    """What the budgets add up to on one GPU, and where the time goes."""
    emilia_shards = max(1, int(args.emilia_hours / 50)) if args.emilia_hours > 0 else 0
    download_gb = 10 + emilia_shards * 2.5 + (104 if "slakh2100" not in args.skip and args.max_gb >= 105 else 0)
    download_h = download_gb / (50 * 3.6)  # 50 MB/s
    prepare_h = (
        args.emilia_hours * 360 / 20 + 15 * 360 / 10
    ) / 3600  # ~20 speech clips/s on a GPU, 10 singing clips/s
    rows = [
        ("data (download + extract)", download_h, f"about {download_gb:.0f} GB at 50 MB/s"),
        ("prepare (measurements on the GPU)", prepare_h, f"{args.emilia_hours:.0f} h speech + the singing sets"),
        ("codec", args.codec_hours, "the heaviest stage: the waveform codec at 24 kHz with its discriminator"),
        ("refiner", args.refiner_hours, "phase-true harmonics, 10 ms noise, residual"),
        ("air", args.air_hours, "only with 48 kHz uploads"),
        ("train TriStream-2", args.train_hours, f"{args.size}, auto batch, Muon"),
        ("eval + push", 1.0, "held-out singers through the fail-safe loop"),
    ]
    total = sum(h for _, h, _ in rows)
    log("ETA on one GPU, by stage (hours):")
    for name, hours, note in rows:
        log(f"  {name:34} {hours:8.1f}   {note}")
    log(f"  {'total':34} {total:8.1f}   = {total / 24:.1f} days")
    log(f"  pushes every {args.push_every_hours:.0f} h to {args.hf_repo or '(no repo)'}")
    log("shorter plan: --codec-hours 72 --refiner-hours 24 --train-hours 72 is about 7.5 days")


# ------------------------------------------------------------------------------------------- swap --
def stage_swap(args) -> None:
    import torch

    from helix_lm.audio import SpeakerEncoder
    from helix_lm.audio.data import read_wave, write_wave
    from helix_lm.music.conversion import Converter, swap_voice
    from helix_lm.music.tristream import TriStreamConfig, TriStreamForSinging

    if not args.song or not args.reference:
        log("swap needs --song and --reference")
        return
    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = load_codec(root, device)
    converter = Converter()
    ckpt = root / "runs" / args.size / "latest.pt"
    if codec is not None and ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        config = TriStreamConfig(**state["config"])
        model = TriStreamForSinging(config).to(device).eval()
        model.load_state_dict(state["model"])
        speaker_encoder = (
            SpeakerEncoder(frame_dim=codec.config.dim, embedding_dim=config.speaker_dim).to(device).eval()
        )
        speaker_encoder.load_state_dict(state["speaker"])
        from helix_lm.music.failsafe import Gates

        converter = Converter(
            model, codec, speaker_encoder, refiner=load_refiner(root, device),
            gates=Gates(steps=config.flow_steps, frame_rate=config.frame_rate),
        )  # fmt: skip
    song = read_wave(args.song, args.sample_rate)
    reference = read_wave(args.reference, args.sample_rate)
    mix, report = swap_voice(song, reference, args.sample_rate, converter=converter)
    out = Path(args.out or (root / "swap.wav"))
    write_wave(out, mix, args.sample_rate)
    log(f"written {out}\n{report.summary()}")


# ------------------------------------------------------------------------------------------- main --
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TriStream-2 on one 96 GB GPU, stage by stage")
    parser.add_argument("--root", default="/workspace/tristream")
    parser.add_argument("--stage", default="all", help="all or one of: " + ", ".join(STAGES))
    parser.add_argument("--size", default="ts-765m", choices=["ts-tiny", "ts-300m", "ts-765m"])
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--hf-repo", default=DEFAULT_REPO, help="the Hub repo that mirrors the run (private)")
    parser.add_argument("--public", action="store_true", help="make the Hub repo public (private by default)")
    parser.add_argument("--push-every-hours", type=float, default=2.0, help="how often checkpoints go to the Hub")
    parser.add_argument("--no-pull", action="store_true", help="do not resume from the Hub's checkpoints")
    parser.add_argument("--no-squash", action="store_true", help="keep every pushed checkpoint in the Hub history")
    parser.add_argument("--retries", type=int, default=1000, help="all: restart a crashed stage this many times")
    parser.add_argument("--retry-wait", type=float, default=120.0, help="seconds between restarts")
    parser.add_argument("--prepare-workers", type=int, default=0, help="prepare: file threads (0 = every core)")
    parser.add_argument(
        "--loader-workers", type=int, default=0, help="training: CPU workers that build batches (0 = all but two)"
    )
    parser.add_argument(
        "--loader-mode", default="auto", choices=["auto", "process", "thread", "thread-forced"],
        help="training: batch workers as processes (fastest), threads, or auto (processes on a GPU machine)",
    )  # fmt: skip
    parser.add_argument(
        "--max-gb", type=float, default=50.0, help="data: skip a source bigger than this (Slakh2100 is 104 GB)"
    )
    parser.add_argument(
        "--emilia-hours", type=float, default=300.0, help="data: hours of Emilia-YODAS speech (CC BY) to fetch"
    )
    parser.add_argument(
        "--data-attempts", type=int, default=5, help="data: runs that insist on every source before going on"
    )
    parser.add_argument("--speech-share-start", type=float, default=0.5, help="train: speech share of a batch, start")
    parser.add_argument("--speech-share-end", type=float, default=0.1, help="train: speech share of a batch, end")
    parser.add_argument("--speech-loss-weight", type=float, default=0.5, help="train: loss weight of a spoken clip")
    parser.add_argument("--augment-prob", type=float, default=0.5, help="rooms, EQ, noise, holes on a clip's input")
    parser.add_argument("--high-register-share", type=float, default=0.25, help="codec: share of high-F0 clips")
    parser.add_argument(
        "--include-research", action="store_true", help="data: also fetch non-commercial sets for evaluation"
    )
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument(
        "--uploads-are-mixtures", action="store_true", help="prepare: the uploads are full songs (separate them)"
    )
    parser.add_argument("--upload-accent", default="unknown")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--codec-hours", type=float, default=240.0)
    parser.add_argument("--codec-rate", type=int, default=25, choices=[25, 50, 75], help="codec frames a second")
    parser.add_argument("--refiner-hours", type=float, default=48.0)
    parser.add_argument("--air-hours", type=float, default=0.0, help="air band from 48 kHz uploads; 0 = skip")
    parser.add_argument("--train-hours", type=float, default=200.0)
    parser.add_argument("--clip-seconds", type=float, default=12.0)
    parser.add_argument("--batch", type=int, default=0, help="clips per step; 0 = auto (largest that fits)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-clips", type=int, default=50)
    parser.add_argument("--song", default="")
    parser.add_argument("--reference", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "eta":
        stage_eta(args)
        return 0
    if args.stage != "all" and args.stage not in STAGES:
        parser.error(f"unknown stage {args.stage}")
    if not take_lock(root):
        return 3
    gpu_max()
    hub_pull(args)
    if args.stage != "all":
        log(f"== stage {args.stage}")
        globals()[f"stage_{args.stage}"](args)
        return 0
    attempt = 0
    while True:
        try:
            for stage in STAGES[:-1]:
                if complete(root, stage, args) and stage != "push":
                    log(f"{stage}: done earlier, skipped (delete markers/{stage} to redo)")
                    continue
                log(f"== stage {stage}")
                globals()[f"stage_{stage}"](args)
                if (root / "stop").exists():
                    log("stop file found; stopping")
                    hub_push(args, force=True, reason="stop file")
                    return 0
            log("all stages done")
            return 0
        except KeyboardInterrupt:
            log("interrupted: saving the state to the Hub")
            hub_push(args, force=True, reason="interrupted")
            return 130
        except Exception as error:
            attempt += 1
            log(f"!! stage failed ({type(error).__name__}: {str(error)[:120]}); attempt {attempt} of {args.retries}")
            traceback.print_exc()
            hub_push(args, force=True, reason=f"after failure {attempt}")
            if attempt >= args.retries:
                raise
            time.sleep(args.retry_wait)
            hub_pull(args)


if __name__ == "__main__":
    raise SystemExit(main())
