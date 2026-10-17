# Copyright 2026 Nathan. Apache-2.0.
"""
A training engine for one machine with several GPUs and a hard time budget -- 8x B300, three hours.

What it does that a plain loop does not:

* **Data-parallel across every GPU.** Launched with ``torchrun --nproc_per_node 8``, each process drives
  one GPU, gradients are averaged over NVLink by DDP, and every rank reads a different shard of the data.
  The same code runs single-process on one GPU or on a CPU (that is how it is tested).
* **A schedule set by the clock, not the step count.** Nobody knows in advance how many steps three
  hours buys at full B300 throughput. The learning rate warms up over a fixed number of steps, then
  follows a cosine to its floor *as a function of elapsed time*, reaching the floor exactly when the
  budget runs out -- and training stops there with time left for the final evaluation, save and upload.
  Resumed runs keep the clock they had.
* **Every checkpoint reaches the Hub.** Checkpoints are written atomically on rank 0 every
  ``checkpoint_minutes``, and each one is queued for upload -- never skipped because the previous upload
  is still going; uploads run in order on their own thread while training continues. The full checkpoint
  (weights, optimizer, clock) replaces ``checkpoints/<run>/latest.pt`` so a dead machine resumes, and the
  weights alone are kept as ``checkpoints/<run>/history/step_N.safetensors`` so every one stays available.
* **bf16 autocast, TF32 matmuls, fused AdamW, optional torch.compile** -- the settings that matter on
  Blackwell, all on by default except compile.
* **All ranks stop together.** Stop and checkpoint decisions are made on rank 0 and broadcast, so one
  slow rank can never leave the others hanging at a collective.
"""

from __future__ import annotations

import json
import math
import os
import queue
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist


@dataclass
class EngineSettings:
    """
    Args:
        run: the run's name; also its folder under ``output`` and its path on the Hub.
        hours: wall-clock budget for training, including the reserve for the end.
        reserve_minutes: kept back from ``hours`` for the final evaluation, save and upload.
        learning_rate / min_lr_ratio / warmup_steps / weight_decay: the optimizer and its schedule.
        batch_size: per GPU.
        precision: ``"bf16"`` (default), ``"fp32"``.
        compile: ``torch.compile`` the trained module.
        checkpoint_minutes / eval_minutes: how often to save (and push) and to evaluate.
        hub_repo: the Hub repository every checkpoint is pushed to; None keeps them local.
        workers: DataLoader workers per GPU.
        max_steps: an optional hard cap, for smoke tests.
    """

    run: str
    output: str
    hours: float = 3.0
    reserve_minutes: float = 10.0
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.05
    warmup_steps: int = 500
    weight_decay: float = 0.05
    batch_size: int = 32
    precision: str = "bf16"
    compile: bool = False
    grad_clip: float = 1.0
    checkpoint_minutes: float = 20.0
    eval_minutes: float = 30.0
    log_every: int = 50
    hub_repo: str | None = None
    workers: int = 12
    seed: int = 0
    max_steps: int | None = None
    find_unused_parameters: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def setup_distributed() -> tuple[int, int, int, torch.device]:
    """``(rank, local_rank, world_size, device)``, initialising NCCL when launched by torchrun."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        # Blackwell runs fp32 matmuls through TF32 tensor cores at a fraction of the cost; the loss of
        # mantissa is far below anything these losses resolve.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return rank, local, world, device


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def broadcast_flag(value: bool, device: torch.device) -> bool:
    """Rank 0's decision, on every rank."""
    if not dist.is_initialized():
        return value
    flag = torch.tensor([1 if value else 0], device=device)
    dist.broadcast(flag, 0)
    return bool(flag.item())


def all_mean(values: dict, device: torch.device) -> dict:
    """The mean of each scalar over every rank -- for logging and evaluation."""
    if not dist.is_initialized() or not values:
        return values
    keys = sorted(values)
    tensor = torch.tensor([float(values[k]) for k in keys], device=device, dtype=torch.float64)
    dist.all_reduce(tensor)
    tensor /= dist.get_world_size()
    return dict(zip(keys, tensor.tolist()))


def lr_factor(step: int, elapsed: float, budget: float, warmup: int, floor: float) -> float:
    """Linear warm-up over ``warmup`` steps, then a cosine in *time* from 1 to ``floor`` at ``budget``."""
    warm = min(1.0, (step + 1) / max(warmup, 1))
    progress = min(max(elapsed / max(budget, 1e-9), 0.0), 1.0)
    return warm * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))


class CheckpointPusher:
    """
    Uploads checkpoints to the Hub in order, on one background thread, never dropping one.

    ``submit`` copies the file first, so the training loop can overwrite ``latest.pt`` straight away.
    """

    def __init__(self, repo_id: str | None, token: str | None = None) -> None:
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN") or None
        self.jobs: queue.Queue = queue.Queue()
        self.pushed: list[str] = []
        self.failed: list[str] = []
        self.thread = None
        if repo_id:
            self.thread = threading.Thread(target=self._work, name="checkpoint-pusher", daemon=True)
            self.thread.start()

    def submit(self, uploads: list[tuple[Path, str]], message: str) -> None:
        if not self.repo_id:
            return
        staged = []
        for local, remote in uploads:
            copy = local.with_name(f".push-{int(time.time() * 1000)}-{local.name}")
            shutil.copy2(local, copy)
            staged.append((copy, remote))
        self.jobs.put((staged, message))

    def _work(self) -> None:
        from huggingface_hub import HfApi

        try:
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()
        except ImportError:
            pass
        api = HfApi(token=self.token)
        created = False
        while True:
            job = self.jobs.get()
            if job is None:
                return
            staged, message = job
            began = time.time()
            for attempt in range(4):
                try:
                    if not created:
                        api.create_repo(self.repo_id, repo_type="model", exist_ok=True)
                        created = True
                    for local, remote in staged:
                        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=self.repo_id,
                                        repo_type="model", commit_message=message)
                    size = sum(local.stat().st_size for local, _ in staged)
                    print(f"[engine] pushed {message} to {self.repo_id} ({size / 1e9:.2f} GB in "
                          f"{time.time() - began:.0f} s)", flush=True)
                    self.pushed.append(message)
                    break
                except Exception as error:                     # a failed upload never stops training
                    if attempt == 3:
                        print(f"[engine] !! push of {message} failed after 4 tries ({type(error).__name__}: "
                              f"{error}); the checkpoint is safe on local disk", flush=True)
                        self.failed.append(message)
                    else:
                        time.sleep(15 * (attempt + 1))
            for local, _ in staged:
                local.unlink(missing_ok=True)
            self.jobs.task_done()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until every queued upload is done (or ``timeout``). Returns whether the queue drained."""
        if self.thread is None:
            return True
        deadline = None if timeout is None else time.time() + timeout
        while self.jobs.unfinished_tasks:
            if deadline is not None and time.time() > deadline:
                return False
            time.sleep(1.0)
        return True


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    module = getattr(module, "_orig_mod", module)                 # torch.compile
    return getattr(module, "module", module)                      # DDP


class Engine:
    """
    Owns the device, the parallel wrapper, the optimizer, the clock and the checkpoints of one run.

    Args:
        settings: :class:`EngineSettings`.
        owner: the whole model whose state is saved (``NatureV1``).
        trained: the module(s) being trained -- a submodule of ``owner`` (the storm network, the retrieval)
            or ``owner`` itself. Wrapped for data parallelism; its forward is what ``engine.model`` calls.
        parameters: optimizer parameter groups; default every trainable parameter of ``trained``.
        config: the owner's configuration dict, saved with every checkpoint.
    """

    def __init__(self, settings: EngineSettings, owner: torch.nn.Module, trained: torch.nn.Module,
                 parameters=None, config: dict | None = None) -> None:
        self.settings = settings
        self.rank, self.local_rank, self.world, self.device = setup_distributed()
        torch.manual_seed(settings.seed + self.rank)
        self.owner = owner.to(self.device)
        self.trained = trained
        self.config = config or {}
        groups = parameters if parameters is not None else [p for p in trained.parameters() if p.requires_grad]
        fused = {"fused": True} if self.device.type == "cuda" else {}
        self.optimizer = torch.optim.AdamW(groups, lr=settings.learning_rate, weight_decay=settings.weight_decay,
                                           betas=(0.9, 0.95), **fused)
        for group in self.optimizer.param_groups:
            group.setdefault("base_lr", group["lr"])
        model = trained
        if self.world > 1:
            from torch.nn.parallel import DistributedDataParallel

            model = DistributedDataParallel(trained, device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                                            find_unused_parameters=settings.find_unused_parameters,
                                            gradient_as_bucket_view=True)
        if settings.compile and self.device.type == "cuda":
            model = torch.compile(model)
        self.model = model
        self.directory = Path(settings.output) / settings.run
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pusher = CheckpointPusher(settings.hub_repo) if is_main() else CheckpointPusher(None)
        self.step = 0
        self.epoch = 0
        self.elapsed_before = 0.0
        self.started = time.time()
        self.last_checkpoint = time.time()
        self.last_eval = time.time()
        self.best = math.inf
        self.history: list[dict] = []
        self._running: dict[str, float] = {}
        self._count = 0

    # ---------------------------------------------------------------------------------- the clock --

    @property
    def budget_seconds(self) -> float:
        return self.settings.hours * 3600.0 - self.settings.reserve_minutes * 60.0

    @property
    def elapsed(self) -> float:
        return self.elapsed_before + (time.time() - self.started)

    def autocast(self):
        import contextlib

        if self.settings.precision == "bf16":
            return torch.autocast(self.device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def poll(self) -> tuple[bool, bool, bool]:
        """
        ``(stop, checkpoint, evaluate)`` -- rank 0's clock, decided once and shared in one broadcast, so
        every rank takes the same branch at the same step.
        """
        now = time.time()
        flags = [self.elapsed >= self.budget_seconds
                 or (self.settings.max_steps is not None and self.step >= self.settings.max_steps),
                 now - self.last_checkpoint >= self.settings.checkpoint_minutes * 60.0,
                 now - self.last_eval >= self.settings.eval_minutes * 60.0]
        if dist.is_initialized():
            tensor = torch.tensor([float(f) for f in flags], device=self.device)
            dist.broadcast(tensor, 0)
            flags = [bool(v) for v in tensor.tolist()]
        return flags[0], flags[1], flags[2]

    def should_stop(self) -> bool:
        local = self.elapsed >= self.budget_seconds or (
            self.settings.max_steps is not None and self.step >= self.settings.max_steps)
        return broadcast_flag(local, self.device)

    # ------------------------------------------------------------------------------- one step --

    def backward_step(self, loss: torch.Tensor, parts: dict | None = None) -> float:
        """Backward, clip, update, schedule. Returns the learning rate used."""
        factor = lr_factor(self.step, self.elapsed, self.budget_seconds, self.settings.warmup_steps,
                           self.settings.min_lr_ratio)
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.settings.grad_clip:
            torch.nn.utils.clip_grad_norm_([p for g in self.optimizer.param_groups for p in g["params"]],
                                           self.settings.grad_clip)
        self.optimizer.step()
        self.step += 1
        for key, value in (parts or {"loss": float(loss.detach())}).items():
            self._running[key] = self._running.get(key, 0.0) + float(value)
        self._count += 1
        if self.step % self.settings.log_every == 0:
            self.log_running(factor)
        return factor * self.settings.learning_rate

    def log_running(self, factor: float) -> None:
        means = all_mean({k: v / max(self._count, 1) for k, v in self._running.items()}, self.device)
        self._running, self._count = {}, 0
        if is_main():
            left = max(self.budget_seconds - self.elapsed, 0.0) / 60
            text = "  ".join(f"{k} {v:.4f}" for k, v in sorted(means.items()))
            print(f"[{self.settings.run}] step {self.step:,}  {text}  lr x{factor:.3f}  "
                  f"{left:.0f} min left", flush=True)

    # ---------------------------------------------------------------------------- checkpoints --

    def due(self, minutes: float, since: float) -> bool:
        return broadcast_flag(time.time() - since >= minutes * 60.0, self.device)

    def checkpoint(self, metrics: dict | None = None, final: bool = False) -> Path | None:
        """Save on rank 0 (atomically) and queue the upload; every rank waits for the save."""
        path = None
        if is_main():
            from .checkpoint import TrainingState, _rng_state
            from .safeio import atomic_torch_save, atomic_write_text

            owner = _unwrap(self.owner)
            state = TrainingState(step=self.step, epoch=self.epoch, best_loss=self.best,
                                  wall_seconds=self.elapsed, history=self.history[-200:])
            payload = {"model": owner.state_dict(), "optimizer": self.optimizer.state_dict(), "scheduler": None,
                       "scaler": None, "state": state.to_dict(), "config": self.config, "rng": _rng_state(),
                       "saved_at": time.time(), "format": 2, "tag": self.settings.run,
                       "engine": {"settings": self.settings.to_dict(), "world_size": self.world,
                                  "metrics": metrics or {}}}
            path = self.directory / "latest.pt"
            atomic_torch_save(payload, path)
            atomic_write_text(self.directory / "state.json", json.dumps(state.to_dict(), indent=2, default=str))
            weights = self.directory / "history" / f"step_{self.step:08d}.safetensors"
            weights.parent.mkdir(exist_ok=True)
            _save_weights(owner, weights, {"run": self.settings.run, "step": str(self.step),
                                           "metrics": json.dumps(metrics or {})})
            remote = f"checkpoints/{self.settings.run}"
            self.pusher.submit([(path, f"{remote}/latest.pt"), (self.directory / "state.json", f"{remote}/state.json"),
                                (weights, f"{remote}/history/{weights.name}")],
                               f"{self.settings.run} step {self.step}" + (" (final)" if final else ""))
            print(f"[{self.settings.run}] checkpoint at step {self.step:,} ({self.elapsed / 60:.0f} min)"
                  + (" -- queued for the Hub" if self.settings.hub_repo else ""), flush=True)
        barrier()
        self.last_checkpoint = time.time()
        return path

    def resume(self) -> bool:
        """Continue from ``latest.pt`` here, or from the Hub copy on a fresh machine. Returns whether it did."""
        path = self.directory / "latest.pt"
        if is_main() and not path.exists() and self.settings.hub_repo:
            try:
                from huggingface_hub import hf_hub_download

                found = hf_hub_download(self.settings.hub_repo, f"checkpoints/{self.settings.run}/latest.pt")
                shutil.copy2(found, path)
                print(f"[{self.settings.run}] pulled the last checkpoint from {self.settings.hub_repo}", flush=True)
            except Exception:
                pass
        barrier()
        if not path.exists():
            return False
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _unwrap(self.owner).load_state_dict(payload["model"])
        try:
            self.optimizer.load_state_dict(payload["optimizer"])
        except (ValueError, KeyError, RuntimeError):
            if is_main():
                print(f"[{self.settings.run}] optimizer state did not match; moments restart", flush=True)
        state = payload.get("state", {})
        self.step = int(state.get("step", 0))
        self.epoch = int(state.get("epoch", 0))
        self.best = float(state.get("best_loss", math.inf))
        self.elapsed_before = float(state.get("wall_seconds", 0.0))
        self.history = list(state.get("history") or [])
        self.started = time.time()
        if is_main():
            print(f"[{self.settings.run}] resumed at step {self.step:,}, {self.elapsed_before / 60:.0f} of "
                  f"{self.settings.hours * 60:.0f} min used", flush=True)
        return True

    def finish(self, metrics: dict | None = None, upload_timeout: float = 1800.0) -> None:
        """Final checkpoint, then wait for every upload to land."""
        self.checkpoint(metrics, final=True)
        if is_main() and self.settings.hub_repo:
            print(f"[{self.settings.run}] waiting for uploads to finish ...", flush=True)
            drained = self.pusher.wait(upload_timeout)
            print(f"[{self.settings.run}] uploads {'complete' if drained else 'still running (left in background)'}: "
                  f"{len(self.pusher.pushed)} pushed, {len(self.pusher.failed)} failed", flush=True)
        barrier()

    # ---------------------------------------------------------------------------------- data --

    def loader(self, dataset, shuffle: bool = True, batch_size: int | None = None, drop_last: bool = True,
               collate_fn=None) -> torch.utils.data.DataLoader:
        """A DataLoader that gives each rank its own shard; call :meth:`set_epoch` before each pass."""
        sampler = None
        if self.world > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle, seed=self.settings.seed,
                                                                      drop_last=drop_last)
        workers = self.settings.workers
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size or self.settings.batch_size, sampler=sampler,
            shuffle=shuffle and sampler is None, num_workers=workers, pin_memory=self.device.type == "cuda",
            drop_last=drop_last, persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None,
            collate_fn=collate_fn)

    def set_epoch(self, loader, epoch: int) -> None:
        self.epoch = epoch
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        dataset = getattr(loader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

    def to_device(self, batch: dict) -> dict:
        return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _save_weights(module: torch.nn.Module, path: Path, metadata: dict) -> None:
    """Weights only, as safetensors (bf16 for floating tensors, to halve the upload)."""
    from .safeio import atomic_write_bytes

    tensors = {}
    for name, value in module.state_dict().items():
        value = value.detach().cpu().contiguous()
        if value.is_floating_point() and value.dtype == torch.float32:
            value = value.to(torch.bfloat16)
        tensors[name] = value
    try:
        from safetensors.torch import save

        atomic_write_bytes(path, save(tensors, metadata={k: str(v) for k, v in metadata.items()}))
    except ImportError:
        torch.save({"weights": tensors, "metadata": metadata}, path.with_suffix(".pt"))


__all__ = ["CheckpointPusher", "Engine", "EngineSettings", "all_mean", "barrier", "broadcast_flag", "is_main",
           "lr_factor", "setup_distributed"]
