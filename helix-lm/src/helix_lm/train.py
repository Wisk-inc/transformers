# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Training HELIX on one machine with any number of GPUs.

    torchrun --nproc_per_node=8 -m helix_lm.train --size 23b --data /data/helix --out /data/helix/runs/helix-23b \\
        --hours 700 --hub-repo CorxLabs/Helix-23B

What it does, in order: builds the model on the meta device and shards it with FSDP2 (bf16 parameters and
gradients, fp32 master weights and reductions, every block its own shard group); opens the prepared data
as a weighted mixture; resumes from the newest checkpoint if there is one; trains with fused AdamW (or Muon) on a
warmup-stable-decay schedule whose length comes from ``--total-tokens`` or, with ``--hours``, from the
measured throughput so the decay ends exactly when the money does; saves a sharded checkpoint every few
minutes and a bf16 safetensors export it pushes to the Hub; reports loss, learning rate, tokens a second,
model-FLOP utilization and memory every few steps; and stops cleanly at the deadline, at a stop file, or
at the planned end, always after a save. A run that died is continued by running the same command.

The same program fine-tunes: point ``--data`` at shards that carry loss masks (``helix sft prepare``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist

from . import arch
from .arch import ALL_SIZES
from .data.catalog import PRETRAIN_SOURCES
from .data.loader import MixtureLoader, load_mixture, mixture_from_catalog, open_mixture
from .hub import HubPusher
from .model import kernels_available
from .optim import build_optimizer, mup_plan, unwrap_optimizers
from .optim.mup import apply_init
from .tokenizer import HelixTokenizer

# Dense BF16 tensor-core peak per GPU, for the MFU column. Vendor numbers; the B300 figure is the one most
# sources agree on (2.25 PFLOPS).
PEAK_FLOPS = {
    "B300": 2.25e15, "GB300": 2.25e15, "B200": 2.25e15, "GB200": 2.25e15,
    "H200": 9.89e14, "H100": 9.89e14, "H800": 9.89e14, "A100": 3.12e14,
    "L40S": 3.62e14, "RTX PRO 6000": 5.0e14, "5090": 2.1e14, "4090": 1.65e14,
}  # fmt: skip


@dataclass
class TrainArgs:
    size: str = "1b"
    data: str = ""
    out: str = ""
    tokenizer: str = ""
    mixture: str = ""
    seq_len: int = 8192
    micro_batch: int = 4
    global_batch_tokens: int = 2_097_152
    total_tokens: float = 0
    steps: int = 0
    hours: float = 0
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 0
    decay_fraction: float = 0.2
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    z_loss: float = 1e-4
    seed: int = 0
    dtype: str = "auto"
    parallel: str = "auto"
    compile: bool = False
    activation_checkpointing: bool = True
    attention_backend: str = "auto"
    recurrent_backend: str = "auto"
    loss_chunk_tokens: int = 4096
    log_every: int = 10
    eval_every: int = 500
    eval_batches: int = 8
    save_every_min: float = 30.0
    save_every_steps: int = 0
    push_every_min: float = 60.0
    keep_checkpoints: int = 2
    hub_repo: str = ""
    hub_path: str = ""
    milestone_tokens: float = 0
    hub_private: bool = True
    run_name: str = ""
    stop_file: str = ""
    measure_steps: int = 30
    holdout_tokens: int = 2_000_000
    init_from: str = ""
    optimizer: str = "adamw"
    muon_momentum: float = 0.95
    mup: bool = False
    mup_base_width: int = 256
    reweight_strength: float = 0.0
    overrides: list[str] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> TrainArgs:
    parser = argparse.ArgumentParser(prog="helix train", description="Train HELIX (pretraining or SFT).")
    defaults = TrainArgs()
    parser.add_argument("--size", default=defaults.size, choices=list(ALL_SIZES), help="named size (helix size)")
    parser.add_argument("--data", required=True, help="prepared data directory (tokens/<source>/*.bin)")
    parser.add_argument("--out", required=True, help="run directory: checkpoints, export, logs")
    parser.add_argument("--tokenizer", default="", help="tokenizer directory (default: <data>/tokenizer)")
    parser.add_argument(
        "--mixture", default="", help="mixture.json with weights and max_epochs (default: the catalog)"
    )
    parser.add_argument("--seq-len", type=int, default=defaults.seq_len)
    parser.add_argument(
        "--micro-batch", type=int, default=defaults.micro_batch, help="sequences per GPU per micro-step"
    )
    parser.add_argument("--global-batch-tokens", type=int, default=defaults.global_batch_tokens)
    parser.add_argument("--total-tokens", type=float, default=0, help="tokens to train on (e.g. 460e9)")
    parser.add_argument("--steps", type=int, default=0, help="or: optimizer steps to run")
    parser.add_argument("--hours", type=float, default=0, help="or: wall-clock budget; the schedule is fitted to it")
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--min-lr-ratio", type=float, default=defaults.min_lr_ratio)
    parser.add_argument("--warmup-steps", type=int, default=0, help="default: 2%% of the run, at least 100")
    parser.add_argument("--decay-fraction", type=float, default=defaults.decay_fraction)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--beta1", type=float, default=defaults.beta1)
    parser.add_argument("--beta2", type=float, default=defaults.beta2)
    parser.add_argument("--grad-clip", type=float, default=defaults.grad_clip)
    parser.add_argument("--z-loss", type=float, default=defaults.z_loss)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    parser.add_argument("--parallel", default="auto", choices=["auto", "fsdp", "ddp", "none"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-activation-checkpointing", dest="activation_checkpointing", action="store_false")
    parser.add_argument("--attention-backend", default="auto", choices=["auto", "flex", "sdpa"])
    parser.add_argument("--recurrent-backend", default="auto", choices=["auto", "fla", "torch"])
    parser.add_argument("--loss-chunk-tokens", type=int, default=defaults.loss_chunk_tokens)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--eval-batches", type=int, default=defaults.eval_batches)
    parser.add_argument("--save-every-min", type=float, default=defaults.save_every_min)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--push-every-min", type=float, default=defaults.push_every_min)
    parser.add_argument("--keep-checkpoints", type=int, default=defaults.keep_checkpoints)
    parser.add_argument("--hub-repo", default="", help="push the export here, e.g. CorxLabs/Helix-23B")
    parser.add_argument("--hub-path", default="", help="folder in the repo (default: checkpoints/<run name>)")
    parser.add_argument(
        "--milestone-tokens", type=float, default=0, help="also keep a copy of the export in the repo every N tokens"
    )
    parser.add_argument("--hub-public", dest="hub_private", action="store_false", help="create the Hub repo public")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--stop-file", default="", help="touch this file to save and stop (also $HELIX_STOP_FILE)")
    parser.add_argument("--measure-steps", type=int, default=defaults.measure_steps)
    parser.add_argument("--holdout-tokens", type=int, default=defaults.holdout_tokens)
    parser.add_argument("--init-from", default="", help="start from this saved model (SFT, continued pretraining)")
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "muon"],
        default=defaults.optimizer,
        help="muon: orthogonalized momentum on the block matrices, AdamW on the rest",
    )
    parser.add_argument("--muon-momentum", type=float, default=defaults.muon_momentum)
    parser.add_argument(
        "--mup", action="store_true", help="muP init and per-width learning rates (transfer the pilot's lr)"
    )
    parser.add_argument("--mup-base-width", type=int, default=defaults.mup_base_width)
    parser.add_argument(
        "--reweight-strength",
        type=float,
        default=defaults.reweight_strength,
        help="online data-mixture reweighting from per-source held-out progress (0 = off)",
    )
    parser.add_argument("--override", dest="overrides", action="append", default=[], help="config field=value")
    namespace = parser.parse_args(argv)
    args = TrainArgs(**vars(namespace))
    if not (args.total_tokens or args.steps or args.hours):
        parser.error("give one of --total-tokens, --steps or --hours")
    args.tokenizer = args.tokenizer or str(Path(args.data) / "tokenizer")
    args.run_name = args.run_name or Path(args.out).name
    args.hub_path = args.hub_path or f"checkpoints/{args.run_name}"
    args.stop_file = args.stop_file or os.environ.get("HELIX_STOP_FILE", "")
    return args


def _parse_override(text: str) -> tuple[str, object]:
    key, _, value = text.partition("=")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


class Trainer:
    def __init__(self, args: TrainArgs) -> None:
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self._setup_distributed()
        torch.manual_seed(args.seed)
        self.started = time.time()
        self.deadline = self.started + args.hours * 3600 if args.hours else float("inf")
        self.log_path = self.out / "log.jsonl" if self.rank == 0 else None  # appended to per record, never held open

        self.tokenizer = HelixTokenizer.from_pretrained(args.tokenizer)
        self.config = self._build_config()
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.loader = self._open_data()
        self.pusher = HubPusher(args.hub_repo or None, args.hub_path, log=self.log, private=args.hub_private)

        self.step = 0
        self.tokens_seen = 0
        self.train_seconds = 0.0
        self.total_steps = self._planned_steps()
        self.step_times: list[float] = []
        self.last_save = time.time()
        self.last_push = time.time()
        self.last_save_duration = 0.0
        self.last_saved_step = -1
        self.last_milestone = 0
        # Seconds kept back from a deadline: to stop (a save must fit) and when fitting the schedule.
        self.stop_margin, self.reserve = 240.0, 480.0
        self._resume()
        if self.rank == 0:
            self.log(self.describe())

    # ------------------------------------------------------------------------------------------ setup --
    def _setup_distributed(self) -> None:
        self.distributed = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
        if self.distributed:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend)
            self.rank, self.world = dist.get_rank(), dist.get_world_size()
        else:
            self.rank, self.world = 0, 1
        if torch.cuda.is_available():
            local = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local)
            self.device = torch.device("cuda", local)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            self.device = torch.device("cpu")
        self.gpu_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
        self.peak_flops = next((v for k, v in PEAK_FLOPS.items() if k.lower() in self.gpu_name.lower()), None)
        dtype = self.args.dtype if self.args.dtype != "auto" else ("bf16" if self.device.type == "cuda" else "fp32")
        self.param_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        parallel = self.args.parallel
        if parallel == "auto":
            parallel = "fsdp" if self.distributed else "none"
        if parallel in ("fsdp", "ddp") and not self.distributed:
            parallel = "none"
        self.parallel = parallel

    def log(self, text: str, record: dict | None = None) -> None:
        if self.rank != 0:
            return
        print(text, flush=True)
        if record is not None and self.log_path is not None:
            with open(self.log_path, "a") as handle:
                handle.write(json.dumps({"time": time.time(), **record}) + "\n")

    def _build_config(self):
        fields = {}
        if arch.family_of(self.args.size) in ("helix", "anansi"):
            fields = {
                "activation_checkpointing": self.args.activation_checkpointing,
                "attention_backend": self.args.attention_backend,
                "recurrent_backend": self.args.recurrent_backend,
                "loss_chunk_tokens": self.args.loss_chunk_tokens,
                "z_loss_weight": self.args.z_loss,
            }
        fields.update(_parse_override(o) for o in self.args.overrides)  # --override wins over the flags
        config = arch.make_config(self.args.size, **fields)
        if self.tokenizer.vocab_size > config.vocab_size:
            raise ValueError(f"tokenizer has {self.tokenizer.vocab_size} tokens, the model {config.vocab_size}")
        config.eos_token_id, config.bos_token_id = self.tokenizer.eos_token_id, self.tokenizer.bos_token_id
        if arch.is_helix(config):  # the transformer keeps padding_idx unset: nothing to zero through a DTensor slice
            config.pad_token_id = self.tokenizer.pad_token_id
        if self.args.mup and hasattr(config, "output_multiplier"):
            config.output_multiplier = self.args.mup_base_width / config.hidden_size
        return config

    def _build_model(self):
        config = self.config
        if self.parallel == "fsdp":
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

            mesh = init_device_mesh(self.device.type, (self.world,))
            policy = MixedPrecisionPolicy(param_dtype=self.param_dtype, reduce_dtype=torch.float32)
            with torch.device("meta"):
                model = arch.construct(config)
            for layer in model.model.layers:
                fully_shard(layer, mesh=mesh, mp_policy=policy)
            fully_shard(model, mesh=mesh, mp_policy=policy)
            model.to_empty(device=self.device)
            arch.initialize(model, config, self.args.seed)
            if config.tie_word_embeddings:
                raise ValueError("tied embeddings are not supported under FSDP; the ladder sizes leave them untied")
        else:
            model = arch.construct(config)
            arch.initialize(model, config, self.args.seed)
            model = model.to(self.device)
            if self.parallel == "ddp":
                from torch.nn.parallel import DistributedDataParallel

                model = DistributedDataParallel(
                    model, device_ids=[self.device.index] if self.device.type == "cuda" else None
                )
        self.mup = None
        if self.args.mup:
            self.mup = self._apply_mup(model, config)
        arch.set_activation_checkpointing(model, config, self.args.activation_checkpointing)
        if self.args.init_from:
            arch.load_initial_weights(model, self.args.init_from, self.parallel)
            self.log(f"[train] weights loaded from {self.args.init_from}")
        if self.args.compile:
            for layer in arch.layers_of(model):
                layer.compile()
        return model

    def _apply_mup(self, model, config):
        """muP: width-scaled init for the matrices (the residual writers keep their depth scaling) and the lr plan."""
        from .optim.muon import partition_parameters

        muon_matrices = {id(p) for p in partition_parameters(model)[0]} if self.args.optimizer == "muon" else set()
        plan = mup_plan(model, config.hidden_size, self.args.mup_base_width, config.initializer_range, muon_matrices)
        writers = ("o_proj.weight", "out_proj.weight", "down_proj.weight", "w_down")
        inner = model.module if hasattr(model, "module") else model
        skip = {id(p) for n, p in inner.named_parameters() if n.endswith(writers)}
        apply_init(model, plan, skip=skip)
        self.log(f"[train] {plan.describe()}")
        return plan

    def _inner(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _build_optimizer(self):
        plan = getattr(self, "mup", None)
        return build_optimizer(
            self.model,
            self.args.optimizer,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(self.args.beta1, self.args.beta2),
            momentum=self.args.muon_momentum,
            fused=self.device.type == "cuda",
            lr_multipliers=plan.lr_multiplier if plan else None,
            wd_multipliers=plan.wd_multiplier if plan else None,
        )

    def _open_data(self) -> MixtureLoader:
        if self.args.mixture:
            weights, caps = load_mixture(self.args.mixture)
        else:
            weights, caps = mixture_from_catalog(PRETRAIN_SOURCES)
            prepared = Path(self.args.data) / "tokens"
            if prepared.exists():  # sources outside the catalog (tests, custom data) get an equal share
                for folder in prepared.iterdir():
                    weights.setdefault(folder.name, 1.0 / max(len(list(prepared.iterdir())), 1))
        planned = self.args.total_tokens or (20 * arch.num_parameters(arch.make_config(self.args.size)))
        return open_mixture(
            self.args.data, weights, caps, self.args.seq_len, self.args.micro_batch, self.rank, self.world,
            self.args.seed, int(planned), self.args.holdout_tokens,
        )  # fmt: skip

    # --------------------------------------------------------------------------------------- schedule --
    @property
    def tokens_per_step(self) -> int:
        return self.accumulation * self.args.micro_batch * self.args.seq_len * self.world

    @property
    def accumulation(self) -> int:
        per_micro = self.args.micro_batch * self.args.seq_len * self.world
        return max(1, round(self.args.global_batch_tokens / per_micro))

    def _planned_steps(self) -> int | None:
        if self.args.steps:
            return self.args.steps
        if self.args.total_tokens:
            return math.ceil(self.args.total_tokens / self.tokens_per_step)
        return None  # from --hours, once the throughput is measured

    @property
    def warmup_steps(self) -> int:
        if self.args.warmup_steps:
            return self.args.warmup_steps
        return max(100, int(0.02 * (self.total_steps or 10_000)))

    def lr_at(self, step: int) -> float:
        """Warmup, stable, then a 1 - sqrt decay over the last ``decay_fraction`` to ``min_lr_ratio``."""
        peak, floor = self.args.lr, self.args.lr * self.args.min_lr_ratio
        warmup = min(self.warmup_steps, max(1, (self.total_steps or 10**9) // 2))
        if step < warmup:
            return peak * (step + 1) / warmup
        if self.total_steps is None:
            return peak
        decay_start = int(self.total_steps * (1 - self.args.decay_fraction))
        if step < decay_start:
            return peak
        progress = min(1.0, (step - decay_start) / max(1, self.total_steps - decay_start))
        return floor + (peak - floor) * (1 - math.sqrt(progress))

    def _in_cooldown(self) -> bool:
        if self.total_steps is None:
            return False
        return self.step >= int(self.total_steps * (1 - self.args.decay_fraction))

    def _fit_steps_to_deadline(self) -> None:
        """With ``--hours``: pick the total so that training, with its decay, ends at the deadline."""
        if not self.args.hours or len(self.step_times) < self.args.measure_steps:
            return
        if self.total_steps is not None and not self._refit_needed:
            return
        seconds_per_step = statistics.median(self.step_times[-self.args.measure_steps :])
        reserve = self.reserve + 3 * self.last_save_duration
        remaining = max(0.0, self.deadline - time.time() - reserve)
        fitted = self.step + int(0.95 * remaining / seconds_per_step)
        if self.args.total_tokens:
            fitted = min(fitted, math.ceil(self.args.total_tokens / self.tokens_per_step))
        self.total_steps = max(fitted, self.step + 1)
        self._refit_needed = False
        hours_left = (self.total_steps - self.step) * seconds_per_step / 3600
        self.log(
            f"[train] schedule fitted to the deadline: {self.total_steps:,} steps "
            f"({self.total_steps * self.tokens_per_step / 1e9:.1f}B tokens), {seconds_per_step:.2f} s/step, "
            f"about {hours_left:.1f} h of training left"
        )

    # ------------------------------------------------------------------------------------------- loop --
    def run(self) -> str:
        args = self.args
        model, optimizer, loader = self.model, self.optimizer, self.loader
        self._refit_needed = args.hours > 0  # fit (or refit after a resume) once throughput is measured
        reason = "planned"
        model.train()
        while self.total_steps is None or self.step < self.total_steps:
            if self._should_stop():
                reason = self._stop_reason
                break
            step_started = time.perf_counter()
            lr = self.lr_at(self.step)
            for group in optimizer.param_groups:
                group["lr"] = lr * group.get("lr_mult", 1.0)
            loss_sum = z_sum = mtp_sum = 0.0
            loops = 0
            for micro in range(self.accumulation):
                ids, _, mask = loader.next_batch()
                ids = ids.to(self.device, non_blocking=True)
                labels = ids.masked_fill(~mask.to(self.device), -100)
                last = micro == self.accumulation - 1
                if self.parallel == "fsdp":
                    model.set_requires_gradient_sync(last)
                context = model.no_sync() if (self.parallel == "ddp" and not last) else _nullcontext()
                with context, torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self._autocast):
                    output = model(ids, labels=labels)
                (output.loss / self.accumulation).backward()
                loss_sum += float(output.loss.detach())
                z_loss = getattr(output, "z_loss", None)
                z_sum += float(z_loss.detach()) if z_loss is not None else 0.0
                mtp = getattr(output, "mtp_loss", None)
                mtp_sum += float(mtp.detach()) if mtp is not None else 0.0
                loops = getattr(output, "loops", 0) or loops
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            self.step += 1
            self.tokens_seen += self.tokens_per_step
            elapsed = time.perf_counter() - step_started
            self.train_seconds += elapsed
            self.step_times.append(elapsed)
            del self.step_times[:-200]
            self._fit_steps_to_deadline()

            loss = self._mean_across_ranks(loss_sum / self.accumulation)
            if self.step % args.log_every == 0 or self.step == 1:
                self._log_step(
                    loss, z_sum / self.accumulation, lr, float(grad_norm), elapsed, mtp_sum / self.accumulation, loops
                )
            if args.eval_every and self.step % args.eval_every == 0:
                results = self.evaluate()
                if args.reweight_strength > 0 and results and not self._in_cooldown():
                    change = loader.reweight(results, args.reweight_strength)
                    if change:
                        self.log(
                            "[data] mixture reweighted: " + change,
                            {"event": "reweight", "step": self.step, "weights": loader.weights},
                        )
            if self._time_to_save():
                self.save_checkpoint()
                if self.pusher.enabled and time.time() - self.last_push >= args.push_every_min * 60:
                    self.export(push=True)
        if reason == "planned":
            self.log(f"[train] planned steps done: {self.step:,} steps, {self.tokens_seen / 1e9:.2f}B tokens")
        if self.last_saved_step != self.step:
            self.save_checkpoint()
        self.export(push=self.pusher.enabled)
        self.pusher.wait()
        final = self.evaluate() if args.eval_every else {}
        self.log(
            f"[train] finished: step {self.step:,}, tokens {self.tokens_seen / 1e9:.3f}B, "
            f"train time {self.train_seconds / 3600:.2f} h, reason={reason}"
            + (f", eval {json.dumps(final)}" if final else ""),
            {"event": "finished", "step": self.step, "tokens": self.tokens_seen, "reason": reason, "eval": final},
        )
        if self.distributed:
            dist.barrier()
            dist.destroy_process_group()
        return reason

    @property
    def _autocast(self) -> bool:
        # FSDP casts parameters itself; autocast is for the DDP / single-process bf16 case on a GPU.
        return self.param_dtype == torch.bfloat16 and self.parallel != "fsdp" and self.device.type == "cuda"

    def _mean_across_ranks(self, value: float) -> float:
        if not self.distributed:
            return value
        tensor = torch.tensor(value, device=self.device if self.device.type == "cuda" else "cpu")
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        return float(tensor)

    def _should_stop(self) -> bool:
        stop = 0
        if self.rank == 0:
            margin = self.stop_margin + 2 * self.last_save_duration
            if self.args.stop_file and Path(self.args.stop_file).exists():
                self._stop_reason, stop = "stop-file", 1
            elif time.time() > self.deadline - margin:
                self._stop_reason, stop = "deadline", 1
        if self.distributed:
            flag = torch.tensor(stop, device=self.device if self.device.type == "cuda" else "cpu")
            dist.broadcast(flag, src=0)
            stop = int(flag)
            if stop and self.rank != 0:
                self._stop_reason = "stop"
        return bool(stop)

    def _time_to_save(self) -> bool:
        if self.args.save_every_steps:
            return self.step % self.args.save_every_steps == 0
        due = time.time() - self.last_save >= self.args.save_every_min * 60
        if self.distributed:  # every rank must agree, and only rank 0's clock counts
            flag = torch.tensor(int(due), device=self.device if self.device.type == "cuda" else "cpu")
            dist.broadcast(flag, src=0)
            due = bool(int(flag))
        return due

    def _log_step(
        self, loss: float, z_loss: float, lr: float, grad_norm: float, elapsed: float, mtp: float = 0.0, loops: int = 0
    ) -> None:
        recent = statistics.median(self.step_times[-20:])
        tokens_per_second = self.tokens_per_step / recent
        mfu = None
        if self.peak_flops:
            mfu = (
                tokens_per_second
                * arch.flops_per_token(self.config, self.args.seq_len)
                / (self.world * self.peak_flops)
            )
        memory = torch.cuda.max_memory_allocated(self.device) / 1e9 if self.device.type == "cuda" else 0.0
        total = f"/{self.total_steps:,}" if self.total_steps else ""
        eta = ((self.total_steps - self.step) * recent / 3600) if self.total_steps else None
        self.log(
            f"[train] step {self.step:,}{total}  loss {loss:.4f}  z {z_loss:.3f}  lr {lr:.2e}  gnorm {grad_norm:.2f}  "
            f"tok/s {tokens_per_second / 1e6:.2f}M"
            + (f"  mtp {mtp:.3f}" if mtp else "")
            + (f"  loops {loops}" if loops else "")
            + (f"  mfu {mfu:.0%}" if mfu is not None else "")
            + (f"  mem {memory:.0f}G" if memory else "")
            + f"  train {self.train_seconds / 3600:.2f}h"
            + (f"  eta {eta:.1f}h" if eta is not None else ""),
            {
                "event": "step",
                "step": self.step,
                "mtp": mtp,
                "loops": loops,
                "tokens": self.tokens_seen,
                "loss": loss,
                "z_loss": z_loss,
                "lr": lr,
                "grad_norm": grad_norm,
                "tokens_per_second": tokens_per_second,
                "mfu": mfu,
                "memory_gb": memory,
                "train_seconds": self.train_seconds,
                "total_steps": self.total_steps,
            },  # fmt: skip
        )

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Held-out loss per source, averaged across ranks."""
        self.model.eval()
        results = {}
        for name in self.loader.names:
            losses = []
            for ids in self.loader.eval_batches(name, self.args.eval_batches):
                ids = ids.to(self.device)
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self._autocast):
                    losses.append(float(self.model(ids, labels=ids).loss))
            if losses:
                results[name] = round(self._mean_across_ranks(sum(losses) / len(losses)), 4)
        self.model.train()
        if results:
            self.log(
                "[eval] step " + f"{self.step:,}  " + "  ".join(f"{k} {v:.3f}" for k, v in results.items()),
                {
                    "event": "eval",
                    "step": self.step,
                    "tokens": self.tokens_seen,
                    **{f"eval/{k}": v for k, v in results.items()},
                },
            )
        return results

    # ---------------------------------------------------------------------------------- checkpoints --
    def _checkpoint_dir(self, step: int) -> Path:
        return self.out / "checkpoints" / f"step-{step:08d}"

    def save_checkpoint(self) -> None:
        started = time.time()
        directory = self._checkpoint_dir(self.step)
        if self.rank == 0:
            directory.mkdir(parents=True, exist_ok=True)
        if self.distributed:
            dist.barrier()
        loader_states = [None] * self.world
        if self.distributed:
            dist.all_gather_object(loader_states, self.loader.state_dict())
        else:
            loader_states = [self.loader.state_dict()]
        if self.parallel == "fsdp":
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import get_state_dict

            model_state, optim_state = get_state_dict(self.model, unwrap_optimizers(self.optimizer))
            dcp.save({"model": model_state, "optim": optim_state}, checkpoint_id=str(directory / "dcp"))
        elif self.rank == 0:
            torch.save(
                {"model": self._inner().state_dict(), "optim": self.optimizer.state_dict()}, directory / "state.pt"
            )
        if self.rank == 0:
            meta = {
                "step": self.step, "tokens": self.tokens_seen, "train_seconds": self.train_seconds,
                "total_steps": self.total_steps, "world": self.world, "parallel": self.parallel,
                "loader": loader_states, "args": asdict(self.args), "config": self.config.to_dict(),
                "last_milestone": self.last_milestone,
            }  # fmt: skip
            (directory / "meta.json").write_text(json.dumps(meta, default=_jsonable))
            (directory / "done").write_text(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
            (self.out / "checkpoints" / "latest").write_text(directory.name)
            for old in sorted(self.out.glob("checkpoints/step-*"))[: -self.args.keep_checkpoints]:
                if old != directory:
                    shutil.rmtree(old, ignore_errors=True)
        if self.distributed:
            dist.barrier()
        self.last_save = time.time()
        self.last_save_duration = self.last_save - started
        self.last_saved_step = self.step
        self.log(
            f"[train] checkpoint at step {self.step:,} ({self.train_seconds / 60:.0f} min) "
            f"in {self.last_save_duration:.0f}s",
            {"event": "checkpoint", "step": self.step, "tokens": self.tokens_seen, "seconds": self.last_save_duration},
        )

    def _resume(self) -> None:
        latest = self.out / "checkpoints" / "latest"
        if not latest.exists():
            return
        directory = self.out / "checkpoints" / latest.read_text().strip()
        if not (directory / "done").exists():
            return
        meta = json.loads((directory / "meta.json").read_text())
        if self.parallel == "fsdp":
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

            model_state, optim_state = get_state_dict(self.model, unwrap_optimizers(self.optimizer))
            dcp.load({"model": model_state, "optim": optim_state}, checkpoint_id=str(directory / "dcp"))
            set_state_dict(
                self.model,
                unwrap_optimizers(self.optimizer),
                model_state_dict=model_state,
                optim_state_dict=optim_state,
            )
        else:
            state = torch.load(directory / "state.pt", map_location=self.device, weights_only=False)
            self._inner().load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optim"])
        self.step, self.tokens_seen = meta["step"], meta["tokens"]
        self.train_seconds = meta.get("train_seconds", 0.0)
        self.last_milestone = meta.get("last_milestone", 0)
        if meta["total_steps"] and not self.args.hours:
            self.total_steps = self.total_steps or meta["total_steps"]
        if meta["world"] == self.world and len(meta["loader"]) == self.world:
            self.loader.load_state_dict(meta["loader"][self.rank])
        else:
            self.log(f"[train] world size changed ({meta['world']} -> {self.world}): data cursors restart")
        self.log(f"[train] resumed from {directory.name}: step {self.step:,}, {self.tokens_seen / 1e9:.3f}B tokens")

    def export(self, push: bool) -> Path:
        """A bf16 safetensors copy of the model, with the tokenizer, under ``out/export``; pushed if asked."""
        if self.parallel == "fsdp":
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            state = get_model_state_dict(self.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        else:
            state = self._inner().state_dict()
        export = self.out / "export"
        if self.rank == 0:
            arch.export(self.model, self.config, state, export, self.args.tokenizer)
            (export / "training_state.json").write_text(
                json.dumps(
                    {
                        "step": self.step,
                        "tokens": self.tokens_seen,
                        "train_hours": self.train_seconds / 3600,
                        "size": self.args.size,
                        "run": self.args.run_name,
                        "gpu": self.gpu_name,
                        "world": self.world,
                    },
                    indent=1,
                )  # fmt: skip
            )
            if push:
                message = f"{self.args.run_name}: step {self.step:,}, {self.tokens_seen / 1e9:.2f}B tokens"
                milestone = None
                if self.args.milestone_tokens and self.tokens_seen // self.args.milestone_tokens > self.last_milestone:
                    self.last_milestone = self.tokens_seen // self.args.milestone_tokens
                    milestone = f"milestones/{self.last_milestone * self.args.milestone_tokens / 1e9:.0f}B-tokens"
                self.pusher.push(export, message, also_copy_to=milestone)
                self.last_push = time.time()
        if self.distributed:
            dist.barrier()
        return export

    def describe(self) -> str:
        kernels = kernels_available()
        flex = "on" if kernels["flex_attention"] else "off"
        fla = "on" if kernels["flash_linear_attention"] else "off"
        lines = [
            f"[train] {self.args.size} ({arch.family_of(self.args.size)}): "
            f"{arch.num_parameters(self.config) / 1e9:.2f}B parameters on {self.world} x "
            f"{self.gpu_name} ({self.parallel}, {str(self.param_dtype).split('.')[-1]}, flex {flex}, fla {fla})",
            f"[train] batch {self.tokens_per_step:,} tokens a step = {self.args.micro_batch} x {self.args.seq_len} x "
            f"{self.world} GPUs x {self.accumulation} accumulated; {self.args.optimizer}"
            f"{' + muP' if self.args.mup else ''}, lr {self.args.lr:.1e}, warmup {self.warmup_steps}, "
            f"decay over the last {self.args.decay_fraction:.0%}",
            "[train] planned: "
            + (
                f"{self.total_steps:,} steps, {self.total_steps * self.tokens_per_step / 1e9:.1f}B tokens"
                if self.total_steps
                else f"fitted to {self.args.hours:g} h once throughput is measured"
            ),  # fmt: skip
            self.loader.describe(self.total_steps * self.tokens_per_step if self.total_steps else None),
        ]
        return "\n".join(lines)


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        trainer = Trainer(args)
    except torch.OutOfMemoryError:
        print(f"[train] out of memory building the model at micro-batch {args.micro_batch}", flush=True)
        return 3
    try:
        trainer.run()
    except torch.OutOfMemoryError:
        print(
            f"[train] out of memory at micro-batch {args.micro_batch}: resume with a smaller --micro-batch", flush=True
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
