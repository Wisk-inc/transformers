# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Muon: momentum orthogonalized by Newton-Schulz, for the 2D weights; AdamW for everything else.

Muon (Jordan et al., 2024; scaled to 16B/1T by Moonshot, "Muon is Scalable for LLM Training", 2025) replaces
the elementwise Adam update of a weight matrix by the nearest orthogonal matrix to its momentum: every singular
direction of the update gets the same magnitude, so the rare directions a matrix has not learned yet move as fast
as the common ones. Moonshot report roughly 2x the token efficiency of AdamW at the same compute, and Kimi K2
(1T parameters) was trained with it. The three things that made it work at scale are all here:

* the **Newton-Schulz quintic** (coefficients 3.4445, -4.7750, 2.0315; five iterations in bf16) that
  orthogonalizes a matrix with matmuls only, no SVD;
* **shape-matched scaling** ``0.2 * sqrt(max(rows, cols))`` so one learning rate serves every matrix and can be
  shared with the AdamW groups (Moonshot's rule);
* **decoupled weight decay** ``w <- w - lr * wd * w`` on the Muon group, without which the weights' RMS grows
  until bf16 loses precision.

Under FSDP2 each rank holds a shard of every matrix, and Newton-Schulz needs the whole matrix: the update
gathers a parameter's momentum (``DTensor.full_tensor``), orthogonalizes it on every rank, and writes back its own
shard. That costs one all-gather per matrix per step, which is the same traffic as the forward pass and is how
Moonshot's distributed Muon works too (they overlap it; this one does not yet).

Embeddings, the output head, norms, biases and the recurrence's scalar parameters stay on AdamW: Muon's
orthogonalization is meaningless for vectors and harmful for the embedding table, whose rows are
independent features, not a linear map.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def newton_schulz(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    The matrix with the same singular vectors as ``matrix`` and all singular values pushed towards 1, by
    ``steps`` iterations of the quintic ``X <- a X + b (X Xᵀ) X + c (X Xᵀ)² X`` on the spectrally normalized input.
    Works on the transpose when the matrix is tall so the Gram matrix is the small one.
    """
    a, b, c = NS_COEFFICIENTS
    x = matrix.to(torch.bfloat16 if matrix.is_cuda else torch.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT
    x = x / (x.norm() + eps)
    for _ in range(steps):
        gram = x @ x.mT
        update = b * gram + c * (gram @ gram)
        x = a * x + update @ x
    if transposed:
        x = x.mT
    return x.to(matrix.dtype)


def muon_scale(shape: torch.Size) -> float:
    """Moonshot's shape-matched factor so that Muon and AdamW share a learning rate: ``0.2 * sqrt(max(m, n))``."""
    return 0.2 * math.sqrt(max(shape[0], shape[1]))


class Muon(torch.optim.Optimizer):
    """
    Muon for the parameters it is given (all must be 2D). Pass the matrices only; build the AdamW for the rest
    with :func:`build_optimizer`, which also knows which parameters are which.

    Args:
        params: 2D parameters (plain tensors or FSDP2 DTensors).
        lr: learning rate, the same number as the AdamW groups thanks to the shape-matched scale.
        momentum: Nesterov momentum coefficient (0.95 is the published default).
        weight_decay: decoupled weight decay.
        ns_steps: Newton-Schulz iterations.
    """

    def __init__(self, params, lr: float = 3e-4, momentum: float = 0.95, weight_decay: float = 0.1, ns_steps: int = 5):
        defaults = {"lr": lr, "momentum": momentum, "weight_decay": weight_decay, "ns_steps": ns_steps}
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() != 2:
                    raise ValueError(f"Muon takes 2D parameters only, got shape {tuple(p.shape)}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mu, wd, steps = group["lr"], group["momentum"], group["weight_decay"], group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(p.grad)
                update = p.grad.add(buf, alpha=mu)  # Nesterov: the gradient plus the decayed momentum
                full = update.full_tensor() if hasattr(update, "full_tensor") else update
                ortho = newton_schulz(full, steps=steps) * muon_scale(full.shape)
                if hasattr(p, "full_tensor"):  # DTensor: write back this rank's shard
                    from torch.distributed.tensor import distribute_tensor

                    ortho = distribute_tensor(ortho.to(p.dtype), p.device_mesh, p.placements)
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(ortho.to(p.dtype), alpha=-lr)
        return loss


def partition_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    """
    ``(muon, adam_decay, adam_no_decay)``: the 2D weights of linear maps inside the blocks go to Muon; the
    embedding table and the output head (2D, but not linear maps between hidden features) and every 1D parameter
    go to AdamW, the 1D ones without weight decay.
    """
    inner = model.module if hasattr(model, "module") else model
    excluded = set()
    for name, module in inner.named_modules():
        if isinstance(module, nn.Embedding) or name.endswith("lm_head") or "mtp" in name and name.endswith("head"):
            excluded.update(id(p) for p in module.parameters(recurse=False))
    muon, decay, no_decay = [], [], []
    for p in inner.parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 2 and id(p) not in excluded:
            muon.append(p)
        elif p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return muon, decay, no_decay


class Chained:
    """
    Several optimizers stepped as one. Not an ``Optimizer`` subclass on purpose: the trainer's loop sees one
    object with ``param_groups``, ``step``, ``zero_grad`` and a state dict, and the distributed checkpoint
    helpers are handed the underlying optimizers (``.optimizers``), which they accept as a list.
    """

    def __init__(self, optimizers: Iterable[torch.optim.Optimizer]):
        self.optimizers = list(optimizers)
        self.param_groups = [g for o in self.optimizers for g in o.param_groups]
        self.defaults = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for o in self.optimizers:
            o.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"optimizers": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state_dict: dict) -> None:
        for o, s in zip(self.optimizers, state_dict["optimizers"], strict=True):
            o.load_state_dict(s)

    @property
    def state(self) -> dict:
        merged = {}
        for o in self.optimizers:
            merged.update(o.state)
        return merged


def unwrap_optimizers(optimizer) -> list[torch.optim.Optimizer]:
    """The real optimizers behind ``optimizer`` (one, or Chained's several), for checkpoint helpers."""
    return list(optimizer.optimizers) if isinstance(optimizer, Chained) else [optimizer]


def build_optimizer(
    model: nn.Module,
    name: str,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    momentum: float = 0.95,
    fused: bool = False,
    lr_multipliers: dict[int, float] | None = None,
    wd_multipliers: dict[int, float] | None = None,
) -> torch.optim.Optimizer:
    """
    ``name`` is ``"adamw"`` (one AdamW, decay on matrices only: the 1.x behaviour) or ``"muon"`` (Muon on the
    block matrices, AdamW on the rest). ``lr_multipliers`` maps ``id(param)`` to a factor (muP); parameters with
    the same factor share a group so the schedule can scale each group's ``lr`` by its ``lr_mult``.
    """
    multipliers, wd_mults = lr_multipliers or {}, wd_multipliers or {}

    def grouped(params: list[nn.Parameter], weight_decay: float, **extra) -> list[dict]:
        buckets: dict[tuple[float, float], list] = {}
        for p in params:
            buckets.setdefault((multipliers.get(id(p), 1.0), wd_mults.get(id(p), 1.0)), []).append(p)
        return [
            {"params": ps, "lr": lr * m, "lr_mult": m, "weight_decay": weight_decay * w, **extra}
            for (m, w), ps in sorted(buckets.items())
        ]

    muon_params, decay, no_decay = partition_parameters(model)
    if name == "adamw":
        groups = grouped(muon_params + decay, weight_decay) + grouped(no_decay, 0.0)
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, fused=fused)
    if name != "muon":
        raise ValueError(f"unknown optimizer {name!r}: adamw or muon")
    adam = torch.optim.AdamW(
        grouped(decay, weight_decay) + grouped(no_decay, 0.0),
        lr=lr, betas=betas, eps=1e-8, fused=fused,
    )  # fmt: skip
    muon = Muon(
        grouped(muon_params, weight_decay, momentum=momentum),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    return Chained([muon, adam])
