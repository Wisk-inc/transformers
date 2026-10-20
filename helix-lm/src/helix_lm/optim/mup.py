# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
muP (maximal update parametrization) as a plan: per-parameter init scales, learning-rate multipliers and an output
multiplier, derived from how much wider a model is than a base width.

With muP (Yang et al., "Tensor Programs V", 2022) the learning rate, init and other hyperparameters tuned on a
small model transfer to a wide one: the pilot's best learning rate is the big run's learning rate. The rules
for Adam-style optimizers, with width multiplier ``m = hidden / base_hidden``:

| parameter kind                      | init std            | lr multiplier | notes                           |
| --- | --- | --- | --- |
| embeddings (input)                  | ``s``               | 1             | rows are independent features   |
| hidden matrices (fan_in grows)      | ``s / sqrt(m)``     | ``1 / m``     | weight decay x m (lr x wd fixed) |
| output head                         | ``s / m`` (or zero) | ``1 / m``     | logits also multiplied by 1/m   |
| vectors (norms, biases, A_log ...)  | unchanged           | 1             |                                 |

Here ``s`` is the base init std (0.02). For Muon the hidden-matrix learning rate does not need the ``1/m``
factor, because the orthogonalized update already has width-independent RMS (Moonshot's scaling rule); the
plan therefore carries one multiplier per parameter and lets the optimizer builder decide which groups see it.
The output multiplier is applied by the model (``AnansiConfig.output_multiplier``), which is the one muP rule a
plan cannot express as init or lr.

Validation at toy scale (``tests/test_optim.py``): with the plan, the mean absolute change of the hidden
activations after one optimizer step is of the same order for a 64-wide and a 512-wide model; without it, the
wide model's changes grow with width.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class MupPlan:
    width_multiplier: float
    init_std: dict[int, float] = field(default_factory=dict)  # id(param) -> std
    lr_multiplier: dict[int, float] = field(default_factory=dict)  # id(param) -> factor
    wd_multiplier: dict[int, float] = field(default_factory=dict)  # id(param) -> factor (keeps lr x wd fixed)
    output_multiplier: float = 1.0

    def describe(self) -> str:
        kinds = {}
        for factor in self.lr_multiplier.values():
            kinds[factor] = kinds.get(factor, 0) + 1
        groups = ", ".join(f"{n} params x{f:.3g}" for f, n in sorted(kinds.items()))
        return f"muP: width x{self.width_multiplier:.3g}, output x{self.output_multiplier:.3g}, lr groups: {groups}"


def mup_plan(
    model: nn.Module,
    hidden_size: int,
    base_hidden: int,
    base_std: float = 0.02,
    muon_matrices: set[int] | None = None,
) -> MupPlan:
    """
    The plan for ``model``: embeddings keep ``base_std`` and lr; the output head and hidden matrices get the
    width-scaled rules; vectors are untouched. ``muon_matrices`` (ids) keep lr multiplier 1 (Muon handles width).
    """
    m = hidden_size / base_hidden
    plan = MupPlan(width_multiplier=m, output_multiplier=1.0 / m)
    inner = model.module if hasattr(model, "module") else model
    muon_matrices = muon_matrices or set()
    for name, module in inner.named_modules():
        is_embedding = isinstance(module, nn.Embedding)
        is_head = name.endswith("lm_head") or name.endswith(".head")
        for p in module.parameters(recurse=False):
            if p.dim() < 2:
                continue
            if is_embedding:
                plan.init_std[id(p)] = base_std
                plan.lr_multiplier[id(p)] = 1.0
            elif is_head:
                plan.init_std[id(p)] = base_std / m
                plan.lr_multiplier[id(p)] = 1.0 / m
            else:
                plan.init_std[id(p)] = base_std / math.sqrt(m)
                plan.lr_multiplier[id(p)] = 1.0 if id(p) in muon_matrices else 1.0 / m
                plan.wd_multiplier[id(p)] = 1.0 if id(p) in muon_matrices else m
    return plan


@torch.no_grad()
def apply_init(model: nn.Module, plan: MupPlan, skip: set[int] | None = None) -> int:
    """Re-draw every planned matrix with its muP std, in place (DTensor-safe). Returns how many were touched."""
    skip = skip or set()
    inner = model.module if hasattr(model, "module") else model
    touched = 0
    for p in inner.parameters():
        std = plan.init_std.get(id(p))
        if std is None or id(p) in skip:
            continue
        p.normal_(mean=0.0, std=std)
        touched += 1
    return touched
