# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Muon and muP: the orthogonalization, the shape-matched scale, the parameter split, and the coordinate check."""

import torch
from torch import nn

from helix_lm.anansi import AnansiConfig, AnansiForCausalLM
from helix_lm.optim import Chained, Muon, build_optimizer, mup_plan
from helix_lm.optim.muon import muon_scale, newton_schulz, partition_parameters


def test_newton_schulz_orthogonalizes_both_orientations():
    torch.manual_seed(0)
    for shape in ((64, 32), (32, 64), (48, 48)):
        out = newton_schulz(torch.randn(*shape), steps=5)
        singular = torch.linalg.svdvals(out.float())
        assert singular.min() > 0.6 and singular.max() < 1.4, (shape, singular.min(), singular.max())


def test_muon_update_rms_is_shape_matched():
    torch.manual_seed(0)
    g = torch.randn(256, 1024)
    update = newton_schulz(g) * muon_scale(g.shape)
    rms = update.pow(2).mean().sqrt().item()
    assert 0.14 < rms < 0.26, rms  # Moonshot's 0.2 target, within 30%


def test_muon_learns_a_linear_map_faster_than_it_forgets():
    torch.manual_seed(0)
    target = torch.randn(32, 16)
    w = nn.Parameter(torch.zeros(32, 16))
    opt = Muon([w], lr=0.3, momentum=0.9, weight_decay=0.0)
    x = torch.randn(256, 16)
    losses = []
    for _ in range(60):
        loss = ((x @ w.T - x @ target.T) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.1 * losses[0]


def test_parameter_split_and_chained_state():
    torch.manual_seed(0)
    model = AnansiForCausalLM(AnansiConfig.named("a-20m", vocab_size=128, ngram_memory_size=256, loss_chunk_tokens=0))
    muon_params, decay, no_decay = partition_parameters(model)
    ids = {id(p) for p in muon_params}
    assert id(model.model.embed_tokens.weight) not in ids and id(model.lm_head.weight) not in ids
    assert all(p.dim() == 2 for p in muon_params) and all(p.dim() < 2 for p in no_decay)
    assert id(model.model.layers[1].mlp.gate_proj.weight) in ids
    optimizer = build_optimizer(model, "muon", lr=1e-3, weight_decay=0.1)
    assert isinstance(optimizer, Chained) and len(optimizer.optimizers) == 2
    x = torch.randint(0, 128, (2, 32))
    model.train()
    model(x, labels=x).loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    state = optimizer.state_dict()
    again = build_optimizer(model, "muon", lr=1e-3, weight_decay=0.1)
    again.load_state_dict(state)
    assert len(again.state) == len(optimizer.state) > 0
    adamw = build_optimizer(model, "adamw", lr=1e-3, weight_decay=0.1)
    assert isinstance(adamw, torch.optim.AdamW) and all("lr_mult" in g for g in adamw.param_groups)


class Toy(nn.Module):
    def __init__(self, width: int, vocab: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.hidden = nn.Linear(width, width, bias=False)
        self.lm_head = nn.Linear(width, vocab, bias=False)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.hidden.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, ids):
        self.activations = torch.relu(self.hidden(self.embed(ids)))
        return self.lm_head(self.activations)


def _activation_change(width: int, mup: bool, base: int = 32) -> float:
    torch.manual_seed(0)
    model = Toy(width)
    plan = mup_plan(model, width, base) if mup else None
    if plan:
        for p in model.parameters():
            if id(p) in plan.init_std:
                nn.init.normal_(p, std=plan.init_std[id(p)])
    optimizer = build_optimizer(
        model, "adamw", lr=1e-2, weight_decay=0.0,
        lr_multipliers=plan.lr_multiplier if plan else None, wd_multipliers=plan.wd_multiplier if plan else None,
    )  # fmt: skip
    ids = torch.randint(0, 64, (64,))
    before = model(ids)
    multiplier = plan.output_multiplier if plan else 1.0
    loss = nn.functional.cross_entropy(before * multiplier, torch.roll(ids, 1))
    acts_before = model.activations.detach().clone()
    loss.backward()
    optimizer.step()
    model(ids)
    return (model.activations - acts_before).abs().mean().item()


def test_mup_keeps_activation_updates_width_independent():
    narrow_sp, wide_sp = _activation_change(32, mup=False), _activation_change(512, mup=False)
    narrow_mup, wide_mup = _activation_change(32, mup=True), _activation_change(512, mup=True)
    assert wide_mup / narrow_mup < wide_sp / narrow_sp, (wide_mup / narrow_mup, wide_sp / narrow_sp)
    assert wide_mup / narrow_mup < 3.0


def test_mup_plan_rules():
    model = Toy(128)
    plan = mup_plan(model, 128, 32)
    assert plan.width_multiplier == 4 and plan.output_multiplier == 0.25
    assert plan.lr_multiplier[id(model.embed.weight)] == 1.0
    assert plan.lr_multiplier[id(model.hidden.weight)] == 0.25 and plan.wd_multiplier[id(model.hidden.weight)] == 4
    assert plan.lr_multiplier[id(model.lm_head.weight)] == 0.25
    assert abs(plan.init_std[id(model.hidden.weight)] - 0.01) < 1e-9
    assert "width x4" in plan.describe()
