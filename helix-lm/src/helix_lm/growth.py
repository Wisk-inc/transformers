# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Progressive growth: turn a trained small model into a bigger one that computes the *same function*, then keep
training. The small model learns the cheap early lessons (token statistics, syntax, short-range structure) at a
fraction of the cost per token; the grown model inherits them instead of re-learning them at full price. The
function-preserving operators here are the ones with the cleanest evidence (Net2Net 2016, bert2BERT 2022,
Apple's HyperCloning 2024, depth up-scaling as in SOLAR 2024):

* **deeper**: new blocks are inserted with every residual-writing projection zeroed, so each new block is the
  identity at first (and its gates start where the gradient can move them);
* **wider MLP**: new hidden units of the SwiGLU get fresh ``gate``/``up`` rows and zero ``down`` columns, so
  they add nothing until trained;
* **more streams** (Anansi): new residual streams are copies of existing ones with the hyper-connection
  weights renormalized, so the mixture the blocks read is unchanged.

Growing works on exported weights, not inside a running job: ``helix grow`` reads an export, writes a bigger
export, and the next ``train --init-from`` continues from it. That keeps growth independent of how the running
job is sharded. ``tests/test_growth.py`` checks the grown model's logits equal the small one's to float precision.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from .anansi import AnansiConfig, AnansiForCausalLM
from .config import HelixConfig
from .model import HelixForCausalLM

_LAYER = re.compile(r"^(model\.layers\.)(\d+)(\..*)$")
_RESIDUAL_WRITERS = ("o_proj.weight", "out_proj.weight", "down_proj.weight", "w_down")


def load_export(directory: str | Path) -> tuple[HelixConfig | AnansiConfig, dict[str, torch.Tensor]]:
    directory = Path(directory)
    raw = json.loads((directory / "config.json").read_text())
    config = (AnansiConfig if raw.get("model_type") == "anansi" else HelixConfig).from_dict(raw)
    return config, HelixForCausalLM.load_state_dict_from(directory)


def _renumber(state: dict[str, torch.Tensor], mapping: dict[int, int]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        match = _LAYER.match(key)
        if match:
            old = int(match.group(2))
            key = f"{match.group(1)}{mapping[old]}{match.group(3)}"
        out[key] = value
    return out


def _layer_keys(state: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{index}."
    return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}


@torch.no_grad()
def grow_depth(
    config,
    state: dict[str, torch.Tensor],
    insert_after: dict[int, int],
    template: dict[int, int] | None = None,
    identity: bool = True,
) -> tuple[object, dict[str, torch.Tensor]]:
    """
    Insert ``insert_after[i]`` new identity blocks after old block ``i``. Each new block copies the weights of
    a template block (``template[i]``, default ``i``) so its norms and strands start in a trained regime, with
    its residual-writing projections zeroed so it adds nothing. For Anansi the hyper-connection ``post``
    logits are pushed far negative for the same reason.
    """
    old_layers = config.num_hidden_layers
    template = template or {}
    mapping, new_index, inserted = {}, 0, {}
    for old in range(old_layers):
        mapping[old] = new_index
        new_index += 1
        for _ in range(insert_after.get(old, 0)):
            inserted[new_index] = template.get(old, old)
            new_index += 1
    grown = _renumber(state, mapping)
    for new, source in inserted.items():
        block = _layer_keys(state, source)
        for name, value in block.items():
            value = value.clone()
            if identity and name.endswith(_RESIDUAL_WRITERS):
                value.zero_()
            if identity and name.endswith("post_logits"):
                value.fill_(-12.0)
            if identity and name.endswith("res_logits"):  # exact identity mixing, not the soft init
                value.fill_(0.0)
                value += 30.0 * torch.eye(value.shape[-1], dtype=value.dtype)
            if identity and name.endswith("dyn.weight"):
                value.zero_()
            grown[f"model.layers.{new}.{name}"] = value
    new_config = _with_layers(config, new_index, mapping, inserted)
    return new_config, grown


def _with_layers(config, num_layers: int, mapping: dict[int, int], inserted: dict[int, int]):
    values = config.to_dict()
    old_types = list(config.layer_types)
    types = [None] * num_layers
    for old, new in mapping.items():
        types[new] = old_types[old]
    for new, source in inserted.items():
        types[new] = old_types[source]
    values["layer_types"] = types
    values["num_hidden_layers"] = num_layers
    if isinstance(config, AnansiConfig):
        # New blocks extend the section they were inserted into.
        prelude, core = config.prelude_layers, config.core_layers
        p = sum(1 for new, src in inserted.items() if src < prelude) + prelude
        c = sum(1 for new, src in inserted.items() if prelude <= src < prelude + core) + core
        values.update(prelude_layers=p, core_layers=c, coda_layers=num_layers - p - c)
        return AnansiConfig(**values)
    return HelixConfig(**values)


@torch.no_grad()
def grow_mlp(config, state: dict[str, torch.Tensor], intermediate_size: int, std: float | None = None):
    """Widen every dense SwiGLU to ``intermediate_size``: new ``gate``/``up`` rows random, new ``down`` columns
    zero."""
    old = config.intermediate_size
    if intermediate_size < old:
        raise ValueError("growth only widens")
    std = config.initializer_range if std is None else std
    extra = intermediate_size - old
    grown = {}
    for key, value in state.items():
        if key.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")) and value.shape[0] == old:
            fresh = torch.randn(extra, value.shape[1], dtype=value.dtype) * std
            value = torch.cat([value, fresh], dim=0)
        elif key.endswith("mlp.down_proj.weight") and value.shape[1] == old:
            value = torch.cat([value, torch.zeros(value.shape[0], extra, dtype=value.dtype)], dim=1)
        grown[key] = value
    values = {**config.to_dict(), "intermediate_size": intermediate_size}
    return type(config)(**values), grown


@torch.no_grad()
def grow_streams(config: AnansiConfig, state: dict[str, torch.Tensor], num_streams: int):
    """
    More residual streams for an Anansi model. Every stream starts as the embedding; the ``pre`` and ``post`` logits
    of new streams are pushed far negative so no block reads or writes them until the gradient says so, ``res`` keeps
    the old block of the mixing matrix with the new streams carried by the identity, the dynamic maps get zero
    columns for the new streams, and the readout gives them no weight. The function is unchanged.
    """
    old = config.num_streams
    if num_streams < old:
        raise ValueError("growth only widens")
    n = num_streams
    grown = {}
    for key, value in state.items():
        if key.endswith(("pre_logits", "post_logits")) and value.shape[-1] == old:
            value = torch.cat([value, torch.full((n - old,), -12.0, dtype=value.dtype)])
        elif key.endswith("res_logits") and value.shape[-1] == old:
            big = torch.full((n, n), -12.0, dtype=value.dtype)
            big[:old, :old] = value
            big[old:, old:] = torch.eye(n - old, dtype=value.dtype) * 12.0 - 12.0 + 6.0
            value = big
        elif key.endswith("dyn.weight") and value.shape[0] == 2 * old + old * old:
            h = value.shape[1] // old
            new = torch.zeros(2 * n + n * n, n * h, dtype=value.dtype)
            cols = slice(0, old * h)  # the old streams keep their columns; new streams' columns start at zero
            new[:old, cols] = value[:old]
            new[n : n + old, cols] = value[old : 2 * old]
            old_res = value[2 * old :].reshape(old, old, -1)
            new_res = new[2 * n :].reshape(n, n, -1)
            new_res[:old, :old, : old * h] = old_res
            new[2 * n :] = new_res.reshape(n * n, -1)
            value = new
        elif key.endswith("model.readout") and value.shape[-1] == old:
            value = torch.cat([value, torch.full((n - old,), -12.0, dtype=value.dtype)])
        grown[key] = value
    values = {**config.to_dict(), "num_streams": n}
    return AnansiConfig(**values), grown


def plan_from_sizes(config, target: str):
    """What a named target size asks for relative to ``config``: the growth steps as a list of (kind, value)."""
    target_config = AnansiConfig.named(target) if target.startswith("a-") else HelixConfig.named(target)
    if (
        target_config.hidden_size != config.hidden_size
        or target_config.num_attention_heads != config.num_attention_heads
    ):
        raise ValueError("growth keeps the width and the heads; pick a target with the same hidden_size and heads")
    steps = []
    if target_config.intermediate_size > config.intermediate_size:
        steps.append(("mlp", target_config.intermediate_size))
    if isinstance(config, AnansiConfig) and target_config.num_streams > config.num_streams:
        steps.append(("streams", target_config.num_streams))
    if target_config.num_hidden_layers > config.num_hidden_layers:
        steps.append(("depth", target_config.num_hidden_layers))
    return target_config, steps


def grow(config, state: dict[str, torch.Tensor], steps: list[tuple[str, int]], mode: str = "identity"):
    """
    Apply growth steps in order; returns the new config and state. ``mode`` for depth: ``"identity"`` inserts
    blocks that compute nothing yet (exactly function-preserving); ``"stack"`` inserts copies of trained blocks
    as they are (G_stack, NeurIPS 2024: not function-preserving, yet it reached a 7B baseline's loss with 45%
    fewer tokens, ahead of the preserving variants). Stack when training continues at once; identity when the
    grown model must behave like the old one immediately.
    """
    if mode not in ("identity", "stack"):
        raise ValueError("mode is identity or stack")
    for kind, value in steps:
        if kind == "mlp":
            config, state = grow_mlp(config, state, value)
        elif kind == "streams":
            config, state = grow_streams(config, state, value)
        elif kind == "depth":
            extra = value - config.num_hidden_layers
            if isinstance(config, AnansiConfig):  # spread new blocks over the core, the looped part
                core = list(config.core_range)
            else:
                core = list(range(config.num_hidden_layers))
            insert_after = {}
            for j in range(extra):
                where = core[(j * len(core)) // max(extra, 1) % len(core)]
                insert_after[where] = insert_after.get(where, 0) + 1
            config, state = grow_depth(config, state, insert_after, identity=mode == "identity")
        else:
            raise ValueError(f"unknown growth step {kind!r}")
    return config, state


def build(config):
    return AnansiForCausalLM(config) if isinstance(config, AnansiConfig) else HelixForCausalLM(config)


def grow_export(source: str | Path, target: str, out: str | Path, mode: str = "stack") -> Path:
    """``helix grow``: read an export, grow it to the named size, write the new export (fp32, with the tokenizer)."""
    config, state = load_export(source)
    target_config, steps = plan_from_sizes(config, target)
    config, state = grow(config, state, steps, mode=mode)
    model = build(config)
    model.load_state_dict(state)
    out = Path(out)
    model.save_pretrained(out, dtype=torch.float32)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = Path(source) / name
        if path.exists():
            (out / name).write_bytes(path.read_bytes())
    (out / "growth.json").write_text(
        json.dumps({"from": str(source), "target": target, "steps": steps, "mode": mode}, indent=1)
    )
    return out
