# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The model families the engine trains.

* ``helix`` -- :class:`helix_lm.model.HelixForCausalLM`, the three-strand architecture.
* ``transformer`` -- a dense decoder of the Qwen2/Llama family through ``transformers`` (RMSNorm, rotary,
  grouped-query attention with QKV bias, SwiGLU), which every serving stack already loads: vLLM, SGLang,
  llama.cpp, ``transformers`` itself. **CorX5.5 is this family at ``t-34b``.**

Both are built the same way: on the meta device, sharded with FSDP2 block by block, materialized, then
initialized in place -- so a 34B model never exists whole on one GPU. The trainer only talks to this module.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from .config import SIZES, HelixConfig

# Dense transformer sizes: hidden, MLP, layers, query heads, key/value heads. The 34b row has Qwen2.5-32B's
# shape with a wider MLP, which with the 65,536-token CorxLabs vocabulary comes to 33.9B parameters.
TRANSFORMER_SIZES: dict[str, dict] = {
    "t-350m": dict(hidden_size=1024, intermediate_size=2816, num_hidden_layers=16, num_attention_heads=16, num_key_value_heads=4),
    "t-1b": dict(hidden_size=2048, intermediate_size=5632, num_hidden_layers=24, num_attention_heads=16, num_key_value_heads=4),
    "t-3b": dict(hidden_size=2560, intermediate_size=6912, num_hidden_layers=32, num_attention_heads=20, num_key_value_heads=4),
    "t-7b": dict(hidden_size=3584, intermediate_size=18944, num_hidden_layers=28, num_attention_heads=28, num_key_value_heads=4),
    "t-14b": dict(hidden_size=5120, intermediate_size=13824, num_hidden_layers=48, num_attention_heads=40, num_key_value_heads=8),
    "t-34b": dict(hidden_size=5120, intermediate_size=29696, num_hidden_layers=64, num_attention_heads=40, num_key_value_heads=8),
}  # fmt: skip

ALL_SIZES: tuple[str, ...] = tuple(SIZES) + tuple(TRANSFORMER_SIZES)


def family_of(size: str) -> str:
    if size in TRANSFORMER_SIZES:
        return "transformer"
    if size in SIZES:
        return "helix"
    raise ValueError(f"unknown size {size!r}; choose from {', '.join(ALL_SIZES)}")


# ---------------------------------------------------------------------------------------- configs --
def make_config(size: str, vocab_size: int = 65536, **overrides):
    """A configuration for a named size: :class:`HelixConfig`, or a ``Qwen2Config`` for the transformer."""
    if family_of(size) == "helix":
        return HelixConfig.named(size, **overrides)
    from transformers import Qwen2Config

    fields = {
        "vocab_size": vocab_size,
        "tie_word_embeddings": False,
        "rope_theta": 1_000_000.0,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "initializer_range": 0.02,
        "use_cache": False,
        "use_sliding_window": False,
        "bos_token_id": 1,
        "eos_token_id": 0,
        **TRANSFORMER_SIZES[size],
        **overrides,
    }
    config = Qwen2Config(**fields)
    config._attn_implementation = "sdpa"
    config.corx_size = size
    return config


def num_parameters(config, embeddings: bool = True) -> int:
    """Parameter count from the shapes, for either family."""
    if isinstance(config, HelixConfig):
        return config.num_parameters(embeddings)
    h, inter, L = config.hidden_size, config.intermediate_size, config.num_hidden_layers
    H, KV = config.num_attention_heads, config.num_key_value_heads
    hd = getattr(config, "head_dim", None) or h // H
    attention = h * H * hd + 2 * h * KV * hd + H * hd * h + (H * hd + 2 * KV * hd)  # q, k, v, o, qkv biases
    mlp = 3 * h * inter
    total = L * (attention + mlp + 2 * h) + h
    if embeddings:
        total += config.vocab_size * h * (1 if config.tie_word_embeddings else 2)
    return int(total)


def flops_per_token(config, seq_len: int) -> float:
    """Training FLOPs per token: ``6N`` plus attention (``12 L h s`` for a dense causal window)."""
    if isinstance(config, HelixConfig):
        return config.flops_per_token()
    weights = 6.0 * num_parameters(config, embeddings=False) + 6.0 * config.vocab_size * config.hidden_size
    attention = 12.0 * config.num_hidden_layers * config.hidden_size * seq_len
    return weights + attention


def describe_transformer_sizes() -> str:
    widths = (
        ("size", 6),
        ("params", 8),
        ("non-embed", 10),
        ("layers", 6),
        ("hidden", 6),
        ("mlp", 6),
        ("tokens x20", 10),
    )
    rows = ["  ".join(f"{name:>{width}}" for name, width in widths)]
    for name in TRANSFORMER_SIZES:
        config = make_config(name)
        total, body = num_parameters(config), num_parameters(config, embeddings=False)
        rows.append(
            f"{name:>6}  {total / 1e9:7.2f}B  {body / 1e9:9.2f}B  {config.num_hidden_layers:>6}  {config.hidden_size:>6}  "
            f"{config.intermediate_size:>6}  {20 * total / 1e9:8.0f}B"
        )
    return "\n".join(rows)


# ------------------------------------------------------------------------------------------ models --
def construct(config) -> nn.Module:
    """The module for a config, on the current default device (call under ``torch.device("meta")``)."""
    if isinstance(config, HelixConfig):
        from .model import HelixForCausalLM

        return HelixForCausalLM(config)
    from transformers import Qwen2ForCausalLM

    return Qwen2ForCausalLM(config)


def layers_of(model: nn.Module) -> list[nn.Module]:
    inner = model.module if hasattr(model, "module") else model
    return list(inner.model.layers)


@torch.no_grad()
def initialize(model: nn.Module, config, seed: int) -> None:
    """
    Fresh weights, in place, on a model that may have come from ``to_empty`` and may hold DTensors: every
    operation is an in-place random fill or a buffer copy. The three residual-writing projections of each
    block get ``std / sqrt(2 * layers)`` in both families.
    """
    torch.manual_seed(seed)
    if isinstance(config, HelixConfig):
        from .model import init_weights, scale_residual_projections

        model.apply(lambda module: init_weights(module, config))
        scale_residual_projections(model.model.layers, config)
        return
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    model.apply(model._init_weights)
    std = config.initializer_range / math.sqrt(2 * config.num_hidden_layers)
    for layer in model.model.layers:
        layer.self_attn.o_proj.weight.normal_(mean=0.0, std=std)
        layer.mlp.down_proj.weight.normal_(mean=0.0, std=std)
    # The rotary frequencies are a buffer computed in __init__, so after `to_empty` they are garbage.
    device = model.model.norm.weight.device
    inv_freq, _ = Qwen2RotaryEmbedding.compute_default_rope_parameters(config, device=device)
    model.model.rotary_emb.inv_freq.copy_(inv_freq.to(model.model.rotary_emb.inv_freq.dtype))
    model.model.rotary_emb.original_inv_freq = model.model.rotary_emb.inv_freq


def set_activation_checkpointing(model: nn.Module, config, enabled: bool) -> None:
    if isinstance(config, HelixConfig):
        config.activation_checkpointing = enabled
    elif enabled:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def load_initial_weights(model: nn.Module, path: str, parallel: str) -> None:
    """Weights from a saved export of either family (``model.safetensors`` or sharded) into a built model."""
    from .model import HelixForCausalLM

    tensors = HelixForCausalLM.load_state_dict_from(path)
    if parallel == "fsdp":
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        set_model_state_dict(
            model, tensors, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=False)
        )
    else:
        (model.module if hasattr(model, "module") else model).load_state_dict(tensors)


def export(
    model: nn.Module, config, state_dict: dict, directory: str | Path, tokenizer_dir: str | Path | None
) -> Path:
    """
    A bf16 safetensors copy with the tokenizer beside it. The transformer family writes a standard
    ``transformers`` folder (``config.json``, ``model-*.safetensors``, ``tokenizer.json`` ...), so the
    export is what vLLM or ``AutoModelForCausalLM.from_pretrained`` loads directly.
    """
    import shutil

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    inner = model.module if hasattr(model, "module") else model
    if isinstance(config, HelixConfig):
        inner.save_pretrained(directory, state_dict=state_dict, dtype=torch.bfloat16)
    else:
        tensors = {k: v.detach().to("cpu", torch.bfloat16).contiguous() for k, v in state_dict.items()}
        inner.config.dtype = torch.bfloat16
        inner.save_pretrained(directory, state_dict=tensors, safe_serialization=True)
        for stale in directory.glob("model.pt"):
            stale.unlink()
    if tokenizer_dir:
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            source = Path(tokenizer_dir) / name
            if source.exists():
                shutil.copy(source, directory / name)
    return directory
