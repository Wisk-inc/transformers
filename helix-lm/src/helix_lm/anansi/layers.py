# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The parts of an Anansi block that HELIX does not have: hyper-connected residual streams, the gated mixer, the
hashed n-gram memory and the fine-grained mixture of experts.
"""

from __future__ import annotations

import torch
from torch import nn

from ..model import ACT2FN, HelixBraid, HelixMLP, HelixRecurrentStrand, HelixRMSNorm
from .config import AnansiConfig


# ------------------------------------------------------------------------------------ hyper-connections --
def sinkhorn(logits: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """
    The doubly-stochastic matrix nearest (in KL) to ``exp(logits)``: alternate row and column normalization in
    the log domain. Every row and every column sums to one, so mixing the streams with it can neither amplify
    nor lose signal -- the manifold constraint that keeps many-stream residuals stable (DeepSeek mHC, 2026).
    """
    dtype = logits.dtype
    logits = logits.float()  # fp32: the alternating normalization is where bf16 drifts
    for _ in range(iterations):
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
    return logits.exp().to(dtype)


class HyperConnections(nn.Module):
    """
    ``num_streams`` residual streams around one block. Reading mixes the streams into the block's input with
    non-negative weights (``pre``); writing mixes the streams among themselves with a doubly-stochastic matrix
    (``res``) and adds the block's output to each with weights in ``(0, 2)`` (``post``, DeepSeek mHC's range).
    Each coefficient is a learned constant plus a small input-dependent term, so the network can route
    differently per token.

    At initialization block ``layer_idx`` reads and writes stream ``layer_idx % num_streams`` and ``res`` is the
    identity: a plain residual network whose depth is spread over the streams, which the training can then
    re-wire. With one stream this is exactly ``x + f(x)``.
    """

    def __init__(self, hidden_size: int, num_streams: int, layer_idx: int, dynamic: bool = True) -> None:
        super().__init__()
        n = self.num_streams = num_streams
        self.layer_idx = layer_idx
        self.pre_logits = nn.Parameter(torch.zeros(n))
        self.post_logits = nn.Parameter(torch.zeros(n))
        self.res_logits = nn.Parameter(torch.zeros(n, n))
        self.dynamic = dynamic and n > 1
        if self.dynamic:  # mHC: the dynamic part reads every (normalized) stream, flattened
            self.norm = HelixRMSNorm(hidden_size)
            self.dyn = nn.Linear(n * hidden_size, 2 * n + n * n, bias=False)
            self.dyn_scale = nn.Parameter(torch.zeros(3))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        n, k = self.num_streams, self.layer_idx % self.num_streams
        self.pre_logits.fill_(-4.0)
        self.pre_logits[k] = 4.0
        self.post_logits.fill_(-6.0)  # post = 2 sigmoid(.): 0 gives exactly 1 on the written stream
        self.post_logits[k] = 0.0
        self.res_logits.fill_(0.0)
        self.res_logits += 6.0 * torch.eye(n, device=self.res_logits.device, dtype=self.res_logits.dtype)
        if self.dynamic:  # small but not zero on both factors, or neither would ever get a gradient
            self.dyn.weight.normal_(mean=0.0, std=0.02)
            self.dyn_scale.fill_(0.01)

    def _coefficients(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.num_streams
        pre, post, res = self.pre_logits, self.post_logits, self.res_logits
        if self.dynamic:
            summary = self.norm(streams).flatten(-2)  # (B, T, n * h): each stream normalized, then concatenated
            dyn = torch.tanh(self.dyn(summary))
            scale = self.dyn_scale
            pre = pre + scale[0] * dyn[..., :n]
            post = post + scale[1] * dyn[..., n : 2 * n]
            res = res + scale[2] * dyn[..., 2 * n :].reshape(*dyn.shape[:-1], n, n)
        return torch.sigmoid(pre), 2.0 * torch.sigmoid(post), sinkhorn(res)

    def read(self, streams: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """``streams`` is ``(B, T, n, h)``; returns the block input ``(B, T, h)`` and the coefficients for write."""
        if self.num_streams == 1:
            return streams[:, :, 0], ()
        pre, post, res = self._coefficients(streams)
        x = torch.einsum("...n,...nh->...h", pre.expand(*streams.shape[:2], -1) if pre.dim() == 1 else pre, streams)
        return x, (post, res)

    def write(self, streams: torch.Tensor, output: torch.Tensor, coefficients: tuple) -> torch.Tensor:
        if self.num_streams == 1:
            return streams + output[:, :, None]
        post, res = coefficients
        mixed = torch.einsum(
            "...ij,...jh->...ih", res.expand(*streams.shape[:2], -1, -1) if res.dim() == 2 else res, streams
        )
        post = post.expand(*streams.shape[:2], -1) if post.dim() == 1 else post
        return mixed + post[..., None] * output[:, :, None]


# ------------------------------------------------------------------------------------------- the mixer --
class GatedMixer(nn.Module):
    """
    Strands L (+I) and R over the same normalized input, with a sigmoid gate on the attention half: the gate is
    computed from the block input, so a head that has nothing useful to say for a token can be switched off
    instead of dumping its mass on a sink token (Qwen3-Next's "gated attention").
    """

    def __init__(self, config: AnansiConfig, layer_idx: int) -> None:
        super().__init__()
        recurrent_only = config.layer_types[layer_idx] == "recurrent"
        self.braid = None if recurrent_only else HelixBraid(config, layer_idx)
        self.recurrent = HelixRecurrentStrand(config, layer_idx)
        # The gate's bias starts at +2 (gate ~0.88): attention begins almost fully open and closes where it learns to.
        self.gate = (
            nn.Linear(config.hidden_size, config.hidden_size)
            if config.gated_attention and not recurrent_only
            else None
        )

    def forward(self, hidden_states, position_embeddings, past_key_values=None, padding_mask=None):
        out = self.recurrent(hidden_states, past_key_values=past_key_values, padding_mask=padding_mask)
        if self.braid is None:
            return out
        attended = self.braid(
            hidden_states,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            padding_mask=padding_mask,
        )
        if self.gate is not None:
            attended = attended * torch.sigmoid(self.gate(hidden_states))
        return out + attended


# ------------------------------------------------------------------------------------- n-gram memory --
_HASH_PRIMES = (1_000_003, 999_983, 2_147_483_647, 97_003, 50_331_653)
_HASH_MASK = (1 << 62) - 1


def ngram_buckets(ids: torch.Tensor, order: int, size: int, prefix: torch.Tensor | None = None) -> torch.Tensor:
    """
    A bucket per position for the n-gram ending there: a multiplicative hash of the last ``order`` ids, with
    ``prefix`` (the ids before this chunk, for decoding) filling the window at the start. Positions whose window
    reaches before the start hash the padding id -1, so the first token of a text has its own bucket too.
    """
    batch, length = ids.shape
    pad = ids.new_full((batch, order - 1), -1)
    if prefix is not None and prefix.shape[1] > 0:
        pad = torch.cat([pad, prefix[:, -(order - 1) :]], dim=1)[:, -(order - 1) :] if order > 1 else pad
    window = torch.cat([pad, ids], dim=1) if order > 1 else ids
    acc = torch.zeros_like(ids)
    for j in range(order):
        shifted = window[:, j : j + length] if order > 1 else ids
        acc = (acc * _HASH_PRIMES[j % len(_HASH_PRIMES)] + (shifted + 2)) & _HASH_MASK
    return acc % size


class NgramMemory(nn.Module):
    """
    Hashed n-gram tables added to the token embeddings: for each order ``n`` the n-gram ending at a position
    picks one of ``size`` rows; the rows are concatenated and projected into the residual width. The projection
    starts at zero, so the memory begins silent and is used only where the gradient finds it useful. It gives
    every token a lookup of "what usually follows these few tokens" for the price of an embedding read, which is
    the kind of conditional memory DeepSeek's Engram (2026) found lets the attention layers spend themselves on
    reasoning instead of n-gram statistics.
    """

    def __init__(self, config: AnansiConfig) -> None:
        super().__init__()
        self.size, self.orders = config.ngram_memory_size, tuple(config.ngram_orders)
        self.tables = nn.ModuleList(nn.Embedding(self.size, config.ngram_dim) for _ in self.orders)
        self.proj = nn.Linear(config.ngram_dim * len(self.orders), config.hidden_size, bias=False)

    def forward(self, ids: torch.Tensor, prefix: torch.Tensor | None = None) -> torch.Tensor:
        rows = [
            table(ngram_buckets(ids, order, self.size, prefix))
            for table, order in zip(self.tables, self.orders, strict=True)
        ]
        return self.proj(torch.cat(rows, dim=-1))


# ------------------------------------------------------------------------------------------- experts --
class MixtureOfExperts(nn.Module):
    """
    Fine-grained routed experts plus a shared expert (DeepSeek-V3's recipe): sigmoid router scores, the top-k
    chosen on scores plus a per-expert bias that is nudged after every training step towards balanced load
    (no auxiliary loss, so the gradient only ever sees the language-modelling objective), the chosen experts'
    outputs weighted by the renormalized scores. The shared expert is a dense MLP every token goes through.
    """

    def __init__(self, config: AnansiConfig) -> None:
        super().__init__()
        self.config = config
        e, h, i = config.num_experts, config.hidden_size, config.expert_intermediate_size
        self.num_experts, self.top_k = e, config.experts_per_token
        self.router = nn.Linear(h, e, bias=False)
        self.register_buffer("balance_bias", torch.zeros(e), persistent=True)
        self.register_buffer("expert_load", torch.zeros(e), persistent=False)
        self.w_gate = nn.Parameter(torch.empty(e, h, i))
        self.w_up = nn.Parameter(torch.empty(e, h, i))
        self.w_down = nn.Parameter(torch.empty(e, i, h))
        self.act = ACT2FN[config.hidden_act]
        shared = config.num_shared_experts * i
        self.shared = HelixMLP(_Shape(config, shared)) if config.num_shared_experts else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        scores = torch.sigmoid(self.router(flat).float())
        chosen = (scores + self.balance_bias).topk(self.top_k, dim=-1).indices  # (N, k)
        weights = scores.gather(1, chosen)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        out = torch.zeros_like(flat)
        for e in range(self.num_experts):
            rows, slot = (chosen == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            h = flat[rows]
            y = (self.act(h @ self.w_gate[e]) * (h @ self.w_up[e])) @ self.w_down[e]
            out.index_add_(0, rows, y * weights[rows, slot, None].to(y.dtype))
        if self.shared is not None:
            out = out + self.shared(flat)
        if self.training and self.config.balance_bias_rate > 0:
            with torch.no_grad():
                load = torch.bincount(chosen.flatten(), minlength=self.num_experts).float()
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(load)  # balance the global load, not one rank's
                self.expert_load.copy_(load)
                self.balance_bias += self.config.balance_bias_rate * torch.sign(load.mean() - load)
        return out.reshape(shape)


class _Shape:
    """A view of the config with another intermediate size, for building the shared expert with HelixMLP."""

    def __init__(self, config: AnansiConfig, intermediate_size: int) -> None:
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.mlp_bias = config.mlp_bias
        self.hidden_act = config.hidden_act


# ---------------------------------------------------------------------------------------------- block --
class AnansiLayer(nn.Module):
    """One block: hyper-connections around (norm -> gated mixer) and (norm -> MLP or experts)."""

    def __init__(self, config: AnansiConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.hc_mixer = HyperConnections(config.hidden_size, config.num_streams, 2 * layer_idx)
        self.hc_mlp = HyperConnections(config.hidden_size, config.num_streams, 2 * layer_idx + 1)
        self.input_layernorm = HelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mixer = GatedMixer(config, layer_idx)
        self.post_attention_layernorm = HelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.is_moe = config.is_moe_layer(layer_idx)
        self.mlp = MixtureOfExperts(config) if self.is_moe else HelixMLP(config)
        # Sandwich norms: each sub-block's output is normalized before it is written into the streams, so a
        # looped core cannot drift in scale from pass to pass or learn to ignore its state (Huginn, Ouro).
        if config.sandwich_norm:
            self.mixer_out_norm = HelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mlp_out_norm = HelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.mixer_out_norm = self.mlp_out_norm = nn.Identity()

    def forward(self, streams, position_embeddings, past_key_values=None, padding_mask=None):
        x, coefficients = self.hc_mixer.read(streams)
        mixed = self.mixer(self.input_layernorm(x), position_embeddings, past_key_values, padding_mask)
        streams = self.hc_mixer.write(streams, self.mixer_out_norm(mixed), coefficients)
        x, coefficients = self.hc_mlp.read(streams)
        return self.hc_mlp.write(streams, self.mlp_out_norm(self.mlp(self.post_attention_layernorm(x))), coefficients)

    def residual_projections(self) -> list[nn.Parameter]:
        """The weights that write into the residual stream, for depth-scaled initialization."""
        params = [self.mixer.recurrent.out_proj.weight]
        if self.mixer.braid is not None:
            params.append(self.mixer.braid.o_proj.weight)
        if self.is_moe:
            params.append(self.mlp.w_down)
            if self.mlp.shared is not None:
                params.append(self.mlp.shared.down_proj.weight)
        else:
            params.append(self.mlp.down_proj.weight)
        return params


__all__ = [
    "AnansiLayer",
    "GatedMixer",
    "HyperConnections",
    "MixtureOfExperts",
    "NgramMemory",
    "ngram_buckets",
    "sinkhorn",
]
