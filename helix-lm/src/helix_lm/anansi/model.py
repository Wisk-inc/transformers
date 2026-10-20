# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The Anansi network: prelude, looped core, coda, over hyper-connected streams; with the n-gram memory at the
input and the multi-token predictor at the output.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..cache import HelixCache
from ..model import (
    HelixDecoderLayer,
    HelixForCausalLM,
    HelixOutput,
    HelixRMSNorm,
    HelixRotaryEmbedding,
    init_weights,
)
from .config import AnansiConfig
from .layers import AnansiLayer, HyperConnections, MixtureOfExperts, NgramMemory


# -------------------------------------------------------------------------------------------- caches --
class AnansiCache:
    """
    Decoding state: one HELIX cache for the prelude and coda, one per core pass (the same layers run several
    times per token, each pass with its own keys and recurrent state), and the last few token ids for the
    n-gram memory.
    """

    def __init__(self, config: AnansiConfig) -> None:
        self.config = config
        self.outer = HelixCache(config)
        self.loops: list[HelixCache] = []
        self.last_ids: torch.Tensor | None = None
        self.seen = 0

    def loop(self, index: int) -> HelixCache:
        while len(self.loops) <= index:
            self.loops.append(HelixCache(self.config))
        return self.loops[index]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seen

    def remember(self, ids: torch.Tensor, keep: int) -> None:
        tail = ids if self.last_ids is None else torch.cat([self.last_ids, ids], dim=1)
        self.last_ids = tail[:, -keep:]

    def byte_size(self) -> int:
        return self.outer.byte_size() + sum(c.byte_size() for c in self.loops)


@dataclass
class AnansiOutput(HelixOutput):
    """``loss`` is the training objective (next token + weighted multi-token term); ``nll`` the next-token loss."""

    nll: torch.Tensor | None = None
    mtp_loss: torch.Tensor | None = None
    loops: int = 0


# --------------------------------------------------------------------------------- multi-token predictor --
class MultiTokenPredictor(nn.Module):
    """
    DeepSeek-V3's sequential module for the token after next: the trunk's representation of position ``t`` and
    the embedding of token ``t+1`` are normalized, concatenated, projected, run through one more HELIX block and
    scored with the shared output head against token ``t+2``. The trunk therefore has to carry two steps of
    future at once, which densifies the training signal; at inference the module is a free speculative drafter.
    """

    def __init__(self, config: AnansiConfig) -> None:
        super().__init__()
        local = copy.copy(config)
        local.layer_types = [*config.layer_types, "helix_local"]
        local.num_hidden_layers = config.num_hidden_layers + 1
        self.norm_hidden = HelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_embed = HelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.merge = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.block = HelixDecoderLayer(local, config.num_hidden_layers)
        self.norm_out = HelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden: torch.Tensor, next_embeds: torch.Tensor, rotary: HelixRotaryEmbedding) -> torch.Tensor:
        """``hidden[:, :-1]`` with the embeddings of the following tokens -> states that predict two ahead."""
        merged = self.merge(torch.cat([self.norm_hidden(hidden[:, :-1]), self.norm_embed(next_embeds)], dim=-1))
        positions = torch.arange(merged.shape[1], device=merged.device)[None]
        states = self.block(merged, position_embeddings=rotary(merged, positions))
        return self.norm_out(states)


# ---------------------------------------------------------------------------------------------- model --
class AnansiModel(nn.Module):
    """The body: embeddings (+ n-gram memory) -> prelude -> core x loops -> coda -> norm."""

    def __init__(self, config: AnansiConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.ngram = NgramMemory(config) if config.ngram_memory_size else None
        self.layers = nn.ModuleList(AnansiLayer(config, i) for i in range(config.num_hidden_layers))
        self.loop_gain = nn.Parameter(torch.ones(config.max_loops, config.hidden_size))
        self.loop_inject = nn.Parameter(torch.ones(config.max_loops, config.hidden_size))
        self.readout = nn.Parameter(torch.zeros(config.num_streams))  # softmax weights that fold the streams
        self.norm = HelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = HelixRotaryEmbedding(config)
        self.register_buffer("forward_count", torch.zeros((), dtype=torch.long), persistent=False)
        self.apply(lambda module: anansi_init_weights(module, config))
        scale_residual_projections(self, config)

    # ------------------------------------------------------------------------------------- loop count --
    def sample_loops(self) -> int:
        """
        Training: ``min_loops + Poisson(mean - min)`` clipped to ``max_loops``, drawn from a counter-seeded
        generator so every data-parallel rank draws the same number (the collectives must agree). Otherwise
        the configured inference count.
        """
        c = self.config
        if not self.training:
            return c.loops_for_inference()
        self.forward_count += 1
        generator = torch.Generator(device="cpu").manual_seed(int(self.forward_count) * 1_000_003 + 12_345)
        extra = torch.poisson(torch.tensor([max(c.mean_loops - c.min_loops, 0.0)]), generator=generator)
        return int(min(c.max_loops, c.min_loops + int(extra.item())))

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: AnansiCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = False,
        num_loops: int | None = None,
        on_loop=None,
    ) -> tuple[torch.Tensor, AnansiCache | None, int]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids or inputs_embeds.")
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError("AnansiModel expects a 2D padding mask (batch, sequence).")
        config = self.config
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = AnansiCache(config)
        if self.ngram is not None and input_ids is not None:
            prefix = past_key_values.last_ids if past_key_values is not None else None
            inputs_embeds = inputs_embeds + self.ngram(input_ids, prefix)
        if past_key_values is not None and input_ids is not None:
            past_key_values.remember(input_ids, max(config.ngram_orders, default=1))
        if position_ids is None:
            seen = past_key_values.seen if past_key_values is not None else 0
            position_ids = torch.arange(seen, seen + inputs_embeds.shape[1], device=inputs_embeds.device)[None]
        rotary = self.rotary_emb(inputs_embeds, position_ids)
        recompute = config.activation_checkpointing and self.training and past_key_values is None

        def run(layer: AnansiLayer, streams: torch.Tensor, cache: HelixCache | None) -> torch.Tensor:
            if recompute and torch.is_grad_enabled():
                return checkpoint(layer, streams, rotary, None, attention_mask, use_reentrant=False)
            return layer(streams, rotary, cache, attention_mask)

        streams = inputs_embeds[:, :, None, :].repeat(1, 1, config.num_streams, 1)
        outer = past_key_values.outer if past_key_values is not None else None
        for i in range(config.prelude_layers):
            streams = run(self.layers[i], streams, outer)

        def finish(state: torch.Tensor) -> torch.Tensor:
            for i in range(config.prelude_layers + config.core_layers, config.num_hidden_layers):
                state = run(self.layers[i], state, outer)
            weights = torch.softmax(self.readout, dim=0).to(state.dtype)
            return self.norm(torch.einsum("n,btnh->bth", weights, state))

        loops = num_loops or self.sample_loops()
        injected = state = streams
        first_with_grad = loops - config.backprop_loops
        used = 0
        for r in range(loops):
            with torch.set_grad_enabled(torch.is_grad_enabled() and r >= first_with_grad):
                if r > 0 and config.loop_injection:
                    idx = min(r, config.max_loops - 1)
                    state = state * self.loop_gain[idx] + injected * self.loop_inject[idx]
                cache = past_key_values.loop(r) if past_key_values is not None else None
                for i in config.core_range:
                    state = run(self.layers[i], state, cache)
            used = r + 1
            # `on_loop(passes_so_far, finish)` may look at the would-be output and stop early (adaptive depth).
            if on_loop is not None and on_loop(used, lambda s=state: finish(s)):
                break

        hidden = finish(state)
        if past_key_values is not None:
            past_key_values.seen += inputs_embeds.shape[1]
        return hidden, past_key_values, used


class AnansiForCausalLM(HelixForCausalLM):
    """Anansi with the output head and the multi-token predictor. Saves and loads like a HELIX model."""

    def __init__(self, config: AnansiConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.model = AnansiModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp = MultiTokenPredictor(config) if config.mtp_depth else None
        init_weights(self.lm_head, config)
        if self.mtp is not None:
            self.mtp.apply(lambda module: anansi_init_weights(module, config))
            scale_residual_projections(self, config)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def new_cache(self) -> AnansiCache:
        return AnansiCache(self.config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: AnansiCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool = False,
        num_loops: int | None = None,
    ) -> AnansiOutput:
        hidden, cache, loops = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            num_loops=num_loops,
        )
        config = self.config
        scaled = hidden * config.output_multiplier if config.output_multiplier != 1.0 else hidden
        mtp_states = None
        if labels is not None and self.mtp is not None and input_ids is not None and input_ids.shape[1] > 2:
            next_embeds = self.model.embed_tokens(input_ids[:, 1:])
            mtp_states = self.mtp(hidden, next_embeds, self.model.rotary_emb)
            if config.output_multiplier != 1.0:
                mtp_states = mtp_states * config.output_multiplier

        if labels is not None and config.loss_chunk_tokens > 0:
            nll, z_loss = self._chunked_loss(scaled, labels)
            mtp_loss = self._chunked_loss(mtp_states, labels[:, 1:])[0] if mtp_states is not None else None
            logits = None
        else:
            logits = self.lm_head(scaled)
            nll = z_loss = mtp_loss = None
            if labels is not None:
                nll, z_loss = self._plain_loss(logits, labels)
                if mtp_states is not None:
                    mtp_loss = self._plain_loss(self.lm_head(mtp_states), labels[:, 1:])[0]
        loss = nll
        if loss is not None and mtp_loss is not None and self.training and config.mtp_weight > 0:
            loss = loss + config.mtp_weight * mtp_loss
        return AnansiOutput(
            logits=logits, loss=loss, past_key_values=cache, hidden_states=hidden, z_loss=z_loss,
            nll=nll, mtp_loss=mtp_loss, loops=loops,
        )  # fmt: skip

    def _plain_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        shifted = logits[:, :-1].reshape(-1, logits.shape[-1]).float()
        targets = labels[:, 1:].reshape(-1)
        loss = F.cross_entropy(shifted, targets, ignore_index=-100)
        z_loss = None
        if self.config.z_loss_weight > 0:
            kept = targets != -100
            z_loss = torch.logsumexp(shifted[kept], dim=-1).pow(2).mean()
            loss = loss + self.config.z_loss_weight * z_loss
        return loss, z_loss

    @torch.no_grad()
    def draft(self, input_ids: torch.Tensor, num_loops: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The next token and the one after it from a single forward pass (next-token head, then the multi-token
        predictor fed with that token): the drafts a speculative decoder verifies. Greedy.
        """
        self.eval()
        hidden, _, _ = self.model(input_ids=input_ids, num_loops=num_loops)
        first = self.lm_head(hidden[:, -1] * self.config.output_multiplier).argmax(-1)
        if self.mtp is None:
            return first, first
        extended = torch.cat([input_ids, first[:, None]], dim=1)
        states = self.mtp(
            hidden[:, -2:].repeat(1, 1, 1) if hidden.shape[1] >= 2 else hidden,
            self.model.embed_tokens(extended[:, -1:]),
            self.model.rotary_emb,
        )
        second = self.lm_head(states[:, -1] * self.config.output_multiplier).argmax(-1)
        return first, second

    @torch.no_grad()
    def think(
        self, input_ids: torch.Tensor, max_loops: int | None = None, kl_threshold: float = 5e-4
    ) -> tuple[torch.Tensor, int]:
        """
        Adaptive depth: run the core one pass at a time and stop when the next-token distribution at the last
        position stops changing (KL between successive passes below ``kl_threshold``, Huginn's exit rule).
        Returns ``(logits (B, T, V), passes_used)``. Easy continuations exit early; hard ones get the budget.
        """
        self.eval()
        max_loops = max_loops or self.config.max_loops
        previous = None

        def on_loop(passes: int, finish) -> bool:
            nonlocal previous
            logits = self.lm_head(finish()[:, -1] * self.config.output_multiplier).float().log_softmax(-1)
            stop = False
            if previous is not None:
                kl = (logits.exp() * (logits - previous)).sum(-1).mean().item()
                stop = kl < kl_threshold
            previous = logits
            return stop

        hidden, _, used = self.model(input_ids=input_ids, num_loops=max_loops, on_loop=on_loop)
        return self.lm_head(hidden * self.config.output_multiplier), used

    @classmethod
    def from_pretrained(cls, directory: str | Path, **overrides) -> AnansiForCausalLM:
        directory = Path(directory)
        config = AnansiConfig.from_dict({**json.loads((directory / "config.json").read_text()), **overrides})
        model = cls(config)
        tensors = cls.load_state_dict_from(directory)
        if config.tie_word_embeddings:
            tensors.setdefault("lm_head.weight", tensors["model.embed_tokens.weight"])
        model.load_state_dict(tensors)
        return model


# ----------------------------------------------------------------------------------- initialization --
@torch.no_grad()
def anansi_init_weights(module: nn.Module, config: AnansiConfig) -> None:
    """HELIX's init for the strands, norms and linear maps; the Anansi parts after their children (``apply`` order)."""
    if isinstance(module, HyperConnections):
        module.reset_parameters()
    elif isinstance(module, MixtureOfExperts):
        module.w_gate.normal_(mean=0.0, std=config.initializer_range)
        module.w_up.normal_(mean=0.0, std=config.initializer_range)
        module.w_down.normal_(mean=0.0, std=config.initializer_range)
        module.balance_bias.zero_()
        module.expert_load.zero_()
    elif isinstance(module, NgramMemory):
        module.proj.weight.zero_()
    elif isinstance(module, AnansiModel):
        module.loop_gain.fill_(1.0)
        module.loop_inject.fill_(1.0)
        module.readout.zero_()
        module.forward_count.zero_()
    else:
        init_weights(module, config)


@torch.no_grad()
def scale_residual_projections(model: nn.Module, config: AnansiConfig) -> None:
    """Every weight that writes into the streams gets ``std / sqrt(2 * layers)`` (see HELIX's rationale)."""
    if not config.scale_residual_init:
        return
    std = config.initializer_range / (2 * config.num_hidden_layers) ** 0.5
    body = model.model if isinstance(model, AnansiForCausalLM) else model
    for layer in body.layers:
        for param in layer.residual_projections():
            param.normal_(mean=0.0, std=std)
    mtp = getattr(model, "mtp", None)
    if mtp is not None:
        for projection in (mtp.block.mixer.o_proj, mtp.block.recurrent.out_proj, mtp.block.mlp.down_proj):
            projection.weight.normal_(mean=0.0, std=std)


# ------------------------------------------------------------------------------------------- counting --
def _meta_model(config: AnansiConfig) -> AnansiForCausalLM:
    with torch.device("meta"):
        return AnansiForCausalLM(config)


def count_parameters(config: AnansiConfig, embeddings: bool = True) -> int:
    model = _meta_model(config)
    total = sum(p.numel() for p in model.parameters())
    if not embeddings:
        total -= model.model.embed_tokens.weight.numel()
        if not config.tie_word_embeddings:
            total -= model.lm_head.weight.numel()
        if model.model.ngram is not None:
            total -= sum(t.weight.numel() for t in model.model.ngram.tables)
    return int(total)


def active_parameters(config: AnansiConfig) -> int:
    """Parameters a token's forward pass multiplies by, with the core counted ``mean_loops`` times and only the
    chosen experts of a mixture: what ``6N`` FLOPs should be computed from."""
    model = _meta_model(config)
    total = 0.0
    for i, layer in enumerate(model.model.layers):
        n = sum(p.numel() for p in layer.parameters())
        if layer.is_moe:
            routed = layer.mlp.w_gate.numel() + layer.mlp.w_up.numel() + layer.mlp.w_down.numel()
            n -= routed * (1 - config.experts_per_token / config.num_experts)
        total += n * (config.mean_loops if i in config.core_range else 1.0)
    total += model.lm_head.weight.numel() + model.model.norm.weight.numel()
    total += model.model.loop_gain.numel() + model.model.loop_inject.numel() + model.model.readout.numel()
    if model.model.ngram is not None:
        total += model.model.ngram.proj.weight.numel()
    if model.mtp is not None:
        total += sum(p.numel() for p in model.mtp.parameters())
    return int(total)
