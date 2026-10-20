# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Configuration for HELIX, and the named sizes the training ladder is built from."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# The named sizes (34b is CorX5.5). Every one shares the HELIX geometry below (block of 128 tokens, an exact
# local window of up to 1,024 tokens on four scales, the index on every other layer reading 16 blocks, a
# 128-wide matrix state per recurrent head) and differs only in width and depth. Parameter counts are from
# ``HelixConfig.num_parameters`` with the 65,536-token CorxLabs vocabulary; run ``helix size`` to print them.
_SIZE_COLUMNS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "num_recurrent_heads",
)
_SIZE_ROWS = {
    #       hidden  layers  heads  kv  head_dim  intermediate  recurrent heads
    "20m": (256, 4, 4, 2, 64, 704, 2),
    "350m": (1024, 16, 16, 4, 64, 2816, 4),
    "1b": (2048, 16, 16, 4, 128, 5632, 8),
    "3b": (2560, 28, 20, 4, 128, 6912, 10),
    "7b": (4096, 32, 32, 8, 128, 11008, 16),
    "23b": (6144, 48, 48, 8, 128, 15360, 24),
    "34b": (6144, 64, 48, 8, 128, 17920, 24),
}
SIZES: dict[str, dict] = {name: dict(zip(_SIZE_COLUMNS, row, strict=True)) for name, row in _SIZE_ROWS.items()}

# What every named size shares. ``vocab_size`` is the CorxLabs tokenizer's; it fits in uint16, which is what
# keeps 460 billion tokens of training data under a terabyte on disk.
LADDER_DEFAULTS: dict = {
    "vocab_size": 65536,
    "block_size": 128,
    "local_blocks": 8,
    "num_window_scales": 4,
    "index_layer_stride": 2,
    "landmark_dim": 128,
    "index_branching": 16,
    "index_beam_width": 4,
    "index_topk": 16,
    "index_max_levels": 8,
    "recurrent_head_dim": 128,
    "recurrent_value_head_dim": 128,
    "recurrent_chunk_size": 64,
    "conv_kernel_size": 4,
    "rope_theta": 1_000_000.0,
    "max_position_embeddings": 1_048_576,
    "tie_word_embeddings": False,
    "bos_token_id": 1,
    "eos_token_id": 0,
    "pad_token_id": 2,
}


@dataclass
class HelixConfig:
    """
    Configuration for a HELIX model.

    Every block braids three sequence mixers over the same residual stream:

    * **strand L** — exact softmax attention over a short, multi-scale sliding window, run on a block
      staircase so compute and activation memory are ``O(N * local_span)``;
    * **strand R** — a gated delta-rule recurrence with a matrix-valued state, trained with the
      chunkwise-parallel (UT transform) form and decoded from a fixed-size state;
    * **strand I** — a hierarchical landmark index over the whole past. Each query *block* descends an
      ``index_branching``-ary tree of pooled summaries with beam width ``index_beam_width``, selects
      ``index_topk`` leaf blocks, and attends exactly over them.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Width of the residual stream.
        intermediate_size: Width of the SwiGLU MLP.
        num_hidden_layers: Number of HELIX blocks.
        num_attention_heads: Query heads shared by strands L and I.
        num_key_value_heads: Key/value (and index-routing) head groups.
        head_dim: Per-head dimension of strands L and I. Defaults to ``hidden_size // num_attention_heads``.
        block_size: Memory block granularity — the unit of landmark pooling, index selection, the local
            staircase, and index routing.
        local_blocks: Number of *previous* memory blocks visible to strand L. The widest per-head window is
            ``local_blocks * block_size`` tokens.
        num_window_scales: Query heads split into this many contiguous groups with geometrically increasing
            windows. This is the multi-scale locality prior; 1 gives every head the same window.
        index_layer_stride: Strand I runs on every ``index_layer_stride``-th layer. The others run L + R and
            keep a window-capped key/value cache instead of a full one.
        landmark_dim: Dimension of the landmark/route space.
        index_branching: Fan-out of the landmark tree.
        index_beam_width: Nodes kept at each level of the beam descent.
        index_topk: Leaf memory blocks each query block attends to.
        index_max_levels: Number of distinct learned per-level biases. The tree itself always grows to
            whatever depth the memory needs.
        index_num_distance_buckets: Logarithmic relative *block* distance buckets biasing the descent.
        attention_tile_blocks: Query blocks attended at once. Caps peak activation memory independently of
            context length; 0 processes the whole sequence in one tile. No effect on the result.
        attention_backend: How strand L is computed on an uncached, unpadded forward -- the training case.
            ``"flex"`` is FlexAttention, a fused block-sparse kernel that reads each head's window directly
            and never copies keys; ``"sdpa"`` is the gathered-block reference; ``"auto"`` picks flex on a
            GPU and the reference elsewhere. Both give the same result (``tests/test_engine.py``), and
            decoding always uses the reference path.
        recurrent_backend: How strand R's chunked delta rule is computed: ``"fla"`` is the Triton kernel from
            ``flash-linear-attention``, ``"torch"`` the reference, ``"auto"`` fla when it is installed and
            the tensors are on a GPU.
        activation_checkpointing: Recompute each block's activations in the backward pass. Trades about a
            third more compute for activation memory that no longer grows with depth.
        scale_residual_init: Initialize the three projections that write into the residual stream with
            ``initializer_range / sqrt(2 * num_hidden_layers)``, so the stream's variance does not grow with
            depth at initialization.
        loss_chunk_tokens: With labels, compute the cross-entropy over this many tokens at a time and never
            hold the whole ``(tokens, vocab)`` logit tensor; 0 computes it in one piece and returns logits.
        z_loss_weight: Weight of the ``log(sum(exp(logits)))^2`` penalty that keeps logits from drifting;
            0 disables it.
        num_recurrent_heads: Heads of the delta-rule recurrence.
        recurrent_head_dim: Key dimension of the recurrent matrix state.
        recurrent_value_head_dim: Value dimension of the recurrent matrix state.
        recurrent_chunk_size: Chunk length of the chunkwise-parallel delta rule.
        conv_kernel_size: Short depthwise causal convolution on the recurrent q/k/v stream.
        use_surprise_gating: Modulate delta-rule write strength by how novel the current key is against a
            short causal pool of the keys before it.
        surprise_kernel_size: Width of the causal pool driving the surprise signal.
        hidden_act: MLP activation (``silu`` or ``gelu``).
        mlp_bias: Whether the MLP projections carry a bias.
        max_position_embeddings: Longest sequence the model is expected to handle.
        rope_theta: RoPE base.
        initializer_range: Standard deviation of the normal initializer.
        rms_norm_eps: Epsilon of the RMS norms.
        tie_word_embeddings: Whether to tie input and output embeddings.
        pad_token_id / bos_token_id / eos_token_id: Special token ids.
        layer_types: Per-layer schedule of ``"helix"`` (L+R+I) or ``"helix_local"`` (L+R). Derived from
            ``index_layer_stride`` when omitted.
        model_name / organization: Who this checkpoint is, for the model card and the identity data.
    """

    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int | None = None

    block_size: int = 64
    local_blocks: int = 4
    num_window_scales: int = 4

    index_layer_stride: int = 3
    landmark_dim: int = 64
    index_branching: int = 8
    index_beam_width: int = 4
    index_topk: int = 8
    index_max_levels: int = 8
    index_num_distance_buckets: int = 32
    attention_tile_blocks: int = 64
    attention_backend: str = "auto"
    recurrent_backend: str = "auto"
    activation_checkpointing: bool = False
    scale_residual_init: bool = True
    loss_chunk_tokens: int = 0
    z_loss_weight: float = 0.0

    num_recurrent_heads: int = 8
    recurrent_head_dim: int = 128
    recurrent_value_head_dim: int = 128
    recurrent_chunk_size: int = 64
    conv_kernel_size: int = 4
    use_surprise_gating: bool = True
    surprise_kernel_size: int = 8

    hidden_act: str = "silu"
    mlp_bias: bool = False
    max_position_embeddings: int = 1048576
    rope_theta: float = 500000.0
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False

    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | None = 2

    layer_types: list[str] = field(default_factory=list)
    model_name: str = "helix"
    organization: str = "CorxLabs"

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})."
            )
        if not 1 <= self.num_window_scales <= self.num_attention_heads:
            raise ValueError(
                f"num_window_scales must be in [1, num_attention_heads]; got {self.num_window_scales} "
                f"for {self.num_attention_heads} heads."
            )
        if self.index_branching < 2:
            raise ValueError(f"index_branching must be >= 2, got {self.index_branching}.")
        if self.index_layer_stride < 1:
            raise ValueError(f"index_layer_stride must be >= 1, got {self.index_layer_stride}.")
        if self.attention_backend not in ("auto", "flex", "sdpa"):
            raise ValueError(f"attention_backend must be 'auto', 'flex' or 'sdpa', got {self.attention_backend!r}.")
        if self.recurrent_backend not in ("auto", "fla", "torch"):
            raise ValueError(f"recurrent_backend must be 'auto', 'fla' or 'torch', got {self.recurrent_backend!r}.")

        if not self.layer_types:
            self.layer_types = [
                "helix" if (i + 1) % self.index_layer_stride == 0 else "helix_local"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but num_hidden_layers is {self.num_hidden_layers}."
            )
        unknown = set(self.layer_types) - {"helix", "helix_local"}
        if unknown:
            raise ValueError(f"layer_types entries must be 'helix' or 'helix_local'; got {sorted(unknown)}.")

        for name, minimum in (
            ("block_size", 1),
            ("local_blocks", 0),
            ("index_topk", 1),
            ("index_beam_width", 1),
            ("landmark_dim", 1),
            ("index_num_distance_buckets", 1),
            ("num_recurrent_heads", 1),
            ("recurrent_head_dim", 1),
            ("recurrent_value_head_dim", 1),
            ("recurrent_chunk_size", 1),
            ("conv_kernel_size", 1),
            # The surprise pool averages the `surprise_kernel_size - 1` positions before the current one,
            # so a width of 1 would leave it with nothing to average.
            ("surprise_kernel_size", 2),
            ("loss_chunk_tokens", 0),
        ):
            if getattr(self, name) < minimum:
                raise ValueError(f"`{name}` must be >= {minimum}, got {getattr(self, name)}.")
        if self.z_loss_weight < 0:
            raise ValueError(f"z_loss_weight must be >= 0, got {self.z_loss_weight}.")
        self.sliding_window = (self.local_blocks + 1) * self.block_size
        self.number_of_conv_states = 2 if self.use_surprise_gating else 1

    # ------------------------------------------------------------------------------------------ sizes --
    @classmethod
    def named(cls, size: str, **overrides) -> HelixConfig:
        """One of the ladder sizes (``helix size`` lists them), with any field overridden."""
        key = size.lower()
        if key not in SIZES:
            raise ValueError(f"unknown size {size!r}; choose from {', '.join(SIZES)}")
        values = {**LADDER_DEFAULTS, **SIZES[key], "model_name": f"helix-{key}", **overrides}
        return cls(**values)

    @property
    def local_span(self) -> int:
        """Widest per-head local window, in tokens."""
        return self.local_blocks * self.block_size

    @property
    def local_window_sizes(self) -> list[int]:
        """Per-head window length, one entry per query head, narrowest group first."""
        # Floored at `block_size`: strand I only reads blocks ending strictly before the query's own
        # block, so the narrowest window must still span a full block or the two would leave a gap.
        heads_per_scale = self.num_attention_heads / self.num_window_scales
        return [
            max(self.block_size, self.local_span >> (self.num_window_scales - 1 - int(head // heads_per_scale)))
            for head in range(self.num_attention_heads)
        ]

    @property
    def num_index_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "helix")

    def index_num_levels(self, num_blocks: int) -> int:
        """Landmark-tree levels built *above* the leaves for a memory of ``num_blocks`` blocks."""
        if num_blocks < self.index_branching:
            return 0
        return int(math.log(num_blocks, self.index_branching))

    def num_parameters(self, embeddings: bool = True) -> int:
        """
        Parameter count from the shapes alone, so a 23B configuration can be sized without building it.
        ``tests/test_engine.py`` checks it against the built model.
        """
        h, hd, H, KV = self.hidden_size, self.head_dim, self.num_attention_heads, self.num_key_value_heads
        ld, nh, dk, dv = (
            self.landmark_dim,
            self.num_recurrent_heads,
            self.recurrent_head_dim,
            self.recurrent_value_head_dim,
        )
        key_dim, value_dim = nh * dk, nh * dv
        conv_dim = 2 * key_dim + value_dim

        braid = 2 * H * hd * h + 2 * KV * hd * h + 2 * hd  # q, o, k, v, q_norm, k_norm
        index = (
            (hd + 2 * hd * ld + ld)  # leaf pooler: query, proj, norm
            + (ld + 2 * ld * ld + ld)  # node pooler
            + h * KV * ld  # route_q_proj
            + KV * (self.index_max_levels + 1)  # level_bias
            + KV * self.index_num_distance_buckets  # distance_bias
            + h * H * 2  # strand_gate
        )
        recurrent = (
            h * (conv_dim + value_dim)  # in_proj_qkvz
            + h * 2 * nh  # in_proj_ba
            + conv_dim * self.conv_kernel_size  # depthwise conv
            + 3 * nh  # A_log, dt_bias, surprise_gamma
            + dv  # gated norm
            + value_dim * h  # out_proj
        )
        mlp = 3 * h * self.intermediate_size + (2 * self.intermediate_size + h if self.mlp_bias else 0)
        per_layer = braid + recurrent + mlp + 2 * h
        total = per_layer * self.num_hidden_layers + index * self.num_index_layers + h  # + final norm
        if embeddings:
            total += self.vocab_size * h * (1 if self.tie_word_embeddings else 2)
        return int(total)

    def flops_per_token(self) -> float:
        """
        Training FLOPs per token (forward + backward), for throughput and MFU reporting: ``6 N`` over the
        non-embedding weights plus the attention and recurrence terms, which depend on the window sizes
        rather than on the context length. An estimate, not a measurement.
        """
        h, hd, H = self.hidden_size, self.head_dim, self.num_attention_heads
        weights = 6.0 * self.num_parameters(embeddings=False)
        lm_head = 6.0 * self.vocab_size * h
        mean_window = sum(self.local_window_sizes) / H
        local = 12.0 * H * hd * mean_window * self.num_hidden_layers
        index = 12.0 * H * hd * self.index_topk * self.block_size * self.num_index_layers
        nh, dk, dv, c = (
            self.num_recurrent_heads,
            self.recurrent_head_dim,
            self.recurrent_value_head_dim,
            self.recurrent_chunk_size,
        )
        recurrent = 12.0 * nh * (dk * dv + c * (dk + dv)) * self.num_hidden_layers
        return weights + lm_head + local + index + recurrent

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> HelixConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})

    def save_pretrained(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "config.json"
        path.write_text(json.dumps({"model_type": "helix", **self.to_dict()}, indent=2) + "\n")
        return path

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> HelixConfig:
        return cls.from_dict(json.loads((Path(directory) / "config.json").read_text()))


def describe_sizes() -> str:
    """The ladder, one line a size: parameters, and what 20 tokens a parameter (Chinchilla) means."""
    widths = (
        ("size", 6),
        ("params", 8),
        ("non-embed", 10),
        ("layers", 6),
        ("hidden", 6),
        ("heads", 5),
        ("tokens x20", 10),
    )
    rows = ["  ".join(f"{name:>{width}}" for name, width in widths)]
    for name in SIZES:
        config = HelixConfig.named(name)
        total, body = config.num_parameters(), config.num_parameters(embeddings=False)
        rows.append(
            f"{name:>6}  {total / 1e9:7.2f}B  {body / 1e9:9.2f}B  {config.num_hidden_layers:>6}  "
            f"{config.hidden_size:>6}  {config.num_attention_heads:>5}  {20 * total / 1e9:8.0f}B"
        )
    return "\n".join(rows)
