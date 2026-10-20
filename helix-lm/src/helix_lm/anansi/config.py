# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The Anansi configuration: a HELIX configuration plus the body around the strands."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import LADDER_DEFAULTS, HelixConfig

# hidden, prelude, core, coda, heads, kv heads, head_dim, mlp, recurrent heads, streams, mean loops, max loops
_A_COLUMNS = (
    "hidden_size",
    "prelude_layers",
    "core_layers",
    "coda_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "num_recurrent_heads",
    "num_streams",
    "mean_loops",
    "max_loops",
)
_A_ROWS = {
    #         hidden pre core coda heads kv  hd   mlp    rh  streams loops max
    "a-20m": (256, 1, 2, 1, 4, 2, 64, 704, 2, 2, 2.0, 4),
    "a-350m": (1024, 2, 6, 2, 16, 4, 64, 2816, 8, 4, 3.0, 6),
    "a-1b": (2048, 2, 8, 2, 16, 4, 128, 5632, 8, 4, 3.0, 6),
    "a-3b": (2560, 3, 10, 3, 20, 4, 128, 6912, 12, 4, 3.0, 6),
    "a-7b": (4096, 4, 12, 4, 32, 8, 128, 11008, 16, 4, 3.0, 6),
    "a-24b": (6144, 8, 24, 8, 48, 8, 128, 17920, 24, 4, 2.0, 4),
    "a-34b": (6144, 8, 40, 8, 48, 8, 128, 17920, 24, 4, 1.5, 3),
}
ANANSI_SIZES: dict[str, dict] = {name: dict(zip(_A_COLUMNS, row, strict=True)) for name, row in _A_ROWS.items()}


@dataclass
class AnansiConfig(HelixConfig):
    """
    Everything :class:`~helix_lm.config.HelixConfig` has, and:

    Args:
        num_streams: Residual streams mixed by the hyper-connections. 1 is an ordinary residual network.
        prelude_layers / core_layers / coda_layers: The layer layout. The prelude reads the input once, the
            core is applied ``loops`` times, the coda reads the result once. ``num_hidden_layers`` is their sum
            (distinct parameter layers); the effective depth is ``prelude + loops * core + coda``.
        mean_loops / min_loops / max_loops: How many times the core runs. Training samples the count around
            the mean so the model works at every depth; evaluation and decoding use ``decode_loops`` (the
            rounded mean when 0). Raising it at inference buys more computation per token.
        backprop_loops: Gradients flow through only the last this-many passes (truncated backpropagation,
            as in recurrent-depth models), which keeps activation memory independent of the loop count.
        loop_injection: Add the prelude's output at the start of every pass, so a pass can never forget the
            input; with per-pass gain and bias vectors (the iteration-conditioned adapters).
        ngram_memory_size: Buckets of the hashed n-gram memory added to the input embeddings; 0 disables it.
        ngram_orders: Which n-gram orders are hashed.
        ngram_dim: Width of each order's table; 0 picks ``hidden_size // 4``.
        num_experts: Routed experts in the core's MLPs; 0 keeps dense MLPs everywhere.
        num_shared_experts: Experts every token always uses (a dense MLP of that many expert widths).
        experts_per_token: Routed experts a token is sent to.
        expert_intermediate_size: Width of one expert; 0 picks ``intermediate_size // experts_per_token`` so
            the active MLP compute per token matches the dense model.
        balance_bias_rate: Step of the loss-free load-balancing bias (DeepSeek-V3); 0 freezes it.
        gated_attention: Sigmoid gate (from the block input) on strand L's output.
        mtp_depth: Multi-token-prediction modules (0 or 1 for now).
        mtp_weight: Weight of the MTP loss added to the next-token loss.
        output_multiplier: Scale on the final hidden states before the output head (muP's output rule).
        decode_loops: Loop count used when not training; 0 means the rounded mean.
        sandwich_norm: Normalize each sub-block's output before it is written into the streams as well as its
            input (Huginn's and Ouro's fix for looped cores that learn to ignore their state).
        prelude_attention: Keep strand L in the prelude; off, the prelude is recurrence + MLP only.
    """

    num_streams: int = 4
    prelude_layers: int = 2
    core_layers: int = 8
    coda_layers: int = 2
    mean_loops: float = 3.0
    min_loops: int = 1
    max_loops: int = 6
    backprop_loops: int = 3
    loop_injection: bool = True
    ngram_memory_size: int = 65536
    ngram_orders: tuple[int, ...] = (2, 3)
    ngram_dim: int = 0
    num_experts: int = 0
    num_shared_experts: int = 1
    experts_per_token: int = 2
    expert_intermediate_size: int = 0
    balance_bias_rate: float = 1e-3
    gated_attention: bool = True
    mtp_depth: int = 1
    mtp_weight: float = 0.1
    sandwich_norm: bool = True
    prelude_attention: bool = False
    output_multiplier: float = 1.0
    decode_loops: int = 0
    model_name: str = "anansi"
    layer_types: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        layout = self.prelude_layers + self.core_layers + self.coda_layers
        if self.num_hidden_layers not in (layout, HelixConfig.num_hidden_layers):
            raise ValueError(
                f"num_hidden_layers ({self.num_hidden_layers}) must equal prelude + core + coda ({layout})"
            )
        self.num_hidden_layers = layout
        if self.core_layers < 1 or self.prelude_layers < 0 or self.coda_layers < 0:
            raise ValueError("core_layers must be >= 1 and prelude/coda >= 0")
        if not 1 <= self.min_loops <= self.max_loops:
            raise ValueError(f"need 1 <= min_loops ({self.min_loops}) <= max_loops ({self.max_loops})")
        if not self.min_loops <= self.mean_loops <= self.max_loops:
            raise ValueError(f"mean_loops ({self.mean_loops}) must lie in [min_loops, max_loops]")
        if self.num_streams < 1:
            raise ValueError("num_streams must be >= 1")
        if self.num_experts and not 1 <= self.experts_per_token <= self.num_experts:
            raise ValueError("experts_per_token must be in [1, num_experts]")
        if self.mtp_depth not in (0, 1):
            raise ValueError("mtp_depth is 0 or 1")
        self.ngram_orders = tuple(int(n) for n in self.ngram_orders)
        if self.ngram_dim == 0:
            self.ngram_dim = max(16, self.hidden_size // 4)
        if self.num_experts and self.expert_intermediate_size == 0:
            self.expert_intermediate_size = max(8, self.intermediate_size // self.experts_per_token)
        # The prelude runs the recurrence only ("recurrent" layers): attention at the very front of a network
        # learns uniform maps (Meta/KAIST hybrid study, 2025), so the first layers are better spent on the delta
        # memory and the n-gram lookup. HelixConfig validates layer types, so the names pass through as local.
        recurrent = [i for i, t in enumerate(self.layer_types) if t == "recurrent"]
        self.layer_types = ["helix_local" if t == "recurrent" else t for t in self.layer_types]
        super().__post_init__()
        if not self.prelude_attention:
            recurrent = sorted(set(recurrent) | set(range(self.prelude_layers)))
        for i in recurrent:
            self.layer_types[i] = "recurrent"

    @classmethod
    def named(cls, size: str, **overrides) -> AnansiConfig:
        key = size.lower()
        if key not in ANANSI_SIZES:
            raise ValueError(f"unknown Anansi size {size!r}; choose from {', '.join(ANANSI_SIZES)}")
        values = {**LADDER_DEFAULTS, **ANANSI_SIZES[key], "model_name": key, **overrides}
        values.pop("num_hidden_layers", None)
        return cls(**values)

    def save_pretrained(self, directory):
        import json
        from pathlib import Path

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "config.json"
        path.write_text(json.dumps({"model_type": "anansi", **self.to_dict()}, indent=2) + "\n")
        return path

    # ----------------------------------------------------------------------------------------- layout --
    @property
    def core_range(self) -> range:
        return range(self.prelude_layers, self.prelude_layers + self.core_layers)

    @property
    def effective_depth(self) -> float:
        return self.prelude_layers + self.mean_loops * self.core_layers + self.coda_layers

    def loops_for_inference(self) -> int:
        return self.decode_loops or max(self.min_loops, min(self.max_loops, round(self.mean_loops)))

    def is_moe_layer(self, layer_idx: int) -> bool:
        return bool(self.num_experts) and layer_idx in self.core_range

    # ------------------------------------------------------------------------------------------ costs --
    def num_parameters(self, embeddings: bool = True) -> int:
        """Counted on the meta device: Anansi has too many optional parts for a closed form to stay honest."""
        from .model import count_parameters

        return count_parameters(self, embeddings=embeddings)

    def flops_per_token(self, seq_len: int | None = None) -> float:
        """Training FLOPs per token: ``6 x`` the parameters a token actually touches, plus the attention terms."""
        from .model import active_parameters

        touched = active_parameters(self)
        attention = 0.0
        for layer_idx in range(self.num_hidden_layers):
            passes = self.mean_loops if layer_idx in self.core_range else 1.0
            span = min(seq_len or self.sliding_window, self.sliding_window)
            attention += passes * 12.0 * self.num_attention_heads * self.head_dim * span
            if self.layer_types[layer_idx] == "helix":
                attention += (
                    passes * 12.0 * self.num_attention_heads * self.head_dim * self.index_topk * self.block_size
                )
        return 6.0 * touched + attention


def describe_anansi_sizes() -> str:
    widths = (
        ("size", 7),
        ("params", 8),
        ("active", 8),
        ("layers", 6),
        ("depth", 6),
        ("hidden", 6),
        ("streams", 7),
        ("TFLOP/tok", 9),
    )
    rows = ["  ".join(f"{name:>{width}}" for name, width in widths)]
    from .model import active_parameters

    for name in ANANSI_SIZES:
        config = AnansiConfig.named(name)
        total, active = config.num_parameters(), active_parameters(config)
        rows.append(
            f"{name:>7}  {total / 1e9:7.2f}B  {active / 1e9:7.2f}B  {config.num_hidden_layers:>6}  "
            f"{config.effective_depth:>6.1f}  {config.hidden_size:>6}  {config.num_streams:>7}  "
            f"{config.flops_per_token(8192) / 1e12:>9.3f}"
        )
    return "\n".join(rows)
