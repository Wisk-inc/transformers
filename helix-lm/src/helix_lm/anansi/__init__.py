# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
ANANSI -- the spider that weaves: HELIX's three strands inside a new body.

Anansi keeps what HELIX proved (local attention, the gated delta-rule memory, the landmark index) and changes
how a network is built around them:

* **many residual streams** mixed by manifold-constrained hyper-connections instead of one residual sum;
* a **looped reasoning core**: a block of layers applied several times per token, with the prelude's output
  injected at every pass and per-pass adapters, so depth (thinking) is a dial at inference, not a fixed cost;
* **gated attention**, a sigmoid gate on strand L's output;
* a **hashed n-gram memory** that gives every token a cheap lookup of what usually follows its last few tokens;
* an optional **fine-grained mixture of experts** with a shared expert and loss-free load balancing in the core;
* a **multi-token prediction** module that trains the trunk to carry two steps of future at once.

See ``docs/ANANSI.md`` for the evidence behind each choice and what has and has not been measured.
"""

from .config import ANANSI_SIZES, AnansiConfig, describe_anansi_sizes
from .model import AnansiCache, AnansiForCausalLM, AnansiModel, AnansiOutput

__all__ = [
    "ANANSI_SIZES",
    "AnansiCache",
    "AnansiConfig",
    "AnansiForCausalLM",
    "AnansiModel",
    "AnansiOutput",
    "describe_anansi_sizes",
]
