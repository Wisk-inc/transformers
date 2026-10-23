# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
HELIX -- Hierarchical Episodic Linear IndeX.

A sequence architecture that keeps what attention is good at while dropping the quadratic bill. Every
block braids three mixers: exact multi-scale local attention, a gated delta-rule recurrence with a
matrix-valued state, and a hierarchical landmark index that can reach an exact token anywhere in the past.

    >>> from helix_lm import HelixConfig, HelixForCausalLM
    >>> model = HelixForCausalLM(HelixConfig(vocab_size=1000, hidden_size=256, num_hidden_layers=4))
    >>> import torch
    >>> model.generate(torch.zeros(1, 8, dtype=torch.long), max_new_tokens=4).shape
    torch.Size([1, 12])

Made by Nathan.
"""

from .anansi import ANANSI_SIZES, AnansiConfig, AnansiForCausalLM, describe_anansi_sizes
from .cache import HelixCache, HelixLayerCache
from .config import SIZES, HelixConfig, describe_sizes
from .model import (
    HelixBraid,
    HelixDecoderLayer,
    HelixForCausalLM,
    HelixLandmarkPooler,
    HelixModel,
    HelixOutput,
    HelixRecurrentStrand,
    kernels_available,
)
from .tokenizer import HelixTokenizer, train_tokenizer

__version__ = "2.3.2"
__author__ = "Nathan"

__all__ = [
    "ANANSI_SIZES",
    "AnansiConfig",
    "AnansiForCausalLM",
    "HelixBraid",
    "HelixCache",
    "HelixConfig",
    "HelixDecoderLayer",
    "HelixForCausalLM",
    "HelixLandmarkPooler",
    "HelixLayerCache",
    "HelixModel",
    "HelixOutput",
    "HelixRecurrentStrand",
    "HelixTokenizer",
    "SIZES",
    "__version__",
    "describe_anansi_sizes",
    "describe_sizes",
    "kernels_available",
    "train_tokenizer",
]
