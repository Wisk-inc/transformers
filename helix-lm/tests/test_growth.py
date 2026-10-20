# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Growth: a grown model computes the same function as the small one (identity mode), or a stacked copy of it."""

import json

import pytest
import torch

from helix_lm.anansi import AnansiConfig, AnansiForCausalLM
from helix_lm.config import HelixConfig
from helix_lm.growth import build, grow, grow_depth, grow_export, grow_mlp, grow_streams, load_export, plan_from_sizes
from helix_lm.model import HelixForCausalLM

VOCAB = 256


def _helix():
    torch.manual_seed(0)
    config = HelixConfig.named("20m", vocab_size=VOCAB, loss_chunk_tokens=0)
    return config, HelixForCausalLM(config).eval()


def _anansi():
    torch.manual_seed(0)
    config = AnansiConfig.named("a-20m", vocab_size=VOCAB, loss_chunk_tokens=0, ngram_memory_size=256)
    return config, AnansiForCausalLM(config).eval()


def _same_function(small, config, state, ids, atol=1e-5):
    big = build(config).eval()
    big.load_state_dict(state)
    with torch.no_grad():
        a = small(ids, num_loops=2).logits if isinstance(small, AnansiForCausalLM) else small(ids).logits
        b = big(ids, num_loops=2).logits if isinstance(big, AnansiForCausalLM) else big(ids).logits
    assert torch.allclose(a, b, atol=atol), (a - b).abs().max()
    return big


def test_identity_depth_growth_preserves_helix():
    config, model = _helix()
    ids = torch.randint(0, VOCAB, (2, 40))
    grown_config, state = grow_depth(config, model.state_dict(), {1: 2, 3: 1})
    assert grown_config.num_hidden_layers == config.num_hidden_layers + 3
    big = _same_function(model, grown_config, state, ids)
    assert len(big.model.layers) == grown_config.num_hidden_layers


def test_mlp_growth_preserves_helix():
    config, model = _helix()
    ids = torch.randint(0, VOCAB, (2, 40))
    grown_config, state = grow_mlp(config, model.state_dict(), config.intermediate_size * 2)
    big = _same_function(model, grown_config, state, ids)
    assert big.model.layers[0].mlp.down_proj.weight.shape[1] == 2 * config.intermediate_size


def test_stream_and_core_growth_preserve_anansi():
    config, model = _anansi()
    ids = torch.randint(0, VOCAB, (2, 40))
    grown_config, state = grow_streams(config, model.state_dict(), 4)
    assert grown_config.num_streams == 4
    _same_function(model, grown_config, state, ids, atol=1e-4)
    deeper_config, state = grow(config, model.state_dict(), [("depth", config.num_hidden_layers + 2)], mode="identity")
    assert (
        deeper_config.core_layers == config.core_layers + 2 and deeper_config.prelude_layers == config.prelude_layers
    )
    _same_function(model, deeper_config, state, ids, atol=1e-4)


def test_stack_mode_copies_trained_blocks():
    config, model = _helix()
    ids = torch.randint(0, VOCAB, (1, 32))
    stacked_config, state = grow(config, model.state_dict(), [("depth", 2 * config.num_hidden_layers)], mode="stack")
    big = build(stacked_config).eval()
    big.load_state_dict(state)
    assert len(big.model.layers) == 2 * config.num_hidden_layers
    with torch.no_grad():
        assert torch.isfinite(big(ids, labels=ids).loss)
        assert not torch.allclose(big(ids).logits, model(ids).logits), "stacking is not function-preserving"
    first_copy = next(k for k in state if k.startswith("model.layers.1.") and k.endswith("gate_proj.weight"))
    assert torch.equal(state[first_copy], model.state_dict()[first_copy.replace("layers.1.", "layers.0.")])


def test_grow_export_end_to_end(tmp_path):
    config, model = _anansi()
    model.save_pretrained(tmp_path / "small", dtype=torch.float32)
    target_config, steps = plan_from_sizes(config, "a-20m")
    assert steps == [], "the same size needs no growth"
    out = grow_export(tmp_path / "small", "a-20m", tmp_path / "same", mode="identity")
    grown_config, state = load_export(out)
    assert grown_config.num_hidden_layers == config.num_hidden_layers
    assert json.loads((out / "growth.json").read_text())["mode"] == "identity"
    with pytest.raises(ValueError):
        plan_from_sizes(config, "a-350m")  # different width: not a growth target
