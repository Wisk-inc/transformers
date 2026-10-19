# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The training engine's upgrades, each checked against the reference it replaces: the fused local attention
against the gathered staircase, the chunked loss against the one-piece loss, recomputation against stored
activations, the sharded safetensors file against the model it came from, and the analytic parameter
count against a built model.
"""

import pytest
import torch

from helix_lm import HelixConfig, HelixForCausalLM
from helix_lm.config import SIZES, describe_sizes
from helix_lm.model import kernels_available

TINY = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "block_size": 8,
    "local_blocks": 3,
    "num_window_scales": 2,
    "index_layer_stride": 2,
    "landmark_dim": 16,
    "index_branching": 2,
    "index_beam_width": 2,
    "index_topk": 2,
    "num_recurrent_heads": 2,
    "recurrent_head_dim": 16,
    "recurrent_value_head_dim": 16,
    "recurrent_chunk_size": 8,
    "conv_kernel_size": 4,
    "surprise_kernel_size": 4,
}


def tiny(dtype=torch.float32, **overrides) -> HelixForCausalLM:
    torch.manual_seed(0)
    return HelixForCausalLM(HelixConfig(**{**TINY, **overrides})).to(dtype)


def copy_weights(source: HelixForCausalLM, target: HelixForCausalLM) -> HelixForCausalLM:
    target.load_state_dict(source.state_dict())
    return target


@pytest.mark.parametrize("length", [37, 128, 300])
def test_flex_local_attention_matches_the_reference(length):
    """Strand L through FlexAttention is the gathered-block staircase, to kernel precision."""
    if not kernels_available()["flex_attention"]:
        pytest.skip("FlexAttention not available")
    reference = tiny(attention_backend="sdpa").eval()
    fused = copy_weights(reference, tiny(attention_backend="flex").eval())
    tokens = torch.randint(0, 128, (2, length))
    with torch.no_grad():
        expected = reference(tokens).logits
        actual = fused(tokens).logits
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


def test_flex_is_only_used_when_it_can_be():
    """A cache or a padding mask sends strand L down the reference path, which handles both."""
    model = tiny(attention_backend="flex").eval()
    braid = model.model.layers[0].mixer
    assert braid._use_flex(None, None, torch.device("cpu"))
    assert not braid._use_flex(None, torch.ones(1, 8, dtype=torch.bool), torch.device("cpu"))
    from helix_lm import HelixCache

    assert not braid._use_flex(HelixCache(model.config), None, torch.device("cpu"))


def test_chunked_loss_equals_one_piece_loss():
    tokens = torch.randint(0, 128, (2, 50))
    labels = tokens.clone()
    labels[0, :7] = -100
    whole = tiny(torch.float64, loss_chunk_tokens=0, z_loss_weight=1e-3)
    chunked = copy_weights(whole, tiny(torch.float64, loss_chunk_tokens=13, z_loss_weight=1e-3))
    a, b = whole(tokens, labels=labels), chunked(tokens, labels=labels)
    assert b.logits is None
    torch.testing.assert_close(b.loss, a.loss)
    torch.testing.assert_close(b.z_loss, a.z_loss)
    a.loss.backward()
    b.loss.backward()
    for (name, p), (_, q) in zip(whole.named_parameters(), chunked.named_parameters(), strict=True):
        torch.testing.assert_close(q.grad, p.grad, msg=name)


def test_activation_checkpointing_changes_nothing():
    tokens = torch.randint(0, 128, (2, 40))
    stored = tiny(torch.float64, attention_backend="sdpa").train()
    recomputed = copy_weights(
        stored, tiny(torch.float64, attention_backend="sdpa", activation_checkpointing=True)
    ).train()
    a, b = stored(tokens, labels=tokens).loss, recomputed(tokens, labels=tokens).loss
    torch.testing.assert_close(b, a)
    a.backward()
    b.backward()
    for (name, p), (_, q) in zip(stored.named_parameters(), recomputed.named_parameters(), strict=True):
        assert q.grad is not None, name
        torch.testing.assert_close(q.grad, p.grad, msg=name)


def test_residual_projections_are_scaled_by_depth():
    config = HelixConfig(**{**TINY, "num_hidden_layers": 8, "hidden_size": 256, "intermediate_size": 512})
    model = HelixForCausalLM(config)
    layer = model.model.layers[0]
    expected = config.initializer_range / (2 * config.num_hidden_layers) ** 0.5
    for projection in (layer.mixer.o_proj, layer.recurrent.out_proj, layer.mlp.down_proj):
        assert abs(projection.weight.std().item() - expected) < 0.25 * expected
    assert abs(layer.mixer.q_proj.weight.std().item() - config.initializer_range) < 0.25 * config.initializer_range
    plain = HelixForCausalLM(HelixConfig(**{**TINY, "scale_residual_init": False}))
    assert abs(plain.model.layers[0].mlp.down_proj.weight.std().item() - 0.02) < 0.01


@pytest.mark.parametrize("tied", [False, True])
def test_sharded_safetensors_round_trip(tmp_path, tied):
    model = tiny(tie_word_embeddings=tied).eval()
    tokens = torch.randint(0, 128, (1, 30))
    with torch.no_grad():
        expected = model(tokens).logits
    model.save_pretrained(tmp_path, max_shard_bytes=20_000)  # tiny shards: several files + an index
    assert (tmp_path / "model.safetensors.index.json").exists()
    assert len(list(tmp_path.glob("model-*.safetensors"))) > 1
    loaded = HelixForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(tokens).logits, expected)
    if tied:
        assert loaded.lm_head.weight.data_ptr() == loaded.model.embed_tokens.weight.data_ptr()


def test_legacy_model_pt_still_loads(tmp_path):
    model = tiny().eval()
    model.save_pretrained(tmp_path, safetensors=False)
    assert (tmp_path / "model.pt").exists()
    loaded = HelixForCausalLM.from_pretrained(tmp_path)
    for (name, p), (_, q) in zip(model.named_parameters(), loaded.named_parameters(), strict=True):
        torch.testing.assert_close(q, p, msg=name)


@pytest.mark.parametrize("size", ["20m"])
def test_named_sizes_count_what_they_build(size):
    config = HelixConfig.named(size)
    assert HelixForCausalLM(config).num_parameters() == config.num_parameters()


def test_analytic_count_matches_tiny_model():
    config = HelixConfig(**TINY)
    assert HelixForCausalLM(config).num_parameters() == config.num_parameters()
    tied = HelixConfig(**{**TINY, "tie_word_embeddings": True})
    assert HelixForCausalLM(tied).num_parameters() == tied.num_parameters()
    assert tied.num_parameters() == config.num_parameters() - config.vocab_size * config.hidden_size


def test_the_ladder_is_what_it_says():
    """23b is 23 billion parameters; every size is heavier than the one before; the table prints."""
    counts = {name: HelixConfig.named(name).num_parameters() for name in SIZES}
    assert 23.0e9 <= counts["23b"] <= 23.6e9
    assert all(a < b for a, b in zip(list(counts.values())[:-1], list(counts.values())[1:], strict=True))
    assert "23b" in describe_sizes()
    assert HelixConfig.named("23b").flops_per_token() > 6 * HelixConfig.named("23b").num_parameters(embeddings=False)
    with pytest.raises(ValueError):
        HelixConfig.named("9000b")


def test_unknown_backends_are_rejected():
    with pytest.raises(ValueError):
        HelixConfig(**{**TINY, "attention_backend": "cudnn"})
    with pytest.raises(ValueError):
        HelixConfig(**{**TINY, "recurrent_backend": "mamba"})
