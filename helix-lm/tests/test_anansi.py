# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Anansi: the looped, hyper-connected body around the HELIX strands, checked at toy scale on a CPU."""

import torch

from helix_lm import arch
from helix_lm.anansi import ANANSI_SIZES, AnansiConfig, AnansiForCausalLM, describe_anansi_sizes
from helix_lm.anansi.layers import HyperConnections, ngram_buckets, sinkhorn
from helix_lm.anansi.model import active_parameters, count_parameters

VOCAB = 512


def tiny(**overrides) -> AnansiConfig:
    fields = {
        "vocab_size": VOCAB,
        "loss_chunk_tokens": 0,
        "z_loss_weight": 1e-4,
        "ngram_memory_size": 1024,
        **overrides,
    }
    return AnansiConfig.named("a-20m", **fields)


def test_named_sizes_count_and_describe():
    for name in ("a-20m", "a-350m", "a-1b"):
        config = AnansiConfig.named(name)
        assert config.num_hidden_layers == config.prelude_layers + config.core_layers + config.coda_layers
        assert config.layer_types[0] == "recurrent", "the prelude runs the recurrence only"
        assert active_parameters(config) > count_parameters(config, embeddings=False)
    table = describe_anansi_sizes()
    assert "a-34b" in table and all(name in table for name in ANANSI_SIZES)


def test_parameter_count_matches_the_built_model():
    config = tiny()
    model = AnansiForCausalLM(config)
    assert sum(p.numel() for p in model.parameters()) == config.num_parameters()
    assert model.model.layers[0].mixer.braid is None and model.model.layers[1].mixer.braid is not None


def test_training_loss_has_mtp_and_eval_loss_does_not():
    torch.manual_seed(0)
    model = AnansiForCausalLM(tiny())
    ids = torch.randint(0, VOCAB, (2, 64))
    model.train()
    out = model(ids, labels=ids)
    assert out.mtp_loss is not None and out.loss.item() > out.nll.item()
    assert 1 <= out.loops <= model.config.max_loops
    out.loss.backward()
    assert all(p.grad is not None for n, p in model.named_parameters() if "mtp" in n or "hc_" in n or "ngram" in n)
    model.eval()
    ev = model(ids, labels=ids)
    assert torch.isclose(ev.loss, ev.nll) and ev.loops == model.config.loops_for_inference()


def test_chunked_loss_equals_plain_loss():
    torch.manual_seed(0)
    plain = AnansiForCausalLM(tiny())
    chunked = AnansiForCausalLM(tiny(loss_chunk_tokens=37))
    chunked.load_state_dict(plain.state_dict())
    plain.eval(), chunked.eval()
    ids = torch.randint(0, VOCAB, (2, 64))
    a, b = plain(ids, labels=ids), chunked(ids, labels=ids)
    assert torch.allclose(a.loss, b.loss, atol=1e-4) and b.logits is None


def test_more_loops_change_the_output_and_think_exits():
    torch.manual_seed(0)
    model = AnansiForCausalLM(tiny()).eval()
    ids = torch.randint(0, VOCAB, (1, 48))
    one, four = model(ids, num_loops=1).logits, model(ids, num_loops=4).logits
    assert not torch.allclose(one, four)
    logits, used = model.think(ids, max_loops=4, kl_threshold=1e9)
    assert used == 2, "a huge threshold stops at the first comparison"
    logits, used = model.think(ids, max_loops=4, kl_threshold=0.0)
    assert used == 4 and torch.allclose(logits, four, atol=1e-4)


def test_cached_decoding_matches_the_full_forward_and_generate_runs():
    torch.manual_seed(0)
    model = AnansiForCausalLM(tiny()).eval()
    ids = torch.randint(0, VOCAB, (2, 20))
    out = model.generate(ids[:, :8], max_new_tokens=4, temperature=0)
    assert out.shape == (2, 12)
    with torch.no_grad():
        full = model(out).logits[:, -2]
        cache = model.new_cache()
        model(out[:, :-2], past_key_values=cache, use_cache=True)
        step = model(out[:, -2:-1], past_key_values=cache, use_cache=True).logits[:, -1]
    assert torch.allclose(full, step, atol=1e-3)
    assert cache.seen == out.shape[1] - 1 and len(cache.loops) == model.config.loops_for_inference()


def test_save_load_roundtrip_and_arch_integration(tmp_path):
    torch.manual_seed(0)
    model = AnansiForCausalLM(tiny()).eval()
    ids = torch.randint(0, VOCAB, (1, 32))
    model.save_pretrained(tmp_path / "m", dtype=torch.float32)
    again = AnansiForCausalLM.from_pretrained(tmp_path / "m").eval()
    assert torch.allclose(again(ids).logits, model(ids).logits, atol=1e-5)
    assert again.config.layer_types == model.config.layer_types
    assert arch.family_of("a-20m") == "anansi" and "a-20m" in arch.ALL_SIZES
    config = arch.make_config("a-20m", vocab_size=VOCAB)
    with torch.device("meta"):
        meta = arch.construct(config)
    assert len(arch.layers_of(meta)) == config.num_hidden_layers
    real = arch.construct(config)
    arch.initialize(real, config, seed=1)
    assert torch.isfinite(real(ids, labels=ids).loss)
    assert arch.flops_per_token(config, 2048) > 6 * arch.num_parameters(config, embeddings=False)


def test_mixture_of_experts_balances_load():
    torch.manual_seed(0)
    config = tiny(num_experts=4, experts_per_token=2, ngram_memory_size=0, mtp_depth=0, balance_bias_rate=0.01)
    model = AnansiForCausalLM(config)
    layer = model.model.layers[config.prelude_layers]
    assert layer.is_moe
    ids = torch.randint(0, VOCAB, (2, 64))
    model.train()
    before = layer.mlp.balance_bias.clone()
    out = model(ids, labels=ids, num_loops=1)  # one pass, so the layer's load and its bias update line up
    out.loss.backward()
    load = layer.mlp.expert_load
    assert load.sum() == 2 * 64 * 2 and not torch.equal(before, layer.mlp.balance_bias)
    # the bias moved towards the under-used experts
    assert torch.all(torch.sign(layer.mlp.balance_bias) == torch.sign(load.mean() - load))
    dense = tiny(
        ngram_memory_size=0, mtp_depth=0, intermediate_size=config.expert_intermediate_size * config.num_experts
    )
    assert active_parameters(config) < active_parameters(dense), (
        "routing touches fewer parameters than a dense MLP of all experts"
    )
    assert count_parameters(config) > count_parameters(tiny(ngram_memory_size=0, mtp_depth=0))


def test_hyper_connections_and_sinkhorn():
    P = sinkhorn(torch.randn(3, 4, 4))
    assert torch.allclose(P.sum(-1), torch.ones(3, 4), atol=1e-4) and torch.allclose(
        P.sum(-2), torch.ones(3, 4), atol=1e-4
    )
    one = HyperConnections(8, 1, 0)
    x, y = torch.randn(2, 5, 1, 8), torch.randn(2, 5, 8)
    read, coefficients = one.read(x)
    assert torch.allclose(one.write(x, y, coefficients)[:, :, 0], x[:, :, 0] + y)
    four = HyperConnections(8, 4, 3)
    x = torch.randn(2, 5, 4, 8)
    read, coefficients = four.read(x)
    assert torch.allclose(read, x[:, :, 3], atol=0.2), "at init block 3 reads stream 3"
    written = four.write(x, y, coefficients)
    assert torch.allclose(written[:, :, 3], x[:, :, 3] + y, atol=0.3) and torch.allclose(
        written[:, :, 0], x[:, :, 0], atol=0.3
    )


def test_ngram_buckets_are_deterministic_and_use_the_prefix():
    ids = torch.tensor([[5, 7, 7, 9, 5, 7]])
    b2 = ngram_buckets(ids, 2, 1000)
    assert b2.shape == ids.shape and b2.min() >= 0 and b2.max() < 1000
    assert b2[0, 1] == b2[0, 5], "the same bigram (5, 7) hashes to the same bucket"
    assert torch.equal(b2, ngram_buckets(ids, 2, 1000))
    with_prefix = ngram_buckets(ids[:, 1:], 2, 1000, prefix=ids[:, :1])
    assert with_prefix[0, 0] == b2[0, 1], "the prefix completes the first position's window"
