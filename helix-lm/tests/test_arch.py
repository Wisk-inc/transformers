# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Both model families through one adapter: counts, meta-device construction, in-place init, export, and a
training run of the transformer twin that reloads with plain ``transformers``."""

import json

import pytest
import torch

from helix_lm import arch
from helix_lm.config import HelixConfig
from helix_lm.train import Trainer, parse_args

TINY_T = [
    "--override", "hidden_size=64", "--override", "intermediate_size=128", "--override", "num_hidden_layers=2",
    "--override", "num_attention_heads=4", "--override", "num_key_value_heads=2", "--override", "head_dim=16",
]  # fmt: skip


def test_families_and_counts():
    assert arch.family_of("34b") == "helix" and arch.family_of("t-34b") == "transformer"
    assert 33.5e9 <= arch.num_parameters(HelixConfig.named("34b")) <= 34.3e9
    t34 = arch.make_config("t-34b")
    assert 33.5e9 <= arch.num_parameters(t34) <= 34.3e9
    with torch.device("meta"):
        built = arch.construct(t34)
    assert sum(p.numel() for p in built.parameters()) == arch.num_parameters(t34)
    assert arch.flops_per_token(t34, 8192) > 6 * arch.num_parameters(t34, embeddings=False)
    assert arch.flops_per_token(HelixConfig.named("34b"), 8192) == HelixConfig.named("34b").flops_per_token()
    with pytest.raises(ValueError):
        arch.family_of("t-9000b")
    assert "t-34b" in arch.describe_transformer_sizes()


def test_transformer_meta_init_gives_real_weights_and_rotary():
    config = arch.make_config("t-350m", hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                              num_attention_heads=4, num_key_value_heads=2, head_dim=16)  # fmt: skip
    with torch.device("meta"):
        model = arch.construct(config)
    model.to_empty(device="cpu")
    arch.initialize(model, config, seed=0)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert torch.isfinite(model.model.rotary_emb.inv_freq).all() and model.model.rotary_emb.inv_freq[0] == 1.0
    std = config.initializer_range / (2 * config.num_hidden_layers) ** 0.5
    assert abs(model.model.layers[0].mlp.down_proj.weight.std().item() - std) < 0.5 * std
    ids = torch.randint(0, config.vocab_size, (2, 12))
    assert torch.isfinite(model(input_ids=ids, labels=ids).loss)


def test_transformer_twin_trains_exports_and_reloads_with_transformers(prepared, tmp_path):
    out = tmp_path / "twin"
    args = parse_args([
        "--data", str(prepared), "--out", str(out), "--tokenizer", str(prepared.parent / "tok"), "--size", "t-350m",
        "--seq-len", "32", "--micro-batch", "2", "--global-batch-tokens", "128", "--steps", "3", "--log-every", "1",
        "--eval-every", "0", "--warmup-steps", "1", "--holdout-tokens", "40", "--save-every-steps", "3", *TINY_T,
    ])  # fmt: skip
    trainer = Trainer(args)
    assert trainer.run() == "planned"
    export = out / "export"
    config = json.loads((export / "config.json").read_text())
    assert config["architectures"] == ["Qwen3ForCausalLM"] and config["vocab_size"] == 65536
    assert (export / "tokenizer.json").exists()
    from transformers import AutoModelForCausalLM

    reloaded = AutoModelForCausalLM.from_pretrained(export, dtype=torch.bfloat16)
    assert next(reloaded.parameters()).dtype == torch.bfloat16
    from safetensors import safe_open

    with safe_open(next(iter(export.glob("model*.safetensors"))), "pt") as handle:
        assert handle.get_tensor(next(iter(handle.keys()))).dtype == torch.bfloat16, "exports are bf16 on disk"
    ids = torch.randint(0, 2048, (1, 16))
    assert torch.isfinite(reloaded(input_ids=ids).logits).all()
    # and the same export starts a fine-tune
    again = Trainer(parse_args([
        "--data", str(prepared), "--out", str(tmp_path / "sft"), "--tokenizer", str(prepared.parent / "tok"),
        "--size", "t-350m", "--seq-len", "32", "--micro-batch", "1", "--global-batch-tokens", "32", "--steps", "1",
        "--eval-every", "0", "--init-from", str(export), *TINY_T,
    ]))  # fmt: skip
    for (name, p), (_, q) in zip(reloaded.named_parameters(), again.model.named_parameters(), strict=True):
        torch.testing.assert_close(q.float(), p.float(), msg=name)


@pytest.mark.skipif(__import__("shutil").which("torchrun") is None, reason="torchrun not on PATH")
def test_transformer_twin_under_fsdp2_on_two_processes(prepared, tmp_path):
    import os
    import subprocess
    from pathlib import Path

    out = tmp_path / "fsdp-t"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"), "OMP_NUM_THREADS": "1"}
    command = [
        "torchrun", "--nproc_per_node=2", "--master_port=29519", "-m", "helix_lm.train", "--data", str(prepared),
        "--out", str(out), "--tokenizer", str(prepared.parent / "tok"), "--parallel", "fsdp", "--size", "t-350m",
        "--seq-len", "32", "--micro-batch", "2", "--global-batch-tokens", "128", "--steps", "3", "--log-every", "1",
        "--eval-every", "0", "--warmup-steps", "1", "--holdout-tokens", "40", "--save-every-steps", "3", *TINY_T,
    ]  # fmt: skip
    run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    assert "[train] finished" in run.stdout and (out / "export" / "config.json").exists()
