# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""SFT data: every converter, the assistant-only mask through packing, and a fine-tune from a saved model.
Evaluation: the lm_eval adapter's log-likelihoods against a direct forward, and the comparison table."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from helix_lm.data.agent import (
    convert_openai_messages,
    convert_opencodeinstruct,
    convert_sharegpt,
    convert_swe_agent,
    convert_swe_smith,
    fit_conversation,
    pack_conversations,
    prepare_sft,
)
from helix_lm.data.catalog import Source
from helix_lm.data.loader import TokenSource
from helix_lm.evaluate import PUBLISHED, compare
from helix_lm.tokenizer import HelixTokenizer
from helix_lm.train import Trainer, parse_args

SWE_ROW = {
    "target": True,
    "exit_status": "submitted (exit_context)",
    "trajectory": [
        {"role": "system", "system_prompt": "SETTING: you are an autonomous programmer.", "text": None, "mask": False},
        {"role": "user", "text": "ISSUE: TypeError in memset provider", "mask": False},
        {"role": "ai", "text": "Let's reproduce it first.\n```\npython -m lexicon memset\n```", "mask": True},
        {"role": "user", "text": "Traceback ... TypeError", "mask": False},
        {"role": "ai", "text": "Found it; fixing line 42.\n```\nedit 42:42\n```\n```\nsubmit\n```", "mask": True},
    ],
}
SMITH_ROW = {
    "resolved": True,
    "messages": json.dumps(
        [
            {"role": "system", "content": "You are a software engineer."},
            {"role": "user", "content": "Fix the failing test."},
            {"role": "assistant", "content": "Running the tests.", "tool_calls": [
                {"function": {"name": "bash", "arguments": "{\"command\": \"pytest -q\"}"}}]},
            {"role": "tool", "content": "1 failed"},
            {"role": "assistant", "content": "Patched; tests pass."},
        ]
    ),
}  # fmt: skip


def test_converters_produce_assistant_turns():
    swe = convert_swe_agent(SWE_ROW)
    assert [t["role"] for t in swe] == ["system", "user", "assistant", "user", "assistant"]
    assert swe[0]["content"].startswith("SETTING")
    assert convert_swe_agent({**SWE_ROW, "target": False}) is None
    assert convert_swe_agent({**SWE_ROW, "exit_status": "autosubmission (max_steps)"}) is None
    smith = convert_swe_smith(SMITH_ROW)
    assert smith[2]["content"].endswith("<|/tool_call|>") and '"command": "pytest -q"' in smith[2]["content"]
    assert smith[3]["role"] == "tool" and smith[3]["content"].startswith("<|tool_result|>")
    assert convert_swe_smith({**SMITH_ROW, "resolved": False}) is None
    assert convert_opencodeinstruct(
        {"input": "Write fizzbuzz", "output": "def fizzbuzz(): ...", "average_test_score": 1.0}
    )
    assert convert_opencodeinstruct({"input": "x", "output": "y", "average_test_score": 0.2}) is None
    share = convert_sharegpt(
        {"conversations": [{"from": "human", "value": "2+2?"}, {"from": "gpt", "value": "<think>4</think>4"}]}
    )
    assert share[1]["content"] == "<|think|>4<|/think|>4"
    assert (
        convert_openai_messages([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])[0]["content"] == "hi"
    )


def test_fit_conversation_cuts_at_an_assistant_turn(corpus):
    tokenizer = HelixTokenizer.from_pretrained(corpus[1])
    conversation = convert_swe_agent(SWE_ROW)
    full_ids, _ = tokenizer.encode_chat(conversation)
    ids, mask = fit_conversation(tokenizer, conversation, len(full_ids) - 1)
    assert len(ids) < len(full_ids)
    assert tokenizer.decode(ids, skip_special_tokens=False).rstrip().endswith("<|im_end|>")
    assert tokenizer.decode(ids, skip_special_tokens=False) == HelixTokenizer.render_chat(conversation[:3])
    assert fit_conversation(tokenizer, conversation, 5) is None


def test_packing_writes_masked_shards_that_the_loader_reads(corpus, tmp_path):
    tokenizer = HelixTokenizer.from_pretrained(corpus[1])
    conversations = [convert_swe_agent(SWE_ROW), convert_swe_smith(SMITH_ROW)] * 5
    stats = pack_conversations(tokenizer, conversations, tmp_path / "sft", "sft", max_tokens=4096, shard_tokens=300)
    assert stats["conversations"] == 10 and stats["dropped"] == 0 and 0 < stats["supervised_tokens"] < stats["tokens"]
    source = TokenSource("sft", tmp_path / "sft")
    assert source.masks is not None and source.total == stats["tokens"]
    ids, mask = source.read(0, source.total), source.read_mask(0, source.total)
    assert int(mask.sum()) == stats["supervised_tokens"]
    supervised = tokenizer.decode(ids[mask.astype(bool)].astype(np.int64).tolist(), skip_special_tokens=False)
    assert "Let's reproduce it first." in supervised and "ISSUE" not in supervised
    assert all(p.with_suffix(".mask").exists() for p in (tmp_path / "sft").glob("*.bin"))


def test_prepare_sft_from_local_files_and_fine_tune(corpus, prepared, tmp_path):
    root, tok_dir, _, _ = corpus
    swe = tmp_path / "swe.parquet"
    pq.write_table(
        pa.table({k: [v] * 4 for k, v in {**SWE_ROW, "trajectory": json.dumps(SWE_ROW["trajectory"])}.items()}), swe
    )
    smith = tmp_path / "smith.jsonl"
    smith.write_text("\n".join(json.dumps(SMITH_ROW) for _ in range(4)) + "\n")
    sources = (
        Source("swe-agent", "local", "trajectory", 0.5, kind="sft"),
        Source("swe-smith", "local", "messages", 0.4, kind="sft"),
        Source("identity-chat", "(generated)", "messages", 0.1, kind="sft"),
    )
    data = tmp_path / "data"
    sft_dir = prepare_sft(
        data, tok_dir, seq_len=512, sources=sources, local_files={"swe-agent": [swe], "swe-smith": [smith]},
        log=lambda *_: None,
    )  # fmt: skip
    mixture = json.loads((sft_dir / "mixture.json").read_text())
    assert set(mixture["weights"]) == {"swe-agent", "swe-smith", "identity-chat"}
    assert (sft_dir / "tokens" / "identity-chat").exists() and (sft_dir / "tokenizer" / "tokenizer.json").exists()
    assert TokenSource("x", sft_dir / "tokens" / "swe-agent").masks is not None

    base = tmp_path / "base"
    small = [
        "--size", "20m", "--seq-len", "32", "--micro-batch", "2", "--global-batch-tokens", "128", "--steps", "2",
        "--log-every", "1", "--eval-every", "0", "--warmup-steps", "1", "--holdout-tokens", "40",
        "--override", "block_size=8", "--override", "local_blocks=2", "--override", "index_branching=2",
        "--override", "index_topk=2", "--override", "landmark_dim=16", "--override", "attention_backend=sdpa",
    ]  # fmt: skip
    Trainer(parse_args(["--data", str(prepared), "--out", str(base), "--tokenizer", str(tok_dir), *small])).run()
    sft_run = tmp_path / "sft-run"
    sft_args = ["--data", str(sft_dir), "--out", str(sft_run), "--mixture", str(sft_dir / "mixture.json")]
    trainer = Trainer(parse_args([*sft_args, "--init-from", str(base / "export"), *small]))
    ids, _, mask = trainer.loader.next_batch()
    assert not mask.all() and mask.any(), "SFT batches carry a real loss mask"
    assert trainer.run() == "planned"


def test_lm_eval_adapter_scores_like_a_direct_forward(prepared, tmp_path):
    pytest.importorskip("lm_eval")
    from helix_lm.evaluate import HelixLM

    out = tmp_path / "m"
    small = ["--size", "20m", "--seq-len", "32", "--micro-batch", "1", "--global-batch-tokens", "32", "--steps", "1",
             "--eval-every", "0", "--override", "block_size=8", "--override", "local_blocks=2",
             "--override", "index_branching=2", "--override", "index_topk=2", "--override", "landmark_dim=16",
             "--override", "attention_backend=sdpa"]  # fmt: skip
    Trainer(
        parse_args(["--data", str(prepared), "--out", str(out), "--tokenizer", str(prepared.parent / "tok"), *small])
    ).run()
    lm = HelixLM(str(out / "export"), device="cpu", batch_size=2, max_length=64)

    class Request:
        def __init__(self, *args):
            self.args = args

    requests = [Request("The quick brown", " fox"), Request("def f(x):", " return x"), Request("", "hello")]
    scores = lm.loglikelihood(requests)
    assert len(scores) == 3 and all(s[0] < 0 for s in scores)
    context, continuation = (
        lm.tok_encode("The quick brown"),
        lm.tok_encode("The quick brown fox")[len(lm.tok_encode("The quick brown")) :],
    )
    ids = torch.tensor([context + continuation])
    logprobs = torch.log_softmax(lm.model(ids).logits.float(), -1)[0]
    expected = sum(float(logprobs[len(context) - 1 + i, t].detach()) for i, t in enumerate(continuation))
    assert abs(scores[0][0] - expected) < 1e-3
    text = lm.generate_until([Request("Mi deh", {"until": ["\n"], "max_gen_toks": 4})])
    assert len(text) == 1 and isinstance(text[0], str)
    rolling = lm.loglikelihood_rolling([Request("the quick brown fox jumps over the lazy dog")])
    assert rolling[0] < 0


def test_comparison_table_lists_the_targets():
    text = compare({"humaneval": 40.0, "mmlu_pro": 30.5})
    assert "Qwen3.5-27B" in text and "72.4" in text and "40.0" in text and "mmlu_pro" in text
    assert PUBLISHED["Qwen3.5-27B"]["swe_bench_verified"] == 72.4
