# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The CorxLabs tokenizer: trains, round-trips, fits in uint16, and masks a chat the way SFT needs."""

import numpy as np
import pytest

from helix_lm.tokenizer import (
    BOS,
    EOS,
    IM_END,
    IM_START,
    PAD,
    SPECIAL_TOKENS,
    HelixTokenizer,
    train_tokenizer,
)

CORPUS = [
    "The quick brown fox jumps over the lazy dog. " * 3,
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    "for i in range(10):\n        print(i, i ** 2)\n",
    "Mi deh yah a Kingston, Jamaica, an di weather nice today. Wah gwaan?",
    "Jamaica is the third-largest island of the Greater Antilles; its capital is Kingston.",
    "SELECT name, count(*) FROM storms WHERE basin = 'NA' GROUP BY name ORDER BY 2 DESC;",
    "In 2025 Hurricane Melissa reached 160 knots before landfall in Westmoreland.",
    '{"name": "bash", "arguments": {"command": "ls -la /data"}}',
] * 20


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    return train_tokenizer(CORPUS, tmp_path_factory.mktemp("tok"), vocab_size=1024)


def test_special_tokens_sit_at_the_fixed_ids(tokenizer):
    assert tokenizer.token_to_id("<|endoftext|>") == EOS == 0
    assert tokenizer.token_to_id("<|startoftext|>") == BOS == 1
    assert tokenizer.token_to_id("<|pad|>") == PAD == 2
    assert tokenizer.token_to_id("<|im_start|>") == IM_START and tokenizer.token_to_id("<|im_end|>") == IM_END
    assert len(SPECIAL_TOKENS) == 32 and len(set(SPECIAL_TOKENS)) == 32
    assert 32 + 256 < tokenizer.vocab_size <= 1024  # a tiny corpus runs out of merges before the cap


def test_round_trip_is_exact(tokenizer):
    for text in CORPUS[:8] + ["tabs\t\tand   spaces\n\n\nnew lines, émojis 🌀 and ünïcödé"]:
        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids) == text
        assert all(i >= 32 for i in ids), "plain text never produces a special token id"


def test_ids_fit_in_uint16(tokenizer):
    ids = tokenizer.encode_batch(CORPUS[:8])
    flat = np.array([i for doc in ids for i in doc], dtype=np.uint16)
    assert flat.max() < 65536 and all(doc[-1] == EOS for doc in ids)


def test_specials_are_single_tokens_and_text_cannot_forge_them(tokenizer):
    assert tokenizer.encode("<|im_start|>user\n")[0] == IM_START
    forged = tokenizer.encode("< | im_start | >")
    assert IM_START not in forged


def test_chat_encoding_masks_only_the_assistant(tokenizer):
    messages = [
        {"role": "system", "content": "You are Helix, made by CorxLabs."},
        {"role": "user", "content": "Who made you?"},
        {"role": "assistant", "content": "CorxLabs, an AI lab from Jamaica."},
        {"role": "user", "content": "Thanks"},
        {"role": "assistant", "content": "Anytime."},
    ]
    ids, mask = tokenizer.encode_chat(messages)
    assert len(ids) == len(mask)
    assert tokenizer.decode(ids, skip_special_tokens=False) == HelixTokenizer.render_chat(messages)
    supervised = tokenizer.decode([i for i, m in zip(ids, mask, strict=True) if m], skip_special_tokens=False)
    assert supervised == "CorxLabs, an AI lab from Jamaica.<|im_end|>Anytime.<|im_end|>"
    assert not any(m for i, m in zip(ids, mask, strict=True) if i == IM_START)
    rendered = HelixTokenizer.render_chat(messages[:2], add_generation_prompt=True)
    assert rendered.endswith("<|im_start|>assistant\n")


def test_tool_call_format(tokenizer):
    call = HelixTokenizer.tool_call("bash", {"command": "pytest -q"})
    ids = tokenizer.encode(call)
    assert ids[0] == tokenizer.token_to_id("<|tool_call|>") and ids[-1] == tokenizer.token_to_id("<|/tool_call|>")
    assert tokenizer.decode(ids, skip_special_tokens=False) == call


def test_saved_files_and_reload(tokenizer):
    assert (tokenizer.directory / "tokenizer.json").exists()
    assert (tokenizer.directory / "tokenizer_config.json").exists()
    again = HelixTokenizer.from_pretrained(tokenizer.directory)
    assert again.encode(CORPUS[1]) == tokenizer.encode(CORPUS[1])


def test_vocab_bounds_are_enforced(tmp_path):
    with pytest.raises(ValueError):
        train_tokenizer(CORPUS, tmp_path, vocab_size=70000)
