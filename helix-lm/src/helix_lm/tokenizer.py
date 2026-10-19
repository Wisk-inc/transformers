# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The CorxLabs tokenizer: a 65,536-token byte-level BPE trained on the model's own data mixture.

65,536 is a deliberate size. Every token id fits in two bytes, so 460 billion tokens of prepared training
data take 920 GB on disk rather than the 1.8 TB a 150k vocabulary would, and the embedding tables of a
23B model stay at 0.8B parameters. Code is tokenized with the GPT-4 pre-tokenization pattern (runs of
spaces merge, numbers split in threes, contractions split), and the training sample includes Jamaican
Patois so its spellings get merges of their own rather than being spelled out byte by byte.

The first 32 ids are special tokens, fixed here so a configuration can name them before any tokenizer
exists: ``<|endoftext|>`` is 0 (the document separator and end of sequence), ``<|startoftext|>`` 1,
``<|pad|>`` 2, then the chat, tool-call, fill-in-the-middle and repository markers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

SPECIAL_TOKENS: tuple[str, ...] = (
    "<|endoftext|>",
    "<|startoftext|>",
    "<|pad|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|tool_call|>",
    "<|/tool_call|>",
    "<|tool_result|>",
    "<|/tool_result|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|repo_name|>",
    "<|file_sep|>",
    "<|think|>",
    "<|/think|>",
) + tuple(f"<|reserved_{i}|>" for i in range(16))

EOS, BOS, PAD = 0, 1, 2
IM_START, IM_END = 3, 4
TOOL_CALL, TOOL_CALL_END, TOOL_RESULT, TOOL_RESULT_END = 5, 6, 7, 8
FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, REPO_NAME, FILE_SEP = 9, 10, 11, 12, 13

# GPT-4's pre-tokenization: contractions, words with one leading symbol, numbers in groups of up to three
# digits, punctuation runs, newlines, and runs of whitespace that leave the last space to the next word.
PRETOKENIZE_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# ChatML with tool calls. The assistant writes a call as `<|tool_call|>{json}<|/tool_call|>`; the tool's
# answer comes back as a `tool` turn. Written as a Jinja template so other loaders can use it unchanged.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

MAX_VOCAB = 65536


def train_tokenizer(
    texts: Iterable[str],
    out_dir: str | Path,
    vocab_size: int = MAX_VOCAB,
    min_frequency: int = 2,
    show_progress: bool = False,
) -> HelixTokenizer:
    """
    Train the byte-level BPE on ``texts`` and write ``tokenizer.json``, ``tokenizer_config.json`` and
    ``special_tokens_map.json`` into ``out_dir``. ``vocab_size`` counts the 32 special tokens and must not
    exceed 65,536 (the uint16 guarantee).
    """
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

    if not 256 + len(SPECIAL_TOKENS) <= vocab_size <= MAX_VOCAB:
        raise ValueError(f"vocab_size must be in [{256 + len(SPECIAL_TOKENS)}, {MAX_VOCAB}], got {vocab_size}")
    tokenizer = Tokenizer(models.BPE(unk_token=None, fuse_unk=False, byte_fallback=False))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(PRETOKENIZE_PATTERN), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show_progress,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_dir / "tokenizer.json"))
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": 1_048_576,
                "bos_token": SPECIAL_TOKENS[BOS],
                "eos_token": SPECIAL_TOKENS[EOS],
                "pad_token": SPECIAL_TOKENS[PAD],
                "additional_special_tokens": list(SPECIAL_TOKENS[3:]),
                "chat_template": CHAT_TEMPLATE,
                "clean_up_tokenization_spaces": False,
                "made_by": "CorxLabs",
            },
            indent=2,
        )
        + "\n"
    )
    (out_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": SPECIAL_TOKENS[BOS],
                "eos_token": SPECIAL_TOKENS[EOS],
                "pad_token": SPECIAL_TOKENS[PAD],
                "additional_special_tokens": list(SPECIAL_TOKENS[3:]),
            },
            indent=2,
        )
        + "\n"
    )
    return HelixTokenizer.from_pretrained(out_dir)


class HelixTokenizer:
    """A thin wrapper over the ``tokenizers`` object: ids in, ids out, and the chat format in one place."""

    def __init__(self, tokenizer, directory: Path | None = None) -> None:
        self._tokenizer = tokenizer
        self.directory = directory
        self.vocab_size = tokenizer.get_vocab_size()
        if self.vocab_size > MAX_VOCAB:
            raise ValueError(f"vocabulary of {self.vocab_size} does not fit in uint16")
        for index, token in enumerate(SPECIAL_TOKENS):
            if tokenizer.token_to_id(token) != index:
                raise ValueError(f"special token {token!r} is id {tokenizer.token_to_id(token)}, expected {index}")
        self.eos_token_id, self.bos_token_id, self.pad_token_id = EOS, BOS, PAD

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> HelixTokenizer:
        from tokenizers import Tokenizer

        directory = Path(directory)
        path = directory / "tokenizer.json" if directory.is_dir() else directory
        return cls(Tokenizer.from_file(str(path)), path.parent)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self._tokenizer.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids = [BOS, *ids]
        if add_eos:
            ids = [*ids, EOS]
        return ids

    def encode_batch(self, texts: list[str], add_eos: bool = True) -> list[list[int]]:
        """Many documents at once (parallel in Rust); ``add_eos`` closes each one, which is how the training
        data separates documents."""
        encodings = self._tokenizer.encode_batch(texts, add_special_tokens=False)
        return [[*e.ids, EOS] if add_eos else list(e.ids) for e in encodings]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> int | None:
        return self._tokenizer.token_to_id(token)

    # ------------------------------------------------------------------------------------- chat --
    @staticmethod
    def render_chat(messages: list[dict], add_generation_prompt: bool = False) -> str:
        """The text form of a conversation, exactly what ``CHAT_TEMPLATE`` renders."""
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def encode_chat(self, messages: list[dict], train_on: str = "assistant") -> tuple[list[int], list[bool]]:
        """
        A conversation as ids plus a per-token loss mask: ``True`` on the assistant's own tokens (its text
        and its closing ``<|im_end|>``), ``False`` on the system, user and tool turns and on every header.
        Turn by turn, so a mask boundary can never fall inside a merge.
        """
        ids: list[int] = []
        mask: list[bool] = []
        for message in messages:
            header = self.encode(f"<|im_start|>{message['role']}\n")
            body = self.encode(str(message["content"]))
            closing = self.encode("<|im_end|>\n")
            supervised = message["role"] == train_on
            ids += header + body + closing
            mask += [False] * len(header) + [supervised] * len(body) + [supervised] * (len(closing) - 1) + [False]
        return ids, mask

    @staticmethod
    def tool_call(name: str, arguments: dict) -> str:
        """How the assistant writes a tool call inside its turn."""
        return f"<|tool_call|>{json.dumps({'name': name, 'arguments': arguments}, ensure_ascii=False)}<|/tool_call|>"

    @staticmethod
    def tool_result(content: str) -> str:
        return f"<|tool_result|>{content}<|/tool_result|>"


def iter_text_sample(files: list[Path], max_bytes: int) -> Iterator[str]:
    """Lines from a list of text files, up to ``max_bytes`` in total -- the tokenizer's training sample."""
    seen = 0
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                yield line
                seen += len(line)
                if seen >= max_bytes:
                    return
