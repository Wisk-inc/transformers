# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Supervised fine-tuning data for a coding agent.

Every source is turned into the same thing -- a conversation of ``system``/``user``/``assistant``/``tool``
turns in the chat format of :mod:`helix_lm.tokenizer` -- then tokenized with the loss mask on the
assistant's turns, packed end to end, and written as the same uint16 shards the trainer already reads,
plus a parallel ``.mask`` tape. ``helix train --data <sft dir> --init-from <pretrained export>`` is the
whole fine-tuning command.

The agent sources are kept honest: SWE-agent runs that did not resolve their issue are dropped, as are
SWE-smith trajectories that were not resolved, so the model imitates runs that worked.
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from ..tokenizer import HelixTokenizer
from .catalog import SFT_SOURCES, Source
from .identity import identity_conversations
from .prepare import DATA_EXTENSIONS, ShardWriter, iter_rows, shard_files, source_tokens

Conversation = list[dict]


# ------------------------------------------------------------------------------------------- rows --
def _as_list(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return list(value or [])


# ------------------------------------------------------------------------------------ converters --
def convert_swe_agent(row: dict, resolved_only: bool = True) -> Conversation | None:
    """nebius/SWE-agent-trajectories: ``trajectory`` is a list of system / user / ai turns."""
    if resolved_only and not row.get("target"):
        return None
    if not str(row.get("exit_status", "")).startswith("submitted"):
        return None
    turns = _as_list(row.get("trajectory"))
    out: Conversation = []
    for turn in turns:
        role = turn.get("role")
        if role == "system":
            content = turn.get("system_prompt") or turn.get("text") or ""
            out.append({"role": "system", "content": content})
        elif role in ("user", "human"):
            out.append({"role": "user", "content": turn.get("text") or ""})
        elif role in ("ai", "assistant"):
            out.append({"role": "assistant", "content": turn.get("text") or ""})
    return out if any(t["role"] == "assistant" for t in out) else None


def convert_openai_messages(messages: Iterable[dict]) -> Conversation:
    """OpenAI-style messages (with ``tool_calls`` and ``tool`` turns) into the HELIX chat format."""
    out: Conversation = []
    for message in messages:
        role, content = message.get("role"), message.get("content") or ""
        if isinstance(content, list):  # multi-part content
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if role == "assistant":
            calls = message.get("tool_calls") or []
            for call in calls:
                function = call.get("function", call)
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}
                content += HelixTokenizer.tool_call(function.get("name", "tool"), arguments)
            out.append({"role": "assistant", "content": content})
        elif role == "tool":
            out.append({"role": "tool", "content": HelixTokenizer.tool_result(str(content))})
        elif role in ("system", "user"):
            out.append({"role": role, "content": str(content)})
    return out


def convert_swe_smith(row: dict, resolved_only: bool = True) -> Conversation | None:
    if resolved_only and row.get("resolved") is False:
        return None
    out = convert_openai_messages(_as_list(row.get("messages")))
    return out if any(t["role"] == "assistant" for t in out) else None


def convert_opencodeinstruct(row: dict) -> Conversation | None:
    score = row.get("average_test_score")
    if score is not None and float(score) < 0.75:
        return None
    if not row.get("input") or not row.get("output"):
        return None
    return [{"role": "user", "content": str(row["input"])}, {"role": "assistant", "content": str(row["output"])}]


def convert_sharegpt(row: dict) -> Conversation | None:
    """``conversations`` of ``{from: human|gpt|system, value}`` (OpenThoughts and friends)."""
    roles = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}
    out = []
    for turn in _as_list(row.get("conversations")):
        role = roles.get(turn.get("from"))
        if role:
            value = str(turn.get("value") or "").replace("<think>", "<|think|>").replace("</think>", "<|/think|>")
            out.append({"role": role, "content": value})
    return out if any(t["role"] == "assistant" for t in out) else None


def convert_messages(row: dict) -> Conversation | None:
    out = convert_openai_messages(_as_list(row.get("messages")))
    return out if any(t["role"] == "assistant" for t in out) else None


def convert_prompt_response(row: dict) -> Conversation | None:
    """``prompt`` / ``response`` pairs (CoCoNot and friends)."""
    prompt = row.get("prompt") or row.get("instruction") or row.get("question")
    response = row.get("response") or row.get("output") or row.get("answer")
    if not prompt or not response:
        return None
    return [{"role": "user", "content": str(prompt)}, {"role": "assistant", "content": str(response)}]


def convert_columns_chat(row: dict) -> Conversation | None:
    """``system`` / ``user`` / ``assistant`` columns (Finance-Instruct), or ``inputs`` / ``answer`` (Sujet)."""
    if row.get("messages"):
        return convert_messages(row)
    user = row.get("user") or row.get("inputs") or row.get("question") or row.get("input")
    assistant = row.get("assistant") or row.get("answer") or row.get("output")
    if not user or not assistant:
        return None
    out = [{"role": "user", "content": str(user)}, {"role": "assistant", "content": str(assistant)}]
    if row.get("system"):
        out.insert(0, {"role": "system", "content": str(row["system"])})
    return out


def convert_medical_o1(row: dict) -> Conversation | None:
    """medical-o1: the question, the reasoning inside think tags, then the answer."""
    question, thinking, answer = row.get("Question"), row.get("Complex_CoT"), row.get("Response")
    if not question or not answer:
        return None
    content = (f"<|think|>{thinking}<|/think|>" if thinking else "") + str(answer)
    return [{"role": "user", "content": str(question)}, {"role": "assistant", "content": content}]


CONVERTERS = {
    "swe-agent": convert_swe_agent,
    "swe-smith": convert_swe_smith,
    "opencodeinstruct": convert_opencodeinstruct,
    "openthoughts": convert_sharegpt,
    "smoltalk": convert_messages,
    "tulu": convert_messages,
    "coconot": convert_prompt_response,
    "finance-chat": convert_columns_chat,
    "medical-chat": convert_medical_o1,
    "codeswitch-chat": convert_messages,
}

GENERATED = {
    "identity-chat": lambda: identity_conversations(),
    "safety-chat": lambda: __import__(
        "helix_lm.data.safety", fromlist=["safety_conversations"]
    ).safety_conversations(),
}


# --------------------------------------------------------------------------------------- packing --
def fit_conversation(
    tokenizer: HelixTokenizer, conversation: Conversation, max_tokens: int
) -> tuple[list, list] | None:
    """
    Tokenize with the assistant mask; a conversation longer than ``max_tokens`` keeps the longest prefix
    of whole turns that still ends on an assistant turn. ``None`` when not even that fits.
    """
    ids, mask = tokenizer.encode_chat(conversation)
    if len(ids) <= max_tokens:
        return ids, mask
    for cut in range(len(conversation) - 1, 0, -1):
        if conversation[cut - 1]["role"] != "assistant":
            continue
        ids, mask = tokenizer.encode_chat(conversation[:cut])
        if len(ids) <= max_tokens:
            return ids, mask
    return None


class MaskedShardWriter(ShardWriter):
    """A :class:`ShardWriter` that also writes the loss mask as a ``.mask`` tape beside each ``.bin``."""

    def __init__(self, directory: Path, prefix: str, shard_tokens: int) -> None:
        super().__init__(directory, prefix, shard_tokens)
        self.mask_buffer: list[np.ndarray] = []

    def add_masked(self, ids: list[int], mask: list[bool]) -> None:
        self.mask_buffer.append(np.asarray(mask, dtype=np.uint8))
        self.add(ids, len(ids))

    def flush(self) -> None:
        if not self.buffered:
            return
        index = len(self.written)
        mask = np.concatenate(self.mask_buffer)
        super().flush()
        path = self.directory / f"{self.prefix}-{index:05d}.mask"
        mask.tofile(path.with_suffix(".mask.part"))
        path.with_suffix(".mask.part").rename(path)
        self.mask_buffer = []


def pack_conversations(
    tokenizer: HelixTokenizer,
    conversations: Iterable[Conversation],
    out_dir: Path,
    prefix: str,
    max_tokens: int,
    shard_tokens: int = 50_000_000,
) -> dict:
    """Tokenize, mask and write conversations end to end. Returns counts."""
    writer = MaskedShardWriter(out_dir, prefix, shard_tokens)
    kept = dropped = supervised = 0
    for conversation in conversations:
        fitted = fit_conversation(tokenizer, conversation, max_tokens)
        if fitted is None:
            dropped += 1
            continue
        ids, mask = fitted
        writer.add_masked(ids, mask)
        kept += 1
        supervised += sum(mask)
    writer.flush()
    return {"conversations": kept, "dropped": dropped, "tokens": writer.total_tokens, "supervised_tokens": supervised}


# ---------------------------------------------------------------------------------------- driver --
def prepare_sft(
    data_dir: str | Path,
    tokenizer_dir: str | Path,
    seq_len: int = 32768,
    sources: tuple[Source, ...] = SFT_SOURCES,
    max_conversations: int = 300_000,
    token: str | None = None,
    local_files: dict[str, list[Path]] | None = None,
    log=print,
) -> Path:
    """
    Build ``data_dir/sft``: one masked source per SFT dataset plus the identity conversations and a
    ``mixture.json`` with the catalog weights, ready for ``helix train --data data_dir/sft``.
    ``local_files`` maps a source name to files on disk (tests, private trajectories).
    """
    from huggingface_hub import HfApi, hf_hub_download

    data_dir = Path(data_dir)
    sft_dir = data_dir / "sft"
    tokens_dir = sft_dir / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = HelixTokenizer.from_pretrained(tokenizer_dir)
    tok_copy = sft_dir / "tokenizer"
    if not tok_copy.exists():
        shutil.copytree(Path(tokenizer_dir) if Path(tokenizer_dir).is_dir() else Path(tokenizer_dir).parent, tok_copy)
    api = HfApi(token=token)
    weights = {}
    for source in sources:
        weights[source.name] = source.weight
        out_dir = tokens_dir / source.name
        if shard_files(out_dir):
            log(f"[sft] {source.name}: already prepared ({source_tokens(out_dir) / 1e6:.1f}M tokens)")
            continue
        if source.name in GENERATED:
            stats = pack_conversations(tokenizer, GENERATED[source.name](), out_dir, source.name, seq_len)
            (out_dir / "manifest.json").write_text(json.dumps({"source": source.name, "complete": True, **stats}))
            log(f"[sft] {source.name}: {stats}")
            continue
        converter = CONVERTERS.get(source.name)
        if converter is None:
            log(f"[sft] {source.name}: no converter -- skipped")
            continue
        if local_files and source.name in local_files:
            files = [(None, p) for p in local_files[source.name]]
        elif source.repo == "(local)":
            folder = data_dir / "local" / source.name
            found = sorted(p for p in folder.iterdir() if p.name.endswith(DATA_EXTENSIONS)) if folder.exists() else []
            if not found:
                log(f"[sft] {source.name}: nothing under {folder} -- skipped")
                continue
            files = [(None, p) for p in found]
        else:
            try:
                names = [
                    f
                    for f in api.list_repo_files(source.repo, repo_type="dataset")
                    if f.endswith(DATA_EXTENSIONS) and (not source.paths or f.startswith(source.paths))
                ]
            except Exception as error:
                log(f"[sft] {source.name}: {type(error).__name__}: {str(error)[:100]} -- skipped")
                continue
            random.Random(f"helix-sft:{source.name}").shuffle(names)
            files = [(n, None) for n in names]

        def conversations(files=files, source=source, converter=converter) -> Iterator[Conversation]:
            seen = 0
            cache = Path(tempfile.mkdtemp(prefix="helix-sft-"))
            try:
                for name, local in files:
                    path = local or Path(
                        hf_hub_download(source.repo, name, repo_type="dataset", cache_dir=cache, token=token)
                    )
                    for row in iter_rows(path):
                        conversation = converter(row)
                        if conversation:
                            seen += 1
                            yield conversation
                            if seen >= max_conversations:
                                return
                    if local is None:
                        shutil.rmtree(cache, ignore_errors=True)
                        cache.mkdir(exist_ok=True)
            finally:
                shutil.rmtree(cache, ignore_errors=True)

        stats = pack_conversations(tokenizer, conversations(), out_dir, source.name, seq_len)
        (out_dir / "manifest.json").write_text(json.dumps({"source": source.name, "complete": True, **stats}))
        log(f"[sft] {source.name}: {stats}")
    (sft_dir / "mixture.json").write_text(
        json.dumps({"weights": weights, "max_epochs": {s.name: max(s.max_epochs, 3.0) for s in sources}}, indent=1)
    )
    return sft_dir
