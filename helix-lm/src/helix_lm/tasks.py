# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Toy tasks that separate architectures before any big run: the suite ``helix anansi bench`` trains a tiny model
of each family on each task for the same number of steps and reports held-out accuracy.

* **recall**: key-value pairs, then queries; the answer is the value last bound to the key (MQAR, Zoology).
  Measures exact associative memory: attention and landmarks ace it, fixed-size states saturate with pairs.
* **parity**: a string of bits, then the parity. A state-tracking task plain attention finds hard at depth;
  recurrences with signed updates (the delta rule's erase) solve it.
* **multihop**: a chain ``a -> b, b -> c, ...`` as pairs, then ``a`` and the number of hops; the answer is the
  node that many steps down the chain. Measures composition: looped cores should match much deeper fixed ones.
* **copynoise**: a span to remember, a long run of noise, then reproduce the span. Measures whether a memory
  survives distraction (fixed states fade; surprise gating and landmarks help).

Every example is one sequence with ``-100`` labels except at the answer positions, so the ordinary next-token
loss trains it. The families run with the same width, depth budget, optimizer and data; the differences are
architectural. Expectations from the literature are in ``docs/ANANSI.md``; this suite exists to check them on
your machine in minutes, not to replace a pretraining run.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import arch

SEP, QUERY, HOP_BASE, COPY, NOISE_BASE = 0, 1, 2, 10, 20  # reserved control tokens; content starts at CONTENT
CONTENT = 64


@dataclass
class Task:
    name: str
    vocab: int
    seq_len: int

    def batch(self, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """``(ids (B, T), labels (B, T))`` with ``-100`` outside the answer positions."""
        raise NotImplementedError


def _fill(ids: torch.Tensor, labels: torch.Tensor, seq_len: int, pad: int = SEP) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.full((ids.shape[0], seq_len), pad, dtype=torch.long)
    lab = torch.full((ids.shape[0], seq_len), -100, dtype=torch.long)
    n = min(seq_len, ids.shape[1])
    out[:, :n], lab[:, :n] = ids[:, :n], labels[:, :n]
    return out, lab


class Recall(Task):
    def __init__(self, seq_len: int = 128, pairs: int = 16, keys: int = 64, values: int = 32) -> None:
        super().__init__("recall", CONTENT + keys + values, seq_len)
        self.pairs, self.keys, self.values = pairs, keys, values

    def batch(self, batch_size, generator):
        b, p = batch_size, self.pairs
        keys = torch.stack([torch.randperm(self.keys, generator=generator)[:p] for _ in range(b)]) + CONTENT
        values = torch.randint(0, self.values, (b, p), generator=generator) + CONTENT + self.keys
        pairs = torch.stack([keys, values], dim=2).reshape(b, 2 * p)
        order = torch.stack([torch.randperm(p, generator=generator) for _ in range(b)])
        q_keys, q_values = keys.gather(1, order), values.gather(1, order)
        queries = torch.stack([q_keys, q_values], dim=2).reshape(b, 2 * p)
        ids = torch.cat([pairs, torch.full((b, 1), SEP), queries], dim=1)
        labels = torch.full_like(ids, -100)
        answer_positions = 2 * p + 1 + 2 * torch.arange(p) + 1
        labels[:, answer_positions] = ids[:, answer_positions]
        return _fill(ids, labels, self.seq_len)


class Parity(Task):
    def __init__(self, seq_len: int = 64, bits: int = 48) -> None:
        super().__init__("parity", CONTENT + 2, seq_len)
        self.bits = bits

    def batch(self, batch_size, generator):
        bits = torch.randint(0, 2, (batch_size, self.bits), generator=generator)
        answer = bits.sum(1) % 2
        ids = torch.cat([bits + CONTENT, torch.full((batch_size, 1), QUERY), (answer + CONTENT)[:, None]], dim=1)
        labels = torch.full_like(ids, -100)
        labels[:, -1] = ids[:, -1]
        return _fill(ids, labels, self.seq_len)


class MultiHop(Task):
    def __init__(self, seq_len: int = 128, chain: int = 12, hops: int = 3, nodes: int = 48) -> None:
        super().__init__("multihop", CONTENT + nodes, seq_len)
        self.chain, self.hops, self.nodes = chain, hops, nodes

    def batch(self, batch_size, generator):
        b = batch_size
        chains = (
            torch.stack([torch.randperm(self.nodes, generator=generator)[: self.chain + 1] for _ in range(b)])
            + CONTENT
        )
        edges = torch.stack([chains[:, :-1], chains[:, 1:]], dim=2)  # (b, chain, 2)
        order = torch.stack([torch.randperm(self.chain, generator=generator) for _ in range(b)])
        shuffled = edges.gather(1, order[..., None].expand(-1, -1, 2)).reshape(b, 2 * self.chain)
        hops = torch.randint(1, self.hops + 1, (b,), generator=generator)
        start = torch.randint(0, self.chain - self.hops + 1, (b,), generator=generator)
        answer = chains[torch.arange(b), start + hops]
        tail = torch.stack([torch.full((b,), QUERY), chains[torch.arange(b), start], HOP_BASE + hops, answer], dim=1)
        ids = torch.cat([shuffled, tail], dim=1)
        labels = torch.full_like(ids, -100)
        labels[:, -1] = ids[:, -1]
        return _fill(ids, labels, self.seq_len)


class CopyNoise(Task):
    def __init__(self, seq_len: int = 160, span: int = 12, noise: int = 96, alphabet: int = 32) -> None:
        super().__init__("copynoise", CONTENT + alphabet, seq_len)
        self.span, self.noise, self.alphabet = span, noise, alphabet

    def batch(self, batch_size, generator):
        b = batch_size
        span = torch.randint(0, self.alphabet, (b, self.span), generator=generator) + CONTENT
        noise = torch.randint(0, 32, (b, self.noise), generator=generator) + NOISE_BASE
        ids = torch.cat([span, noise, torch.full((b, 1), COPY), span], dim=1)
        labels = torch.full_like(ids, -100)
        labels[:, -self.span :] = span
        return _fill(ids, labels, self.seq_len)


TASKS: dict[str, type[Task]] = {"recall": Recall, "parity": Parity, "multihop": MultiHop, "copynoise": CopyNoise}
FAMILIES = ("transformer", "helix", "anansi")


def tiny_config(family: str, vocab: int):
    """Comparable toy models: width 128, four layers of compute (Anansi: 1 + 2 x 1 loop... grown by loops)."""
    common = {"vocab_size": vocab}
    if family == "transformer":
        return arch.make_config(
            "t-350m", hidden_size=128, intermediate_size=352, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=4096, **common,
        )  # fmt: skip
    helix_fields = {
        "hidden_size": 128, "intermediate_size": 352, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 32, "num_recurrent_heads": 2, "block_size": 8, "local_blocks": 2, "index_branching": 2,
        "index_topk": 2, "landmark_dim": 16, "attention_backend": "sdpa", "loss_chunk_tokens": 0, **common,
    }  # fmt: skip
    if family == "helix":
        return arch.make_config("20m", num_hidden_layers=4, **helix_fields)
    return arch.make_config(
        "a-20m", prelude_layers=1, core_layers=2, coda_layers=1, mean_loops=2.0, min_loops=1, max_loops=3,
        num_streams=2, ngram_memory_size=2048, mtp_depth=0, **helix_fields,
    )  # fmt: skip


def accuracy(model, task: Task, generator: torch.Generator, batches: int = 4, batch_size: int = 32) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(batches):
            ids, labels = task.batch(batch_size, generator)
            logits = model(ids).logits
            pred = logits[:, :-1].argmax(-1)
            target = labels[:, 1:]
            mask = target != -100
            correct += int((pred[mask] == target[mask]).sum())
            total += int(mask.sum())
    return correct / max(total, 1)


def train_and_score(family: str, task: Task, steps: int, batch_size: int, lr: float, seed: int, log=None) -> dict:
    torch.manual_seed(seed)
    config = tiny_config(family, task.vocab)
    model = arch.construct(config)
    arch.initialize(model, config, seed)
    from .optim import build_optimizer

    optimizer = build_optimizer(model, "muon", lr=lr, weight_decay=0.01)
    generator = torch.Generator().manual_seed(seed + 1)
    started = time.time()
    model.train()
    for step in range(steps):
        warm = min(1.0, (step + 1) / max(1, steps // 10))
        cool = 0.5 * (1 + math.cos(math.pi * step / steps))
        for group in optimizer.param_groups:
            group["lr"] = lr * warm * max(cool, 0.1) * group.get("lr_mult", 1.0)
        ids, labels = task.batch(batch_size, generator)
        out = model(ids, labels=labels)
        optimizer.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if log and (step + 1) % max(1, steps // 5) == 0:
            log(f"  {family:11} {task.name:9} step {step + 1:4}/{steps} loss {out.loss.item():.3f}")
    held_out = torch.Generator().manual_seed(seed + 99)
    return {
        "family": family, "task": task.name, "accuracy": accuracy(model, task, held_out),
        "params": sum(p.numel() for p in model.parameters()), "seconds": time.time() - started,
    }  # fmt: skip


def compare(
    tasks=tuple(TASKS),
    families=FAMILIES,
    steps: int = 300,
    batch_size: int = 32,
    lr: float = 2e-3,
    seed: int = 0,
    log=print,
) -> list[dict]:
    results = []
    for name in tasks:
        task = TASKS[name]()
        for family in families:
            results.append(train_and_score(family, task, steps, batch_size, lr, seed, log))
            log(f"{family:11} {task.name:9} accuracy {results[-1]['accuracy']:.1%}  ({results[-1]['seconds']:.0f}s)")
    return results


def describe_results(results: list[dict]) -> str:
    tasks = list(dict.fromkeys(r["task"] for r in results))
    families = list(dict.fromkeys(r["family"] for r in results))
    rows = [f"{'task':10} " + " ".join(f"{f:>12}" for f in families)]
    for t in tasks:
        cells = []
        for f in families:
            hit = next((r for r in results if r["task"] == t and r["family"] == f), None)
            cells.append(f"{hit['accuracy']:>11.1%} " if hit else f"{'-':>12}")
        rows.append(f"{t:10} " + " ".join(cells))
    return "\n".join(rows)


_ = F  # noqa: F841 -- torch.nn.functional is imported for callers extending the tasks
