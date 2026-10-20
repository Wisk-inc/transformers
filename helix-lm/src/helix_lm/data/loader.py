# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Reading the shards back: a weighted, deterministic, resumable stream of training sequences.

Each source is a run of memory-mapped uint16 shards read as one long tape. Every rank owns a stripe of
every tape (its start is ``rank / world_size`` of the way in), reads it sequentially in ``seq_len`` steps,
and wraps -- which counts as an epoch. A batch draws its sources from a seeded generator, so the mixture is
honoured in expectation and the whole stream is a pure function of ``(seed, rank, step)``: save the cursors
and the generator state, load them, and the next batch is the same batch.

The last ``holdout_tokens`` of every tape are never read for training; ``eval_batches`` reads them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .prepare import shard_files


class TokenSource:
    """One source's shards as a single addressable tape of uint16 ids."""

    def __init__(self, name: str, directory: str | Path, holdout_tokens: int = 0) -> None:
        self.name = name
        self.directory = Path(directory)
        paths = shard_files(self.directory)
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        if not self.shards:
            raise FileNotFoundError(f"no shards in {self.directory}")
        # SFT shards carry a parallel `.mask` tape: one byte per token, 1 where the loss applies.
        masks = [p.with_suffix(".mask") for p in paths]
        self.masks = (
            [np.memmap(m, dtype=np.uint8, mode="r") for m in masks] if all(m.exists() for m in masks) else None
        )
        self.sizes = np.array([s.shape[0] for s in self.shards], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.total = int(self.offsets[-1])
        self.holdout = min(holdout_tokens, self.total // 2)
        self.train_tokens = self.total - self.holdout

    def read(self, start: int, length: int, tapes=None) -> np.ndarray:
        """``length`` ids from absolute position ``start``, across shard boundaries (no wrapping)."""
        tapes = self.shards if tapes is None else tapes
        out = np.empty(length, dtype=tapes[0].dtype)
        filled = 0
        shard = int(np.searchsorted(self.offsets, start, side="right") - 1)
        position = start - int(self.offsets[shard])
        while filled < length:
            take = min(length - filled, int(self.sizes[shard]) - position)
            out[filled : filled + take] = tapes[shard][position : position + take]
            filled += take
            shard, position = shard + 1, 0
        return out

    def read_mask(self, start: int, length: int) -> np.ndarray:
        """The loss mask for the same window: all ones unless this source carries a mask tape."""
        if self.masks is None:
            return np.ones(length, dtype=np.uint8)
        return self.read(start, length, self.masks)

    def read_wrapped(self, start: int, length: int, limit: int) -> tuple[np.ndarray, np.ndarray, int, bool]:
        """
        ``length`` ids (and their mask) from ``start`` inside ``[0, limit)``, wrapping to 0 at ``limit``.
        Returns the ids, the mask, the next position, and whether the read wrapped (one epoch of this
        stripe's tape finished).
        """
        if start + length <= limit:
            return self.read(start, length), self.read_mask(start, length), start + length, False
        head, head_mask = self.read(start, limit - start), self.read_mask(start, limit - start)
        rest = length - head.shape[0]
        tail, tail_mask = self.read(0, rest), self.read_mask(0, rest)
        return np.concatenate([head, tail]), np.concatenate([head_mask, tail_mask]), rest, True


class MixtureLoader:
    """
    Draws micro-batches of ``(seq_len + 1)``-token windows from several sources by weight.

    Args:
        sources: ``{name: TokenSource}``.
        weights: ``{name: relative weight}``; normalized here.
        seq_len: tokens per training sequence (the window read is one longer, for the labels).
        micro_batch: sequences per call to :meth:`next_batch`.
        rank / world_size: this process's stripe of every tape.
        seed: with rank and the step counter, determines everything.
        max_epochs: ``{name: cap}``; with ``total_tokens`` known, a weight is lowered so that its expected
            draws stay under ``cap`` passes over its train tokens, and the slack goes to the other sources.
        total_tokens: tokens the run will consume in total, for the epoch caps.
    """

    def __init__(
        self,
        sources: dict[str, TokenSource],
        weights: dict[str, float],
        seq_len: int,
        micro_batch: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        max_epochs: dict[str, float] | None = None,
        total_tokens: int | None = None,
    ) -> None:
        self.names = [n for n in sources if weights.get(n, 0) > 0]
        if not self.names:
            raise ValueError("no source with a positive weight")
        self.sources = {n: sources[n] for n in self.names}
        self.seq_len, self.micro_batch = seq_len, micro_batch
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.weights = self.resolve_weights(
            {n: weights[n] for n in self.names}, max_epochs or {}, total_tokens, world_size
        )
        self.probabilities = np.array([self.weights[n] for n in self.names], dtype=np.float64)
        self.probabilities /= self.probabilities.sum()
        # Each rank's stripe: start a fraction of the way into every tape, on a sequence boundary.
        self.cursors = {
            n: ((s.train_tokens * rank) // world_size // seq_len) * seq_len for n, s in self.sources.items()
        }
        self.epochs = dict.fromkeys(self.names, 0)
        self.drawn = dict.fromkeys(self.names, 0)
        self.step = 0
        self.generator = np.random.default_rng([seed, rank])
        self.prior = dict(self.weights)
        self.max_epochs, self.total_tokens = dict(max_epochs or {}), total_tokens
        self.last_eval: dict[str, float] = {}

    def resolve_weights(self, weights, max_epochs, total_tokens, world_size) -> dict[str, float]:
        """Lower the weight of any source that would exceed its epoch cap; hand the slack to the rest."""
        if not total_tokens:
            return dict(weights)
        weights = dict(weights)
        total = sum(weights.values())
        capped: dict[str, float] = {}
        for _ in range(len(weights)):
            changed = False
            free = {n: w for n, w in weights.items() if n not in capped}
            free_total = sum(free.values())
            remaining = total - sum(capped.values())
            for name, weight in free.items():
                cap = max_epochs.get(name)
                if cap is None:
                    continue
                share = remaining * weight / free_total if free_total else 0
                allowed = cap * self.sources[name].train_tokens / total_tokens * total
                if share > allowed:
                    capped[name] = allowed
                    changed = True
            if not changed:
                break
        free = {n: w for n, w in weights.items() if n not in capped}
        free_total = sum(free.values())
        remaining = total - sum(capped.values())
        for name in free:
            weights[name] = remaining * free[name] / free_total if free_total else 0.0
        weights.update(capped)
        return weights

    def reweight(self, eval_losses: dict[str, float], strength: float = 1.0) -> str:
        """
        Online mixture reweighting from learning progress (the idea of ADO / Aioli, in its simplest form): a
        source whose held-out loss fell faster than average since the last evaluation gets more of the next
        batches, one whose loss stalled gets less. New weight ``w_i = prior_i * (progress_i / mean progress) ^
        strength``, clipped to ``[0.5, 2] x prior`` so no source is starved or allowed to dominate, then the
        epoch caps are re-applied. The first call only records the losses. Returns a short description of the
        change, empty when nothing changed.
        """
        current = {n: float(v) for n, v in eval_losses.items() if n in self.names}
        if not self.last_eval:
            self.last_eval = current
            return ""
        progress = {n: max(self.last_eval.get(n, v) - v, 0.0) for n, v in current.items()}
        self.last_eval = current
        mean = sum(progress.values()) / max(len(progress), 1)
        if mean <= 0:
            return ""
        proposed = {}
        for n in self.names:
            score = max(0.25, min(4.0, progress.get(n, mean) / mean)) ** strength
            proposed[n] = min(2.0 * self.prior[n], max(0.5 * self.prior[n], self.prior[n] * score))
        self.set_weights(proposed)
        total = sum(self.weights.values())
        return ", ".join(f"{n} {self.weights[n] / total:.1%}" for n in self.names)

    def set_weights(self, weights: dict[str, float]) -> None:
        self.weights = self.resolve_weights(
            {n: weights[n] for n in self.names}, self.max_epochs, self.total_tokens, self.world_size
        )
        self.probabilities = np.array([self.weights[n] for n in self.names], dtype=np.float64)
        self.probabilities /= self.probabilities.sum()

    def expected_epochs(self, total_tokens: int) -> dict[str, float]:
        """How many passes over each source a run of ``total_tokens`` makes with the resolved weights."""
        total = sum(self.weights.values())
        return {n: total_tokens * self.weights[n] / total / self.sources[n].train_tokens for n in self.names}

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ``(input_ids (micro_batch, seq_len + 1) int64, source_index (micro_batch,) int64,
        loss_mask (micro_batch, seq_len + 1) bool)``. The mask is all ``True`` for pretraining sources.
        """
        choices = self.generator.choice(len(self.names), size=self.micro_batch, p=self.probabilities)
        rows, masks, ids = [], [], []
        for choice in choices:
            name = self.names[choice]
            source = self.sources[name]
            window, mask, self.cursors[name], wrapped = source.read_wrapped(
                self.cursors[name], self.seq_len + 1, source.train_tokens
            )
            # Stay on the sequence grid after a wrap so the stripe is re-read in the same windows.
            if wrapped:
                self.epochs[name] += 1
                self.cursors[name] = (self.cursors[name] // self.seq_len) * self.seq_len
            rows.append(torch.from_numpy(window.astype(np.int64)))
            masks.append(torch.from_numpy(mask.astype(bool)))
            ids.append(choice)
            self.drawn[name] += 1
        self.step += 1
        return torch.stack(rows), torch.tensor(ids, dtype=torch.int64), torch.stack(masks)

    def eval_batches(self, name: str, count: int, micro_batch: int | None = None):
        """Deterministic windows from the held-out tail of one source."""
        source = self.sources[name]
        micro_batch = micro_batch or self.micro_batch
        if source.holdout < self.seq_len + 1:
            return
        start = source.train_tokens
        limit = source.total - (self.seq_len + 1)
        position = start + ((limit - start) * self.rank // self.world_size // self.seq_len) * self.seq_len
        for _ in range(count):
            rows = []
            for _ in range(micro_batch):
                if position > limit:
                    position = start
                rows.append(torch.from_numpy(source.read(position, self.seq_len + 1).astype(np.int64)))
                position += self.seq_len
            yield torch.stack(rows)

    def state_dict(self) -> dict:
        return {
            "cursors": dict(self.cursors),
            "epochs": dict(self.epochs),
            "drawn": dict(self.drawn),
            "step": self.step,
            "generator": self.generator.bit_generator.state,
            "names": list(self.names),
            "weights": dict(self.weights),
            "last_eval": dict(self.last_eval),
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Continue where the checkpoint stopped. Sources that were prepared after the run started (the data keeps
        streaming in during a long run) join the mixture with fresh cursors; a source that disappeared is an error,
        because the mixture the checkpoint was trained on can no longer be reproduced.
        """
        lost = [n for n in state["names"] if n not in self.names]
        if lost:
            raise ValueError(f"the mixture lost sources since the checkpoint: {lost} (saved {state['names']})")
        added = [n for n in self.names if n not in state["names"]]
        if added:
            print(f"[loader] sources prepared since the checkpoint join the mixture now: {added}", flush=True)
        self.cursors.update({n: c for n, c in state["cursors"].items() if n in self.names})
        self.epochs.update({n: e for n, e in state["epochs"].items() if n in self.names})
        self.drawn.update({n: d for n, d in state["drawn"].items() if n in self.names})
        self.step = state["step"]
        self.generator.bit_generator.state = state["generator"]
        self.last_eval = dict(state.get("last_eval", {}))
        saved = state.get("weights")
        if saved and all(n in saved for n in self.names):
            self.set_weights({n: saved[n] for n in self.names})

    def describe(self, total_tokens: int | None = None) -> str:
        lines = [f"{'source':18} {'tokens':>10} {'weight':>7}" + (f" {'epochs':>7}" if total_tokens else "")]
        epochs = self.expected_epochs(total_tokens) if total_tokens else {}
        total = sum(self.weights.values())
        for name in self.names:
            lines.append(
                f"{name:18} {self.sources[name].train_tokens / 1e9:9.3f}B {self.weights[name] / total:7.3%}"
                + (f" {epochs[name]:7.2f}" if total_tokens else "")
            )
        return "\n".join(lines)


def open_mixture(
    data_dir: str | Path,
    weights: dict[str, float],
    max_epochs: dict[str, float],
    seq_len: int,
    micro_batch: int,
    rank: int,
    world_size: int,
    seed: int,
    total_tokens: int | None,
    holdout_tokens: int = 2_000_000,
) -> MixtureLoader:
    """Every prepared source under ``data_dir/tokens`` that the mixture names, as a loader."""
    tokens_dir = Path(data_dir) / "tokens"
    sources, present = {}, {}
    for name, weight in weights.items():
        folder = tokens_dir / name
        if weight > 0 and folder.exists() and shard_files(folder):
            sources[name] = TokenSource(name, folder, holdout_tokens)
            present[name] = weight
    if not sources:
        raise FileNotFoundError(f"no prepared sources under {tokens_dir}; run `helix data prepare` first")
    return MixtureLoader(sources, present, seq_len, micro_batch, rank, world_size, seed, max_epochs, total_tokens)


def mixture_from_catalog(sources) -> tuple[dict[str, float], dict[str, float]]:
    """``(weights, max_epochs)`` from a tuple of :class:`Source`."""
    return {s.name: s.weight for s in sources}, {s.name: s.max_epochs for s in sources}


def save_mixture(path: str | Path, weights: dict[str, float], max_epochs: dict[str, float]) -> None:
    Path(path).write_text(json.dumps({"weights": weights, "max_epochs": max_epochs}, indent=1))


def load_mixture(path: str | Path) -> tuple[dict[str, float], dict[str, float]]:
    data = json.loads(Path(path).read_text())
    return data["weights"], data.get("max_epochs", {})
