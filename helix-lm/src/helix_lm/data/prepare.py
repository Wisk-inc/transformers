# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
From the Hub to uint16 shards.

``prepare_source`` streams one source file by file -- download, filter, tokenize, write -- with a pool of
worker processes, each file becoming its own run of shards, so a stopped preparation resumes at the next
file and nothing is ever half-written. A shard is a flat ``uint16`` array of token ids with documents
closed by ``<|endoftext|>`` and a JSON sidecar saying how many tokens and documents it holds; the loader
memory-maps them. Per-source manifests record which files are done and how many tokens exist.

Budgets are in tokens: a source stops once its shards hold its share of the run's total. Files are
visited in a seeded shuffle, so a budget that stops early still saw the whole dataset's spread (web
crawls are stored by date; the first files alone would be one month of the internet).
"""

from __future__ import annotations

import contextlib
import gzip
import io
import json
import os
import random
import shutil
import tempfile
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from ..tokenizer import HelixTokenizer
from .catalog import PRETRAIN_SOURCES, Source, verify_sources
from .identity import identity_documents
from .jamaica import is_about_caribbean, is_about_jamaica
from .renderers import RENDERERS

DATA_EXTENSIONS = (".parquet", ".jsonl", ".jsonl.gz", ".jsonl.zst", ".jsonl.xz", ".json.gz", ".json")
FALLBACK_COLUMNS = ("text", "content", "code", "body", "document")
DEFAULT_SHARD_TOKENS = 100_000_000  # 200 MB of uint16


# ------------------------------------------------------------------------------------- shard writing --
class ShardWriter:
    """Accumulates token ids and writes ``<prefix>-<n>.bin`` + ``.json`` every ``shard_tokens``."""

    def __init__(self, directory: Path, prefix: str, shard_tokens: int = DEFAULT_SHARD_TOKENS) -> None:
        self.directory, self.prefix, self.shard_tokens = Path(directory), prefix, shard_tokens
        self.directory.mkdir(parents=True, exist_ok=True)
        self.buffer: list[np.ndarray] = []
        self.buffered = 0
        self.documents = 0
        self.bytes = 0
        self.written: list[dict] = []

    def add(self, ids: list[int], nbytes: int) -> None:
        self.buffer.append(np.asarray(ids, dtype=np.uint16))
        self.buffered += len(ids)
        self.documents += 1
        self.bytes += nbytes
        if self.buffered >= self.shard_tokens:
            self.flush()

    def flush(self) -> None:
        if not self.buffered:
            return
        tokens = np.concatenate(self.buffer)
        index = len(self.written)
        stem = self.directory / f"{self.prefix}-{index:05d}"
        part = stem.with_suffix(".bin.part")
        tokens.tofile(part)
        part.rename(stem.with_suffix(".bin"))
        meta = {"tokens": int(tokens.size), "documents": self.documents, "bytes": self.bytes, "dtype": "uint16"}
        stem.with_suffix(".json").write_text(json.dumps(meta))
        self.written.append({"file": stem.with_suffix(".bin").name, **meta})
        self.buffer, self.buffered, self.documents, self.bytes = [], 0, 0, 0

    @property
    def total_tokens(self) -> int:
        return sum(w["tokens"] for w in self.written) + self.buffered


def shard_files(directory: Path) -> list[Path]:
    return sorted(p for p in Path(directory).glob("*.bin") if p.with_suffix(".json").exists())


def source_tokens(directory: Path) -> int:
    return sum(json.loads(p.with_suffix(".json").read_text())["tokens"] for p in shard_files(directory))


# --------------------------------------------------------------------------------------- reading rows --
@contextlib.contextmanager
def _open_text(path: Path):
    """A text handle over a plain, gzip or zstd newline-delimited JSON file."""
    if path.name.endswith(".zst"):
        import zstandard

        with open(path, "rb") as raw:
            yield io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8")
    elif path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield handle
    elif path.name.endswith(".xz"):
        import lzma

        with lzma.open(path, "rt", encoding="utf-8") as handle:
            yield handle
    else:
        with open(path, encoding="utf-8") as handle:
            yield handle


def iter_rows(path: Path, columns: tuple[str, ...] | None = None) -> Iterator[dict]:
    """Whole rows (as dicts) from a parquet or newline-delimited JSON file."""
    if path.name.endswith(".parquet"):
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        wanted = [c for c in (columns or parquet.schema_arrow.names) if c in parquet.schema_arrow.names] or None
        for batch in parquet.iter_batches(batch_size=256, columns=wanted):
            yield from batch.to_pylist()
        return
    with _open_text(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_documents(path: Path, text: str | tuple[str, ...], render: str | None = None) -> Iterator[str]:
    """
    Documents from one data file: parquet, or newline-delimited JSON (plain, gzip, zstd or xz). With ``render``,
    each row goes through that renderer instead of being read as text. A file without any of the named
    columns falls back to the usual ones (``text``, ``content`` ...) and, failing that, yields nothing.
    """
    if render:
        renderer = RENDERERS[render]
        for row in iter_rows(path):
            document = renderer(row)
            if document:
                yield document
        return
    columns = (text,) if isinstance(text, str) else tuple(text)
    if path.name.endswith(".parquet"):
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        wanted = [c for c in columns if c in available] or [c for c in FALLBACK_COLUMNS if c in available][:1]
        if not wanted:
            return
        for batch in parquet.iter_batches(batch_size=512, columns=wanted):
            rows = batch.to_pydict()
            for i in range(batch.num_rows):
                parts = [rows[c][i] for c in wanted]
                yield "\n\n".join(str(p) for p in parts if p)
        return
    with _open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            parts = [row.get(c) for c in columns]
            if not any(parts):
                parts = [next((row[c] for c in FALLBACK_COLUMNS if row.get(c)), None)]
            if any(parts):
                yield "\n\n".join(str(p) for p in parts if p)


# -------------------------------------------------------------------------------------------- filters --
def keep_document(text: str, kind: str) -> bool:
    """
    Cheap quality gates, by kind. Code: no minified or generated monsters. Web: not a stub. Patois: anything
    with words in it -- there is too little of it to be choosy.
    """
    n = len(text)
    if kind == "code":
        if n < 64 or n > 1_000_000:
            return False
        lines = text.split("\n")
        longest = max(len(line) for line in lines)
        if longest > 10_000 or n / max(len(lines), 1) > 400:
            return False
        alnum = sum(ch.isalnum() for ch in text[:20_000]) / max(min(n, 20_000), 1)
        return alnum >= 0.25
    if kind in ("patois", "jamaica"):
        return n >= 20
    return n >= 100


# --------------------------------------------------------------------------------------- the workers --
@dataclass
class FileTask:
    source_name: str
    repo: str
    filename: str
    text: str | tuple[str, ...]
    kind: str
    tag_jamaica: bool
    out_dir: str
    jamaica_dir: str
    tokenizer_dir: str
    shard_tokens: int
    local_path: str | None = None
    token: str | None = None
    render: str | None = None
    caribbean_dir: str | None = None


def _worker_init() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _download(task: FileTask, cache: Path) -> Path:
    if task.local_path:
        return Path(task.local_path)
    from huggingface_hub import hf_hub_download

    for attempt in range(5):
        try:
            return Path(
                hf_hub_download(task.repo, task.filename, repo_type="dataset", cache_dir=cache, token=task.token)
            )
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def process_file(task: FileTask) -> dict:
    """Download one file, tokenize what passes the filter, write its shards. Runs in a worker process."""
    tokenizer = HelixTokenizer.from_pretrained(task.tokenizer_dir)
    stem = Path(task.filename).name.rsplit(".", 1)[0].replace(".jsonl", "")
    prefix = f"{task.source_name}-{stem}"
    writer = ShardWriter(Path(task.out_dir), prefix, task.shard_tokens)
    jamaica = ShardWriter(Path(task.jamaica_dir), f"jamaica_en-{task.source_name}-{stem}", task.shard_tokens)
    caribbean = ShardWriter(
        Path(task.caribbean_dir or task.jamaica_dir), f"caribbean_en-{task.source_name}-{stem}", task.shard_tokens
    )
    cache = Path(tempfile.mkdtemp(prefix="helix-dl-"))
    seen = kept = 0
    try:
        path = _download(task, cache)
        batch: list[str] = []

        def flush_batch() -> None:
            for text, ids in zip(batch, tokenizer.encode_batch(batch), strict=True):
                writer.add(ids, len(text.encode("utf-8", "ignore")))
                if task.tag_jamaica:
                    if is_about_jamaica(text):
                        jamaica.add(ids, len(text))
                    elif is_about_caribbean(text):
                        caribbean.add(ids, len(text))
            batch.clear()

        for text in iter_documents(path, task.text, task.render):
            seen += 1
            if not keep_document(text, task.kind):
                continue
            kept += 1
            batch.append(text)
            if len(batch) >= 256:
                flush_batch()
        if batch:
            flush_batch()
        writer.flush()
        jamaica.flush()
        caribbean.flush()
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    return {
        "file": task.filename,
        "documents_seen": seen,
        "documents_kept": kept,
        "tokens": sum(w["tokens"] for w in writer.written),
        "jamaica_tokens": sum(w["tokens"] for w in jamaica.written) + sum(w["tokens"] for w in caribbean.written),
        "shards": [w["file"] for w in writer.written],
        "jamaica_shards": [w["file"] for w in jamaica.written],
    }


# ---------------------------------------------------------------------------------------- the driver --
def list_source_files(source: Source, token: str | None = None) -> list[str]:
    """Data files of a source on the Hub, under its path prefixes, in a seeded shuffle."""
    from huggingface_hub import HfApi

    files = [
        f
        for f in HfApi(token=token).list_repo_files(source.repo, repo_type="dataset")
        if f.endswith(DATA_EXTENSIONS) and (not source.paths or f.startswith(source.paths))
    ]
    if source.name == "patois-madlad":
        files = [f for f in files if "clean" in Path(f).name]
    files = sorted(files)
    random.Random(f"helix:{source.name}").shuffle(files)
    return files


def prepare_source(
    source: Source,
    data_dir: str | Path,
    tokenizer_dir: str | Path,
    token_budget: int | None,
    workers: int = 8,
    shard_tokens: int = DEFAULT_SHARD_TOKENS,
    token: str | None = None,
    files: list[str] | None = None,
    local_files: list[Path] | None = None,
    min_free_gb: float = 300.0,
    log=print,
) -> dict:
    """
    Prepare one source into ``data_dir/tokens/<name>/``. Resumes: files named in the manifest are skipped.
    Stops when the shards hold ``token_budget`` tokens (``None``: the whole source), or when the volume has
    less than ``min_free_gb`` left. ``local_files`` reads files on disk instead of the Hub (tests, private data).
    """
    data_dir = Path(data_dir)
    out_dir = data_dir / "tokens" / source.name
    jamaica_dir = data_dir / "tokens" / "jamaica_en"
    caribbean_dir = data_dir / "tokens" / "caribbean_en"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("source", source.name)
    manifest.setdefault("repo", source.repo)
    manifest.setdefault("done", [])
    manifest.setdefault("complete", False)

    if local_files is not None:
        todo = [(str(p), str(p)) for p in local_files]
    else:
        todo = [(f, None) for f in (files if files is not None else list_source_files(source, token))]
    done = set(manifest["done"])
    todo = [(f, local) for f, local in todo if f not in done]
    have = source_tokens(out_dir)
    log(
        f"[{source.name}] {len(done)} files done, {len(todo)} to go, {have / 1e9:.2f}B tokens on disk"
        + (f", budget {token_budget / 1e9:.2f}B" if token_budget else "")
    )
    if not todo or (token_budget is not None and have >= token_budget):
        manifest["complete"] = not todo
        manifest["tokens"] = have
        manifest_path.write_text(json.dumps(manifest, indent=1))
        return manifest

    def tasks() -> Iterable[FileTask]:
        for filename, local in todo:
            yield FileTask(
                source.name, source.repo, filename, source.text, source.kind, source.tag_jamaica, str(out_dir),
                str(jamaica_dir), str(tokenizer_dir), shard_tokens, local, token, source.render, str(caribbean_dir),
            )  # fmt: skip

    started = time.time()
    context = get_context("fork") if os.name == "posix" else get_context("spawn")
    pending = list(tasks())
    with context.Pool(max(1, workers), initializer=_worker_init) as pool:
        while pending:
            free_gb = shutil.disk_usage(data_dir).free / 1e9
            if free_gb < min_free_gb:
                log(
                    f"[{source.name}] only {free_gb:.0f} GB free on the volume: stopping here (keeps "
                    f"{min_free_gb:.0f} GB for checkpoints); free space or a bigger volume continues it"
                )
                break
            wave, pending = pending[: max(1, workers)], pending[max(1, workers) :]
            for result in pool.imap_unordered(process_file, wave):
                manifest["done"].append(result["file"])
                have += result["tokens"]
                manifest["tokens"] = have
                manifest_path.write_text(json.dumps(manifest, indent=1))
                rate = have / max(time.time() - started, 1e-6)
                log(
                    f"[{source.name}] {Path(result['file']).name}: {result['documents_kept']:,}/"
                    f"{result['documents_seen']:,} docs, {result['tokens'] / 1e6:.1f}M tokens"
                    + (f", {result['jamaica_tokens'] / 1e6:.2f}M about Jamaica" if result["jamaica_tokens"] else "")
                    + f" | total {have / 1e9:.3f}B @ {rate / 1e6:.1f}M tok/s"
                )
            if token_budget is not None and have >= token_budget:
                log(f"[{source.name}] budget reached: {have / 1e9:.2f}B tokens")
                break
    manifest["complete"] = not pending and (token_budget is None or have < token_budget)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    return manifest


def prepare_identity(data_dir: str | Path, tokenizer_dir: str | Path, log=print) -> int:
    """The identity documents, tokenized into their own small source."""
    out_dir = Path(data_dir) / "tokens" / "identity"
    if shard_files(out_dir):
        return source_tokens(out_dir)
    tokenizer = HelixTokenizer.from_pretrained(tokenizer_dir)
    writer = ShardWriter(out_dir, "identity")
    docs = list(identity_documents())
    for text, ids in zip(docs, tokenizer.encode_batch(docs), strict=True):
        writer.add(ids, len(text))
    writer.flush()
    (out_dir / "manifest.json").write_text(
        json.dumps({"source": "identity", "complete": True, "tokens": writer.total_tokens})
    )
    log(f"[identity] {len(docs)} documents, {writer.total_tokens:,} tokens")
    return writer.total_tokens


def plan_budgets(sources: Iterable[Source], total_tokens: int, overshoot: float = 1.15) -> dict[str, int]:
    """
    Tokens to prepare per source: its normalized weight times the total, with a margin so the loader can
    hold every source to one pass. Derived, generated and local sources are not downloaded and get no budget.
    """
    downloadable = [s for s in sources if not s.local]
    total_weight = sum(s.weight for s in downloadable)
    return {s.name: int(total_tokens * overshoot * s.weight / total_weight) for s in downloadable}


def prepare_all(
    data_dir: str | Path,
    tokenizer_dir: str | Path,
    total_tokens: int,
    sources: tuple[Source, ...] = PRETRAIN_SOURCES,
    workers: int = 8,
    token: str | None = None,
    only: tuple[str, ...] = (),
    min_free_gb: float = 300.0,
    log=print,
) -> dict[str, int]:
    """
    The whole pretraining mixture for a run of ``total_tokens``. Verifies access first; a gated source the
    token cannot read is dropped and its weight handed to its ungated stand-in (``starcoder`` ->
    ``github-code``). Returns tokens on disk per source.
    """
    data_dir = Path(data_dir)
    report = {r["name"]: r for r in verify_sources(sources, token)}
    usable = []
    weights = {s.name: s.weight for s in sources}
    for s in sources:
        if s.local:
            continue
        if not report[s.name]["ok"]:
            log(f"!! {s.name}: {report[s.name]['status']} -- skipped")
            if s.gated and s.kind == "code" and "github-code" in weights:
                weights["github-code"] += weights[s.name]
                log(f"   github-code takes {s.name}'s share of the code mixture")
            continue
        usable.append(s)
    usable = [Source(**{**s.__dict__, "weight": weights[s.name]}) for s in usable]
    budgets = plan_budgets(usable, total_tokens)
    (data_dir / "plan.json").write_text(json.dumps({"total_tokens": total_tokens, "budgets": budgets}, indent=1))
    prepared = {}
    # Small sources first: the Patois corpora and the math are ready in minutes, and the GPU can start on a
    # mixture that already contains them while the web and code keep streaming.
    order = sorted(usable, key=lambda s: report[s.name]["gb"])
    for s in order:
        if only and s.name not in only:
            continue
        manifest = prepare_source(
            s, data_dir, tokenizer_dir, budgets[s.name], workers, token=token, min_free_gb=min_free_gb, log=log
        )
        prepared[s.name] = manifest.get("tokens", 0)
    prepared["identity"] = prepare_identity(data_dir, tokenizer_dir, log)
    for s in sources:  # data made on this machine: data_dir/local/<name>/*.jsonl (the crawler, the synthesizer)
        folder = data_dir / "local" / s.name
        if s.repo == "(local)" and folder.exists() and (not only or s.name in only):
            local_files = sorted(p for p in folder.iterdir() if p.name.endswith(DATA_EXTENSIONS))
            if local_files:
                manifest = prepare_source(
                    s,
                    data_dir,
                    tokenizer_dir,
                    None,
                    workers,
                    token=token,
                    local_files=local_files,
                    min_free_gb=min_free_gb,
                    log=log,
                )
                prepared[s.name] = manifest.get("tokens", 0)
    for derived in ("jamaica_en", "caribbean_en"):
        folder = data_dir / "tokens" / derived
        if folder.exists():
            prepared[derived] = source_tokens(folder)
            (folder / "manifest.json").write_text(
                json.dumps({"source": derived, "complete": True, "tokens": prepared[derived]})
            )
    (data_dir / "prepared.json").write_text(json.dumps(prepared, indent=1))
    return prepared


def sample_text(
    data_dir: str | Path,
    sources: tuple[Source, ...] = PRETRAIN_SOURCES,
    bytes_per_source: int = 400_000_000,
    token: str | None = None,
    log=print,
) -> Path:
    """
    A text sample of the mixture for training the tokenizer, before any tokenizer exists: up to
    ``bytes_per_source`` from each downloadable source (the Patois corpora in full, they are smaller than
    that) plus the identity documents, one document per line with newlines escaped, into
    ``data_dir/tokenizer_sample.txt``.
    """
    from huggingface_hub import hf_hub_download

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / "tokenizer_sample.txt"
    cache = data_dir / "tmp_sample"
    with open(out, "w", encoding="utf-8") as handle:
        for doc in identity_documents():
            handle.write(doc.replace("\n", "\\n") + "\n")
        for source in sources:
            if source.local:
                continue
            try:
                files = list_source_files(source, token)
            except Exception as error:
                log(f"[sample] {source.name}: {type(error).__name__}: {str(error)[:80]} -- skipped")
                continue
            written = 0
            for filename in files:
                if written >= bytes_per_source:
                    break
                try:
                    path = hf_hub_download(source.repo, filename, repo_type="dataset", cache_dir=cache, token=token)
                    for text in iter_documents(Path(path), source.text, source.render):
                        if not keep_document(text, source.kind):
                            continue
                        handle.write(text.replace("\n", "\\n") + "\n")
                        written += len(text)
                        if written >= bytes_per_source:
                            break
                finally:
                    shutil.rmtree(cache, ignore_errors=True)
            log(f"[sample] {source.name}: {written / 1e6:.0f} MB")
    return out


def describe_prepared(data_dir: str | Path, sources: tuple[Source, ...] = PRETRAIN_SOURCES) -> str:
    """Tokens on disk per source, against the plan."""
    data_dir = Path(data_dir)
    plan = json.loads((data_dir / "plan.json").read_text()) if (data_dir / "plan.json").exists() else {"budgets": {}}
    lines = [f"{'source':18} {'tokens':>10} {'budget':>10}  shards  complete"]
    total = 0
    for folder in sorted((data_dir / "tokens").glob("*")) if (data_dir / "tokens").exists() else []:
        tokens = source_tokens(folder)
        total += tokens
        manifest = json.loads((folder / "manifest.json").read_text()) if (folder / "manifest.json").exists() else {}
        budget = plan["budgets"].get(folder.name)
        lines.append(
            f"{folder.name:18} {tokens / 1e9:9.3f}B {(budget or 0) / 1e9:9.3f}B  {len(shard_files(folder)):6}  "
            f"{'yes' if manifest.get('complete') else 'no'}"
        )
    lines.append(f"{'total':18} {total / 1e9:9.3f}B")
    return "\n".join(lines)
