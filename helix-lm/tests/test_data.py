# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The data pipeline on a tiny local corpus: files become shards, Jamaica is found, the identity is written,
and the loader is deterministic, resumable, striped across ranks, and honours its epoch caps.
"""

import json

import numpy as np
import pytest
import torch
from conftest import JAMAICA

from helix_lm.data.catalog import PRETRAIN_SOURCES, SFT_SOURCES, Source, describe_report, source_by_name
from helix_lm.data.identity import IDENTITY, identity_conversations, identity_documents
from helix_lm.data.jamaica import FACTS, PARISHES, is_about_jamaica, jamaica_score
from helix_lm.data.loader import MixtureLoader, TokenSource, mixture_from_catalog, open_mixture
from helix_lm.data.prepare import (
    ShardWriter,
    describe_prepared,
    iter_documents,
    keep_document,
    plan_budgets,
    prepare_source,
    shard_files,
    source_tokens,
)
from helix_lm.tokenizer import EOS


# ------------------------------------------------------------------------------------------ preparing --
def test_shards_hold_every_document_closed_by_eos(prepared, corpus):
    _, tok_dir, _, _ = corpus
    web_dir = prepared / "tokens" / "web"
    shards = shard_files(web_dir)
    assert len(shards) > 3, "500-token shards should split each file several times"
    docs = sum(json.loads(p.with_suffix(".json").read_text())["documents"] for p in shards)
    assert docs == 302
    tape = np.concatenate([np.fromfile(p, dtype=np.uint16) for p in shards])
    assert (tape == EOS).sum() == 302
    manifest = json.loads((web_dir / "manifest.json").read_text())
    assert manifest["complete"] and len(manifest["done"]) == 3 and manifest["tokens"] == source_tokens(web_dir)


def test_jamaica_documents_are_also_written_to_their_own_source(prepared):
    jamaica = prepared / "tokens" / "jamaica_en"
    assert shard_files(jamaica), "the two Jamaica documents must have been tagged while streaming the web files"
    docs = sum(json.loads(p.with_suffix(".json").read_text())["documents"] for p in shard_files(jamaica))
    assert docs == 2


def test_prepare_resumes_and_skips_done_files(prepared, corpus):
    _, tok_dir, web_files, _ = corpus
    before = source_tokens(prepared / "tokens" / "web")
    web = Source("web", "local", "text", 0.7, kind="web", tag_jamaica=True)
    manifest = prepare_source(web, prepared, tok_dir, None, workers=1, local_files=web_files, log=lambda *_: None)
    assert manifest["complete"] and source_tokens(prepared / "tokens" / "web") == before


def test_identity_source_and_description(prepared):
    identity = prepared / "tokens" / "identity"
    assert source_tokens(identity) > 500
    text = describe_prepared(prepared)
    assert "identity" in text and "jamaica_en" in text and "total" in text


def test_code_filter_drops_minified_and_keeps_code():
    assert keep_document("def f():\n    return 1\n" * 10, "code")
    assert not keep_document("x" * 20_000, "code")  # one 20k-character line
    assert not keep_document("short", "web")
    assert keep_document("Mi deh yah a Kingston an everyting criss.", "patois")


def test_jsonl_reader_and_column_join(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"title": "T", "text": "body"}) + "\n" + json.dumps({"title": "", "text": "only"}) + "\n"
    )
    assert list(iter_documents(path, ("title", "text"))) == ["T\n\nbody", "only"]


def test_shard_writer_is_atomic_and_counts(tmp_path):
    writer = ShardWriter(tmp_path, "x", shard_tokens=10)
    for _ in range(5):
        writer.add([1, 2, 3, 4], 8)
    writer.flush()
    assert writer.total_tokens == 20 and len(writer.written) == 2 and not list(tmp_path.glob("*.part"))


# --------------------------------------------------------------------------------------------- loader --
def make_loader(prepared, rank=0, world_size=1, seed=0, total_tokens=None, max_epochs=None, micro_batch=4):
    sources = {n: TokenSource(n, prepared / "tokens" / n, holdout_tokens=200) for n in ("web", "code", "identity")}
    return MixtureLoader(
        sources, {"web": 0.6, "code": 0.3, "identity": 0.1}, seq_len=32, micro_batch=micro_batch, rank=rank,
        world_size=world_size, seed=seed, max_epochs=max_epochs, total_tokens=total_tokens,
    )  # fmt: skip


def test_batches_have_the_right_shape_and_values(prepared):
    loader = make_loader(prepared)
    ids, src, mask = loader.next_batch()
    assert ids.shape == (4, 33) and ids.dtype == torch.int64 and src.shape == (4,)
    assert mask.shape == (4, 33) and mask.dtype == torch.bool and mask.all()
    assert ids.max() < 2048


def test_the_stream_is_deterministic_and_resumable(prepared):
    a, b = make_loader(prepared, seed=3), make_loader(prepared, seed=3)
    for _ in range(5):
        torch.testing.assert_close(a.next_batch()[0], b.next_batch()[0])
    state = json.loads(json.dumps(a.state_dict(), default=lambda o: o.tolist() if hasattr(o, "tolist") else o))
    c = make_loader(prepared, seed=3)
    c.load_state_dict(state)
    for _ in range(5):
        torch.testing.assert_close(a.next_batch()[0], c.next_batch()[0])
    assert (
        make_loader(prepared, seed=4).next_batch()[0].tolist()
        != make_loader(prepared, seed=3).next_batch()[0].tolist()
    )


def test_ranks_read_different_stripes(prepared):
    first, second = make_loader(prepared, rank=0, world_size=2), make_loader(prepared, rank=1, world_size=2)
    assert first.cursors["web"] != second.cursors["web"]
    assert first.cursors["web"] % 32 == 0 and second.cursors["web"] % 32 == 0


def test_epochs_are_counted_and_tapes_wrap(prepared):
    loader = make_loader(prepared, micro_batch=8)
    for _ in range(600):
        loader.next_batch()
    assert loader.epochs["identity"] >= 1, "the tiny identity source must have wrapped"
    assert all(c < loader.sources[n].train_tokens for n, c in loader.cursors.items())


def test_epoch_caps_lower_a_small_sources_weight(prepared):
    loader = make_loader(prepared, total_tokens=200_000, max_epochs={"identity": 2.0, "code": 1.0})
    epochs = loader.expected_epochs(200_000)
    assert epochs["identity"] <= 2.0 + 1e-6 and epochs["code"] <= 1.0 + 1e-6
    assert abs(sum(loader.weights.values()) - 1.0) < 1e-9
    assert "epochs" in loader.describe(200_000)


def test_holdout_is_never_trained_on_and_eval_reads_it(prepared):
    loader = make_loader(prepared, micro_batch=2)
    web = loader.sources["web"]
    assert web.holdout == 200 and web.train_tokens == web.total - 200
    batches = list(loader.eval_batches("web", 3))
    assert len(batches) == 3 and batches[0].shape == (2, 33)
    for _ in range(50):
        loader.next_batch()
    assert loader.cursors["web"] < web.train_tokens


def test_open_mixture_and_catalog_weights(prepared):
    weights, caps = mixture_from_catalog(PRETRAIN_SOURCES)
    assert abs(sum(weights.values()) - 1.0) < 0.02 and caps["patois-fineweb2"] == 4.0
    loader = open_mixture(prepared, {"web": 1.0, "identity": 0.5, "missing": 0.2}, {}, 16, 2, 0, 1, 0, None, 100)
    assert set(loader.names) == {"web", "identity"}
    with pytest.raises(FileNotFoundError):
        open_mixture(prepared, {"missing": 1.0}, {}, 16, 2, 0, 1, 0, None)


def test_loader_rejects_a_changed_mixture(prepared):
    loader = make_loader(prepared)
    state = loader.state_dict()
    state["names"] = ["web"]
    with pytest.raises(ValueError):
        loader.load_state_dict(state)


# -------------------------------------------------------------------------------------- Jamaica, identity --
def test_jamaica_filter_wants_more_than_one_mention():
    assert is_about_jamaica(JAMAICA[0]) and is_about_jamaica(JAMAICA[1])
    assert not is_about_jamaica("Countries visited: France, Jamaica, Peru and Japan.")
    assert not is_about_jamaica("Manchester United won again; the club from Manchester, England, leads the league.")
    assert jamaica_score("jamaicans in jamaica") == 4
    assert len(PARISHES) == 14 and any("Melissa" in f for f in FACTS)


def test_identity_documents_and_chats_say_who_made_the_model():
    docs = list(identity_documents())
    assert any("CorxLabs" in d and "Jamaica" in d and IDENTITY["architecture"] in d for d in docs)
    chats = list(identity_conversations(50))
    assert len(chats) == 50
    assert all(c[-1]["role"] == "assistant" for c in chats)
    assert sum("CorxLabs" in c[-1]["content"] for c in chats) > 10


def test_catalog_shapes():
    assert source_by_name("fineweb-edu").tag_jamaica and source_by_name("starcoder").gated
    assert abs(sum(s.weight for s in SFT_SOURCES) - 1.0) < 1e-9
    budgets = plan_budgets(PRETRAIN_SOURCES, 1_000_000_000)
    assert "identity" not in budgets and "jamaica_en" not in budgets and budgets["fineweb-edu"] > budgets["finemath"]
    report = [{"name": "x", "repo": "r", "ok": False, "files": 0, "gb": 0.0, "status": "NOT FOUND"}]
    assert "NOT FOUND" in describe_report(report)
    with pytest.raises(KeyError):
        source_by_name("nope")
