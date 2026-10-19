# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""A tiny corpus, tokenizer and prepared shards shared by the data and training tests."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from helix_lm.data.catalog import Source
from helix_lm.data.prepare import prepare_identity, prepare_source
from helix_lm.tokenizer import train_tokenizer

WEB = [
    f"Document {i}: the quick brown fox number {i} jumps over the lazy dog and writes a long enough paragraph "
    "to pass the web filter with room to spare, because stubs are dropped on purpose."
    for i in range(300)
]
JAMAICA = [
    "Kingston, Jamaica -- The Jamaica Gleaner reports that Montego Bay and Negril hotels in Westmoreland and "
    "St. James are preparing for the hurricane season, the Bank of Jamaica said in Kingston on Monday.",
    "Jamaican sprinter Usain Bolt returned to Trelawny for a reggae festival near Falmouth, Jamaica, where "
    "patois poetry and ackee breakfasts drew visitors from across Jamaica and the wider Caribbean.",
]
CODE = [
    f'def f{i}(x):\n    """Multiply by {i}: a function with a docstring long enough for the code filter."""\n'
    f"    return x * {i}\n\n\nclass K{i}:\n    pass\n"
    for i in range(50)
]


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus")
    tok_dir = train_tokenizer(WEB + JAMAICA + CODE, root / "tok", vocab_size=2048).directory
    web_files = []
    for i in range(3):
        path = root / f"web-{i}.parquet"
        texts = WEB[i * 100 : (i + 1) * 100] + (JAMAICA if i == 1 else [])
        pq.write_table(pa.table({"text": texts, "id": list(range(len(texts)))}), path)
        web_files.append(path)
    code_file = root / "code.jsonl"
    code_file.write_text("\n".join(json.dumps({"content": c}) for c in CODE) + "\n")
    return root, tok_dir, web_files, code_file


@pytest.fixture(scope="session")
def prepared(corpus):
    root, tok_dir, web_files, code_file = corpus
    data = root / "data"
    quiet = lambda *_: None  # noqa: E731
    web = Source("web", "local", "text", 0.7, kind="web", tag_jamaica=True)
    code = Source("code", "local", "content", 0.3, kind="code")
    prepare_source(web, data, tok_dir, None, workers=2, shard_tokens=500, local_files=web_files, log=quiet)
    prepare_source(code, data, tok_dir, None, workers=1, shard_tokens=500, local_files=[code_file], log=quiet)
    prepare_identity(data, tok_dir, log=quiet)
    return data
