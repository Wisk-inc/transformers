# Copyright 2026 Nathan. Apache-2.0.
"""
The published weights of the AI weather models whose forecasts Pluvium reads, and how to keep a copy of them.

=========================  ==========================================================================================
:mod:`.registry`           which models, where each publisher keeps the weights, under which licence
:mod:`.fetch`              copy one to a local folder, checked, with its licence and a record of the source
:mod:`.mirror`             keep the copies inside a Hugging Face model repository, under ``upstream/``
=========================  ==========================================================================================

From the command line: ``pluvium upstream list``, ``pluvium upstream fetch`` and
``pluvium upstream mirror --repo <organisation>/<name>``.
"""

from .fetch import DEFAULT_CACHE, download, fetch_model, licence_text, plan, sha256_of, size_of, source_note
from .mirror import index_text, is_mirrored, mirror, repo_files, summary
from .registry import MODELS, Upstream, get, select

__all__ = [
    "DEFAULT_CACHE", "MODELS", "Upstream", "download", "fetch_model", "get", "index_text", "is_mirrored",
    "licence_text", "mirror", "plan", "repo_files", "select", "sha256_of", "size_of", "source_note", "summary",
]
