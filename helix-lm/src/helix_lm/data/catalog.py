# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Every source Helix trains on, in one place, with what it is for and under which license.

Weights are shares of the pretraining token budget. They were chosen for a coding agent: about half code
and code-adjacent text, a third educational web text, the rest mathematics, reference, synthetic
textbooks, and the Jamaican material -- which is small in the world and therefore up-weighted here, with
the Patois corpora capped at a few passes so they are learned rather than memorized.

Nothing here is downloaded at import time. ``helix data verify`` checks that each source exists, that the
token has access to the gated ones, and reports sizes; ``helix data prepare`` streams them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    """
    One dataset on the Hub.

    Args:
        name: Short name used for the shard files and the mixture.
        repo: Hub dataset id.
        text: Column holding the document text, or a tuple of columns joined with blank lines.
        weight: Share of the pretraining mixture (relative; normalized at load time).
        paths: Prefixes inside the repo to read (``()`` reads every parquet file).
        license: As declared on the dataset card.
        kind: ``web``, ``code``, ``math``, ``reference``, ``synthetic``, ``patois``, ``jamaica``, ``sft``.
        filter: Name of a document filter in :mod:`prepare` applied before tokenizing.
        tag_jamaica: Also write documents about Jamaica to the ``jamaica_en`` source while streaming.
        max_epochs: Cap on how many times the loader may pass over this source.
        gated: The user must accept the dataset's terms on the Hub before the token can read it.
        note: Anything the person running this should know.
    """

    name: str
    repo: str
    text: str | tuple[str, ...]
    weight: float
    paths: tuple[str, ...] = ()
    license: str = ""
    kind: str = "web"
    filter: str | None = None
    tag_jamaica: bool = False
    max_epochs: float = 1.0
    gated: bool = False
    note: str = ""
    extra: dict = field(default_factory=dict)


PRETRAIN_SOURCES: tuple[Source, ...] = (
    # ---- code: about half of what a coding agent reads -------------------------------------------------
    Source(
        "starcoder", "bigcode/starcoderdata", "content", 0.32,
        license="other (The Stack v1 terms; permissive-licensed code only)", kind="code", gated=True,
        note="Accept the terms at https://huggingface.co/datasets/bigcode/starcoderdata once with the CorxLabs "
        "account. "
             "Includes github-issues, git-commits and jupyter subsets: the agent's natural habitat.",
    ),
    Source(
        "github-code", "codeparrot/github-code-clean", "code", 0.08,
        license="apache-2.0 (dataset); files carry permissive licenses", kind="code",
        note="Ungated fallback for the code share if starcoder's terms are not accepted; the preparer raises its "
        "weight.",
    ),
    Source(
        "code-web", "OpenCoder-LLM/opc-fineweb-code-corpus", "text", 0.04,
        license="mit", kind="code", note="Code-related web pages: documentation, tutorials, Q&A.",
    ),
    Source(
        "code-anneal", "OpenCoder-LLM/opc-annealing-corpus", "text", 0.02,
        license="odc-by", kind="code", max_epochs=2.0,
        note="Algorithmic corpora and synthetic code QA; read twice, it is small and dense.",
    ),
    # ---- the English web, educational slice ----------------------------------------------------------
    Source(
        "fineweb-edu", "HuggingFaceFW/fineweb-edu", "text", 0.36, paths=("sample/350BT/",),
        license="odc-by", kind="web", tag_jamaica=True,
    ),
    # ---- mathematics -----------------------------------------------------------------------------------
    Source("finemath", "HuggingFaceTB/finemath", "text", 0.07, paths=("finemath-3plus/",), license="odc-by",
           kind="math"),
    # ---- reference and synthetic textbooks ----------------------------------------------------------------
    Source(
        "wikipedia", "wikimedia/wikipedia", ("title", "text"), 0.04, paths=("20231101.en/",),
        license="cc-by-sa-3.0", kind="reference", tag_jamaica=True, max_epochs=2.0,
    ),
    Source(
        "cosmopedia", "HuggingFaceTB/smollm-corpus", "text", 0.05, paths=("cosmopedia-v2/",),
        license="odc-by", kind="synthetic",
    ),
    # ---- Jamaica -------------------------------------------------------------------------------------------
    # The Patois corpora are small (tens of megabytes between them). They are given a share far above their
    # size, capped at four passes, so the model learns the language without memorizing the documents.
    Source(
        "patois-fineweb2", "HuggingFaceFW/fineweb-2", "text", 0.004, paths=("data/jam_Latn/train/",),
        license="odc-by", kind="patois", max_epochs=4.0, note="Jamaican Patois web text, FineWeb-2's jam_Latn split.",
    ),
    Source(
        "patois-glotcc", "cis-lmu/GlotCC-V1", "content", 0.002, paths=("v1.0/jam-Latn/",),
        license="cc0-1.0", kind="patois", max_epochs=4.0,
    ),
    Source(
        "patois-madlad", "allenai/MADLAD-400", "text", 0.002, paths=("data/jam/",),
        license="odc-by", kind="patois", max_epochs=4.0, note="MADLAD-400's clean 'jam' split.",
    ),
    Source(
        "patois-wikipedia", "wikimedia/wikipedia", ("title", "text"), 0.001, paths=("20231101.jam/",),
        license="cc-by-sa-3.0", kind="patois", max_epochs=4.0, note="The Jamaican Patois Wikipedia (jam).",
    ),
    # Written while streaming fineweb-edu and Wikipedia: documents the Jamaica filter accepts. Never
    # downloaded on its own; its weight is on top of the copies that stay in their original sources.
    Source(
        "jamaica_en", "(derived)", "text", 0.010, kind="jamaica", max_epochs=3.0,
        note="English documents about Jamaica, mined from the web and Wikipedia passes; read up to three times.",
    ),
    # The model's own identity and facts about Jamaica, generated by `helix_lm.data.identity`.
    Source("identity", "(generated)", "text", 0.001, kind="jamaica", max_epochs=8.0),
)  # fmt: skip

SFT_SOURCES: tuple[Source, ...] = (
    Source("swe-agent", "nebius/SWE-agent-trajectories", "trajectory", 0.25, license="cc-by-4.0", kind="sft",
           note="80k SWE-agent runs on real GitHub issues; `agent.py` keeps the ones that submitted a patch."),
    Source("swe-smith", "SWE-bench/SWE-smith-trajectories", "messages", 0.15, license="mit", kind="sft",
           note="5k resolved trajectories from SWE-smith, OpenAI message format with tool calls."),
    Source("opencodeinstruct", "nvidia/OpenCodeInstruct", ("input", "output"), 0.25, license="cc-by-4.0", kind="sft"),
    Source("openthoughts", "open-thoughts/OpenThoughts3-1.2M", "conversations", 0.15, license="apache-2.0", kind="sft",
           note="Reasoning traces for math, code and science."),
    Source("smoltalk", "HuggingFaceTB/smoltalk2", "messages", 0.10, license="apache-2.0 (per subset)", kind="sft",
           paths=("SFT/",)),
    Source("tulu", "allenai/tulu-3-sft-mixture", "messages", 0.09, license="odc-by", kind="sft"),
    Source("identity-chat", "(generated)", "messages", 0.01, kind="sft", max_epochs=4.0),
)  # fmt: skip


def source_by_name(name: str, sources: tuple[Source, ...] = PRETRAIN_SOURCES) -> Source:
    for source in sources:
        if source.name == name:
            return source
    raise KeyError(f"no source named {name!r}")


def verify_sources(sources: tuple[Source, ...] = PRETRAIN_SOURCES, token: str | None = None) -> list[dict]:
    """
    Ask the Hub about every source: does it exist, can this token read it, how big is the part we read.
    Returns one record per source; ``helix data verify`` prints them. Derived and generated sources are
    reported as such and never queried.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    api = HfApi(token=token)
    report = []
    for source in sources:
        record = {"name": source.name, "repo": source.repo, "ok": True, "files": 0, "gb": 0.0, "status": ""}
        if source.repo.startswith("("):
            record["status"] = source.repo
            report.append(record)
            continue
        try:
            info = api.dataset_info(source.repo, files_metadata=True)
        except GatedRepoError:
            record.update(ok=False, status="GATED: accept the terms on the Hub with this account")
            report.append(record)
            continue
        except RepositoryNotFoundError:
            record.update(ok=False, status="NOT FOUND")
            report.append(record)
            continue
        except Exception as error:  # network, auth
            record.update(ok=False, status=f"{type(error).__name__}: {str(error)[:80]}")
            report.append(record)
            continue
        files = [
            s for s in info.siblings
            if s.rfilename.endswith((".parquet", ".jsonl", ".json.gz", ".jsonl.zst", ".jsonl.gz", ".zip"))
            and (not source.paths or s.rfilename.startswith(source.paths))
        ]  # fmt: skip
        record["files"] = len(files)
        record["gb"] = round(sum((s.size or 0) for s in files) / 1e9, 2)
        if not files:
            record.update(ok=False, status="no data files under " + ", ".join(source.paths or ("/",)))
        elif info.gated and source.gated:
            record["status"] = "gated; readable with this token"
        report.append(record)
    return report


def describe_report(report: list[dict]) -> str:
    lines = [f"{'source':18} {'ok':3} {'files':>6} {'GB':>9}  repo / status"]
    for r in report:
        lines.append(
            f"{r['name']:18} {'ok' if r['ok'] else '!!':3} {r['files']:>6} {r['gb']:>9.2f}  {r['repo']}  {r['status']}"
        )
    return "\n".join(lines)
