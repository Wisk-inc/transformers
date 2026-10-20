# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Every source CorX5.5 trains on, in one place, with what it is for and under which license.

Weights are shares of the pretraining token budget (870B tokens for the one-year run; the same catalog
scales to 4T). The shape: 40% general web, 30% code and code-adjacent text (repositories, commits, issues,
programs with the tests they were run against and what happened), 6% mathematics, 10% the domains the
model is for (legal, finance, health, education), and 14% nominally Jamaican and Caribbean material.
"Nominally" because the Patois web is tens of megabytes and every Jamaican source is capped at a few passes
so it is learned rather than memorized; the loader hands the unfillable share back to the other sources
and prints the real shares at the start of training.

Nothing here is downloaded at import time. ``helix data verify`` checks that each source exists, that the
token has access to the gated ones, and reports sizes; ``helix data prepare`` streams them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    """
    One dataset on the Hub (or made here).

    Args:
        name: Short name used for the shard files and the mixture.
        repo: Hub dataset id; ``(derived)``, ``(generated)`` or ``(local)`` for data made on this machine.
        text: Column holding the document text, or a tuple of columns joined with blank lines.
        weight: Share of the pretraining mixture (relative; normalized at load time).
        paths: Prefixes inside the repo to read (``()`` reads every data file).
        license: As declared on the dataset card.
        kind: ``web``, ``code``, ``math``, ``reference``, ``synthetic``, ``legal``, ``finance``, ``health``,
            ``education``, ``patois``, ``jamaica``, ``sft``.
        render: Name of a renderer in :mod:`renderers` that turns a structured row into text.
        tag_jamaica: Also write documents about Jamaica (and the Caribbean) to their own sources while streaming.
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
    render: str | None = None
    tag_jamaica: bool = False
    max_epochs: float = 1.0
    gated: bool = False
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def local(self) -> bool:
        return self.repo.startswith("(")


STACK_LANGUAGES = (
    "python", "javascript", "typescript", "java", "go", "rust", "c", "c++", "c-sharp", "php", "ruby", "shell",
    "sql", "kotlin", "swift", "scala", "dart", "lua", "r", "julia", "haskell", "elixir", "markdown", "dockerfile",
    "makefile", "cmake", "yaml", "toml", "html", "css", "jupyter-notebook", "tex",
)  # fmt: skip

PRETRAIN_SOURCES: tuple[Source, ...] = (
    # ---- the general web: 40% -----------------------------------------------------------------------------
    Source(
        "dclm",
        "mlfoundations/dclm-baseline-1.0",
        "text",
        0.25,
        license="cc-by-4.0",
        kind="web",
        tag_jamaica=True,
        note="DCLM-Baseline, 4T tokens of filtered Common Crawl; the backbone of the mixture.",
    ),
    Source(
        "fineweb-edu",
        "HuggingFaceFW/fineweb-edu",
        "text",
        0.10,
        paths=("sample/350BT/",),
        license="odc-by",
        kind="web",
        tag_jamaica=True,
    ),
    Source(
        "finepdfs",
        "HuggingFaceFW/finepdfs",
        "text",
        0.05,
        paths=("data/eng_Latn/train/",),
        license="odc-by",
        kind="web",
        tag_jamaica=True,
        note="Long documents from PDFs: reports, filings, judgments, papers.",
    ),
    # ---- code and what happens when it runs: 30% -------------------------------------------------------------
    Source(
        "stack",
        "bigcode/the-stack-dedup",
        "content",
        0.14,
        paths=tuple(f"data/{lang}/" for lang in STACK_LANGUAGES),
        license="other (The Stack terms; permissive-licensed code only)",
        kind="code",
        gated=True,
        note="Accept the terms at https://huggingface.co/datasets/bigcode/the-stack-dedup once with the CorxLabs "
        "account.",
    ),
    Source(
        "starcoder",
        "bigcode/starcoderdata",
        "content",
        0.08,
        license="other (The Stack terms)",
        kind="code",
        gated=True,
        note="Includes github-issues, git-commits and jupyter subsets. Accept the terms once on the Hub.",
    ),
    Source(
        "github-code",
        "codeparrot/github-code-clean",
        "code",
        0.03,
        license="apache-2.0 (permissive files)",
        kind="code",
        note="Ungated; takes the gated sources' share when their terms are not accepted.",
    ),
    Source("code-web", "OpenCoder-LLM/opc-fineweb-code-corpus", "text", 0.02, license="mit", kind="code"),
    Source(
        "commits",
        "bigcode/commitpackft",
        "message",
        0.005,
        license="mit",
        kind="code",
        render="commit",
        max_epochs=2.0,
    ),
    Source(
        "issues",
        "bigcode/the-stack-github-issues",
        "content",
        0.01,
        license="other (The Stack terms)",
        kind="code",
        gated=True,
        render="github_issue",
    ),
    Source(
        "traces-instruct",
        "nvidia/OpenCodeInstruct",
        "output",
        0.007,
        license="cc-by-4.0",
        kind="code",
        render="execution_trace",
        max_epochs=2.0,
        note="Programs with their unit tests and the recorded result of running them: synthetic execution traces.",
    ),
    Source(
        "traces-feedback",
        "m-a-p/CodeFeedback-Filtered-Instruction",
        "answer",
        0.003,
        license="apache-2.0",
        kind="code",
        render="code_feedback",
        max_epochs=2.0,
    ),
    Source(
        "code-reasoning",
        "nvidia/OpenCodeReasoning",
        "output",
        0.005,
        license="cc-by-4.0",
        kind="code",
        render="problem_solution",
        max_epochs=2.0,
    ),
    # ---- mathematics: 6% ----------------------------------------------------------------------------------
    Source(
        "finemath", "HuggingFaceTB/finemath", "text", 0.03, paths=("finemath-3plus/",), license="odc-by", kind="math"
    ),
    Source(
        "infiwebmath",
        "HuggingFaceTB/finemath",
        "text",
        0.015,
        paths=("infiwebmath-3plus/",),
        license="odc-by",
        kind="math",
    ),
    Source(
        "numina",
        "AI-MO/NuminaMath-1.5",
        "solution",
        0.005,
        license="apache-2.0",
        kind="math",
        render="problem_solution",
        max_epochs=2.0,
    ),
    Source("nemotron-math", "nvidia/Nemotron-Math-v2", "text", 0.01, license="cc-by-4.0", kind="math"),
    # ---- the domains CorX5.5 is for: 10% -------------------------------------------------------------------
    Source(
        "cosmopedia",
        "HuggingFaceTB/smollm-corpus",
        "text",
        0.025,
        paths=("cosmopedia-v2/",),
        license="odc-by",
        kind="education",
    ),
    Source("cosmopedia-v1", "HuggingFaceTB/cosmopedia", "text", 0.01, license="apache-2.0", kind="education"),
    Source(
        "webinstruct",
        "TIGER-Lab/WebInstructSub",
        "answer",
        0.01,
        license="apache-2.0",
        kind="education",
        render="problem_solution",
        max_epochs=2.0,
    ),
    Source(
        "nemotron-specialized",
        "nvidia/Nemotron-Pretraining-Specialized-v1",
        "text",
        0.02,
        license="cc-by-4.0",
        kind="synthetic",
    ),
    Source(
        "caselaw",
        "free-law/Caselaw_Access_Project",
        "text",
        0.015,
        license="cc0-1.0",
        kind="legal",
        gated=True,
        tag_jamaica=True,
        note="US case law, public domain. Accept the terms once on the Hub.",
    ),
    Source(
        "medmcqa",
        "openlifescienceai/medmcqa",
        "question",
        0.002,
        license="apache-2.0",
        kind="health",
        render="multiple_choice",
        max_epochs=2.0,
    ),
    Source(
        "pubmedqa",
        "qiaojin/PubMedQA",
        "question",
        0.002,
        paths=("pqa_artificial/", "pqa_labeled/"),
        license="mit",
        kind="health",
        render="pubmed_qa",
        max_epochs=2.0,
    ),
    Source(
        "medical-o1",
        "FreedomIntelligence/medical-o1-reasoning-SFT",
        "Response",
        0.003,
        license="apache-2.0",
        kind="health",
        render="problem_solution",
        max_epochs=2.0,
    ),
    Source(
        "guidelines",
        "epfl-llm/guidelines",
        "clean_text",
        0.003,
        license="other (per-source, mostly CC-BY)",
        kind="health",
        render="guideline",
        max_epochs=2.0,
    ),
    Source(
        "finance-instruct",
        "Josephgflowers/Finance-Instruct-500k",
        "assistant",
        0.005,
        license="apache-2.0",
        kind="finance",
        render="chat_transcript",
        max_epochs=2.0,
    ),
    Source(
        "finance-sujet",
        "sujet-ai/Sujet-Finance-Instruct-177k",
        "answer",
        0.005,
        license="apache-2.0",
        kind="finance",
        render="chat_transcript",
        max_epochs=2.0,
    ),
    # ---- Jamaica and the Caribbean: 14% nominal, capped by passes ----------------------------------------------
    Source(
        "patois-fineweb2",
        "HuggingFaceFW/fineweb-2",
        "text",
        0.01,
        paths=("data/jam_Latn/train/",),
        license="odc-by",
        kind="patois",
        max_epochs=4.0,
        note="Jamaican Patois web text, FineWeb-2's jam_Latn split.",
    ),
    Source(
        "patois-glotcc",
        "cis-lmu/GlotCC-V1",
        "content",
        0.005,
        paths=("v1.0/jam-Latn/",),
        license="cc0-1.0",
        kind="patois",
        max_epochs=4.0,
    ),
    Source(
        "patois-madlad",
        "allenai/MADLAD-400",
        "text",
        0.005,
        paths=("data/jam/",),
        license="odc-by",
        kind="patois",
        max_epochs=4.0,
    ),
    Source(
        "patois-wikipedia",
        "wikimedia/wikipedia",
        ("title", "text"),
        0.003,
        paths=("20231101.jam/",),
        license="cc-by-sa-3.0",
        kind="patois",
        max_epochs=4.0,
    ),
    Source(
        "jamaica_en",
        "(derived)",
        "text",
        0.04,
        kind="jamaica",
        max_epochs=3.0,
        note="English documents about Jamaica, found while the web, PDFs and case law stream past.",
    ),
    Source(
        "caribbean_en",
        "(derived)",
        "text",
        0.02,
        kind="jamaica",
        max_epochs=2.0,
        note="Documents about the wider Caribbean, found the same way.",
    ),
    Source(
        "jamaica_gov",
        "(local)",
        "text",
        0.03,
        kind="jamaica",
        max_epochs=3.0,
        note="Jamaican public records crawled by `helix jamaica crawl`: judgments, Hansard, gazettes, statistics, "
        "JIS.",
    ),
    Source(
        "codeswitch",
        "(local)",
        "text",
        0.025,
        kind="jamaica",
        max_epochs=2.0,
        note="Synthetic SJE/Patois code-switched dialogues from `helix jamaica synth` (banking, legal, health, "
        "education).",
    ),
    Source("identity", "(generated)", "text", 0.002, kind="jamaica", max_epochs=8.0),
)

SFT_SOURCES: tuple[Source, ...] = (
    Source(
        "swe-agent",
        "nebius/SWE-agent-trajectories",
        "trajectory",
        0.20,
        license="cc-by-4.0",
        kind="sft",
        note="80k SWE-agent runs on real GitHub issues; `agent.py` keeps the ones that resolved the issue.",
    ),
    Source("swe-smith", "SWE-bench/SWE-smith-trajectories", "messages", 0.12, license="mit", kind="sft"),
    Source("opencodeinstruct", "nvidia/OpenCodeInstruct", ("input", "output"), 0.18, license="cc-by-4.0", kind="sft"),
    Source(
        "openthoughts", "open-thoughts/OpenThoughts3-1.2M", "conversations", 0.12, license="apache-2.0", kind="sft"
    ),
    Source(
        "smoltalk",
        "HuggingFaceTB/smoltalk2",
        "messages",
        0.08,
        license="apache-2.0 (per subset)",
        kind="sft",
        paths=("SFT/",),
    ),
    Source(
        "tulu",
        "allenai/tulu-3-sft-mixture",
        "messages",
        0.10,
        license="odc-by",
        kind="sft",
        note="Includes the CoCoNot and WildJailbreak safety subsets.",
    ),
    Source(
        "coconot",
        "allenai/coconot",
        "messages",
        0.02,
        license="odc-by",
        kind="sft",
        note="Noncompliance: requests a model should decline or qualify, with good responses.",
    ),
    Source("finance-chat", "Josephgflowers/Finance-Instruct-500k", "messages", 0.03, license="apache-2.0", kind="sft"),
    Source(
        "medical-chat",
        "FreedomIntelligence/medical-o1-reasoning-SFT",
        "messages",
        0.03,
        license="apache-2.0",
        kind="sft",
    ),
    Source(
        "codeswitch-chat",
        "(local)",
        "messages",
        0.07,
        kind="sft",
        max_epochs=3.0,
        note="The synthetic SJE/Patois dialogues as conversations.",
    ),
    Source(
        "safety-chat",
        "(generated)",
        "messages",
        0.03,
        kind="sft",
        max_epochs=4.0,
        note="Generated by `helix_lm.data.safety`: boundaries, benign look-alikes and agent conduct.",
    ),
    Source("identity-chat", "(generated)", "messages", 0.02, kind="sft", max_epochs=4.0),
)


def source_by_name(name: str, sources: tuple[Source, ...] = PRETRAIN_SOURCES) -> Source:
    for source in sources:
        if source.name == name:
            return source
    raise KeyError(f"no source named {name!r}")


def verify_sources(sources: tuple[Source, ...] = PRETRAIN_SOURCES, token: str | None = None) -> list[dict]:
    """
    Ask the Hub about every source: does it exist, can this token read it, how big is the part we read.
    Returns one record per source; ``helix data verify`` prints them. Derived, generated and local sources are
    reported as such and never queried.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    api = HfApi(token=token)
    report = []
    for source in sources:
        record = {"name": source.name, "repo": source.repo, "ok": True, "files": 0, "gb": 0.0, "status": ""}
        if source.local:
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
            if s.rfilename.endswith((".parquet", ".jsonl", ".json.gz", ".jsonl.zst", ".jsonl.gz", ".jsonl.xz", ".zip"))
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
    lines = [f"{'source':20} {'ok':3} {'files':>6} {'GB':>9}  repo / status"]
    for r in report:
        lines.append(
            f"{r['name']:20} {'ok' if r['ok'] else '!!':3} {r['files']:>6} {r['gb']:>9.2f}  {r['repo']}  {r['status']}"
        )
    return "\n".join(lines)
