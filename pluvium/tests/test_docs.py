# Copyright 2026 Nathan. Apache-2.0.
"""The documents do not rot: every relative link points at a file that exists, every command they show is one the
command line has, and the Hub cell stages the whole package."""

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = sorted([ROOT / "README.md", *ROOT.glob("docs/*.md"), *ROOT.glob("docs/papers/*.md"), *ROOT.glob("hub/*.md")])
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)\)|\!\[[^\]]*\]\(([^)\s]+)\)")


def test_relative_links_resolve():
    missing = []
    for doc in DOCS:
        for a, b in LINK.findall(doc.read_text()):
            target = (a or b).split("#")[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            base = ROOT if doc.parent.name == "hub" else doc.parent       # the card's links are from the repository's top
            if not (base / target).exists():
                missing.append(f"{doc.relative_to(ROOT)} -> {target}")
    assert not missing, missing


def test_commands_in_the_documents_exist():
    from pluvium import cli

    text = "\n".join(doc.read_text() for doc in DOCS)
    used = set(re.findall(r"^pluvium ([a-z-]+)", text, re.M)) | set(re.findall(r"`pluvium ([a-z-]+)", text))
    source = Path(cli.__file__).read_text()
    known = set(re.findall(r'add_parser\("([a-z-]+)"', source))
    assert used and used <= known, sorted(used - known)
    modules = set(re.findall(r"python -m (pluvium[\w.]+)", text))
    for name in modules:
        assert importlib.util.find_spec(name) is not None, name
    scripts = set(re.findall(r"python (scripts/\w+\.py|hub/\w+\.py)", text))
    assert scripts and all((ROOT / s).exists() for s in scripts), sorted(s for s in scripts if not (ROOT / s).exists())


def test_every_module_started_with_dash_m_really_starts():
    """``python -m pluvium.program`` imported the module and ended with exit code 0 in every release before 0.4.3:
    the file had no ``if __name__ == "__main__"``. Every notebook cell started the GPU training that way, so none
    of them trained anything. A module that exists is not a module that runs: both are checked here."""
    import subprocess

    sources = [ROOT / "README.md", ROOT / "CHANGELOG.md", *ROOT.glob("docs/*.md"), *ROOT.glob("hub/*.md"),
               *ROOT.glob("hub/*.py"), *ROOT.glob("notebooks/*.py")]
    started = set()
    for path in sources:
        started |= set(re.findall(r"-m[\"', ]+(pluvium(?:\.\w+)+)", path.read_text()))
    assert "pluvium.cli" in started
    for name in sorted(started | {"pluvium.program"}):
        source = Path(importlib.util.find_spec(name).origin).read_text()
        assert 'if __name__ == "__main__":' in source, f"python -m {name} would do nothing: {name} has no main guard"
    ran = subprocess.run([sys.executable, "-m", "pluvium.program", "--help"], capture_output=True, text=True, cwd=ROOT,
                         timeout=300)
    assert ran.returncode == 0 and "--hours" in ran.stdout and "--vram-share" in ran.stdout, ran.stdout[-400:] + ran.stderr[-400:]
    wrong = subprocess.run([sys.executable, "-m", "pluvium.program", "--no-such-flag"], capture_output=True, text=True,
                           cwd=ROOT, timeout=300)
    assert wrong.returncode == 2                                           # it parsed its arguments: it ran


def test_cells_compile_and_call_commands_that_exist():
    from pluvium import cli

    known = set(re.findall(r'add_parser\("([a-z-]+)"', Path(cli.__file__).read_text()))
    cells = sorted(ROOT.glob("notebooks/*.py")) + [ROOT / "hub" / "upload_to_hub.py"]
    assert len(cells) >= 4
    called = r"\{pluvium\} ([a-z-]+)|PLUVIUM \+ \[\"([a-z-]+)\""            # both ways the cells call the command line
    for cell in cells:
        source = cell.read_text()
        compile(source, str(cell), "exec")                                 # a cell that does not parse helps nobody
        assert "PASTE_YOUR_ORG_TOKEN_HERE" in source or "HF_TOKEN" not in source   # tokens are pasted, never shipped
        assert not re.search(r"hf_[A-Za-z0-9]{20,}", source), cell.name    # and no real token has slipped in
        for a, b in re.findall(called, source):
            assert (a or b) in known, (cell.name, a or b)
    blend = (ROOT / "notebooks" / "pluvium_blend_cell.py").read_text()
    assert {"setup", "blend", "cyclone"} <= set(re.findall(r"\{pluvium\} ([a-z-]+)", blend))
    everything = (ROOT / "notebooks" / "pluvium_everything.py").read_text()
    assert {"now", "upstream", "blend", "cyclone"} <= {b for _, b in re.findall(called, everything)}
    assert everything.index("HF_TOKEN = ") < everything.index("import ")    # the settings come before any code


def test_papers_and_card_have_none_of_the_signs_the_style_check_lists():
    """``scripts/check_style.py`` encodes Wikipedia's field guide to signs of AI writing: its word lists, the
    formulas it describes and its formatting habits. The papers, the model card, the readme and the two long pages
    behind them carry none."""
    spec = importlib.util.spec_from_file_location("check_style", ROOT / "scripts" / "check_style.py")
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    pages = sorted(ROOT.glob("docs/papers/*.md")) + [ROOT / "hub" / "README.md", ROOT / "README.md",
                                                     ROOT / "docs" / "GUIDE.md", ROOT / "docs" / "MODEL_CARD.md"]
    assert len(pages) >= 7
    for page in pages:
        found = style.findings(page.read_text())
        assert not found, (page.name, found[:5])
    for page in sorted(ROOT.glob("docs/papers/0*.md")):
        text = page.read_text()
        rhythm = style.rhythm(text)
        assert rhythm["share_8_to_20"] < 0.75, page.name                   # sentences of every length, as people write
        body, refs = text.split("## References")
        for author in re.findall(r"^- ([A-Z][a-z]+),", refs, re.M):         # every reference is used in the text
            assert author in body, (page.name, author)
        assert all("doi:" in line or "arXiv:" in line for line in refs.strip().split("\n- "))
    assert style.findings("It stands as a testament, not just a tool.\n- **Speed:** fast\n## A Title In Title Case Here")


def test_code_comments_docstrings_and_messages_carry_none_of_the_signs():
    """The same check on what a person reads in the code: every comment, docstring and longer message."""
    spec = importlib.util.spec_from_file_location("check_style", ROOT / "scripts" / "check_style.py")
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    files = [p for folder in ("pluvium", "tests", "notebooks", "scripts", "hub") for p in sorted((ROOT / folder).rglob("*.py"))
             if str(p.relative_to(ROOT)) not in style.CODE_SKIP]
    assert len(files) > 150
    found = []
    for path in files:
        prose = style.python_prose(path.read_text())
        found += [(str(path.relative_to(ROOT)), line, what) for line, check, what in style.findings(prose)
                  if not what.startswith(("title-case", "bold phrase"))]
    assert not found, found[:8]
    assert style.findings(style.python_prose('x = 1  # this is not just a number, it is a testament\n'))


def test_the_readme_is_short_and_the_card_on_the_hub_is_the_same_text():
    """The readme is three short pieces in the author's own words and tables for the rest; the long version is
    ``docs/GUIDE.md``. The card on the Hub is that same text under its front matter, so the two cannot drift."""
    readme = (ROOT / "README.md").read_text()
    card = (ROOT / "hub" / "README.md").read_text()
    assert card.startswith("---\n") and card.split("\n---\n", 1)[1].lstrip("\n") == readme
    prose = [line for line in re.sub(r"```.*?```", "", readme, flags=re.S).split("\n")
             if line.strip() and not line.startswith(("#", "|"))]
    words = sum(len(line.split()) for line in prose)
    assert 250 <= words <= 450, words
    assert readme.count("```") == 2                                        # one block: how to install and run it
    for page in ("docs/GUIDE.md", "docs/TRAINING.md", "CHANGELOG.md"):
        assert f"]({page})" in readme and (ROOT / page).exists()


def test_headline_results_are_the_same_text_wherever_they_appear():
    """The manual and the long model card carry the benchmark page's headline results between markers, written there
    by ``scripts/doc_frontier.py``, so the three cannot drift apart. (The readme and the card on the Hub are short
    and point at them.)"""
    page = (ROOT / "docs" / "FRONTIER_BENCHMARK.md").read_text()
    blocks = []
    for doc in (ROOT / "docs" / "GUIDE.md", ROOT / "docs" / "MODEL_CARD.md"):
        text = doc.read_text()
        assert text.count("<!-- frontier-summary:start -->") == 1 and text.count("<!-- frontier-summary:end -->") == 1
        block = text.split("<!-- frontier-summary:start -->")[1].split("<!-- frontier-summary:end -->")[0].strip()
        bullets = [line for line in block.splitlines() if line.startswith("- ")]
        assert len(bullets) >= 4 and bullets[0].startswith("- At 50 mm, Brier skill improved on the first Blend")
        for bullet in bullets:
            assert bullet in page, bullet[:60]                             # word for word what the benchmark page says
        blocks.append(block)
    assert blocks[0] == blocks[1]


def test_no_document_is_left_unfinished():
    for doc in DOCS:
        text = doc.read_text()
        for mark in ("TODO", "TBD", "FIXME", "XXX"):
            assert mark not in text, (doc.name, mark)


def test_hub_cell_stages_the_package_with_the_card_on_top(monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("upload_to_hub", ROOT / "hub" / "upload_to_hub.py")
    cell = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cell)
    assert cell.HF_TOKEN == "PASTE_YOUR_ORG_TOKEN_HERE" and cell.HF_REPO.endswith("/Pluvium-500")   # no token in the file
    monkeypatch.setenv("PLUVIUM_HUB_DRY_RUN", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(cell, "SOURCE", str(ROOT))
    cell.main()
    out = capsys.readouterr().out
    assert "dry run" in out and "pluvium" in out and "docs" in out and "README.md" in out
    count = int(re.search(r"dry run: (\d+) files", out).group(1))
    assert count > 200
    monkeypatch.delenv("PLUVIUM_HUB_DRY_RUN")
    try:
        cell.main()
    except SystemExit as stop:
        assert "token" in str(stop).lower()
    else:
        raise AssertionError("uploading without a token must stop")
    assert "upload_to_hub" not in sys.modules
