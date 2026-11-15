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


def test_cells_compile_and_call_commands_that_exist():
    from pluvium import cli

    known = set(re.findall(r'add_parser\("([a-z-]+)"', Path(cli.__file__).read_text()))
    cells = sorted(ROOT.glob("notebooks/*.py")) + [ROOT / "hub" / "upload_to_hub.py"]
    assert len(cells) >= 3
    for cell in cells:
        source = cell.read_text()
        compile(source, str(cell), "exec")                                 # a cell that does not parse helps nobody
        assert "PASTE_YOUR_ORG_TOKEN_HERE" in source or "HF_TOKEN" not in source   # tokens are pasted, never shipped
        for command in re.findall(r"\{pluvium\} ([a-z-]+)", source):
            assert command in known, (cell.name, command)
    blend = (ROOT / "notebooks" / "pluvium_blend_cell.py").read_text()
    assert {"setup", "blend", "cyclone"} <= set(re.findall(r"\{pluvium\} ([a-z-]+)", blend))


def test_headline_results_are_the_same_text_wherever_they_appear():
    """The README and the model card carry the benchmark page's headline results between markers, written there by
    ``scripts/doc_frontier.py``, so the three cannot drift apart."""
    page = (ROOT / "docs" / "FRONTIER_BENCHMARK.md").read_text()
    blocks = []
    for doc in (ROOT / "README.md", ROOT / "hub" / "README.md"):
        text = doc.read_text()
        assert text.count("<!-- frontier-summary:start -->") == 1 and text.count("<!-- frontier-summary:end -->") == 1
        block = text.split("<!-- frontier-summary:start -->")[1].split("<!-- frontier-summary:end -->")[0].strip()
        bullets = [line for line in block.splitlines() if line.startswith("- **")]
        assert len(bullets) >= 4 and bullets[0].startswith("- **Heavy rain (50 mm).**")
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
