# Copyright 2026 Nathan. Apache-2.0.
"""Write docs/MODULES.md: every module of the package, its size, and the first sentence of its docstring.

    python scripts/doc_modules.py            # run from the pluvium folder
"""
import ast
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
rows, total = [], 0
for path in sorted(root.glob("pluvium/**/*.py")):
    if "__pycache__" in path.parts:
        continue
    text = path.read_text()
    doc = ast.get_docstring(ast.parse(text)) or ""
    first_paragraph = " ".join(doc.strip().split("\n\n")[0].split())
    match = re.search(r"^(.+?[.?])(\s|$)", first_paragraph)
    sentence = (match.group(1) if match else first_paragraph).replace("|", "/")
    lines = len(text.splitlines())
    total += lines
    rows.append(f"| `{path.relative_to(root).as_posix()}` | {lines} | {sentence} |")
out = ["# Module map", "", "Every module of the package, what it is for, and its size. Generated from the modules' own "
       "docstrings by `scripts/doc_modules.py`.", "", "| Module | Lines | Purpose |", "|---|---|---|"] + rows + [
    "", f"{len(rows)} modules, {total:,} lines."]
(root / "docs" / "MODULES.md").write_text("\n".join(out) + "\n")
print(f"{len(rows)} modules, {total:,} lines -> docs/MODULES.md")
