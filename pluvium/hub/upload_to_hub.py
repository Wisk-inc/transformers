# Put Pluvium on the Hugging Face Hub: the model card, all the code the architecture is built from, the papers and
# the benchmarks, and a collection in the organisation that holds the model and the papers it builds on.
# Paste into one notebook cell, or run as `python hub/upload_to_hub.py` from the pluvium folder.
#
# It creates the model repository if it does not exist and uploads in one commit. Running it again updates it.
# Two more things belong in the same repository and have their own commands:
#   pluvium upstream mirror --repo <repo>          the weights of GraphCast, NOAA's GraphCastGFS/AIGFS and ECMWF's AIFS
#   pluvium blend publish --root <folder> --repo <repo>     a trained blend, so that others can skip training
# ruff: noqa: E402  (the settings come first on purpose: they are what a person edits)

HF_TOKEN = "PASTE_YOUR_ORG_TOKEN_HERE"      # a write token for the organisation (Hub -> Settings -> Access Tokens)
HF_REPO = "CorxLabs/Pluvium-500"            # the model repository to create or update
PRIVATE = False                             # True to keep it private until you are ready
COLLECTION = "Pluvium"                      # the collection to create in the organisation; "" for none
SOURCE = None                               # the pluvium folder; None = find it, or clone it from GitHub
DRY_RUN = False                             # True to list what would be uploaded and stop (needs no token)

import os
import subprocess
import sys
import tempfile
from pathlib import Path

GITHUB = "https://github.com/Wisk-inc/transformers.git"
BRANCH = "claude/linear-context-architecture-a35c6h"
# what goes in the repository: the card is hub/README.md; everything else keeps its path
INCLUDE = ["pluvium/**", "tests/**", "scripts/**", "notebooks/**", "docs/**", "hub/upload_to_hub.py", "pyproject.toml",
           "pytest.ini", "LICENSE", "README.md", "CHANGELOG.md"]
IGNORE = ["**/__pycache__/**", "**/*.pyc", "**/.pytest_cache/**", "**/.ruff_cache/**", "dist/**", "**/*.egg-info/**"]
COLLECTION_NOTE = "Rain forecasts for any country and hurricane tracks, from six operational systems. Code, papers and benchmarks."
MODEL_NOTE = ("The model, all its code, four papers, the benchmarks with losses beside wins, and copies of the "
              "upstream AI models' weights.")
# papers Pluvium builds on that have a page on the Hub (checked on 15 November 2026): arXiv id -> why it is here
PAPERS = {
    "2212.12794": "GraphCast. NOAA runs a fine-tuned GraphCast every six hours; it is one of the six systems Pluvium reads.",
    "2406.01465": "AIFS, ECMWF's AI forecast model: another of the six systems Pluvium reads.",
    "2509.18994": "The update that made AIFS operational at ECMWF.",
    "2604.25559": "The surface ocean in AIFS 2.0, the version ECMWF has run since May 2026.",
}


def find_source() -> Path:
    """The pluvium project folder: SOURCE, the folder this file is in, the working folder, or a fresh clone."""
    here = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
    for folder in ([Path(SOURCE)] if SOURCE else []) + [here, Path.cwd(), Path.cwd() / "pluvium"]:
        if (folder / "pyproject.toml").exists() and (folder / "pluvium" / "__init__.py").exists():
            return folder
    clone = Path(tempfile.mkdtemp(prefix="pluvium-src-"))
    print(f"no local copy found; cloning {GITHUB} ({BRANCH}) ...", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, GITHUB, str(clone)], check=True)
    return clone / "pluvium"


def make_collection(api) -> str | None:
    """Create the collection in the repository's organisation (or find it), put the model and the papers in it,
    and return its address. A paper the Hub does not know is skipped; nothing here stops the upload."""
    if not COLLECTION:
        return None
    owner = HF_REPO.split("/")[0]
    try:
        collection = api.create_collection(title=COLLECTION, namespace=owner, description=COLLECTION_NOTE[:150],
                                           private=PRIVATE, exists_ok=True)
        api.add_collection_item(collection.slug, item_id=HF_REPO, item_type="model", note=MODEL_NOTE[:500], exists_ok=True)
    except Exception as error:
        print(f"collection not made ({type(error).__name__}: {error})")
        return None
    added = 0
    for paper, note in PAPERS.items():
        try:
            api.add_collection_item(collection.slug, item_id=paper, item_type="paper", note=note[:500], exists_ok=True)
            added += 1
        except Exception as error:
            print(f"  paper {paper} not added ({type(error).__name__}: {str(error)[:120]})")
    where = f"https://huggingface.co/collections/{collection.slug}"
    print(f"collection: {where} (the model and {added} papers)")
    return where


def main() -> None:
    dry = DRY_RUN or os.environ.get("PLUVIUM_HUB_DRY_RUN") == "1"
    token = os.environ.get("HF_TOKEN") or HF_TOKEN
    if not dry and (not token or token.startswith("PASTE")):
        raise SystemExit("Paste a write token into HF_TOKEN at the top of this cell (or set the HF_TOKEN variable).")
    source = find_source()
    card = source / "hub" / "README.md"
    if not card.exists():
        raise SystemExit(f"{card} is missing: this needs the full pluvium folder")
    # stage a copy with the card at the top, so one commit carries everything
    stage = Path(tempfile.mkdtemp(prefix="pluvium-hub-"))
    for pattern in INCLUDE:
        for path in source.glob(pattern):
            if path.is_file() and not any(part in ("__pycache__", ".pytest_cache", ".ruff_cache") for part in path.parts) \
                    and path.suffix != ".pyc":
                target = stage / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
    (stage / "PACKAGE_README.md").write_bytes((stage / "README.md").read_bytes())      # the package's own readme
    (stage / "README.md").write_bytes(card.read_bytes())                               # the model card
    files = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file())
    size = sum((stage / f).stat().st_size for f in files) / 1e6
    if dry:
        tops = sorted({f.split("/")[0] for f in files})
        print(f"dry run: {len(files)} files, {size:.1f} MB, from {source}: {', '.join(tops)}")
        if COLLECTION:
            print(f"dry run: collection '{COLLECTION}' in {HF_REPO.split('/')[0]} with the model and {len(PAPERS)} papers")
        return
    try:
        from huggingface_hub import HfApi
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.20"], check=True)
        from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(HF_REPO, repo_type="model", exist_ok=True, private=PRIVATE)
    print(f"uploading {len(files)} files ({size:.1f} MB) from {source} to https://huggingface.co/{HF_REPO} ...", flush=True)
    api.upload_folder(folder_path=str(stage), repo_id=HF_REPO, repo_type="model", ignore_patterns=IGNORE,
                      commit_message="Pluvium: code, model card, papers and benchmarks")
    print(f"done: https://huggingface.co/{HF_REPO}")
    make_collection(api)


if __name__ == "__main__":
    main()
