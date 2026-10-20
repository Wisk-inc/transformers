"""
Create the CorX5.5 model repository on the Hugging Face Hub and upload the code: the wheel (under code/), the
startup scripts and the plan. Reads the token from $HF_TOKEN (never from a chat). Run from the helix-lm directory:

    python scripts/build_corx_startup.py
    HF_TOKEN=... python scripts/corx_publish.py [--public] [--repo CorxLabs/CorX5.5]

The startup script does the same on the training machine (stage 0), so this is only for publishing ahead of time.
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).parent
REPO_DIR = HERE.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="CorxLabs/CorX5.5")
    parser.add_argument("--public", action="store_true", help="make the repository public (private by default)")
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN in the environment (a write token); do not paste it anywhere else")
    api = HfApi(token=token)
    api.create_repo(args.repo, exist_ok=True, private=not args.public)
    uploads = {
        **{f"code/{p.name}": p for p in sorted((REPO_DIR / "dist").glob("helix_lm-*.whl"))},
        "code/corx_startup.sh": HERE / "corx_startup.sh",
        "code/corx_startup_bundle.sh": REPO_DIR / "dist" / "corx_startup_bundle.sh",
        "code/CORX5.5.md": REPO_DIR / "docs" / "CORX5.5.md",
    }
    missing = [str(p) for p in uploads.values() if not p.exists()]
    if missing:
        raise SystemExit("build first (python scripts/build_corx_startup.py); missing: " + ", ".join(missing))
    for path_in_repo, path in uploads.items():
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo,
            repo_id=args.repo,
            commit_message=f"code: {path.name}",
        )
        print(f"uploaded {path_in_repo} ({path.stat().st_size / 1024:.0f} KB)")
    if "README.md" not in set(api.list_repo_files(args.repo)):
        card = (
            "---\nlicense: apache-2.0\nlanguage: [en, jam]\n"
            "tags: [helix, corxlabs, jamaica, coding-agent, long-context]\n---\n"
            "# CorX5.5\n\nCorxLabs' 34B HELIX coding agent, trained from scratch in Kingston, Jamaica. Training has "
            "not started yet: `code/` holds the software and the startup script; checkpoints arrive under "
            "`checkpoints/corx5.5` and `milestones/` once it runs. The plan is `code/CORX5.5.md`.\n"
        )
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=args.repo,
            commit_message="CorX5.5: created",
        )
        print("uploaded README.md")
    print(f"{'public' if args.public else 'private'} repository ready: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
