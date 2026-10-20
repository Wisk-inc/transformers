# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Checkpoints to and from the Hugging Face Hub: uploads in the background, downloads on resume.

A year-long run pushes its export hundreds of times. Every push to the same path would otherwise leave
the previous files in the repository's LFS history, so after each upload the history is squashed to a
single commit: the repository holds the current export plus the milestone copies, nothing else.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path


class HubPusher:
    """
    Uploads a folder to ``repo/path_in_repo`` without blocking training. One upload at a time: a new push
    waits for the previous one, so a slow link delays the next push rather than piling them up.
    """

    def __init__(
        self, repo: str | None, path_in_repo: str, token: str | None = None, log=print, private: bool = True,
        squash: bool = True,
    ) -> None:  # fmt: skip
        self.repo, self.path_in_repo, self.log = repo, path_in_repo, log
        self.token = token or os.environ.get("HF_TOKEN")
        self.private, self.squash = private, squash
        self.thread: threading.Thread | None = None
        self.pushed = 0
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.repo)

    def push(self, folder: str | Path, message: str, also_copy_to: str | None = None) -> None:
        """Upload ``folder`` to the pusher's path; ``also_copy_to`` keeps a second copy under that path."""
        if not self.enabled:
            return
        self.wait()
        self.thread = threading.Thread(target=self._upload, args=(Path(folder), message, also_copy_to), daemon=True)
        self.thread.start()

    def wait(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            self.thread.join()

    def _upload(self, folder: Path, message: str, also_copy_to: str | None) -> None:
        from huggingface_hub import HfApi

        api = HfApi(token=self.token)
        started = time.time()
        for attempt in range(4):
            try:
                api.create_repo(self.repo, exist_ok=True, private=self.private)
                targets = [self.path_in_repo] + ([also_copy_to] if also_copy_to else [])
                for target in targets:
                    api.upload_folder(
                        folder_path=str(folder),
                        repo_id=self.repo,
                        path_in_repo=target,
                        commit_message=message if target == self.path_in_repo else f"{message} (milestone)",
                        ignore_patterns=["*.part", "*.tmp"],
                    )
                if self.squash:
                    try:
                        api.super_squash_history(repo_id=self.repo)
                    except Exception as error:  # a squash failure costs storage, not correctness
                        self.log(f"[hub] history squash skipped: {type(error).__name__}: {str(error)[:120]}")
                self.pushed += 1
                self.last_error = None
                self.log(
                    f"[hub] pushed {folder.name} to {self.repo}/{self.path_in_repo}"
                    + (f" and {also_copy_to}" if also_copy_to else "")
                    + f" in {time.time() - started:.0f}s"
                )
                return
            except Exception as error:
                self.last_error = f"{type(error).__name__}: {error}"
                self.log(f"[hub] push attempt {attempt + 1} failed: {self.last_error[:200]}")
                time.sleep(15 * (attempt + 1))


def upload_text(
    repo: str, path_in_repo: str, text: str, message: str, token: str | None = None, private: bool = True
) -> None:
    """One small file (a model card, a manifest) straight to the repo."""
    from huggingface_hub import HfApi

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    api.create_repo(repo, exist_ok=True, private=private)
    api.upload_file(path_or_fileobj=text.encode(), path_in_repo=path_in_repo, repo_id=repo, commit_message=message)


def pull_folder(repo: str, path_in_repo: str, local_dir: str | Path, token: str | None = None) -> Path | None:
    """Download ``repo/path_in_repo`` into ``local_dir`` (a resume on a fresh machine). ``None`` if absent."""
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    if not api.repo_exists(repo):
        return None
    files = [f for f in api.list_repo_files(repo) if f.startswith(path_in_repo.rstrip("/") + "/")]
    if not files:
        return None
    snapshot_download(repo, allow_patterns=[path_in_repo.rstrip("/") + "/*"], local_dir=local_dir, token=api.token)
    return Path(local_dir) / path_in_repo
