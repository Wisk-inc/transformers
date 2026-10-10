# Copyright 2026 Nathan. Apache-2.0.
"""
Keep what a session built on the Hub, so the next session starts where this one stopped.

A hosted notebook (molab, Colab) survives nothing: the session ends on a timer, and its disk goes with
it -- the 45 GB of staged reanalysis, the checkpoints, the storm feature banks, the remembered batch
size. The Hugging Face Hub does survive, and uploads from these machines run at tens of megabytes a
second. So everything expensive is mirrored there as soon as it exists, in the background, and pulled
back at the start of the next session before anything is rebuilt. Staging that took half an hour from
the cloud archive takes minutes from the Hub, and training resumes from the last pushed checkpoint.

Nothing here is required: without a repository or a token the mirror says so once and does nothing,
and the pipeline behaves exactly as it does on a machine with a disk that lasts.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path

from .safeio import read_json


class HubMirror:
    """
    Push and pull files against one Hub repository.

    Args:
        repo_id: ``"you/NatureV1"``. None disables everything.
        token: a write token. Without one, pulls from a public repository still work; pushes do not.
    """

    def __init__(self, repo_id: str | None, token: str | None = None, private: bool = False) -> None:
        self.repo_id = repo_id
        self.private = private
        self.token = token or os.environ.get("HF_TOKEN") or None
        if self.token is None and repo_id:
            try:
                from huggingface_hub import get_token

                self.token = get_token()
            except Exception:
                self.token = None
        self._threads: list[threading.Thread] = []
        self._said_no_push = False
        self._lock = threading.Lock()

    @property
    def can_pull(self) -> bool:
        return bool(self.repo_id)

    @property
    def can_push(self) -> bool:
        return bool(self.repo_id and self.token)

    def _note_no_push(self) -> None:
        with self._lock:
            if not self._said_no_push:
                where = "no HF_REPO" if not self.repo_id else "no HF_TOKEN"
                print(f"[mirror] {where}: nothing is mirrored to the Hub, so a new session starts from scratch",
                      flush=True)
                self._said_no_push = True

    # ------------------------------------------------------------------------------------ pulling --

    def remote_size(self, repo_path: str) -> int | None:
        """Size in bytes of a file in the repository, or None if it is not there (or cannot be asked)."""
        if not self.can_pull:
            return None
        try:
            from huggingface_hub import HfApi

            info = HfApi(token=self.token).get_paths_info(self.repo_id, [repo_path], repo_type="model")
            for entry in info:
                size = getattr(entry, "size", None)
                if size is None and getattr(entry, "lfs", None) is not None:
                    size = entry.lfs.get("size") if isinstance(entry.lfs, dict) else getattr(entry.lfs, "size", None)
                return int(size) if size is not None else None
        except Exception:
            return None
        return None

    def pull(self, local: str | os.PathLike, repo_path: str, what: str | None = None) -> bool:
        """
        Download ``repo_path`` to ``local`` if ``local`` does not exist yet. Returns True if it did.

        The download lands in a side folder and is moved into place whole, so a session that ends
        mid-download leaves nothing that looks like a finished file.
        """
        local = Path(local)
        if local.exists() or not self.can_pull:
            return False
        if self.remote_size(repo_path) is None:
            return False
        what = what or local.name
        staging = local.parent / ".hub"
        try:
            from huggingface_hub import hf_hub_download

            try:
                from huggingface_hub.utils import disable_progress_bars

                disable_progress_bars()
            except ImportError:
                pass
            began = time.time()
            print(f"[mirror] pulling {what} from {self.repo_id} ...", flush=True)
            downloaded = hf_hub_download(self.repo_id, repo_path, repo_type="model", local_dir=str(staging),
                                         token=self.token)
            local.parent.mkdir(parents=True, exist_ok=True)
            os.replace(downloaded, local)
            size = local.stat().st_size / 1e9
            print(f"[mirror] pulled {what}: {size:.2f} GB in {time.time() - began:.0f} s", flush=True)
            return True
        except Exception as error:
            print(f"[mirror] could not pull {what} ({type(error).__name__}: {error}); building it here", flush=True)
            return False
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def pull_staging(self, local_npy: str | os.PathLike, repo_stem: str, variables, indices) -> bool:
        """
        Pull a staged array and its sidecar, if the Hub holds one for these variables that covers
        any of the steps wanted -- :func:`naturev1.materialise` then reuses or grows it as usual.
        """
        local_npy = Path(local_npy)
        sidecar = local_npy.with_suffix(".json")
        if local_npy.exists() or not self.can_pull:
            return False
        if not self.pull(sidecar, f"{repo_stem}.json", what=f"{local_npy.stem} metadata"):
            return False
        meta = read_json(sidecar, default={}) or {}
        wanted = {int(i) for i in indices}
        held = set(meta.get("indices") or [])
        if list(meta.get("variables") or []) != list(variables) or not (wanted & held):
            sidecar.unlink(missing_ok=True)
            print(f"[mirror] the Hub's {local_npy.name} is for different variables or years; staging here", flush=True)
            return False
        if self.pull(local_npy, f"{repo_stem}.npy", what=local_npy.name):
            return True
        sidecar.unlink(missing_ok=True)                # no array to go with it
        return False

    # ------------------------------------------------------------------------------------ pushing --

    def push(self, local: str | os.PathLike, repo_path: str, background: bool = True,
             what: str | None = None) -> bool:
        """
        Upload ``local`` to ``repo_path`` unless the Hub already holds a file of the same size.

        Background by default: a 40 GB staging uploads while training runs. Returns True if an upload
        was started (or done, when ``background`` is False).
        """
        local = Path(local)
        if not local.exists():
            return False
        if not self.can_push:
            self._note_no_push()
            return False
        what = what or local.name
        size = local.stat().st_size
        if self.remote_size(repo_path) == size:
            return False                              # already there, byte count and all

        def upload() -> None:
            began = time.time()
            try:
                from huggingface_hub import HfApi

                try:
                    from huggingface_hub.utils import disable_progress_bars

                    disable_progress_bars()
                except ImportError:
                    pass
                api = HfApi(token=self.token)
                api.create_repo(self.repo_id, repo_type="model", exist_ok=True, private=self.private)
                api.upload_file(path_or_fileobj=str(local), path_in_repo=repo_path, repo_id=self.repo_id,
                                repo_type="model", commit_message=f"mirror {what}")
                print(f"[mirror] pushed {what} to {self.repo_id} ({size / 1e9:.2f} GB in {time.time() - began:.0f} s)",
                      flush=True)
            except Exception as error:                # a failed mirror must never stop a run
                print(f"[mirror] push of {what} skipped ({type(error).__name__}: {error})", flush=True)

        if not background:
            upload()
            return True
        thread = threading.Thread(target=upload, name=f"mirror-{what}", daemon=True)
        thread.start()
        self._threads.append(thread)
        return True

    def push_staging(self, local_npy: str | os.PathLike, repo_stem: str) -> bool:
        """Push a staged array, then its sidecar -- so a sidecar on the Hub always has its array."""
        local_npy = Path(local_npy)
        sidecar = local_npy.with_suffix(".json")
        if not (local_npy.exists() and sidecar.exists()) or not self.can_push:
            if not self.can_push:
                self._note_no_push()
            return False
        if self.remote_size(f"{repo_stem}.npy") == local_npy.stat().st_size:
            return False

        def both() -> None:
            self.push(local_npy, f"{repo_stem}.npy", background=False, what=local_npy.name)
            self.push(sidecar, f"{repo_stem}.json", background=False, what=sidecar.name)

        thread = threading.Thread(target=both, name=f"mirror-{local_npy.stem}", daemon=True)
        thread.start()
        self._threads.append(thread)
        return True

    def wait(self, timeout: float | None = None) -> None:
        """Block until background uploads finish -- before a session is allowed to end."""
        for thread in list(self._threads):
            thread.join(timeout)

    def describe(self) -> str:
        if not self.repo_id:
            return "mirror: off (no HF_REPO)"
        if not self.token:
            return f"mirror: pull-only from {self.repo_id} (no HF_TOKEN, so nothing is pushed)"
        return f"mirror: {self.repo_id} (pull and push)"


def session_started() -> float | None:
    """
    When this machine's session began, as a unix time -- on a container, the start of process 1.

    Hosted notebooks end a session on a timer counted from here, not from when training started.
    """
    try:
        return os.stat("/proc/1").st_ctime
    except OSError:
        return None


__all__ = ["HubMirror", "session_started"]

