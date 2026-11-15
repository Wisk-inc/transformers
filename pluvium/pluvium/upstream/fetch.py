# Copyright 2026 Nathan. Apache-2.0.
"""
Copy a published set of weights to a local folder, unchanged, with its licence and a record of where it came from.

``plan`` asks the publisher which files there are and how large they are. ``fetch_model`` downloads them (a
download that stopped is continued, not started again), checks every size and, where the publisher gives one,
every checksum, and writes two files beside the copy:

``LICENSE``
    The text of the licence or licences the publisher states for the weights.
``SOURCE.md``
    What the files are, who made them, where and when they were copied from, the notice the licence asks to be
    kept, and the SHA-256 of every file.

Nothing here needs a Hugging Face account: the buckets and ECMWF's repositories are public.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from ..util import FetchError, log
from .registry import Upstream

HUB = "https://huggingface.co"
AGENT = {"User-Agent": "pluvium-upstream/1"}
LICENCES = Path(__file__).parent / "licences"
DEFAULT_CACHE = "~/.cache/pluvium/upstream"


def _local(url: str) -> str | None:
    """The path behind a ``file://`` address or a plain path (how the tests stand in for a bucket), else None."""
    if url.startswith("file://"):
        return urllib.parse.unquote(url[7:])
    return None if url.startswith("http") else url


def _open(url: str, headers: dict | None = None, method: str = "GET", timeout: float = 120.0):
    request = urllib.request.Request(url, headers={**AGENT, **(headers or {})}, method=method)
    return urllib.request.urlopen(request, timeout=timeout)


def size_of(url: str) -> int:
    """How many bytes the file at ``url`` holds, asked without downloading it."""
    path = _local(url)
    if path is not None:
        return os.path.getsize(path)
    last = None
    for k in range(4):
        try:
            with _open(url, method="HEAD", timeout=60.0) as response:
                return int(response.headers["Content-Length"])
        except urllib.error.HTTPError as error:
            if error.code in (400, 401, 403, 404):
                raise FetchError(f"{url}: HTTP {error.code}") from error
            last = error
        except Exception as error:                              # timeouts, resets: try again
            last = error
        time.sleep(1.5 * 2**k)
    raise FetchError(f"{url}: {type(last).__name__}: {last}")


def plan(model: Upstream) -> list:
    """The files to copy: ``[{"path", "url", "size", "sha256"}]``, sizes as the publisher reports them now.
    ``sha256`` is None where the publisher gives no checksum."""
    if model.kind == "http":
        out = []
        for entry in model.files:
            name, saved_as = entry if isinstance(entry, tuple) else (entry, entry)
            url = model.location + (name if _local(model.location) is not None else urllib.parse.quote(name))
            out.append({"path": saved_as, "url": url, "size": size_of(url), "sha256": None})
        return out
    if model.kind == "hub":
        with _open(f"{HUB}/api/models/{model.location}?blobs=true", timeout=60.0) as response:
            info = json.load(response)
        commit = info["sha"]                                    # one commit, so a copy cannot mix two versions
        out = []
        for item in info["siblings"]:
            name = item["rfilename"]
            if name == ".gitattributes":
                continue
            lfs = item.get("lfs") or {}
            out.append({"path": name, "url": f"{HUB}/{model.location}/resolve/{commit}/{urllib.parse.quote(name)}",
                        "size": int(lfs.get("size", item.get("size") or 0)), "sha256": lfs.get("sha256"),
                        "commit": commit})
        return out
    raise ValueError(f"{model.name}: unknown kind {model.kind!r}")


def sha256_of(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path, size: int | None = None, sha256: str | None = None, attempts: int = 6,
             chunk: int = 1 << 20) -> Path:
    """Stream ``url`` into ``target``. A file already there with the right size is kept. A partial download
    (``target`` + ``.part``) is continued from where it stopped. The size, and the checksum when one is given, must
    match, or the file is removed and a :class:`pluvium.util.FetchError` raised."""
    target = Path(target)
    if target.exists() and (size is None or target.stat().st_size == size):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    path = _local(url)
    last = None
    for k in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        if size is not None and have > size:                    # a stale partial file from another version
            part.unlink()
            have = 0
        if size is not None and have == size:
            break
        try:
            if path is not None:
                with open(path, "rb") as source, open(part, "ab") as out:
                    source.seek(have)
                    while block := source.read(chunk):
                        out.write(block)
            else:
                with _open(url, {"Range": f"bytes={have}-"} if have else None) as response:
                    if have and response.status != 206:         # the server sent the whole file: start again
                        have = 0
                    with open(part, "ab" if have else "wb") as out:
                        while block := response.read(chunk):
                            out.write(block)
            break
        except urllib.error.HTTPError as error:
            if error.code == 416 and have:                      # nothing after what is already here: it is whole
                break
            if error.code in (400, 401, 403, 404):
                raise FetchError(f"{url}: HTTP {error.code}") from error
            last = error
        except Exception as error:                              # a dropped connection: continue from the part
            last = error
        time.sleep(min(30.0, 1.5 * 2**k))
    else:
        raise FetchError(f"{url}: {type(last).__name__}: {last}")
    got = part.stat().st_size
    if size is not None and got != size:
        part.unlink()
        raise FetchError(f"{url}: {got:,} bytes arrived, {size:,} expected")
    if sha256 is not None and sha256_of(part) != sha256:
        part.unlink()
        raise FetchError(f"{url}: the checksum does not match the one the publisher gives")
    part.replace(target)
    return target


def licence_text(model: Upstream) -> str:
    """The licence or licences of a model, as one text."""
    parts = []
    for name, address, file in model.licences:
        parts.append(f"{name}\n{address}\n\n" + (LICENCES / file).read_text(encoding="utf-8"))
    if len(parts) == 1:
        return parts[0]
    rule = "\n\n" + "=" * 78 + "\n\n"
    return (f"Two licences apply to the files in this folder. {model.attribution}\n\n" + "=" * 78 + "\n\n"
            + rule.join(parts))


def source_note(model: Upstream, files: list, when: str) -> str:
    """The text of ``SOURCE.md``: what was copied, from where, under which terms."""
    commits = sorted({f["commit"] for f in files if f.get("commit")})
    where = f"https://huggingface.co/{model.location}" if model.kind == "hub" else model.location
    lines = [f"# {model.title}", "", model.note, "",
             f"These files were copied unchanged from {where} on {when}"
             + (f", at commit `{commits[0]}`" if commits else "") + ". Pluvium did not make them and does not change "
             "them.", "",
             f"Publisher: {model.producer}. Its own page for them: {model.homepage}", "",
             "## Licence", "",
             f"{model.licence}. The full text is in `LICENSE` in this folder. " + model.attribution, "",
             "The rest of this repository is Pluvium's own work under Apache 2.0. That licence does not cover this "
             "folder.", ""]
    if model.papers:
        lines += ["## Papers", ""] + [f"- {p}" for p in model.papers] + [""]
    lines += ["## Files", "", "| File | Bytes | SHA-256 |", "|---|---:|---|"]
    lines += [f"| `{f['path']}` | {f['size']:,} | `{f['sha256']}` |" for f in files]
    return "\n".join(lines) + "\n"


def fetch_model(model: Upstream, cache=DEFAULT_CACHE, files: list | None = None, say=log) -> Path:
    """Copy one model into ``cache/<name>`` and return that folder. ``files`` is a plan made earlier (see
    :func:`plan`), to save asking the publisher twice."""
    folder = Path(cache).expanduser() / model.name
    files = plan(model) if files is None else files
    total = sum(f["size"] for f in files)
    say(f"upstream: {model.title}: {len(files)} files, {total / 1e6:,.0f} MB -> {folder}")
    done = []
    for f in files:
        target = folder / f["path"]
        fresh = not (target.exists() and target.stat().st_size == f["size"])
        download(f["url"], target, f["size"], f["sha256"] if fresh else None)
        done.append({**f, "sha256": f["sha256"] or sha256_of(target)})
        if fresh and f["size"] > 50e6:
            say(f"upstream:   {f['path'][:70]} ({f['size'] / 1e6:,.0f} MB)")
    now = datetime.now(timezone.utc)
    when = f"{now.day} {now:%B %Y}"
    (folder / "LICENSE").write_text(licence_text(model), encoding="utf-8")
    (folder / "SOURCE.md").write_text(source_note(model, done, when), encoding="utf-8")
    return folder


__all__ = ["DEFAULT_CACHE", "download", "fetch_model", "licence_text", "plan", "sha256_of", "size_of", "source_note"]
