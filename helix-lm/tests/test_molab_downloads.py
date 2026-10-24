# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The molab program keeps only whole downloads: a body cut short by the server or a proxy (which ``urllib``
returns as if complete) is resumed and verified against the published size and md5, a corrupt archive is
deleted and fetched again, two runs cannot share one root, and the data stage is complete only when every
source it wants is on disk."""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from helix_lm.scripts import tristream_molab as molab

PAYLOAD = bytes(range(256)) * 1200  # 300 KB
MD5 = "md5:" + hashlib.md5(PAYLOAD).hexdigest()  # noqa: S324


class Server:
    """A file server that can cut the first answer short, ignore ranges, or answer with HTML."""

    def __init__(self, cut: int = 0, ignore_range: bool = False, html_first: bool = False) -> None:
        self.cut, self.ignore_range, self.html_first = cut, ignore_range, html_first
        self.requests: list[str] = []
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):  # quiet
                pass

            def do_GET(self):  # noqa: N802
                server.requests.append(self.headers.get("Range", ""))
                if server.html_first and len(server.requests) == 1:
                    body = b"<html>rate limited</html>"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                start, end = 0, len(PAYLOAD) - 1
                rng = self.headers.get("Range")
                if rng and not server.ignore_range:
                    first, _, last = rng.split("=")[1].partition("-")
                    start, end = int(first), int(last) if last else len(PAYLOAD) - 1
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(PAYLOAD)}")
                else:
                    self.send_response(200)
                body = PAYLOAD[start : end + 1]
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if server.cut and len(server.requests) == 1:
                    self.wfile.write(body[: server.cut])
                    self.wfile.flush()
                    self.connection.close()  # the body stops here: urllib sees a clean end of stream
                    return
                self.wfile.write(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/file.zip"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(molab.time, "sleep", lambda *_: None)


def test_cut_short_download_is_resumed_and_verified(tmp_path):
    server = Server(cut=100_000)
    try:
        target = molab.download(
            server.url, tmp_path / "file.zip", size=len(PAYLOAD), checksum=MD5, log=lambda *_: None
        )
    finally:
        server.close()
    assert target.read_bytes() == PAYLOAD
    assert not (tmp_path / "file.zip.part").exists()
    assert server.requests[0] == "" and server.requests[1] == "bytes=100000-"


def test_server_that_ignores_ranges_starts_over(tmp_path):
    server = Server(cut=100_000, ignore_range=True)
    try:
        target = molab.download(
            server.url, tmp_path / "file.zip", size=len(PAYLOAD), checksum=MD5, log=lambda *_: None
        )
    finally:
        server.close()
    assert target.read_bytes() == PAYLOAD


def test_html_answer_falls_back_to_the_other_route(tmp_path):
    bad, good = Server(html_first=True), Server()
    try:
        target = molab.download(
            bad.url, tmp_path / "file.zip", size=len(PAYLOAD), checksum=MD5, alternate=good.url, log=lambda *_: None
        )
    finally:
        bad.close()
        good.close()
    assert target.read_bytes() == PAYLOAD
    assert len(bad.requests) == 1 and len(good.requests) == 1


def test_wrong_checksum_never_gets_the_final_name(tmp_path):
    server = Server()
    try:
        with pytest.raises(OSError, match="md5 mismatch"):
            molab.download(server.url, tmp_path / "file.zip", size=len(PAYLOAD), checksum="md5:" + "0" * 32,
                           attempts=2, log=lambda *_: None)  # fmt: skip
    finally:
        server.close()
    assert not (tmp_path / "file.zip").exists() and not (tmp_path / "file.zip.part").exists()


def test_bad_file_on_disk_is_replaced(tmp_path):
    (tmp_path / "file.zip").write_bytes(b"not the file")
    server = Server()
    try:
        molab.download(server.url, tmp_path / "file.zip", size=len(PAYLOAD), checksum=MD5, log=lambda *_: None)
    finally:
        server.close()
    assert (tmp_path / "file.zip").read_bytes() == PAYLOAD


def test_extract_archives_deletes_a_corrupt_archive_and_extracts_a_whole_one(tmp_path):
    (tmp_path / "broken.zip").write_bytes(b"PK\x03\x04 this is not a zip file" * 100)
    with zipfile.ZipFile(tmp_path / "good.zip", "w") as handle:
        handle.writestr("clip.wav", b"RIFF" + bytes(100))
    assert molab.extract_archives(tmp_path) == 1
    assert not (tmp_path / "broken.zip").exists()
    assert (tmp_path / "extracted" / "good" / "clip.wav").exists()
    assert (tmp_path / "extracted" / "good" / ".extracted").exists()
    assert molab.extract_archives(tmp_path) == 0  # done once


def test_run_lock_refuses_a_live_holder_and_takes_over_from_a_dead_one(tmp_path):
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (tmp_path / "run.lock").write_text(json.dumps({"pid": other.pid, "started": "now"}))
        assert molab.take_lock(tmp_path) is False
    finally:
        other.kill()
        other.wait()
    (tmp_path / "run.lock").write_text(json.dumps({"pid": 2**22 - 1, "started": "long ago"}))
    assert molab.take_lock(tmp_path) is True
    assert json.loads((tmp_path / "run.lock").read_text())["pid"] == os.getpid()


def test_data_stage_is_complete_only_with_every_source_on_disk(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(molab, "token", lambda: None)
    args = SimpleNamespace(skip=[], emilia_hours=300.0, data_attempts=5)
    (tmp_path / "markers").mkdir()
    (tmp_path / "markers" / "data").write_text("x")
    found = dict.fromkeys(molab.CLEAN_SOURCES, 1)
    (tmp_path / "data_found.json").write_text(json.dumps(found))
    assert molab.missing_sources(tmp_path, args) == ["vocalset", "dagstuhl-choirset", "esmuc"]
    assert molab.complete(tmp_path, "data", args) is False
    for name in ("vocalset", "dagstuhl-choirset", "esmuc"):
        marker = tmp_path / "raw" / name / "extracted" / name / ".extracted"
        marker.parent.mkdir(parents=True)
        marker.write_text("done")
    assert molab.complete(tmp_path, "data", args) is True
    args.skip = ["esmuc"]
    (tmp_path / "raw" / "esmuc" / "extracted" / "esmuc" / ".extracted").unlink()
    assert molab.complete(tmp_path, "data", args) is True
    # with a token, Emilia is wanted too
    monkeypatch.setattr(molab, "token", lambda: "hf_x")
    assert molab.missing_sources(tmp_path, args) == ["emilia-yodas"]


def test_prepare_is_redone_for_a_source_that_arrived_later(tmp_path):
    args = SimpleNamespace(skip=[], emilia_hours=0.0)
    (tmp_path / "markers").mkdir()
    (tmp_path / "markers" / "prepare").write_text("x")
    (tmp_path / "prepared" / "uploads").mkdir(parents=True)
    (tmp_path / "prepared" / "uploads" / "manifest.json").write_text("{}")
    assert molab.complete(tmp_path, "prepare", args) is True
    (tmp_path / "raw" / "vocalset" / "extracted").mkdir(parents=True)
    (tmp_path / "raw" / "vocalset" / "extracted" / "a.wav").write_bytes(b"RIFF")
    assert molab.complete(tmp_path, "prepare", args) is False
    (tmp_path / "prepared" / "raw_vocalset").mkdir()
    (tmp_path / "prepared" / "raw_vocalset" / "manifest.json").write_text("{}")
    assert molab.complete(tmp_path, "prepare", args) is True


def test_prepare_folders_match_the_prepare_stage():
    source = Path(molab.__file__).read_text()
    for folder in molab.PREPARE_FOLDERS:
        assert f'("{folder}",' in source


def test_parallel_pieces_join_into_the_whole_file(tmp_path):
    server = Server(cut=50_000)  # the first connection is cut short; its piece resumes
    try:
        target = molab.download(server.url, tmp_path / "file.zip", size=len(PAYLOAD), checksum=MD5,
                                parallel_from=1, log=lambda *_: None)  # fmt: skip
    finally:
        server.close()
    assert target.read_bytes() == PAYLOAD
    assert not list(tmp_path.glob("file.zip.part*"))
    assert sum(r.startswith("bytes=") for r in server.requests) >= 16


def test_parallel_falls_back_to_one_connection_when_ranges_are_refused(tmp_path):
    server = Server(ignore_range=True)
    try:
        target = molab.download(server.url, tmp_path / "file.zip", size=len(PAYLOAD), checksum=MD5,
                                parallel_from=1, log=lambda *_: None)  # fmt: skip
    finally:
        server.close()
    assert target.read_bytes() == PAYLOAD
    assert not list(tmp_path.glob("file.zip.part*"))


def test_a_file_already_on_disk_is_checked_once_then_never_fetched(tmp_path, monkeypatch):
    (tmp_path / "file.zip").write_bytes(PAYLOAD)
    server = Server()
    lines: list[str] = []
    try:
        molab.download(server.url, tmp_path / "file.zip", size=len(PAYLOAD), checksum=MD5, log=lines.append)
        assert server.requests == [] and "whole, not downloaded again" in lines[-1]
        monkeypatch.setattr(molab, "file_md5", lambda *_: pytest.fail("checked again"))
        molab.download(server.url, tmp_path / "file.zip", size=len(PAYLOAD), checksum=MD5, log=lines.append)
        assert server.requests == [] and "already on disk and checked" in lines[-1]
    finally:
        server.close()


def test_a_fresh_download_is_stamped_so_the_next_run_skips_the_check(tmp_path, monkeypatch):
    server = Server()
    try:
        molab.download(server.url, tmp_path / "file.zip", size=len(PAYLOAD), checksum=MD5, log=lambda *_: None)
        monkeypatch.setattr(molab, "file_md5", lambda *_: pytest.fail("checked again"))
        molab.download(server.url, tmp_path / "file.zip", size=len(PAYLOAD), checksum=MD5, log=lambda *_: None)
    finally:
        server.close()
    assert len(server.requests) == 1


def test_an_unpacked_archive_is_not_fetched_again_even_when_the_zip_is_gone(tmp_path):
    assert not molab.archive_extracted(tmp_path, "VocalSet.zip")
    marker = tmp_path / "extracted" / "VocalSet" / ".extracted"
    marker.parent.mkdir(parents=True)
    marker.write_text("done")
    assert molab.archive_extracted(tmp_path, "VocalSet.zip")
    assert molab.archive_extracted(tmp_path / "x", "EN-B000000.tar") is False
    assert not molab.archive_extracted(tmp_path, "notes.csv")


def test_leftover_archive_of_an_unpacked_shard_is_removed(tmp_path):
    import tarfile

    shard = tmp_path / "Emilia-YODAS" / "EN" / "EN-B000000.tar"
    shard.parent.mkdir(parents=True)
    with tarfile.open(shard, "w"):
        pass
    marker = tmp_path / "extracted" / "EN-B000000" / ".extracted"
    marker.parent.mkdir(parents=True)
    marker.write_text("done")
    assert molab.extract_archives(tmp_path, delete=True) == 0
    assert not shard.exists()


def test_an_old_single_connection_partial_is_reused_by_the_parallel_download(tmp_path):
    (tmp_path / "file.zip.part").write_bytes(PAYLOAD[:200_000])
    server = Server()
    lines: list[str] = []
    try:
        target = molab.download(server.url, tmp_path / "file.zip", size=len(PAYLOAD), checksum=MD5,
                                parallel_from=1, log=lines.append)  # fmt: skip
    finally:
        server.close()
    assert target.read_bytes() == PAYLOAD
    assert any("resuming" in line for line in lines)
    fetched = 0
    for r in server.requests:
        first, _, last = r.split("=")[1].partition("-")
        fetched += int(last) - int(first) + 1
    assert fetched <= len(PAYLOAD) - 200_000 + 20_000  # only what the old file did not already hold


def test_the_data_stage_run_twice_downloads_nothing_the_second_time(tmp_path, monkeypatch):
    import io
    import tarfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as handle:
        handle.writestr("singer/clip.wav", b"RIFF" + bytes(2000))
    archive = buffer.getvalue()
    md5 = "md5:" + hashlib.md5(archive).hexdigest()  # noqa: S324
    served: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):  # noqa: N802
            served.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(archive)))
            self.end_headers()
            self.wfile.write(archive)

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def fake_zenodo(record):
        name = {"1193957": "VocalSet.zip", "4607766": "Dagstuhl.zip", "5848990": "Esmuc.zip"}[record]
        return [{"name": name, "url": f"{base}/{name}", "size": len(archive), "checksum": md5, "record": ""}]

    snapshots: list[list[str]] = []

    def fake_snapshot(repo, repo_type, allow_patterns, local_dir, token):
        snapshots.append(list(allow_patterns))
        for pattern in allow_patterns:
            shard = Path(local_dir) / pattern
            shard.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(shard, "w") as handle:
                data = b"RIFF" + bytes(2000)
                info = tarfile.TarInfo("clip.mp3")
                info.size = len(data)
                handle.addfile(info, io.BytesIO(data))

    import huggingface_hub

    monkeypatch.setattr(molab, "zenodo_files", fake_zenodo)
    monkeypatch.setattr(molab, "token", lambda: "hf_x")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    args = SimpleNamespace(root=str(tmp_path), skip=["slakh2100"], max_gb=50.0, emilia_hours=150.0,
                           include_research=False, data_attempts=5)  # fmt: skip
    try:
        molab.stage_data(args)
        assert len(served) == 3 and snapshots == [[f"Emilia-YODAS/EN/EN-B{i:06d}.tar" for i in range(3)]]
        assert molab.complete(tmp_path, "data", args)
        assert not list((tmp_path / "raw" / "emilia-yodas").rglob("*.tar"))  # unpacked, then removed
        # second run: everything is on disk, so nothing is fetched
        molab.stage_data(args)
        assert len(served) == 3 and len(snapshots) == 1
        # a zip removed by hand: still unpacked, so still not fetched
        (tmp_path / "raw" / "vocalset" / "VocalSet.zip").unlink()
        molab.stage_data(args)
        assert len(served) == 3 and len(snapshots) == 1
        # one more hour of speech asked for: only the new shard is fetched
        args.emilia_hours = 200.0
        molab.stage_data(args)
        assert snapshots[-1] == ["Emilia-YODAS/EN/EN-B000003.tar"]
    finally:
        httpd.shutdown()
        httpd.server_close()
