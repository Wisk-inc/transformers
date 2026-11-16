# Copyright 2026 Nathan. Apache-2.0.
"""The upstream weights: the registry, the checked copy, and the mirror into a Hub repository."""

import dataclasses
import hashlib
import sys
from types import SimpleNamespace

import pytest

from pluvium import upstream
from pluvium.upstream import fetch as F
from pluvium.upstream.registry import CC0, CC_BY, MODELS, Upstream
from pluvium.util import FetchError, PluviumError, hub_refusal


def bucket(tmp_path, files):
    """A folder standing in for a public bucket, and the registry entry that points at it."""
    root = tmp_path / "bucket"
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return Upstream(name="toy", title="Toy weights", producer="Somebody", feeds="graphcast", kind="http",
                    location=str(root) + "/", files=tuple(files), licences=(CC0, CC_BY),
                    attribution="Toy weights by Somebody.", homepage="https://example.org/toy", papers=("arXiv:0000.00000",),
                    note="Weights for a test.", size_mb=1)


FILES = {"params/a b.npz": b"x" * 5000, "stats/mean.nc": b"y" * 300}


def test_registry_is_complete_and_names_what_feeds_pluvium():
    names = [m.name for m in MODELS]
    assert len(names) == len(set(names)) and {m.feeds for m in MODELS} == {"graphcast", "aifs"}
    for m in MODELS:
        assert m.kind in ("http", "hub") and m.licences and m.attribution and m.homepage.startswith("https://")
        assert all((F.LICENCES / file).exists() for _, _, file in m.licences)      # the licence text is shipped
        assert (m.kind == "http") == bool(m.files)
        if m.kind == "http":
            assert m.location.startswith("https://") and m.location.endswith("/")
    assert {m.feeds for m in MODELS if m.current} == {"graphcast", "aifs"}        # one in use today behind each input
    assert upstream.get("graphcast").licence == "CC BY 4.0"
    assert upstream.get("noaa-graphcastgfs").licence == "CC0 1.0 and CC BY 4.0"   # NOAA's terms and DeepMind's
    assert [m.name for m in upstream.select(["graphcast"])] == ["graphcast"]
    assert len(upstream.select()) == len(MODELS) and all(m.current for m in upstream.select(current_only=True))
    with pytest.raises(KeyError):
        upstream.select(["no-such-model"])


def test_copy_is_checked_and_carries_licence_and_source(tmp_path):
    model = bucket(tmp_path, FILES)
    files = upstream.plan(model)
    assert [(f["path"], f["size"]) for f in files] == [("params/a b.npz", 5000), ("stats/mean.nc", 300)]
    said = []
    folder = upstream.fetch_model(model, tmp_path / "cache", say=said.append)
    assert (folder / "params" / "a b.npz").read_bytes() == FILES["params/a b.npz"]
    licence = (folder / "LICENSE").read_text()
    assert "CC0 1.0 Universal" in licence and "Attribution 4.0 International" in licence and "Two licences" in licence
    note = (folder / "SOURCE.md").read_text()
    assert "copied unchanged" in note and "Toy weights by Somebody." in note and "arXiv:0000.00000" in note
    assert hashlib.sha256(FILES["stats/mean.nc"]).hexdigest() in note and "Apache 2.0" in note
    assert not list(folder.rglob("*.part"))
    upstream.fetch_model(model, tmp_path / "cache", say=said.append)               # again: nothing is copied twice
    assert len([s for s in said if "Toy weights" in s]) == 2


def test_stopped_download_continues_and_a_wrong_file_is_refused(tmp_path):
    source = tmp_path / "big.bin"
    source.write_bytes(bytes(range(256)) * 400)
    target = tmp_path / "out" / "big.bin"
    target.parent.mkdir()
    (target.parent / "big.bin.part").write_bytes(source.read_bytes()[:30_000])     # a download that stopped
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    F.download(str(source), target, size=102_400, sha256=digest)
    assert target.read_bytes() == source.read_bytes() and not (target.parent / "big.bin.part").exists()
    with pytest.raises(FetchError, match="expected"):
        F.download(str(source), tmp_path / "out" / "short.bin", size=999)
    with pytest.raises(FetchError, match="checksum"):
        F.download(str(source), tmp_path / "out" / "bad.bin", size=102_400, sha256="0" * 64)
    assert not (tmp_path / "out" / "bad.bin").exists() and not (tmp_path / "out" / "bad.bin.part").exists()


class FakeApi:
    """The few methods of ``huggingface_hub.HfApi`` the mirror uses, recording what was uploaded."""

    def __init__(self):
        self.files, self.commits, self.created = {}, [], None

    def create_repo(self, repo, repo_type="model", exist_ok=True, private=False):
        self.created = (repo, private)

    def model_info(self, repo, files_metadata=True):
        return SimpleNamespace(siblings=[SimpleNamespace(rfilename=k, size=v) for k, v in self.files.items()])

    def upload_folder(self, folder_path, path_in_repo, repo_id, repo_type, commit_message, ignore_patterns=None):
        from pathlib import Path

        for path in Path(folder_path).rglob("*"):
            if path.is_file() and not path.name.endswith(".part"):
                self.files[f"{path_in_repo}/{path.relative_to(folder_path).as_posix()}"] = path.stat().st_size
        self.commits.append(commit_message)

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        self.files[path_in_repo] = len(path_or_fileobj)
        self.index = path_or_fileobj.decode()
        self.commits.append(commit_message)


def test_mirror_copies_once_skips_what_is_there_and_survives_a_failure(tmp_path, monkeypatch):
    good = bucket(tmp_path, FILES)
    broken = Upstream(name="gone", title="Gone weights", producer="Nobody", feeds="aifs", kind="http",
                      location=str(tmp_path / "nowhere") + "/", files=("x.bin",), licences=(CC_BY,), attribution="None.",
                      homepage="https://example.org/gone", papers=(), note="Not there.", size_mb=1)
    # the package exports the function `mirror`, which hides the module of the same name: take the module itself
    monkeypatch.setattr(sys.modules["pluvium.upstream.mirror"], "select", lambda names=None: [good, broken])
    api, said = FakeApi(), []
    results = upstream.mirror("org/repo", cache=tmp_path / "cache", api=api, say=said.append)
    assert results["toy"] == "copied" and results["gone"].startswith("failed")
    assert api.created == ("org/repo", False)
    assert api.files["upstream/toy/params/a b.npz"] == 5000 and "upstream/toy/SOURCE.md" in api.files
    assert "upstream/toy/LICENSE" in api.files and "upstream/README.md" in api.files
    assert "Toy weights" in api.index and "Gone weights" not in api.index and "does not run these models" in api.index
    assert "unchanged" in api.commits[0] and "CC0 1.0 and CC BY 4.0" in api.commits[0]
    again = upstream.mirror("org/repo", cache=tmp_path / "cache", api=api, say=said.append)
    assert again["toy"] == "already there" and len([c for c in api.commits if c.startswith("Mirror")]) == 1
    assert upstream.summary(again) == "0 copied, 1 already there, 1 failed (gone)"
    with pytest.raises(SystemExit):                                                 # without a token nothing goes up
        monkeypatch.delenv("HF_TOKEN", raising=False)
        upstream.mirror("org/repo", cache=tmp_path / "cache")

    class Refused(Exception):
        response = SimpleNamespace(status_code=403)

    def refuse(*args, **kwargs):
        raise Refused("403 Forbidden")

    monkeypatch.setattr(api, "create_repo", refuse)                                 # a token that may not write there
    with pytest.raises(PluviumError, match="needs write access to org"):
        upstream.mirror("org/repo", cache=tmp_path / "cache", api=api)
    assert "pasted whole" in str(hub_refusal(SimpleNamespace(response=SimpleNamespace(status_code=401)), "org/repo"))
    assert hub_refusal(ValueError("something else"), "org/repo") is None


def test_mirroring_one_model_keeps_the_others_in_the_index(tmp_path, monkeypatch):
    first = bucket(tmp_path, FILES)
    second = dataclasses.replace(first, name="other", title="Other weights")
    everything = [first, second]
    monkeypatch.setattr(sys.modules["pluvium.upstream.mirror"], "select",
                        lambda names=None: [m for m in everything if not names or m.name in names])
    api = FakeApi()
    both = upstream.mirror("org/repo", cache=tmp_path / "cache", api=api, say=lambda *said: None)
    assert both == {"toy": "copied", "other": "copied"}
    assert "(toy/SOURCE.md)" in api.index and "(other/SOURCE.md)" in api.index
    one = upstream.mirror("org/repo", names=["other"], cache=tmp_path / "cache", api=api, say=lambda *said: None)
    assert one == {"other": "already there"}
    assert "(toy/SOURCE.md)" in api.index and "(other/SOURCE.md)" in api.index      # the one not asked for is still listed
