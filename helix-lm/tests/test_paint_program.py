# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The Pinxit training program and the ``helix paint`` command, without a network: every stage on a few
pictures of one's own, a session that ends in the middle, the Hub mirror through a stand-in, the second process
that fetches pictures while the practice sheets train, and the notebook cell."""

import json
import sys
from pathlib import Path

import pytest
import torch

import helix_lm
from helix_lm.molab import Session
from helix_lm.paint.io import save_image
from helix_lm.scripts import pinxit_molab as program

CELL = Path(__file__).resolve().parents[1] / "notebooks" / "pinxit_molab_cell.py"


def _pictures(root: Path, photos: int = 3, paintings: int = 2) -> None:
    for kind, count in (("photos", photos), ("paintings", paintings)):
        folder = root / "uploads" / kind
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            g = torch.Generator().manual_seed(i + (10 if kind == "paintings" else 0))
            img = torch.rand(3, 1, 1, generator=g) * 0.5 + torch.rand(3, 96, 128, generator=g) * 0.25
            img[:, 20:60, 30 + 10 * i:80 + 10 * i] = torch.rand(3, 1, 1, generator=g)
            save_image(img.clamp(0, 1), folder / f"{kind}_{i}.png")


def _args(root: Path, *more) -> list:
    return ["--root", str(root), "--stage", "all", "--size", "px-tiny", "--eye", "tiny", "--hf-repo", "",
            "--chicago", "0", "--cleveland", "0", "--washington", "0", "--photos", "0", "--sheet-hours", "0.5",
            "--sheet-steps", "2", "--picture-hours", "0.5", "--picture-steps", "2", "--batch", "4",
            "--train-work", "96", "--workers", "0", "--gallery", "1", "--gallery-work", "96",
            "--pictures-per-step", "2", "--picture-cells", "24", "--retries", "1", *more]


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(program, "SESSION", None)
    monkeypatch.setattr(program, "FETCHER", None)
    monkeypatch.setattr(program, "LAST_PUSH", 0.0)
    monkeypatch.setattr(program, "TAG", "")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(program, "token", lambda: None)
    monkeypatch.setattr(Session, "find", classmethod(lambda cls, env=None: Session(None)))


def test_every_stage_runs_on_ones_own_pictures_and_a_second_run_has_nothing_left_to_do(tmp_path, capfd):
    capsys = capfd                                                # the second process writes past sys.stdout
    _pictures(tmp_path)
    assert program.main(_args(tmp_path)) == 0
    said = capsys.readouterr().out
    for stage in program.STAGES:
        assert (tmp_path / "markers" / stage).exists(), stage
    assert "[fetch] data: 3 photos, 2 paintings" in said          # the second process did the data stage,
    assert "== stage data" not in said and "== stage pictures" in said            # so the run did not have to
    assert len(list((tmp_path / "uploads" / "paintings").glob("*.card.json"))) == 2
    side = json.loads((tmp_path / "runs" / "px-tiny" / "latest.json").read_text())
    assert side["stage"] == "pictures" and side["step"] == 2
    state = torch.load(tmp_path / "runs" / "px-tiny" / "latest.pt", weights_only=True)
    assert state["eye"] == "tiny" and state["ema"] is not None
    rows = json.loads((tmp_path / "gallery" / "metrics.json").read_text())
    assert {r["hand"] for r in rows} == {"rules", "pinxit"} and all(r["strokes"] > 0 for r in rows)
    assert (tmp_path / "gallery" / "00_pinxit.png").exists() and (tmp_path / "gallery" / "00_passes.png").exists()
    assert (tmp_path / "run.lock").read_text() != "" and "all stages done" in said
    assert program.main(_args(tmp_path)) == 0
    again = capsys.readouterr().out
    for stage in ("setup", "sheets", "data", "pictures", "gallery"):
        assert f"{stage}: done earlier, skipped" in again
    assert "[fetch]" not in again                                 # nothing to fetch, so no second process


def test_more_hours_asked_later_continue_a_stage(tmp_path, capsys):
    _pictures(tmp_path)
    assert program.main(_args(tmp_path)) == 0
    capsys.readouterr()
    more = [a if a != "0.5" else "0.6" for a in _args(tmp_path)]
    more[more.index("--picture-steps") + 1] = "4"
    more[more.index("--sheet-steps") + 1] = "3"
    assert program.main(more) == 0
    said = capsys.readouterr().out
    assert "[sheets] went on from step 2" in said and "[pictures] went on from step 2" in said
    side = json.loads((tmp_path / "runs" / "px-tiny" / "latest.json").read_text())
    assert side["progress"] == {"sheets": {"step": 3, "hours": side["progress"]["sheets"]["hours"]},
                                "pictures": {"step": 4, "hours": side["progress"]["pictures"]["hours"]}}


def test_a_session_that_ends_saves_and_the_next_one_goes_on(tmp_path, capsys, monkeypatch):
    _pictures(tmp_path)
    calls = {"n": 0}

    def ending(cls, env=None):                                    # the session ends once training has begun
        session = Session(None)
        session.wrap_up = lambda now=None: calls["n"] > 0
        return session

    monkeypatch.setattr(Session, "find", classmethod(ending))
    monkeypatch.setattr(program, "start_fetcher", lambda args: None)
    real = program.stage_sheets

    def sheets(args):
        real(args)
        calls["n"] += 1
        program.stop_if_session_ends(args, "sheets")

    monkeypatch.setattr(program, "stage_sheets", sheets)
    assert program.main(_args(tmp_path)) == 0
    said = capsys.readouterr().out
    assert "run the cell again in a new session" in said and not (tmp_path / "markers" / "data").exists()
    assert (tmp_path / "runs" / "px-tiny" / "latest.pt").exists()
    monkeypatch.setattr(Session, "find", classmethod(lambda cls, env=None: Session(None)))
    monkeypatch.setattr(program, "SESSION", None)
    monkeypatch.setattr(program, "stage_sheets", real)
    assert program.main(_args(tmp_path)) == 0
    assert (tmp_path / "markers" / "push").exists() and "== stage data" in capsys.readouterr().out


class _Hub:
    def __init__(self, files=()):
        self.files, self.commits, self.made = list(files), [], []

    def create_repo(self, repo, exist_ok=True, private=True):
        self.made.append((repo, private))

    def create_commit(self, repo, operations, commit_message):
        self.commits.append((repo, [op.path_in_repo for op in operations], commit_message, operations))

    def super_squash_history(self, repo):
        pass

    def list_repo_files(self, repo):
        return self.files


def test_the_run_mirrors_itself_to_the_hub_and_the_card_says_how_to_use_the_model(tmp_path, monkeypatch, capsys):
    _pictures(tmp_path)
    hub = _Hub()
    monkeypatch.setattr(program, "token", lambda: "a-token-for-tests")
    monkeypatch.setattr(program, "hub_api", lambda: hub)
    monkeypatch.setattr(program, "start_fetcher", lambda args: None)
    args = [a if a != "" else "Someone/Painter" for a in _args(tmp_path)]
    assert program.main(args) == 0
    assert hub.made and hub.made[0] == ("Someone/Painter", True)                 # private unless asked otherwise
    repo, paths, message, ops = hub.commits[-1]
    assert message.startswith("release") and {"README.md", "pinxit/px-tiny/latest.pt", "pinxit/px-tiny/latest.json",
                                              "markers/pictures", "gallery/metrics.json"} <= set(paths)
    card = next(op for op in ops if op.path_in_repo == "README.md").path_or_fileobj.decode()
    assert 'load_hand("Someone/Painter", "px-tiny")' in card and "a-token-for-tests" not in card
    assert "license: apache-2.0" in card and "| rules" in card and "| pinxit" in card
    assert "a-token-for-tests" not in capsys.readouterr().out


def test_a_failure_that_keeps_coming_back_stops_the_run(tmp_path, monkeypatch, capsys):
    _pictures(tmp_path)
    monkeypatch.setattr(program, "start_fetcher", lambda args: None)

    def broken(args):
        raise RuntimeError("the same thing again")

    monkeypatch.setattr(program, "stage_sheets", broken)
    args = _args(tmp_path, "--retry-wait", "0")
    args[args.index("--retries") + 1] = "9"
    with pytest.raises(RuntimeError, match="the same thing again"):
        program.main(args)
    assert "came back 3 times in a row" in capsys.readouterr().out


def test_two_runs_cannot_share_a_folder(tmp_path, capsys):
    (tmp_path / "run.lock").write_text("1")                       # pid 1 is always there
    assert program.main(_args(tmp_path)) == 3
    assert "another run (pid 1)" in capsys.readouterr().out


def test_the_command_paints_reads_and_changes(tmp_path, capsys):
    from helix_lm.cli import main as helix
    from helix_lm.paint.cli import main as paint_cli

    _pictures(tmp_path, photos=1, paintings=1)
    photo = tmp_path / "uploads" / "photos" / "photos_0.png"
    style = tmp_path / "uploads" / "paintings" / "paintings_0.png"
    kept = tmp_path / "p.pinxit"
    assert paint_cli(["make", str(photo), "--style", str(style), "--emotion", "majesty", "--work", "96",
                      "--out", str(tmp_path / "p.png"), "--keep", str(kept), "--sheet", str(tmp_path / "s.png")]) == 0
    made = capsys.readouterr().out
    assert "plan:" in made and (tmp_path / "p.png").exists() and (tmp_path / "s.png").exists() and kept.exists()
    strokes = json.loads(made[made.index("{"):])["strokes"]
    assert paint_cli(["change", str(kept), "--add", str(style), "--box", "0.1,0.1,0.6,0.6",
                      "--out", str(tmp_path / "p2.png")]) == 0
    changed = json.loads(capsys.readouterr().out.split("\n{", 1)[1].join(["{", ""]))
    assert changed["strokes"] > strokes and (tmp_path / "p2.png").exists()
    assert paint_cli(["change", str(kept), "--undo"]) == 0
    assert json.loads(capsys.readouterr().out)["strokes"] == strokes
    assert paint_cli(["read", str(style), "--out", str(tmp_path / "card.json")]) == 0
    assert "feeling:" in capsys.readouterr().out and (tmp_path / "card.json").exists()
    with pytest.raises(SystemExit) as done:                       # reached through the main command too
        helix(["paint", "sizes"])
    assert done.value.code == 0 and "px-base" in capsys.readouterr().out
    with pytest.raises(ValueError, match="needs an eye that reads text"):
        paint_cli(["change", str(kept), "--words", "a red boat"])


def test_the_cell_is_plain_python_asks_for_this_version_and_holds_no_token():
    text = CELL.read_text()
    compile(text, str(CELL), "exec")
    assert 'HF_TOKEN = "PASTE_YOUR_ORG_TOKEN_HERE"' in text and f'HELIX = "{helix_lm.__version__}"' in text
    assert 'PROGRAM = "helix_lm.scripts.pinxit_molab"' in text and 'HF_REPO = "CorxLabs/Pinxit-1"' in text
    assert "helix-lm[paint]==" in text and "hf_" + "A" not in text
    scope: dict = {}
    exec(text[text.index("def plan("):text.index("code = run(")], scope)   # the split of the hours, on its own
    hours = dict(zip(scope["plan"](20)[::2], scope["plan"](20)[1::2], strict=True))
    sheets, pictures = float(hours["--sheet-hours"]), float(hours["--picture-hours"])
    assert 0.35 < sheets / (sheets + pictures) < 0.45 and 18.5 < sheets + pictures <= 20
    assert sys.version_info >= (3, 10)
