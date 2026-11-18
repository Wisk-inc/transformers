# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The TriStream program on molab: a session that ends after 12 hours, a prepare stage that goes on from the clips
already done, and a failure that keeps coming back stopping the run instead of being tried for days."""

import math
import time
from pathlib import Path

import pytest
import torch

from helix_lm import molab
from helix_lm.audio.data import write_wave
from helix_lm.molab import Session, SessionEnding


def test_the_session_end_comes_from_the_setting_or_from_molab():
    assert molab.session_limit({"HELIX_SESSION_HOURS": "6"}) == (6.0, "HELIX_SESSION_HOURS")
    assert molab.session_limit({"HELIX_SESSION_HOURS": "0"})[0] is None
    on = molab.on_molab({}, places=["/tmp/uv-venv/lib/python3.13/site-packages/helix_lm/molab.py"], commands=[])
    assert on and molab.on_molab({}, places=["/usr"], commands=["python -m marimo edit nb.py"])
    assert not molab.on_molab({"HOME": "/root"}, places=["/usr/lib"], commands=["bash"])
    began = time.time() - 3600
    assert molab.session_start(12, {"HELIX_SESSION_START": str(began)}) == began
    found = molab.session_start(12, {})
    assert time.time() - 12 * 3600 < found <= time.time()


def test_a_session_wraps_up_half_an_hour_before_its_end():
    now = time.time()
    s = Session(12.0, began=now - 11 * 3600, why="test")
    assert not s.wrap_up(now) and math.isclose(s.left_hours(now), 1.0, abs_tol=1e-6)
    assert s.wrap_up(now + 31 * 60) and "stops by" in s.describe()
    assert s.environment()["HELIX_SESSION_HOURS"] == "12"
    free = Session(None)
    assert not free.wrap_up() and free.left_hours() == float("inf") and "no session end" in free.describe()


def _songs(folder: Path, n: int, sample_rate: int = 24000) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    t = torch.arange(sample_rate) / sample_rate
    out = []
    for k in range(n):
        wave = 0.3 * torch.sin(2 * math.pi * (220 + 20 * k) * t) * (1 + 0.1 * torch.sin(2 * math.pi * 5 * t))
        path = folder / f"take{k}.wav"
        write_wave(path, wave, sample_rate)
        out.append(path)
    return out


def test_prepare_goes_on_from_the_clips_done_before_a_stop(tmp_path, monkeypatch):
    from helix_lm.music import data as D

    songs = _songs(tmp_path / "in", 4)
    out = tmp_path / "out"
    measured = []
    real = D.measure_clip

    def counting(path, *args, **kwargs):
        measured.append(Path(path).name)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(D, "measure_clip", counting)
    calls = {"n": 0}

    def stop_after_two() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(SessionEnding, match="2 of 4"):
        D.prepare_uploads([tmp_path / "in"], out, analysis=False, should_stop=stop_after_two)
    assert not (out / "manifest.json").exists() and sorted(measured) == ["take0.wav", "take1.wav"]
    manifest = D.prepare_uploads([tmp_path / "in"], out, analysis=False)
    assert sorted(measured) == ["take0.wav", "take1.wav", "take2.wav", "take3.wav"]   # the first two not again
    assert len(manifest["clips"]) == 4 and (out / "manifest.json").exists()
    (out / "take3.f0.pt").unlink()                                                     # an output lost: redone
    D.prepare_uploads([tmp_path / "in"], out, analysis=False)
    assert measured.count("take3.wav") == 2 and measured.count("take0.wav") == 1
    assert len(songs) == 4


def test_two_files_with_one_name_do_not_overwrite_each_other(tmp_path):
    from helix_lm.music.data import prepare_uploads

    _songs(tmp_path / "a", 1)
    _songs(tmp_path / "b", 1)                                                         # also take0.wav
    manifest = prepare_uploads([tmp_path / "a", tmp_path / "b"], tmp_path / "out", analysis=False)
    reasons = [c["reason"] for c in manifest["clips"]]
    assert sum(r.startswith("same file name as") for r in reasons) == 1


def test_the_program_saves_pushes_and_stops_before_the_session_ends(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from helix_lm.scripts import tristream_molab as M

    pushed = []
    monkeypatch.setattr(M, "hub_push", lambda args, force=False, reason="": pushed.append(reason) or True)
    monkeypatch.setattr(M, "SESSION", Session(12.0, began=time.time() - 11.8 * 3600, why="test"))
    budget = M.Budget(10.0)
    assert not M.keep_going(budget, tmp_path)
    with pytest.raises(SessionEnding):
        M.stop_if_session_ends(SimpleNamespace(), "train")
    assert pushed == ["session end"]
    monkeypatch.setattr(M, "SESSION", Session(None))
    assert M.keep_going(budget, tmp_path)
    (tmp_path / "stop").write_text("")
    assert not M.keep_going(budget, tmp_path)


def test_a_failure_that_keeps_coming_back_stops_the_run(tmp_path, monkeypatch):
    from helix_lm.scripts import tristream_molab as M

    tries = []

    def broken(args):
        tries.append(1)
        raise RuntimeError("the same fault")

    monkeypatch.setattr(M, "STAGES", ("setup", "swap"))
    monkeypatch.setattr(M, "stage_setup", broken)
    for name in ("hub_push", "hub_pull", "gpu_max"):
        monkeypatch.setattr(M, name, lambda *a, **k: None)
    monkeypatch.setenv("HELIX_SESSION_HOURS", "0")
    with pytest.raises(RuntimeError, match="the same fault"):
        M.main(["--root", str(tmp_path), "--retries", "100", "--retry-wait", "0", "--hf-repo", ""])
    assert len(tries) == M.SAME_FAILURE_LIMIT


def test_a_catch_all_cannot_swallow_the_session_end(monkeypatch):
    """The data stage catches Exception around each source; in 2.4.0 that handler swallowed the session end."""
    from types import SimpleNamespace

    from helix_lm.scripts import tristream_molab as M

    pushes = []
    monkeypatch.setattr(M, "hub_push", lambda args, force=False, reason="": pushes.append(reason) or True)
    monkeypatch.setattr(M, "SESSION", Session(12.0, began=time.time() - 11.9 * 3600, why="test"))
    assert not issubclass(SessionEnding, Exception)
    reached = []
    with pytest.raises(SessionEnding) as ending:
        for name in ("vocalset", "dagstuhl", "esmuc"):
            try:
                M.stop_if_session_ends(SimpleNamespace(), f"data ({name})")
                reached.append(name)
            except Exception:  # what the data stage does around each source
                continue
    assert reached == [] and pushes == ["session end"] and ending.value.pushed


def _prepared(out: Path, stem: str, f0: torch.Tensor) -> dict:
    t = torch.arange(24000 * 2) / 24000
    write_wave(out / f"{stem}.vocal.wav", 0.3 * torch.sin(2 * math.pi * 110.0 * t), 24000)
    torch.save(f0, out / f"{stem}.f0.pt")
    return {"path": f"/data/{stem}.wav", "seconds": 2.0, "tags": {"singer": "s", "f0_low": 200.0, "f0_high": 230.0},
            "analysis": {}, "f0_path": str(out / f"{stem}.f0.pt"), "accepted": True, "reason": ""}


def test_pitch_files_from_the_old_tracker_are_made_again(tmp_path):
    import json

    from helix_lm.music.data import PITCH_TRACKER, refresh_pitch

    out = tmp_path / "prepared" / "raw_vocalset"
    out.mkdir(parents=True)
    octave_up = torch.full((51,), 220.0)                       # the old tracker's error on a 110 Hz voice
    manifest = {"clips": [_prepared(out, f"take{k}", octave_up) for k in range(3)], "accepted": 3}
    (out / "manifest.json").write_text(json.dumps(manifest))
    calls = {"n": 0}

    def stop_after_one() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    with pytest.raises(SessionEnding):
        refresh_pitch(out, json.loads((out / "manifest.json").read_text()), 24000, should_stop=stop_after_one,
                      log=lambda *_: None)
    assert list(json.loads((out / "pitch_refresh.json").read_text())) == ["take0"]
    done = refresh_pitch(out, json.loads((out / "manifest.json").read_text()), 24000, log=lambda *_: None)
    assert done["pitch_tracker"] == PITCH_TRACKER and not (out / "pitch_refresh.json").exists()
    for k in range(3):
        f0 = torch.load(out / f"take{k}.f0.pt")
        assert abs(float(f0[f0 > 0].median()) - 110.0) < 2.0
    assert abs(done["clips"][0]["tags"]["f0_high"] - 110.0) < 3.0
    assert json.loads((out / "manifest.json").read_text())["pitch_tracker"] == PITCH_TRACKER


def test_the_benchmark_singers_are_never_trained_on():
    from types import SimpleNamespace

    from helix_lm.scripts import tristream_molab as M

    tags = [SimpleNamespace(singer=s) for s in ("vocalset/f2", "vocalset/m10", "vocalset/f1", "emilia/x", "f2")]
    split = M.singer_split(tags)
    assert split["vocalset/f2"] == split["vocalset/m10"] == "validation"
    from helix_lm.music.regularize import split_singers

    plain = split_singers([t.singer for t in tags])
    assert all(split[s] == plain[s] for s in ("vocalset/f1", "emilia/x", "f2"))  # everyone else as before
    assert M.benchmark_file(Path("/r/prepared/raw_vocalset/m5_arpeggios_belt_c_a.vocal.wav"))
    assert not M.benchmark_file(Path("/r/prepared/raw_vocalset/m4_arpeggios_belt_c_a.vocal.wav"))
    assert not M.benchmark_file(Path("/r/prepared/uploads/f2_take.vocal.wav"))
