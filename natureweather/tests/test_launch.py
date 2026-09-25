# Copyright 2026 Nathan. Apache-2.0.
"""
Background training: the run must outlive the notebook that started it.

A multi-day run in a notebook cell dies with the kernel -- a dropped wifi connection was enough. These
pin the properties that make the detached run trustworthy: it survives its launcher exiting, only one
runs at a time, stopping it lets it checkpoint, and it runs exactly the pipeline the cell runs.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import torch
from naturev1 import Trainer, TrainSettings, follow, launch, status, stop
from naturev1.launch import _set_setting


HERE = Path(__file__).resolve().parents[1]

FAKE = textwrap.dedent('''
    SMOKE_TEST    = False     # a comment that must survive
    ROLLOUT_TRAIN = 4_000     # steps
    import pathlib, sys, time
    out = pathlib.Path.cwd()
    print("settings", SMOKE_TEST, ROLLOUT_TRAIN, flush=True)
    for i in range(3):
        print(f"\\rstaging {i}/3", end="", flush=True)
    print(flush=True)
    try:
        for step in range(600):
            print(f"step {step}", flush=True)
            time.sleep(0.1)
    finally:
        (out / "checkpointed").write_text("yes")
''')


def _wait(condition, seconds=20.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.2)
    return False


def test_the_packaged_pipeline_is_the_cell():
    """Two copies of the pipeline would drift; the packaged one must be the cell, byte for byte."""
    packaged = (HERE / "naturev1" / "_pipeline.py").read_text()
    cell = (HERE / "colab_naturev1.py").read_text()
    assert packaged.endswith(cell), "naturev1/_pipeline.py is out of date: regenerate it from the cell"


def test_importing_the_pipeline_does_not_start_training():
    with pytest.raises(ImportError, match="launch"):
        import naturev1._pipeline  # noqa: F401


def test_settings_are_overridden_in_place():
    text = _set_setting(FAKE, "ROLLOUT_TRAIN", 1000)
    assert "ROLLOUT_TRAIN = 1000     # steps" in text
    assert "SMOKE_TEST    = False     # a comment that must survive" in text
    with pytest.raises(KeyError, match="not a pipeline setting"):
        _set_setting(FAKE, "NOT_A_SETTING", 1)


def test_run_survives_its_launcher_and_stops_cleanly(tmp_path):
    fake = tmp_path / "fake_pipeline.py"
    fake.write_text(FAKE)
    home = tmp_path / "data"

    # Launch from a separate Python process that then exits -- a notebook kernel dying, in miniature.
    code = (f"import naturev1; naturev1.launch(directory={str(home)!r}, script={str(fake)!r}, "
            f"ROLLOUT_TRAIN=7, SMOKE_TEST=True)")
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, timeout=60)

    assert _wait(lambda: status(home) is not None), "the run should be alive after its launcher exited"
    assert _wait(lambda: "step 5" in (home / "train.log").read_text())
    shown = follow(directory=home)
    assert "settings True 7" in shown, "the overrides reached the running pipeline"
    assert "staging 2/3" in shown and "staging 0/3" not in shown, "progress bars collapse to one line"
    assert "RUNNING" in shown

    pid = status(home)["pid"]
    assert launch(directory=home, script=fake) == pid, "a second launch must not start a second run"

    assert stop(directory=home, wait=20)
    assert status(home) is None
    assert (home / "checkpointed").exists(), "stopping must let the run's finally-blocks checkpoint"
    assert "NOT RUNNING" in follow(directory=home)


def test_token_goes_through_the_environment_not_the_file(tmp_path):
    fake = tmp_path / "fake_pipeline.py"
    fake.write_text(FAKE.replace('print("settings"', 'import os; print("token", os.environ.get("HF_TOKEN"))\nprint("settings"'))
    home = tmp_path / "data"
    launch(directory=home, script=fake, HF_TOKEN="hf_secret_for_test")
    try:
        assert _wait(lambda: "token hf_secret_for_test" in (home / "train.log").read_text())
        assert "hf_secret_for_test" not in (home / "naturev1_pipeline.py").read_text()
    finally:
        stop(directory=home, wait=20)


def test_an_interrupted_stage_says_so(tmp_path):
    """The pipeline must not start the next stage from a half-trained model after a stop."""
    from naturev1 import NatureConfig, NatureV1

    from ihelix import fibonacci_sphere

    model = NatureV1(NatureConfig(analysis_channels=4, hidden_size=16, num_layers=1, num_heads=2,
                                  num_kv_heads=1, head_dim=8, intermediate_size=32, latent_points=32,
                                  history_frames=2, lead_times_hours=(6,)),
                     fibonacci_sphere(32, num_neighbours=8, cluster_size=8))
    grid = fibonacci_sphere(40, num_neighbours=8, cluster_size=8)

    def batches():
        for index in range(50):
            if index == 2:
                os.kill(os.getpid(), signal.SIGINT)   # what the Stop button sends
            yield {"analysis": torch.randn(1, 2, 40, 4), "analysis_grid": grid, "output_grid": grid,
                   "calendar": torch.zeros(1, 2, 6), "field_target": torch.zeros(1, 40, 1, 7),
                   "field_mask": torch.ones(1, 40, 1, 7)}

    trainer = Trainer(model, TrainSettings(max_steps=50, checkpoint_dir=str(tmp_path), precision="fp32",
                                           warmup_steps=1, log_every=100), device="cpu")
    state = trainer.fit(batches(), epochs=1)
    assert trainer.interrupted and state.step < 50
    assert json.loads((tmp_path / "state.json").read_text())["step"] == state.step, "checkpointed on the way out"


# ----------------------------------------------------------------------------------- the supervisor --

def _script(tmp_path, body: str) -> Path:
    fake = tmp_path / "fake_pipeline.py"
    fake.write_text("SMOKE_TEST = False\nimport pathlib, sys, time\nhere = pathlib.Path.cwd()\n"
                    "attempt = int((here / 'attempts').read_text()) + 1 if (here / 'attempts').exists() else 1\n"
                    "(here / 'attempts').write_text(str(attempt))\nprint('attempt', attempt, flush=True)\n"
                    + textwrap.dedent(body))
    return fake


def _finished(home):
    return _wait(lambda: status(home) is None, seconds=60)


def test_a_stalled_run_is_restarted_from_its_checkpoint(tmp_path):
    """A read that never returns: alive, silent, stuck. The supervisor must notice and restart it."""
    fake = _script(tmp_path, """
        if attempt == 1:
            time.sleep(600)          # hangs, and never touches the heartbeat
        print('finished', flush=True)
    """)
    home = tmp_path / "data"
    launch(directory=home, script=fake, stall_minutes=0.05, poll_seconds=0.5)
    assert _finished(home)
    log = (home / "train.log").read_text()
    assert "no progress for" in log and "attempt 2" in log and "the pipeline finished" in log


def test_a_crashed_run_is_restarted(tmp_path):
    fake = _script(tmp_path, """
        if attempt == 1:
            raise RuntimeError('CUDA out of memory (simulated)')
        print('finished', flush=True)
    """)
    home = tmp_path / "data"
    launch(directory=home, script=fake, stall_minutes=1.0, poll_seconds=0.5)
    assert _finished(home)
    log = (home / "train.log").read_text()
    assert "exited with code 1" in log and "attempt 2" in log and "the pipeline finished" in log


def test_a_broken_run_is_not_restarted_forever(tmp_path):
    """Failing within minutes, three times running, is an error to fix, not a blip to retry."""
    fake = _script(tmp_path, "raise SystemExit('DATA PROBLEM: simulated')\n")
    home = tmp_path / "data"
    launch(directory=home, script=fake, stall_minutes=1.0, poll_seconds=0.5)
    assert _finished(home)
    log = (home / "train.log").read_text()
    assert "3 times in a row" in log and "Not restarting" in log
    assert (home / "attempts").read_text() == "3"


def test_stop_ends_the_supervisor_and_the_run(tmp_path):
    fake = _script(tmp_path, """
        try:
            time.sleep(600)
        finally:
            (here / 'checkpointed').write_text('yes')
    """)
    home = tmp_path / "data"
    launch(directory=home, script=fake, stall_minutes=10.0, poll_seconds=0.5)
    assert _wait(lambda: (home / "train.log").exists() and "attempt 1" in (home / "train.log").read_text())
    child = json.loads((home / "train.pid").read_text()).get("child")
    assert stop(directory=home, wait=30)
    assert status(home) is None
    assert (home / "checkpointed").exists(), "the run must get to checkpoint on the way out"
    assert child and not Path(f"/proc/{child}").exists(), "no orphaned run left behind"
    assert "stopped on request" in (home / "train.log").read_text()
    assert (home / "attempts").read_text() == "1", "a requested stop is not a crash to restart"
