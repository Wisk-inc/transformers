# Copyright 2026 Nathan. Apache-2.0.
"""
Run the whole training pipeline in the background, so a dropped connection cannot stop it.

Training inside a notebook cell lives exactly as long as the notebook's kernel. Close the laptop, lose
the wifi, restart the kernel -- the run dies with it, and a multi-day run on a remote GPU dies many
times. :func:`launch` starts the same pipeline as a separate process on the machine the GPU is in,
detached from the notebook. The notebook can disconnect, restart or close; the run carries on, and
:func:`follow` picks up its log from any new session.

Everything stays pure Python, so it works in marimo, Colab and Jupyter alike -- no shell, no ``!``.

    import naturev1
    naturev1.launch(ROLLOUT_TRAIN=1000)   # returns at once; training runs on the server
    naturev1.follow()                     # the latest log lines, from any session, any time
    naturev1.stop()                       # checkpoint and end the run
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


LOG = "train.log"
PID = "train.pid"
SCRIPT = "naturev1_pipeline.py"
RUNNER = "naturev1_runner.py"

#: The runner is what the process executes. It exists for one reason: a DataLoader that uses the
#: ``spawn`` start method re-imports the main module in every worker, and a pipeline script run as
#: ``__main__`` would start a second training run inside each one. The guard below is the fix.
#: SIGTERM becomes KeyboardInterrupt, so every stage's ``finally`` writes its checkpoint on the way out.
_RUNNER = '''import runpy
import signal


def _interrupt(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _interrupt)
    runpy.run_path({script!r}, run_name="__naturev1_pipeline__")
'''


def data_directory(directory: str | os.PathLike | None = None) -> Path:
    """Where the pipeline keeps its data, log and checkpoints -- the same rule the cell uses."""
    if directory is not None:
        return Path(directory)
    return Path("/content") if os.path.isdir("/content") else Path.home() / "naturev1_data"


def _set_setting(text: str, name: str, value) -> str:
    """Replace one ``NAME = value`` line in the pipeline's settings block, keeping its comment."""
    pattern = re.compile(rf"^({re.escape(name)}\s*=\s*)([^#\n]*?)(\s*(#.*)?)$", re.MULTILINE)
    if not pattern.search(text):
        known = re.findall(r"^([A-Z][A-Z0-9_]*)\s*=", text.split("# ═══ 0 ═══")[0], re.MULTILINE)
        raise KeyError(f"{name} is not a pipeline setting. Settings: {', '.join(known)}")
    return pattern.sub(lambda m: f"{m.group(1)}{value!r}{m.group(3)}", text, count=1)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    cmdline = Path(f"/proc/{pid}/cmdline")
    if cmdline.exists():
        # A recycled pid belonging to some other program is not our run.
        return RUNNER.encode() in cmdline.read_bytes() or b"naturev1" in cmdline.read_bytes()
    return True


def status(directory: str | os.PathLike | None = None) -> dict | None:
    """The running pipeline's pid, start time and log path, or None if nothing is running."""
    marker = data_directory(directory) / PID
    if not marker.exists():
        return None
    try:
        info = json.loads(marker.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return info if _alive(int(info["pid"])) else None


def launch(directory: str | os.PathLike | None = None, script: str | os.PathLike | None = None,
           restart: bool = False, **settings) -> int:
    """
    Start the training pipeline as a background process on this machine. Returns its pid.

    The process is started in its own session, so it is not a child the notebook can take down with
    it: the kernel can restart, the browser can disconnect, the notebook can close. It writes to
    ``train.log`` in the data directory; :func:`follow` shows it. Checkpoints go where the cell puts
    them, so a run that is launched after an in-notebook run resumes from it rather than starting over.

    Only one run at a time: two would fight over the GPU and write the same checkpoints. If one is
    already going, this says so and returns its pid; ``restart=True`` stops it first.

    Args:
        settings: any setting from the top of the cell, by name -- ``ROLLOUT_TRAIN=1000``,
            ``SMOKE_TEST=True``, ``RUN_PUBLISH=True``. ``HF_TOKEN`` is passed through the environment,
            never written into the script on disk.
        script: run this file instead of the packaged pipeline (for testing).
    """
    home = data_directory(directory)
    home.mkdir(parents=True, exist_ok=True)
    running = status(home)
    if running and not restart:
        print(f"[launch] already running (pid {running['pid']}, started {running['started']}). "
              "naturev1.follow() to watch it, naturev1.stop() to end it, or launch(restart=True).")
        return int(running["pid"])
    if running:
        stop(home)

    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    token = settings.pop("HF_TOKEN", None) or settings.pop("hf_token", None)
    if token:
        environment["HF_TOKEN"] = token

    source = Path(script) if script is not None else Path(__file__).with_name("_pipeline.py")
    text = source.read_text()
    for name, value in settings.items():
        text = _set_setting(text, name.upper(), value)
    target = home / SCRIPT
    target.write_text(text)
    runner = home / RUNNER
    runner.write_text(_RUNNER.format(script=str(target)))

    log = home / LOG
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    with open(log, "a") as handle:
        handle.write(f"\n{'=' * 99}\n[launch] {stamp}  settings {settings or 'defaults'}\n{'=' * 99}\n")
        process = subprocess.Popen(
            [sys.executable, "-u", str(runner)], cwd=str(home), env=environment,
            stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,          # detached: its own session, not the notebook's child
        )
    (home / PID).write_text(json.dumps({"pid": process.pid, "started": stamp, "log": str(log),
                                        "settings": {k: repr(v) for k, v in settings.items()}}))
    print(f"[launch] training started in the background (pid {process.pid}).\n"
          f"         It keeps running if this notebook disconnects or restarts.\n"
          f"         Log: {log}\n"
          f"         naturev1.follow() shows progress; naturev1.stop() ends it.")
    return process.pid


def follow(lines: int = 40, directory: str | os.PathLike | None = None) -> str:
    """
    Print (and return) the last lines of the run's log, with whether it is still going.

    Progress bars write ``\\r`` to redraw a line in place; only the latest state of each such line is
    shown, so a staging progress bar is one line here and not two thousand.
    """
    home = data_directory(directory)
    log = home / LOG
    if not log.exists():
        text = f"no log at {log} -- nothing has been launched from this directory"
        print(text)
        return text
    with open(log, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(size - 400_000, 0))
        tail = handle.read().decode("utf-8", errors="replace")
    rows = [row.split("\r")[-1] for row in tail.split("\n")]
    shown = "\n".join(rows[-lines:])
    running = status(home)
    state = (f"[follow] RUNNING, pid {running['pid']}, started {running['started']}" if running
             else "[follow] NOT RUNNING -- finished, stopped, or crashed; the lines above say which")
    text = f"{shown}\n{state}"
    print(text)
    return text


def stop(directory: str | os.PathLike | None = None, wait: float = 60.0) -> bool:
    """
    End the run: ask it to stop, give it ``wait`` seconds to checkpoint, then make sure.

    SIGTERM first -- the trainer catches it, finishes its step and writes a checkpoint -- and SIGKILL
    to the whole process group only if it is still there after ``wait``. Checkpoints go down every
    minute regardless, so the worst case is losing the last minute. Returns True if a run was stopped.
    """
    home = data_directory(directory)
    running = status(home)
    if not running:
        print("[stop] nothing is running")
        return False
    pid = int(running["pid"])
    # The main process only: its DataLoader workers are shut down by it on the way out. Signalling the
    # whole group would kill the workers first and fail the step the trainer is trying to finish.
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    deadline = time.time() + wait
    while time.time() < deadline and _alive(pid):
        time.sleep(1.0)
    if _alive(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        time.sleep(1.0)
    (home / PID).unlink(missing_ok=True)
    print(f"[stop] run {pid} ended; re-launching resumes from its last checkpoint")
    return True
