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
SUPERVISOR = "naturev1_supervisor.py"
HEARTBEAT = "heartbeat"

#: The supervisor loads this file directly rather than importing the package, so it stays a small
#: process that never imports torch and never touches the GPU the run is using.
_SUPERVISOR = """import importlib.util
spec = importlib.util.spec_from_file_location("naturev1_launch", {module!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.supervise({runner!r}, {home!r}, stall_minutes={stall_minutes!r}, max_restarts={max_restarts!r},
                 poll_seconds={poll_seconds!r})
"""

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
    proc = Path(f"/proc/{pid}")
    if proc.exists():
        try:
            state = (proc / "stat").read_text().rpartition(")")[2].split()[0]
            text = (proc / "cmdline").read_bytes()
        except (OSError, IndexError):
            return False                    # gone between the two reads
        if state in ("Z", "X"):
            return False                    # exited, and only waiting for its parent to reap it
        if not text:
            # Empty for a moment while a new process is still starting up. Reading that as "not
            # running" made follow() report a run it had just launched as stopped.
            return True
        # A recycled pid belonging to some other program is not our run.
        return RUNNER.encode() in text or SUPERVISOR.encode() in text or b"naturev1" in text
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
           restart: bool = False, supervise: bool = True, stall_minutes: float = 30.0,
           max_restarts: int = 20, poll_seconds: float = 30.0, **settings) -> int:
    """
    Start the training pipeline as a background process on this machine. Returns its pid.

    The process is started in its own session, so it is not a child the notebook can take down with
    it: the kernel can restart, the browser can disconnect, the notebook can close. It writes to
    ``train.log`` in the data directory; :func:`follow` shows it. Checkpoints go where the cell puts
    them, so a run that is launched after an in-notebook run resumes from it rather than starting over.

    Only one run at a time: two would fight over the GPU and write the same checkpoints. If one is
    already going, this says so and returns its pid; ``restart=True`` stops it first.

    With ``supervise`` (the default) the run is watched, and brought back when it stops for any reason
    but finishing: a crash, an out-of-memory error, a cloud read that hangs, a worker that deadlocks.
    See :func:`supervise`.

    Args:
        settings: any setting from the top of the cell, by name -- ``ROLLOUT_TRAIN=1000``,
            ``RUN_PUBLISH=True``. ``HF_TOKEN`` is passed through the environment, never written into the
            script on disk.
        stall_minutes: with ``supervise``, restart a run that has made no progress for this long.
        max_restarts: with ``supervise``, give up after this many restarts in total.
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

    environment = dict(os.environ, PYTHONUNBUFFERED="1", NATUREV1_BACKGROUND_CHILD="1")
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
    shown = {k: ("<set>" if "TOKEN" in k.upper() else v) for k, v in settings.items()}
    command = [sys.executable, "-u", str(runner)]
    if supervise:
        watcher = home / SUPERVISOR
        watcher.write_text(_SUPERVISOR.format(module=str(Path(__file__).resolve()), runner=str(runner),
                                              home=str(home), stall_minutes=float(stall_minutes),
                                              max_restarts=int(max_restarts),
                                              poll_seconds=float(poll_seconds)))
        command = [sys.executable, "-u", str(watcher)]
    with open(log, "a") as handle:
        handle.write(f"\n{'=' * 99}\n[launch] {stamp}  settings {shown or 'defaults'}"
                     f"{'  (supervised)' if supervise else ''}\n{'=' * 99}\n")
        process = subprocess.Popen(
            command, cwd=str(home), env=environment,
            stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,          # detached: its own session, not the notebook's child
        )
    (home / PID).write_text(json.dumps({"pid": process.pid, "started": stamp, "log": str(log),
                                        "supervised": bool(supervise),
                                        "settings": {k: repr(v) for k, v in shown.items()}}))
    print(f"[launch] training started in the background (pid {process.pid}).\n"
          f"         It keeps running if this notebook disconnects or restarts"
          f"{', and restarts itself if it crashes or stalls' if supervise else ''}.\n"
          f"         Log: {log}\n"
          f"         naturev1.follow() shows progress; naturev1.stop() ends it.")
    return process.pid


def supervise(runner: str, directory: str, stall_minutes: float = 30.0, max_restarts: int = 20,
              poll_seconds: float = 30.0, quick_failure_seconds: float = 600.0,
              max_quick_failures: int = 3) -> None:
    """
    Run the pipeline, and bring it back whenever it stops for any reason other than finishing.

    Every loop in the pipeline that makes progress -- a training step, a staged chunk, a storm, a
    scored batch -- touches a heartbeat file (:func:`naturev1.heartbeat.beat`). A run whose heartbeat is
    older than ``stall_minutes`` is stuck, whatever the cause: a cloud read that never returns, a
    deadlocked DataLoader worker, a GPU driver that has stopped answering. It is stopped and started
    again, and resumes from its last checkpoint -- at most a minute behind.

    A run that crashes is restarted too, after a pause that grows with each restart. But a run that
    fails within ``quick_failure_seconds`` of starting, ``max_quick_failures`` times in a row, is not
    unlucky: it is broken, and restarting it forever would only hide the error. Then the supervisor
    stops and says so, and the error is in the log above.
    """
    home = Path(directory)
    log, heartbeat, marker = home / LOG, home / HEARTBEAT, home / PID
    environment = dict(os.environ, NATUREV1_HEARTBEAT=str(heartbeat))
    stopping, child = [False], [None]

    def say(message: str) -> None:
        with open(log, "a") as handle:
            handle.write(f"[supervisor] {dt.datetime.now().isoformat(timespec='seconds')}  {message}\n")

    def on_stop(signum, frame):
        stopping[0] = True
        if child[0] is not None and child[0].poll() is None:
            try:
                os.kill(child[0].pid, signal.SIGTERM)       # the run checkpoints on its way out
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)

    def end_child(grace: float) -> None:
        process = child[0]
        try:
            os.kill(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.time() + grace
        while time.time() < deadline and process.poll() is None:
            time.sleep(0.5)
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        process.wait()

    restarts = quick = 0
    while True:
        heartbeat.write_text(f"{time.time():.0f} starting\n")
        started = time.time()
        with open(log, "a") as handle:
            child[0] = subprocess.Popen([sys.executable, "-u", runner], cwd=str(home), env=environment,
                                        stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        try:
            info = json.loads(marker.read_text())
            info["child"] = child[0].pid
            marker.write_text(json.dumps(info))
        except (OSError, json.JSONDecodeError):
            pass

        stalled, checked = False, time.time()
        while child[0].poll() is None and not stopping[0]:
            time.sleep(min(1.0, poll_seconds))              # wakes every second, so a stop is prompt
            if time.time() - checked < poll_seconds:
                continue
            checked = time.time()
            try:
                age = time.time() - heartbeat.stat().st_mtime
                last = heartbeat.read_text().strip().partition(" ")[2]
            except OSError:
                age, last = 0.0, ""
            if age > stall_minutes * 60 and child[0].poll() is None:
                stalled = True
                say(f"no progress for {age / 60:.0f} min (last: {last or 'nothing'}) -- stopping the run "
                    "to restart it from its last checkpoint")
                end_child(grace=90.0)
        if stopping[0]:
            end_child(grace=120.0)
            say("stopped on request")
            return
        code = child[0].wait()
        if code == 0 and not stalled:
            say("the pipeline finished")
            return

        ran = time.time() - started
        quick = quick + 1 if ran < quick_failure_seconds else 0
        restarts += 1
        reason = "stalled" if stalled else f"exited with code {code}"
        if quick >= max_quick_failures:
            say(f"the run {reason} within {quick_failure_seconds / 60:.0f} min of starting, {quick} times in "
                "a row: that is an error, not a stall or a dropped connection. Not restarting -- the "
                "cause is in the lines above.")
            return
        if restarts > max_restarts:
            say(f"{max_restarts} restarts used up; not restarting. The last error is above.")
            return
        pause = min(60 * restarts, 600) if poll_seconds >= 5 else poll_seconds
        say(f"the run {reason} after {ran / 60:.0f} min; restart {restarts} of {max_restarts} in "
            f"{pause:.0f} s, from its last checkpoint")
        deadline = time.time() + pause
        while time.time() < deadline and not stopping[0]:
            time.sleep(min(1.0, pause))
        if stopping[0]:
            say("stopped on request")
            return


def _tail(log: Path, size: int = 400_000) -> str:
    with open(log, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        end = handle.tell()
        handle.seek(max(end - size, 0))
        return handle.read().decode("utf-8", errors="replace")


def _only_banner(tail: str) -> bool:
    """True while the newest run has written nothing yet: its launch banner is all there is."""
    start = tail.rfind("[launch] ")
    return start >= 0 and tail[start:].count("\n") <= 2


def follow(lines: int = 40, directory: str | os.PathLike | None = None, wait: float = 0.0) -> None:
    """
    Print the last lines of the run's log, with whether it is still going.

    It prints and returns nothing, so as the last line of a notebook cell the log is shown once -- not a
    second time as the escaped string a returned value would be displayed as.

    Progress bars write ``\\r`` to redraw a line in place; only the latest state of each such line is
    shown, so a staging progress bar is one line here and not two thousand.

    Args:
        wait: for a run launched a moment ago, wait up to this many seconds for its first lines, so a
            run that fails at once shows its error here rather than on the next look. A run that has
            already written something is shown at once.
    """
    home = data_directory(directory)
    log = home / LOG
    if not log.exists():
        print(f"no log at {log} -- nothing has been launched from this directory")
        return
    if wait > 0 and _only_banner(_tail(log, 20_000)) and status(home):
        print(f"[follow] waiting up to {wait:.0f} s for the run's first lines ...")
        deadline = time.time() + wait
        while time.time() < deadline and status(home) and _only_banner(_tail(log, 20_000)):
            time.sleep(1.0)
        # It has started talking: let that first burst land, until the log has been still for 3 s.
        size, still = log.stat().st_size, 0
        while time.time() < deadline and still < 3 and status(home):
            time.sleep(1.0)
            grown = log.stat().st_size
            still, size = (still + 1 if grown == size else 0), grown
    tail = _tail(log)
    rows = [row.split("\r")[-1] for row in tail.split("\n")]
    shown = "\n".join(rows[-lines:])
    running = status(home)
    state = (f"[follow] RUNNING, pid {running['pid']}, started {running['started']}" if running
             else "[follow] NOT RUNNING -- finished, stopped, or crashed; the lines above say which")
    print(f"{shown}\n{state}")


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
    child = running.get("child")
    # The main process only: its DataLoader workers are shut down by it on the way out. Signalling the
    # whole group would kill the workers first and fail the step the trainer is trying to finish. A
    # supervisor passes the signal on to the run it is watching and does not restart it.
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    deadline = time.time() + wait
    while time.time() < deadline and (_alive(pid) or (child and _alive(int(child)))):
        time.sleep(1.0)
    for leftover in (child, pid):
        if leftover and _alive(int(leftover)):
            try:
                os.killpg(int(leftover), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    time.sleep(1.0)
    (home / PID).unlink(missing_ok=True)
    print(f"[stop] run {pid} ended; re-launching resumes from its last checkpoint")
    return True
