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
            # A read that fails or comes back empty while the process is still being set up, or just
            # as it leaves. The signal above reached it, so it is alive; a dead one is caught next time.
            return True
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
    progress(home)                        # what this run continues from

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


#: While the run is downloading, a chunk lands every few seconds; nothing for this long is a hung
#: connection, not a slow step, and waiting the full training stall limit would waste half an hour.
STAGING_STALL_MINUTES = 12.0


def _stall_limit(note: str, stall_minutes: float) -> float:
    """How long the heartbeat may be silent before the run is called stalled, given what it was doing."""
    if note.startswith("staging"):
        return min(stall_minutes, STAGING_STALL_MINUTES)
    return stall_minutes


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

    A run that crashes is restarted too, after a pause that grows with each restart; ``max_restarts``
    counts restarts in a row, and an hour of healthy running starts the count again. But a run that
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
            if age > _stall_limit(last, stall_minutes) * 60 and child[0].poll() is None:
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
        if ran >= 3600:
            # An hour of good progress since the last trouble: a fresh allowance. Counting every
            # restart since launch would end a week-long run over a few unrelated blips days apart.
            restarts = 0
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


STAGES = (("stage1", "stage one, pretraining"), ("rollout", "stage 1b, rollout"),
          ("stage2", "stage two, storm heads"))


def backup_directory() -> Path:
    """Where the second copy of every checkpoint goes: ``NATUREV1_BACKUP_DIR``, else ``~/naturev1_backup``."""
    return Path(os.environ.get("NATUREV1_BACKUP_DIR") or Path.home() / "naturev1_backup")


def progress(directory: str | os.PathLike | None = None) -> None:
    """
    Print where a run would continue from: each stage's newest checkpoint, its age, and its copies.

    Nothing is changed. Running the cell again, or :func:`launch`, continues from exactly these.
    """
    home = data_directory(directory)
    roots = [home / "naturev1_ckpt", Path("/content/drive/MyDrive/naturev1_ckpt")]
    root = next((r for r in roots if r.is_dir()), roots[0])
    backup = backup_directory()
    from .safeio import read_json

    print(f"[progress] checkpoints in {root}; backups in {backup}")
    for tag, label in STAGES:
        latest = root / tag / "latest.pt"
        copy = backup / tag / "latest.pt"
        snapshots = sorted((root / tag).glob("step_*.pt")) if (root / tag).is_dir() else []
        if not latest.exists() and not copy.exists():
            print(f"  {label:26} not started")
            continue
        state = (read_json(root / tag / "state.json", default=None)
                 or read_json(backup / tag / "state.json", default=None) or {})
        newest = max((p for p in (latest, copy) if p.exists()), key=lambda p: p.stat().st_mtime)
        age = (time.time() - newest.stat().st_mtime) / 60
        when = f"{age:.0f} min ago" if age < 120 else f"{age / 60:.1f} h ago"
        step = f"step {state['step']:,}, epoch {state.get('epoch', 0)}" if "step" in state else "saved"
        print(f"  {label:26} {step}, {when}; {len(snapshots)} snapshot(s)"
              f"{', backup copy' if copy.exists() else ', NO backup copy yet'}")
    bank = [p.name for p in (root.glob("storm_bank_*.pt") if root.is_dir() else [])]
    if bank:
        print(f"  storm feature banks        {', '.join(sorted(bank))}")
    for name in ("era5.npy", "era5_val.npy"):
        if (home / name).exists():
            staged = read_json(home / name.replace(".npy", ".json"), default={}) or {}
            done = "complete" if staged.get("indices") and not (home / name.replace(".npy", ".progress.json")).exists() \
                else "partly downloaded -- resumes where it stopped"
            print(f"  {name:26} {(home / name).stat().st_size / 1e9:.1f} GB, {done}")


#: In a live watch, how often a line is printed when the run itself has printed nothing -- proof of
#: life, with what the run is doing, so a quiet stretch never looks like a dead one.
ALIVE_EVERY = 60.0
#: And how often the fuller status (every stage, the session clock, the GPU) is printed.
STATUS_EVERY = 600.0


def follow(lines: int = 40, directory: str | os.PathLike | None = None, wait: float = 0.0,
           watch: float | None = 0.0) -> None:
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
        watch: after the snapshot, keep showing the log live for this many seconds -- new lines as
            they arrive, a progress bar's latest state every ten seconds -- then stop, with a line
            saying the run goes on. ``None`` watches until the run ends, printing a proof-of-life
            line every minute when the run is quiet: the cell stays visibly active, which also keeps a
            hosted notebook from deciding it is idle. Stopping the cell stops only the watching.
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
    if running and (watch is None or watch > 0):
        _watch(log, home, watch)


def _gpu_line() -> str:
    """Utilisation and memory of the first GPU, from nvidia-smi; empty when there is none."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        used, total = None, None
        for row in out.stdout.strip().splitlines()[:1]:
            util, used, total = (part.strip() for part in row.split(","))
            return f"GPU {util}% busy, {float(used) / 1024:.0f}/{float(total) / 1024:.0f} GB"
    except Exception:
        return ""                                   # no GPU, or no nvidia-smi: nothing to report
    return ""


def _session_line() -> str:
    """How long this machine's session has been up -- what a hosted notebook's timer counts."""
    from .mirror import session_started

    began = session_started()
    if began is None:
        return ""
    hours = (time.time() - began) / 3600
    return f"session up {hours:.1f} h"


def _stage_lines(home: Path) -> list[str]:
    from .safeio import read_json

    roots = [home / "naturev1_ckpt", Path("/content/drive/MyDrive/naturev1_ckpt")]
    root = next((r for r in roots if r.is_dir()), roots[0])
    rows = []
    for tag, label in STAGES:
        state = read_json(root / tag / "state.json", default=None)
        if state and "step" in state:
            age = (time.time() - (root / tag / "state.json").stat().st_mtime) / 60
            rows.append(f"{label}: step {state['step']:,} (saved {age:.0f} min ago)")
    return rows


def _heartbeat_line(home: Path) -> str:
    try:
        text = (home / HEARTBEAT).read_text().strip()
        age = time.time() - (home / HEARTBEAT).stat().st_mtime
        return f"doing: {text.partition(' ')[2] or 'starting'} ({age:.0f} s ago)"
    except OSError:
        return ""


def _watch(log: Path, home: Path, seconds: float | None) -> None:
    """
    Tail the log live: whole lines as they land, a progress bar at most every 10 s, and when the run
    has printed nothing for a minute, one line saying it is alive and what it is doing.
    """
    if seconds is None:
        print("[follow] watching until the run finishes. This cell stays active (so the notebook is never "
              "idle); stopping the cell stops only the watching, never the run.", flush=True)
    else:
        print(f"[follow] watching live for {seconds:.0f} s (the run does not depend on this cell) ...", flush=True)
    position = log.stat().st_size
    partial, last_bar = "", 0.0
    now = time.time()
    deadline = None if seconds is None else now + seconds
    last_output, last_status, pushes = now, now, 0
    try:
        while deadline is None or time.time() < deadline:
            time.sleep(2.0)
            size = log.stat().st_size
            if size < position:
                position = 0                                                # the log was truncated
            if size > position:
                with open(log, "rb") as handle:
                    handle.seek(position)
                    partial += handle.read(size - position).decode("utf-8", errors="replace")
                position = size
                *whole, partial = partial.split("\n")
                for row in whole:
                    shown = row.split("\r")[-1]
                    if shown.strip():
                        print(shown, flush=True)
                        last_output = time.time()
                        pushes += "pushed" in shown and "[checkpoint]" in shown
            bar = partial.split("\r")[-1]
            if bar.strip() and time.time() - last_bar >= 10.0:
                print(bar, flush=True)
                last_bar = last_output = time.time()
            if not status(home):
                print("[follow] the run has ended; the lines above say how", flush=True)
                return
            quiet = time.time() - last_output
            if quiet >= ALIVE_EVERY:
                parts = [p for p in (_heartbeat_line(home), *_stage_lines(home), _gpu_line(), _session_line()) if p]
                print(f"[alive] {time.strftime('%H:%M:%S')}  " + " | ".join(parts), flush=True)
                last_output = time.time()
            if time.time() - last_status >= STATUS_EVERY:
                print(f"[status] {time.strftime('%H:%M:%S')}  Hub pushes seen this watch: {pushes}; "
                      f"log {log.stat().st_size / 1e6:.1f} MB; naturev1.stop() ends the run, nothing else does",
                      flush=True)
                last_status = time.time()
    except KeyboardInterrupt:
        print("[follow] stopped watching; the run continues (naturev1.stop() is the only thing that ends it)",
              flush=True)
        return
    print("[follow] still running; run this cell or naturev1.follow() again any time", flush=True)


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
