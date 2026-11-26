# Pinxit-1: one cell, plain Python. Runs in molab, Jupyter, or as "python pinxit_molab_cell.py" on a rented GPU.
# Paste your Hugging Face token below and run it. It stops any earlier run by itself, installs helix-lm 2.6.0,
# pulls the latest checkpoint from the private repo CorxLabs/Pinxit-1 and goes on with the hours that are left.
# The GPU starts on practice sheets at once; the paintings and photos are fetched beside it. A molab session ends
# after 12 hours: the run saves, pushes and stops 30 minutes before that, so run this cell again in the next
# session and it carries on. Stopping this cell stops the run too.
# Your own pictures are learned from too: put photos in <ROOT>/uploads/photos and paintings in
# <ROOT>/uploads/paintings before you run the cell (they are met twenty times as often as the rest).
# ruff: noqa: E402  (the settings come first, so they are the first thing seen in the notebook)

HF_TOKEN = "PASTE_YOUR_ORG_TOKEN_HERE"      # a Hugging Face write token for the CorxLabs organisation
HF_REPO = "CorxLabs/Pinxit-1"               # private: checkpoints, markers, the gallery with its scores
GPU_HOURS = 20                              # the GPU hours you have in total for this model
SIZE = "px-base"                            # 45M parameters; "px-small" (11M) is faster, "px-large" is bigger
ROOT = "auto"                               # "auto": the folder that already holds this run, else /workspace/pinxit
HELIX = "2.6.0"

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

if not HF_TOKEN.startswith("PASTE"):
    os.environ["HF_TOKEN"] = HF_TOKEN       # for the program below only; never printed or written to a file
if not os.environ.get("HF_TOKEN"):
    raise SystemExit("Paste a Hugging Face write token into HF_TOKEN first: the run resumes from the private repo.")
PROGRAM = "helix_lm.scripts.pinxit_molab"


def run(args: list) -> int:
    """Run a command and show its output as it comes. Stopping the cell stops the command."""
    print("$ " + " ".join(str(a) for a in args[:8]) + (" ..." if len(args) > 8 else ""), flush=True)
    proc = subprocess.Popen([str(a) for a in args], env=dict(os.environ, PYTHONUNBUFFERED="1"),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
    finally:
        if proc.poll() is None:             # the cell was stopped: let the program save and push first
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                proc.kill()
    return proc.wait()


def installed() -> str:
    out = subprocess.run([sys.executable, "-c", "import helix_lm; print(helix_lm.__version__)"],
                         capture_output=True, text=True)
    return out.stdout.strip()


# 1. any earlier run of the program in this machine is stopped (a cell run twice would start a second one)
def earlier_runs() -> list:
    pids = []
    for proc in Path("/proc").iterdir():
        if proc.name.isdigit() and int(proc.name) != os.getpid():
            try:
                if PROGRAM.encode() in (proc / "cmdline").read_bytes():
                    pids.append(int(proc.name))
            except OSError:
                pass
    return pids


for pid in earlier_runs():
    print(f"stopping the earlier run (pid {pid}); it saves and pushes first", flush=True)
    with contextlib.suppress(ProcessLookupError):           # it ended by itself in the meantime
        os.kill(pid, signal.SIGINT)
for _ in range(120):
    if not earlier_runs():
        break
    time.sleep(5)
for pid in earlier_runs():
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)

# 2. helix-lm at this version: from PyPI, or from the source tree when PyPI does not have it
pypi = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", f"helix-lm[paint]=={HELIX}"]
if installed() != HELIX and (run(pypi) != 0 or installed() != HELIX):
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "helix-lm[paint] @ git+https://github.com/"
         "Wisk-inc/transformers.git@claude/linear-context-architecture-a35c6h#subdirectory=helix-lm"])
if installed() != HELIX:
    raise SystemExit(f"helix-lm {HELIX} could not be installed (found {installed() or 'none'}); the error is above.")
print(f"helix-lm {installed()}", flush=True)


# 3. the working folder: the one that already holds this run, so nothing is fetched twice
def progress(folder: Path) -> int:
    marks = folder / "markers"
    return (len(list(marks.iterdir())) if marks.is_dir() else 0) * 1000 + (folder / "uploads").is_dir()


if ROOT == "auto":
    places = [Path("/workspace/pinxit"), Path("/home/marimo/pinxit"), Path.home() / "pinxit"]
    best = max(places, key=progress)
    root = best if progress(best) else (places[0] if os.access("/workspace", os.W_OK) else places[2])
else:
    root = Path(os.path.expanduser(ROOT))
root.mkdir(parents=True, exist_ok=True)
(root / "stop").unlink(missing_ok=True)     # a stop file left by an earlier run would end this one at once
print(f"working folder {root}", flush=True)


# 4. the hours: two fifths on practice sheets, three fifths on real pictures; a stage resumes with what is left
def plan(gpu_hours: float) -> list:
    left = max(1.0, gpu_hours - 0.5)        # half an hour is kept for the gallery and the pushes
    hours = {"--sheet-hours": f"{0.4 * left:.1f}", "--picture-hours": f"{0.6 * left:.1f}",
             "--push-every-hours": "1"}
    return [x for kv in hours.items() for x in kv]


code = run([sys.executable, "-m", PROGRAM, "--root", root, "--stage", "all", "--hf-repo", HF_REPO, "--size", SIZE]
           + plan(GPU_HOURS))
print(f"\nthe program ended with code {code}.", flush=True)
print(f"checkpoints, the gallery and its scores: https://huggingface.co/{HF_REPO}/tree/main", flush=True)
print("If it stopped for the session's end, run this cell again in a new session: it goes on from there.")
