# TriStream-V2-SVS: one cell, plain Python. Runs in molab, Jupyter, or as "python tristream_molab_cell.py" on
# a rented GPU. Paste your Hugging Face token below and run it. It stops any earlier run by itself, installs
# helix-lm 2.4.0, pulls the latest checkpoints from the private repo CorxLabs/Tristream-V2-SVS and goes on with
# the hours that are left. A molab session ends after 12 hours: the run saves, pushes and stops 30 minutes before
# that, so run this cell again in the next session and it carries on. Stopping this cell stops the run too.
# ruff: noqa: E402  (the settings come first, so they are the first thing seen in the notebook)

HF_TOKEN = "PASTE_YOUR_ORG_TOKEN_HERE"      # a Hugging Face write token for the CorxLabs organisation
HF_REPO = "CorxLabs/Tristream-V2-SVS"       # private: checkpoints, markers, benchmark results
GPU_HOURS = 32                              # the GPU hours you have in total for this model; 0 = the full budgets
SIZE = "ts-765m"                            # "ts-300m" for a smaller, faster model
ROOT = "auto"                               # "auto": the folder that already holds this run, else /workspace/tristream
HELIX = "2.4.0"

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
PROGRAM = "helix_lm.scripts.tristream_molab"


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
    os.kill(pid, signal.SIGINT)
for _ in range(120):
    if not earlier_runs():
        break
    time.sleep(5)
for pid in earlier_runs():
    os.kill(pid, signal.SIGKILL)

# 2. helix-lm at this version: from PyPI, or from the source tree when PyPI does not have it
pypi = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", f"helix-lm[music]=={HELIX}"]
if installed() != HELIX and (run(pypi) != 0 or installed() != HELIX):
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "helix-lm[music] @ git+https://github.com/"
         "Wisk-inc/transformers.git@claude/linear-context-architecture-a35c6h#subdirectory=helix-lm"])
if installed() != HELIX:
    raise SystemExit(f"helix-lm {HELIX} could not be installed (found {installed() or 'none'}); the error is above.")
print(f"helix-lm {installed()}", flush=True)


# 3. the working folder: the one that already holds this run, so nothing is downloaded or prepared twice
def progress(folder: Path) -> int:
    marks = folder / "markers"
    return (len(list(marks.iterdir())) if marks.is_dir() else 0) * 1000 + (folder / "raw").is_dir()


if ROOT == "auto":
    places = [Path("/workspace/tristream"), Path("/home/marimo/tristream"), Path.home() / "tristream"]
    best = max(places, key=progress)
    root = best if progress(best) else (places[0] if os.access("/workspace", os.W_OK) else places[2])
else:
    root = Path(os.path.expanduser(ROOT))
root.mkdir(parents=True, exist_ok=True)
(root / "stop").unlink(missing_ok=True)     # a stop file left by an earlier run would end this one at once
print(f"working folder {root}", flush=True)


# 4. the hours: split between the codec, the refiner and the singing model; a stage resumes with what is left
def plan(gpu_hours: float) -> list:
    if gpu_hours <= 0:
        hours = {"--emilia-hours": "300", "--codec-hours": "240", "--refiner-hours": "48", "--train-hours": "200",
                 "--push-every-hours": "2"}
    else:
        left = max(1.0, gpu_hours - (1.5 if gpu_hours < 60 else 2.5) - 1.0)
        hours = {"--emilia-hours": "100" if gpu_hours < 24 else ("150" if gpu_hours < 60 else "300"),
                 "--codec-hours": f"{0.47 * left:.1f}", "--refiner-hours": f"{0.10 * left:.1f}",
                 "--train-hours": f"{0.43 * left:.1f}", "--push-every-hours": "1" if gpu_hours < 48 else "2"}
    return [x for kv in hours.items() for x in kv]


code = run([sys.executable, "-m", PROGRAM, "--root", root, "--stage", "all", "--hf-repo", HF_REPO, "--size", SIZE]
           + plan(GPU_HOURS))
print(f"\nthe program ended with code {code}.", flush=True)
print(f"checkpoints and benchmark results: https://huggingface.co/{HF_REPO}/tree/main", flush=True)
print("If it stopped for the session's end, run this cell again in a new session: it goes on from there.")
