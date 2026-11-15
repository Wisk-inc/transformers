# Pluvium, everything in one cell: the newest package, the Hugging Face repository (code, papers, a collection, the
# upstream AI models' weights), the hourly network on a GPU, your country's 14-day rain forecast and hurricane tracks.
# Paste into one notebook cell (Colab, Kaggle, Jupyter, a rented GPU box) or run as `python pluvium_everything.py`.
#
# With nothing changed it needs no account and no GPU: it sets the country up, gets a trained blend (downloads one
# if the repository has it, otherwise trains one in about an hour and a half) and prints the forecast.
# With a token pasted in, it also builds the repository. On a GPU of 24 GB or more it also trains the network.
# Every step can be run again: what is already downloaded, uploaded or trained is not done twice.

COUNTRY = "JM"                              # ISO codes: "JM", "TT", "DO,HT", "PH", "KE" ...
ROOT = "~/pluvium-JM"                       # the country's working folder; 41 MB for Jamaica, a few GB for a large country
HANDOVER = "auto"                           # record storms: "auto" follows the storms staged in training; "always"; "never"
LEARN = False                               # True: first learn from the rain NOAA has published since (see step 5)
HURRICANES = True                           # a bulletin for every tropical cyclone alive now
BASINS = "al,ep"                            # al: North Atlantic, ep: East Pacific

HF_TOKEN = "PASTE_YOUR_ORG_TOKEN_HERE"      # a Hugging Face write token; leave as it is to skip everything on the Hub
HF_REPO = "CorxLabs/Pluvium-500"            # the model repository to create or update
PRIVATE = False                             # True to keep the repository and the collection private
MIRROR_UPSTREAM = True                      # copy GraphCast's, NOAA's and ECMWF's weights into the repository (4.9 GB)

TRAIN_NETWORK = "auto"                      # the hourly network: "auto" trains it when a GPU of 24 GB or more is here; "yes"; "no"
GPU_HOURS = 24                              # wall-clock budget for the network
NETWORK_ROOT = "/workspace/pluvium"         # its working folder; about 150 GB for six years of data
SIZE = "pluvium-500"                        # or "pluvium-base" for a smaller, faster network
VRAM = "fill"                               # "fill": every training step is sized to the GPU's memory; "fastest": the old rule
VRAM_SHARE = 0.95                           # how much of the GPU's memory a step may need; the rest is the safety margin

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

if not HF_TOKEN.startswith("PASTE"):
    os.environ["HF_TOKEN"] = HF_TOKEN       # for the commands below only; it is never printed or written to a file
TOKEN = os.environ.get("HF_TOKEN", "")
PLUVIUM = [sys.executable, "-m", "pluvium.cli"]
root = Path(os.path.expanduser(ROOT))
steps = []                                  # (what, outcome), printed at the end


def run(args: list, env: dict | None = None) -> None:
    """Run a command and show its output line by line, so a notebook shows progress. Stopping the cell stops it."""
    print("$ " + " ".join(str(a) for a in args), flush=True)
    proc = subprocess.Popen([str(a) for a in args], env=dict(os.environ, PYTHONUNBUFFERED="1", **(env or {})),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
    finally:
        if proc.poll() is None:
            proc.terminate()
    if proc.wait() != 0:
        raise RuntimeError(f"`{' '.join(str(a) for a in args[:5])} ...` ended with exit code {proc.returncode}")


def took(started: float) -> str:
    seconds = time.time() - started
    return f"{seconds:.0f} s" if seconds < 100 else f"{seconds / 60:.0f} min"


def step(what: str, action, skip: str = "") -> bool:
    """Do one step. A step that fails is reported and the cell goes on: the steps do not depend on each other."""
    print(f"\n=== {what} " + "=" * max(4, 96 - len(what)), flush=True)
    if skip:
        print(f"skipped: {skip}", flush=True)
        steps.append((what, f"skipped ({skip})"))
        return False
    started = time.time()
    try:
        action()
    except BaseException as error:          # also a stopped cell: say so, then let the stop go through
        steps.append((what, f"FAILED after {took(started)}: {type(error).__name__}: {error}"))
        print(f"FAILED: {type(error).__name__}: {error}", flush=True)
        if isinstance(error, KeyboardInterrupt):
            raise
        return False
    steps.append((what, f"done in {took(started)}"))
    return True


# 1. the package: from PyPI, or from the source tree when PyPI cannot be reached
def install() -> None:
    try:
        run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pluvium>=0.5.0"])
    except RuntimeError:
        run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/Wisk-inc/transformers.git"
             "@claude/linear-context-architecture-a35c6h#subdirectory=pluvium"])
    run(PLUVIUM + ["upstream", "list"])     # also proves the command line works


def version() -> str:
    out = subprocess.run([sys.executable, "-c", "import pluvium; print(pluvium.__version__)"], capture_output=True, text=True)
    return out.stdout.strip()


if not step("1. Pluvium, the newest version", install):
    raise SystemExit("Pluvium could not be installed, so nothing else can run. The error is above.")
print(f"pluvium {version()}")


# 2. the repository: the model card, all the code, the papers, the benchmarks, and a collection in the organisation
def source_tree() -> Path:
    """The full Pluvium project (code, tests, docs, papers, card) for the installed version: the source archive
    from PyPI, or the source tree on GitHub when this version is not on PyPI."""
    work = Path(tempfile.mkdtemp(prefix="pluvium-src-"))
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/pluvium/{version()}/json", timeout=60) as response:
            url = next(f["url"] for f in json.load(response)["urls"] if f["packagetype"] == "sdist")
        urllib.request.urlretrieve(url, work / "pluvium.tar.gz")
        with tarfile.open(work / "pluvium.tar.gz") as archive:
            try:
                archive.extractall(work, filter="data")
            except TypeError:               # Pythons from before the filter existed
                archive.extractall(work)
        return next(p for p in work.iterdir() if p.is_dir())
    except OSError as error:
        print(f"no source archive on PyPI for this version ({error}); cloning the source tree", flush=True)
        run(["git", "clone", "--depth", "1", "--branch", "claude/linear-context-architecture-a35c6h",
             "https://github.com/Wisk-inc/transformers.git", work / "clone"])
        return work / "clone" / "pluvium"


def upload_code() -> None:
    source = source_tree()
    spec = importlib.util.spec_from_file_location("pluvium_hub_upload", source / "hub" / "upload_to_hub.py")
    hub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hub)
    hub.HF_REPO, hub.PRIVATE, hub.SOURCE = HF_REPO, PRIVATE, str(source)
    hub.main()                              # reads the token from the environment


NO_TOKEN = "" if TOKEN else "no token pasted into HF_TOKEN"
step(f"2. {HF_REPO}: code, model card, papers, benchmarks, collection", upload_code, NO_TOKEN)

# 3. the weights of the AI models whose forecasts Pluvium reads, copied unchanged with their licences
step("3. Upstream weights: GraphCast (DeepMind), GraphCastGFS and AIGFS (NOAA), AIFS (ECMWF)",
     lambda: run(PLUVIUM + ["upstream", "mirror", "--repo", HF_REPO] + (["--private"] if PRIVATE else [])),
     NO_TOKEN or ("" if MIRROR_UPSTREAM else "MIRROR_UPSTREAM is False"))


# 4. the hourly network, on a GPU. It goes before the rest because GPU time is the part with a clock on it.
def gpu_gb() -> float:
    """Memory of the largest GPU here, in GB; 0 when there is none."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60).stdout
        return max(float(x) for x in out.split()) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def network() -> None:
    probe = ("import torch; assert torch.cuda.is_available(), 'no GPU visible';"
             "x = torch.ones(8, device='cuda'); print(torch.cuda.get_device_name(0), torch.__version__, float(x @ x))")
    if subprocess.run([sys.executable, "-c", probe]).returncode != 0:
        print("PyTorch here has no kernels for this GPU; installing the CUDA 12.8 build", flush=True)
        run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "torch", "--index-url",
             "https://download.pytorch.org/whl/cu128"])
        subprocess.run([sys.executable, "-c", probe], check=True)
    if not TOKEN:
        print("No Hub token: nothing is mirrored, so a stopped session cannot resume and nothing is published.")
    home = Path(os.path.expanduser(NETWORK_ROOT))
    run(PLUVIUM + ["program"],
        env={"GPU_HOURS": str(GPU_HOURS), "HF_REPO": HF_REPO, "PLUVIUM_ROOT": NETWORK_ROOT, "PLUVIUM_SIZE": SIZE,
             "PLUVIUM_VRAM": VRAM, "PLUVIUM_VRAM_SHARE": str(VRAM_SHARE),
             "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false"})
    if not (home / "program.json").exists():        # the program records each stage it finishes there
        raise RuntimeError("the training program ended without doing anything")
    report = home / "forecast" / "report.txt"
    print(f"the network's own forecast is in {report}" if report.exists() else
          f"the network is trained; no forecast was written (the lines above say why). Its files are in {home}")


gpu = gpu_gb()
resuming = (Path(os.path.expanduser(NETWORK_ROOT)) / "program.json").exists()
place = Path(os.path.expanduser(NETWORK_ROOT))
while not place.exists():
    place = place.parent
free_gb = shutil.disk_usage(place).free / 1e9
why_not = ("TRAIN_NETWORK is 'no'" if TRAIN_NETWORK == "no" else
           "no GPU here; the rain forecast and the hurricane tracks need none" if gpu == 0 else
           "" if TRAIN_NETWORK == "yes" else
           f"the GPU has {gpu:.0f} GB and the network wants 24 or more (TRAIN_NETWORK = 'yes' with SIZE = "
           "'pluvium-base' trains a smaller one)" if gpu < 22 else
           f"{free_gb:.0f} GB free at {place} and the data needs about 150 (point NETWORK_ROOT at a bigger disk)"
           if free_gb < 150 and not resuming else "")
step(f"4. The Pluvium-500 network on the GPU ({GPU_HOURS} hours)", network, why_not)

# 5. rain for the country, 14 days: sets the country up, downloads a trained blend or trains one, reads the newest runs.
#    LEARN = True first updates the blend's weights from rain published since it last learned. NOAA publishes the rain
#    record about five months late and a month at a time, so that is worth doing about once a month. Each run learns
#    from two weeks of rain and takes about ten minutes, nearly all of it reading the two big ensembles.
step(f"5. Rain for {COUNTRY}, 14 days",
     lambda: run(PLUVIUM + ["now", "--country", COUNTRY, "--root", root, "--repo", HF_REPO]
                 + (["--handover", HANDOVER] if HANDOVER != "auto" else []) + (["--learn"] if LEARN else [])))

# 6. share the blend trained here, so that the next person's step 5 is a download
trained_here = (root / "blend" / "blend.json").exists() and not (root / "blend" / "FETCHED").exists()
step(f"6. Publish the trained blend for {COUNTRY}",
     lambda: run(PLUVIUM + ["blend", "publish", "--root", root, "--repo", HF_REPO] + (["--private"] if PRIVATE else [])),
     NO_TOKEN or ("" if trained_here else "the blend here was downloaded, not trained here"))

# 7. hurricanes: track, cone and the chance of passing near each place, for every storm alive now
step("7. Hurricanes alive now", lambda: run(PLUVIUM + ["cyclone", "--root", root, "--basins", BASINS]),
     "" if HURRICANES else "HURRICANES is False")

print("\n" + "=" * 100)
for what, outcome in steps:
    print(f"{what}: {outcome}")
tables = sorted(root.glob("blend_*.csv"))
print(f"\nforecast table: {tables[-1] if tables else 'none written'}")
print("its columns: the place and its parish; the chance of 1, 10, 25 and 50 mm; median, 90th percentile and expected "
      "rain; how many raw systems call for 50 mm; the alert tier; whether the forecast is beyond the training record; "
      "how rare the amount is")
print(f"by parish: pluvium now --country {COUNTRY} --root {root} --parishes")
print(f"when in the day, first three days: pluvium now --country {COUNTRY} --root {root} --when")
if HURRICANES:
    print(f"hurricane bulletins and data: {root / 'cyclone'}")
if TOKEN:
    print(f"repository: https://huggingface.co/{HF_REPO}")
print(f"every day after this: pluvium now --country {COUNTRY} --root {root}      (about a minute)")
failed = [what for what, outcome in steps if outcome.startswith("FAILED")]
if failed:
    print(f"\n{len(failed)} step(s) failed and are listed above. Running the cell again repeats only what is missing.")
