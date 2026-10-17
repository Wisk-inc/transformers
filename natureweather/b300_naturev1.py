# ═══════════════════════════════════════════════════════════════════════════════════════════════════
#  NatureV1 0.12 on 8x NVIDIA B300 -- the satellite path, then every basin.
#
#  Cells for JupyterLab or VS Code ("# %%"), or run top to bottom as a script. Every long step runs as
#  its own detached process with a log under DATA/logs: closing the notebook or losing the connection
#  never stops it, and running the same cell again reattaches to it instead of starting it twice.
#
#  What happens, and roughly how long it takes on 8x B300 with a fast NVMe array and a fast network:
#
#    1  install, preflight                     minutes   checks GPUs, disks, network, HF token; measures a step
#    2  base model + statistics from the Hub   minutes   the trained 0.10/0.11 release is the starting point
#    3  labels                                 minutes   IBTrACS since 1980, every basin: ~230,000 storm fixes
#    4  stage the record (two jobs at once)    1-4 h     ERA5 1980-2021 (~0.33 TB) and GridSat-B1 1980-2021
#                                                        (~4-6 TB read, ~0.8 TB of training shards written)
#    5  stage A, 3 h: both satellite networks  3 h       GPUs 0-3: the storm network; GPUs 4-7: the retrieval
#    6  stage B: heads + backbone tail         2 h + ~1  every basin, satellites in the loop; then banks with
#                                                        rollout members, consensus, calibration, test scores
#    7  publish                                minutes   only replaces the release if it beat the old one
#
#  Every checkpoint of every run is pushed to HF_REPO as it is written (checkpoints/<run>/latest.pt, and
#  the weights of each one under checkpoints/<run>/history/), so a dead machine resumes on a new one.
#
#  The training time is the GPU budget you set (STAGE_A_HOURS, STAGE_B_HOURS); staging is separate and
#  bound by network and disk, not by the GPUs. Nothing here has been run on a B300 by its author -- the
#  code is tested end to end on CPU (including real two-process data parallelism); the preflight is the
#  first thing that touches your hardware, and it says so if anything is wrong.
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

# %% [0] settings ──────────────────────────────────────────────────────────────────────────────────────
HF_REPO        = "CorxLabs/NatureV1-500"
HF_TOKEN       = ""            # paste a WRITE token here for this session, then clear it afterwards
DATA           = "/data/naturev1"   # the NVMe array: >= 2 TB free (staged shards + ERA5 + working space)
FIRST_YEAR     = 1980          # GridSat-B1 begins 1980
LAST_YEAR      = 2021          # WeatherBench 2's 1.5-degree ERA5 ends 2021-12-31
SAT_RESOLUTION = 0.25          # degrees of the staged global frames (0.1 writes ~4 TB of shards instead of ~0.8)
STAGE_A_HOURS  = 3.0           # both satellite networks, side by side, four GPUs each
STAGE_B_HOURS  = 2.0           # storm heads + backbone tail on all eight GPUs (banks and scoring come after)
TAIL_BLOCKS    = 4             # backbone blocks fine-tuned in stage B (of 17)
ROTATIONS      = 4             # test-time rotation ensemble kept in the banks (1: off); used only if it helps
STAGING_WORKERS = None         # None: all CPUs but 16
KEEP_RAW       = False         # keep the raw GridSat files (4-6 TB) after staging
BRANCH         = "claude/linear-context-architecture-a35c6h"

# %% [1] install and preflight ──────────────────────────────────────────────────────────────────────────
import os, subprocess, sys
from pathlib import Path

REPO_URL = "git+https://github.com/Wisk-inc/transformers@" + BRANCH
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                f"ihelix @ {REPO_URL}#subdirectory=ihelix",
                f"naturev1[all] @ {REPO_URL}#subdirectory=natureweather",
                "safetensors", "netCDF4", "h5netcdf"], check=True)
if HF_TOKEN.strip():
    os.environ["HF_TOKEN"] = HF_TOKEN.strip()       # inherited by every job below; never written to disk
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
DATA = Path(DATA)
LOGS, SAT, CKPT = DATA / "logs", DATA / "satellite", DATA / "runs"
for folder in (LOGS, SAT, CKPT):
    folder.mkdir(parents=True, exist_ok=True)

import naturev1
from naturev1.hardware import hardware_preflight

print(f"naturev1 {naturev1.__version__}")
READY = hardware_preflight(DATA, HF_REPO, need_tb=1.6 if SAT_RESOLUTION >= 0.25 else 5.0, hours=STAGE_A_HOURS)
assert READY, "fix the lines marked !! above before spending GPU hours"

# %% [2] the starting model and its statistics, from the Hub ───────────────────────────────────────────
import shutil
import torch
from huggingface_hub import hf_hub_download

BASE = DATA / "base.pt"
if not BASE.exists():
    for tag in ("stage3", "stage2"):                 # the newest released stage
        try:
            shutil.copy2(hf_hub_download(HF_REPO, f"checkpoints/{tag}/latest.pt"), BASE)
            print(f"starting from checkpoints/{tag}/latest.pt")
            break
        except Exception as error:
            print(f"no {tag} on the Hub ({type(error).__name__})")
STATS = DATA / "stats.json"
if not STATS.exists():
    shutil.copy2(hf_hub_download(HF_REPO, "staging/stats.json"), STATS)
base = torch.load(BASE, map_location="cpu", weights_only=False)
print("base config:", {k: base["config"][k] for k in ("hidden_size", "num_layers", "analysis_channels", "state_channels")})
ANALYSIS_CHANNELS = base["config"]["analysis_channels"]
TENDENCY = DATA / "tendency.pt"
recorded = base["model"].get("state_tendency")
if recorded is not None and float(recorded.abs().sum()) > 0:
    torch.save(recorded.float(), TENDENCY)
    print("tendency scale: recorded in the checkpoint")
del base

# %% [3] labels: every basin since 1980 ────────────────────────────────────────────────────────────────
import numpy as np
from naturev1.ibtracs import describe_table, fetch_ibtracs, parse_ibtracs, storm_label_table

IBTRACS = fetch_ibtracs(DATA)
LABELS = SAT / "labels.npz"
if not LABELS.exists():
    np.savez(LABELS, **storm_label_table(parse_ibtracs(IBTRACS, since=FIRST_YEAR, until=LAST_YEAR)))
print(describe_table(dict(np.load(LABELS))))

# %% [4] stage the record: ERA5 and GridSat, as two detached jobs ──────────────────────────────────────
from naturev1.b300 import follow, start_job
from naturev1.satstage import staging_plan

print(staging_plan(range(FIRST_YEAR, LAST_YEAR + 1), SAT_RESOLUTION, len(np.load(LABELS)["time"])))
ERA5 = DATA / f"era5_{FIRST_YEAR}_{LAST_YEAR}.npy"
start_job("era5", [sys.executable, "-c",
                   f"from naturev1.b300 import stage_era5_record; "
                   f"stage_era5_record({str(DATA)!r}, {FIRST_YEAR}, {LAST_YEAR}, {str(STATS)!r}, workers=32)"], LOGS)
workers = STAGING_WORKERS or max((os.cpu_count() or 32) - 16, 4)
start_job("gridsat", [sys.executable, "-m", "naturev1.satstage", "--root", str(SAT), "--ibtracs", str(IBTRACS),
                      "--years", f"{FIRST_YEAR}-{LAST_YEAR}", "--workers", str(workers),
                      "--resolution", str(SAT_RESOLUTION)] + (["--keep-raw", "--raw-dir", str(DATA / "gridsat_raw")]
                                                              if KEEP_RAW else []), LOGS)
follow(LOGS, ["era5", "gridsat"])

# %% [5] stage A: the storm network on GPUs 0-3, the retrieval on GPUs 4-7, side by side ─────────────────
if not TENDENCY.exists():                    # a checkpoint from before 0.11 does not record it: measure it
    from naturev1.b300 import memmap_tendency

    torch.save(memmap_tendency(ERA5, ANALYSIS_CHANNELS), TENDENCY)
common = ["--checkpoint", str(BASE), "--root", str(SAT), "--table", str(LABELS), "--output", str(CKPT),
          "--hours", str(STAGE_A_HOURS), "--hub-repo", HF_REPO]
start_job("stage-a-satnet", ["-m", "naturev1.sattrain", "satnet", *common, "--workers", "12"], LOGS,
          gpus=[0, 1, 2, 3])
start_job("stage-a-retrieval", ["-m", "naturev1.sattrain", "retrieval", *common, "--era5", str(ERA5),
                                "--tendency", str(TENDENCY), "--workers", "10"], LOGS, gpus=[4, 5, 6, 7])
follow(LOGS, ["stage-a-satnet", "stage-a-retrieval"])

# %% [6] stage B: storm heads + the backbone's last blocks, every basin, satellites in the loop ───────────
start_job("stage-b", ["-m", "naturev1.stormtrain", "--base", str(BASE),
                      "--satnet", str(CKPT / "stage-a-satnet" / "latest.pt"),
                      "--retrieval", str(CKPT / "stage-a-retrieval" / "latest.pt"),
                      "--era5", str(ERA5), "--tendency", str(TENDENCY), "--ibtracs", str(IBTRACS),
                      "--table", str(LABELS), "--root", str(SAT), "--output", str(CKPT),
                      "--hours", str(STAGE_B_HOURS), "--tail", str(TAIL_BLOCKS), "--rotations", str(ROTATIONS),
                      "--hub-repo", HF_REPO], LOGS,
          gpus=list(range(8)))
follow(LOGS, ["stage-b"])
print((CKPT / "stage-b" / "report.txt").read_text() if (CKPT / "stage-b" / "report.txt").exists() else "no report yet")

# %% [7] publish: the release only if it beat the model it started from ──────────────────────────────────
from naturev1.b300 import publish_release

where = publish_release(CKPT / "stage-b" / "latest.pt", STATS, HF_REPO, DATA / "release",
                        comparison=CKPT / "comparison.json", report=CKPT / "stage-b" / "report.txt")
print("published to", where, "-- now clear HF_TOKEN in cell [0]")
