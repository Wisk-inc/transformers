# Pluvium-500 on one NVIDIA RTX PRO 6000 Blackwell (96 GB): train, calibrate, score, publish, forecast.
# Paste into one notebook cell (Jupyter, molab, a rented GPU box) or run as `python rtx6000_pluvium.py`.
# Re-running continues where the last session stopped (data and checkpoints are mirrored to the Hub).

GPU_HOURS = 24                              # wall-clock budget for this session (training gets what the rest leaves)
HF_TOKEN = "PASTE_YOUR_ORG_TOKEN_HERE"      # a write token for the Hub organisation (Settings -> Access Tokens)
HF_REPO = "CorxLabs/Pluvium-500"            # where data, checkpoints and the finished model are kept
ROOT = "/workspace/pluvium"                 # local working folder (needs ~150 GB free for six years of data)
SIZE = "pluvium-500"                        # or pluvium-base for a faster, smaller model

import os
import subprocess
import sys


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


# 1. packages: Pluvium from PyPI (falls back to the source tree), and a PyTorch that has Blackwell (sm_120) kernels
try:
    sh(f"{sys.executable} -m pip install -q --upgrade pluvium")
except subprocess.CalledProcessError:
    sh(f"{sys.executable} -m pip install -q 'git+https://github.com/Wisk-inc/transformers.git"
       f"@claude/linear-context-architecture-a35c6h#subdirectory=pluvium'")
probe = ("import torch; assert torch.cuda.is_available(), 'no GPU visible';"
         "x = torch.ones(8, device='cuda'); print(torch.cuda.get_device_name(0), torch.__version__, float((x @ x)))")
if subprocess.run([sys.executable, "-c", probe]).returncode != 0:
    print("PyTorch here has no kernels for this GPU; installing the CUDA 12.8 build", flush=True)
    sh(f"{sys.executable} -m pip install -q --upgrade torch --index-url https://download.pytorch.org/whl/cu128")
    subprocess.run([sys.executable, "-c", probe], check=True)

# 2. the program, streamed line by line (stopping the cell stops it; running the cell again resumes it)
if HF_TOKEN.startswith("PASTE"):
    print("No Hub token: nothing will be mirrored, so a stopped session cannot resume. Paste a token above.")
env = dict(os.environ, GPU_HOURS=str(GPU_HOURS), HF_REPO=HF_REPO, PLUVIUM_ROOT=ROOT, PLUVIUM_SIZE=SIZE,
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
if not HF_TOKEN.startswith("PASTE"):
    env["HF_TOKEN"] = HF_TOKEN
proc = subprocess.Popen([sys.executable, "-u", "-m", "pluvium.program"], env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1)
try:
    for line in proc.stdout:
        print(line, end="", flush=True)
finally:
    if proc.poll() is None:
        proc.terminate()
print(f"program finished with exit code {proc.wait()}; the forecast is in {ROOT}/forecast/report.txt")
