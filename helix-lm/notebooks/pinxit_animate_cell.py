# Pinxit Animate: one cell, plain Python. Runs in molab, Jupyter, or as "python pinxit_animate_cell.py" on a GPU.
# Say what the animation shows. An open video model imagines the motion (Wan 2.2 on a large GPU, AnimateLCM
# otherwise), then Pinxit draws it by hand, frame by frame, on a whiteboard: the lines with a pen, the colour with
# markers, every frame traced over the last one so the character keeps its look. It writes the animation, a video
# of it being drawn, and a .pinxa file that keeps everything, to change or carry on later. No token is needed.
# ruff: noqa: E402  (the settings come first, so they are the first thing seen in the notebook)

WORDS = "a cheerful orange fox running through a snowy forest, side view"
NEXT = "the fox jumps over a log"          # a second clip that starts where the first ends; "" for none
CHARACTER = ""                            # a picture of the character to start from (a path); "" for none
MODEL = "auto"                            # auto: Wan 2.2 TI2V 5B on a GPU with 40 GB or more, else AnimateLCM
LORA = ""                                 # a LoRA for the video model (a style, a character); "" for none
FRAMES = 0                                # 0: the model's own length (Wan 2.2: 121 frames, 5 seconds)
LINES = "pen"                             # pen, brush pen or pencil
OUT = "auto"                              # auto: /workspace/pinxit_animations, else your home folder
HELIX = "2.6.0"

import os
import subprocess
import sys
import time
from pathlib import Path


def installed() -> str:
    out = subprocess.run([sys.executable, "-c", "import helix_lm; print(helix_lm.__version__)"],
                         capture_output=True, text=True)
    return out.stdout.strip()


if installed() != HELIX:
    pypi = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", f"helix-lm[paint]=={HELIX}"]
    if subprocess.run(pypi).returncode != 0 or installed() != HELIX:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "helix-lm[paint] @ git+https://"
                        "github.com/Wisk-inc/transformers.git@claude/linear-context-architecture-a35c6h"
                        "#subdirectory=helix-lm"])
if installed() != HELIX:
    raise SystemExit(f"helix-lm {HELIX} could not be installed (found {installed() or 'none'}); the error is above.")

import torch

from helix_lm.paint.animate import animate
from helix_lm.paint.video import load_animator

out = Path("/workspace/pinxit_animations" if OUT == "auto" and os.access("/workspace", os.W_OK) else
           (Path.home() / "pinxit_animations" if OUT == "auto" else OUT)).expanduser()
out.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(out.parent / "hf"))
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU (this will take a long time)"
print(f"helix-lm {installed()} | {gpu} | writing to {out}", flush=True)

began = time.time()
animator = load_animator(MODEL, lora=LORA or None)
print(f"video model {animator.name} loaded in {time.time() - began:.0f} s", flush=True)
name = "".join(c if c.isalnum() else "_" for c in WORDS.lower())[:40].strip("_")
clip = animate(WORDS, animator=animator, frames=FRAMES or None, character=CHARACTER or None, lines=LINES,
               log=lambda line: print(line, flush=True))
if NEXT:
    clip.extend(NEXT, animator=animator, frames=FRAMES or None, log=lambda line: print(line, flush=True))
clip.save(out / f"{name}.pinxa")
clip.save(out / f"{name}.mp4")
clip.save(out / f"{name}_frames.png", width=256)
clip.making_of(out / f"{name}_being_drawn.mp4")
print(f"\ndone in {(time.time() - began) / 60:.0f} minutes:", flush=True)
for file in sorted(out.glob(f"{name}*")):
    print(f"  {file}  ({file.stat().st_size / 1e6:.1f} MB)")
print("Open the .pinxa later to change it: Animation.load(path).recolor(...), .retool(...), .extend(...).")
