# Pinxit Animate: one cell, plain Python. Runs in molab, Jupyter, or as "python pinxit_animate_cell.py" on a GPU.
# Say what the animation shows. An open video model imagines the motion (Wan 2.2 14B on a GPU with 80 GB or more),
# then Pinxit draws it as anime, the way a studio does: the background painted once and held, the character inked
# and painted as a cel on every drawing, timed on ones, twos and threes, faces drawn with care, every line steadied
# through time, every drawing checked against its frame. It writes the animation, a video of it being drawn at the
# desk with a hand holding the pen, and a .pinxa file that keeps everything, to change or carry on later.
# No token is needed.
# ruff: noqa: E402  (the settings come first, so they are the first thing seen in the notebook)

WORDS = "a cheerful anime girl with short red hair and green eyes waving hello, upper body, in a sunny park"
NEXT = "she laughs and turns to run down the path"   # a second clip that starts where the first ends; "" for none
CHARACTER = ""                            # a picture of the character to start from (a path); "" for none
MOTION = ""                               # a video of someone moving (a path): with MODEL "wan-animate" and a
                                          # CHARACTER, the character performs that motion; "" for none
MODEL = "auto"                            # auto: Wan 2.2 14B from 80 GB, TI2V 5B from 40 GB, else AnimateLCM;
                                          # or wan-animate, wan-vace, ltx, hunyuan, cogvideox, skyreels, a Hub repo
LORA = ""                                 # a LoRA for the video model (a style, a character); "" for none
FRAMES = 0                                # 0: the model's own length (Wan 2.2: 81 or 121 frames)
LINES = "brush pen"                       # brush pen (bold anime lines), pen (fine) or pencil
TIMING = 2                                # most movement drawn on 1s, 2s or 3s (fast action is always on ones)
OUT_FPS = 24                              # frames a second of the video written out
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


def say(line: str) -> None:
    print(line, flush=True)


began = time.time()
animator = load_animator(MODEL, lora=LORA or None)
print(f"video model {animator.name} loaded in {time.time() - began:.0f} s", flush=True)
name = "".join(c if c.isalnum() else "_" for c in WORDS.lower())[:40].strip("_")
clip = animate(WORDS, animator=animator, frames=FRAMES or None, character=CHARACTER or None,
               motion=MOTION or None, lines=LINES, on=TIMING, log=say)
if NEXT:
    clip.extend(NEXT, animator=animator, frames=FRAMES or None, log=say)
del animator                              # the video model is not needed for the rest
if torch.cuda.is_available():
    torch.cuda.empty_cache()
clip.save(out / f"{name}.pinxa")
clip.save(out / f"{name}.mp4", fps=OUT_FPS)
clip.save(out / f"{name}_drawings.png", width=256)
clip.making_of(out / f"{name}_being_drawn.mp4")
checks = clip.settings.get("checks", [])
if checks:
    worst = max(checks, key=lambda c: c["error"])
    print(f"\nchecked {len(checks)} drawings against their frames: colour off by "
          f"{sum(c['error'] for c in checks) / len(checks):.3f} on average, the worst frame {worst['frame'] + 1} "
          f"({worst['error']:.3f})", flush=True)
print(f"done in {(time.time() - began) / 60:.0f} minutes:", flush=True)
for file in sorted(out.glob(f"{name}*")):
    print(f"  {file}  ({file.stat().st_size / 1e6:.1f} MB)")
print("Open the .pinxa later to change it: Animation.load(path).recolor(...), .retool(...), .extend(...).")
