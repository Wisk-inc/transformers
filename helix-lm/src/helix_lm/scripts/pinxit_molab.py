# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Pinxit, trained on one GPU in notebook sessions.

    python -m helix_lm.scripts.pinxit_molab --root /workspace/pinxit --stage all --hf-repo CorxLabs/Pinxit-1 \\
        --size px-base --sheet-hours 8 --picture-hours 12

Stages, each resumable, each leaving a marker under ``<root>/markers``:

    setup     folders, the GPU, the token (``$HF_TOKEN`` or ``/root/hf_token``)
    sheets    the first training stage: practice sheets (no data needed)
    data      public-domain paintings from three museums, CC BY photos from Open Images, your own pictures in
              ``<root>/uploads/photos`` and ``<root>/uploads/paintings``; a style card for every painting
    pictures  the second: real photos, the mind's plans, the pretrained eye with its adapters
    gallery   paintings of test photos by the trained model and by the untrained hand, with scores
    push      everything to the Hub with a model card

The practice sheets need no pictures, so the GPU starts on them at once and the data stage runs beside them as a
second process; by the time its turn comes it has usually finished. If that process fails nothing is lost: the
stage then runs in its place and takes up where the other stopped.

The run mirrors itself to a private Hub repository (checkpoints with their hours, markers, the gallery) every
``--push-every-hours`` and whenever it stops, and starts by pulling whatever the Hub has that is newer. A
notebook session that ends after 12 hours is no problem: the run saves, pushes and stops half an hour before
(:mod:`helix_lm.molab`), and the next session's run goes on from there. A stop file ``<root>/stop`` ends a run
cleanly. Hours trained are kept in the checkpoint, so asking for more hours later continues a stage.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

STAGES = ("setup", "sheets", "data", "pictures", "gallery", "push")
DEFAULT_REPO = "CorxLabs/Pinxit-1"
SESSION = None
LAST_PUSH = 0.0
SAME_FAILURE_LIMIT = 3
TAG = ""                # put before every line this process logs ("[fetch]" in the one that fetches pictures)
FETCHER = None          # the data stage running beside the first training stage


def log(*parts) -> None:
    print(time.strftime("[%H:%M:%S]"), *(([TAG] if TAG else []) + list(parts)), flush=True)


def token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if not tok and Path("/root/hf_token").exists():
        tok = Path("/root/hf_token").read_text().strip()
    return tok or None


def session():
    from helix_lm.molab import Session

    global SESSION
    if SESSION is None:
        SESSION = Session(None)
    return SESSION


def keep_going(root: Path) -> bool:
    return not (root / "stop").exists() and not session().wrap_up()


def stop_if_session_ends(args, what: str) -> None:
    from helix_lm.molab import SessionEnding

    if session().wrap_up():
        hub_push(args, force=True, reason="session end")
        raise SessionEnding(f"{what}: saved and pushed before the session ends", pushed=True)


def take_lock(root: Path) -> bool:
    """One run per folder: a second one would write over the first one's checkpoints."""
    lock = root / "run.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
            if pid != os.getpid() and Path(f"/proc/{pid}").exists():
                log(f"!! another run (pid {pid}) is using {root}; stop it first, or delete run.lock")
                return False
        except ValueError:
            pass
    lock.write_text(str(os.getpid()))
    return True


def checkpoint(root: Path, size: str) -> Path:
    return root / "runs" / size / "latest.pt"


def mark(root: Path, stage: str, hours: float = 0.0) -> None:
    (root / "markers").mkdir(parents=True, exist_ok=True)
    (root / "markers" / stage).write_text(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "hours": hours}))


def marked_hours(root: Path, stage: str) -> float | None:
    path = root / "markers" / stage
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text()).get("hours", 0.0))
    except ValueError:
        return 0.0


def complete(root: Path, stage: str, args) -> bool:
    """Whether a stage has nothing left to do: its marker is there, and a training stage trained the hours now
    asked for; the gallery is older than the checkpoint it would show."""
    hours = marked_hours(root, stage)
    if hours is None:
        return False
    if stage == "sheets":
        return hours >= args.sheet_hours - 0.02
    if stage == "pictures":
        return hours >= args.picture_hours - 0.02
    if stage == "gallery":
        ckpt, shown = checkpoint(root, args.size), root / "gallery" / "metrics.json"
        return shown.exists() and (not ckpt.exists() or shown.stat().st_mtime >= ckpt.stat().st_mtime)
    return True


# ---------------------------------------------------------------------------------------------- hub --
def hub_api():
    tok = token()
    if not tok:
        return None
    from huggingface_hub import HfApi

    return HfApi(token=tok)


def model_card(root: Path, args) -> str:
    ckpt = checkpoint(root, args.size).with_suffix(".json")
    trained = json.loads(ckpt.read_text()) if ckpt.exists() else {}
    paintings = len(list((root / "paintings").glob("*.jpg"))) if (root / "paintings").exists() else 0
    photos = len(list((root / "photos").glob("*.jpg"))) if (root / "photos").exists() else 0
    table = (root / "gallery" / "README.md").read_text() if (root / "gallery" / "README.md").exists() else \
        "No gallery yet: the run has not reached that stage."
    return f"""---
license: apache-2.0
library_name: helix-lm
tags: [painting, stroke-based-rendering, image-to-image, helix]
---

# Pinxit-1 ({args.size})

Pinxit paints a photo. It does not start from noise: it looks at the picture, decides what it is about, mixes a
palette, and lays brush strokes from broad to fine, each with a brush, two colours, a pressure and a speed. What
it gives back is the list of strokes, so a painting can be rendered at any size, replayed, and changed later.

Built on the Helix trunk (`helix_lm.anansi`) with a pretrained eye (`{args.eye}`) that is fine-tuned through
small adapters. Code and documentation: `pip install "helix-lm[paint]"`, `docs/PINXIT.md`.

```python
from helix_lm.paint import load_hand, paint
hand, eye = load_hand("{args.hf_repo or DEFAULT_REPO}", "{args.size}")       # the trained painter and its eye
painting = paint("photo.jpg", style="a_painting.jpg", emotion="majesty", hand=hand, eye=eye)
painting.save("painting.png")
painting.save("painting.pinxit")      # with its memory, to change later
```

## Training so far

Stage `{trained.get('stage', 'not started')}`, step {trained.get('step', 0)}, {trained.get('hours', 0.0):.1f} hours
in that stage (saved {trained.get('saved', 'never')}). Paintings learned from: {paintings} (public domain, from the
Art Institute of Chicago, the Cleveland Museum of Art and the National Gallery of Art; listed in
`data/paintings.jsonl`). Photos: {photos} (Open Images, CC BY 2.0; authors listed in `data/photos.jsonl`).

## Gallery and scores

{table}

## What it is not

The scores say whether a painting did what was asked of it, not whether it is good. The composition, the pushed
colours and what is left unfinished come from written rules (the experimental part), not from the trained
weights. Painting something in from words depends on the eye's sense of the words and is loose.
"""


def hub_push(args, force: bool = False, reason: str = "checkpoint") -> bool:
    """Checkpoint, markers, records, gallery and the model card to the Hub. Never raises."""
    global LAST_PUSH
    if not args.hf_repo or not token():
        return False
    if not force and time.time() - LAST_PUSH < args.push_every_hours * 3600:
        return False
    try:
        from huggingface_hub import CommitOperationAdd

        api, root = hub_api(), Path(args.root)
        try:
            api.create_repo(args.hf_repo, exist_ok=True, private=not args.public)
        except Exception as error:
            log(f"hub: create_repo skipped ({type(error).__name__})")
        ops = [CommitOperationAdd("README.md", model_card(root, args).encode())]
        small = list((root / "markers").glob("*")) if (root / "markers").exists() else []
        for path in small:
            ops.append(CommitOperationAdd(f"markers/{path.name}", str(path)))
        for record in ("paintings/paintings.jsonl", "photos/photos.jsonl"):
            if (root / record).exists():
                ops.append(CommitOperationAdd(f"data/{Path(record).name}", str(root / record)))
        if (root / "gallery").exists():
            for path in sorted((root / "gallery").iterdir()):
                if path.suffix in (".png", ".json", ".md", ".gif"):
                    ops.append(CommitOperationAdd(f"gallery/{path.name}", str(path)))
        ckpt = checkpoint(root, args.size)
        if ckpt.exists():
            ops.append(CommitOperationAdd(f"pinxit/{args.size}/latest.pt", str(ckpt)))
            if ckpt.with_suffix(".json").exists():
                ops.append(CommitOperationAdd(f"pinxit/{args.size}/latest.json", str(ckpt.with_suffix(".json"))))
        for attempt in range(3):
            try:
                api.create_commit(args.hf_repo, operations=ops,
                                  commit_message=f"{reason} {time.strftime('%Y-%m-%d %H:%M')}")
                break
            except Exception as error:
                log(f"hub: push attempt {attempt + 1} failed ({type(error).__name__}: {str(error)[:80]})")
                time.sleep(30 * (attempt + 1))
        else:
            return False
        if not args.no_squash:
            try:
                api.super_squash_history(args.hf_repo)
            except Exception as error:
                log(f"hub: history not squashed ({type(error).__name__})")
        LAST_PUSH = time.time()
        log(f"hub: pushed ({reason}) to https://huggingface.co/{args.hf_repo}")
        return True
    except Exception as error:
        log(f"hub: push failed ({type(error).__name__}: {str(error)[:100]}); the run continues")
        return False


def hub_pull(args) -> None:
    """Take the Hub's checkpoint when it has trained longer than the one here, and its markers."""
    api = hub_api()
    if api is None or not args.hf_repo or args.no_pull:
        return
    from huggingface_hub import hf_hub_download

    root = Path(args.root)
    try:
        files = set(api.list_repo_files(args.hf_repo))
    except Exception as error:
        log(f"hub: nothing to pull ({type(error).__name__}: {str(error)[:80]})")
        return
    cache = root / "hf" / "pull"
    remote, side = f"pinxit/{args.size}/latest.pt", f"pinxit/{args.size}/latest.json"
    ckpt = checkpoint(root, args.size)
    if remote in files and side in files:
        theirs = json.loads(Path(hf_hub_download(args.hf_repo, side, token=token(), local_dir=str(cache))).read_text())
        ours = json.loads(ckpt.with_suffix(".json").read_text()) if ckpt.with_suffix(".json").exists() else {}
        # the one that has trained longer, over all its stages
        ahead = theirs.get("total_hours", theirs.get("hours", 0.0)) > ours.get("total_hours", ours.get("hours", 0.0))
        if not ckpt.exists() or ahead:
            log(f"hub: pulling the checkpoint ({theirs.get('stage')}, step {theirs.get('step')}, "
                f"{theirs.get('hours', 0):.1f} h)")
            got = Path(hf_hub_download(args.hf_repo, remote, token=token(), local_dir=str(cache)))
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(got), str(ckpt))
            ckpt.with_suffix(".json").write_text(json.dumps(theirs))
            for name in ("sheets", "pictures"):
                if f"markers/{name}" in files:
                    got = Path(hf_hub_download(args.hf_repo, f"markers/{name}", token=token(), local_dir=str(cache)))
                    (root / "markers").mkdir(parents=True, exist_ok=True)
                    shutil.move(str(got), str(root / "markers" / name))
    log(f"hub: in step with https://huggingface.co/{args.hf_repo}")


# ------------------------------------------------------------------------------------------- fetcher --
def start_fetcher(args) -> None:
    """Run the data stage as a second process, beside the practice sheets (see the module's notes)."""
    global FETCHER
    if FETCHER is not None and FETCHER.poll() is None:
        return
    command = [sys.executable, "-m", "helix_lm.scripts.pinxit_molab", "--root", str(args.root), "--stage", "data",
               "--beside", "--hf-repo", "", "--chicago", str(args.chicago), "--cleveland", str(args.cleveland),
               "--washington", str(args.washington), "--photos", str(args.photos)]
    FETCHER = subprocess.Popen(command, env=dict(os.environ, **session().environment()))
    log("the pictures are being fetched by a second process (its lines say [fetch]) while the practice sheets train")


def stop_fetcher() -> None:
    if FETCHER is not None and FETCHER.poll() is None:
        FETCHER.terminate()
        try:
            FETCHER.wait(timeout=30)
        except subprocess.TimeoutExpired:
            FETCHER.kill()


def fetcher_finished(args) -> bool:
    """Wait for the second process when the data stage's turn comes. True when it did the whole stage."""
    if FETCHER is None:
        return False
    waited = False
    while FETCHER.poll() is None and keep_going(Path(args.root)):
        if not waited:
            log("data: waiting for the second process to finish fetching")
            waited = True
        time.sleep(5)
    return FETCHER.poll() == 0 and marked_hours(Path(args.root), "data") is not None


# -------------------------------------------------------------------------------------------- stages --
def stage_setup(args) -> None:
    import torch

    root = Path(args.root)
    for sub in ("paintings", "photos", "uploads/photos", "uploads/paintings", "runs", "gallery", "markers", "hf"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root / "hf"))
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log(f"GPU: {props.name}, {props.total_memory / 2 ** 30:.0f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        log("!! no GPU visible: the data stage runs, training will be very slow")
    log("token:", "found" if token() else "NOT SET (set HF_TOKEN; the run cannot mirror itself to the Hub without)")
    free = shutil.disk_usage(root).free / 2 ** 30
    log(f"disk: {free:.0f} GB free under {root}" + ("" if free > 20 else " (little: the pictures need about 6 GB)"))
    mark(root, "setup")


def stage_data(args) -> None:
    from helix_lm.paint.data import SOURCES, download, find_pictures, open_images, style_cards

    root = Path(args.root)
    if fetcher_finished(args):
        log("data: fetched beside the practice sheets")
        return
    stop_fetcher()                                   # whatever it left undone is done here, by one process
    stop_if_session_ends(args, "data")
    wanted = {"chicago": args.chicago, "cleveland": args.cleveland, "washington": args.washington}
    for name, limit in wanted.items():
        if limit <= 0:
            continue
        stop_if_session_ends(args, f"data ({name})")
        try:
            have = download(SOURCES[name](limit), root / "paintings", "paintings.jsonl", should_stop=session().wrap_up,
                            log=log)
            log(f"{name}: the paintings folder now holds {have}")
        except Exception as error:                   # one museum out of reach must not cost the others
            log(f"{name}: not fetched ({type(error).__name__}: {str(error)[:100]}); going on without it")
    if args.photos > 0:
        stop_if_session_ends(args, "data (photos)")
        try:
            have = download(open_images(args.photos), root / "photos", "photos.jsonl", max_side=640,
                            should_stop=session().wrap_up, log=log)
            log(f"open images: the photos folder now holds {have}")
        except Exception as error:
            log(f"open images: not fetched ({type(error).__name__}: {str(error)[:100]})")
    stop_if_session_ends(args, "data")
    for folder in (root / "paintings", root / "uploads" / "paintings"):
        made = style_cards(folder, log, session().wrap_up)
        if made:
            log(f"{folder.name}: {made} style cards measured")
    stop_if_session_ends(args, "data (style cards)")
    photos = len(find_pictures(root / "photos")) + len(find_pictures(root / "uploads" / "photos"))
    paintings = len(find_pictures(root / "paintings")) + len(find_pictures(root / "uploads" / "paintings"))
    log(f"data: {photos} photos, {paintings} paintings")
    if photos == 0:
        raise RuntimeError("no photos could be fetched and uploads/photos is empty: nothing to train on")
    mark(root, "data")


def build_model(args, device):
    from helix_lm.paint.eye import eye_width
    from helix_lm.paint.model import Pinxit, PinxitConfig

    model = Pinxit(PinxitConfig.named(args.size, eye_dim=eye_width(args.eye))).to(device)
    log(f"Pinxit {args.size}: {model.num_parameters() / 1e6:.1f}M parameters on the Helix trunk, eye {args.eye}")
    return model


def stage_sheets(args) -> None:
    import torch

    from helix_lm.paint.train import train_sheets

    root = Path(args.root)
    if args.sheet_hours <= 0:
        mark(root, "sheets")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    done = train_sheets(model, hours=args.sheet_hours, steps=args.sheet_steps or None, batch=args.batch,
                        checkpoint=checkpoint(root, args.size), log=log, keep_going=lambda: keep_going(root),
                        save_minutes=args.save_minutes,
                        on_save=lambda step, hours: hub_push(args, reason=f"sheets step {step}"))
    log(f"sheets: step {done['step']}, {done['hours']:.2f} h")
    stop_if_session_ends(args, "sheets")
    if not (root / "stop").exists():
        mark(root, "sheets", args.sheet_hours if args.sheet_steps else done["hours"])


def picture_lists(root: Path) -> tuple[list, list]:
    from helix_lm.paint.data import find_pictures

    own_photos = find_pictures(root / "uploads" / "photos")
    own_paintings = find_pictures(root / "uploads" / "paintings")
    photos = find_pictures(root / "photos") + own_photos * 20            # your own pictures are met more often
    paintings = [p for p in find_pictures(root / "paintings") + own_paintings * 20
                 if p.with_suffix(".card.json").exists()]
    return photos, paintings


def stage_pictures(args) -> None:
    import torch

    from helix_lm.paint.data import plan_stream
    from helix_lm.paint.eye import load_eye
    from helix_lm.paint.train import train_pictures

    root = Path(args.root)
    if args.picture_hours <= 0:
        mark(root, "pictures")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    eye = load_eye(args.eye, device, lora=args.lora) if args.eye not in ("", "none") else None
    photos, paintings = picture_lists(root)
    log(f"pictures: {len(set(photos))} photos, {len(set(paintings))} paintings with cards (your own are met "
        f"twenty times as often)")
    plans = plan_stream(photos, paintings, work=args.train_work, workers=args.workers)
    done = train_pictures(model, plans, eye=eye, hours=args.picture_hours, steps=args.picture_steps or None,
                          cells=args.picture_cells, pictures=args.pictures_per_step, pool=args.plan_pool,
                          checkpoint=checkpoint(root, args.size), log=log, keep_going=lambda: keep_going(root),
                          save_minutes=args.save_minutes,
                          on_save=lambda step, hours: hub_push(args, reason=f"pictures step {step}"))
    log(f"pictures: step {done['step']}, {done['hours']:.2f} h")
    stop_if_session_ends(args, "pictures")
    if not (root / "stop").exists():
        mark(root, "pictures", args.picture_hours if args.picture_steps else done["hours"])


def stage_gallery(args) -> None:

    from helix_lm.paint.bench import compare, judge
    from helix_lm.paint.data import find_pictures
    from helix_lm.paint.model import load_hand
    from helix_lm.paint.painting import paint

    root = Path(args.root)
    ckpt = checkpoint(root, args.size)
    out = root / "gallery"
    out.mkdir(parents=True, exist_ok=True)
    hands, eye = {"rules": None}, None
    if ckpt.exists():
        hands["pinxit"], eye = load_hand(ckpt, args.size, polish=args.polish)
    photos = (find_pictures(root / "uploads" / "photos") + find_pictures(root / "photos"))[: args.gallery]
    styles = [None] + (find_pictures(root / "uploads" / "paintings") + find_pictures(root / "paintings"))[:2]
    rows = []
    for i, photo in enumerate(photos):
        style = styles[i % len(styles)]
        emotion = "majesty" if style is None else None
        for name, hand in hands.items():
            if session().left_hours() < 0.6:
                log("gallery: stopped early, the session is nearly over")
                break
            began = time.time()
            try:
                # both hands are given the same plan (the eye helps make it), so their scores compare
                painting = paint(photo, style=style, emotion=emotion, hand=hand, eye=eye, work=args.gallery_work)
            except Exception as error:               # one picture that fails must not cost the gallery
                log(f"gallery: {photo.name} with {name} failed ({type(error).__name__}: {str(error)[:120]})")
                continue
            score = judge(painting)
            score.update(name=photo.stem[:24], hand=name, seconds=time.time() - began,
                         style=style.stem if style else "own colours, majesty")
            rows.append(score)
            painting.save(out / f"{i:02d}_{name}.png", width=1024)
            if name == "pinxit" or len(hands) == 1:
                painting.sheet(out / f"{i:02d}_passes.png", width=256)
            log(f"gallery: {photo.name} by {name}: closeness {score['closeness']:.3f}, {score['strokes']} strokes, "
                f"{score['seconds']:.0f} s")
    (out / "metrics.json").write_text(json.dumps(rows, indent=1))
    (out / "README.md").write_text(compare(rows) + "\n\n`rules` is the untrained hand (written rules, refined by "
                                   "gradient descent); `pinxit` is the trained model with a quarter of that "
                                   "refinement. Lower closeness is nearer the plan.\n")
    mark(root, "gallery")


def stage_push(args) -> None:
    hub_push(args, force=True, reason="release")
    mark(Path(args.root), "push")


# ---------------------------------------------------------------------------------------------- main --
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Pinxit on one GPU, in sessions")
    parser.add_argument("--root", default="/workspace/pinxit")
    parser.add_argument("--stage", default="all")
    parser.add_argument("--size", default="px-base", choices=["px-tiny", "px-small", "px-base", "px-large"])
    parser.add_argument("--eye", default="google/siglip2-base-patch16-256",
                        help="the pretrained eye; 'none' or 'tiny'")
    parser.add_argument("--lora", type=int, default=16)
    parser.add_argument("--sheet-hours", type=float, default=8.0)
    parser.add_argument("--picture-hours", type=float, default=12.0)
    parser.add_argument("--sheet-steps", type=int, default=0, help="stop the first stage after this many steps")
    parser.add_argument("--picture-steps", type=int, default=0)
    parser.add_argument("--batch", type=int, default=0, help="practice sheets a step; 0 fills the GPU")
    parser.add_argument("--picture-cells", type=int, default=0,
                        help="canvas cells a picture step may hold; 0 fills the GPU")
    parser.add_argument("--pictures-per-step", type=int, default=48, help="the most pictures in one step")
    parser.add_argument("--plan-pool", type=int, default=256, help="plans kept ready for the picture stage")
    parser.add_argument("--train-work", type=int, default=256)
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 4) - 2)))
    parser.add_argument("--chicago", type=int, default=1000)
    parser.add_argument("--cleveland", type=int, default=2500)
    parser.add_argument("--washington", type=int, default=2500)
    parser.add_argument("--photos", type=int, default=10000)
    parser.add_argument("--gallery", type=int, default=6)
    parser.add_argument("--gallery-work", type=int, default=384)
    parser.add_argument("--polish", type=float, default=0.25)
    parser.add_argument("--save-minutes", type=float, default=20.0)
    parser.add_argument("--hf-repo", default=DEFAULT_REPO)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--push-every-hours", type=float, default=1.0)
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument("--no-squash", action="store_true")
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--retry-wait", type=float, default=30.0)
    parser.add_argument("--beside", action="store_true", help=argparse.SUPPRESS)   # the second process (see above)
    args = parser.parse_args(argv)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if args.stage != "all" and args.stage not in STAGES:
        parser.error(f"unknown stage {args.stage}; there are {', '.join(STAGES)} and all")
    from helix_lm.molab import Session, SessionEnding

    global SESSION, TAG
    SESSION = Session.find()
    if args.beside:                                   # no lock and no Hub: the run it works beside has both
        TAG = "[fetch]"
        try:
            stage_data(args)
        except SessionEnding:
            log("stopped: the session is ending")
        return 0
    if not take_lock(root):
        return 3
    log(SESSION.describe())
    hub_pull(args)
    stages = STAGES if args.stage == "all" else (args.stage,)
    if args.stage == "all" and args.sheet_hours > 0 and not complete(root, "data", args) \
            and not complete(root, "sheets", args):
        start_fetcher(args)
    try:
        return run_stages(args, stages)
    finally:
        stop_fetcher()


def run_stages(args, stages) -> int:
    from helix_lm.molab import SessionEnding

    root = Path(args.root)
    attempt, last_failure, repeats = 0, "", 0
    while True:
        try:
            for stage in stages:
                if args.stage == "all" and complete(root, stage, args) and stage != "push":
                    log(f"{stage}: done earlier, skipped (delete markers/{stage} to redo)")
                    continue
                log(f"== stage {stage}")
                globals()[f"stage_{stage}"](args)
                if (root / "stop").exists():
                    log("stop file found; stopping")
                    hub_push(args, force=True, reason="stop file")
                    return 0
            log("all stages done")
            return 0
        except SessionEnding as ending:
            if not ending.pushed:
                hub_push(args, force=True, reason="session end")
            log(f"{ending}. Everything is on the Hub; run the cell again in a new session to go on from here")
            return 0
        except KeyboardInterrupt:
            log("interrupted: saving the state to the Hub")
            hub_push(args, force=True, reason="interrupted")
            return 130
        except Exception as error:
            attempt += 1
            failure = f"{type(error).__name__}: {str(error)[:200]}"
            repeats = repeats + 1 if failure == last_failure else 1
            last_failure = failure
            log(f"!! stage failed ({failure[:160]}); attempt {attempt} of {args.retries}")
            traceback.print_exc()
            hub_push(args, force=True, reason=f"after failure {attempt}")
            if attempt >= args.retries or repeats >= SAME_FAILURE_LIMIT:
                if repeats >= SAME_FAILURE_LIMIT:
                    log(f"!! the same failure came back {repeats} times in a row; stopping instead of trying again")
                raise
            if session().wrap_up():
                log("the session is about to end; everything is on the Hub, run the cell again to go on")
                return 0
            time.sleep(args.retry_wait)
            hub_pull(args)


if __name__ == "__main__":
    sys.exit(main())
