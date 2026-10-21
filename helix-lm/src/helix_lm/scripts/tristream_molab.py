# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
TriStream-2 on one RTX PRO 6000 Blackwell (96 GB): the whole program as one script for a notebook.

In a notebook cell:

    !pip install -q "helix-lm[music]"
    !python -m helix_lm.scripts.tristream_molab --root /workspace/tristream --stage all \
        --hf-repo CorxLabs/TriStream-2-765M

or stage by stage (``setup``, ``data``, ``prepare``, ``codec``, ``train``, ``eval``, ``push``, ``swap``). Every
stage is resumable and writes a ``done`` marker; the token is read from ``$HF_TOKEN`` or ``/root/hf_token``
(never pasted into a notebook that gets shared). Stop with Ctrl-C or the stop file ``<root>/stop``; start again
to continue.

What the stages do and how long they take on this GPU (estimates; the log prints measured rates):

| stage   | what                                                                                   | time        |
| ---     | ---                                                                                    | ---         |
| setup   | packages, GPU check, folders                                                           | minutes     |
| data    | the commercially clean sources (VocalSet, Dagstuhl ChoirSet, ESMUC, Emilia-YODAS       | hours       |
|         | subset, Slakh instrumentals, FMA CC-BY filter), plus your uploads; research-only sets  | (network)   |
|         | only with ``--include-research`` and only for evaluation                              |             |
| prepare | separation (HT-Demucs, MIT) for mixtures, F0, beats/key/sections, breath, dryness,     | ~1 day      |
|         | tags, singer-disjoint split, codec frames                                              | per 5k h    |
| codec   | the 25 Hz split-RVQ codec on the audio pool (or ``--codec-hours 0`` to skip and use a  | ~12 days    |
|         | pretrained open codec where its licence allows)                                       |             |
| train   | TriStream-2 765M with Muon, bf16, speech first then singing, singer-disjoint validation| ~7-10 days  |
|         | per 15B frames; checkpoints every 30 min, pushed to the Hub every 6 h                  |             |
| eval    | F0 RMSE/FFE, speaker similarity, breath/technique agreement, glitch rate, bleed, the   | hours       |
|         | critic's score of plans; SingMOS-Pro when installed                                   |             |
| swap    | the demo: a song + a reference voice -> the song in that voice, with the report        | minutes     |

The honest part: nothing here has been trained yet; the compute numbers come from the GPU's datasheet (about
500 TFLOPS dense bf16 inferred) and published codec/LM throughputs (docs/TRISTREAM.md). The clean-licence
singing data in the open is under a hundred hours; the script says what it found and what it did not.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import zipfile
from pathlib import Path

DEFAULT_REPO = "CorxLabs/Tristream-V2-SVS"
LAST_PUSH = 0.0

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

STAGES = ("setup", "data", "prepare", "codec", "refiner", "air", "train", "eval", "push", "swap")


def log(*parts) -> None:
    print(time.strftime("[%H:%M:%S]"), *parts, flush=True)


def token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if not tok and Path("/root/hf_token").exists():
        tok = Path("/root/hf_token").read_text().strip()
    return tok or None


def done(root: Path, stage: str) -> bool:
    return (root / "markers" / stage).exists()


def complete(root: Path, stage: str, args) -> bool:
    """A stage counts as done when its marker AND its artifact exist: a marker pulled from the Hub without the
    file behind it, or a prepared folder that was never copied to this machine, is not done."""
    if not done(root, stage):
        return False
    if stage == "prepare":
        return any((root / "prepared").rglob("manifest.json"))
    if stage in ("refiner", "air") and getattr(args, f"{stage}_hours", 0) <= 0:
        return True
    artifact = {
        "codec": root / "codec" / "codec.pt",
        "refiner": root / "refiner" / "refiner.pt",
        "air": root / "air" / "air.pt",
        "train": root / "runs" / args.size / "latest.pt",
        "eval": root / "eval" / "summary.json",
    }.get(stage)
    return artifact is None or artifact.exists()


class Budget:
    """Hours of a stage that carry across restarts: a stage stopped at hour 30 of 48 resumes with 18 left,
    not 48 again."""

    def __init__(self, hours: float, elapsed: float = 0.0) -> None:
        self.hours, self.before, self.start = hours, elapsed, time.time()

    @property
    def elapsed(self) -> float:
        return self.before + (time.time() - self.start) / 3600

    @property
    def remaining(self) -> float:
        return max(0.0, self.hours - self.elapsed)

    def over(self) -> bool:
        return self.remaining <= 0


def save_checkpoint(obj: dict, path: Path) -> None:
    """Atomic save (a crash mid-write never leaves a torn file) plus a small sidecar with the step and the hours,
    so a pull can compare checkpoints without loading them."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    side = {
        "step": int(obj.get("step", 0)),
        "hours": float(obj.get("hours", 0.0)),
        "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.with_suffix(".json").write_text(json.dumps(side))


def checkpoint_meta(path: Path) -> dict:
    side = path.with_suffix(".json")
    if side.exists():
        try:
            return json.loads(side.read_text())
        except Exception:
            pass
    return {"step": 0 if path.exists() else -1, "hours": 0.0}


class Prefetcher:
    """A thread that builds the next batches while the GPU works on this one (file reads are the CPU cost)."""

    def __init__(self, make, depth: int = 3) -> None:
        self.make, self.queue, self.stop = make, queue.Queue(depth), threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                item = self.make()
            except Exception as error:  # surfaces in get()
                item = error
            while not self.stop.is_set():
                try:
                    self.queue.put(item, timeout=0.5)
                    break
                except queue.Full:
                    continue

    def get(self):
        item = self.queue.get()
        if isinstance(item, Exception):
            raise item
        return item

    def drain(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        self.stop.set()


def gpu_max() -> None:
    """The settings that let a Blackwell card run flat out: TF32 matmuls, cuDNN autotuning, high precision mode."""
    import torch

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def extract_archives(folder: Path, delete: bool = False) -> int:
    """Unpack every zip / tar under ``folder`` into ``folder/extracted/<name>`` once (a marker per archive)."""
    count = 0
    for archive in sorted(folder.rglob("*")):
        if archive.is_dir() or "extracted" in archive.parts:
            continue
        lower = archive.name.lower()
        if not lower.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
            continue
        target = folder / "extracted" / re.sub(r"\.(zip|tar|tar\.gz|tgz)$", "", archive.name)
        marker = target / ".extracted"
        if marker.exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        log(f"extracting {archive.name} ({archive.stat().st_size / 1e9:.1f} GB)")
        try:
            if lower.endswith(".zip"):
                with zipfile.ZipFile(archive) as handle:
                    handle.extractall(target)
            else:
                with tarfile.open(archive) as handle:
                    handle.extractall(target, filter="data")
            marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            count += 1
            if delete:
                archive.unlink()
        except Exception as error:
            log(f"  could not extract {archive.name}: {type(error).__name__}: {str(error)[:80]}")
    return count


# -------------------------------------------------------------------------------------------- hub --
def prepared_records(root: Path) -> list[dict]:
    records = []
    for manifest in (root / "prepared").rglob("manifest.json"):
        data = json.loads(manifest.read_text())
        records += [{**clip, "dir": str(manifest.parent)} for clip in data["clips"] if clip.get("accepted")]
    return records


def hub_api():
    tok = token()
    if not tok:
        return None
    from huggingface_hub import HfApi

    return HfApi(token=tok)


def checkpoint_files(root: Path, size: str) -> list[tuple[Path, str]]:
    return [
        (root / "codec" / "codec.pt", "checkpoints/codec/codec.pt"),
        (root / "refiner" / "refiner.pt", "checkpoints/refiner/refiner.pt"),
        (root / "air" / "air.pt", "checkpoints/air/air.pt"),
        (root / "runs" / size / "latest.pt", f"checkpoints/runs/{size}/latest.pt"),
    ]


def hub_pull(args) -> None:
    """Resume from the Hub: every checkpoint the repo holds that is newer (more steps) than the local one is
    downloaded, with its marker and the eval summary, so a fresh machine continues where the last one stopped."""
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
    pulled = 0
    for local, remote in checkpoint_files(root, args.size):
        if remote not in files:
            continue
        remote_meta = {"step": 0, "hours": 0.0}
        side = remote[:-3] + ".json"
        if side in files:
            remote_meta = json.loads(
                Path(hf_hub_download(args.hf_repo, side, token=token(), local_dir=str(cache))).read_text()
            )
        local_meta = checkpoint_meta(local)
        if local.exists() and local_meta.get("step", 0) >= remote_meta.get("step", 0):
            continue
        log(f"hub: pulling {remote} (step {remote_meta.get('step')}, {remote_meta.get('hours', 0):.1f} h)")
        got = Path(hf_hub_download(args.hf_repo, remote, token=token(), local_dir=str(cache)))
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(got), str(local))
        local.with_suffix(".json").write_text(json.dumps(remote_meta))
        pulled += 1
    for name in ("codec", "refiner", "air", "train", "eval"):
        if f"markers/{name}" in files and not (root / "markers" / name).exists():
            got = Path(hf_hub_download(args.hf_repo, f"markers/{name}", token=token(), local_dir=str(cache)))
            (root / "markers").mkdir(parents=True, exist_ok=True)
            shutil.move(str(got), str(root / "markers" / name))
    if "eval/summary.json" in files and not (root / "eval" / "summary.json").exists():
        got = Path(hf_hub_download(args.hf_repo, "eval/summary.json", token=token(), local_dir=str(cache)))
        (root / "eval").mkdir(parents=True, exist_ok=True)
        shutil.move(str(got), str(root / "eval" / "summary.json"))
    log(f"hub: {pulled} checkpoint(s) pulled from https://huggingface.co/{args.hf_repo}")


def model_card(root: Path, args) -> str:
    rows = []
    for stage in STAGES:
        marker = root / "markers" / stage
        rows.append(f"| {stage} | {'done ' + marker.read_text().strip() if marker.exists() else 'pending'} |")
    checkpoints = []
    for local, remote in checkpoint_files(root, args.size):
        if local.exists():
            meta = checkpoint_meta(local)
            size_gb = local.stat().st_size / 1e9
            checkpoints.append(
                f"| `{remote}` | {meta.get('step', 0):,} | {meta.get('hours', 0.0):.1f} h | {size_gb:.2f} GB |"
            )
    summary = root / "prepared" / "summary.json"
    data = json.loads(summary.read_text()) if summary.exists() else {}
    evals = root / "eval" / "summary.json"
    results = json.loads(evals.read_text()) if evals.exists() else {}
    import helix_lm

    return "\n".join(
        [
            "---",
            "license: apache-2.0",
            "library_name: helix-lm",
            "pipeline_tag: audio-to-audio",
            "tags:",
            "- singing-voice-synthesis",
            "- svs",
            "- voice-conversion",
            "- zero-shot-voice-cloning",
            "- tristream",
            "- anansi",
            "- helix-lm",
            "---",
            "",
            "# TriStream-V2-SVS",
            "",
            f"TriStream-2 `{args.size}` singing voice synthesis and conversion, trained by the helix-lm",
            f"{helix_lm.__version__} molab program (`python -m helix_lm.scripts.tristream_molab`).",
            "This repository is the live state: pushed on a schedule while training runs and pulled back on",
            "restart, so a stopped run",
            "continues from here. Nothing here is a finished model until the `eval` row says done.",
            "",
            "## Status",
            "",
            "| stage | state |",
            "|---|---|",
            *rows,
            "",
            "## Checkpoints",
            "",
            "| file | steps | hours trained | size |",
            "|---|---|---|---|",
            *(checkpoints or ["| none yet | | | |"]),
            "",
            "## Data",
            "",
            f"Prepared: {data.get('hours', 0):.1f} h accepted over {data.get('accepted', 0)} clips",
            f"from {data.get('sources', [])}. Singing and speech are sampled by domain share (speech share annealed",
            f"{args.speech_share_start:.0%} -> {args.speech_share_end:.0%}), never by corpus size.",
            "Sources are the CC BY 4.0 sets in `helix music sources --commercial` (VocalSet, Dagstuhl ChoirSet,",
            "Emilia-YODAS speech, Slakh2100) plus the uploads folder; non-commercial research sets are kept out",
            "unless `--include-research` was given, and then only for evaluation.",
            "",
            "## Evaluation",
            "",
            "```json",
            json.dumps(results, indent=1) if results else "{}",
            "```",
            "",
            "## Use",
            "",
            "```bash",
            "pip install 'helix-lm[music]'",
            f"helix music swap song.wav reference.wav --out out.wav --root <folder with codec/ and runs/{args.size}/>",
            "```",
            "",
            "Clone only voices you own or have written permission for. Outputs pass through the fail-safe loop",
            "(docs/TRISTREAM.md section 12) and carry a report naming what was done.",
        ]
    )


def hub_push(args, force: bool = False, reason: str = "checkpoint") -> bool:
    """Push the checkpoints, markers, manifests, eval and the model card; throttled to --push-every-hours unless
    forced; never raises (a failed push must not stop a run); squashes the repo history so the Hub holds one
    copy of each file instead of every checkpoint ever pushed."""
    global LAST_PUSH
    if not args.hf_repo or not token():
        return False
    if not force and time.time() - LAST_PUSH < args.push_every_hours * 3600:
        return False
    try:
        from huggingface_hub import CommitOperationAdd

        api = hub_api()
        root = Path(args.root)
        try:
            api.create_repo(args.hf_repo, exist_ok=True, private=not args.public)
        except Exception as error:  # a token that may write but not create: the repo exists, carry on
            log(f"hub: create_repo skipped ({type(error).__name__})")
        ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=model_card(root, args).encode())]
        small = list((root / "markers").glob("*")) + list((root / "eval").glob("*.json"))
        small += list((root / "prepared").rglob("manifest.json")) + list((root / "prepared").glob("summary.json"))
        small += [p for p in (root / "data_found.json", root / "codec" / "audit.json") if p.exists()]
        for path in small:
            ops.append(CommitOperationAdd(path_in_repo=str(path.relative_to(root)), path_or_fileobj=str(path)))
        for local, remote in checkpoint_files(root, args.size):
            if local.exists():
                ops.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)))
                side = local.with_suffix(".json")
                if side.exists():
                    ops.append(CommitOperationAdd(path_in_repo=remote[:-3] + ".json", path_or_fileobj=str(side)))
        for attempt in range(3):
            try:
                api.create_commit(
                    args.hf_repo, operations=ops, commit_message=f"{reason} {time.strftime('%Y-%m-%d %H:%M')}"
                )
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


def mark(root: Path, stage: str) -> None:
    (root / "markers").mkdir(parents=True, exist_ok=True)
    (root / "markers" / stage).write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


# ------------------------------------------------------------------------------------------ setup --
def stage_setup(args) -> None:
    import torch

    root = Path(args.root)
    for sub in ("raw", "uploads", "prepared", "codec", "runs", "eval", "markers", "hf"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root / "hf"))
    if not torch.cuda.is_available():
        log("!! no GPU visible; the data stages run, training will not")
    else:
        props = torch.cuda.get_device_properties(0)
        log(f"GPU: {props.name}, {props.total_memory / 1e9:.0f} GB")
    for package in ("demucs", "soundfile", "huggingface_hub"):
        try:
            __import__(package)
        except ImportError:
            log(f"installing {package}")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=False)
    log(
        "token:",
        "present"
        if token()
        else "NOT SET (set HF_TOKEN or write /root/hf_token; downloads of gated sets and the push need it)",
    )
    mark(root, "setup")


# ------------------------------------------------------------------------------------------- data --
CLEAN_SOURCES = {
    "vocalset": {"zenodo": "1193957", "note": "20 pro singers, techniques, CC BY 4.0"},
    "dagstuhl-choirset": {"zenodo": "4607766", "note": "choir close mics with F0, CC BY 4.0"},
    "esmuc": {"zenodo": "5848990", "note": "12 singers, F0 + notes, CC BY 4.0"},
    "slakh2100": {"zenodo": "4599666", "note": "synthesized instrumentals for augmentation, CC BY 4.0"},
}
RESEARCH_SOURCES = {
    "gtsinger": "AaronZ345/GTSinger",  # CC BY-NC-SA: evaluation and research only
}


def zenodo_files(record: str) -> list[dict]:
    import urllib.request

    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=60) as response:
        payload = json.load(response)
    return [{"name": f["key"], "url": f["links"]["self"], "size": f.get("size", 0)} for f in payload.get("files", [])]


def download(url: str, target: Path, log=log) -> Path:
    import urllib.request

    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    log(f"downloading {url} -> {target.name}")
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as handle:
        shutil.copyfileobj(response, handle, length=1 << 20)
    tmp.rename(target)
    return target


def stage_data(args) -> None:
    root = Path(args.root)
    raw = root / "raw"
    found = {}
    for name, info in CLEAN_SOURCES.items():
        if name in args.skip:
            continue
        try:
            files = zenodo_files(info["zenodo"])
            kept = [
                f
                for f in files
                if f["name"].lower().endswith((".zip", ".tar.gz", ".tgz", ".wav", ".flac", ".csv", ".json"))
            ]
            gb = sum(f["size"] for f in kept) / 1e9
            log(f"{name}: {len(kept)} files, {gb:.1f} GB ({info['note']})")
            if gb > args.max_gb:
                log(f"  skipped: over --max-gb {args.max_gb}")
                continue
            for f in kept:
                download(f["url"], raw / name / f["name"])
            extract_archives(raw / name)
            found[name] = len(kept)
        except Exception as error:
            log(f"{name}: not fetched ({type(error).__name__}: {str(error)[:80]}); continue without it")
    if args.emilia_hours > 0 and token():
        try:
            from huggingface_hub import snapshot_download

            # Emilia-YODAS is the CC BY 4.0 half (1,362 EN archives named EN-B000000.tar ...; about 50 h each):
            # fetch exactly the first N, by name
            patterns = [f"Emilia-YODAS/EN/EN-B{i:06d}.tar" for i in range(max(1, int(args.emilia_hours / 50)))]
            snapshot_download(
                "amphion/Emilia-Dataset",
                repo_type="dataset",
                allow_patterns=patterns,
                local_dir=str(raw / "emilia-yodas"),
                token=token(),
            )
            found["emilia-yodas"] = len(list((raw / "emilia-yodas").rglob("*.tar")))
            log(f"emilia-yodas: {found['emilia-yodas']} archives (CC BY 4.0 half only)")
            extract_archives(raw / "emilia-yodas", delete=True)  # the tar is gone once its clips are out
        except Exception as error:
            log(f"emilia-yodas: not fetched ({type(error).__name__}: {str(error)[:80]})")
    if args.include_research and token():
        for name, repo in RESEARCH_SOURCES.items():
            try:
                from huggingface_hub import snapshot_download

                snapshot_download(repo, repo_type="dataset", local_dir=str(raw / name), token=token(), max_workers=4)
                found[name] = 1
                log(f"{name}: fetched for EVALUATION ONLY (non-commercial licence)")
            except Exception as error:
                log(f"{name}: not fetched ({type(error).__name__})")
    uploads = list((root / "uploads").rglob("*"))
    log(f"uploads folder: {sum(p.suffix.lower() in AUDIO_SUFFIXES for p in uploads)} audio files")
    (root / "markers").mkdir(exist_ok=True)
    (root / "data_found.json").write_text(json.dumps(found, indent=1))
    mark(root, "data")


# ---------------------------------------------------------------------------------------- prepare --
def emilia_speaker(path: Path) -> str:
    """The speaker id of an Emilia clip: from the JSON beside it, else the ``EN_B000000_S00000`` prefix."""
    side = path.with_suffix(".json")
    if side.exists():
        try:
            data = json.loads(side.read_text())
            if isinstance(data, dict) and data.get("speaker"):
                return str(data["speaker"])
        except Exception:
            pass
    match = re.match(r"([A-Z]{2}_B\d+_S\d+)", path.stem)
    return match.group(1) if match else path.parent.name


def stage_prepare(args) -> None:
    import torch

    from helix_lm.music.data import find_audio, prepare_uploads
    from helix_lm.music.separation import try_pretrained

    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    separate = try_pretrained("htdemucs")
    log("separator:", "HT-Demucs (MIT)" if separate else "none (install demucs); full songs will be rejected")
    log(f"measurements on {device}, {args.prepare_workers} workers")
    manifests = {}
    sources = [
        ("uploads", args.uploads_are_mixtures, args.upload_accent, True, None, "singing"),
        ("raw/vocalset", False, "unknown", True, lambda p: "vocalset/" + p.stem.split("_")[0], "singing"),
        ("raw/dagstuhl-choirset", False, "unknown", True, None, "choir"),
        ("raw/esmuc", False, "unknown", True, None, "choir"),
        ("raw/emilia-yodas", False, "unknown", False, emilia_speaker, "speech"),  # speech: no song analysis
    ]
    for folder, is_mixture, accent, analysis, singer_of, domain in sources:
        path = root / folder
        if not path.exists() or not find_audio([path]):
            continue
        out = root / "prepared" / folder.replace("/", "_")
        if (out / "manifest.json").exists() and not args.force:
            manifests[folder] = json.loads((out / "manifest.json").read_text())
            continue
        log(f"preparing {folder}: {len(find_audio([path]))} files")
        manifests[folder] = prepare_uploads(
            [path],
            out,
            sample_rate=args.sample_rate,
            singer=folder,
            accent=accent,
            mixtures=is_mixture,
            separate=separate,
            log=log if folder == "uploads" else (lambda *_: None),
            device=device,
            analysis=analysis,
            workers=args.prepare_workers,
            singer_of=singer_of,
            domain=domain,
        )
        log(f"  {folder}: {manifests[folder]['accepted']} clips accepted, {manifests[folder]['hours']:.1f} h")
    hours = sum(m.get("hours", 0) for m in manifests.values())
    accepted = sum(m.get("accepted", 0) for m in manifests.values())
    all_gaps = [g for m in manifests.values() for g in m.get("gaps", [])]
    log(f"prepared: {accepted} clips accepted, {hours:.1f} h; gaps: {all_gaps[:6]}")
    (root / "prepared" / "summary.json").write_text(
        json.dumps({"hours": hours, "accepted": accepted, "sources": list(manifests)}, indent=1)
    )
    mark(root, "prepare")


# ------------------------------------------------------------------------------------------ codec --
def stage_codec(args) -> None:
    import torch

    from helix_lm.audio import AudioCodec, CodecConfig, codec_losses
    from helix_lm.audio.codec import MultiPeriodDiscriminator, discriminator_loss
    from helix_lm.audio.data import read_wave
    from helix_lm.music.data import find_audio

    root = Path(args.root)
    if args.codec_hours <= 0:
        log("codec stage skipped (--codec-hours 0)")
        mark(root, "codec")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = CodecConfig.at_frame_rate(args.codec_rate, sample_rate=args.sample_rate)
    codec, disc = AudioCodec(config).to(device), MultiPeriodDiscriminator().to(device)
    files = find_audio([root / "prepared"])
    files = [f for f in files if f.name.endswith(".vocal.wav") or f.name.endswith(".instrumental.wav")]
    if not files:
        log("!! no prepared audio; run the prepare stage")
        return
    (root / "codec").mkdir(parents=True, exist_ok=True)  # every stage makes its own folder: no order dependence
    ckpt = root / "codec" / "codec.pt"
    step = 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        codec.load_state_dict(state["codec"])
        disc.load_state_dict(state["disc"])
        step = state["step"]
    opt_g = torch.optim.AdamW(codec.parameters(), lr=1e-4, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-4, betas=(0.8, 0.99))
    budget = Budget(args.codec_hours, checkpoint_meta(ckpt).get("hours", 0.0) if ckpt.exists() else 0.0)
    crop = int(1.5 * args.sample_rate)
    g = torch.Generator().manual_seed(step)
    batch_size = args.batch if args.batch > 0 else (96 if device.type == "cuda" else 4)
    speech_files = [f for f in files if "emilia" in str(f)]
    sung_files = [f for f in files if "emilia" not in str(f)] or files
    log(
        f"codec: {len(sung_files)} sung + {len(speech_files)} spoken files (drawn half and half), "
        f"{budget.remaining:.1f} h left of {args.codec_hours} h, resuming at step {step}"
    )

    # the high register must not starve: clips whose F0 reaches 500 Hz get a fixed share of every batch
    high_files = [
        Path(r["dir"]) / f"{Path(r['path']).stem}.vocal.wav"
        for r in prepared_records(root)
        if r["tags"].get("f0_high", 0) >= 500
    ]
    high_files = [f for f in high_files if f.exists()]
    low_files = [f for f in sung_files if f not in set(high_files)] or sung_files
    log(f"codec: {len(high_files)} high-register files get {args.high_register_share:.0%} of each batch")
    from helix_lm.audio.codec import codebook_usage_by_register
    from helix_lm.audio.refiner import high_band_loss
    from helix_lm.music.regularize import augment

    def pick_file() -> Path:
        draw = float(torch.rand((), generator=g))
        if high_files and draw < args.high_register_share:
            pool = high_files
        elif speech_files and draw < args.high_register_share + 0.5 * (1 - args.high_register_share):
            pool = speech_files
        else:
            pool = sung_files
        return pool[int(torch.randint(0, len(pool), (), generator=g))]

    def crop_of(path: Path, length: int) -> torch.Tensor:
        wave = read_wave(path, args.sample_rate)
        if wave.shape[-1] < length:
            wave = torch.nn.functional.pad(wave, (0, length - wave.shape[-1]))
        start = int(torch.randint(0, wave.shape[-1] - length + 1, (), generator=g))
        return wave[start : start + length]

    def make_batch() -> torch.Tensor:
        batch = []
        for _ in range(batch_size):
            wave = crop_of(pick_file(), crop)
            if float(torch.rand((), generator=g)) < args.augment_prob:  # rooms, EQ, noise: "clean" is not a domain
                wave, _ = augment(wave, args.sample_rate, seed=int(torch.randint(0, 1 << 30, (), generator=g)))
            batch.append(wave)
        return torch.stack(batch)[:, None]

    audit_high = torch.stack([crop_of(f, crop) for f in high_files[:16]]) if high_files else None
    audit_low = torch.stack([crop_of(f, crop) for f in low_files[:16]]) if high_files else None
    prefetch = Prefetcher(make_batch)
    last_save = time.time()
    while not budget.over() and not (root / "stop").exists():
        wave = prefetch.get().to(device, non_blocking=True)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = codec(wave)
                terms = codec_losses(
                    out,
                    wave,
                    config,
                    discriminator=disc if step > 5000 else None,
                    time_weight=1.0 if step < 20000 else 0.1,
                )
                terms["high_band"] = 2.0 * high_band_loss(out["recon"].float(), wave.float(), args.sample_rate)
                terms["total"] = terms["total"] + terms["high_band"]
            opt_g.zero_grad()
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 10.0)
            opt_g.step()
            if step > 5000:
                d_loss = discriminator_loss(disc, out["recon"].float(), wave)
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
        except torch.cuda.OutOfMemoryError:
            opt_g.zero_grad(set_to_none=True)
            opt_d.zero_grad(set_to_none=True)
            del wave
            torch.cuda.empty_cache()
            batch_size = max(2, batch_size // 2)
            prefetch.drain()
            log(f"codec out of memory: batch halved to {batch_size}")
            continue
        step += 1
        if step % 100 == 0:
            log(
                f"codec step {step}: mel {terms['mel'].item():.3f} high-band {terms['high_band'].item():.3f} "
                f"commit {terms['commitment'].item():.4f} usage {codec.quantizer.semantic.usage():.0%}"
            )
        if step % 2000 == 0 and audit_high is not None:
            codec.eval()
            audit = codebook_usage_by_register(codec, audit_high.to(device), audit_low.to(device))
            codec.train()
            (root / "codec" / "audit.json").write_text(json.dumps({"step": step, **audit}, indent=1))
            log(
                "codebook audit: high-register "
                + " ".join(f"{u:.0%}" for u in audit["high"])
                + " | low-register "
                + " ".join(f"{u:.0%}" for u in audit["low"])
                + (" !! upper tiers starved on the high register" if audit["starved"] else " (ok)")
            )
        if step % 2000 == 0 or time.time() - last_save > 1800:
            save_checkpoint(
                {
                    "codec": codec.state_dict(),
                    "disc": disc.state_dict(),
                    "step": step,
                    "config": config.__dict__,
                    "hours": budget.elapsed,
                },
                ckpt,
            )
            last_save = time.time()
            hub_push(args, reason=f"codec step {step}")
    prefetch.close()
    save_checkpoint(
        {
            "codec": codec.state_dict(),
            "disc": disc.state_dict(),
            "step": step,
            "config": config.__dict__,
            "hours": budget.elapsed,
        },
        ckpt,
    )
    log(f"codec saved at step {step}")
    mark(root, "codec")


# ------------------------------------------------------------------------------------------ train --
def load_codec(root: Path, device):
    import torch

    from helix_lm.audio import AudioCodec, CodecConfig

    ckpt = root / "codec" / "codec.pt"
    if not ckpt.exists():
        return None
    state = torch.load(ckpt, map_location=device)
    codec = AudioCodec(CodecConfig(**state["config"])).to(device).eval()
    codec.load_state_dict(state["codec"])
    return codec


# ----------------------------------------------------------------------------------------- refiner --
def load_refiner(root: Path, device):
    import torch

    from helix_lm.audio.refiner import HarmonicNoiseRefiner, RefinerConfig

    ckpt = root / "refiner" / "refiner.pt"
    if not ckpt.exists():
        return None
    state = torch.load(ckpt, map_location=device)
    refiner = HarmonicNoiseRefiner(RefinerConfig(**state["config"])).to(device).eval()
    refiner.load_state_dict(state["refiner"])
    return refiner


def stage_refiner(args) -> None:
    """The refiner learns to put back what the codec averages away: trained on (codec reconstruction -> original)
    pairs of the prepared dry vocals, with the multi-period discriminator after the first 2,000 steps."""
    import torch

    from helix_lm.audio.codec import MultiPeriodDiscriminator, discriminator_loss
    from helix_lm.audio.data import read_wave
    from helix_lm.audio.refiner import HarmonicNoiseRefiner, RefinerConfig, refiner_losses
    from helix_lm.audio.singing import estimate_f0

    root = Path(args.root)
    if args.refiner_hours <= 0:
        log("refiner stage skipped (--refiner-hours 0)")
        mark(root, "refiner")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = load_codec(root, device)
    if codec is None:
        log("!! no codec (run the codec stage first)")
        return
    files = sorted((root / "prepared").rglob("*.vocal.wav"))
    if not files:
        log("!! no prepared vocals")
        return
    (root / "refiner").mkdir(parents=True, exist_ok=True)
    ckpt = root / "refiner" / "refiner.pt"
    config = RefinerConfig(
        sample_rate=args.sample_rate, frame_dim=codec.config.dim, frame_rate=codec.config.frame_rate
    )
    refiner = HarmonicNoiseRefiner(config).to(device)
    disc = MultiPeriodDiscriminator().to(device)
    opt = torch.optim.AdamW(refiner.parameters(), lr=2e-4, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=2e-4, betas=(0.8, 0.99))
    step = 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        refiner.load_state_dict(state["refiner"])
        disc.load_state_dict(state["disc"])
        step = state["step"]
        log(f"refiner resumed at step {step}")
    log(f"refiner: {refiner.num_parameters() / 1e6:.1f}M parameters, {len(files)} clips, {args.refiner_hours:.0f} h")
    budget = Budget(args.refiner_hours, checkpoint_meta(ckpt).get("hours", 0.0) if ckpt.exists() else 0.0)
    g = torch.Generator().manual_seed(step)
    batch_size = max(4, args.batch if args.batch > 0 else (48 if device.type == "cuda" else 4))
    need = int(2.0 * args.sample_rate)
    hop_fine = config.hop
    tf = need // hop_fine
    need = tf * hop_fine
    last_save = time.time()

    def save() -> None:
        save_checkpoint(
            {
                "refiner": refiner.state_dict(),
                "disc": disc.state_dict(),
                "step": step,
                "config": config.__dict__,
                "hours": budget.elapsed,
            },
            ckpt,
        )

    from helix_lm.music.regularize import augment

    def make_batch() -> torch.Tensor:
        waves = []
        for _ in range(batch_size):
            wave = read_wave(files[int(torch.randint(0, len(files), (), generator=g))], args.sample_rate)
            if wave.shape[-1] < need:
                wave = torch.nn.functional.pad(wave, (0, need - wave.shape[-1]))
            start = int(torch.randint(0, wave.shape[-1] - need + 1, (), generator=g))
            wave = wave[start : start + need]
            if float(torch.rand((), generator=g)) < args.augment_prob:
                wave, _ = augment(wave, args.sample_rate, seed=int(torch.randint(0, 1 << 30, (), generator=g)))
            waves.append(wave)
        return torch.stack(waves)

    prefetch = Prefetcher(make_batch)
    while not budget.over() and not (root / "stop").exists():
        wave = prefetch.get().to(device, non_blocking=True)
        try:
            with torch.no_grad():
                frames = codec.listen(wave)
                recon = codec.decode_frames(frames)[:, :, :need]
                f0_fine = estimate_f0(wave.cpu(), args.sample_rate, frame_rate=config.fine_rate).to(device)[:, :tf]
                f0_fine = torch.nn.functional.pad(f0_fine, (0, tf - f0_fine.shape[-1]))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = refiner(recon, frames, f0_fine)
                terms = refiner_losses(out, wave, args.sample_rate, disc if step > 2000 else None)
            opt.zero_grad()
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            opt.step()
            if step > 2000:
                opt_d.zero_grad()
                discriminator_loss(disc, out["wave"].detach(), wave[:, None]).backward()
                opt_d.step()
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            batch_size = max(2, batch_size // 2)
            prefetch.drain()
            log(f"refiner out of memory: batch halved to {batch_size}")
            continue
        step += 1
        if step % 50 == 0:
            log(f"refiner step {step}: " + " ".join(f"{k} {float(v.detach()):.3f}" for k, v in terms.items()))
        if time.time() - last_save > 1800:
            save()
            last_save = time.time()
            hub_push(args, reason=f"refiner step {step}")
    prefetch.close()
    save()
    log(f"refiner saved at step {step}")
    mark(root, "refiner")


# --------------------------------------------------------------------------------------------- air --
def stage_air(args) -> None:
    """The air band above the codec's Nyquist, learned from your own recordings at 44.1/48 kHz (the uploads
    folder, read at 48 kHz). Opt in with --air-hours; without 48 kHz sources there is nothing to learn from."""
    import torch

    from helix_lm.audio.data import read_wave
    from helix_lm.audio.refiner import AirBandExtension, AirConfig, air_losses
    from helix_lm.music.data import find_audio

    root = Path(args.root)
    if args.air_hours <= 0:
        log("air stage skipped (--air-hours 0)")
        mark(root, "air")
        return
    files = find_audio([root / "uploads"])
    if not files:
        log("!! no uploads to learn the air band from")
        mark(root, "air")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (root / "air").mkdir(parents=True, exist_ok=True)
    ckpt = root / "air" / "air.pt"
    config = AirConfig(in_rate=args.sample_rate, out_rate=48000)
    air = AirBandExtension(config).to(device)
    opt = torch.optim.AdamW(air.parameters(), lr=3e-4)
    step = 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        air.load_state_dict(state["air"])
        step = state["step"]
    budget = Budget(args.air_hours, checkpoint_meta(ckpt).get("hours", 0.0) if ckpt.exists() else 0.0)
    g = torch.Generator().manual_seed(step)
    need = int(2.0 * 48000)
    ratio = 48000 // args.sample_rate
    batch_size = max(2, (args.batch if args.batch > 0 else 16) // 2)
    while not budget.over() and not (root / "stop").exists():
        waves = []
        for _ in range(batch_size):
            wave = read_wave(files[int(torch.randint(0, len(files), (), generator=g))], 48000)
            if wave.shape[-1] < need:
                wave = torch.nn.functional.pad(wave, (0, need - wave.shape[-1]))
            start = int(torch.randint(0, wave.shape[-1] - need + 1, (), generator=g))
            waves.append(wave[start : start + need])
        target = torch.stack(waves).to(device)
        body = torch.nn.functional.avg_pool1d(target[:, None], ratio, stride=ratio)  # the 24 kHz body of the take
        out = air(body)
        terms = air_losses(out, target, 48000, cutoff=args.sample_rate / 2 * 0.95)
        opt.zero_grad()
        terms["total"].backward()
        opt.step()
        step += 1
        if step % 50 == 0:
            log(f"air step {step}: " + " ".join(f"{k} {float(v.detach()):.3f}" for k, v in terms.items()))
        if step % 500 == 0:
            save_checkpoint(
                {"air": air.state_dict(), "step": step, "config": config.__dict__, "hours": budget.elapsed}, ckpt
            )
            hub_push(args, reason=f"air step {step}")
    save_checkpoint({"air": air.state_dict(), "step": step, "config": config.__dict__, "hours": budget.elapsed}, ckpt)
    log(f"air band saved at step {step}")
    mark(root, "air")


def stage_train(args) -> None:
    import torch

    from helix_lm.audio import SpeakerEncoder
    from helix_lm.audio.data import read_wave
    from helix_lm.music.regularize import ACCENTS, ClipTags, augment, mixture_weights, speech_share_at, split_singers
    from helix_lm.music.tristream import TriStreamConfig, TriStreamForSinging, noise_features, source_features
    from helix_lm.optim import build_optimizer

    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = load_codec(root, device)
    if codec is None:
        log("!! no codec (run the codec stage, or point --codec at a saved one)")
        return
    if args.batch <= 0:
        args.batch = 64  # auto: the largest batch that fits is found by halving on out-of-memory
        log("auto batch size: starting at 64 clips and halving on out-of-memory until it fits")
    records = []
    for manifest in (root / "prepared").rglob("manifest.json"):
        data = json.loads(manifest.read_text())
        records += [{**clip, "dir": str(manifest.parent)} for clip in data["clips"] if clip["accepted"]]
    if not records:
        log("!! no accepted clips")
        return
    tags = [ClipTags(**r["tags"]) for r in records]
    split = split_singers([t.singer for t in tags])
    train_idx = [i for i, t in enumerate(tags) if split[t.singer] == "train"]
    train_tags = [tags[i] for i in train_idx]
    sung = sum(t.domain != "speech" for t in train_tags)
    log(
        f"domains: {sung} sung clips, {len(train_tags) - sung} spoken clips; speech share annealed "
        f"{args.speech_share_start:.0%} -> {args.speech_share_end:.0%} of each batch, spoken loss weight "
        f"{args.speech_loss_weight}"
    )
    weights = mixture_weights(train_tags, speech_share_at(0.0, args.speech_share_start, args.speech_share_end))
    config = TriStreamConfig.named(
        args.size,
        frame_dim=codec.config.dim,
        sample_rate=args.sample_rate,
        semantic_codebook=codec.config.codebook_size,  # the content codes come from this codec
        frame_rate=codec.config.frame_rate,
    )
    model = TriStreamForSinging(config).to(device)
    speaker_encoder = SpeakerEncoder(frame_dim=codec.config.dim, embedding_dim=config.speaker_dim).to(device)
    params = list(model.parameters()) + list(speaker_encoder.parameters())
    optimizer = build_optimizer(torch.nn.ModuleList([model, speaker_encoder]), "muon", lr=args.lr, weight_decay=0.05)
    run = root / "runs" / args.size
    run.mkdir(parents=True, exist_ok=True)
    step = 0
    ckpt = run / "latest.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        speaker_encoder.load_state_dict(state["speaker"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        log(f"resumed at step {step}")
    log(
        f"TriStream-2 {args.size}: {model.num_parameters() / 1e6:.0f}M parameters, {len(train_idx)} training clips, "
        f"{len(records) - len(train_idx)} validation clips (singer-disjoint)"
    )
    budget = Budget(args.train_hours, checkpoint_meta(ckpt).get("hours", 0.0) if ckpt.exists() else 0.0)
    log(f"train: {budget.remaining:.1f} h left of {args.train_hours} h")
    frames_per_clip = int(args.clip_seconds * config.frame_rate)
    g = torch.Generator().manual_seed(step)
    singer_ids = {t.singer: i % config.num_singers for i, t in enumerate(tags)}
    last_save = time.time()

    def make_batch():
        chosen = torch.multinomial(weights, args.batch, replacement=True, generator=g).tolist()
        batch_rows = [records[train_idx[i]] for i in chosen]
        waves, f0s, accents, singers = [], [], [], []
        loss_weights = [
            1.0 if r["tags"].get("domain", "singing") != "speech" else args.speech_loss_weight for r in batch_rows
        ]
        for r in batch_rows:
            stem = Path(r["path"]).stem
            wave = read_wave(Path(r["dir"]) / f"{stem}.vocal.wav", args.sample_rate)
            f0 = torch.load(Path(r["dir"]) / f"{stem}.f0.pt")
            need = int(args.clip_seconds * args.sample_rate)
            if wave.shape[-1] < need:
                wave = torch.nn.functional.pad(wave, (0, need - wave.shape[-1]))
                f0 = torch.nn.functional.pad(f0, (0, max(0, frames_per_clip - f0.shape[-1])))
            start = int(torch.randint(0, wave.shape[-1] - need + 1, (), generator=g))
            f_start = int(start / args.sample_rate * config.frame_rate)
            waves.append(wave[start : start + need])
            f0s.append(f0[f_start : f_start + frames_per_clip])
            accents.append(ACCENTS.index(r["tags"]["accent"]) if r["tags"]["accent"] in ACCENTS else 0)
            singers.append(singer_ids[r["tags"]["singer"]])
        clean = torch.stack(waves)
        dirty = (
            clean.clone()
        )  # what the model hears: the dry take through rooms, EQ, noise, holes; the target stays dry
        for i in range(clean.shape[0]):
            if float(torch.rand((), generator=g)) < args.augment_prob:
                dirty[i], _ = augment(clean[i], args.sample_rate, seed=int(torch.randint(0, 1 << 30, (), generator=g)))
        return clean, dirty, f0s, accents, singers, len(batch_rows), torch.tensor(loss_weights)

    prefetch = Prefetcher(make_batch)
    last_share = -1.0
    while not budget.over() and not (root / "stop").exists():
        share = speech_share_at(
            budget.elapsed / max(budget.hours, 1e-6), args.speech_share_start, args.speech_share_end
        )
        if abs(share - last_share) >= 0.05:  # re-derive the sampler as the curriculum moves
            weights = mixture_weights(train_tags, share)
            last_share = share
            log(f"curriculum: speech share now {share:.0%}")
        wave, dirty, f0s, accents, singers, rows, loss_weights = prefetch.get()
        wave = wave.to(device, non_blocking=True)
        dirty = dirty.to(device, non_blocking=True)
        with torch.no_grad():
            frames = codec.listen(wave)[:, :frames_per_clip]
            frames_in = codec.listen(dirty)[:, :frames_per_clip]
        t = frames.shape[1]
        f0 = torch.stack([torch.nn.functional.pad(f, (0, max(0, t - f.shape[0])))[:t] for f in f0s]).to(device)
        energy = frames.pow(2).mean(-1)
        speaker = speaker_encoder(frames)  # the clip's own voice: the model learns to copy it (in-context cloning)
        batch = {
            "source": source_features(f0, energy), "noise": noise_features(frames_in), "frames": frames_in,
            "target_frames": frames,
            "phonemes": torch.full((rows, 1), -1, dtype=torch.long, device=device),
            "speaker": speaker, "singer": torch.tensor(singers, device=device),
            "accent": torch.tensor(accents, device=device), "f0_hz": f0, "weights": loss_weights.to(device),
        }  # fmt: skip
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                losses = model(batch)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        except torch.cuda.OutOfMemoryError:
            # pushing the 96 GB to the edge: start big, halve on the first out-of-memory, keep going
            optimizer.zero_grad(set_to_none=True)
            del batch, frames
            torch.cuda.empty_cache()
            args.batch = max(1, args.batch // 2)
            prefetch.drain()
            log(f"out of memory: batch halved to {args.batch} clips of {args.clip_seconds:.0f} s")
            continue
        step += 1
        if step % 20 == 0:
            log(
                f"step {step}: "
                + " ".join(f"{k} {float(v.detach()):.3f}" for k, v in losses.items() if k != "mask_fraction")
            )
        if time.time() - last_save > 1800:
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "speaker": speaker_encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": config.__dict__,
                    "hours": budget.elapsed,
                },
                ckpt,
            )
            last_save = time.time()
            log(f"checkpoint at step {step} ({budget.elapsed:.1f} h trained)")
            hub_push(args, reason=f"train step {step}")
    prefetch.close()
    save_checkpoint(
        {
            "model": model.state_dict(),
            "speaker": speaker_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config.__dict__,
            "hours": budget.elapsed,
        },
        ckpt,
    )
    log(f"saved at step {step}")
    mark(root, "train")


# ------------------------------------------------------------------------------------------- eval --
def stage_eval(args) -> None:
    import torch

    from helix_lm.audio import SpeakerEncoder
    from helix_lm.audio.data import read_wave
    from helix_lm.audio.singing import estimate_f0
    from helix_lm.audio.speaker import speaker_similarity
    from helix_lm.music.failsafe import Gates, synthesize_with_gates
    from helix_lm.music.register import register_break_report
    from helix_lm.music.regularize import ClipTags, split_singers
    from helix_lm.music.tristream import TriStreamConfig, TriStreamForSinging, noise_features, source_features

    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = load_codec(root, device)
    refiner = load_refiner(root, device)
    ckpt = root / "runs" / args.size / "latest.pt"
    if codec is None or not ckpt.exists():
        log("!! nothing to evaluate yet")
        return
    state = torch.load(ckpt, map_location=device)
    config = TriStreamConfig(**state["config"])
    model = TriStreamForSinging(config).to(device).eval()
    model.load_state_dict(state["model"])
    speaker_encoder = SpeakerEncoder(frame_dim=codec.config.dim, embedding_dim=config.speaker_dim).to(device).eval()
    speaker_encoder.load_state_dict(state["speaker"])
    records = []
    for manifest in (root / "prepared").rglob("manifest.json"):
        data = json.loads(manifest.read_text())
        records += [{**clip, "dir": str(manifest.parent)} for clip in data["clips"] if clip["accepted"]]
    tags = [ClipTags(**r["tags"]) for r in records]
    split = split_singers([t.singer for t in tags])
    paired = list(zip(records, tags, strict=True))
    sung = [(r, t) for r, t in paired if t.domain != "speech"] or paired
    held = [r for r, t in sung if split[t.singer] == "validation"][: args.eval_clips]
    if not held and sung:
        log("no validation singers yet (too few singers for the split): evaluating on training clips, optimistic")
        held = [r for r, _ in sung][: args.eval_clips]
    results = {
        "clips": len(held), "f0_rmse_cents": [], "speaker_similarity": [], "glitch_rate": [], "buzz_db": [],
        "hnr_db": [], "sync_lag_ms": [], "robotic": [], "register_pops_per_minute": [], "rounds": [], "fallbacks": [],
        "passed": [],
    }  # fmt: skip
    gates = Gates(steps=config.flow_steps, min_similarity=None, frame_rate=config.frame_rate)
    for r in held:
        stem = Path(r["path"]).stem
        wave = read_wave(Path(r["dir"]) / f"{stem}.vocal.wav", args.sample_rate)[: int(10 * args.sample_rate)].to(
            device
        )
        with torch.no_grad():
            frames = codec.listen(wave[None])
            f0 = estimate_f0(wave[None].cpu(), args.sample_rate, config.frame_rate)[:, : frames.shape[1]].to(device)
            batch = {
                "source": source_features(f0, frames.pow(2).mean(-1)),
                "noise": noise_features(frames),
                "frames": frames,
                "phonemes": torch.full((1, 1), -1, dtype=torch.long, device=device),
                "speaker": speaker_encoder(frames),
            }
            audio, report = synthesize_with_gates(
                model, codec, batch, args.sample_rate, f0[0], refiner=refiner, source_wave=wave, gates=gates
            )
            audio = audio.cpu()
        final = report.final
        results["f0_rmse_cents"].append(final["f0_rmse_cents"])
        with torch.no_grad():
            sim = speaker_similarity(speaker_encoder(codec.listen(audio[None].to(device))), speaker_encoder(frames))
        results["speaker_similarity"].append(float(sim))
        glitches = final["glitches"]
        results["glitch_rate"].append(
            float(glitches["clicks"].numel() + glitches["dropouts"].numel())
            / max(audio.shape[-1] / args.sample_rate, 1e-6)
        )
        results["buzz_db"].append(final["buzz"]["buzz_db"])
        results["hnr_db"].append(final.get("coherence", {}).get("hnr_db", 0.0))
        results["sync_lag_ms"].append(abs(final.get("sync", {}).get("lag_ms", 0.0)))
        results["robotic"].append(final.get("robotic", {}).get("score", 0.0))
        top = float(r["tags"].get("f0_high", 0.0)) * 0.85  # the clip's own upper register
        if top > 0:
            pops = register_break_report(audio, f0[0].cpu(), args.sample_rate, top, config.frame_rate)
            results["register_pops_per_minute"].append(pops["pops_per_minute"])
        results["rounds"].append(len(report.rounds))
        results["fallbacks"].append(float(any("sketch" in a or "transplant" in a for a in report.applied)))
        results["passed"].append(float(report.ok))
    summary = {k: (sum(v) / len(v) if isinstance(v, list) and v else v) for k, v in results.items()}
    (root / "eval").mkdir(parents=True, exist_ok=True)
    (root / "eval" / "summary.json").write_text(json.dumps(summary, indent=1))
    log("eval:", summary)
    mark(root, "eval")


# ------------------------------------------------------------------------------------------- push --
def stage_push(args, quiet: bool = False) -> None:
    if not args.hf_repo or not token():
        log("push skipped: --hf-repo and a token are needed")
        return
    if hub_push(args, force=True, reason="push stage") and not quiet:
        mark(Path(args.root), "push")


# -------------------------------------------------------------------------------------------- eta --
def stage_eta(args) -> None:
    """What the budgets add up to on one GPU, and where the time goes."""
    emilia_shards = max(1, int(args.emilia_hours / 50)) if args.emilia_hours > 0 else 0
    download_gb = 10 + emilia_shards * 2.5 + (104 if "slakh2100" not in args.skip and args.max_gb >= 105 else 0)
    download_h = download_gb / (50 * 3.6)  # 50 MB/s
    prepare_h = (
        args.emilia_hours * 360 / 20 + 15 * 360 / 10
    ) / 3600  # ~20 speech clips/s on a GPU, 10 singing clips/s
    rows = [
        ("data (download + extract)", download_h, f"about {download_gb:.0f} GB at 50 MB/s"),
        ("prepare (measurements on the GPU)", prepare_h, f"{args.emilia_hours:.0f} h speech + the singing sets"),
        ("codec", args.codec_hours, "the heaviest stage: the waveform codec at 24 kHz with its discriminator"),
        ("refiner", args.refiner_hours, "phase-true harmonics, 10 ms noise, residual"),
        ("air", args.air_hours, "only with 48 kHz uploads"),
        ("train TriStream-2", args.train_hours, f"{args.size}, auto batch, Muon"),
        ("eval + push", 1.0, "held-out singers through the fail-safe loop"),
    ]
    total = sum(h for _, h, _ in rows)
    log("ETA on one GPU, by stage (hours):")
    for name, hours, note in rows:
        log(f"  {name:34} {hours:8.1f}   {note}")
    log(f"  {'total':34} {total:8.1f}   = {total / 24:.1f} days")
    log(f"  pushes every {args.push_every_hours:.0f} h to {args.hf_repo or '(no repo)'}")
    log("shorter plan: --codec-hours 72 --refiner-hours 24 --train-hours 72 is about 7.5 days")


# ------------------------------------------------------------------------------------------- swap --
def stage_swap(args) -> None:
    import torch

    from helix_lm.audio import SpeakerEncoder
    from helix_lm.audio.data import read_wave, write_wave
    from helix_lm.music.conversion import Converter, swap_voice
    from helix_lm.music.tristream import TriStreamConfig, TriStreamForSinging

    if not args.song or not args.reference:
        log("swap needs --song and --reference")
        return
    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = load_codec(root, device)
    converter = Converter()
    ckpt = root / "runs" / args.size / "latest.pt"
    if codec is not None and ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        config = TriStreamConfig(**state["config"])
        model = TriStreamForSinging(config).to(device).eval()
        model.load_state_dict(state["model"])
        speaker_encoder = (
            SpeakerEncoder(frame_dim=codec.config.dim, embedding_dim=config.speaker_dim).to(device).eval()
        )
        speaker_encoder.load_state_dict(state["speaker"])
        from helix_lm.music.failsafe import Gates

        converter = Converter(
            model, codec, speaker_encoder, refiner=load_refiner(root, device),
            gates=Gates(steps=config.flow_steps, frame_rate=config.frame_rate),
        )  # fmt: skip
    song = read_wave(args.song, args.sample_rate)
    reference = read_wave(args.reference, args.sample_rate)
    mix, report = swap_voice(song, reference, args.sample_rate, converter=converter)
    out = Path(args.out or (root / "swap.wav"))
    write_wave(out, mix, args.sample_rate)
    log(f"written {out}\n{report.summary()}")


# ------------------------------------------------------------------------------------------- main --
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TriStream-2 on one 96 GB GPU, stage by stage")
    parser.add_argument("--root", default="/workspace/tristream")
    parser.add_argument("--stage", default="all", help="all or one of: " + ", ".join(STAGES))
    parser.add_argument("--size", default="ts-765m", choices=["ts-tiny", "ts-300m", "ts-765m"])
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--hf-repo", default=DEFAULT_REPO, help="the Hub repo that mirrors the run (private)")
    parser.add_argument("--public", action="store_true", help="make the Hub repo public (private by default)")
    parser.add_argument("--push-every-hours", type=float, default=2.0, help="how often checkpoints go to the Hub")
    parser.add_argument("--no-pull", action="store_true", help="do not resume from the Hub's checkpoints")
    parser.add_argument("--no-squash", action="store_true", help="keep every pushed checkpoint in the Hub history")
    parser.add_argument("--retries", type=int, default=1000, help="all: restart a crashed stage this many times")
    parser.add_argument("--retry-wait", type=float, default=120.0, help="seconds between restarts")
    parser.add_argument("--prepare-workers", type=int, default=8, help="prepare: file threads")
    parser.add_argument(
        "--max-gb", type=float, default=50.0, help="data: skip a source bigger than this (Slakh2100 is 104 GB)"
    )
    parser.add_argument(
        "--emilia-hours", type=float, default=300.0, help="data: hours of Emilia-YODAS speech (CC BY) to fetch"
    )
    parser.add_argument("--speech-share-start", type=float, default=0.5, help="train: speech share of a batch, start")
    parser.add_argument("--speech-share-end", type=float, default=0.1, help="train: speech share of a batch, end")
    parser.add_argument("--speech-loss-weight", type=float, default=0.5, help="train: loss weight of a spoken clip")
    parser.add_argument("--augment-prob", type=float, default=0.5, help="rooms, EQ, noise, holes on a clip's input")
    parser.add_argument("--high-register-share", type=float, default=0.25, help="codec: share of high-F0 clips")
    parser.add_argument(
        "--include-research", action="store_true", help="data: also fetch non-commercial sets for evaluation"
    )
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument(
        "--uploads-are-mixtures", action="store_true", help="prepare: the uploads are full songs (separate them)"
    )
    parser.add_argument("--upload-accent", default="unknown")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--codec-hours", type=float, default=240.0)
    parser.add_argument("--codec-rate", type=int, default=25, choices=[25, 50, 75], help="codec frames a second")
    parser.add_argument("--refiner-hours", type=float, default=48.0)
    parser.add_argument("--air-hours", type=float, default=0.0, help="air band from 48 kHz uploads; 0 = skip")
    parser.add_argument("--train-hours", type=float, default=200.0)
    parser.add_argument("--clip-seconds", type=float, default=12.0)
    parser.add_argument("--batch", type=int, default=0, help="clips per step; 0 = auto (largest that fits)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-clips", type=int, default=50)
    parser.add_argument("--song", default="")
    parser.add_argument("--reference", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "eta":
        stage_eta(args)
        return 0
    if args.stage != "all" and args.stage not in STAGES:
        parser.error(f"unknown stage {args.stage}")
    gpu_max()
    hub_pull(args)
    if args.stage != "all":
        log(f"== stage {args.stage}")
        globals()[f"stage_{args.stage}"](args)
        return 0
    attempt = 0
    while True:
        try:
            for stage in STAGES[:-1]:
                if complete(root, stage, args) and stage != "push":
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
        except KeyboardInterrupt:
            log("interrupted: saving the state to the Hub")
            hub_push(args, force=True, reason="interrupted")
            return 130
        except Exception as error:
            attempt += 1
            log(f"!! stage failed ({type(error).__name__}: {str(error)[:120]}); attempt {attempt} of {args.retries}")
            traceback.print_exc()
            hub_push(args, force=True, reason=f"after failure {attempt}")
            if attempt >= args.retries:
                raise
            time.sleep(args.retry_wait)
            hub_pull(args)


if __name__ == "__main__":
    raise SystemExit(main())
