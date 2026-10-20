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
import shutil
import subprocess
import sys
import time
from pathlib import Path

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

STAGES = ("setup", "data", "prepare", "codec", "train", "eval", "push", "swap")


def log(*parts) -> None:
    print(time.strftime("[%H:%M:%S]"), *parts, flush=True)


def token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if not tok and Path("/root/hf_token").exists():
        tok = Path("/root/hf_token").read_text().strip()
    return tok or None


def done(root: Path, stage: str) -> bool:
    return (root / "markers" / stage).exists()


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
            found[name] = len(kept)
        except Exception as error:
            log(f"{name}: not fetched ({type(error).__name__}: {str(error)[:80]}); continue without it")
    if args.emilia_hours > 0 and token():
        try:
            from huggingface_hub import snapshot_download

            # Emilia-YODAS is the CC BY 4.0 half; the EN shard list is large, so take the first N archives only.
            patterns = [f"Emilia-YODAS/EN/*{i:03d}*" for i in range(max(1, int(args.emilia_hours / 50)))]
            snapshot_download(
                "amphion/Emilia-Dataset",
                repo_type="dataset",
                allow_patterns=patterns,
                local_dir=str(raw / "emilia-yodas"),
                token=token(),
            )
            found["emilia-yodas"] = len(list((raw / "emilia-yodas").rglob("*.tar")))
            log(f"emilia-yodas: {found['emilia-yodas']} archives (CC BY 4.0 half only)")
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
def stage_prepare(args) -> None:
    from helix_lm.music.data import find_audio, prepare_uploads
    from helix_lm.music.separation import try_pretrained

    root = Path(args.root)
    separate = try_pretrained("htdemucs")
    log("separator:", "HT-Demucs (MIT)" if separate else "none (install demucs); full songs will be rejected")
    manifests = {}
    for folder, is_mixture, accent in (
        ("uploads", args.uploads_are_mixtures, args.upload_accent),
        ("raw/vocalset", False, "unknown"),
        ("raw/dagstuhl-choirset", False, "unknown"),
        ("raw/esmuc", False, "unknown"),
    ):
        path = root / folder
        if not path.exists() or not find_audio([path]):
            continue
        out = root / "prepared" / folder.replace("/", "_")
        if (out / "manifest.json").exists() and not args.force:
            manifests[folder] = json.loads((out / "manifest.json").read_text())
            continue
        log(f"preparing {folder}")
        manifests[folder] = prepare_uploads(
            [path],
            out,
            sample_rate=args.sample_rate,
            singer=folder,
            accent=accent,
            mixtures=is_mixture,
            separate=separate,
            log=log,
        )
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
    config = CodecConfig(sample_rate=args.sample_rate)
    codec, disc = AudioCodec(config).to(device), MultiPeriodDiscriminator().to(device)
    files = find_audio([root / "prepared"])
    files = [f for f in files if f.name.endswith(".vocal.wav") or f.name.endswith(".instrumental.wav")]
    if not files:
        log("!! no prepared audio; run the prepare stage")
        return
    ckpt = root / "codec" / "codec.pt"
    step = 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        codec.load_state_dict(state["codec"])
        disc.load_state_dict(state["disc"])
        step = state["step"]
    opt_g = torch.optim.AdamW(codec.parameters(), lr=1e-4, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-4, betas=(0.8, 0.99))
    deadline = time.time() + args.codec_hours * 3600
    crop = int(1.5 * args.sample_rate)
    g = torch.Generator().manual_seed(step)
    log(f"codec: {len(files)} files, {args.codec_hours} h budget, resuming at step {step}")
    while time.time() < deadline and not (root / "stop").exists():
        batch = []
        for _ in range(16):
            wave = read_wave(files[int(torch.randint(0, len(files), (), generator=g))], args.sample_rate)
            if wave.shape[-1] < crop:
                wave = torch.nn.functional.pad(wave, (0, crop - wave.shape[-1]))
            start = int(torch.randint(0, wave.shape[-1] - crop + 1, (), generator=g))
            batch.append(wave[start : start + crop])
        wave = torch.stack(batch).to(device)[:, None]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = codec(wave)
            terms = codec_losses(
                out,
                wave,
                config,
                discriminator=disc if step > 5000 else None,
                time_weight=1.0 if step < 20000 else 0.1,
            )
        opt_g.zero_grad()
        terms["total"].backward()
        torch.nn.utils.clip_grad_norm_(codec.parameters(), 10.0)
        opt_g.step()
        if step > 5000:
            d_loss = discriminator_loss(disc, out["recon"].float(), wave)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()
        step += 1
        if step % 100 == 0:
            log(
                f"codec step {step}: mel {terms['mel'].item():.3f} commit {terms['commitment'].item():.4f} "
                f"usage {codec.quantizer.semantic.usage():.0%}"
            )
        if step % 2000 == 0:
            torch.save(
                {"codec": codec.state_dict(), "disc": disc.state_dict(), "step": step, "config": config.__dict__}, ckpt
            )
    torch.save({"codec": codec.state_dict(), "disc": disc.state_dict(), "step": step, "config": config.__dict__}, ckpt)
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


def stage_train(args) -> None:
    import torch

    from helix_lm.audio import SpeakerEncoder
    from helix_lm.audio.data import read_wave
    from helix_lm.music.regularize import ACCENTS, ClipTags, balanced_weights, split_singers
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
    weights = balanced_weights([tags[i] for i in train_idx], by="accent")
    config = TriStreamConfig.named(args.size, frame_dim=codec.config.dim, sample_rate=args.sample_rate)
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
    deadline = time.time() + args.train_hours * 3600
    frames_per_clip = int(args.clip_seconds * config.frame_rate)
    g = torch.Generator().manual_seed(step)
    singer_ids = {t.singer: i % config.num_singers for i, t in enumerate(tags)}
    last_save = time.time()
    while time.time() < deadline and not (root / "stop").exists():
        chosen = torch.multinomial(weights, args.batch, replacement=True, generator=g).tolist()
        batch_rows = [records[train_idx[i]] for i in chosen]
        waves, f0s, accents, singers = [], [], [], []
        for r in batch_rows:
            stem = Path(r["path"]).stem
            wave = read_wave(Path(r["dir"]) / f"{stem}.vocal.wav", args.sample_rate)
            f0 = torch.load(Path(r["dir"]) / f"{stem}.f0.pt")
            need = int(args.clip_seconds * args.sample_rate)
            if wave.shape[-1] < need:
                wave = torch.nn.functional.pad(wave, (0, need - wave.shape[-1]))
                f0 = torch.nn.functional.pad(f0, (0, frames_per_clip - f0.shape[-1]))
            start = int(torch.randint(0, wave.shape[-1] - need + 1, (), generator=g))
            f_start = int(start / args.sample_rate * config.frame_rate)
            waves.append(wave[start : start + need])
            f0s.append(f0[f_start : f_start + frames_per_clip])
            accents.append(ACCENTS.index(r["tags"]["accent"]) if r["tags"]["accent"] in ACCENTS else 0)
            singers.append(singer_ids[r["tags"]["singer"]])
        wave = torch.stack(waves).to(device)
        with torch.no_grad():
            frames = codec.listen(wave)[:, :frames_per_clip]
        t = frames.shape[1]
        f0 = torch.stack([torch.nn.functional.pad(f, (0, max(0, t - f.shape[0])))[:t] for f in f0s]).to(device)
        energy = frames.pow(2).mean(-1)
        speaker = speaker_encoder(frames)  # the clip's own voice: the model learns to copy it (in-context cloning)
        batch = {
            "source": source_features(f0, energy), "noise": noise_features(frames), "frames": frames,
            "phonemes": torch.full((len(batch_rows), 1), -1, dtype=torch.long, device=device),
            "speaker": speaker, "singer": torch.tensor(singers, device=device),
            "accent": torch.tensor(accents, device=device), "f0_hz": f0,
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
            log(f"out of memory: batch halved to {args.batch} clips of {args.clip_seconds:.0f} s")
            continue
        step += 1
        if step % 20 == 0:
            log(f"step {step}: " + " ".join(f"{k} {float(v):.3f}" for k, v in losses.items() if k != "mask_fraction"))
        if time.time() - last_save > 1800:
            torch.save(
                {
                    "model": model.state_dict(),
                    "speaker": speaker_encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": config.__dict__,
                },
                ckpt,
            )
            last_save = time.time()
            log(f"checkpoint at step {step}")
            if args.hf_repo and token():
                stage_push(args, quiet=True)
    torch.save(
        {
            "model": model.state_dict(),
            "speaker": speaker_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config.__dict__,
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
    from helix_lm.music.effects import detect_glitches
    from helix_lm.music.regularize import ClipTags, split_singers
    from helix_lm.music.tristream import TriStreamConfig, TriStreamForSinging, noise_features, source_features

    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = load_codec(root, device)
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
    if args.batch <= 0:
        args.batch = 64  # auto: the largest batch that fits is found by halving on out-of-memory
        log("auto batch size: starting at 64 clips and halving on out-of-memory until it fits")
    records = []
    for manifest in (root / "prepared").rglob("manifest.json"):
        data = json.loads(manifest.read_text())
        records += [{**clip, "dir": str(manifest.parent)} for clip in data["clips"] if clip["accepted"]]
    tags = [ClipTags(**r["tags"]) for r in records]
    split = split_singers([t.singer for t in tags])
    held = [r for r, t in zip(records, tags, strict=True) if split[t.singer] == "validation"][: args.eval_clips]
    results = {"clips": len(held), "f0_rmse_cents": [], "speaker_similarity": [], "glitch_rate": []}
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
            out = model.sample(batch, steps=config.flow_steps)
            audio = codec.decode_frames(out)[0, 0].cpu()
        f0_out = estimate_f0(audio[None], args.sample_rate, config.frame_rate)[0]
        n = min(f0_out.shape[0], f0.shape[1])
        voiced = (f0[0, :n].cpu() > 0) & (f0_out[:n] > 0)
        if voiced.any():
            cents = 1200 * torch.log2(f0_out[:n][voiced] / f0[0, :n].cpu()[voiced])
            results["f0_rmse_cents"].append(float(cents.pow(2).mean().sqrt()))
        with torch.no_grad():
            sim = speaker_similarity(speaker_encoder(codec.listen(audio[None].to(device))), speaker_encoder(frames))
        results["speaker_similarity"].append(float(sim))
        glitches = detect_glitches(audio, args.sample_rate)
        results["glitch_rate"].append(
            float(glitches["clicks"].numel() + glitches["dropouts"].numel())
            / max(audio.shape[-1] / args.sample_rate, 1e-6)
        )
    summary = {k: (sum(v) / len(v) if isinstance(v, list) and v else v) for k, v in results.items()}
    (root / "eval" / "summary.json").write_text(json.dumps(summary, indent=1))
    log("eval:", summary)
    mark(root, "eval")


# ------------------------------------------------------------------------------------------- push --
def stage_push(args, quiet: bool = False) -> None:
    if not args.hf_repo or not token():
        log("push skipped: --hf-repo and a token are needed")
        return
    from huggingface_hub import HfApi

    root = Path(args.root)
    api = HfApi(token=token())
    api.create_repo(args.hf_repo, exist_ok=True, private=not args.public)
    for folder, path_in_repo in (
        (root / "runs" / args.size, f"runs/{args.size}"),
        (root / "codec", "codec"),
        (root / "eval", "eval"),
    ):
        if folder.exists() and any(folder.iterdir()):
            api.upload_folder(
                folder_path=str(folder),
                repo_id=args.hf_repo,
                path_in_repo=path_in_repo,
                commit_message=f"{path_in_repo} {time.strftime('%Y-%m-%d %H:%M')}",
                ignore_patterns=["*.part"],
            )
    if not quiet:
        log(f"pushed to https://huggingface.co/{args.hf_repo}")
        mark(root, "push")


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
        converter = Converter(model, codec, speaker_encoder)
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
    parser.add_argument("--hf-repo", default="", help="e.g. CorxLabs/TriStream-2-765M")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--max-gb", type=float, default=200.0, help="data: skip a source bigger than this")
    parser.add_argument(
        "--emilia-hours", type=float, default=1000.0, help="data: hours of Emilia-YODAS (CC BY) to fetch"
    )
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
    stages = list(STAGES[:-1]) if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage not in STAGES:
            parser.error(f"unknown stage {stage}")
        if args.stage == "all" and done(root, stage) and stage not in ("push",):
            log(f"{stage}: done earlier, skipped (delete markers/{stage} to redo)")
            continue
        log(f"== stage {stage}")
        globals()[f"stage_{stage}"](args)
        if (root / "stop").exists():
            log("stop file found; stopping")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
