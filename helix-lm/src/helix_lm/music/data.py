# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Music and singing data: the catalogue with licences, multi-file ingestion, and the preparation pipeline that
turns uploads into TriStream's training records.

The licence column decides everything downstream. Commercially clean singing audio in the open is under a
hundred hours (VocalSet, the choir sets); the large singing corpora are non-commercial and are used here for
**evaluation and research only**; the plan's commercial path is licensed or recorded singing plus CC-BY speech
(Emilia-YODAS) and CC-BY music (FMA's per-track CC filter, Slakh instrumentals). MUSDB18-HQ and MoisesDB are
evaluation sets, never training sets, for the same reason. The research behind each row is in
``docs/TRISTREAM.md``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..audio.data import read_wave, write_wave
from ..audio.singing import estimate_f0
from .analysis import analyze
from .regularize import ClipTags, breath_events, f0_range, is_dry_enough

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif")


@dataclass(frozen=True)
class MusicSource:
    name: str
    where: str
    kind: str  # singing | stems | speech | music | choir | eval
    hours: float
    license: str
    commercial_ok: bool
    contents: str = ""
    role: str = "train"  # train | eval | research


MUSIC_SOURCES: tuple[MusicSource, ...] = (
    MusicSource(
        "vocalset", "zenodo 1193957", "singing", 10.1, "CC BY 4.0", True, "20 pro singers, techniques, vowels"
    ),
    MusicSource("dagstuhl-choirset", "zenodo 4607766", "choir", 1.0, "CC BY 4.0", True, "close mics, F0"),
    MusicSource("esmuc", "zenodo 5848990", "choir", 0.5, "CC BY 4.0", True, "12 singers, F0 + notes"),
    MusicSource("vocadito", "zenodo", "singing", 0.3, "CC BY 4.0", True, "40 excerpts, f0, notes, lyrics", "eval"),
    MusicSource("emilia-yodas", "amphion/Emilia-Dataset", "speech", 114_000, "CC BY 4.0", True, "the YODAS half only"),
    MusicSource("slakh2100", "zenodo 4599666", "stems", 145, "CC BY 4.0", True, "synthesized multitrack, no vocals"),
    MusicSource(
        "fma-cc",
        "mdeff/fma",
        "music",
        2_000,
        "per-track CC (filter)",
        True,
        "filter genres.csv for CC-BY tracks; reggae tag exists",
    ),
    MusicSource(
        "gtsinger",
        "AaronZ345/GTSinger",
        "singing",
        80.6,
        "CC BY-NC-SA 4.0",
        False,
        "techniques, scores, 9 languages",
        "research",
    ),
    MusicSource("opensinger", "request form", "singing", 50, "CC BY-NC-SA", False, "66 singers", "research"),
    MusicSource(
        "m4singer", "request form", "singing", 29.8, "non-commercial", False, "20 singers, scores", "research"
    ),
    MusicSource("opencpop", "request form", "singing", 5.2, "CC BY-NC-ND", False, "1 singer, scores", "research"),
    MusicSource("popcs", "request form", "singing", 5.9, "CC BY-NC-SA", False, "", "research"),
    MusicSource("nus-48e", "request", "singing", 1.9, "research only", False, "12 singers, phone labels", "research"),
    MusicSource(
        "singstyle111", "request", "singing", 12.8, "restricted", False, "styles, breath labels, emotions", "research"
    ),
    MusicSource("popbutfy", "request", "singing", 50.8, "unverified", False, "amateur/pro pairs", "research"),
    MusicSource("csd", "zenodo", "choir", 1.0, "unread", False, "SATB + MIDI", "research"),
    MusicSource(
        "jacappella", "jaCappella/jaCappella", "choir", 1.5, "research only", False, "Japanese a cappella", "research"
    ),
    MusicSource("musdb18-hq", "zenodo 3338373", "eval", 10, "educational only", False, "150 songs, 4 stems", "eval"),
    MusicSource(
        "moisesdb",
        "music.ai",
        "eval",
        14,
        "CC BY-NC-SA 4.0",
        False,
        "240 tracks, 12 genres, bleed annotations",
        "eval",
    ),
    MusicSource(
        "medleydb", "medleydb.weebly.com", "eval", 7.2, "CC BY-NC-SA 4.0", False, "multitrack + melody", "eval"
    ),
    MusicSource("medleyvox", "paper 2211.07302", "eval", 1.0, "research", False, "duet / unison / N-singer", "eval"),
    MusicSource(
        "mtg-jamendo", "mtg", "music", 3_800, "non-commercial research", False, "55k tracks, tags", "research"
    ),
    MusicSource(
        "deam",
        "cvml.unige.ch",
        "eval",
        30,
        "CC (unverified)",
        False,
        "arousal/valence at 2 Hz; critic training",
        "research",
    ),
    MusicSource(
        "pmemo", "HuiZhangDB/PMEmo", "eval", 5, "MIT repo", True, "794 choruses, dynamic V/A, EDA", "research"
    ),
    MusicSource(
        "ochim",
        "osf.io/x59fm",
        "eval",
        0,
        "unstated",
        False,
        "1,806 chill onsets, 1,019 pieces (no audio)",
        "research",
    ),
    MusicSource(
        "patois-music-asr",
        "paper 2507.16834",
        "singing",
        42.6,
        "no public release",
        False,
        "Jamaican Patois lyrics, transcribed",
        "research",
    ),
)


def describe_music_sources(commercial_only: bool = False) -> str:
    rows = [f"{'source':18} {'kind':8} {'hours':>8}  {'commercial':10} {'role':8}  licence / contents"]
    for s in MUSIC_SOURCES:
        if commercial_only and not s.commercial_ok:
            continue
        rows.append(
            f"{s.name:18} {s.kind:8} {s.hours:>8,.1f}  {'yes' if s.commercial_ok else 'no':10} {s.role:8}  "
            f"{s.license}; {s.contents}"
        )
    return "\n".join(rows)


# ----------------------------------------------------------------------------------- ingestion --
def find_audio(paths: list[str | Path]) -> list[Path]:
    """Every audio file under the given files and folders (multi-file uploads), sorted."""
    found: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            found += [q for q in p.rglob("*") if q.suffix.lower() in AUDIO_SUFFIXES]
        elif p.suffix.lower() in AUDIO_SUFFIXES:
            found.append(p)
    return sorted(set(found))


@dataclass
class ClipRecord:
    path: str
    seconds: float
    tags: dict
    analysis: dict
    f0_path: str
    accepted: bool
    reason: str = ""


def prepare_clip(
    path: Path,
    out_dir: Path,
    sample_rate: int,
    singer: str,
    accent: str = "unknown",
    emotion: str = "unknown",
    language: str = "en",
    is_mixture: bool = False,
    separate=None,
    log=print,
) -> ClipRecord:
    """
    One upload -> a record: resampled mono audio, the vocal (separated when the file is a full song and a
    separator is given), F0 at 25 Hz, breath events, range, dryness check, and the song analysis (tempo, key,
    sections) so the model can be trained with beat-aware timing. Rejected clips are kept in the manifest with
    the reason, never silently dropped.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wave = read_wave(path, sample_rate)
    seconds = wave.shape[-1] / sample_rate
    stem = path.stem
    vocal = wave
    reason = ""
    if is_mixture:
        if separate is None:
            return ClipRecord(
                str(path), seconds, {}, {}, "", False, "a full song needs a separator (install demucs or pass one)"
            )
        stems = separate(wave, sample_rate)
        vocal = stems["vocals"]
        write_wave(
            out_dir / f"{stem}.instrumental.wav", sum(v for k, v in stems.items() if k != "vocals"), sample_rate
        )
    dry, dryness = is_dry_enough(vocal, sample_rate)
    low, high = f0_range(vocal, sample_rate)
    tags = ClipTags(
        singer=singer,
        accent=accent,
        emotion=emotion,
        language=language,
        f0_low=low,
        f0_high=high,
        dry=dry,
        breaths=breath_events(vocal, sample_rate),
    )
    f0 = estimate_f0(vocal[None], sample_rate, frame_rate=25)[0]
    torch.save(f0, out_dir / f"{stem}.f0.pt")
    write_wave(out_dir / f"{stem}.vocal.wav", vocal, sample_rate)
    try:
        analysis = analyze(wave if is_mixture else vocal, sample_rate)
        summary = {
            "tempo": analysis.tempo,
            "key": analysis.key_name,
            "sections": analysis.segments,
            "reverb": analysis.reverb,
            "cues": analysis.vibe.get("cues"),
        }
    except Exception as error:  # analysis must never lose a clip
        summary = {"error": f"{type(error).__name__}: {error}"}
    accepted = dry and high > 0
    if not dry:
        reason = f"not dry enough (reverb {dryness['reverb']:.2f}s): kept for listening data, not as a target"
    elif high <= 0:
        reason = "no voiced frames found"
    log(
        f"[clip] {path.name}: {seconds:.1f}s, range {low:.0f}-{high:.0f} Hz, breaths {len(tags.breaths)}, "
        f"{'ok' if accepted else reason}"
    )
    return ClipRecord(str(path), seconds, asdict(tags), summary, str(out_dir / f"{stem}.f0.pt"), accepted, reason)


def prepare_uploads(
    paths: list[str | Path],
    out_dir: str | Path,
    sample_rate: int = 24000,
    singer: str = "upload",
    accent: str = "unknown",
    emotion: str = "unknown",
    mixtures: bool = False,
    separate=None,
    log=print,
) -> dict:
    """Many files or folders -> ``manifest.json`` with one record per clip and the data gaps."""
    out_dir = Path(out_dir)
    files = find_audio(paths)
    records = [
        prepare_clip(f, out_dir, sample_rate, singer, accent, emotion, is_mixture=mixtures, separate=separate, log=log)
        for f in files
    ]
    from .regularize import report_gaps

    tags = [ClipTags(**r.tags) for r in records if r.tags]
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_rate": sample_rate,
        "clips": [asdict(r) for r in records],
        "accepted": sum(r.accepted for r in records),
        "hours": sum(r.seconds for r in records) / 3600,
        "gaps": report_gaps(tags) if tags else ["no clips"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    log(
        f"[data] {manifest['accepted']}/{len(records)} clips accepted, {manifest['hours']:.2f} h; "
        f"gaps: {manifest['gaps']}"
    )
    return manifest
