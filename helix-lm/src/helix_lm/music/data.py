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
import os
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


#: the pitch tracker a manifest's ``<stem>.f0.pt`` files were made with; manifests from before 2.4 have none
PITCH_TRACKER = "yin"


def _write_json(path: Path, data) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str))
    os.replace(tmp, path)


def refresh_pitch(out_dir: Path, manifest: dict, sample_rate: int, device=None, should_stop=None, log=print) -> dict:
    """
    The pitch files of a source prepared before 2.4, made again with YIN from each accepted clip's vocal (on disk
    as ``<stem>.vocal.wav``): the old autocorrelation tracker put 29% of a low voice's frames an octave off, and
    training learns its pitch head from these files. The separation and the other measurements stay; each clip's
    pitch range is measured again too (it picks the high-register clips). ``should_stop()`` True saves the
    progress (``pitch_refresh.json``) and raises :class:`helix_lm.molab.SessionEnding`; the next run goes on.
    """
    out_dir = Path(out_dir)
    progress_file = out_dir / "pitch_refresh.json"
    measured_ranges: dict = json.loads(progress_file.read_text()) if progress_file.exists() else {}  # stem: range
    clips = [c for c in manifest.get("clips", []) if c.get("accepted")]
    todo = [c for c in clips if Path(c["path"]).stem not in measured_ranges]
    log(f"[data] {out_dir.name}: pitch files from an older tracker; {len(todo)} of {len(clips)} to redo with YIN")
    for k, clip in enumerate(todo):
        if should_stop is not None and should_stop():
            _write_json(progress_file, measured_ranges)
            from ..molab import SessionEnding

            raise SessionEnding(f"{out_dir.name}: pitch redone for {len(measured_ranges)} of {len(clips)} clips; "
                                "the rest next time")
        stem = Path(clip["path"]).stem
        vocal = out_dir / f"{stem}.vocal.wav"
        if vocal.exists():
            wave = read_wave(vocal, sample_rate)
            measured = wave.to(device) if device is not None else wave
            with torch.no_grad():
                f0 = estimate_f0(measured[None], sample_rate, frame_rate=25)[0].cpu()
                low, high = f0_range(measured, sample_rate)
            tmp = out_dir / f"{stem}.f0.pt.{os.getpid()}.tmp"
            torch.save(f0, tmp)
            os.replace(tmp, out_dir / f"{stem}.f0.pt")
            measured_ranges[stem] = [low, high]
        else:
            measured_ranges[stem] = None                     # nothing to measure: the clip stays as it was
        if k % 500 == 499:
            _write_json(progress_file, measured_ranges)
    # every clip's new range, including those measured by an earlier, interrupted run
    for clip in manifest.get("clips", []):
        found = measured_ranges.get(Path(clip["path"]).stem) if clip.get("accepted") else None
        if found is not None:
            low, high = found
            clip.setdefault("tags", {}).update(f0_low=low, f0_high=high)
            if high <= 0:
                clip.update(accepted=False, reason="no voiced frames found (pitch measured again with YIN)")
    manifest["pitch_tracker"] = PITCH_TRACKER
    manifest["accepted"] = sum(bool(c.get("accepted")) for c in manifest.get("clips", []))
    _write_json(out_dir / "manifest.json", manifest)
    progress_file.unlink(missing_ok=True)
    return manifest


def prepare_clip(path: Path, out_dir: Path, *args, **kwargs) -> ClipRecord:
    """:func:`measure_clip`, reusing the clip's finished record when it and the clip's outputs are on disk
    (:mod:`helix_lm.music.prepare_cache`), and leaving a record when it is done."""
    import contextlib

    from .prepare_cache import load_record, save_record

    cached = load_record(Path(out_dir), Path(path))
    if cached is not None:
        return ClipRecord(**cached)
    record = measure_clip(Path(path), Path(out_dir), *args, **kwargs)
    if record.reason.startswith("unreadable"):       # may read next time (a file still being written): no record
        return record
    with contextlib.suppress(OSError):                 # a full disk must not lose the clip already measured
        save_record(Path(out_dir), asdict(record))
    return record


def measure_clip(
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
    device=None,
    analysis: bool = True,
    domain: str = "singing",
) -> ClipRecord:
    """
    One upload -> a record: resampled mono audio, the vocal (separated when the file is a full song and a
    separator is given), F0 at 25 Hz, breath events, range, dryness check, and the song analysis (tempo, key,
    sections) so the model can be trained with beat-aware timing. Rejected clips are kept in the manifest with
    the reason, never silently dropped. ``device`` runs the pitch, breath and dryness measurements on a GPU
    (the difference between days and hours on a thousand hours of speech); ``analysis=False`` skips the song
    analysis, which means nothing for speech.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    try:
        wave = read_wave(path, sample_rate)
    except Exception as error:  # an unreadable file is a record with a reason, not a crash
        log(f"[clip] {path.name}: unreadable ({type(error).__name__}: {str(error)[:60]})")
        return ClipRecord(str(path), 0.0, {}, {}, "", False, f"unreadable: {type(error).__name__}")
    seconds = wave.shape[-1] / sample_rate
    if seconds < 0.5:
        return ClipRecord(str(path), seconds, {}, {}, "", False, "shorter than half a second")
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
    measured = vocal.to(device) if device is not None else vocal
    with torch.no_grad():
        dry, dryness = is_dry_enough(measured, sample_rate)
        low, high = f0_range(measured, sample_rate)
        breaths = breath_events(measured, sample_rate)
        f0 = estimate_f0(measured[None], sample_rate, frame_rate=25)[0].cpu()
    tags = ClipTags(
        singer=singer,
        accent=accent,
        emotion=emotion,
        language=language,
        f0_low=low,
        f0_high=high,
        dry=dry,
        breaths=breaths,
        domain=domain,
    )
    torch.save(f0, out_dir / f"{stem}.f0.pt")
    write_wave(out_dir / f"{stem}.vocal.wav", vocal, sample_rate)
    summary: dict = {}
    if analysis:
        try:
            result = analyze(wave if is_mixture else vocal, sample_rate)
            summary = {
                "tempo": result.tempo,
                "key": result.key_name,
                "sections": result.segments,
                "reverb": result.reverb,
                "cues": result.vibe.get("cues"),
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
    device=None,
    analysis: bool = True,
    workers: int = 1,
    singer_of=None,
    language: str = "en",
    domain: str = "singing",
    should_stop=None,
) -> dict:
    """
    Many files or folders -> ``manifest.json`` with one record per clip and the data gaps. ``singer_of(path)``
    names the singer per file (one corpus, many voices); ``workers`` threads overlap file reading and writing
    with the measurements; ``device`` puts the measurements on a GPU. Clips finished by an earlier run are read
    back, not measured again. ``should_stop()`` True stops before the next clip and raises
    :class:`helix_lm.molab.SessionEnding`, with no manifest written: the next run goes on from the clips done.
    """
    from concurrent.futures import ThreadPoolExecutor

    from .prepare_cache import duplicate_stems

    out_dir = Path(out_dir)
    files = find_audio(paths)
    twins = duplicate_stems(files)

    def one(f: Path) -> ClipRecord | None:
        if f in twins:                                 # its outputs would overwrite the first file's
            return ClipRecord(str(f), 0.0, {}, {}, "", False, f"same file name as {twins[f]}")
        if should_stop is not None and should_stop():
            return None
        return prepare_clip(
            f, out_dir, sample_rate, singer_of(f) if singer_of else singer, accent, emotion, language,
            is_mixture=mixtures, separate=separate, log=log, device=device, analysis=analysis, domain=domain,
        )  # fmt: skip

    if workers > 1 and len(files) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(one, files))
    else:
        records = [one(f) for f in files]
    if any(r is None for r in records):
        from ..molab import SessionEnding

        done = sum(r is not None for r in records)
        raise SessionEnding(f"{out_dir.name}: stopped at {done} of {len(records)} clips; the rest are done next time")
    from .regularize import report_gaps

    tags = [ClipTags(**r.tags) for r in records if r.tags]
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_rate": sample_rate,
        "clips": [asdict(r) for r in records],
        "accepted": sum(r.accepted for r in records),
        "hours": sum(r.seconds for r in records) / 3600,
        "gaps": report_gaps(tags) if tags else ["no clips"],
        "pitch_tracker": PITCH_TRACKER,
    }
    _write_json(out_dir / "manifest.json", manifest)
    log(
        f"[data] {manifest['accepted']}/{len(records)} clips accepted, {manifest['hours']:.2f} h; "
        f"gaps: {manifest['gaps']}"
    )
    return manifest
