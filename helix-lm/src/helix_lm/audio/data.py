# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Audio data: the catalogue (what exists, how big, under what licence) and the readers.

The licences matter more than usual here: most singing corpora and some of the largest speech sets are
non-commercial. The catalogue carries the licence on every row so a run can be filtered to what a product may
use (``commercial_ok``). Hours are the publishers' figures (see ``docs/ANANSI.md`` for sources).
"""

from __future__ import annotations

import json
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class AudioSource:
    name: str
    repo: str
    kind: str  # speech | singing | music | general | instruction
    hours: float
    license: str
    commercial_ok: bool
    note: str = ""


AUDIO_SOURCES: tuple[AudioSource, ...] = (
    AudioSource(
        "emilia-yodas",
        "amphion/Emilia-Dataset",
        "speech",
        114_000,
        "CC BY 4.0",
        True,
        "the YODAS half; the Emilia half is CC BY-NC",
    ),
    AudioSource("librispeech", "openslr/librispeech_asr", "speech", 960, "CC BY 4.0", True),
    AudioSource("libritts-r", "mythicinfinity/libritts_r", "speech", 585, "CC BY 4.0", True, "studio-quality TTS set"),
    AudioSource("peoples-speech", "MLCommons/peoples_speech", "speech", 30_000, "CC BY / CC BY-SA", True),
    AudioSource(
        "common-voice", "mozilla-foundation/common_voice_17_0", "speech", 22_000, "CC0", True, "137 languages"
    ),
    AudioSource("mls", "facebook/multilingual_librispeech", "speech", 50_000, "CC BY 4.0", True),
    AudioSource("voxpopuli", "facebook/voxpopuli", "speech", 1_800, "CC0", True, "400k h unlabelled besides"),
    AudioSource("loquacious", "speechbrain/LoquaciousSet", "speech", 25_000, "commercial-OK mix", True),
    AudioSource("gigaspeech", "speechcolab/gigaspeech", "speech", 10_000, "non-commercial", False),
    AudioSource("emilia", "amphion/Emilia-Dataset", "speech", 101_000, "CC BY-NC 4.0", False),
    AudioSource(
        "patois-music-asr",
        "(paper 2507.16834)",
        "speech",
        42.6,
        "research",
        False,
        "Jamaican Patois lyrics with transcriptions; the only Patois speech set found",
    ),
    AudioSource(
        "caribbean-voices",
        "(zindi caribbean-voices)",
        "speech",
        230,
        "competition terms",
        False,
        "BBC Caribbean Service archive clips",
    ),
    AudioSource("vocalset", "(zenodo 1442513)", "singing", 10.1, "CC BY 4.0", True, "20 singers, techniques"),
    AudioSource(
        "gtsinger", "AaronZ345/GTSinger", "singing", 80.6, "custom (research)", False, "9 languages, technique tags"
    ),
    AudioSource("m4singer", "(request form)", "singing", 29.8, "non-commercial", False),
    AudioSource("opensinger", "(request form)", "singing", 50, "CC BY-NC", False),
    AudioSource("opencpop", "(request form)", "singing", 5.2, "CC BY-NC-ND", False),
    AudioSource("ace-opencpop", "(request form)", "singing", 128.9, "CC BY-NC 4.0", False, "synthetic"),
    AudioSource("jamendo", "(mtg-jamendo)", "music", 3_800, "CC mix", True),
    AudioSource("fsd50k", "Fhrozen/FSD50k", "general", 108, "CC", True),
    AudioSource("audioset", "agkphysics/AudioSet", "general", 5_800, "CC BY 4.0 labels; YouTube audio", False),
    AudioSource("audiosetcaps", "baijs/AudioSetCaps", "instruction", 0, "CC BY 4.0", True, "6.1M captions, 18.4M QA"),
    AudioSource(
        "sift-50m", "amazon-agi/SIFT-50M", "instruction", 14_000, "CDLA-Sharing", True, "50M speech instructions"
    ),
    AudioSource("voicebench", "hlt-lab/voicebench", "instruction", 0, "eval", True, "evaluation only"),
)


def describe_audio_sources(commercial_only: bool = False) -> str:
    rows = [f"{'source':18} {'kind':11} {'hours':>9}  {'commercial':10}  licence / note"]
    for s in AUDIO_SOURCES:
        if commercial_only and not s.commercial_ok:
            continue
        note = f"; {s.note}" if s.note else ""
        ok = "yes" if s.commercial_ok else "no"
        rows.append(f"{s.name:18} {s.kind:11} {s.hours:>9,.0f}  {ok:10}  {s.license}{note}")
    return "\n".join(rows)


# ------------------------------------------------------------------------------------------ readers --
def read_wave(path: str | Path, sample_rate: int = 24000) -> torch.Tensor:
    """
    A PCM WAV file -> mono float32 ``(samples,)`` in ``[-1, 1]`` at ``sample_rate`` (linear resampling). Uses
    ``soundfile`` when installed (any format), the standard library otherwise (8/16/24/32-bit PCM WAV).
    """
    path = Path(path)
    try:
        import soundfile  # type: ignore

        data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        audio = torch.from_numpy(data.mean(axis=1))
    except ImportError:
        with wave.open(str(path), "rb") as handle:
            channels, width, rate, frames = (
                handle.getnchannels(),
                handle.getsampwidth(),
                handle.getframerate(),
                handle.getnframes(),
            )
            raw = handle.readframes(frames)
        if width == 2:
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 4:
            samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        elif width == 1:
            samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif width == 3:
            triplets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            ints = (
                triplets[:, 0].astype(np.int32)
                | (triplets[:, 1].astype(np.int32) << 8)
                | (triplets[:, 2].astype(np.int32) << 16)
            )
            ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
            samples = ints.astype(np.float32) / 8388608.0
        else:
            raise ValueError(f"unsupported sample width {width}") from None
        audio = torch.from_numpy(samples.reshape(-1, channels).mean(axis=1).copy())
    return resample(audio, rate, sample_rate)


def resample(audio: torch.Tensor, rate: int, target: int) -> torch.Tensor:
    if rate == target:
        return audio
    length = int(round(audio.shape[-1] * target / rate))
    return torch.nn.functional.interpolate(audio[None, None], size=length, mode="linear", align_corners=False)[0, 0]


def write_wave(path: str | Path, audio: torch.Tensor, sample_rate: int = 24000) -> Path:
    """Mono float ``(samples,)`` -> 16-bit PCM WAV."""
    path = Path(path)
    clipped = audio.detach().cpu().clamp(-1, 1).numpy()
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path


def synthetic_voice(
    seconds: float = 1.0,
    sample_rate: int = 24000,
    f0: float = 220.0,
    harmonics: int = 8,
    vibrato_hz: float = 5.5,
    seed: int = 0,
) -> torch.Tensor:
    """A harmonic tone with vibrato and breath noise: the test signal for codec, pitch and speaker tests."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(int(seconds * sample_rate)) / sample_rate
    pitch = f0 * (1 + 0.02 * torch.sin(2 * torch.pi * vibrato_hz * t))
    phase = 2 * torch.pi * torch.cumsum(pitch, dim=0) / sample_rate
    wave_ = sum(torch.sin(k * phase) / k for k in range(1, harmonics + 1))
    envelope = torch.sin(torch.pi * t / seconds).clamp(min=0.05)
    noise = 0.01 * torch.randn(t.shape, generator=g)
    return (0.5 * wave_ * envelope + noise).clamp(-1, 1)


# ------------------------------------------------------------------------------------------ shards --
def encode_files(codec, files: list[Path], out_dir: str | Path, sample_rate: int, batch_seconds: float = 20.0) -> dict:
    """
    Codec tokens for a list of audio files into ``out_dir``: one ``.codes`` file of uint16 ``(Q, frames)`` per
    input plus ``manifest.json`` (frames, seconds, codebooks). The language model's audio stream reads these.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"codebooks": codec.config.num_codebooks, "frame_rate": codec.config.frame_rate, "files": []}
    codec.eval()
    for path in files:
        audio = read_wave(path, sample_rate)
        codes = codec.encode(audio[None])[0].to(torch.int32).numpy().astype(np.uint16)
        target = out_dir / (Path(path).stem + ".codes")
        with open(target, "wb") as handle:
            handle.write(struct.pack("<II", codes.shape[0], codes.shape[1]))
            handle.write(codes.tobytes())
        manifest["files"].append(
            {
                "source": str(path),
                "codes": target.name,
                "frames": int(codes.shape[1]),
                "seconds": audio.shape[-1] / sample_rate,
            }
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def read_codes(path: str | Path) -> torch.Tensor:
    with open(path, "rb") as handle:
        q, t = struct.unpack("<II", handle.read(8))
        data = np.frombuffer(handle.read(), dtype=np.uint16).reshape(q, t)
    return torch.from_numpy(data.astype(np.int64))
