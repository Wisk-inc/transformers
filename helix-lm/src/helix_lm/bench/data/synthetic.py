# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
A synthetic singing set where every answer is known: the pitch curve (notes, glides, vibrato), the vowel's formants,
the singer's vocal-tract length and breathiness. Each take is a harmonic series shaped by a formant envelope, with
band-limited breath noise; every metric of the suite can be checked against the truth here before it is trusted
on recordings, and the whole suite runs on a CPU in seconds.

Formant centres are the classic adult male averages (Peterson and Barney, 1952) scaled by the singer's tract.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch

from .registry import Item

VOWELS = {"a": (730, 1090, 2440), "e": (530, 1840, 2480), "i": (270, 2290, 3010), "o": (570, 840, 2410),
          "u": (300, 870, 2240)}
SINGERS = {
    "low": {"tract": 1.0, "tilt_db": -12.0, "breath": 0.015, "range": (45, 57)},
    "mid": {"tract": 1.1, "tilt_db": -10.0, "breath": 0.02, "range": (52, 64)},
    "high": {"tract": 1.2, "tilt_db": -8.0, "breath": 0.03, "range": (60, 72)},
}
TRUTH_RATE = 100.0


def midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69.0) / 12.0)


def pitch_curve(notes, seconds: float, rate: float, vibrato_hz: float = 5.5, vibrato_cents: float = 35.0,
                glide_s: float = 0.06, vibrato_delay_s: float = 0.25) -> torch.Tensor:
    """F0 in Hz at ``rate`` frames a second: the notes, an exponential glide into each, vibrato after a delay."""
    n = int(round(seconds * rate))
    f0 = torch.zeros(n, dtype=torch.float64)
    prev = None
    for onset, offset, midi in notes:
        a, b = int(round(onset * rate)), min(n, int(round(offset * rate)))
        if b <= a:
            continue
        t = torch.arange(b - a, dtype=torch.float64) / rate
        cents = torch.zeros_like(t)
        if prev is not None and abs(onset - prev[1]) < 1e-6:          # a slur from the note before: glide
            cents = (prev[2] - midi) * 100.0 * torch.exp(-t / max(glide_s / 3, 1e-3))
        vib = vibrato_cents * torch.sin(2 * math.pi * vibrato_hz * t) * (t > vibrato_delay_s).double()
        f0[a:b] = midi_hz(midi) * 2 ** ((cents + vib) / 1200.0)
        prev = (onset, offset, midi)
    return f0


BANDWIDTHS = (80.0, 100.0, 140.0)


def formant_gain(freqs: torch.Tensor, vowel: str, tract: float) -> torch.Tensor:
    """The magnitude of three second-order resonators in cascade (how the vocal tract shapes a voice), normalised to
    1 at 0 Hz. The skirts are wide, as in real voices, so no single harmonic swamps the others."""
    gain = torch.ones_like(freqs, dtype=torch.float64)
    f = freqs.double()
    for centre, bw in zip(VOWELS[vowel], BANDWIDTHS, strict=True):
        c = centre * tract
        gain = gain / torch.sqrt((1 - (f / c) ** 2) ** 2 + (f * bw / c**2) ** 2)
    return gain / gain.max().clamp(min=1e-9) if gain.numel() > 1 else gain


def render(notes, sample_rate: int, vowel: str, singer: str, seconds: float, seed: int = 0, **pitch) -> tuple:
    """``(wave, f0 at TRUTH_RATE)``."""
    s = SINGERS[singer]
    f0 = pitch_curve(notes, seconds, TRUTH_RATE, **pitch)
    n = int(round(seconds * sample_rate))
    f0_s = torch.nn.functional.interpolate(f0[None, None], size=n, mode="linear", align_corners=False)[0, 0]
    voiced = f0_s > 0
    phase = 2 * math.pi * torch.cumsum(f0_s, 0) / sample_rate
    wave = torch.zeros(n, dtype=torch.float64)
    top = int(sample_rate / 2 / max(float(f0[f0 > 0].min()) if (f0 > 0).any() else 100.0, 50.0))
    for k in range(1, min(top, 60) + 1):
        fk = f0_s * k
        alive = voiced & (fk < sample_rate / 2 * 0.95)
        amp = 10 ** (s["tilt_db"] * math.log2(k) / 20) * formant_gain(fk, vowel, s["tract"])
        wave += torch.where(alive, amp * torch.sin(k * phase), torch.zeros_like(fk))
    env = torch.zeros(n, dtype=torch.float64)
    for onset, offset, _ in notes:
        a, b = int(onset * sample_rate), min(n, int(offset * sample_rate))
        if b > a:
            length = b - a
            ramp_in, ramp_out = min(int(0.03 * sample_rate), length // 2), min(int(0.05 * sample_rate), length // 2)
            seg = torch.ones(length, dtype=torch.float64)
            seg[:ramp_in] = torch.linspace(0, 1, ramp_in, dtype=torch.float64)
            if ramp_out:
                seg[-ramp_out:] = torch.linspace(1, 0, ramp_out, dtype=torch.float64)
            env[a:b] = torch.maximum(env[a:b], seg)
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(n, generator=g, dtype=torch.float64)
    spec = torch.fft.rfft(noise)
    freqs = torch.fft.rfftfreq(n, 1.0 / sample_rate).double()
    noise = torch.fft.irfft(spec * formant_gain(freqs, vowel, s["tract"]) * (freqs > 1500), n=n)
    noise = noise / noise.abs().max().clamp(min=1e-9)
    out = wave * env + s["breath"] * noise * (0.3 + env)
    return (0.5 * out / out.abs().max().clamp(min=1e-9)).float(), f0


def melody(rng: random.Random, low: int, high: int, seconds: float) -> list:
    notes, t = [], 0.1
    while t < seconds - 0.4:
        d = rng.choice((0.3, 0.45, 0.6, 0.8))
        end = min(seconds - 0.1, t + d)
        notes.append((round(t, 3), round(end, 3), rng.randint(low, high)))
        t = end + (0.0 if rng.random() < 0.6 else rng.choice((0.15, 0.3)))     # slurred, or a short rest
    return notes


def make_set(out_dir: str | Path, takes_per_singer: int = 4, seconds: float = 4.0, sample_rate: int = 24000,
             seed: int = 0) -> list[Item]:
    """Write the takes and their truth (``<id>.truth.json``: notes, F0 at 100 Hz, vowel, vibrato) to ``out_dir``."""
    from ...audio.data import write_wave

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    items = []
    for singer, s in SINGERS.items():
        for k in range(takes_per_singer):
            vowel = rng.choice(sorted(VOWELS))
            notes = melody(rng, *s["range"], seconds)
            vib = {"vibrato_hz": rng.uniform(4.8, 6.5), "vibrato_cents": rng.uniform(20.0, 50.0)}
            wave, f0 = render(notes, sample_rate, vowel, singer, seconds, seed=seed * 1000 + len(items), **vib)
            item_id = f"syn-{singer}-{k}"
            path = out_dir / f"{item_id}.wav"
            write_wave(path, wave, sample_rate)
            (out_dir / f"{item_id}.truth.json").write_text(json.dumps(
                {"notes": notes, "vowel": vowel, "f0": f0.tolist(), "rate": TRUTH_RATE, **vib}))
            items.append(Item(item_id, path, singer, "synthetic", notes=notes, technique=vowel))
    return items


def truth(item: Item) -> dict:
    data = json.loads(Path(item.path).with_suffix(".truth.json").read_text())
    data["f0"] = torch.tensor(data["f0"], dtype=torch.float64)
    return data


__all__ = ["SINGERS", "TRUTH_RATE", "VOWELS", "make_set", "melody", "pitch_curve", "render", "truth"]
