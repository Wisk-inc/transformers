# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Humanize: the pitch, loudness and breath of a sung line as a body produces it, from a list of notes.

A synthesizer sounds like one because its pitch arrives on the note instantly and sits there, its vibrato is a
sine from a function, and every note starts at the same loudness. Singers do none of that, and the ways they
do not are well measured (Sundberg's *The Science of the Singing Voice*; Prame 1994/1997 on vibrato rate and
extent; Seashore 1938 on portamento and the vibrato's onset; Saitou et al. 2005 on preparation, overshoot and
fine fluctuation in singing F0):

* **portamento**: the pitch glides between notes over 40-200 ms, longer for larger intervals, with an
  S-shaped (not linear) trajectory;
* **overshoot**: an upward leap passes its target by 10-30 cents and settles within ~80 ms; a downward leap
  undershoots;
* **preparation (scoop)**: a phrase-initial note is approached from below, 50-200 cents, over 60-120 ms;
* **vibrato**: it starts 150-400 ms into a held note, its extent ramps in over ~300 ms, its rate sits at 5-7 Hz
  and drifts, its extent grows towards the end of the note and the end of the phrase;
* **fine fluctuation**: a slow wander (1-4 Hz, 10-20 cents) and a fast jitter (a few cents) are always present;
* **release**: the last note of a phrase falls away 50-200 cents as the voice relaxes;
* **timing**: onsets land 10-25 ms early or late, never exactly on the grid;
* **loudness**: each note has an attack of 30-80 ms and a small decay; a phrase swells towards its highest note
  and lets go after; a breath, with its own noise, precedes a phrase.

:func:`humanize` turns notes into those curves at a fine rate (100 Hz by default) so the source stream and the
refiner's oscillator can follow a run at 10 ms, and :func:`decimate` folds them to the codec's 25 Hz for the
streams that live there. A :class:`HumanStyle` is one singer's habits; the same seed gives the same singer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

Note = tuple[int, float, float]  # (midi, start seconds, end seconds)


@dataclass
class HumanStyle:
    portamento_ms: float = 90.0  # glide time for a whole tone; scales with sqrt(interval / 2)
    overshoot_cents: float = 20.0
    scoop_cents: float = 120.0
    scoop_ms: float = 90.0
    vibrato_rate_hz: float = 5.8
    vibrato_depth_cents: float = 45.0
    vibrato_onset_ms: float = 280.0
    vibrato_rate_drift_hz: float = 0.25
    wander_cents: float = 12.0  # slow 1-4 Hz fluctuation
    jitter_cents: float = 3.0  # fast fluctuation
    release_cents: float = -150.0
    release_ms: float = 80.0
    timing_ms: float = 15.0  # onset offset standard deviation
    attack_ms: float = 50.0
    decay: float = 0.15  # loudness drop after the attack
    phrase_arch: float = 0.3  # swell towards the phrase's peak note
    breath_ms: float = 220.0
    phrase_gap_s: float = 0.25  # a rest longer than this ends a phrase

    @classmethod
    def named(cls, name: str) -> HumanStyle:
        presets = {
            "neutral": {},
            "pop": {"vibrato_onset_ms": 350.0, "vibrato_depth_cents": 35.0, "scoop_cents": 150.0, "decay": 0.1},
            "soul": {
                "vibrato_depth_cents": 60.0,
                "vibrato_rate_hz": 6.2,
                "scoop_cents": 200.0,
                "portamento_ms": 120.0,
            },
            "classical": {
                "vibrato_onset_ms": 120.0,
                "vibrato_depth_cents": 70.0,
                "scoop_cents": 40.0,
                "timing_ms": 8.0,
            },
            "reggae": {"vibrato_depth_cents": 30.0, "portamento_ms": 70.0, "timing_ms": 20.0, "phrase_arch": 0.2},
            "ethereal": {
                "vibrato_depth_cents": 25.0,
                "vibrato_onset_ms": 450.0,
                "attack_ms": 120.0,
                "wander_cents": 16.0,
            },
            "aggressive": {"attack_ms": 25.0, "overshoot_cents": 35.0, "vibrato_rate_hz": 6.5, "scoop_cents": 80.0},
        }
        if name not in presets:
            raise ValueError(f"unknown style {name!r}; choose from {', '.join(presets)}")
        return cls(**presets[name])


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0, 1)
    return x * x * (3 - 2 * x)


def _slow_noise(n: int, rate: float, cutoff_hz: float, g: torch.Generator) -> torch.Tensor:
    """White noise low-passed to ``cutoff_hz`` by a moving average, unit variance."""
    window = max(1, int(rate / cutoff_hz)) | 1
    x = torch.randn(n + window, generator=g)
    x = F.avg_pool1d(x[None, None], window, stride=1)[0, 0][:n]
    return x / (x.std() + 1e-6)


def phrases(notes: list[Note], gap_s: float) -> list[list[int]]:
    """Group note indices into phrases separated by rests longer than ``gap_s``."""
    groups: list[list[int]] = []
    for i, (_, start, _) in enumerate(notes):
        if groups and start - notes[groups[-1][-1]][2] <= gap_s:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def humanize(
    notes: list[Note],
    style: HumanStyle | None = None,
    fine_rate: float = 100.0,
    total_seconds: float | None = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """
    ``notes``: (midi, start, end) in seconds, non-overlapping, in time order. Returns curves at ``fine_rate``:
    ``f0_hz`` (0 where silent), ``energy`` (0..1), ``breath`` (0..1), ``note_index`` (-1 for rest), and the
    ``onsets`` actually used (seconds, after the timing offsets).
    """
    style = style or HumanStyle()
    g = torch.Generator().manual_seed(seed)
    notes = sorted(notes, key=lambda n: n[1])
    end = total_seconds if total_seconds is not None else (notes[-1][2] + 0.5 if notes else 1.0)
    n = int(math.ceil(end * fine_rate)) + 1
    t = torch.arange(n) / fine_rate
    cents = torch.zeros(n)
    voiced = torch.zeros(n, dtype=torch.bool)
    energy = torch.zeros(n)
    breath = torch.zeros(n)
    note_index = torch.full((n,), -1, dtype=torch.long)
    onsets = []
    groups = phrases(notes, style.phrase_gap_s)
    vib_phase = 0.0
    for group in groups:
        # a breath before the phrase: noise rises and falls in the gap before the first onset
        first_start = notes[group[0]][1]
        b0, b1 = max(0, int((first_start - style.breath_ms / 1000) * fine_rate)), int(first_start * fine_rate)
        if b1 > b0:
            breath[b0:b1] = torch.sin(torch.linspace(0, math.pi, b1 - b0)).clamp(min=0).pow(0.7)
        peak_note = max(group, key=lambda i: notes[i][0])
        prev_cents = None
        for k, i in enumerate(group):
            midi, start, stop = notes[i]
            offset = float(torch.randn((), generator=g)) * style.timing_ms / 1000
            start = max(0.0, start + offset)
            onsets.append(start)
            a, z = int(start * fine_rate), min(n, int(stop * fine_rate))
            if z <= a + 1:
                continue
            target = (midi - 69) * 100.0
            seg = t[a:z] - t[a]
            seg_ms = seg * 1000
            curve = torch.full((z - a,), target)
            if prev_cents is None:
                # preparation: come from below and settle
                curve = curve - style.scoop_cents * torch.exp(-seg_ms / style.scoop_ms)
            else:
                interval = target - prev_cents
                glide_ms = min(200.0, max(40.0, style.portamento_ms * math.sqrt(abs(interval) / 200.0 + 1e-6)))
                shape = _smoothstep(seg_ms / glide_ms)
                over = math.copysign(style.overshoot_cents, interval) * min(1.0, abs(interval) / 300.0)
                settle = torch.exp(-(seg_ms - glide_ms).clamp(min=0) / 80.0) * (seg_ms >= glide_ms * 0.7).float()
                curve = prev_cents + interval * shape + over * settle
            # vibrato: delayed onset, ramped extent, drifting rate, growing towards the end of the note
            dur_ms = (z - a) / fine_rate * 1000
            if dur_ms > style.vibrato_onset_ms + 120:
                ramp = ((seg_ms - style.vibrato_onset_ms) / 300.0).clamp(0, 1)
                grow = 1 + 0.25 * (seg_ms / dur_ms)
                rate = style.vibrato_rate_hz + style.vibrato_rate_drift_hz * _slow_noise(z - a, fine_rate, 0.7, g)
                phase = vib_phase + 2 * math.pi * torch.cumsum(rate, 0) / fine_rate
                vib_phase = float(phase[-1]) % (2 * math.pi)
                extent = style.vibrato_depth_cents * (0.85 + 0.15 * _slow_noise(z - a, fine_rate, 1.0, g).abs())
                curve = curve + ramp * grow * extent * torch.sin(phase)
            # release at the end of the phrase
            if k == len(group) - 1:
                rel = int(style.release_ms / 1000 * fine_rate)
                if rel > 1 and rel < z - a:
                    fall = torch.linspace(0, 1, rel) ** 2 * style.release_cents
                    curve[-rel:] = curve[-rel:] + fall
            # loudness: attack, small decay, phrase arch, release fade
            attack = _smoothstep(seg_ms / style.attack_ms)
            decay = 1 - style.decay * (1 - torch.exp(-seg_ms / 150.0))
            arch = (
                1 + style.phrase_arch * (1 - abs(k - group.index(peak_note)) / max(1, len(group))) - style.phrase_arch
            )
            fade = torch.ones(z - a)
            tail = int(0.06 * fine_rate)
            if k == len(group) - 1 and tail < z - a:
                fade[-tail:] = torch.linspace(1, 0.2, tail)
            energy[a:z] = (attack * decay * arch * fade).clamp(0.05, 1.0)
            cents[a:z] = curve
            voiced[a:z] = True
            note_index[a:z] = i
            prev_cents = target
    # fine fluctuation everywhere the voice is on
    wander = _slow_noise(n, fine_rate, 3.0, g) * style.wander_cents
    jitter = torch.randn(n, generator=g) * style.jitter_cents
    cents = cents + (wander + jitter) * voiced
    f0 = torch.where(voiced, 440.0 * torch.pow(2.0, cents / 1200.0), torch.zeros(n))
    return {"f0_hz": f0, "energy": energy, "breath": breath, "note_index": note_index, "onsets": torch.tensor(onsets)}


def decimate(curve: torch.Tensor, fine_rate: float, frame_rate: float, frames: int | None = None) -> torch.Tensor:
    """Average-pool a fine-rate curve to the frame rate (zeros stay zeros only if the whole frame is silent)."""
    factor = int(round(fine_rate / frame_rate))
    n = curve.shape[-1] // factor * factor
    out = curve[..., :n].reshape(*curve.shape[:-1], -1, factor)
    voiced = (out > 0).float().mean(-1) > 0.5 if curve.min() >= 0 else torch.ones(out.shape[:-1], dtype=torch.bool)
    pooled = torch.where(voiced, out.sum(-1) / (out > 0).float().sum(-1).clamp(min=1), torch.zeros_like(out[..., 0]))
    if frames is not None:
        pooled = F.pad(pooled, (0, max(0, frames - pooled.shape[-1])))[..., :frames]
    return pooled


def fine_from_frames(curve: torch.Tensor, fine_rate: float, frame_rate: float) -> torch.Tensor:
    """The other way: a frame-rate curve to the fine rate by linear interpolation (voicing kept hard)."""
    factor = int(round(fine_rate / frame_rate))
    x = curve.float()[None, None]
    fine = F.interpolate(x, size=curve.shape[-1] * factor, mode="linear", align_corners=False)[0, 0]
    voiced = F.interpolate((x > 0).float(), size=fine.shape[-1], mode="nearest")[0, 0] > 0.5
    return torch.where(voiced, fine, torch.zeros_like(fine))


def notes_from_plan(plan, analysis, frame_rate: float = 25.0) -> list[Note]:
    """The conductor's half-beat notes as (midi, start, end) with tied repeats merged into held notes."""
    beat_times = analysis.beat_times()
    beat = 60.0 / max(analysis.tempo, 1)
    out: list[Note] = []
    for phrase in plan.phrases:
        for k, half_note in enumerate(phrase.notes):
            b = phrase.start_beat + k / 2
            t0 = (float(beat_times[int(b)]) if int(b) < beat_times.numel() else int(b) * beat) + (b - int(b)) * beat
            t0 += (phrase.swing * beat if k % 2 else 0.0) + phrase.human_offset_ms / 1000
            t1 = t0 + beat / 2
            if half_note < 0:
                continue
            if out and out[-1][0] == half_note and abs(out[-1][2] - t0) < 1e-3:
                out[-1] = (half_note, out[-1][1], t1)
            else:
                out.append((half_note, t0, t1))
    return out
