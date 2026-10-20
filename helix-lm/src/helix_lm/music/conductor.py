# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The conductor: from a song's analysis and the critic's timeline to a vocal plan TriStream can sing.

What it decides, and the rule behind each decision:

* **where to sing** -- by section: verses get the lead line; a build gets a rising line over its last bars;
  the chorus entry lands exactly on the downbeat, with a beat of rest before it (anticipation); a bridge is
  sparse; an instrumental song gets the single strongest moment the critic finds, entered after a rest
  ("the sudden soloist");
* **which notes** -- a contour per phrase (arch or ramp), chord tones on strong beats, scale steps between,
  every note snapped to the key, the whole line fitted to the singer's range by octaves first and
  compression last; the chorus opens with an octave-leap climax figure when the range allows;
* **harmony** -- two parts from chord tones with voice-leading on choruses; none in verses;
* **how** -- sliders mapped from the backing track: quiet sections raise breathiness and whisper and lower
  volume; loud sections raise power, tension and vibrato; a bright track makes the voice warmer (formant ratio
  below one), a dark track brighter; the room follows the track's reverb; an "ethereal" cue adds echo and
  long tails; "aggressive" cuts them;
* **timing** -- notes are quantized softly to the beat grid with a little swing and a human offset (perfect
  quantization is the clearest tell of a synthetic voice; TriStream-1's report said the same);
* **restraint** -- candidate plans are scored by :func:`frisson.score_plan`, which penalizes density, and the
  best natural one wins; interventions are capped per minute.

The plan is explicit data (:class:`VocalPlan`) with a log of why, so a person can read and edit it before
anything is rendered.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch

from .analysis import AnalysisResult, scale_pitch_classes
from .frisson import frisson_peaks, score_plan
from .harmony import climax_figure, fit_to_range, harmonize, nearest_chord_tone, rising_line, snap_to_scale


@dataclass
class Phrase:
    section: str
    start_beat: int
    end_beat: int
    notes: list[int]  # one MIDI note per half-beat, -1 for rest
    chords: list[str]  # one per half-beat
    sliders: dict[str, float]
    technique: str = "plain"
    harmonies: list[list[int]] = field(default_factory=list)
    interventions: list[str] = field(default_factory=list)
    flags: dict = field(default_factory=dict)
    swing: float = 0.08
    human_offset_ms: float = 12.0


@dataclass
class VocalPlan:
    tempo: float
    key: str
    phrases: list[Phrase]
    log: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"{self.tempo:.0f} BPM, {self.key}: {len(self.phrases)} phrases"]
        for p in self.phrases:
            notes = [n for n in p.notes if n >= 0]
            span = f"{min(notes)}-{max(notes)}" if notes else "rest"
            lines.append(f"  {p.section:7} beats {p.start_beat:3}-{p.end_beat:3}  notes {span:7}  {p.technique:8} "
                         f"power {p.sliders['power']:.2f} breath {p.sliders['breathiness']:.2f} "
                         f"{'+' + ','.join(p.interventions) if p.interventions else ''}")  # fmt: skip
        return "\n".join(lines + [f"  why: {w}" for w in self.log])


SLIDERS = ("breathiness", "whisper", "power", "tension", "vibrato", "warmth", "echo", "room", "volume", "brightness")

# Timing rules from the chills literature (docs/TRISTREAM.md): the dopaminergic anticipation window is about 15 s
# before a peak and a chill lasts about 7 s, so planned peaks are kept at least this far apart; the pre-chorus
# ramp runs 4-8 bars; the entry gets a 1-2 beat gap before it; the held top note sits 1-2 beats after the downbeat.
MIN_PEAK_SPACING_SECONDS = 20.0
PRE_CHORUS_RAMP_BARS = (4, 8)
ENTRY_GAP_BEATS = 1


def sliders_for(analysis: AnalysisResult, start_beat: int, end_beat: int) -> dict[str, float]:
    """Slider values for a span from the backing track's loudness, brightness, reverb and vibe cues."""
    beats = analysis.beats
    a = int(beats[min(start_beat, beats.numel() - 1)]) if beats.numel() else 0
    b = int(beats[min(end_beat, beats.numel() - 1)]) if beats.numel() else analysis.rms.shape[0]
    b = max(b, a + 1)
    rms_db = 20 * analysis.rms.clamp(min=1e-6).log10()
    whole = rms_db.quantile(0.95), rms_db.quantile(0.1)
    energy = float(((rms_db[a:b].mean() - whole[1]) / max(float(whole[0] - whole[1]), 1e-6)).clamp(0, 1))
    bright = float((analysis.centroid[a:b].mean() / 4000).clamp(0, 1))
    cues = set(analysis.vibe.get("cues", []))
    sliders = {
        "breathiness": 0.75 - 0.6 * energy,
        "whisper": max(0.0, 0.5 - energy * 1.2),
        "power": 0.2 + 0.75 * energy,
        "tension": 0.1 + 0.7 * energy,
        "vibrato": 0.25 + 0.5 * energy,
        "warmth": 0.35 + 0.5 * bright,  # a bright track wants a warmer voice to balance the spectrum
        "brightness": 0.65 - 0.4 * bright,
        "echo": 0.1,
        "room": min(1.0, analysis.reverb / 2.0),
        "volume": 0.55 + 0.35 * energy,
    }
    if "ethereal" in cues:
        sliders["echo"], sliders["room"], sliders["breathiness"] = (
            0.5,
            max(sliders["room"], 0.6),
            min(1.0, sliders["breathiness"] + 0.15),
        )
    if "aggressive" in cues:
        sliders["echo"], sliders["room"], sliders["tension"] = (
            0.0,
            min(sliders["room"], 0.2),
            min(1.0, sliders["tension"] + 0.2),
        )
    if "lo-fi" in cues:
        sliders["brightness"] = min(sliders["brightness"], 0.4)
    return {k: float(min(1.0, max(0.0, v))) for k, v in sliders.items()}


def technique_for(sliders: dict[str, float], section: str) -> str:
    if sliders["whisper"] > 0.3:
        return "breathy"
    if sliders["power"] > 0.8 and section in ("chorus", "drop"):
        return "belting"
    if sliders["power"] > 0.65:
        return "mixed"
    if sliders["vibrato"] > 0.6:
        return "vibrato"
    return "plain"


def _contour(length: int, shape: str, low: int, high: int) -> list[float]:
    if length <= 1:
        return [float((low + high) / 2)]
    xs = [i / (length - 1) for i in range(length)]
    if shape == "arch":
        return [low + (high - low) * math.sin(math.pi * x) for x in xs]
    if shape == "ramp":
        return [low + (high - low) * x for x in xs]
    if shape == "fall":
        return [high - (high - low) * x for x in xs]
    return [float((low + high) / 2)] * length


def melody_for(
    chords: list[str], tonic: int, mode: str, low: int, high: int, shape: str, rng: random.Random, rest_every: int = 8
) -> list[int]:
    """A line over ``chords`` (one per half-beat): chord tones on beats, scale steps off, following a contour."""
    n = len(chords)
    target = _contour(n, shape, low + 3, high - 3)
    notes, previous = [], None
    for i, chord in enumerate(chords):
        if rest_every and i % rest_every == rest_every - 1:
            notes.append(-1)
            continue
        wish = int(round(target[i] + rng.uniform(-2, 2)))
        note = nearest_chord_tone(wish, chord) if i % 2 == 0 else snap_to_scale(wish, tonic, mode)
        if previous is not None and abs(note - previous) > 7 and i % 4 != 0:  # resolve big leaps by step
            note = snap_to_scale(previous + (2 if note > previous else -2), tonic, mode)
        notes.append(note)
        previous = note
    return notes


def plan_vocals(
    analysis: AnalysisResult,
    singer_range: tuple[int, int] = (55, 79),
    density: str = "normal",
    seed: int = 0,
    allow_compress: bool = True,
) -> VocalPlan:
    """
    The plan for one song. ``density`` is ``"sparse"`` (choruses and one climax only), ``"normal"`` or
    ``"full"`` (verses too, harmonies on every chorus). The two or three candidate plans differ in their
    interventions; the one the critic scores best after the naturalness penalties is returned.
    """
    rng = random.Random(seed)
    tonic, mode, _ = analysis.key
    key_name = analysis.key_name
    chords = analysis.chords
    n_beats = len(chords)
    low, high = singer_range
    half_chords = [c for c in chords for _ in range(2)]
    candidates: list[VocalPlan] = []
    for variant in ("plain", "frisson", "frisson+harmony"):
        phrases, log = [], []
        for seg in analysis.segments:
            a, b = seg["start_beat"], min(seg["end_beat"], n_beats)
            if b - a < 4:
                continue
            label = seg["label"]
            sliders = sliders_for(analysis, a, b)
            technique = technique_for(sliders, label)
            interventions: list[str] = []
            if label in ("intro", "outro"):
                continue
            if label == "verse" and density == "sparse":
                continue
            if label == "bridge" and density != "full":
                continue
            start, end = a, b
            if label == "build":
                if variant == "plain":
                    continue
                start = max(a, b - 8)
                notes = rising_line(low + 4, min(high - 2, low + 16), (end - start) * 2, tonic, mode)
                interventions += ["rising_line"]
                log.append(f"build at beat {a}: rising line over the last {end - start} beats into the chorus")
            elif label in ("chorus", "drop"):
                start = a
                notes = melody_for(half_chords[start * 2 : end * 2], tonic, mode, low, high, "arch", rng)
                if variant != "plain" and high - low >= 14:
                    figure = climax_figure(low + 7, chords[start] if start < n_beats else "C", tonic, mode)
                    notes[: len(figure)] = figure
                    interventions += ["octave_leap", "held_high_note", "pre_entry_rest"]
                    log.append(f"chorus at beat {a}: octave-leap climax on the downbeat, a beat of rest before it")
                if variant != "plain" and sliders_for(analysis, max(0, a - 8), a)["breathiness"] > 0.5:
                    interventions.append("breathy_to_belt")
                technique = "belting" if sliders["power"] > 0.7 else technique
            else:  # verse / bridge
                notes = melody_for(
                    half_chords[start * 2 : end * 2],
                    tonic,
                    mode,
                    low,
                    high - 5,
                    "arch" if label == "verse" else "fall",
                    rng,
                )
                technique = "breathy" if label == "bridge" else technique
            notes, how = fit_to_range(notes, low, high, allow_compress)
            flags = {
                "out_of_range": how.endswith("still out of range"),
                "out_of_key": any(n >= 0 and n % 12 not in scale_pitch_classes(tonic, mode) for n in notes),
            }
            if how != "as is":
                log.append(f"{label} at beat {a}: line moved into the singer's range ({how})")
            phrase = Phrase(
                label,
                start,
                end,
                notes,
                half_chords[start * 2 : end * 2],
                sliders,
                technique,
                [],
                interventions,
                flags,
            )
            if variant == "frisson+harmony" and label in ("chorus", "drop"):
                lead = list(notes)
                phrase.harmonies = harmonize(lead, phrase.chords, parts=2, tonic=tonic, mode=mode)
                phrase.interventions.append("harmony_entry")
                log.append(f"chorus at beat {a}: two harmony parts below the lead from chord tones")
            if label in ("chorus", "drop") and not any(p.section in ("chorus", "drop") for p in phrases):
                phrase.interventions.append("voice_entry")
            phrases.append(phrase)
        if not phrases:  # an instrumental: the single strongest moment, after a rest
            peaks = frisson_peaks(analysis, top=1)
            if peaks:
                b0 = max(0, peaks[0]["beat"] - peaks[0]["beat"] % 4)
                end = min(n_beats, b0 + 8)
                sliders = sliders_for(analysis, b0, end)
                notes = melody_for(half_chords[b0 * 2 : end * 2], tonic, mode, low, high, "arch", rng)
                notes, _ = fit_to_range(notes, low, high, allow_compress)
                phrases.append(
                    Phrase(
                        "solo",
                        b0,
                        end,
                        notes,
                        half_chords[b0 * 2 : end * 2],
                        sliders,
                        technique_for(sliders, "chorus"),
                        [],
                        ["voice_entry", "pre_entry_rest"],
                        {},
                    )
                )
                log.append(
                    f"instrumental: the soloist enters at beat {b0}, the strongest moment the critic found "
                    f"({', '.join(peaks[0]['because'])})"
                )
        candidates.append(VocalPlan(analysis.tempo, key_name, phrases, log))
    beat_seconds = 60.0 / max(analysis.tempo, 1)
    for plan in candidates:  # keep planned peaks at least MIN_PEAK_SPACING_SECONDS apart
        last_peak = -1e9
        for phrase in plan.phrases:
            if "octave_leap" in phrase.interventions:
                if (phrase.start_beat - last_peak) * beat_seconds < MIN_PEAK_SPACING_SECONDS:
                    phrase.interventions = [
                        t for t in phrase.interventions if t not in ("octave_leap", "held_high_note")
                    ]
                else:
                    last_peak = phrase.start_beat
    scored = [(score_plan(analysis, plan)["total"], i, plan) for i, plan in enumerate(candidates)]
    scored.sort(key=lambda t: -t[0])
    best = scored[0][2]
    best.log.append(f"chose variant {scored[0][1]} of {len(candidates)} (critic score {scored[0][0]:.2f})")
    return best


# ------------------------------------------------------------------------------------- to controls --
def render_controls(
    plan: VocalPlan, analysis: AnalysisResult, frame_rate: float = 25.0, vibrato_depth: float = 0.35
) -> dict[str, torch.Tensor]:
    """
    Per-frame control streams for TriStream: ``f0_hz`` (0 where silent; vibrato ramps in on held notes, a
    scoop into each new note), ``note`` (MIDI or 128 for rest), ``technique`` ids, ``harmony_f0`` (parts,
    frames), and one curve per slider, all at ``frame_rate``; notes are placed on the beat grid with swing
    and a small human offset.
    """
    beat_times = analysis.beat_times()
    if beat_times.numel() == 0:
        beat_times = torch.arange(1) * 0.0
    total_seconds = float(beat_times[-1]) + 60.0 / max(analysis.tempo, 1) * 2 if beat_times.numel() else 1.0
    frames = int(total_seconds * frame_rate) + 1
    f0 = torch.zeros(frames)
    note = torch.full((frames,), 128, dtype=torch.long)
    technique = torch.zeros(frames, dtype=torch.long)
    parts = max((len(p.harmonies) for p in plan.phrases), default=0)
    harmony = torch.zeros(parts, frames)
    curves = {k: torch.zeros(frames) for k in SLIDERS}
    techniques = ("plain", "breathy", "falsetto", "vibrato", "glissando", "mixed", "belting", "pharyngeal")
    beat = 60.0 / max(analysis.tempo, 1)
    for phrase in plan.phrases:
        tech_id = techniques.index(phrase.technique) if phrase.technique in techniques else 0
        for k, half_note in enumerate(phrase.notes):
            b = phrase.start_beat + k / 2
            swing = phrase.swing * beat if k % 2 else 0.0
            t0 = (
                (float(beat_times[int(b)]) if int(b) < beat_times.numel() else int(b) * beat)
                + (b - int(b)) * beat
                + swing
                + phrase.human_offset_ms / 1000
            )
            t1 = t0 + beat / 2
            a, z = int(t0 * frame_rate), min(frames, int(t1 * frame_rate))
            if z <= a:
                continue
            for name, value in phrase.sliders.items():
                curves[name][a:z] = value
            technique[a:z] = tech_id
            if half_note < 0:
                continue
            hz = 440.0 * 2 ** ((half_note - 69) / 12)
            seg = torch.arange(z - a) / frame_rate
            scoop = torch.pow(
                2.0, -1.5 / 12 * torch.exp(-seg / 0.06)
            )  # start a semitone and a half under, settle fast
            vib = torch.pow(
                2.0,
                vibrato_depth
                / 12
                * torch.clamp(seg / 0.3, max=1)
                * torch.sin(2 * math.pi * 5.5 * seg)
                * phrase.sliders["vibrato"],
            )
            f0[a:z] = hz * scoop * vib
            note[a:z] = half_note
            for p, line in enumerate(phrase.harmonies):
                if k < len(line) and line[k] >= 0:
                    harmony[p, a:z] = 440.0 * 2 ** ((line[k] - 69) / 12) * vib
    return {"f0_hz": f0, "note": note, "technique": technique, "harmony_f0": harmony, **curves}
