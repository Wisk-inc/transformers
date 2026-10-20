# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Music analysis from the waveform, in PyTorch, with the standard algorithms written out:

* **tempo and beats**: a spectral-flux onset envelope, a tempo estimate from its autocorrelation under a
  log-normal tempo prior, and Ellis's (2007) dynamic-programming beat tracker (what ``librosa.beat`` does);
* **downbeats**: the beat phase whose candidates carry the most onset energy and harmonic change (a meter-4
  heuristic; a trained downbeat model such as Beat This! is better and can replace it);
* **key**: Krumhansl-Schmuckler profile correlation on the chroma;
* **chords and tension**: triad template matching per beat; tension as sensory roughness (Plomp-Levelt
  intervals on the chroma) plus harmonic distance from the key; resolutions as tension falling onto a
  diatonic tonic/subdominant/dominant chord;
* **structure**: Foote's checkerboard novelty on a beat-synchronous self-similarity matrix, segments labelled
  by energy rank and position (intro, verse, build, chorus, drop, bridge, outro);
* **reverb**: a decay-time proxy from the energy decay after onsets;
* **groove**: syncopation (energy on off-beats relative to on-beats) and beat regularity;
* a **vibe vector** (tempo, brightness, density, percussiveness, dynamic range, mode, reverb, syncopation)
  and the named cues a conductor reads from it ("ethereal", "aggressive", "acoustic", "one-drop", ...).

Every function takes a mono float waveform and a sample rate. Accuracy is that of the classical methods:
good enough to drive a vocal plan, not a replacement for trained MIR models where those are available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..audio.codec import mel_filterbank

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR_PROFILE = torch.tensor([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = torch.tensor([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)
# Plomp-Levelt style roughness of each pitch-class interval (0 = unison ... 6 = tritone), normalized.
INTERVAL_ROUGHNESS = torch.tensor([0.0, 1.0, 0.55, 0.3, 0.2, 0.1, 0.8])


# ------------------------------------------------------------------------------------------- spectra --
def stft_magnitude(wave: torch.Tensor, n_fft: int = 2048, hop: int = 512) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=wave.device)
    return torch.stft(wave, n_fft, hop_length=hop, window=window, return_complex=True, center=True).abs()


def log_mel(wave: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop: int = 512, n_mels: int = 80) -> torch.Tensor:
    """``(n_mels, frames)`` log-mel spectrogram."""
    mag = stft_magnitude(wave, n_fft, hop)
    bank = mel_filterbank(sample_rate, n_fft, n_mels, wave.device)
    return (bank @ mag).clamp(min=1e-6).log()


def chroma(wave: torch.Tensor, sample_rate: int, n_fft: int = 4096, hop: int = 512) -> torch.Tensor:
    """``(12, frames)`` pitch-class energy, from the STFT bins between 60 Hz and 5 kHz, normalized per frame."""
    mag = stft_magnitude(wave, n_fft, hop)
    freqs = torch.fft.rfftfreq(n_fft, 1 / sample_rate).to(wave.device)
    keep = (freqs > 60) & (freqs < 5000)
    midi = 69 + 12 * torch.log2(freqs[keep] / 440.0)
    classes = torch.round(midi).long() % 12
    weights = mag[keep] ** 2
    out = torch.zeros(12, mag.shape[1], device=wave.device)
    out.index_add_(0, classes, weights)
    return out / (out.sum(0, keepdim=True) + 1e-9)


# -------------------------------------------------------------------------------------- onsets, tempo --
def onset_envelope(wave: torch.Tensor, sample_rate: int, hop: int = 512) -> torch.Tensor:
    """Spectral flux: the mean positive change of the log-mel spectrogram, one value per frame."""
    mel = log_mel(wave, sample_rate, hop=hop)
    flux = (mel[:, 1:] - mel[:, :-1]).clamp(min=0).mean(0)
    flux = F.pad(flux, (1, 0))
    return flux / (flux.max() + 1e-9)


def estimate_tempo(
    onset: torch.Tensor, sample_rate: int, hop: int = 512, bpm_range=(50.0, 220.0), prior_bpm: float = 120.0
) -> float:
    """BPM from the autocorrelation of the onset envelope, weighted by a log-normal prior around ``prior_bpm``."""
    x = onset - onset.mean()
    n = x.shape[0]
    spectrum = torch.fft.rfft(x, n=2 * n)
    acf = torch.fft.irfft(spectrum * spectrum.conj(), n=2 * n)[:n]
    acf = acf / (acf[0] + 1e-9)
    frame_rate = sample_rate / hop
    lags = torch.arange(1, n, device=onset.device).float()
    bpms = 60.0 * frame_rate / lags
    prior = torch.exp(-0.5 * ((torch.log2(bpms / prior_bpm)) / 1.0) ** 2)
    score = acf[1:] * prior
    score[(bpms < bpm_range[0]) | (bpms > bpm_range[1])] = -1
    best = int(score.argmax())
    return float(bpms[best])


def track_beats(
    onset: torch.Tensor, bpm: float, sample_rate: int, hop: int = 512, tightness: float = 100.0
) -> torch.Tensor:
    """
    Ellis's dynamic-programming beat tracker: beats maximize onset strength while keeping their spacing close
    to the tempo period (a log-squared penalty scaled by ``tightness``). Returns frame indices.
    """
    period = 60.0 * sample_rate / hop / bpm
    n = onset.shape[0]
    local = onset.clone()
    window = torch.arange(-int(2 * period), -int(period / 2) + 1, device=onset.device)
    window = window[window < 0]
    penalty = -tightness * (torch.log(-window.float() / period)) ** 2
    score = torch.zeros(n, device=onset.device)
    back = torch.full((n,), -1, dtype=torch.long, device=onset.device)
    for t in range(n):
        candidates = t + window
        valid = candidates >= 0
        if not valid.any():
            score[t] = local[t]
            continue
        cand = candidates[valid]
        total = score[cand] + penalty[valid]
        best = int(total.argmax())
        score[t] = local[t] + total[best]
        back[t] = cand[best]
    # Trace back from the best final beat within the last period.
    tail_start = max(0, n - int(period))
    t = tail_start + int(score[tail_start:].argmax())
    beats = []
    while t >= 0:
        beats.append(t)
        t = int(back[t])
    return torch.tensor(beats[::-1], dtype=torch.long)


def downbeats(beats: torch.Tensor, onset: torch.Tensor, chroma_frames: torch.Tensor, meter: int = 4) -> torch.Tensor:
    """The beat phase (0..meter-1) whose beats carry the most onset energy and chroma change; returns their indices."""
    if beats.numel() < meter:
        return beats[:1]
    change = torch.zeros(chroma_frames.shape[1], device=onset.device)
    change[1:] = (chroma_frames[:, 1:] - chroma_frames[:, :-1]).abs().sum(0)
    best_phase, best_score = 0, -1.0
    for phase in range(meter):
        idx = beats[phase::meter]
        score = float(onset[idx].mean() + change[idx].mean())
        if score > best_score:
            best_phase, best_score = phase, score
    return beats[best_phase::meter]


# --------------------------------------------------------------------------------------- key, chords --
def estimate_key(chroma_frames: torch.Tensor) -> tuple[int, str, float]:
    """Krumhansl-Schmuckler: correlate the mean chroma with the 24 rotated profiles -> (tonic, mode, confidence)."""
    mean = chroma_frames.mean(1)
    best = (0, "major", -2.0)
    scores = []
    for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
        for tonic in range(12):
            rotated = torch.roll(profile.to(mean.device), tonic)
            r = torch.corrcoef(torch.stack([mean, rotated]))[0, 1].item()
            scores.append(r)
            if r > best[2]:
                best = (tonic, mode, r)
    scores.sort(reverse=True)
    confidence = max(0.0, min(1.0, (scores[0] - scores[1]) * 5 + 0.5))
    return best[0], best[1], confidence


def scale_pitch_classes(tonic: int, mode: str) -> tuple[int, ...]:
    steps = MAJOR_SCALE if mode == "major" else MINOR_SCALE
    return tuple((tonic + s) % 12 for s in steps)


def snap_midi_to_scale(midi: torch.Tensor, tonic: int, mode: str) -> torch.Tensor:
    """Move every note to the nearest scale degree (ties go down), so a planned line never sings a wrong note."""
    allowed = torch.tensor(scale_pitch_classes(tonic, mode))
    out = midi.clone()
    for i, note in enumerate(midi.tolist()):
        if note < 0:
            continue
        candidates = torch.arange(int(note) - 6, int(note) + 7)
        ok = candidates[torch.isin(candidates % 12, allowed)]
        distances = (ok - note).abs()
        out[i] = ok[int(distances.argmin())]
    return out


def chord_templates() -> tuple[list[str], torch.Tensor]:
    names, rows = [], []
    for tonic in range(12):
        for quality, intervals in (("", (0, 4, 7)), ("m", (0, 3, 7))):
            row = torch.zeros(12)
            for k, i in enumerate(intervals):
                row[(tonic + i) % 12] = 1.0 if k else 1.2
            names.append(NOTE_NAMES[tonic] + quality)
            rows.append(row / row.sum())
    return names, torch.stack(rows)


def chords_per_beat(chroma_frames: torch.Tensor, beats: torch.Tensor) -> list[str]:
    names, templates = chord_templates()
    templates = templates.to(chroma_frames.device)
    out = []
    bounds = beats.tolist() + [chroma_frames.shape[1]]
    for a, b in zip(bounds[:-1], bounds[1:], strict=False):
        segment = chroma_frames[:, a : max(b, a + 1)].mean(1)
        out.append(names[int((templates @ segment).argmax())])
    return out


def roughness(chroma_frames: torch.Tensor) -> torch.Tensor:
    """Sensory dissonance per frame: pairwise pitch-class energies weighted by interval roughness."""
    table = INTERVAL_ROUGHNESS.to(chroma_frames.device)
    c = chroma_frames
    total = torch.zeros(c.shape[1], device=c.device)
    for i in range(12):
        for j in range(i + 1, 12):
            interval = min((j - i) % 12, (i - j) % 12)
            total += table[interval] * c[i] * c[j]
    return total / (total.max() + 1e-9)


def tension_curve(chroma_frames: torch.Tensor, tonic: int, mode: str) -> torch.Tensor:
    """Roughness plus the energy outside the key, smoothed: high before a resolution, low on the tonic."""
    allowed = torch.zeros(12, device=chroma_frames.device)
    allowed[list(scale_pitch_classes(tonic, mode))] = 1.0
    outside = (chroma_frames * (1 - allowed)[:, None]).sum(0)
    tonic_weight = chroma_frames[tonic]
    raw = 0.5 * roughness(chroma_frames) + 0.35 * outside + 0.15 * (1 - tonic_weight)
    kernel = torch.ones(1, 1, 9, device=raw.device) / 9
    return F.conv1d(raw[None, None], kernel, padding=4)[0, 0]


def resolutions(
    tension: torch.Tensor, chords: list[str], beats: torch.Tensor, tonic: int, mode: str, drop: float = 0.15
) -> list[int]:
    """Beat indices where tension falls by at least ``drop`` onto a diatonic tonic / dominant / subdominant chord."""
    home = {
        NOTE_NAMES[tonic] + ("" if mode == "major" else "m"),
        NOTE_NAMES[(tonic + 7) % 12],
        NOTE_NAMES[(tonic + 5) % 12] + ("" if mode == "major" else "m"),
    }
    events = []
    for i in range(1, len(chords)):
        if chords[i] in home and beats[i] < tension.shape[0]:
            before = tension[max(0, beats[i - 1]) : beats[i]].mean() if beats[i] > beats[i - 1] else tension[beats[i]]
            after = tension[beats[i] : min(tension.shape[0], beats[i] + 8)].mean()
            if before - after >= drop * max(float(before), 1e-6) and before > after:
                events.append(i)
    return events


# ---------------------------------------------------------------------------------- loudness, colour --
def rms(wave: torch.Tensor, hop: int = 512, n_fft: int = 2048) -> torch.Tensor:
    frames = F.pad(wave, (n_fft // 2, n_fft // 2)).unfold(-1, n_fft, hop)
    return frames.pow(2).mean(-1).sqrt()


def spectral_centroid(wave: torch.Tensor, sample_rate: int, hop: int = 512, n_fft: int = 2048) -> torch.Tensor:
    mag = stft_magnitude(wave, n_fft, hop)
    freqs = torch.fft.rfftfreq(n_fft, 1 / sample_rate).to(wave.device)
    return (mag * freqs[:, None]).sum(0) / (mag.sum(0) + 1e-9)


def spectral_rolloff(
    wave: torch.Tensor, sample_rate: int, percent: float = 0.85, hop: int = 512, n_fft: int = 2048
) -> torch.Tensor:
    mag = stft_magnitude(wave, n_fft, hop)
    cumulative = mag.cumsum(0) / (mag.sum(0, keepdim=True) + 1e-9)
    idx = (cumulative >= percent).float().argmax(0)
    return torch.fft.rfftfreq(n_fft, 1 / sample_rate).to(wave.device)[idx]


def reverb_time(wave: torch.Tensor, sample_rate: int, hop: int = 512) -> float:
    """
    A decay-time proxy (seconds): in the band above 2 kHz, where sustained tones carry little energy, the time
    the level takes to fall 20 dB after each strong onset, median over onsets, extrapolated to 60 dB and capped
    at 4 s. Dry vocals give about 0.1-0.3 s, a hall 1.5 s and more. Sustained harmonic content fools any
    blind estimate; treat the number as a hint.
    """
    n_fft = 2048
    mag = stft_magnitude(wave, n_fft, hop)
    freqs = torch.fft.rfftfreq(n_fft, 1 / sample_rate).to(wave.device)
    high = mag[freqs > 2000].pow(2).mean(0).clamp(min=1e-10)
    env = 10 * high.log10()
    onset = onset_envelope(wave, sample_rate, hop)[: env.shape[0]]
    peaks = (onset > onset.mean() + 2 * onset.std()).nonzero().flatten()
    times = []
    for p in peaks.tolist():
        level = env[p]
        below = (env[p:] < level - 20).nonzero()
        if below.numel():
            times.append(min(4.0, int(below[0]) * hop / sample_rate * 3))
    return float(torch.tensor(times).median()) if times else 0.0


def groove(onset: torch.Tensor, beats: torch.Tensor) -> tuple[float, float]:
    """``(syncopation, regularity)``: off-beat onset energy relative to on-beat, and how even the beat spacing is."""
    if beats.numel() < 3:
        return 0.0, 0.0
    on = onset[beats].mean()
    mid = (beats[:-1] + beats[1:]) // 2
    off = onset[mid].mean()
    syncopation = float(off / (on + off + 1e-9))
    gaps = (beats[1:] - beats[:-1]).float()
    regularity = float(1.0 - (gaps.std() / (gaps.mean() + 1e-9)).clamp(max=1.0))
    return syncopation, regularity


# ------------------------------------------------------------------------------------------ structure --
def beat_synchronous(features: torch.Tensor, beats: torch.Tensor) -> torch.Tensor:
    bounds = beats.tolist() + [features.shape[1]]
    cols = [features[:, a : max(b, a + 1)].mean(1) for a, b in zip(bounds[:-1], bounds[1:], strict=False)]
    return torch.stack(cols, dim=1) if cols else features.mean(1, keepdim=True)


def novelty(features: torch.Tensor, kernel: int = 8) -> torch.Tensor:
    """Foote's checkerboard novelty along the diagonal of the cosine self-similarity matrix."""
    x = F.normalize(features, dim=0)
    ssm = x.T @ x
    n = ssm.shape[0]
    half = kernel // 2
    sign = torch.cat([-torch.ones(half), torch.ones(half)])
    checker = sign[:, None] * sign[None, :]
    gauss = torch.exp(-((torch.arange(kernel) - half + 0.5) ** 2) / (2 * (half / 2) ** 2))
    checker = checker * gauss[:, None] * gauss[None, :]
    out = torch.zeros(n)
    for i in range(half, n - half):
        out[i] = (ssm[i - half : i + half, i - half : i + half] * checker).sum()
    return out.clamp(min=0) / (out.max() + 1e-9)


def segment_structure(
    beat_features: torch.Tensor, beat_energy: torch.Tensor, min_beats: int = 8, threshold: float = 0.3
) -> list[dict]:
    """
    Boundaries at novelty peaks; each segment gets its mean energy, an energy rank, and a label from position
    and energy: the loudest segments are ``chorus``, a rising segment right before one is a ``build``, the beat
    where a quiet-to-loud jump lands is the ``drop``, quiet segments between are ``verse``, the first and last
    ``intro`` / ``outro``, and a late low-energy segment after a chorus the ``bridge``.
    """
    n = beat_features.shape[1]
    nov = novelty(beat_features)
    peaks = [i for i in range(1, n - 1) if nov[i] >= threshold and nov[i] >= nov[i - 1] and nov[i] >= nov[i + 1]]
    bounds = [0]
    for p in peaks:
        if p - bounds[-1] >= min_beats:
            bounds.append(p)
    bounds.append(n)
    segments = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=False):
        segments.append({"start_beat": a, "end_beat": b, "energy": float(beat_energy[a:b].mean())})
    energies = sorted(s["energy"] for s in segments)
    loud = energies[int(0.7 * (len(energies) - 1))] if energies else 0.0
    for s in segments:
        s["label"] = "chorus" if s["energy"] >= loud and len(segments) > 1 else "verse"
    if segments:
        segments[0]["label"] = "intro" if len(segments) > 2 and segments[0]["energy"] < loud else segments[0]["label"]
        if len(segments) > 2 and segments[-1]["energy"] < loud:
            segments[-1]["label"] = "outro"
    for i in range(1, len(segments)):
        prev, cur = segments[i - 1], segments[i]
        if cur["label"] == "chorus" and prev["label"] in ("verse", "intro"):
            a, b = prev["start_beat"], prev["end_beat"]
            rising = beat_energy[(a + b) // 2 : b].mean() > beat_energy[a : (a + b) // 2].mean() * 1.15
            if rising and b - a >= min_beats:
                prev["label"] = "build"
            cur["drop_beat"] = cur["start_beat"]
    for i in range(2, len(segments) - 1):
        if (
            segments[i]["label"] == "verse"
            and segments[i - 1]["label"] == "chorus"
            and segments[i + 1]["label"] == "chorus"
        ):
            segments[i]["label"] = "bridge"
    return segments


# ----------------------------------------------------------------------------------------------- vibe --
VIBE_CUES = {
    "ethereal": "slow, bright and airy, little percussion, long reverb",
    "aggressive": "fast, loud, dense, bright, dry",
    "warm acoustic": "moderate tempo, dark spectrum, light percussion, wide dynamics",
    "dancehall": "90-110 BPM, strong syncopation, heavy low end",
    "reggae": "60-90 BPM, off-beat accents, moderate density",
    "ballad": "slow, minor or major, wide dynamics, moderate reverb",
    "club": "120-135 BPM, very regular beat, high density",
    "lo-fi": "dark, muffled spectrum, soft dynamics, slow",
}


def vibe(
    tempo: float,
    brightness: float,
    density: float,
    percussive: float,
    dynamic_range: float,
    mode: str,
    reverb: float,
    syncopation: float,
    regularity: float,
) -> dict:
    """A vector of descriptors in [0, 1] and the named cues that fit; the conductor maps these to vocal controls."""
    vec = {
        "tempo": max(0.0, min(1.0, (tempo - 50) / 150)),
        "brightness": max(0.0, min(1.0, brightness / 4000)),
        "density": max(0.0, min(1.0, density)),
        "percussive": max(0.0, min(1.0, percussive)),
        "dynamic_range": max(0.0, min(1.0, dynamic_range / 30)),
        "minor": 1.0 if mode == "minor" else 0.0,
        "reverb": max(0.0, min(1.0, reverb / 2.0)),
        "syncopation": max(0.0, min(1.0, syncopation * 2)),
        "regularity": max(0.0, min(1.0, regularity)),
    }
    cues = []
    if tempo < 95 and vec["brightness"] > 0.4 and vec["percussive"] < 0.4 and vec["reverb"] > 0.4:
        cues.append("ethereal")
    if tempo > 130 and vec["density"] > 0.6 and vec["brightness"] > 0.5 and vec["reverb"] < 0.3:
        cues.append("aggressive")
    if vec["brightness"] < 0.35 and vec["percussive"] < 0.5 and vec["dynamic_range"] > 0.4:
        cues.append("warm acoustic")
    if 88 <= tempo <= 112 and vec["syncopation"] > 0.5:
        cues.append("dancehall")
    if 60 <= tempo < 92 and vec["syncopation"] > 0.45 and vec["density"] < 0.6:
        cues.append("reggae")
    if tempo < 85 and vec["dynamic_range"] > 0.45:
        cues.append("ballad")
    if 118 <= tempo <= 136 and vec["regularity"] > 0.85 and vec["density"] > 0.5:
        cues.append("club")
    if vec["brightness"] < 0.25 and vec["dynamic_range"] < 0.3 and tempo < 100:
        cues.append("lo-fi")
    return {"vector": vec, "cues": cues or ["neutral"]}


# ---------------------------------------------------------------------------------------------- whole --
@dataclass
class AnalysisResult:
    sample_rate: int
    hop: int
    tempo: float
    beats: torch.Tensor  # frame indices
    downbeats: torch.Tensor
    key: tuple[int, str, float]
    chords: list[str]
    tension: torch.Tensor  # per frame
    resolutions: list[int]  # beat indices
    rms: torch.Tensor
    centroid: torch.Tensor
    segments: list[dict]
    reverb: float
    syncopation: float
    regularity: float
    vibe: dict = field(default_factory=dict)

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop

    def beat_times(self) -> torch.Tensor:
        return self.beats.float() / self.frame_rate

    def downbeat_times(self) -> torch.Tensor:
        return self.downbeats.float() / self.frame_rate

    @property
    def key_name(self) -> str:
        tonic, mode, _ = self.key
        return f"{NOTE_NAMES[tonic]} {mode}"

    def describe(self) -> str:
        labels = ", ".join(f"{s['label']}@{s['start_beat']}" for s in self.segments)
        return (
            f"{self.tempo:.0f} BPM, {self.key_name} (confidence {self.key[2]:.2f}), {len(self.beats)} beats, "
            f"{len(self.downbeats)} bars, reverb {self.reverb:.2f}s, syncopation {self.syncopation:.2f}, "
            f"cues {self.vibe.get('cues')}; sections: {labels}"
        )


def analyze(wave: torch.Tensor, sample_rate: int, hop: int = 512) -> AnalysisResult:
    """Everything above on one mono waveform."""
    wave = wave.float()
    if wave.dim() > 1:
        wave = wave.mean(0)
    onset = onset_envelope(wave, sample_rate, hop)
    tempo = estimate_tempo(onset, sample_rate, hop)
    beats = track_beats(onset, tempo, sample_rate, hop)
    chroma_frames = chroma(wave, sample_rate, hop=hop)
    frames = min(onset.shape[0], chroma_frames.shape[1])
    onset, chroma_frames = onset[:frames], chroma_frames[:, :frames]
    beats = beats[beats < frames]
    bars = downbeats(beats, onset, chroma_frames)
    key = estimate_key(chroma_frames)
    chords = chords_per_beat(chroma_frames, beats) if beats.numel() else []
    tension = tension_curve(chroma_frames, key[0], key[1])
    events = resolutions(tension, chords, beats, key[0], key[1]) if chords else []
    loudness = rms(wave, hop)[:frames]
    centroid = spectral_centroid(wave, sample_rate, hop)[:frames]
    mel = log_mel(wave, sample_rate, hop=hop)[:, :frames]
    beat_feats = (
        beat_synchronous(torch.cat([chroma_frames, mel / 10], dim=0), beats)
        if beats.numel()
        else chroma_frames.mean(1, keepdim=True)
    )
    beat_energy = beat_synchronous(loudness[None], beats)[0] if beats.numel() else loudness.mean()[None]
    segments = (
        segment_structure(beat_feats, beat_energy)
        if beats.numel() > 8
        else [{"start_beat": 0, "end_beat": int(beats.numel()), "energy": float(loudness.mean()), "label": "verse"}]
    )
    reverb = reverb_time(wave, sample_rate, hop)
    syncopation, regularity = groove(onset, beats)
    db = 20 * loudness.clamp(min=1e-6).log10()
    dynamic_range = float(db.quantile(0.95) - db.quantile(0.1))
    percussive = float(onset.mean() * 4)
    density = float((mel > mel.mean()).float().mean())
    result = AnalysisResult(
        sample_rate, hop, tempo, beats, bars, key, chords, tension, events, loudness, centroid, segments,
        reverb, syncopation, regularity,
    )  # fmt: skip
    result.vibe = vibe(
        tempo, float(centroid.mean()), density, percussive, dynamic_range, key[1], reverb, syncopation, regularity
    )
    return result


def synthetic_song(
    seconds: float = 12.0,
    sample_rate: int = 16000,
    bpm: float = 120.0,
    key: int = 7,
    minor: bool = False,
    seed: int = 0,
) -> torch.Tensor:
    """
    A test song: a kick on every beat, a hat on the off-beats, a bass note and a triad per bar following
    I-V-vi-IV in the key, quiet for the first half and loud for the second (a verse and a chorus).
    """
    g = torch.Generator().manual_seed(seed)
    n = int(seconds * sample_rate)
    t = torch.arange(n) / sample_rate
    out = torch.zeros(n)
    beat = 60.0 / bpm
    scale = MINOR_SCALE if minor else MAJOR_SCALE
    degrees = [0, 4, 5, 3]  # I V vi IV
    for b in range(int(seconds / beat)):
        start = int(b * beat * sample_rate)
        bar = b // 4
        loud = 1.0 if t[start] >= seconds / 2 else 0.45
        kick = torch.exp(-torch.arange(int(0.12 * sample_rate)) / (0.03 * sample_rate)) * torch.sin(
            2 * math.pi * 55 * torch.arange(int(0.12 * sample_rate)) / sample_rate
        )
        out[start : start + kick.shape[0]] += 0.8 * loud * kick[: n - start]
        hat_start = start + int(beat / 2 * sample_rate)
        if hat_start < n:
            hat = torch.randn(int(0.03 * sample_rate), generator=g) * torch.exp(
                -torch.arange(int(0.03 * sample_rate)) / (0.008 * sample_rate)
            )
            out[hat_start : hat_start + hat.shape[0]] += 0.15 * loud * hat[: n - hat_start]
        if b % 4 == 0:
            degree = degrees[bar % 4]
            root = key + scale[degree % 7] + 48
            third = root + (3 if (degree in (5, 2, 1) and not minor) or (minor and degree in (0, 3, 4)) else 4)
            fifth = root + 7
            length = int(4 * beat * sample_rate)
            seg = torch.arange(min(length, n - start)) / sample_rate
            tone = sum(
                torch.sin(2 * math.pi * 440 * 2 ** ((m - 69) / 12) * seg) for m in (root, third, fifth, root - 12)
            )
            out[start : start + seg.shape[0]] += 0.12 * loud * tone * torch.exp(-seg / 3)
    return (out / (out.abs().max() + 1e-9)) * 0.9
