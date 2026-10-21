# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The data rules: what goes into TriStream's training set, tagged how, mixed how, and what is kept out.

Each rule answers a failure the brief names:

* **accents must not blend into a song** -- every clip carries an explicit accent tag (from metadata, a
  classifier, or ``unknown``); the sampler balances accents per batch instead of letting the biggest corpus
  set the default; the model always sees the tag, so at inference the tag is chosen, never inherited; and the
  content stream is trained against an accent classifier through a gradient-reversal head
  (:func:`accent_adversarial_loss`), the same device TriStream-1 used against singer memorisation;
* **low notes, high notes, whistle, bass** -- clips are bucketed by their F0 range and the buckets are
  sampled to a target distribution (:func:`range_buckets`, :func:`balanced_weights`), so the model is not
  mostly alto-tenor by accident; the gaps (whistle register, true bass) are reported as gaps;
* **echo, muffle, autotune, loud instruments, duets** -- :func:`augment` makes the dry target into the
  distorted *input* conditions the model must understand (the target stays dry), including a second singer
  mixed in with a duet label, so the model learns to listen through effects and to separate roles;
* **breath and emotion** -- :func:`breath_events` finds inhalations (low-energy, high-flatness bursts before
  voiced onsets) so they become tokens, and emotion tags are carried from the corpus or left ``unknown``,
  never guessed from lyrics;
* **dry targets only** -- :func:`is_dry_enough` rejects a target with heavy room or instrumental bleed (the
  TriStream-1 rule), measured with the reverb proxy and the bleed report;
* **singer-disjoint splits** -- :func:`split_singers` hashes singer ids so validation voices are never trained on.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from ..audio.singing import estimate_f0
from .analysis import reverb_time, spectral_rolloff
from .effects import echo, lowpass, muffle, retune, reverb

ACCENTS = (
    "unknown",
    "jamaican",
    "trinidadian",
    "us_general",
    "us_southern",
    "uk_rp",
    "uk_london",
    "nigerian",
    "indian",
    "spanish_l2",
    "french_l2",
    "mandarin_l2",
    "other",
)
EMOTIONS = ("unknown", "neutral", "joy", "sadness", "anger", "tenderness", "longing", "triumph", "fear", "calm")
RANGE_BUCKETS = (
    ("bass", 0, 55),
    ("baritone", 55, 60),
    ("tenor", 60, 67),
    ("alto", 67, 72),
    ("soprano", 72, 84),
    ("whistle", 84, 128),
)


@dataclass
class ClipTags:
    singer: str
    accent: str = "unknown"
    emotion: str = "unknown"
    language: str = "en"
    technique: str = "plain"
    f0_low: float = 0.0
    f0_high: float = 0.0
    duet: bool = False
    dry: bool = True
    breaths: list[float] = field(default_factory=list)  # seconds

    @property
    def range_bucket(self) -> str:
        if self.f0_high <= 0:
            return "unknown"
        midi = 69 + 12 * math.log2(self.f0_high / 440.0)
        for name, lo, hi in RANGE_BUCKETS:
            if lo <= midi < hi:
                return name
        return "whistle"


# ------------------------------------------------------------------------------------------ splits --
def split_singers(singer_ids: list[str], validation_fraction: float = 0.05) -> dict[str, str]:
    """A stable singer-disjoint split: the hash of the id decides, so it never changes between runs."""
    out = {}
    for sid in singer_ids:
        h = int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        out[sid] = "validation" if h < validation_fraction else "train"
    return out


# ------------------------------------------------------------------------------------- range, balance --
def f0_range(wave: torch.Tensor, sample_rate: int) -> tuple[float, float]:
    f0 = estimate_f0(wave[None], sample_rate, frame_rate=25)[0]
    voiced = f0[f0 > 0]
    if voiced.numel() < 5:
        return 0.0, 0.0
    return float(voiced.quantile(0.05)), float(voiced.quantile(0.95))


def range_buckets(tags: list[ClipTags]) -> dict[str, int]:
    counts = {name: 0 for name, _, _ in RANGE_BUCKETS}
    counts["unknown"] = 0
    for t in tags:
        counts[t.range_bucket] += 1
    return counts


def balanced_weights(
    tags: list[ClipTags],
    target: dict[str, float] | None = None,
    by: str = "accent",
    floor: float = 0.25,
    ceiling: float = 4.0,
) -> torch.Tensor:
    """
    Sampling weights that pull the empirical distribution of ``by`` (``accent`` or ``range_bucket``) towards
    ``target`` (uniform over the present classes by default), clipped so no clip is weighted below ``floor``
    or above ``ceiling`` times its natural share.
    """
    keys = [getattr(t, by) if by != "range_bucket" else t.range_bucket for t in tags]
    counts: dict[str, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    n = len(keys)
    present = sorted(counts)
    target = target or {k: 1.0 / len(present) for k in present}
    weights = []
    for k in keys:
        natural = counts[k] / n
        wanted = target.get(k, 0.0)
        w = (wanted / natural) if natural > 0 else 0.0
        weights.append(min(ceiling, max(floor, w)))
    out = torch.tensor(weights, dtype=torch.float)
    return out / out.sum()


def report_gaps(tags: list[ClipTags]) -> list[str]:
    """What the data cannot teach: ranges and accents with fewer than a token amount of clips."""
    gaps = []
    buckets = range_buckets(tags)
    for name, count in buckets.items():
        if name != "unknown" and count < 20:
            gaps.append(f"range '{name}': {count} clips (record more; the model will not sing it well)")
    accents: dict[str, int] = {}
    for t in tags:
        accents[t.accent] = accents.get(t.accent, 0) + 1
    for name, count in accents.items():
        if name != "unknown" and count < 50:
            gaps.append(f"accent '{name}': {count} clips")
    if accents.get("unknown", 0) > 0.5 * len(tags):
        gaps.append("more than half of the clips have no accent tag: the tag cannot do its job; run the classifier")
    return gaps


# -------------------------------------------------------------------------------------- dryness --
def is_dry_enough(
    wave: torch.Tensor, sample_rate: int, max_reverb: float = 0.6, max_rolloff_hz: float | None = None
) -> tuple[bool, dict]:
    """TriStream-1's rule, measured: reject heavy room (decay proxy) and obviously muffled or band-limited audio."""
    decay = reverb_time(wave, sample_rate)
    rolloff = float(spectral_rolloff(wave, sample_rate).median())
    ok = decay <= max_reverb and (max_rolloff_hz is None or rolloff >= max_rolloff_hz)
    return ok, {"reverb": decay, "rolloff_hz": rolloff}


# ------------------------------------------------------------------------------------------ breath --
def breath_events(wave: torch.Tensor, sample_rate: int, frame_rate: float = 25.0) -> list[float]:
    """
    Inhalations: frames that are quiet but not silent, noise-like (high spectral flatness), unvoiced, and
    followed within a second by a voiced onset. Returns their times in seconds.
    """
    hop = int(sample_rate / frame_rate)
    n_fft = 1024
    window = torch.hann_window(n_fft)
    spec = torch.stft(wave, n_fft, hop, window=window, return_complex=True).abs() + 1e-8
    flatness = spec.log().mean(0).exp() / spec.mean(0)
    energy = spec.pow(2).mean(0)
    level = 10 * energy.log10()
    f0 = estimate_f0(wave[None], sample_rate, frame_rate)[0]
    frames = min(flatness.shape[0], f0.shape[0])
    peak = level[:frames].max()
    events = []
    for t in range(frames):
        quiet = peak - 35 < level[t] < peak - 12
        noisy = flatness[t] > 0.3
        unvoiced = f0[t] == 0
        upcoming = f0[t + 1 : t + int(frame_rate)].gt(0).any() if t + 1 < frames else False
        if quiet and noisy and unvoiced and bool(upcoming) and (not events or t / frame_rate - events[-1] > 0.4):
            events.append(t / frame_rate)
    return events


# ----------------------------------------------------------------------------------- augmentation --
def augment(
    dry: torch.Tensor,
    sample_rate: int,
    instrumental: torch.Tensor | None = None,
    second_singer: torch.Tensor | None = None,
    seed: int = 0,
    bpm: float = 120.0,
) -> tuple[torch.Tensor, dict]:
    """
    The distorted *input* for a dry target: a random subset of room, tempo echo, muffle, autotune-style
    retune, instrumental louder than the voice, and a second singer (duet). Returns the input and the record
    of what was applied (the model is told the conditions as tokens, so it learns to listen through them).
    """
    g = torch.Generator().manual_seed(seed)
    draw = lambda: float(torch.rand((), generator=g))  # noqa: E731
    x = dry.clone()
    applied = {}
    if draw() < 0.5:
        decay = 0.3 + 1.7 * draw()
        x = reverb(x, sample_rate, decay_seconds=decay, mix=0.15 + 0.3 * draw())
        applied["reverb"] = decay
    if draw() < 0.25:
        x = echo(x, sample_rate, delay_seconds=60.0 / bpm * (0.5 if draw() < 0.5 else 0.75), feedback=0.3, mix=0.25)
        applied["echo"] = True
    if draw() < 0.2:
        cutoff = 800 + 2200 * draw()
        x = muffle(x, sample_rate, cutoff=cutoff, amount=0.7 + 0.3 * draw())
        applied["muffle_hz"] = cutoff
    if draw() < 0.15:
        applied["retune"] = True  # the F0 stream is retuned in the feature pipeline; the waveform is left as is
    if instrumental is not None and draw() < 0.6:
        ratio = 0.5 + 2.5 * draw()  # instrumental up to 2.5x louder than the voice
        inst = instrumental[: x.shape[-1]]
        if inst.shape[-1] < x.shape[-1]:
            inst = F.pad(inst, (0, x.shape[-1] - inst.shape[-1]))
        x = x + ratio * inst * (x.pow(2).mean().sqrt() / (inst.pow(2).mean().sqrt() + 1e-9))
        applied["instrumental_ratio"] = ratio
    if second_singer is not None and draw() < 0.3:
        other = second_singer[: x.shape[-1]]
        if other.shape[-1] < x.shape[-1]:
            other = F.pad(other, (0, x.shape[-1] - other.shape[-1]))
        x = x + (0.6 + 0.6 * draw()) * other
        applied["duet"] = True
    if draw() < 0.2:
        x = lowpass(x, 4000 + 8000 * draw(), sample_rate)
        applied["bandlimited"] = True
    if draw() < 0.2:  # a low-bitrate rip: band limit plus the holes a perceptual codec punches
        depth = 30 + 20 * draw()
        x = codec_hole_augment(x, sample_rate, depth_db=depth, seed=int(draw() * 1e6))
        applied["codec_holes_db"] = depth
    if draw() < 0.25:  # a noise floor (phone, room tone, tape)
        level_db = -45 + 20 * draw()
        noise = torch.randn(x.shape[-1], generator=g)
        noise = lowpass(noise, 6000.0, sample_rate)
        x = x + noise / (noise.pow(2).mean().sqrt() + 1e-9) * x.pow(2).mean().sqrt() * 10 ** (level_db / 20)
        applied["noise_floor_db"] = level_db
    if instrumental is not None and draw() < 0.2:  # separator residue: anti-phase, low-passed instrumental
        inst = instrumental[: x.shape[-1]]
        if inst.shape[-1] < x.shape[-1]:
            inst = F.pad(inst, (0, x.shape[-1] - inst.shape[-1]))
        x = x - 0.1 * lowpass(inst, 2000.0, sample_rate) * (
            x.pow(2).mean().sqrt() / (inst.pow(2).mean().sqrt() + 1e-9)
        )
        applied["phase_residue"] = True
    peak = x.abs().max()
    return (x / peak * 0.95 if peak > 0.95 else x), applied


def codec_hole_augment(x: torch.Tensor, sample_rate: int, depth_db: float = 36.0, seed: int = 0) -> torch.Tensor:
    """Zero the time-frequency bins above 4 kHz that sit ``depth_db`` under their frame's peak, and cut above
    16 kHz: what an MP3 at 128 kbps leaves of a vocal."""
    n_fft, hop = 2048, 512
    window = torch.hann_window(n_fft)
    spec = torch.stft(x, n_fft, hop, window=window, return_complex=True)
    db = 20 * torch.log10(spec.abs() + 1e-7)
    freqs = torch.fft.rfftfreq(n_fft, 1 / sample_rate)
    peak = db.max(0, keepdim=True).values
    holes = (db < peak - depth_db) & (freqs >= 4000)[:, None]
    spec = spec.masked_fill(holes, 0)
    spec = spec.masked_fill((freqs >= 16000)[:, None], 0)
    return torch.istft(spec, n_fft, hop, window=window, length=x.shape[-1])


def retune_f0_for_training(f0_hz: torch.Tensor, tonic: int, mode: str, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    speed = 0.6 + 0.4 * float(torch.rand((), generator=g))
    return retune(f0_hz, tonic, mode, speed=speed)


# -------------------------------------------------------------------------------------- adversary --
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None


class AccentAdversary(nn.Module):
    """A classifier over accents on the content stream through a gradient-reversal layer."""

    def __init__(self, dim: int, num_accents: int = len(ACCENTS), scale: float = 0.1) -> None:
        super().__init__()
        self.scale = scale
        self.head = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, num_accents))

    def forward(self, content: torch.Tensor, accent_ids: torch.Tensor) -> torch.Tensor:
        pooled = GradientReversal.apply(content.mean(dim=1), self.scale)
        return F.cross_entropy(self.head(pooled), accent_ids)


def accent_adversarial_loss(
    adversary: AccentAdversary, content: torch.Tensor, accent_ids: torch.Tensor
) -> torch.Tensor:
    return adversary(content, accent_ids)
