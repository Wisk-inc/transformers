# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Source separation with bleed control, in PyTorch.

:class:`BandSplitSeparator` is a band-split, dual-path network in the family that leads the public benchmarks
(BS-RoFormer / Mel-Band RoFormer, ByteDance 2023-24: about 11 dB vocal SDR on MUSDB18-HQ; SCNet; the
HT-Demucs line): the STFT is cut into frequency bands of growing width, each band is embedded, a sequence
model runs along time within each band and another across bands at each time step, and a complex mask per
stem is produced for every bin. It is written small enough to train on a CPU in a test and wide enough to
scale on one GPU; a product should start from open weights when the licence allows (``docs/TRISTREAM.md``
lists which do) and fine-tune.

Bleed control, the part the benchmarks under-measure:

* **mask budgeting**: the stems' masks are normalized so a bin's energy is split, never duplicated;
* **harmonic gating** for the vocal stem: bins far from the tracked F0's harmonics are attenuated in voiced
  frames, which removes the hi-hat and string leakage that a mask alone lets through;
* **target-singer extraction** (:meth:`extract`): a speaker embedding conditions the network through FiLM, so
  in a duet each singer is pulled out by their own reference and the two masks are made mutually exclusive;
* **residual check**: :func:`bleed_report` measures how much of the instrumental's energy leaked into the
  vocal in the frames where the vocal is silent, the number to watch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

STEMS = ("vocals", "drums", "bass", "other")


@dataclass
class SeparatorConfig:
    sample_rate: int = 24000
    n_fft: int = 1024
    hop: int = 256
    num_bands: int = 16
    dim: int = 64
    layers: int = 2
    stems: tuple[str, ...] = STEMS
    speaker_dim: int = 0  # > 0 enables target-singer extraction


def band_edges(n_bins: int, num_bands: int) -> list[tuple[int, int]]:
    """Band boundaries that grow roughly geometrically, the first bands one or two bins wide."""
    edges = torch.unique(torch.round(torch.logspace(0, math.log10(n_bins), num_bands + 1)).long().clamp(max=n_bins))
    edges = torch.cat([torch.tensor([0]), edges]) if edges[0] != 0 else edges
    edges = torch.unique(edges)
    pairs = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=False) if b > a]
    if pairs[-1][1] < n_bins:
        pairs.append((pairs[-1][1], n_bins))
    return pairs


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return x * (1 + scale[:, None, None]) + shift[:, None, None]


class BandSplitSeparator(nn.Module):
    def __init__(self, config: SeparatorConfig | None = None) -> None:
        super().__init__()
        self.config = c = config or SeparatorConfig()
        self.n_bins = c.n_fft // 2 + 1
        self.bands = band_edges(self.n_bins, c.num_bands)
        self.embed = nn.ModuleList(nn.Linear(2 * (b - a), c.dim) for a, b in self.bands)
        self.time_rnn = nn.ModuleList(
            nn.GRU(c.dim, c.dim // 2, batch_first=True, bidirectional=True) for _ in range(c.layers)
        )
        self.band_rnn = nn.ModuleList(
            nn.GRU(c.dim, c.dim // 2, batch_first=True, bidirectional=True) for _ in range(c.layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(c.dim) for _ in range(2 * c.layers))
        self.film = FiLM(c.speaker_dim, c.dim) if c.speaker_dim else None
        self.mask = nn.ModuleList(
            nn.Sequential(nn.Linear(c.dim, 2 * c.dim), nn.GELU(), nn.Linear(2 * c.dim, 2 * (b - a) * len(c.stems)))
            for a, b in self.bands
        )
        self.register_buffer("window", torch.hann_window(c.n_fft), persistent=False)

    def stft(self, wave: torch.Tensor) -> torch.Tensor:
        return torch.stft(wave, self.config.n_fft, self.config.hop, window=self.window, return_complex=True)

    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(spec, self.config.n_fft, self.config.hop, window=self.window, length=length)

    def masks(self, spec: torch.Tensor, speaker: torch.Tensor | None = None) -> torch.Tensor:
        """``spec (B, bins, T)`` complex -> complex masks ``(B, stems, bins, T)``, budgeted across stems."""
        b, _, t = spec.shape
        feats = []
        for (lo, hi), proj in zip(self.bands, self.embed, strict=True):
            band = spec[:, lo:hi].transpose(1, 2)  # (B, T, w)
            feats.append(proj(torch.cat([band.real, band.imag], dim=-1)))
        x = torch.stack(feats, dim=2)  # (B, T, bands, dim)
        if self.film is not None and speaker is not None:
            x = self.film(x.reshape(b, t, -1, x.shape[-1]).flatten(1, 2), speaker).reshape(b, t, len(self.bands), -1)
        n_bands = len(self.bands)
        for i in range(self.config.layers):
            y = x.permute(0, 2, 1, 3).reshape(b * n_bands, t, -1)
            y, _ = self.time_rnn[i](y)
            x = x + self.norms[2 * i](y).reshape(b, n_bands, t, -1).permute(0, 2, 1, 3)
            y = x.reshape(b * t, n_bands, -1)
            y, _ = self.band_rnn[i](y)
            x = x + self.norms[2 * i + 1](y).reshape(b, t, n_bands, -1)
        stems = len(self.config.stems)
        out = spec.new_zeros(b, stems, self.n_bins, t)
        for k, ((lo, hi), head) in enumerate(zip(self.bands, self.mask, strict=True)):
            m = head(x[:, :, k]).reshape(b, t, stems, hi - lo, 2)
            complex_mask = torch.complex(m[..., 0], m[..., 1]).permute(0, 2, 3, 1)  # (B, stems, w, T)
            out[:, :, lo:hi] = complex_mask
        # Budget: the magnitudes of the stems' masks at a bin sum to at most 1 (no energy invented twice).
        mag = out.abs()
        total = mag.sum(1, keepdim=True).clamp(min=1.0)
        return out / total

    def forward(self, mix: torch.Tensor, speaker: torch.Tensor | None = None) -> torch.Tensor:
        """``mix (B, samples)`` -> stems ``(B, stems, samples)``."""
        spec = self.stft(mix)
        masks = self.masks(spec, speaker)
        separated = masks * spec[:, None]
        b, s, _, _ = separated.shape
        waves = self.istft(separated.reshape(b * s, *separated.shape[2:]), mix.shape[-1])
        return waves.reshape(b, s, -1)

    def extract(self, mix: torch.Tensor, speakers: torch.Tensor) -> torch.Tensor:
        """
        Target-singer extraction for ``speakers (B, singers, speaker_dim)``: a vocal mask per singer, made
        mutually exclusive by a softmax over singers (plus a residual "neither"), then the budgeted stems.
        Returns ``(B, singers, samples)``.
        """
        if self.film is None:
            raise ValueError("build the separator with speaker_dim > 0 for extraction")
        spec = self.stft(mix)
        vocal_index = self.config.stems.index("vocals")
        logits = []
        for k in range(speakers.shape[1]):
            logits.append(self.masks(spec, speakers[:, k])[:, vocal_index].abs())
        stack = torch.stack(logits + [torch.full_like(logits[0], 0.15)], dim=1)  # (B, singers + 1, bins, T)
        share = stack / stack.sum(1, keepdim=True)
        phase = spec / (spec.abs() + 1e-8)
        waves = []
        for k in range(speakers.shape[1]):
            waves.append(self.istft(share[:, k] * spec.abs() * phase, mix.shape[-1]))
        return torch.stack(waves, dim=1)


# ----------------------------------------------------------------------------------------- losses --
def si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR in dB, per example."""
    target = target - target.mean(-1, keepdim=True)
    estimate = estimate - estimate.mean(-1, keepdim=True)
    scale = (estimate * target).sum(-1, keepdim=True) / (target.pow(2).sum(-1, keepdim=True) + eps)
    projection = scale * target
    noise = estimate - projection
    return 10 * torch.log10(projection.pow(2).sum(-1) / (noise.pow(2).sum(-1) + eps) + eps)


def separation_loss(estimates: torch.Tensor, targets: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Negative SI-SDR averaged over stems plus a multi-resolution spectral L1 (the DAC/Demucs recipe)."""
    from ..audio.codec import multiscale_mel_loss

    sdr = -si_sdr(estimates, targets).mean()
    spectral = multiscale_mel_loss(
        estimates.reshape(-1, estimates.shape[-1]),
        targets.reshape(-1, targets.shape[-1]),
        sample_rate,
        (256, 512, 1024),
    )
    return sdr + 5.0 * spectral


# ------------------------------------------------------------------------------------- bleed tools --
def harmonic_gate(
    vocal_spec: torch.Tensor,
    f0_hz: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    width_semitones: float = 0.6,
    floor: float = 0.15,
) -> torch.Tensor:
    """
    Keep the bins near the harmonics of the tracked F0 in voiced frames and attenuate the rest to ``floor``
    (unvoiced frames pass untouched: consonants and breath live between harmonics). ``f0_hz`` is per frame.
    """
    freqs = torch.fft.rfftfreq(n_fft, 1 / sample_rate).to(vocal_spec.device)
    gate = torch.ones_like(vocal_spec.real)
    for t in range(min(vocal_spec.shape[-1], f0_hz.shape[0])):
        f0 = float(f0_hz[t])
        if f0 <= 0:
            continue
        harmonics = torch.arange(1, int(sample_rate / 2 / f0) + 1, device=freqs.device) * f0
        ratio = freqs[:, None] / harmonics[None]
        distance_semitones = (12 * torch.log2(ratio.clamp(min=1e-6))).abs().min(1).values
        gate[..., t] = torch.where(
            distance_semitones < width_semitones,
            torch.ones_like(distance_semitones),
            torch.full_like(distance_semitones, floor),
        )
    return vocal_spec * gate


def bleed_report(vocal: torch.Tensor, instrumental: torch.Tensor, sample_rate: int, hop_ms: float = 20.0) -> dict:
    """Leakage: the vocal stem's energy in frames where the true vocal is silent, relative to the instrumental's."""
    n = max(1, int(hop_ms / 1000 * sample_rate))
    frames_v = vocal.unfold(-1, n, n).pow(2).mean(-1)
    frames_i = instrumental.unfold(-1, n, n).pow(2).mean(-1)
    silent = frames_v < frames_v.max() * 1e-3
    leak = float(frames_v[silent].mean() / (frames_i[silent].mean() + 1e-12)) if silent.any() else 0.0
    return {
        "leak_ratio": leak,
        "leak_db": 10 * math.log10(leak + 1e-12),
        "silent_fraction": float(silent.float().mean()),
    }


def try_pretrained(name: str = "htdemucs"):
    """
    The open separators, when their packages are installed: ``demucs`` (MIT code; HT-Demucs weights MIT) and
    ``audio-separator`` (which serves BS-RoFormer / Mel-Band RoFormer checkpoints; check each checkpoint's
    licence before shipping). Returns a callable ``separate(wave, sample_rate) -> dict[stem, wave]`` or None.
    """
    try:
        import demucs.apply  # type: ignore
        import demucs.pretrained  # type: ignore
        import torchaudio  # type: ignore  # noqa: F401  (the resampler the separator needs; checked up front)
    except ImportError:
        return None
    try:
        model = demucs.pretrained.get_model(name)
    except Exception:  # weights not downloadable here: the pipeline falls back instead of crashing later
        return None
    model.eval()

    def separate(wave: torch.Tensor, sample_rate: int) -> dict[str, torch.Tensor]:
        import torchaudio  # type: ignore

        x = (
            torchaudio.functional.resample(wave, sample_rate, model.samplerate)
            if sample_rate != model.samplerate
            else wave
        )
        x = x[None].repeat(2, 1) if x.dim() == 1 else x
        with torch.no_grad():
            out = demucs.apply.apply_model(model, x[None], progress=False)[0]
        return {src: out[i].mean(0) for i, src in enumerate(model.sources)}

    return separate
