# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The refiner: air, transients and phase-true harmonics at the sample rate, after the codec.

A codec at 25 frames a second cannot carry everything: the frame is 40 ms, so a consonant burst of 5 ms, the
first 10 ms of a breath, and the exact phase relations between harmonics all get averaged into the frame and
reconstructed by a decoder that has to guess them back. That guess is where metallic phasing, smeared
fricatives and dulled air come from. The refiner stops asking the codec to do those jobs:

* a **harmonic path** that synthesizes the voiced part additively, with phase accumulated sample by sample
  from a 100 Hz F0 curve (DDSP, Engel et al. 2020; NSF, Wang et al. 2019). Harmonics are coherent by
  construction, aliasing is excluded by construction, and a fast run or a wide vibrato is followed at 10 ms
  resolution instead of 40 ms;
* a **noise path**: white noise shaped by band gains predicted at 100 Hz, so a /t/ or an inhale lands with
  10 ms timing and its own spectrum, not the frame's average;
* a **residual path**: a small dilated 1-D network with snake activations (BigVGAN, Lee et al. 2023) over
  [codec output, harmonic, noise] that corrects what the two physical paths cannot express;
* a learned **gate** on the codec's own output, so the refiner can lean on the codec where it is good and
  replace it where it is not.

Training targets are the original waveforms; the losses are multi-resolution STFT, a **transient loss** on
spectral-flux curves at 5 ms (the consonants), a **high-band loss** above 6 kHz (the air), a time-domain L1,
and optionally the codec's multi-period discriminator. Two fail-safes live here too: :func:`render_sketch`
turns the model's source-filter sketch into audio with no learned parameters at all (always coherent, never
metallic, used when a span fails its gates), and :func:`transient_transplant` keeps the source vocal's own
unvoiced high band where the conversion smeared it (unvoiced bursts carry little identity).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .codec import MultiPeriodDiscriminator, mel_filterbank


@dataclass
class RefinerConfig:
    sample_rate: int = 24000
    frame_dim: int = 256  # the codec's latent width
    frame_rate: float = 25.0  # the codec's frame rate
    fine_rate: float = 100.0  # the synthesis paths' control rate
    harmonics: int = 64
    noise_bands: int = 32
    dim: int = 128
    residual_dim: int = 32
    residual_dilations: tuple[int, ...] = (1, 3, 9, 27)
    kernel: int = 7

    @property
    def hop(self) -> int:
        return int(round(self.sample_rate / self.fine_rate))

    @property
    def upsample(self) -> int:
        return int(round(self.fine_rate / self.frame_rate))

    @property
    def noise_fft(self) -> int:
        return 2 * self.hop


class Snake(nn.Module):
    """x + sin^2(a x) / a with a learned per-channel a: periodic, so it continues oscillations instead of clipping."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.log_alpha.exp()
        return x + torch.sin(alpha * x).pow(2) / (alpha + 1e-9)


def upsample_curve(curve: torch.Tensor, length: int) -> torch.Tensor:
    """Linear interpolation of ``(B, T)`` or ``(B, T, C)`` control curves to ``length`` steps along time."""
    squeeze = curve.dim() == 2
    x = curve[..., None] if squeeze else curve
    x = F.interpolate(x.transpose(1, 2), size=length, mode="linear", align_corners=False).transpose(1, 2)
    return x[..., 0] if squeeze else x


def harmonic_oscillator(
    f0_fine: torch.Tensor, amplitudes: torch.Tensor, sample_rate: int, hop: int, chunk: int = 16
) -> torch.Tensor:
    """
    Additive synthesis. ``f0_fine (B, Tf)`` in Hz (0 = unvoiced), ``amplitudes (B, Tf, K)`` >= 0. Phase is the
    cumulative sum of the instantaneous frequency at the sample rate, so harmonics stay locked through any run;
    harmonics above Nyquist are silenced (no aliasing). Returns ``(B, N)`` with ``N = Tf * hop``.
    """
    b, tf = f0_fine.shape
    n = tf * hop
    f0 = upsample_curve(f0_fine, n)
    voiced = upsample_curve((f0_fine > 0).float(), n)
    # phase accumulated in float64 and wrapped: harmonic 40 at minute five is as locked as at second one
    phase = (2 * math.pi * torch.cumsum(f0.double() / sample_rate, dim=-1)).remainder(2 * math.pi).to(f0.dtype)
    amps = upsample_curve(amplitudes, n)  # (B, N, K)
    k = amplitudes.shape[-1]
    out = torch.zeros(b, n, device=f0.device, dtype=f0.dtype)
    for start in range(0, k, chunk):  # chunked over harmonics to bound memory
        idx = torch.arange(start + 1, min(k, start + chunk) + 1, device=f0.device, dtype=f0.dtype)
        alias = (f0[..., None] * idx < sample_rate / 2).to(f0.dtype)
        out = out + (torch.sin(phase[..., None] * idx) * amps[..., start : start + idx.numel()] * alias).sum(-1)
    return out * voiced


def shaped_noise(
    gains: torch.Tensor, sample_rate: int, hop: int, n_fft: int, length: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """White noise filtered by band gains ``(B, Tf, bands)`` at the fine rate; the bands are mel-spaced."""
    b, tf, bands = gains.shape
    device = gains.device
    noise = torch.randn(b, length, device=device, generator=generator)
    window = torch.hann_window(n_fft, device=device)
    spec = torch.stft(noise, n_fft, hop, window=window, return_complex=True)  # (B, bins, frames)
    fb = mel_filterbank(sample_rate, n_fft, bands, device)  # (bands, bins)
    fb = fb / (fb.sum(0, keepdim=True) + 1e-6)
    mag = upsample_curve(gains, spec.shape[-1]) @ fb  # (B, frames, bins)
    shaped = spec * mag.transpose(1, 2).to(spec.dtype)
    return torch.istft(shaped, n_fft, hop, window=window, length=length)


class HarmonicNoiseRefiner(nn.Module):
    def __init__(self, config: RefinerConfig | None = None) -> None:
        super().__init__()
        self.config = c = config or RefinerConfig()
        d = c.dim
        self.cond_in = nn.Linear(c.frame_dim + 3, d)  # frames + (log f0, voicing, log f0 delta) at the fine rate
        self.cond = nn.Sequential(nn.Conv1d(d, d, 5, padding=2), nn.GELU(), nn.Conv1d(d, d, 5, padding=2), nn.GELU())
        self.harmonic_head = nn.Linear(d, c.harmonics + 1)  # amplitude distribution + overall gain
        self.noise_head = nn.Linear(d, c.noise_bands)
        self.gate_head = nn.Linear(d, 1)
        layers: list[nn.Module] = [nn.Conv1d(3, c.residual_dim, c.kernel, padding=c.kernel // 2)]
        for dil in c.residual_dilations:
            layers += [
                Snake(c.residual_dim),
                nn.Conv1d(c.residual_dim, c.residual_dim, c.kernel, dilation=dil, padding=dil * (c.kernel // 2)),
            ]
        layers += [Snake(c.residual_dim), nn.Conv1d(c.residual_dim, 1, c.kernel, padding=c.kernel // 2)]
        self.residual = nn.Sequential(*layers)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def controls(self, frames: torch.Tensor, f0_fine: torch.Tensor) -> dict[str, torch.Tensor]:
        tf = f0_fine.shape[1]
        x = upsample_curve(frames, tf)
        voiced = (f0_fine > 0).float()
        log_f0 = torch.log(f0_fine.clamp(min=1.0)) / 7 * voiced
        delta = F.pad(log_f0[:, 1:] - log_f0[:, :-1], (1, 0)) * 10
        h = self.cond_in(torch.cat([x, log_f0[..., None], voiced[..., None], delta[..., None]], dim=-1))
        h = self.cond(h.transpose(1, 2)).transpose(1, 2)
        harm = self.harmonic_head(h)
        amplitudes = torch.softmax(harm[..., :-1], dim=-1) * F.softplus(harm[..., -1:])
        return {
            "amplitudes": amplitudes,  # (B, Tf, K)
            "noise_gains": F.softplus(self.noise_head(h)) * 0.1,  # (B, Tf, bands)
            "gate": torch.sigmoid(self.gate_head(h) + 2.0),  # (B, Tf, 1); starts at ~0.88: trust the codec first
        }

    def forward(
        self,
        codec_wave: torch.Tensor | None,
        frames: torch.Tensor,
        f0_fine: torch.Tensor,
        generator: torch.Generator | None = None,
        trust: torch.Tensor | float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        ``codec_wave (B, 1, N)`` or None (pure synthesis), ``frames (B, T, frame_dim)`` codec latents at the frame
        rate, ``f0_fine (B, Tf)`` Hz at the fine rate. ``trust`` (0..1, per batch item or scalar) is the intake's
        input quality: the residual path, which can sharpen damage as well as detail, is scaled by it, so a
        rough input gets the physical paths and the codec but not an amplifier. Returns ``wave (B, 1, N)``
        and its parts.
        """
        c = self.config
        b, tf = f0_fine.shape
        n = tf * c.hop
        ctrl = self.controls(frames, f0_fine)
        harmonic = harmonic_oscillator(f0_fine, ctrl["amplitudes"], c.sample_rate, c.hop)
        noise = shaped_noise(ctrl["noise_gains"], c.sample_rate, c.hop, c.noise_fft, n, generator)
        if codec_wave is None:
            base = torch.zeros(b, n, device=f0_fine.device, dtype=harmonic.dtype)
        else:
            base = codec_wave[:, 0]
            base = F.pad(base, (0, max(0, n - base.shape[-1])))[:, :n]
        gate = upsample_curve(ctrl["gate"][..., 0], n)
        stack = torch.stack([base, harmonic, noise], dim=1)
        residual = self.residual(stack)[:, 0]
        if torch.is_tensor(trust):
            residual = residual * trust.to(residual.dtype).reshape(-1, 1)
        else:
            residual = residual * float(trust)
        wave = gate * base + harmonic + noise + residual
        return {
            "wave": wave[:, None],
            "harmonic": harmonic,
            "noise": noise,
            "residual": residual,
            "gate": gate,
            "amplitudes": ctrl["amplitudes"],
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------------------------------- air --
@dataclass
class AirConfig:
    in_rate: int = 24000
    out_rate: int = 48000
    bands: int = 16  # air bands between the input's top and the output's Nyquist
    context_bands: int = 32  # mel bands of the body the gains are predicted from
    dim: int = 64
    hop_ms: float = 10.0


class AirBandExtension(nn.Module):
    """
    The band above the codec's Nyquist, for studio delivery. A 24 kHz pipeline has nothing above 12 kHz; the
    air of a sibilant and the breath of a close microphone live there. This module takes the 24 kHz output
    up to 48 kHz and fills the top band with shaped noise whose per-band gains are predicted from the body's
    spectrum and whose time envelope follows the 4-11 kHz band (sibilance drives air), the spectral band
    replication idea with the gains learned from 48 kHz vocals instead of transmitted. Trained with
    :func:`air_losses` on 48 kHz studio recordings when you have them.
    """

    def __init__(self, config: AirConfig | None = None) -> None:
        super().__init__()
        self.config = c = config or AirConfig()
        self.net = nn.Sequential(
            nn.Linear(c.context_bands + 1, c.dim),
            nn.GELU(),
            nn.Linear(c.dim, c.dim),
            nn.GELU(),
            nn.Linear(c.dim, c.bands),
        )

    @property
    def hop(self) -> int:
        return int(self.config.out_rate * self.config.hop_ms / 1000)

    def forward(self, wave_in: torch.Tensor, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """``wave_in (B, 1, N)`` at ``in_rate`` -> ``wave (B, 1, N * out_rate / in_rate)`` at ``out_rate``."""
        from ..music.effects import lowpass

        c = self.config
        b = wave_in.shape[0]
        n_out = int(wave_in.shape[-1] * c.out_rate / c.in_rate)
        body = F.interpolate(wave_in, size=n_out, mode="linear", align_corners=False)[:, 0]
        body = torch.stack([lowpass(x, c.in_rate / 2 * 0.95, c.out_rate) for x in body])  # no imaging
        n_fft, hop = 4 * self.hop, self.hop
        window = torch.hann_window(n_fft, device=body.device)
        spec = torch.stft(body, n_fft, hop, window=window, return_complex=True)
        mag = spec.abs()
        fb = mel_filterbank(c.out_rate, n_fft, c.context_bands * 2, body.device)
        top = int(c.in_rate / 2 / (c.out_rate / 2) * (n_fft // 2))
        context = torch.log1p((fb[: c.context_bands, :top] @ mag[:, :top]).transpose(1, 2))  # (B, frames, ctx)
        freqs = torch.fft.rfftfreq(n_fft, 1 / c.out_rate).to(body.device)
        sib = (freqs >= 4000) & (freqs < c.in_rate / 2)
        sib_env = mag[:, sib].pow(2).mean(1).sqrt()  # (B, frames)
        sib_norm = sib_env / (sib_env.amax(-1, keepdim=True) + 1e-6)
        gains = F.softplus(self.net(torch.cat([context, sib_norm[..., None]], dim=-1))) * 0.1  # (B, frames, bands)
        gains = gains * sib_norm[..., None]
        # the air bands: linear between the input's Nyquist and the output's
        edges = torch.linspace(c.in_rate / 2, c.out_rate / 2, c.bands + 1, device=body.device)
        air_fb = torch.zeros(c.bands, n_fft // 2 + 1, device=body.device)
        for k in range(c.bands):
            air_fb[k] = ((freqs >= edges[k]) & (freqs < edges[k + 1])).float()
        noise = torch.randn(b, n_out, device=body.device, generator=generator)
        nspec = torch.stft(noise, n_fft, hop, window=window, return_complex=True)
        shaped = nspec * (gains @ air_fb).transpose(1, 2).to(nspec.dtype)
        air = torch.istft(shaped, n_fft, hop, window=window, length=n_out)
        air = air * (body.pow(2).mean(-1, keepdim=True).sqrt() / (air.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6))
        return {"wave": (body + air)[:, None], "body": body, "air": air, "gains": gains}


def air_losses(output: dict, target_48k: torch.Tensor, sample_rate: int, cutoff: float = 11000.0) -> dict:
    pred = output["wave"]
    target = target_48k.reshape(pred.shape)
    terms = {
        "air_band": high_band_loss(pred, target, sample_rate, cutoff=cutoff),
        "air_transient": transient_loss(pred, target, sample_rate),
    }
    terms["total"] = sum(terms.values())
    return terms


# ------------------------------------------------------------------------------------ fail-safes --
def render_sketch(
    f0_fine: torch.Tensor,
    envelope_log: torch.Tensor,
    noise_log: torch.Tensor,
    sample_rate: int,
    fine_rate: float = 100.0,
    harmonics: int = 48,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    The physical fallback: audio from the model's source-filter sketch with no learned parameters. ``envelope_log``
    and ``noise_log`` are ``(B, T, bands)`` log band energies (bands log-spaced 60..8000 Hz as in
    ``tristream.source_filter_sketch``), ``f0_fine (B, Tf)``. Harmonic amplitudes are the envelope read at each
    harmonic's frequency, so the result is coherent by construction. Returns ``(B, N)``.
    """
    hop = int(round(sample_rate / fine_rate))
    b, tf = f0_fine.shape
    bands = envelope_log.shape[-1]
    centres = torch.logspace(math.log10(60.0), math.log10(8000.0), bands, device=f0_fine.device)
    env = upsample_curve(envelope_log, tf).exp()  # (B, Tf, bands)
    k = torch.arange(1, harmonics + 1, device=f0_fine.device).float()
    freqs = f0_fine.clamp(min=1.0)[..., None] * k  # (B, Tf, K)
    # read the envelope at each harmonic: soft nearest band in log frequency
    dist = (torch.log2(freqs[..., None] / centres) * 12).abs()  # (B, Tf, K, bands)
    weights = torch.softmax(-dist, dim=-1)
    amplitudes = (weights * env[:, :, None, :]).sum(-1) / k.sqrt()  # a gentle 1/sqrt(k) roll-off
    amplitudes = amplitudes / (amplitudes.sum(-1, keepdim=True) + 1e-6) * env.mean(-1, keepdim=True).sqrt()
    harmonic = harmonic_oscillator(f0_fine, amplitudes, sample_rate, hop)
    gains = upsample_curve(noise_log, tf).exp().sqrt() * 0.05
    voiced = (f0_fine > 0).float()
    gains = gains * (1 - 0.7 * voiced)[..., None]
    noise = shaped_noise(gains, sample_rate, hop, 2 * hop, tf * hop, generator)
    out = harmonic + noise
    peak = out.abs().amax(-1, keepdim=True).clamp(min=1e-6)
    return out / peak * 0.5


def transient_transplant(
    generated: torch.Tensor,
    source: torch.Tensor,
    f0_hz: torch.Tensor,
    sample_rate: int,
    frame_rate: float = 25.0,
    cutoff: float = 4000.0,
    strength: float = 0.8,
) -> torch.Tensor:
    """
    Keep the source vocal's own unvoiced high band (bursts, fricatives, breath air) where the conversion
    smeared it: ``generated``/``source`` ``(N,)`` aligned, ``f0_hz (T,)`` at ``frame_rate`` (0 = unvoiced).
    Unvoiced sounds carry little speaker identity, so this is nearly free in similarity and large in crispness.
    """
    from ..music.effects import highpass

    n = min(generated.shape[-1], source.shape[-1])
    gen, src = generated[..., :n], source[..., :n]
    unvoiced = (f0_hz <= 0).float()[None, None]
    gate = F.interpolate(unvoiced, size=n, mode="linear", align_corners=False)[0, 0]
    k = max(1, int(0.005 * sample_rate))
    gate = F.avg_pool1d(F.pad(gate[None, None], (k // 2, k - 1 - k // 2), mode="replicate"), k, stride=1)[0, 0]
    gen_hp, src_hp = highpass(gen, cutoff, sample_rate), highpass(src, cutoff, sample_rate)
    level = (gen.pow(2).mean().sqrt() / (src.pow(2).mean().sqrt() + 1e-9)).clamp(0.1, 10.0)
    return gen - strength * gate * gen_hp + strength * gate * src_hp * level


# --------------------------------------------------------------------------------------- losses --
def _spectrum(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=x.device)
    return torch.stft(x.reshape(-1, x.shape[-1]), n_fft, hop, window=window, return_complex=True).abs()


def multi_resolution_stft_loss(
    pred: torch.Tensor, target: torch.Tensor, scales: tuple[int, ...] = (256, 512, 1024, 2048)
) -> torch.Tensor:
    total = pred.new_zeros(())
    for n_fft in scales:
        p, t = _spectrum(pred, n_fft, n_fft // 4), _spectrum(target, n_fft, n_fft // 4)
        log_l1 = (torch.log(p + 1e-5) - torch.log(t + 1e-5)).abs().mean()
        convergence = (t - p).norm() / (t.norm() + 1e-6)
        total = total + log_l1 + convergence
    return total / len(scales)


def flux_curve(x: torch.Tensor, sample_rate: int, hop_ms: float = 5.0, n_fft: int = 512) -> torch.Tensor:
    """Half-wave rectified spectral flux at ``hop_ms``: the curve that peaks on consonants and breath onsets."""
    hop = max(1, int(sample_rate * hop_ms / 1000))
    mag = torch.log(_spectrum(x, n_fft, hop) + 1e-4)
    return F.relu(mag[..., 1:] - mag[..., :-1]).sum(-2)


def transient_loss(pred: torch.Tensor, target: torch.Tensor, sample_rate: int) -> torch.Tensor:
    fp, ft = flux_curve(pred, sample_rate), flux_curve(target, sample_rate)
    return (fp - ft).abs().mean() / (ft.abs().mean() + 1e-6)


def high_band_loss(
    pred: torch.Tensor, target: torch.Tensor, sample_rate: int, cutoff: float = 6000.0, n_fft: int = 1024
) -> torch.Tensor:
    p, t = _spectrum(pred, n_fft, n_fft // 4), _spectrum(target, n_fft, n_fft // 4)
    start = int(cutoff / (sample_rate / 2) * (n_fft // 2))
    return (torch.log(p[:, start:] + 1e-5) - torch.log(t[:, start:] + 1e-5)).abs().mean()


def refiner_losses(
    output: dict,
    target: torch.Tensor,
    sample_rate: int,
    discriminator: MultiPeriodDiscriminator | None = None,
) -> dict[str, torch.Tensor]:
    """The refiner's training terms; ``total`` is what to backpropagate (the discriminator keeps its own loss)."""
    pred = output["wave"]
    target = target.reshape(pred.shape)
    terms = {
        "stft": 1.0 * multi_resolution_stft_loss(pred, target),
        "transient": 2.0 * transient_loss(pred, target, sample_rate),
        "high_band": 1.0 * high_band_loss(pred, target, sample_rate),
        "time": 0.5 * F.l1_loss(pred, target),
    }
    if discriminator is not None:
        fake, real = discriminator(pred), discriminator(target)
        terms["adversarial"] = sum(((1 - f) ** 2).mean() for f, _ in fake) / len(fake)
        terms["feature_matching"] = (
            2.0
            * sum(
                F.l1_loss(ff, rf.detach())
                for (_, ffs), (_, rfs) in zip(fake, real, strict=True)
                for ff, rf in zip(ffs, rfs, strict=True)
            )
            / len(fake)
        )
    terms["total"] = sum(terms.values())
    return terms
