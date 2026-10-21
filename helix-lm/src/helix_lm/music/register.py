# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Register: what a vocal tract can and cannot do when the melody leaves the singer's range.

A baritone sent two octaves up is not a soprano; the tract is the same length and its resonances do not
move with the pitch. What real singers do, measured (Sundberg 1987; Joliveau, Smith & Wolfe 2004, *Nature*
427: sopranos tune the first resonance to the pitch once F0 climbs above it; Titze 2008 on the resonance
and register transitions), is:

* keep the formants nearly where they are: the tract-length scale between a bass and a soprano is about
  15-20 %, never an octave;
* **tune F1 to F0** when F0 rises above the resting F1 (the sung vowel opens), so the harmonic keeps a
  resonance under it; above that the voice is in head voice or falsetto with its own, thinner spectrum;
* beyond about an octave and a half over the comfortable top there is no technique left: the plan has to
  move the melody (octave down, compression), and say so.

This module measures the reference voice's tract scale from its formants (LPC), plans a physically
bounded formant ratio and the F1 tuning for a target melody, warps the model's sketched envelope by that
ratio, and detects the hollow, resonance-less result when a conversion was pushed past the limit anyway.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from ..audio.singing import estimate_f0

NEUTRAL_FORMANTS_HZ = (500.0, 1500.0, 2500.0)  # a 17.5 cm tract on a neutral vowel


def lpc(frame: torch.Tensor, order: int) -> torch.Tensor:
    """Autocorrelation-method LPC coefficients ``a[1..order]`` (``a[0] = 1`` implied) of one frame."""
    x = frame - frame.mean()
    n = x.numel()
    r = torch.stack([(x[: n - k] * x[k:]).sum() for k in range(order + 1)])
    r[0] = r[0] * 1.0001 + 1e-9  # a little white-noise floor for stability
    toeplitz = torch.stack([torch.roll(r[:-1], k)[:order] for k in range(order)])
    toeplitz = torch.tril(toeplitz) + torch.tril(toeplitz, -1).T
    try:
        a = torch.linalg.solve(toeplitz, r[1:])
    except RuntimeError:
        a = torch.zeros(order)
    return a


def formants(wave: torch.Tensor, sample_rate: int, count: int = 3, frame_ms: float = 30.0) -> list[float]:
    """Median formant frequencies (Hz) over voiced frames from LPC roots on the pre-emphasized, 8 kHz-limited
    signal: the usual phonetics recipe, good to a few percent on clean vowels."""
    from .effects import lowpass

    target_sr = 8000
    x = lowpass(wave, 3800.0, sample_rate) if sample_rate > target_sr else wave
    n = int(x.numel() * target_sr / sample_rate)
    x = F.interpolate(x[None, None], size=n, mode="linear", align_corners=False)[0, 0]
    x = torch.cat([x[:1], x[1:] - 0.97 * x[:-1]])  # pre-emphasis
    f0 = estimate_f0(x[None], target_sr, frame_rate=50)[0]
    frame, hop, order = int(frame_ms / 1000 * target_sr), target_sr // 50, 2 + target_sr // 1000
    window = torch.hann_window(frame)
    found: list[list[float]] = []
    for i in range(0, n - frame, hop):
        if i // hop >= f0.numel() or f0[i // hop] <= 0:
            continue
        a = lpc(x[i : i + frame] * window, order)
        roots = np.roots(np.concatenate([[1.0], a.numpy()]))
        roots = roots[np.imag(roots) > 0]
        freqs = np.sort(np.angle(roots) * target_sr / (2 * math.pi))
        bandwidths = -0.5 * target_sr / math.pi * np.log(np.abs(roots[np.argsort(np.angle(roots))]) + 1e-9)
        keep = [float(f) for f, b in zip(freqs, bandwidths, strict=True) if 90 < f < target_sr / 2 - 50 and b < 400]
        if len(keep) >= count:
            found.append(keep[:count])
    if not found:
        return list(NEUTRAL_FORMANTS_HZ[:count])
    return torch.tensor(found).median(0).values.tolist()


def tract_scale(wave: torch.Tensor, sample_rate: int) -> float:
    """Tract length relative to the neutral 17.5 cm tract (> 1 = longer, darker: a bass; < 1 = a child, a
    soprano), from the first three formants. The number real conversions may move formants by."""
    f = formants(wave, sample_rate, 3)
    ratios = [ref / max(measured, 1.0) for ref, measured in zip(NEUTRAL_FORMANTS_HZ, f, strict=True)]
    return float(min(1.6, max(0.6, sum(ratios) / len(ratios))))


def comfortable_range(wave: torch.Tensor, sample_rate: int) -> tuple[float, float]:
    """The 5th and 95th percentile F0 (Hz) of a reference: where that voice actually lives."""
    f0 = estimate_f0(wave[None], sample_rate, frame_rate=25)[0]
    voiced = f0[f0 > 0]
    if voiced.numel() < 5:
        return 110.0, 440.0
    return float(voiced.quantile(0.05)), float(voiced.quantile(0.95))


def register_plan(
    reference_range_hz: tuple[float, float], melody_range_hz: tuple[float, float], scale: float = 1.0
) -> dict:
    """
    How far the melody leaves the reference voice's comfortable range, what the voice will do there, and
    the formant ratio the model may apply (a quarter of the pitch excursion in log terms, capped at the 15-20 %
    a real tract spans), with the plan action when physics runs out.
    """
    lo, hi = reference_range_hz
    m_lo, m_hi = melody_range_hz
    above = max(0.0, 12 * math.log2(max(m_hi, 1.0) / max(hi, 1.0)))
    below = max(0.0, 12 * math.log2(max(lo, 1.0) / max(m_lo, 1.0)))
    ratio = 2 ** ((above - below) / 12 * 0.25)
    ratio = float(min(1.2, max(0.85, ratio)))
    if above <= 2:
        register, action = "chest / mixed", "sing as written"
    elif above <= 7:
        register, action = "head voice", "sing as written; F1 tunes to F0 on the top notes"
    elif above <= 12:
        register, action = "falsetto", "sing as written in falsetto, or move the top line down an octave"
    elif above <= 19:
        register, action = "whistle / strain", "move the melody down an octave or compress it; falsetto will thin out"
    else:
        register, action = "beyond physics", "move the melody down; no tract reaches this from that voice"
    if below > 7:
        register, action = "fry / sub-bass", "move the melody up an octave; the voice will fry under its range"
    return {
        "semitones_above": above,
        "semitones_below": below,
        "formant_ratio": ratio,
        "tract_scale": scale,
        "register": register,
        "action": action,
        "possible": above <= 19 and below <= 12,
    }


def formant_tuning(f0_hz: torch.Tensor, f1_rest: float) -> torch.Tensor:
    """Per-frame F1 target: the resting F1 until F0 climbs past it, then a little above F0 (Joliveau 2004)."""
    voiced = f0_hz > 0
    return torch.where(voiced & (f0_hz > f1_rest), f0_hz * 1.05, torch.full_like(f0_hz, f1_rest))


def warp_envelope(
    envelope_log: torch.Tensor, ratio: torch.Tensor | float, low_hz: float = 60.0, high_hz: float = 8000.0
) -> torch.Tensor:
    """Move a log-frequency band envelope ``(B, T, bands)`` by ``ratio`` in frequency, per batch item ``(B,)`` or per
    frame ``(B, T)``, by linear interpolation along the band axis with border clamping; what the model's sketch
    does when the tract is scaled. ``ratio = 1`` is the identity."""
    b, t, bands = envelope_log.shape
    per_octave = (bands - 1) / math.log2(high_hz / low_hz)
    if not torch.is_tensor(ratio):
        ratio = torch.full((b,), float(ratio), device=envelope_log.device)
    ratio = ratio.float().to(envelope_log.device)
    if ratio.dim() == 1:
        ratio = ratio[:, None].expand(b, t)
    shift = torch.log2(ratio.clamp(min=0.25, max=4.0)) * per_octave  # (B, T) in bands; > 0 moves energy up
    src = torch.arange(bands, device=envelope_log.device).float()[None, None, :] - shift[:, :, None]  # (B, T, bands)
    lo = src.floor().clamp(0, bands - 1).long()
    hi = (lo + 1).clamp(max=bands - 1)
    w = (src - lo.float()).clamp(0, 1)
    return envelope_log.gather(-1, lo) * (1 - w) + envelope_log.gather(-1, hi) * w


def tract_ratio_curve(
    f0_hz: torch.Tensor,
    reference_range_hz: tuple[float, float],
    frame_rate: float = 25.0,
    ramp_semitones: float = 3.0,
    max_ratio: float = 1.2,
    smooth_ms: float = 150.0,
) -> torch.Tensor:
    """
    The per-frame formant ratio for a melody against a voice's comfortable range: nothing inside the range, a
    smooth ramp (over ``ramp_semitones``) once the pitch climbs past the top, a quarter of the excess in log
    terms, capped at ``max_ratio``, held through unvoiced gaps and low-passed over ``smooth_ms``. The wall at a
    register break is a step in this curve; this curve has no steps.
    """
    f0 = f0_hz.float()
    lo, hi = reference_range_hz
    voiced = f0 > 0
    excess = torch.where(voiced, 12 * torch.log2(f0.clamp(min=1.0) / max(hi, 1.0)), torch.zeros_like(f0))
    deficit = torch.where(voiced, 12 * torch.log2(max(lo, 1.0) / f0.clamp(min=1.0)), torch.zeros_like(f0))
    up = torch.sigmoid((excess - ramp_semitones / 2) / (ramp_semitones / 4)) * excess.clamp(min=0)
    down = torch.sigmoid((deficit - ramp_semitones / 2) / (ramp_semitones / 4)) * deficit.clamp(min=0)
    log_ratio = (up - down) / 12 * 0.25
    # hold through unvoiced frames (no jump at an onset), then smooth
    held = log_ratio.clone()
    for i in range(1, held.numel()):
        if not voiced[i]:
            held[i] = held[i - 1]
    k = max(1, int(smooth_ms / 1000 * frame_rate)) | 1
    smooth = F.avg_pool1d(F.pad(held[None, None], (k // 2, k // 2), mode="replicate"), k, stride=1)[0, 0]
    return torch.pow(2.0, smooth).clamp(1 / max_ratio, max_ratio)


def register_break_report(
    wave: torch.Tensor,
    f0_hz: torch.Tensor,
    sample_rate: int,
    boundary_hz: float,
    frame_rate: float = 25.0,
    window_ms: float = 100.0,
    jump_db: float = 12.0,
) -> dict:
    """
    Pops at register breaks: around every frame where the pitch crosses ``boundary_hz`` (the voice's comfortable
    top), the level envelope is checked for a jump over ``jump_db`` within 5 ms and the click detector for
    clicks. Returns the crossings, the pops found, pops per minute and ``ok`` (no pops).
    """
    from .effects import detect_glitches, rms_envelope

    f0 = f0_hz.float()
    above = f0 > boundary_hz
    voiced = f0 > 0
    cross = (above[1:] != above[:-1]) & voiced[1:] & voiced[:-1]
    crossings = cross.nonzero().flatten() + 1
    if crossings.numel() == 0:
        return {"crossings": 0, "pops": 0, "pops_per_minute": 0.0, "ok": True}
    env = rms_envelope(wave, sample_rate, 5.0)
    hop = max(1, int(sample_rate * 0.005))
    db = 20 * torch.log10(env[::hop] + 1e-6)
    clicks = detect_glitches(wave, sample_rate)["clicks"]
    pops = 0
    half = max(3, int(window_ms / 1000 * frame_rate))
    for frame in crossings.tolist():
        a = max(0, int((frame - half) / frame_rate * sample_rate / hop))
        z = min(db.numel(), int((frame + half) / frame_rate * sample_rate / hop))
        jump = float((db[a + 1 : z] - db[a : z - 1]).abs().max()) if z - a > 2 else 0.0
        s0, s1 = int((frame - half) / frame_rate * sample_rate), int((frame + half) / frame_rate * sample_rate)
        clicked = bool(((clicks >= s0) & (clicks < s1)).any()) if clicks.numel() else False
        pops += int(jump > jump_db or clicked)
    minutes = wave.shape[-1] / sample_rate / 60
    return {
        "crossings": int(crossings.numel()),
        "pops": pops,
        "pops_per_minute": pops / max(minutes, 1e-6),
        "ok": pops == 0,
    }


def hollow_score(wave: torch.Tensor, reference: torch.Tensor, sample_rate: int) -> dict:
    """The sound of a tract morphed past its limits: the F1 region (300-1200 Hz) under-represented against
    the reference, and a flat, low-contrast spectrum between 300 Hz and 4 kHz where formants should stand."""
    n_fft = 2048
    window = torch.hann_window(n_fft)

    def profile(x):
        mag = torch.stft(x, n_fft, 512, window=window, return_complex=True).abs().mean(-1)
        return 20 * torch.log10(mag + 1e-7)

    freqs = torch.fft.rfftfreq(n_fft, 1 / sample_rate)
    p, r = profile(wave), profile(reference)
    f1 = (freqs >= 300) & (freqs < 1200)
    mid = (freqs >= 1200) & (freqs < 4000)
    share_p = float(p[f1].mean() - p[mid].mean())
    share_r = float(r[f1].mean() - r[mid].mean())
    band = (freqs >= 300) & (freqs < 4000)
    k = 15
    smooth_p = F.avg_pool1d(F.pad(p[band][None, None], (k // 2, k // 2), mode="replicate"), k, stride=1)[0, 0]
    smooth_r = F.avg_pool1d(F.pad(r[band][None, None], (k // 2, k // 2), mode="replicate"), k, stride=1)[0, 0]
    contrast_p, contrast_r = float(smooth_p.std()), float(smooth_r.std())
    deficit = share_r - share_p
    return {
        "f1_deficit_db": deficit,
        "formant_contrast_db": contrast_p,
        "reference_contrast_db": contrast_r,
        "ok": deficit < 6.0 and contrast_p > 0.5 * contrast_r,
    }
