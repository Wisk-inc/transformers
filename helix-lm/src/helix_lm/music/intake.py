# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Intake: what comes in from the real world, measured, cleaned and labelled before any model hears it.

The models are developed on dry stems and studio references. Users bring room reverb, background noise,
low-bitrate rips with codec holes, clipped phone recordings, and separator output with phase-cancelled
residue. A refiner fed that will sharpen the damage along with the voice. So the intake:

* **measures** (:func:`assess_input`): bandwidth (a 128 kbps rip stops near 16 kHz, a phone call near 4), the
  noise floor, clipping, the room (T60), codec holes (bins far under their own envelope), and a 0..1 quality
  number the model receives as a condition, so "this input is dirty" is something it is told, not something
  it has to infer and then imitate;
* **cleans** (:func:`clean_input`): declipping, spectral-gate denoising from the signal's own quiet frames,
  late-reverb suppression (Lebart et al. 2001, the statistical late-tail estimate), click repair; every step
  is recorded and none runs unless the measurement called for it;
* **checks a separation** (:func:`separation_sanity`): phase-cancelled residue shows as negative correlation
  between the vocal and the instrumental in the voice band, and as energy the mix never had;
* **embeds a reference voice robustly** (:func:`reference_embedding`): the reference is cleaned, cut into
  windows, and windows that are whispered, shouted, nearly silent or buried in music are dropped; the rest
  are embedded one by one and combined by a trimmed mean, so a cough, a chorus or a strained line cannot
  become part of the singer's "permanent" timbre. The report says what was dropped and how far the raw
  pooled embedding sat from the clean one (contamination);
* **trains the encoder to not care** (:func:`condition_invariance_loss`): the embedding of a clip and of its
  augmented (reverberant, noisy, music-mixed, band-limited) twin must agree.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..audio.singing import estimate_f0
from .analysis import spectral_centroid, spectral_rolloff
from .effects import decay_time, detect_glitches, repair_glitches


def _stft(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=x.device)
    return torch.stft(x, n_fft, hop, window=window, return_complex=True)


def _istft(spec: torch.Tensor, n_fft: int, hop: int, length: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=spec.device)
    return torch.istft(spec, n_fft, hop, window=window, length=length)


# ------------------------------------------------------------------------------------------ measure --
def codec_holes(wave: torch.Tensor, sample_rate: int, low_hz: float = 4000.0, depth_db: float = 55.0) -> float:
    """The share of time-frequency bins above ``low_hz`` that sit more than ``depth_db`` under the smoothed
    envelope of their own frame: the holes a perceptual codec punches where it decided nobody would hear."""
    n_fft = 2048
    mag = _stft(wave, n_fft, 512).abs()
    db = 20 * torch.log10(mag + 1e-7)
    start = int(low_hz / (sample_rate / 2) * (n_fft // 2))
    stop = int(min(16000.0, sample_rate / 2 - 500) / (sample_rate / 2) * (n_fft // 2))
    if stop - start < 16:
        return 0.0
    band = db[start:stop]
    k = 31
    envelope = F.max_pool1d(F.pad(band.T[None], (k // 2, k // 2), mode="replicate"), k, stride=1)[0].T
    loud = band.max(0).values > band.max() - 40  # only frames with signal
    holes = (band < envelope - depth_db) & loud[None, :]
    return float(holes.float().sum() / max(1, int(loud.sum()) * (stop - start)))


def bandwidth_edge(wave: torch.Tensor, sample_rate: int, depth_db: float = 60.0) -> float:
    """The highest frequency whose long-term level is within ``depth_db`` of the spectrum's peak: where the
    recording stops (a phone line near 4 kHz, a 128 kbps rip near 16 kHz, a studio file at Nyquist)."""
    n_fft = 2048
    ltas = 20 * torch.log10(_stft(wave, n_fft, 512).abs().pow(2).mean(-1).sqrt() + 1e-7)
    freqs = torch.fft.rfftfreq(n_fft, 1 / sample_rate)
    above = (ltas > ltas.max() - depth_db).nonzero().flatten()
    return float(freqs[above[-1]]) if above.numel() else 0.0


def noise_floor_db(wave: torch.Tensor, sample_rate: int) -> float:
    """The 5th percentile of each bin's level over time (what is there when the voice is not), averaged, in dB
    under the loudest frame: a clean recording sits 60 dB down, room tone and hiss 30-45 dB."""
    n_fft = 1024
    power = _stft(wave, n_fft, 256).abs().pow(2)
    floor = power.quantile(0.05, dim=-1).mean()
    loudest = power.mean(0).max()
    return float(10 * torch.log10((floor + 1e-12) / (loudest + 1e-12)))


def assess_input(wave: torch.Tensor, sample_rate: int) -> dict:
    """Bandwidth, noise floor, clipping, room, holes and a quality score in [0, 1] (1 = studio-clean)."""
    from .effects import bandpass

    wave = wave.float()
    peak = float(wave.abs().max())
    clipped = float((wave.abs() >= 0.985 * max(peak, 1e-6)).float().mean()) if peak > 0.9 else 0.0
    edge = bandwidth_edge(wave, sample_rate)
    rolloff = float(spectral_rolloff(wave, sample_rate).median())
    noise_floor = noise_floor_db(wave, sample_rate)
    # the room is read above 1 kHz, where a voice's own releases are fast and a tail is not
    high = min(6000.0, sample_rate / 2 - 200)
    room = decay_time(bandpass(wave, 1000.0, high, sample_rate), sample_rate)
    holes = codec_holes(wave, sample_rate)
    centroid = float(spectral_centroid(wave, sample_rate).median())
    full_credit = min(12000.0, sample_rate / 2 * 0.9)
    bandwidth_ok = float(max(0.0, min(1.0, (edge - 3500.0) / max(1.0, full_credit - 3500.0))))
    quality = 1.0
    quality *= 0.5 + 0.5 * bandwidth_ok
    quality *= 1.0 - min(1.0, clipped * 50)
    quality *= 1.0 if noise_floor < -45 else max(0.3, 1.0 - (noise_floor + 45) / 30)
    quality *= 1.0 if room < 0.6 else max(0.4, 1.0 - (room - 0.6) / 2.0)
    quality *= 1.0 - min(0.6, holes * 4)
    kind = "studio" if quality > 0.85 else "consumer" if quality > 0.6 else "rough"
    return {
        "bandwidth_hz": edge,
        "rolloff_hz": rolloff,
        "noise_floor_db": noise_floor,
        "clipped_fraction": clipped,
        "room_t60": room,
        "codec_holes": holes,
        "centroid_hz": centroid,
        "quality": float(max(0.0, min(1.0, quality))),
        "kind": kind,
    }


def separation_sanity(vocal: torch.Tensor, instrumental: torch.Tensor, mix: torch.Tensor, sample_rate: int) -> dict:
    """Phase-cancelled residue and invented energy after a stem split: negative vocal/instrumental correlation in
    the voice band, and stems that together carry more than the mix did."""
    from .effects import bandpass

    n = min(vocal.numel(), instrumental.numel(), mix.numel())
    v, i, m = vocal[:n], instrumental[:n], mix[:n]
    vb, ib = bandpass(v, 200.0, 2000.0, sample_rate), bandpass(i, 200.0, 2000.0, sample_rate)
    corr = float((vb * ib).sum() / (vb.norm() * ib.norm() + 1e-9))
    leak = float((vb * ib).sum() / (ib.pow(2).sum() + 1e-9))  # the instrumental's gain inside the vocal stem
    invented = float(10 * torch.log10((v.pow(2).sum() + i.pow(2).sum() + 1e-9) / (m.pow(2).sum() + 1e-9)))
    residual = float(10 * torch.log10(((v + i - m).pow(2).mean() + 1e-9) / (m.pow(2).mean() + 1e-9)))
    return {
        "voice_band_correlation": corr,
        "leak_gain": leak,
        "invented_db": invented,
        "residual_db": residual,
        "phase_cancelled": leak < -0.2 or corr < -0.3,
        "ok": leak >= -0.2 and corr >= -0.3 and invented < 3.0,
    }


# -------------------------------------------------------------------------------------------- clean --
def declip(wave: torch.Tensor, threshold: float = 0.985) -> tuple[torch.Tensor, int]:
    """Runs of samples stuck at the rails are bridged by a smooth arc that overshoots in proportion to their
    length (the waveform that was cut off kept going)."""
    x = wave.clone()
    peak = float(x.abs().max())
    if peak < 0.9:
        return x, 0
    stuck = x.abs() >= threshold * peak
    runs, i, n = [], 0, x.numel()
    while i < n:
        if stuck[i]:
            j = i
            while j < n and stuck[j]:
                j += 1
            if j - i >= 2:
                runs.append((i, j))
            i = j
        else:
            i += 1
    for a, z in runs:
        left, right = x[max(0, a - 1)], x[min(n - 1, z)]
        length = z - a
        arc = torch.sin(torch.linspace(0, math.pi, length)) * torch.sign(x[a]) * peak * (0.08 * math.log1p(length))
        x[a:z] = torch.linspace(float(left), float(right), length) + arc
    return x, len(runs)


def denoise(wave: torch.Tensor, sample_rate: int, strength: float = 1.5, floor: float = 0.1) -> torch.Tensor:
    """Spectral gating from the signal's own quietest frames (the 10th percentile per bin is the noise)."""
    n_fft, hop = 1024, 256
    spec = _stft(wave, n_fft, hop)
    mag = spec.abs()
    noise = mag.quantile(0.1, dim=-1, keepdim=True)
    gain = ((mag - strength * noise) / (mag + 1e-8)).clamp(min=floor, max=1.0)
    gain = F.avg_pool2d(F.pad(gain[None, None], (1, 1, 1, 1), mode="replicate"), 3, stride=1)[0, 0]
    return _istft(spec * gain, n_fft, hop, wave.numel())


def suppress_late_reverb(
    wave: torch.Tensor, sample_rate: int, t60: float, start_ms: float = 50.0, floor: float = 0.15
) -> torch.Tensor:
    """Lebart's late-reverb estimate: the reverberant power ``start_ms`` ago, decayed at the room's rate, is
    subtracted as noise would be. Early reflections and the direct sound are left alone."""
    n_fft, hop = 1024, 256
    spec = _stft(wave, n_fft, hop)
    power = spec.abs().pow(2)
    frames_delay = max(1, int(start_ms / 1000 * sample_rate / hop))
    decay_per_frame = math.exp(-6.9 * hop / sample_rate / max(t60, 0.1))
    late = torch.zeros_like(power)
    acc = torch.zeros_like(power[:, 0])
    for t in range(power.shape[-1]):
        acc = acc * decay_per_frame + (power[:, t - frames_delay] if t >= frames_delay else 0) * (1 - decay_per_frame)
        late[:, t] = acc
    gain = (1 - late / (power + 1e-8)).clamp(min=floor, max=1.0).sqrt()
    return _istft(spec * gain, n_fft, hop, wave.numel())


def clean_input(wave: torch.Tensor, sample_rate: int, report: dict | None = None) -> tuple[torch.Tensor, list[str]]:
    """The steps the measurement calls for, in order, each recorded; nothing runs on a clean input."""
    report = report or assess_input(wave, sample_rate)
    applied = []
    x = wave.float()
    if report["clipped_fraction"] > 1e-4:
        x, runs = declip(x)
        if runs:
            applied.append(f"declipped {runs} runs")
    glitches = detect_glitches(x, sample_rate)
    if glitches["clicks"].numel() or glitches["dropouts"].numel():
        x = repair_glitches(x, sample_rate, glitches)
        applied.append(
            f"repaired {int(glitches['clicks'].numel())} clicks, {int(glitches['dropouts'].numel())} dropouts"
        )
    if report["noise_floor_db"] > -45:
        x = denoise(x, sample_rate)
        applied.append(f"denoised (floor {report['noise_floor_db']:.0f} dB)")
    if report["room_t60"] > 0.6:
        x = suppress_late_reverb(x, sample_rate, report["room_t60"])
        applied.append(f"late reverb suppressed (T60 {report['room_t60']:.1f} s)")
    peak = float(x.abs().max())
    if peak > 0.99:
        x = x / peak * 0.99
    return x, applied


# ---------------------------------------------------------------------------------------- reference --
def reference_windows(
    wave: torch.Tensor, sample_rate: int, window_s: float = 1.0, hop_s: float = 0.5
) -> tuple[list[tuple[int, int]], dict]:
    """Windows of a reference clip worth embedding, and why the others were not: whispered or silent (little
    voicing), shouted (loud and bright well above the clip's own norm), too quiet, or buried (noise-like)."""
    n = wave.numel()
    win, hop = int(window_s * sample_rate), int(hop_s * sample_rate)
    if n < win:
        return [(0, n)], {"kept": 1, "dropped": {}}
    f0 = estimate_f0(wave[None], sample_rate, frame_rate=50)[0]
    frames_per_window = int(window_s * 50)
    candidates = []
    for start in range(0, n - win + 1, hop):
        seg = wave[start : start + win]
        f = int(start / sample_rate * 50)
        voiced = float((f0[f : f + frames_per_window] > 0).float().mean()) if f0.numel() > f else 0.0
        level = float(20 * torch.log10(seg.pow(2).mean().sqrt() + 1e-6))
        centroid = float(spectral_centroid(seg, sample_rate).median())
        candidates.append((start, start + win, voiced, level, centroid))
    levels = torch.tensor([c[3] for c in candidates])
    centroids = torch.tensor([c[4] for c in candidates])
    med_level, med_centroid = float(levels.median()), float(centroids.median())
    kept, dropped = [], {"whisper_or_silence": 0, "shout": 0, "quiet": 0}
    for start, stop, voiced, level, centroid in candidates:
        if voiced < 0.6:
            dropped["whisper_or_silence"] += 1
        elif level > med_level + 8 or (level > med_level + 5 and centroid > med_centroid * 1.15):
            dropped["shout"] += 1
        elif level < med_level - 12:
            dropped["quiet"] += 1
        else:
            kept.append((start, stop))
    if not kept:  # keep the most voiced windows rather than nothing
        kept = [(c[0], c[1]) for c in sorted(candidates, key=lambda c: -c[2])[:3]]
    return kept, {"kept": len(kept), "dropped": dropped}


def trimmed_mean_embedding(embeddings: torch.Tensor, trim: float = 0.2) -> torch.Tensor:
    """Normalize, drop the ``trim`` share farthest from the medoid, average, normalize: one outlier window
    cannot steer the voice."""
    e = F.normalize(embeddings, dim=-1)
    if e.shape[0] <= 2:
        return F.normalize(e.mean(0, keepdim=True), dim=-1)
    sims = e @ e.T
    medoid = e[int(sims.sum(-1).argmax())]
    distance = 1 - e @ medoid
    keep = int(math.ceil(e.shape[0] * (1 - trim)))
    chosen = e[distance.argsort()[:keep]]
    return F.normalize(chosen.mean(0, keepdim=True), dim=-1)


@torch.no_grad()
def reference_embedding(
    reference: torch.Tensor, sample_rate: int, codec, speaker_encoder, clean: bool = True
) -> tuple[torch.Tensor, dict]:
    """The robust voice embedding ``(1, D)`` of a reference clip and a report (quality, cleaning, windows kept
    and dropped, contamination = how far the naive pooled embedding sat from the robust one)."""
    assessment = assess_input(reference, sample_rate)
    cleaned, applied = clean_input(reference, sample_rate, assessment) if clean else (reference, [])
    windows, selection = reference_windows(cleaned, sample_rate)
    embeddings = torch.cat([speaker_encoder(codec.listen(cleaned[a:z][None])) for a, z in windows])
    robust = trimmed_mean_embedding(embeddings)
    naive = F.normalize(speaker_encoder(codec.listen(reference[None])), dim=-1)
    contamination = float(1 - (naive * robust).sum(-1))
    report = {
        **assessment,
        "cleaning": applied,
        **selection,
        "contamination": contamination,
        "ok": contamination < 0.15 and selection["kept"] >= 1,
    }
    return robust, report


def condition_invariance_loss(speaker_encoder, codec, clean: torch.Tensor, dirty: torch.Tensor) -> torch.Tensor:
    """``clean``/``dirty`` ``(B, N)`` are the same voices, the second through :func:`regularize.augment`: the
    embeddings must agree (cosine), and different voices in the batch must not (a margin)."""
    with torch.no_grad():
        frames_clean, frames_dirty = codec.listen(clean), codec.listen(dirty)
    a = F.normalize(speaker_encoder(frames_clean), dim=-1)
    b = F.normalize(speaker_encoder(frames_dirty), dim=-1)
    agree = (1 - (a * b).sum(-1)).mean()
    if a.shape[0] > 1:
        off = (a @ b.T).fill_diagonal_(0)
        apart = F.relu(off - 0.5).sum() / (a.shape[0] * (a.shape[0] - 1))
    else:
        apart = a.new_zeros(())
    return agree + apart
