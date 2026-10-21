# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The ear: detectors for the ways a synthetic singing voice gives itself away.

Each function measures one thing a listener hears and returns a number with a pass/fail against a threshold,
so the generation loop (:mod:`failsafe`) can act on it instead of hoping:

* :func:`frame_rate_buzz` -- a codec's frame rate leaking into the amplitude envelope (a 25 Hz flutter);
* :func:`harmonic_coherence` -- harmonics-to-noise along the F0 track and how much it flickers on held notes
  (metallic, phasey voices flicker; real ones do not);
* :func:`hf_character` -- whether the band above 6 kHz is noise-like as in a real voice or tonal and peaky
  (metallic) compared with a reference;
* :func:`consonant_sharpness` -- how much of the reference's onset energy survives at its onsets (smearing);
* :func:`robotic_score` -- the F0 tells: no micro-variation, machine-perfect vibrato, instantaneous jumps;
* :func:`sync_lag` -- the lag between the control curve and what the audio does (stream desynchronization);
* :func:`artifact_report` -- all of them, plus the click / dropout / clipping report, with a list of failures.

Thresholds are starting points from the literature on jitter, vibrato and modulation perception; they are
meant to be re-tuned on listening tests and are exposed as arguments.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .effects import detect_glitches


def _stft(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=x.device)
    return torch.stft(x.reshape(-1, x.shape[-1]), n_fft, hop, window=window, return_complex=True).abs()[0]


def amplitude_envelope(wave: torch.Tensor, sample_rate: int, hop_ms: float = 1.0) -> torch.Tensor:
    hop = max(1, int(sample_rate * hop_ms / 1000))
    n = wave.shape[-1] // hop * hop
    return wave[..., :n].reshape(-1, hop).pow(2).mean(-1).sqrt()


def frame_rate_buzz(wave: torch.Tensor, sample_rate: int, frame_rate: float = 25.0, limit_db: float = 8.0) -> dict:
    """Peak of the envelope's modulation spectrum at the frame rate (and its double) above the floor, in dB."""
    env = amplitude_envelope(wave, sample_rate, 1.0)  # 1 kHz
    if env.numel() < 400:
        return {"buzz_db": 0.0, "ok": True}
    k = 101
    trend = F.avg_pool1d(F.pad(env[None, None], (k // 2, k // 2), mode="replicate"), k, stride=1)[0, 0]
    detrended = (env - trend) * torch.hann_window(env.numel())
    spectrum = torch.fft.rfft(detrended).abs()
    freqs = torch.fft.rfftfreq(env.numel(), d=1e-3)

    def band(lo, hi):
        sel = (freqs >= lo) & (freqs <= hi)
        return spectrum[sel].max() if sel.any() else spectrum.new_zeros(())

    peak = torch.maximum(band(frame_rate - 1, frame_rate + 1), band(2 * frame_rate - 1, 2 * frame_rate + 1))
    floor_sel = (freqs >= 3) & (freqs <= 60) & ((freqs - frame_rate).abs() > 2) & ((freqs - 2 * frame_rate).abs() > 2)
    floor = spectrum[floor_sel].median() if floor_sel.any() else spectrum.new_tensor(1.0)
    buzz = float(20 * torch.log10((peak + 1e-9) / (floor + 1e-9)))
    return {"buzz_db": buzz, "ok": buzz < limit_db}


def harmonic_coherence(
    wave: torch.Tensor,
    f0_hz: torch.Tensor,
    sample_rate: int,
    frame_rate: float = 25.0,
    min_hnr_db: float = 8.0,
    max_flicker_db: float = 3.0,
) -> dict:
    """
    Harmonics-to-noise ratio per voiced frame along ``f0_hz`` (energy at the first 20 harmonics against the
    energy half-way between them), its mean, and its frame-to-frame flicker on held notes (F0 within a quarter
    tone of its neighbours). Phasey or metallic output has a low or flickering HNR where the pitch is steady.
    """
    n_fft = 2048
    hop = int(sample_rate / frame_rate)
    mag = _stft(wave, n_fft, hop)  # (bins, frames)
    frames = min(mag.shape[-1], f0_hz.shape[-1])
    mag, f0 = mag[:, :frames], f0_hz[:frames]
    voiced = f0 > 0
    if voiced.sum() < 4:
        return {"hnr_db": 0.0, "flicker_db": 0.0, "ok": True, "voiced_frames": int(voiced.sum())}
    bin_hz = sample_rate / n_fft
    k = torch.arange(1, 21, device=wave.device).float()
    harm_bins = (f0[:, None] * k / bin_hz).round().long()  # (frames, 20)
    between = ((f0[:, None] * (k + 0.5)) / bin_hz).round().long()
    valid = (harm_bins < mag.shape[0] - 1) & (between < mag.shape[0] - 1) & (f0[:, None] * k < 8000)
    harm_bins, between = harm_bins.clamp(max=mag.shape[0] - 1), between.clamp(max=mag.shape[0] - 1)
    idx = torch.arange(frames, device=wave.device)[:, None].expand_as(harm_bins)
    h = (mag[harm_bins, idx] ** 2 * valid).sum(-1)
    b = (mag[between, idx] ** 2 * valid).sum(-1)
    hnr = 10 * torch.log10((h + 1e-9) / (b + 1e-9))
    hnr_v = hnr[voiced]
    log_f0 = torch.log2(f0.clamp(min=1.0)) * 12
    steady = voiced.clone()
    steady[1:] &= (log_f0[1:] - log_f0[:-1]).abs() < 0.5
    steady[:-1] &= (log_f0[1:] - log_f0[:-1]).abs() < 0.5
    if steady.sum() >= 4:
        s = hnr[steady]
        flicker = float((s[1:] - s[:-1]).abs().mean())
    else:
        flicker = 0.0
    mean = float(hnr_v.mean())
    # a very clean harmonic signal (HNR far above the floor) is not phasey whatever its flicker in dB
    return {
        "hnr_db": mean,
        "flicker_db": flicker,
        "ok": mean >= min_hnr_db and (flicker <= max_flicker_db or mean >= 25.0),
        "voiced_frames": int(voiced.sum()),
    }


def spectral_flatness(mag: torch.Tensor) -> torch.Tensor:
    power = mag.pow(2) + 1e-10
    return torch.exp(power.log().mean(0)) / power.mean(0)


def hf_character(
    pred: torch.Tensor, ref: torch.Tensor, sample_rate: int, cutoff: float = 6000.0, limit_db: float = 6.0
) -> dict:
    """Flatness of the band above ``cutoff`` in the output against the reference: tonal where it should be noisy
    reads as metallic; noisy where it should be tonal reads as hiss. Both are reported as a dB difference."""
    n_fft = 1024
    start = int(cutoff / (sample_rate / 2) * (n_fft // 2))
    fp = spectral_flatness(_stft(pred, n_fft, n_fft // 4)[start:]).mean()
    fr = spectral_flatness(_stft(ref, n_fft, n_fft // 4)[start:]).mean()
    diff = float(10 * torch.log10((fr + 1e-6) / (fp + 1e-6)))  # > 0: output more tonal (metallic) than the reference
    return {"flatness_pred": float(fp), "flatness_ref": float(fr), "metallic_db": diff, "ok": abs(diff) < limit_db}


def onsets(flux: torch.Tensor, min_gap: int = 10) -> torch.Tensor:
    """Indices of flux peaks above mean + 2 std, at least ``min_gap`` frames apart."""
    threshold = flux.mean() + 2 * flux.std()
    peaks = (flux[1:-1] > flux[:-2]) & (flux[1:-1] >= flux[2:]) & (flux[1:-1] > threshold)
    idx = peaks.nonzero().flatten() + 1
    kept = []
    for i in idx.tolist():
        if not kept or i - kept[-1] >= min_gap:
            kept.append(i)
    return torch.tensor(kept, dtype=torch.long, device=flux.device)


def consonant_sharpness(pred: torch.Tensor, ref: torch.Tensor, sample_rate: int, min_ratio: float = 0.6) -> dict:
    """How much of the reference's onset energy the output keeps at the reference's onsets (5 ms flux)."""
    from ..audio.refiner import flux_curve

    fp, fr = flux_curve(pred[None], sample_rate)[0], flux_curve(ref[None], sample_rate)[0]
    frames = min(fp.numel(), fr.numel())
    fp, fr = fp[:frames], fr[:frames]
    where = onsets(fr)
    if where.numel() == 0:
        return {"ratio": 1.0, "onsets": 0, "ok": True}

    def around(curve, idx):
        lo, hi = (idx - 1).clamp(min=0), (idx + 2).clamp(max=curve.numel())
        return torch.stack([curve[a:b].max() for a, b in zip(lo.tolist(), hi.tolist(), strict=True)])

    ratio = float((around(fp, where) / (around(fr, where) + 1e-6)).clamp(max=3.0).mean())
    return {"ratio": ratio, "onsets": int(where.numel()), "ok": ratio >= min_ratio}


def _lowpass_curve(x: torch.Tensor, window: int) -> torch.Tensor:
    window = max(1, window | 1)
    return F.avg_pool1d(F.pad(x[None, None], (window // 2, window // 2), mode="replicate"), window, stride=1)[0, 0]


def held_segments(
    cents: torch.Tensor, voiced: torch.Tensor, frame_rate: float, min_seconds: float = 0.3, depart_cents: float = 70.0
):
    """
    Held notes: voiced runs on one pitch. A run starts at a voicing onset or where the 50 ms-smoothed pitch
    departs more than ``depart_cents`` from the note's reference (the median of its first 150 ms), which
    catches steps and glides alike. Returns ``(start, end)`` pairs at least ``min_seconds`` long.
    """
    n = cents.numel()
    sm = _lowpass_curve(cents, max(1, int(0.05 * frame_rate)))
    settle = max(2, int(0.15 * frame_rate))
    out, start, ref = [], None, None
    for i in range(n):
        if not voiced[i]:
            if start is not None and i - start >= int(min_seconds * frame_rate):
                out.append((start, i))
            start, ref = None, None
            continue
        if start is None:
            start, ref = i, None
            continue
        if ref is None:
            if i - start >= settle:
                ref = float(sm[start:i].median())
            continue
        if abs(float(sm[i]) - ref) > depart_cents:
            if i - start >= int(min_seconds * frame_rate):
                out.append((start, i))
            start, ref = i, None
    if start is not None and n - start >= int(min_seconds * frame_rate):
        out.append((start, n))
    return out


def robotic_score(f0_hz: torch.Tensor, frame_rate: float = 100.0) -> dict:
    """
    Three tells of a machine pitch curve, each in [0, 1] (1 = robotic), and their mean:

    * ``fluctuation``: the fine fluctuation left on held notes once the trend and the 4-8 Hz vibrato band are
      removed (spectral notch). Sung voices keep a wander and jitter of 5-20 cents (Saitou 2005); a flat
      residual under 2 cents is a function, not a larynx;
    * ``vibrato_regularity``: how much of the 3-10 Hz power sits in the three bins around the vibrato peak.
      A sine from a function puts almost all of it there; a human vibrato drifts in rate and extent;
    * ``jumps``: the share of note transitions taken as one step over 100 cents instead of a glide. Singers
      glide over 40-150 ms; only measured at >= 50 Hz, where a glide spans several frames.
    """
    f0 = f0_hz.float()
    voiced = f0 > 0
    if voiced.sum() < int(frame_rate):
        return {"fluctuation_cents": 0.0, "vibrato_regularity": 0.0, "jump_fraction": 0.0, "score": 0.0, "ok": True}
    cents = torch.where(voiced, 1200 * torch.log2(f0.clamp(min=1.0) / 440.0), torch.zeros_like(f0))
    segments = held_segments(cents, voiced, frame_rate)
    residual_power, vibrato_peak, vibrato_band = [], [], []
    trim = int(0.12 * frame_rate)
    for a, z in segments:
        a = a + trim if z - a - trim >= int(0.2 * frame_rate) else a
        x = cents[a:z]
        t = torch.arange(z - a, dtype=torch.float)
        t = t - t.mean()
        slope = (t * (x - x.mean())).sum() / (t.pow(2).sum() + 1e-9)
        x = x - x.mean() - slope * t
        window = torch.hann_window(z - a)
        spec = torch.fft.rfft(x * window)
        freqs = torch.fft.rfftfreq(z - a, d=1.0 / frame_rate)
        power = spec.abs().pow(2)
        band = (freqs >= 3) & (freqs <= 10)
        if band.any():
            peak_bin = int((power * band).argmax())
            around = power[max(0, peak_bin - 1) : peak_bin + 2].sum()
            vibrato_peak.append(float(around))
            vibrato_band.append(float(power[band].sum()))
            notch = spec.clone()
            notch[max(0, peak_bin - 2) : peak_bin + 3] = 0
        else:
            notch = spec.clone()
        notch[freqs < 1.0] = 0
        resid = torch.fft.irfft(notch, n=z - a)
        residual_power.append(float(resid.pow(2).mean() / (window.pow(2).mean() + 1e-9)))
    fluctuation = math.sqrt(sum(residual_power) / len(residual_power)) if residual_power else 0.0
    if vibrato_band and sum(vibrato_band) > 0:
        peakiness = min(1.0, sum(vibrato_peak) / sum(vibrato_band))
        vibrato_rms = math.sqrt(sum(vibrato_band) / max(1, len(vibrato_band)) / max(1, int(frame_rate)))
    else:
        peakiness, vibrato_rms = 0.5, 0.0
    # vibrato regularity: how constant the extent is along the held notes. A sung vibrato is absent at the
    # onset, ramps in, grows towards the end and breathes; a function's is the same in every 150 ms chunk.
    chunk = max(2, int(0.15 * frame_rate))
    extents = []
    for a, z in segments:
        x = cents[a:z]
        x = x - _lowpass_curve(x, max(1, int(0.3 * frame_rate)))
        for start in range(0, z - a - chunk + 1, chunk):
            extents.append(float(x[start : start + chunk].pow(2).mean().sqrt()))
    if len(extents) >= 3 and vibrato_rms >= 3.0:
        e = torch.tensor(extents)
        extent_cv = float(e.std() / (e.mean() + 1e-6))
        regularity = max(0.0, 1.0 - extent_cv)
    else:
        regularity = 0.5  # too little vibrato to judge
    # note transitions: steps versus glides
    jump_fraction, steps, glides = 0.0, 0, 0
    if frame_rate >= 50:
        delta = (cents[1:] - cents[:-1]) * (voiced[1:] & voiced[:-1]).float()
        big = delta.abs() > 100
        window = max(2, int(0.2 * frame_rate))
        cumulative = F.conv1d(F.pad(delta[None, None], (window - 1, 0)), torch.ones(1, 1, window))[0, 0].abs()
        i, n = 0, delta.numel()
        while i < n:
            if big[i]:
                steps += 1
                i += window
            elif cumulative[i] > 100 and not big[max(0, i - window) : i + 1].any():
                glides += 1
                i += window
            else:
                i += 1
        jump_fraction = steps / max(1, steps + glides)
    parts = {
        "fluctuation": float(max(0.0, min(1.0, (4.0 - fluctuation) / 4.0))),
        "vibrato": float(max(0.0, min(1.0, (regularity - 0.6) / 0.3))),
        "jumps": float(max(0.0, min(1.0, jump_fraction))) if frame_rate >= 50 else 0.0,
    }
    weights = {"fluctuation": 1.0, "vibrato": 1.0, "jumps": 1.0 if frame_rate >= 50 else 0.0}
    score = sum(parts[k] * weights[k] for k in parts) / max(1e-6, sum(weights.values()))
    return {
        "fluctuation_cents": fluctuation,
        "vibrato_regularity": regularity,
        "vibrato_peakiness": peakiness,
        "vibrato_rms_cents": vibrato_rms,
        "jump_fraction": jump_fraction,
        "transitions": steps + glides,
        "held_notes": len(segments),
        "parts": parts,
        "score": float(score),
        "ok": score < 0.5,
    }


def sync_lag(
    f0_hz: torch.Tensor,
    wave: torch.Tensor,
    sample_rate: int,
    frame_rate: float = 25.0,
    max_lag_ms: float = 200.0,
    limit_ms: float = 50.0,
) -> dict:
    """
    Lag, in ms, between the control events (voicing onsets and note changes over a semitone) and the audio's
    onsets (spectral flux), from their cross-correlation at 10 ms resolution. Positive: the audio moves after
    the control. Vibrato is deliberately not an event, so a wobble cannot masquerade as a lag.
    """
    hop = int(sample_rate / 100)
    f0 = f0_hz.float()
    n = wave.shape[-1] // hop
    f0_fine = F.interpolate(f0[None, None], size=n, mode="linear", align_corners=False)[0, 0]
    voiced = (f0_fine > 0).float()
    cents = torch.log2(f0_fine.clamp(min=1.0)) * 1200 * voiced
    onset = F.relu(voiced[1:] - voiced[:-1])
    change = ((cents[1:] - cents[:-1]).abs() > 100).float() * voiced[1:] * voiced[:-1]
    control = onset * 2 + change
    kernel = torch.ones(1, 1, 3) / 3
    control = F.conv1d(F.pad(control[None, None], (1, 1)), kernel)[0, 0]
    mag = torch.log(_stft(wave, 512, hop) + 1e-4)
    flux = F.relu(mag[:, 1:] - mag[:, :-1]).sum(0)
    flux = F.conv1d(F.pad(flux[None, None], (1, 1)), kernel)[0, 0]
    m = min(control.numel(), flux.numel())
    c, o = control[:m], flux[:m]
    if c.sum() < 1e-6 or o.abs().sum() < 1e-6:
        return {"lag_ms": 0.0, "correlation": 0.0, "events": 0, "ok": True}
    c, o = c - c.mean(), o - o.mean()
    max_lag = int(max_lag_ms / 10)
    best, best_lag = -2.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = c[: m - lag], o[lag:]
        else:
            a, b = c[-lag:], o[: m + lag]
        r = float((a * b).sum() / (a.norm() * b.norm() + 1e-9))
        if r > best + 1e-6 or (abs(r - best) <= 1e-6 and abs(lag) < abs(best_lag)):
            best, best_lag = r, lag
    lag_ms = best_lag * 10.0
    events = int((onset > 0).sum() + (change > 0).sum())
    # one sustained note has nothing to align on; a verdict needs several events and a real correlation
    reliable = events >= 4 and best >= 0.3
    return {
        "lag_ms": lag_ms,
        "correlation": best,
        "events": events,
        "reliable": reliable,
        "ok": (not reliable) or abs(lag_ms) <= limit_ms,
    }


def air_report(
    pred: torch.Tensor, ref: torch.Tensor, sample_rate: int, cutoff: float = 11000.0, limit_db: float = 8.0
) -> dict:
    """Energy above ``cutoff`` relative to the 4 kHz-``cutoff`` band, output against reference: the air a codec
    or a 24 kHz pipeline removed, in dB. Only meaningful when the sample rate carries that band."""
    if sample_rate / 2 <= cutoff + 1000:
        return {"air_deficit_db": 0.0, "ok": True, "measured": False}
    n_fft = 2048
    freqs = torch.fft.rfftfreq(n_fft, 1 / sample_rate)
    hi, mid = freqs >= cutoff, (freqs >= 4000) & (freqs < cutoff)

    def share(x):
        p = _stft(x, n_fft, n_fft // 4).pow(2).mean(-1)
        return float(10 * torch.log10((p[hi].mean() + 1e-12) / (p[mid].mean() + 1e-12)))

    deficit = share(ref) - share(pred)
    return {"air_deficit_db": deficit, "ok": abs(deficit) < limit_db, "measured": True}


def pre_echo_score(pred: torch.Tensor, ref: torch.Tensor, sample_rate: int, limit_db: float = 6.0) -> dict:
    """Energy in the 20 ms before the reference's onsets against the 20 ms after, output minus reference: a
    transform codec smears a transient backwards in time (pre-echo), which is the 'mp3' sound on sibilants."""
    from ..audio.refiner import flux_curve

    fr = flux_curve(ref[None], sample_rate)[0]
    where = onsets(fr)
    if where.numel() == 0:
        return {"pre_echo_db": 0.0, "onsets": 0, "ok": True}
    hop = int(sample_rate * 0.005)
    before, after = [], []
    for x in (pred, ref):
        b_sum, a_sum = 0.0, 0.0
        for i in where.tolist():
            centre = i * hop
            b_sum += float(x[max(0, centre - 4 * hop) : centre].pow(2).mean()) if centre > 4 * hop else 0.0
            a_sum += float(x[centre : centre + 4 * hop].pow(2).mean())
        before.append(b_sum)
        after.append(a_sum)
    ratio = [10 * math.log10((b + 1e-12) / (a + 1e-12)) for b, a in zip(before, after, strict=True)]
    excess = ratio[0] - ratio[1]
    return {"pre_echo_db": excess, "onsets": int(where.numel()), "ok": excess < limit_db}


def artifact_report(
    pred: torch.Tensor,
    sample_rate: int,
    ref: torch.Tensor | None = None,
    f0_hz: torch.Tensor | None = None,
    frame_rate: float = 25.0,
    f0_fine: torch.Tensor | None = None,
    fine_rate: float = 100.0,
) -> dict:
    """Every detector that applies, a ``failed`` list and an overall ``ok``."""
    report: dict = {
        "glitches": detect_glitches(pred, sample_rate),
        "buzz": frame_rate_buzz(pred, sample_rate, frame_rate),
    }
    if f0_hz is not None:
        report["coherence"] = harmonic_coherence(pred, f0_hz, sample_rate, frame_rate)
        report["sync"] = sync_lag(f0_hz, pred, sample_rate, frame_rate)
    if f0_fine is not None:
        report["robotic"] = robotic_score(f0_fine, fine_rate)
    elif f0_hz is not None:
        report["robotic"] = robotic_score(f0_hz, frame_rate)
    if ref is not None:
        report["hf"] = hf_character(pred, ref, sample_rate)
        report["consonants"] = consonant_sharpness(pred, ref, sample_rate)
        report["pre_echo"] = pre_echo_score(pred, ref, sample_rate)
        if sample_rate >= 44100:
            report["air"] = air_report(pred, ref, sample_rate)
    failed = [name for name, value in report.items() if isinstance(value, dict) and not value.get("ok", True)]
    report["failed"] = failed
    report["ok"] = not failed
    return report


def summarize_report(report: dict) -> str:
    lines = []
    for name, value in report.items():
        if name in ("failed", "ok") or not isinstance(value, dict):
            continue
        shown = {
            k: (round(v, 2) if isinstance(v, float) else v)
            for k, v in value.items()
            if k not in ("parts", "clicks", "dropouts")
        }
        if name == "glitches":
            shown = {
                "clicks": int(value["clicks"].numel()),
                "dropouts": int(value["dropouts"].numel()),
                "ok": value["ok"],
            }
        lines.append(f"  {name:10} {'ok  ' if value.get('ok', True) else 'FAIL'} {shown}")
    lines.append(f"  overall: {'ok' if report['ok'] else 'failed ' + ', '.join(report['failed'])}")
    return "\n".join(lines)


__all__ = [
    name for name in dir() if not name.startswith("_") and name not in ("math", "torch", "F", "detect_glitches")
]
