# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Effects and repairs on waveforms, differentiable where it matters, in PyTorch:

* room: :func:`reverb` (synthetic exponentially decaying impulse response at a chosen decay time, with
  pre-delay and damping), :func:`echo` (feedback delay in beats), :func:`match_reverb` (give a dry vocal the
  room the backing track has, from :func:`analysis.reverb_time`);
* tone: :func:`muffle` (low-pass, "singing from the next room" / phone / lo-fi), :func:`presence` (a gentle
  high shelf), :func:`de_ess` (compress the 5-9 kHz band when it spikes), :func:`formant_shift` (resample the
  spectral envelope without moving the pitch: the register trick for notes a voice cannot support);
* pitch: :func:`retune` (pull an F0 curve towards the scale with a retune speed, from transparent to the
  hard "autotune" sound), :func:`add_vibrato`;
* arrangement: :func:`double` (a slightly detuned, delayed copy for width), :func:`duck` (sidechain the
  instrumental under the voice so the voice is never buried), :func:`place_in_mix` (level and tone matching);
* dynamics: :func:`compress`, :func:`limit`, :func:`normalize_loudness`;
* breath: :func:`insert_breaths` (shaped noise bursts at phrase starts), and the opposite, :func:`strip_breaths`;
* quality control: :func:`detect_glitches` (clicks, dropouts, clipping, DC) and :func:`repair_glitches`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .analysis import scale_pitch_classes


# ------------------------------------------------------------------------------------------- helpers --
def _fft_convolve(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1] + kernel.shape[-1] - 1
    size = 1 << (n - 1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(x, size) * torch.fft.rfft(kernel, size), size)[..., : x.shape[-1]]
    return y


def _biquad_lowpass(x: torch.Tensor, cutoff: float, sample_rate: int, q: float = 0.707) -> torch.Tensor:
    w0 = 2 * math.pi * cutoff / sample_rate
    alpha = math.sin(w0) / (2 * q)
    b0 = (1 - math.cos(w0)) / 2
    b1 = 1 - math.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha
    b = torch.tensor([b0, b1, b2]) / a0
    a = torch.tensor([1.0, a1 / a0, a2 / a0])
    return _lfilter(x, b, a)


def _lfilter(x: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Direct-form IIR filter (a slow but dependency-free loop over samples; fine for seconds of audio)."""
    y = torch.zeros_like(x)
    x_hist = torch.zeros(3)
    y_hist = torch.zeros(3)
    out = []
    for n in range(x.shape[-1]):
        x_hist = torch.roll(x_hist, 1)
        x_hist[0] = x[..., n]
        value = (b * x_hist).sum() - (a[1:] * y_hist[:2]).sum()
        y_hist = torch.roll(y_hist, 1)
        y_hist[0] = value
        out.append(value)
    y = torch.stack(out, dim=-1)
    return y


def lowpass(x: torch.Tensor, cutoff: float, sample_rate: int, taps: int = 255) -> torch.Tensor:
    """Linear-phase FIR low-pass (windowed sinc); fast through FFT convolution."""
    t = torch.arange(taps) - (taps - 1) / 2
    fc = cutoff / sample_rate
    kernel = 2 * fc * torch.sinc(2 * fc * t) * torch.hann_window(taps, periodic=False)
    kernel = kernel / kernel.sum()
    y = _fft_convolve(x, kernel)
    return torch.roll(y, -(taps - 1) // 2, dims=-1)


def highpass(x: torch.Tensor, cutoff: float, sample_rate: int, taps: int = 255) -> torch.Tensor:
    return x - lowpass(x, cutoff, sample_rate, taps)


def bandpass(x: torch.Tensor, low: float, high: float, sample_rate: int) -> torch.Tensor:
    return lowpass(highpass(x, low, sample_rate), high, sample_rate)


# ---------------------------------------------------------------------------------------------- room --
def impulse_response(
    decay_seconds: float, sample_rate: int, pre_delay: float = 0.01, damping: float = 0.5, seed: int = 0
) -> torch.Tensor:
    """Exponentially decaying noise with high frequencies dying faster (``damping``): a plausible room."""
    g = torch.Generator().manual_seed(seed)
    length = int(max(decay_seconds, 0.05) * sample_rate)
    t = torch.arange(length) / sample_rate
    noise = torch.randn(length, generator=g)
    tau = max(decay_seconds, 0.05) / 6.9  # -60 dB at decay_seconds
    tail = noise * torch.exp(-t / tau)
    if damping > 0:
        tail = lowpass(tail, 8000 * (1 - 0.8 * damping), sample_rate)
    ir = torch.zeros(int(pre_delay * sample_rate) + length)
    ir[int(pre_delay * sample_rate) :] = tail
    ir[0] = 1.0
    return ir / ir.abs().max()


def reverb(
    x: torch.Tensor,
    sample_rate: int,
    decay_seconds: float = 1.2,
    mix: float = 0.25,
    pre_delay: float = 0.02,
    damping: float = 0.5,
) -> torch.Tensor:
    ir = impulse_response(decay_seconds, sample_rate, pre_delay, damping)
    wet = _fft_convolve(x, ir)
    wet = wet / (wet.abs().max() + 1e-9) * (x.abs().max() + 1e-9)
    return (1 - mix) * x + mix * wet


def echo(
    x: torch.Tensor, sample_rate: int, delay_seconds: float, feedback: float = 0.35, mix: float = 0.3, repeats: int = 4
) -> torch.Tensor:
    out = x.clone()
    delay = int(delay_seconds * sample_rate)
    tap = x
    for _ in range(repeats):
        tap = torch.roll(tap, delay, dims=-1) * feedback
        tap[..., :delay] = 0
        out = out + mix * tap
    return out


def beat_echo(x: torch.Tensor, sample_rate: int, bpm: float, beats: float = 0.75, **kw) -> torch.Tensor:
    """An echo locked to the tempo (dotted eighth by default)."""
    return echo(x, sample_rate, delay_seconds=60.0 / bpm * beats, **kw)


ROOM_BANDS = ((60.0, 250.0), (250.0, 1000.0), (1000.0, 4000.0), (4000.0, 11000.0))


def decay_time(wave: torch.Tensor, sample_rate: int, hop_ms: float = 10.0, max_seconds: float = 3.0) -> float:
    """
    T60 from the decays that actually happen in the signal: after each level peak the envelope is followed
    down to 25 dB under the peak, and the slope of its last 20 dB (the part after the sound itself has stopped)
    is extrapolated to 60 dB; median over peaks. A held note's length is not counted as reverb. Returns 0 when
    nothing decays far enough.
    """
    env = rms_envelope(wave, sample_rate, 30.0)
    hop = max(1, int(sample_rate * hop_ms / 1000))
    db = 20 * torch.log10(env[::hop] + 1e-6)
    n = db.numel()
    if n < 20:
        return 0.0
    floor = float(db.max()) - 30
    w = max(1, int(100 / hop_ms))
    limit = int(max_seconds * 1000 / hop_ms)
    times, i = [], 0
    while i < n:
        if db[i] >= floor and db[i] == db[max(0, i - w) : i + w + 1].max():
            top = float(db[i])
            j, last, end = i + 1, top, None
            while j < n and j - i < limit:
                value = float(db[j])
                if value > last + 3.0:  # a new onset: stop following
                    break
                last = min(last, value)
                if value <= top - 25:
                    end = j
                    break
                j += 1
            span = 20.0
            if end is None and top - last >= 15:  # the signal ended mid-decay: use the last 10 dB of what there is
                end, span = int(db[i:j].argmin()) + i, 10.0
            if end is not None:
                target = float(db[end]) + span
                k = end
                while k > i and float(db[k]) < target:
                    k -= 1
                if end > k:
                    times.append(60.0 * (end - k) * hop_ms / 1000 / span)
            i = max(j, i + 1)
        else:
            i += 1
    # a voice's own releases can only make a decay look slower than the room, never faster: take the fast quartile
    return min(max_seconds, float(torch.tensor(times).quantile(0.25))) if times else 0.0


def multiband_reverb_time(wave: torch.Tensor, sample_rate: int) -> list[float]:
    """The decay per octave-ish band: real rooms lose highs faster than lows, and a voice swap that gives the
    new voice one broadband decay sounds pasted on. Returns seconds per band of ``ROOM_BANDS``."""
    broadband = decay_time(wave, sample_rate)
    out = []
    for lo, hi in ROOM_BANDS:
        hi = min(hi, sample_rate / 2 - 100)
        band = bandpass(wave, lo, hi, sample_rate) if hi > lo else wave
        value = decay_time(band, sample_rate)
        out.append(value if value > 0 else broadband)
    return out


def estimate_room(wave: torch.Tensor, sample_rate: int) -> dict:
    """What can be read of a room from a vocal alone: per-band decays, a pre-delay guess from the envelope's
    first echo (10-40 ms autocorrelation peak) and a wet fraction from the broadband decay."""
    decays = multiband_reverb_time(wave, sample_rate)
    env = rms_envelope(wave, sample_rate, 2.0)
    env = env - env.mean()
    lo, hi = int(0.010 * sample_rate), int(0.040 * sample_rate)
    best, pre = 0.0, 0.015
    for lag in range(lo, hi, max(1, (hi - lo) // 30)):
        r = float((env[:-lag] * env[lag:]).mean() / (env.pow(2).mean() + 1e-9))
        if r > best:
            best, pre = r, lag / sample_rate
    broadband = sum(decays) / len(decays)
    return {"decays": decays, "pre_delay": pre, "wet": float(min(0.5, max(0.0, (broadband - 0.3) / 2.0)))}


def build_rir(
    decays: list[float],
    sample_rate: int,
    pre_delay: float = 0.012,
    early: tuple[float, ...] = (0.007, 0.013, 0.021, 0.034),
    seed: int = 0,
) -> torch.Tensor:
    """A room impulse response from per-band decays: a direct tap, a few early reflections, and one band-limited
    decaying noise tail per band of ``ROOM_BANDS`` with its own time constant."""
    g = torch.Generator().manual_seed(seed)
    longest = max(max(decays), 0.05)
    length = int(longest * sample_rate)
    t = torch.arange(length) / sample_rate
    tail = torch.zeros(length)
    for (lo, hi), decay in zip(ROOM_BANDS, decays, strict=True):
        hi = min(hi, sample_rate / 2 - 100)
        if hi <= lo:
            continue
        noise = torch.randn(length, generator=g)
        tau = max(decay, 0.05) / 6.9
        band = bandpass(noise * torch.exp(-t / tau), lo, hi, sample_rate)
        tail = tail + band
    ir = torch.zeros(int(pre_delay * sample_rate) + length)
    ir[0] = 1.0
    for k, delay in enumerate(early):
        pos = int(delay * sample_rate)
        if pos < ir.numel():
            ir[pos] += 0.4 / (k + 1) * (1 if k % 2 == 0 else -1)
    ir[int(pre_delay * sample_rate) :] += tail / (tail.abs().max() + 1e-9) * 0.6
    return ir / ir.abs().max()


def apply_room(x: torch.Tensor, room: dict, sample_rate: int, mix: float | None = None) -> torch.Tensor:
    ir = build_rir(room["decays"], sample_rate, room.get("pre_delay", 0.012))
    wet = _fft_convolve(x, ir)
    wet = wet / (wet.abs().max() + 1e-9) * (x.abs().max() + 1e-9)
    mix = room.get("wet", 0.2) if mix is None else mix
    return (1 - mix) * x + mix * wet


def match_reverb(
    vocal: torch.Tensor,
    sample_rate: int,
    backing_decay: float,
    vocal_decay: float = 0.0,
    mix: float = 0.2,
    room: dict | None = None,
) -> torch.Tensor:
    """Give a dry vocal the room of the original (only adds; it never tries to remove reverb). With ``room`` from
    :func:`estimate_room` the per-band decays and the pre-delay are matched, not one broadband tail."""
    if room is not None and max(room["decays"]) > 0.1:
        return apply_room(vocal, room, sample_rate, mix if mix != 0.2 else None)
    extra = max(0.0, backing_decay - vocal_decay)
    return reverb(vocal, sample_rate, decay_seconds=extra, mix=mix) if extra > 0.1 else vocal


# ---------------------------------------------------------------------------------------------- tone --
def muffle(x: torch.Tensor, sample_rate: int, cutoff: float = 1500.0, amount: float = 1.0) -> torch.Tensor:
    return (1 - amount) * x + amount * lowpass(x, cutoff, sample_rate)


def presence(x: torch.Tensor, sample_rate: int, gain_db: float = 3.0, corner: float = 3000.0) -> torch.Tensor:
    high = highpass(x, corner, sample_rate)
    return x + (10 ** (gain_db / 20) - 1) * high


def de_ess(x: torch.Tensor, sample_rate: int, threshold: float = 0.1, ratio: float = 4.0) -> torch.Tensor:
    band = bandpass(x, 5000, min(9000, sample_rate / 2 - 100), sample_rate)
    env = rms_envelope(band, sample_rate)
    gain = torch.where(env > threshold, (threshold / env.clamp(min=1e-9)) ** (1 - 1 / ratio), torch.ones_like(env))
    return x - band + band * gain


def rms_envelope(x: torch.Tensor, sample_rate: int, window_ms: float = 10.0) -> torch.Tensor:
    n = max(1, int(window_ms / 1000 * sample_rate))
    kernel = torch.ones(1, 1, n) / n
    env = F.conv1d(F.pad(x.pow(2)[None, None], (n // 2, n - 1 - n // 2)), kernel)[0, 0]
    return env.sqrt()


def formant_shift(x: torch.Tensor, sample_rate: int, ratio: float, n_fft: int = 1024, hop: int = 256) -> torch.Tensor:
    """
    Move the spectral envelope by ``ratio`` (1.1 = brighter / smaller tract, 0.9 = darker / larger) and keep the
    harmonics where they are: cepstral envelope, warped, re-applied. The register trick: a tenor line sung by a
    bass voice an octave down keeps its own formants; a high note moved into range keeps the singer's size.
    """
    window = torch.hann_window(n_fft)
    spec = torch.stft(x, n_fft, hop, window=window, return_complex=True)
    mag, phase = spec.abs(), spec.angle()
    log_mag = mag.clamp(min=1e-6).log()
    cepstrum = torch.fft.irfft(log_mag, dim=0)
    lifter = torch.zeros(cepstrum.shape[0])
    lifter[:30] = 1.0
    envelope = torch.fft.rfft(cepstrum * lifter[:, None], dim=0).real
    bins = torch.arange(mag.shape[0]).float()
    source_bins = (bins / ratio).clamp(max=mag.shape[0] - 1)
    lo = source_bins.floor().long()
    hi = (lo + 1).clamp(max=mag.shape[0] - 1)
    frac = (source_bins - lo.float())[:, None]
    warped = envelope[lo] * (1 - frac) + envelope[hi] * frac
    new_mag = (log_mag - envelope + warped).exp()
    out = torch.istft(torch.polar(new_mag, phase), n_fft, hop, window=window, length=x.shape[-1])
    return out


# --------------------------------------------------------------------------------------------- pitch --
def retune(f0_hz: torch.Tensor, tonic: int, mode: str, speed: float = 0.5, frame_rate: float = 25.0) -> torch.Tensor:
    """
    Pull an F0 curve towards the nearest scale note. ``speed`` 0 leaves it, 1 snaps instantly (the hard
    autotune sound); in between the retune follows with a time constant, which keeps vibrato and scoops.
    """
    allowed = scale_pitch_classes(tonic, mode)
    voiced = f0_hz > 0
    midi = 69 + 12 * torch.log2(f0_hz.clamp(min=1.0) / 440.0)
    target = midi.clone()
    for i, m in enumerate(midi.tolist()):
        if not voiced[i]:
            continue
        candidates = [n for n in range(int(m) - 6, int(m) + 7) if n % 12 in allowed]
        target[i] = min(candidates, key=lambda n: abs(n - m))
    if speed >= 1.0:
        out = target
    else:
        alpha = 1 - math.exp(-speed * 8 / frame_rate)  # time constant shrinks as speed grows
        out = midi.clone()
        state = midi[0]
        for i in range(midi.shape[0]):
            state = state + alpha * (target[i] - state) if voiced[i] else midi[i]
            out[i] = state
        out = speed * out + (1 - speed) * midi
    hz = 440.0 * torch.pow(2.0, (out - 69) / 12)
    return torch.where(voiced, hz, torch.zeros_like(hz))


def add_vibrato(
    f0_hz: torch.Tensor,
    frame_rate: float = 25.0,
    rate_hz: float = 5.5,
    depth_semitones: float = 0.4,
    onset_frames: int = 6,
) -> torch.Tensor:
    t = torch.arange(f0_hz.shape[0]) / frame_rate
    ramp = torch.clamp(torch.arange(f0_hz.shape[0]) / max(onset_frames, 1), max=1.0)
    cents = depth_semitones * ramp * torch.sin(2 * math.pi * rate_hz * t)
    return torch.where(f0_hz > 0, f0_hz * torch.pow(2.0, cents / 12), f0_hz)


# ---------------------------------------------------------------------------------------- arrangement --
def double(
    x: torch.Tensor, sample_rate: int, detune_cents: float = 8.0, delay_ms: float = 18.0, mix: float = 0.5
) -> torch.Tensor:
    ratio = 2 ** (detune_cents / 1200)
    n = x.shape[-1]
    idx = torch.arange(n).float() / ratio
    lo = idx.floor().long().clamp(max=n - 1)
    hi = (lo + 1).clamp(max=n - 1)
    frac = idx - lo.float()
    detuned = x[..., lo] * (1 - frac) + x[..., hi] * frac
    detuned = torch.roll(detuned, int(delay_ms / 1000 * sample_rate), dims=-1)
    return x + mix * detuned


def duck(
    instrumental: torch.Tensor,
    vocal: torch.Tensor,
    sample_rate: int,
    depth_db: float = 4.0,
    attack_ms: float = 10.0,
    release_ms: float = 150.0,
) -> torch.Tensor:
    """Lower the instrumental while the voice is present (sidechain): the voice is never buried."""
    env = rms_envelope(vocal, sample_rate, window_ms=attack_ms)
    env = env / (env.max() + 1e-9)
    release = int(release_ms / 1000 * sample_rate)
    smoothed = env.clone()
    for i in range(1, env.shape[-1]):  # one-pole release
        smoothed[i] = max(float(env[i]), float(smoothed[i - 1]) * (1 - 1 / max(release, 1)))
    gain = 10 ** (-depth_db / 20 * smoothed)
    return instrumental * gain


def place_in_mix(
    vocal: torch.Tensor,
    instrumental: torch.Tensor,
    sample_rate: int,
    vocal_db_above: float = 2.0,
    duck_db: float = 3.0,
    room_decay: float = 0.0,
) -> torch.Tensor:
    """Level the vocal to sit ``vocal_db_above`` dB over the instrumental, duck the instrumental, add the room."""
    v = vocal / (vocal.pow(2).mean().sqrt() + 1e-9)
    i = instrumental / (instrumental.pow(2).mean().sqrt() + 1e-9)
    v = v * 10 ** (vocal_db_above / 20)
    if room_decay > 0.1:
        v = reverb(v, sample_rate, decay_seconds=room_decay, mix=0.15)
    mix = duck(i, v, sample_rate, depth_db=duck_db) + v
    return limit(mix)


# ------------------------------------------------------------------------------------------ dynamics --
def compress(
    x: torch.Tensor, sample_rate: int, threshold_db: float = -18.0, ratio: float = 3.0, makeup_db: float = 0.0
) -> torch.Tensor:
    env_db = 20 * rms_envelope(x, sample_rate, 20.0).clamp(min=1e-6).log10()
    over = (env_db - threshold_db).clamp(min=0)
    gain_db = -over * (1 - 1 / ratio) + makeup_db
    return x * 10 ** (gain_db / 20)


def limit(x: torch.Tensor, ceiling: float = 0.98) -> torch.Tensor:
    peak = x.abs().max()
    y = x * (ceiling / peak) if peak > ceiling else x
    return torch.tanh(y * 1.2) / math.tanh(1.2) * ceiling if y.abs().max() > ceiling * 0.999 else y


def normalize_loudness(x: torch.Tensor, target_rms_db: float = -20.0) -> torch.Tensor:
    current = 20 * (x.pow(2).mean().sqrt() + 1e-9).log10()
    return x * 10 ** ((target_rms_db - current) / 20)


# -------------------------------------------------------------------------------------------- breath --
def insert_breaths(
    x: torch.Tensor,
    sample_rate: int,
    phrase_starts: list[float],
    length_ms: float = 220.0,
    level_db: float = -30.0,
    seed: int = 0,
) -> torch.Tensor:
    """A shaped, band-limited noise burst before each phrase start (seconds): the sound of a singer taking air."""
    g = torch.Generator().manual_seed(seed)
    out = x.clone()
    n = int(length_ms / 1000 * sample_rate)
    t = torch.arange(n) / n
    shape = torch.sin(math.pi * t) ** 1.5
    for start in phrase_starts:
        pos = int(start * sample_rate) - n
        if pos < 0 or pos + n > out.shape[-1]:
            continue
        noise = bandpass(torch.randn(n, generator=g), 400, min(6000, sample_rate / 2 - 100), sample_rate)
        noise = noise / (noise.abs().max() + 1e-9) * 10 ** (level_db / 20) * shape
        out[..., pos : pos + n] += noise
    return out


def strip_breaths(x: torch.Tensor, sample_rate: int, threshold_db: float = -32.0) -> torch.Tensor:
    """Attenuate noise-like low-level passages between phrases (a gentle gate on the unvoiced floor)."""
    env_db = 20 * rms_envelope(x, sample_rate, 30.0).clamp(min=1e-6).log10()
    gain = torch.where(env_db < threshold_db, torch.full_like(env_db, 0.25), torch.ones_like(env_db))
    return x * gain


# -------------------------------------------------------------------------------------------- glitches --
def detect_glitches(x: torch.Tensor, sample_rate: int) -> dict:
    """Clicks (sample-to-sample jumps far above the local level), dropouts (sudden silence), clipping, DC offset."""
    diff = (x[..., 1:] - x[..., :-1]).abs()
    local = rms_envelope(diff, sample_rate, 20.0) + 1e-4  # the usual sample-to-sample motion around each point
    clicks = ((diff > 6 * local) & (diff > 0.05)).nonzero().flatten()
    env = rms_envelope(x, sample_rate, 5.0)
    quiet = env < 1e-4
    loud_before = torch.roll(env, int(0.02 * sample_rate)) > 0.02
    dropouts = (quiet & loud_before).nonzero().flatten()
    clipped = (x.abs() >= 0.999).float().mean().item()
    dc = x.mean().item()
    return {"clicks": clicks, "dropouts": dropouts, "clipped_fraction": clipped, "dc_offset": dc,
            "ok": clicks.numel() == 0 and dropouts.numel() == 0 and clipped < 1e-4 and abs(dc) < 0.02}  # fmt: skip


def repair_glitches(x: torch.Tensor, sample_rate: int, report: dict | None = None) -> torch.Tensor:
    report = report or detect_glitches(x, sample_rate)
    out = x - report["dc_offset"]
    for pos in report["clicks"].tolist():
        a, b = max(0, pos - 32), min(out.shape[-1], pos + 32)
        if b - a > 2:
            ramp = torch.linspace(float(out[..., a]), float(out[..., b - 1]), b - a)
            out[..., a:b] = ramp
    for pos in report["dropouts"].tolist():  # cross-fade over a dropout with the last good audio
        a, b = max(0, pos - int(0.01 * sample_rate)), min(out.shape[-1], pos + int(0.02 * sample_rate))
        if a > b - a:
            out[..., a:b] = out[..., a - (b - a) : a] * torch.linspace(1, 0, b - a) + out[..., a:b] * torch.linspace(
                0, 1, b - a
            )
    return limit(out)
