# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Long form: a five-minute track without drift.

Drift is what happens when time is accumulated instead of addressed. If every chunk's clock starts where the
previous chunk's output happened to end, a 5 ms error per chunk is a quarter of a beat by the last chorus.
Here nothing accumulates:

* **one global grid**: a frame ``i`` is the samples ``[i * hop, (i + 1) * hop)`` of the track, always, for the
  codec frames, the F0 curve, the phoneme durations and the output; every chunk is cut from that grid and
  pasted back at its absolute position;
* **beat-anchored chunks** (:func:`plan_chunks`): chunk edges snap to beats (or any anchors given), so a
  crossfade happens where the music turns, never in the middle of a run, and overlaps are generous;
* **re-anchoring per chunk**: durations and the alignment prior are normalized inside each chunk
  (``tristream.lyric_alignment`` fits the predicted total to the chunk), so an alignment error cannot be
  carried into the next chunk;
* **drift measured, not assumed** (:func:`drift_curve`, :func:`drift_rate`): the sync lag is estimated in
  20 s windows along the whole track; a slope in ms per minute is a rate to correct with :func:`time_warp`
  and regenerate, a flat line with an offset is a constant latency to shift;
* **phase that does not lose precision**: the refiner's oscillator accumulates phase in float64 and wraps
  it, so harmonic 40 at minute five is as locked as at second one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .artifacts import sync_lag


def plan_chunks(
    total_frames: int,
    chunk_frames: int = 750,
    overlap_frames: int = 50,
    anchors: torch.Tensor | None = None,
    snap_frames: int | None = None,
) -> list[tuple[int, int]]:
    """``(start, end)`` frame spans covering ``[0, total_frames)``; consecutive spans overlap by ``overlap_frames``
    and their edges move to the nearest anchor (frame index) within ``snap_frames`` when anchors are given."""
    if total_frames <= chunk_frames:
        return [(0, total_frames)]
    snap = overlap_frames // 2 if snap_frames is None else snap_frames
    spans, start = [], 0
    while start < total_frames:
        end = min(total_frames, start + chunk_frames)
        if anchors is not None and anchors.numel() and end < total_frames:
            distance = (anchors.float() - end).abs()
            k = int(distance.argmin())
            if distance[k] <= snap:
                end = int(anchors[k])
        spans.append((start, end))
        if end >= total_frames:
            break
        start = max(start + 1, end - overlap_frames)
    return spans


def time_warp(curve: torch.Tensor, factor: float) -> torch.Tensor:
    """Resample a control curve along time by ``factor`` (1.001 = 0.1 % slower); voicing stays hard."""
    n = curve.shape[-1]
    m = max(1, int(round(n * factor)))
    x = curve.float()[None, None]
    out = F.interpolate(x, size=m, mode="linear", align_corners=False)[0, 0]
    voiced = F.interpolate((x > 0).float(), size=m, mode="nearest")[0, 0] > 0.5
    return torch.where(voiced, out, torch.zeros_like(out))


def drift_curve(
    f0_hz: torch.Tensor,
    wave: torch.Tensor,
    sample_rate: int,
    frame_rate: float = 25.0,
    window_s: float = 20.0,
    max_lag_ms: float = 400.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sync lag per window along the track: ``(centre times s, lags ms, reliable flags)``."""
    hop = int(sample_rate / frame_rate)
    frames_per_window = int(window_s * frame_rate)
    times, lags, flags = [], [], []
    for start in range(0, f0_hz.numel(), frames_per_window):
        stop = min(f0_hz.numel(), start + frames_per_window)
        if stop - start < frames_per_window // 2:
            break
        seg = wave[start * hop : stop * hop]
        r = sync_lag(f0_hz[start:stop], seg, sample_rate, frame_rate, max_lag_ms=max_lag_ms)
        times.append((start + stop) / 2 / frame_rate)
        lags.append(r["lag_ms"])
        flags.append(r.get("reliable", True))
    return torch.tensor(times), torch.tensor(lags), torch.tensor(flags, dtype=torch.bool)


def drift_rate(times: torch.Tensor, lags: torch.Tensor, reliable: torch.Tensor | None = None) -> dict:
    """Least-squares slope of lag against time: ``ms_per_minute`` (drift), ``offset_ms`` (constant latency),
    and the warp factor that would cancel the drift."""
    if reliable is not None and reliable.any():
        times, lags = times[reliable], lags[reliable]
    if times.numel() < 2:
        offset = float(lags.mean()) if lags.numel() else 0.0
        return {"ms_per_minute": 0.0, "offset_ms": offset, "warp": 1.0, "windows": int(times.numel())}
    t = times - times.mean()
    slope = float((t * (lags - lags.mean())).sum() / (t.pow(2).sum() + 1e-9))  # ms per second
    offset = float(lags.mean() - slope * times.mean())
    return {
        "ms_per_minute": slope * 60.0,
        "offset_ms": offset,
        "warp": 1.0 / (1.0 + slope / 1000.0),  # the audio runs slow by `slope` ms per s: play the controls faster
        "windows": int(times.numel()),
    }


def render_long(
    synth,
    total_frames: int,
    hop: int,
    chunk_frames: int = 750,
    overlap_frames: int = 50,
    anchors: torch.Tensor | None = None,
    log=None,
) -> torch.Tensor:
    """
    ``synth(start, end) -> wave`` renders the frames ``[start, end)`` and returns ``(end - start) * hop`` samples (or
    fewer; the rest is padded). Chunks are pasted at ``start * hop`` on the global grid and overlaps are
    crossfaded with raised cosines. Returns ``(total_frames * hop,)``.
    """
    out = torch.zeros(total_frames * hop)
    weight = torch.zeros(total_frames * hop)
    for k, (start, end) in enumerate(plan_chunks(total_frames, chunk_frames, overlap_frames, anchors)):
        wave = synth(start, end)
        n = (end - start) * hop
        wave = F.pad(wave[:n], (0, max(0, n - wave.shape[-1])))
        fade = torch.ones(n)
        ramp = min(overlap_frames * hop, n // 2)
        if ramp > 0:
            edge = 0.5 - 0.5 * torch.cos(torch.linspace(0, torch.pi, ramp))
            if start > 0:
                fade[:ramp] = edge
            if end < total_frames:
                fade[-ramp:] = edge.flip(0)
        out[start * hop : end * hop] += wave * fade
        weight[start * hop : end * hop] += fade
        if log:
            log(f"[longform] chunk {k}: frames {start}-{end}")
    return out / weight.clamp(min=1e-6)
