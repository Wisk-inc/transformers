# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
F1 and F2 from linear prediction (the autocorrelation method at about 11 kHz, order 2 + kHz, the roots of the
predictor), per voiced frame. A conversion that keeps the singer's vowels keeps F1 and F2 near the reference's
(after the speaker's own vocal-tract scaling, which a voice conversion is meant to change; compare the source's
vowels with the output's for content, the target's for identity).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..audio_io import resample


def lpc(frame: np.ndarray, order: int) -> np.ndarray:
    """Predictor coefficients ``a`` (``a[0] = 1``) by Levinson-Durbin."""
    r = np.correlate(frame, frame, mode="full")[len(frame) - 1 : len(frame) + order]
    a = np.zeros(order + 1)
    a[0], err = 1.0, r[0] + 1e-12
    for i in range(1, order + 1):
        k = -(r[i] + a[1:i] @ r[i - 1 : 0 : -1]) / err
        a[1 : i + 1] = a[1 : i + 1] + k * a[i - 1 :: -1][: i]
        err *= 1 - k * k
    return a


def formants(wave: torch.Tensor, sample_rate: int, frame_ms: float = 25.0, hop_ms: float = 10.0,
             max_frames: int = 400) -> np.ndarray:
    """``(frames, 2)`` F1 and F2 in Hz of the loud frames (NaN where none was found)."""
    target = 11025
    x = resample(wave.flatten().double(), sample_rate, target).numpy()
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])                               # pre-emphasis
    n, hop = int(target * frame_ms / 1000), int(target * hop_ms / 1000)
    order = 2 + target // 1000
    if len(x) < n:
        return np.zeros((0, 2))
    starts = np.arange(0, len(x) - n, hop)
    energy = np.array([np.mean(x[s : s + n] ** 2) for s in starts])
    loud = starts[energy > energy.max() * 10 ** (-3.0)]
    if len(loud) > max_frames:
        loud = loud[np.linspace(0, len(loud) - 1, max_frames).astype(int)]
    win = np.hamming(n)
    out = []
    for s in loud:
        a = lpc(x[s : s + n] * win, order)
        roots = np.roots(a)
        roots = roots[np.imag(roots) > 0]
        freqs = np.angle(roots) * target / (2 * math.pi)
        bw = -0.5 * target / math.pi * np.log(np.abs(roots) + 1e-12)
        cand = np.sort(freqs[(freqs > 90) & (bw < 400)])
        out.append([cand[0] if len(cand) > 0 else np.nan, cand[1] if len(cand) > 1 else np.nan])
    return np.array(out, dtype=np.float64).reshape(-1, 2)


def formant_metrics(pred: torch.Tensor, ref: torch.Tensor, sample_rate: int) -> dict:
    """The difference of the median F1 and F2 of the two takes (a level of vowel quality, robust to timing)."""
    p, r = formants(pred, sample_rate), formants(ref, sample_rate)
    if len(p) == 0 or len(r) == 0:
        return {"formant_error_hz": math.nan, "f1_error_hz": math.nan, "f2_error_hz": math.nan}
    mp, mr = np.nanmedian(p, 0), np.nanmedian(r, 0)
    e1, e2 = abs(mp[0] - mr[0]), abs(mp[1] - mr[1])
    return {"formant_error_hz": float(np.nanmean([e1, e2])), "f1_error_hz": float(e1), "f2_error_hz": float(e2)}


__all__ = ["formant_metrics", "formants", "lpc"]
