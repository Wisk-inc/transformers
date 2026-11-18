# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Vibrato rate and depth of a pitch track: the strongest periodicity between 3 and 9 Hz of the detrended cents
contour of its voiced frames. Needs a pitch track at 50 frames a second or more."""

from __future__ import annotations

import math

import torch

RATE_BAND = (3.0, 9.0)


def vibrato(f0: torch.Tensor, frame_rate: float, trend_seconds: float = 0.5) -> dict:
    f0 = f0.flatten().double()
    voiced = f0 > 0
    if int(voiced.sum()) < int(frame_rate * 0.6) or frame_rate < 20:
        return {"rate_hz": math.nan, "extent_cents": math.nan, "present": False}
    c = 1200 * torch.log2(f0[voiced] / f0[voiced].median())
    k = max(3, int(trend_seconds * frame_rate) | 1)
    trend = torch.nn.functional.avg_pool1d(
        torch.nn.functional.pad(c[None, None], (k // 2, k // 2), mode="replicate"), k, stride=1
    )[0, 0]
    d = (c - trend) * torch.hann_window(c.numel(), dtype=torch.float64)
    spec = torch.fft.rfft(d).abs()
    freqs = torch.fft.rfftfreq(c.numel(), d=1.0 / frame_rate)
    band = (freqs >= RATE_BAND[0]) & (freqs <= RATE_BAND[1])
    if not band.any():
        return {"rate_hz": math.nan, "extent_cents": math.nan, "present": False}
    idx = torch.nonzero(band).flatten()
    peak = idx[spec[idx].argmax()]
    floor = spec[(freqs > 0.5) & ~band].median() if ((freqs > 0.5) & ~band).any() else spec.new_tensor(1e-9)
    extent = float((c - trend).std() * math.sqrt(2.0))
    return {"rate_hz": float(freqs[peak]), "extent_cents": extent, "present": bool(spec[peak] > 3 * floor)}


def vibrato_metrics(pred: torch.Tensor, ref: torch.Tensor, frame_rate: float) -> dict:
    p, r = vibrato(pred, frame_rate), vibrato(ref, frame_rate)
    both = p["present"] and r["present"]
    return {"vibrato_rate_error_hz": abs(p["rate_hz"] - r["rate_hz"]) if both else math.nan,
            "vibrato_extent_error_cents": abs(p["extent_cents"] - r["extent_cents"])
            if not (math.isnan(p["extent_cents"]) or math.isnan(r["extent_cents"])) else math.nan,
            "vibrato_kept": float(p["present"] == r["present"])}


__all__ = ["RATE_BAND", "vibrato", "vibrato_metrics"]
