# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""An output's loudness made to follow its source: the same integrated loudness (BS.1770) and, with ``contour``, the
same slow dynamics (a gain curve smoothed over ``smoothing_ms``, limited to ``max_gain_db``). A conversion that
flattens the dynamics of a performance loses its phrasing; the benchmark's loudness correlation measures that."""

from __future__ import annotations

import math

import torch

from ..bench.loudness import integrated_loudness


def match_loudness(out: torch.Tensor, reference: torch.Tensor, sample_rate: int, peak: float = 0.99) -> torch.Tensor:
    a, b = integrated_loudness(out, sample_rate), integrated_loudness(reference, sample_rate)
    if not (math.isfinite(a) and math.isfinite(b)):
        return out
    y = out * 10 ** ((b - a) / 20)
    top = float(y.abs().max())
    return y * (peak / top) if top > peak else y


def _envelope_db(x: torch.Tensor, sample_rate: int, smoothing_ms: float) -> torch.Tensor:
    k = max(1, int(sample_rate * smoothing_ms / 1000))
    power = torch.nn.functional.avg_pool1d(x.flatten()[None, None].pow(2), k, stride=1, padding=k // 2,
                                           count_include_pad=False)[0, 0][: x.numel()]
    return 10 * torch.log10(power + 1e-10)


def match_contour(out: torch.Tensor, reference: torch.Tensor, sample_rate: int, smoothing_ms: float = 200.0,
                  max_gain_db: float = 12.0) -> torch.Tensor:
    n = min(out.numel(), reference.numel())
    o, r = out.flatten()[:n], reference.flatten()[:n]
    gain = (_envelope_db(r, sample_rate, smoothing_ms) - _envelope_db(o, sample_rate, smoothing_ms))
    gain = gain - gain.median()
    gain = gain.clamp(-max_gain_db, max_gain_db)
    y = o * 10 ** (gain / 20)
    return match_loudness(y, reference, sample_rate)


__all__ = ["match_contour", "match_loudness"]
