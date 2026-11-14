# Copyright 2026 Nathan. Apache-2.0.
"""
The hand-over: when a forecast leaves the range calibration was trained on, the raw consensus takes it back.

    final = (1 - g) * calibrated + g * raw consensus,      g = clip(support score, 0, 1) * agreement

``g`` is the share of trust handed to the raw systems. Inside the training range the support score is negative and
``g`` is exactly zero, so every ordinary day, and every score computed on ordinary days, is untouched by
construction. Beyond the edge ``g`` rises linearly in the log of the amount; with the default width the raw consensus
is the forecast at twice the edge.

``agreement`` guards against one wild system: it is the share of the systems that themselves call for heavy rain
(here, raw amount at least ``heavy`` mm), scaled so that half of them agreeing is enough. One model drawing a bullseye
where the others are dry does not take the forecast away from the calibration.

The mixture is taken threshold by threshold, in probability. Both parts are decreasing in the threshold, so the result
is too, and the expected amount is the same mixture of the two expected amounts.
"""

from __future__ import annotations

import numpy as np

from .support import Support


def handover(support: Support, raw_mean: np.ndarray, agree_share: np.ndarray | None = None,
             cap: float = 0.95) -> np.ndarray:
    """The share of trust handed to the raw consensus per row, in ``[0, cap]``: zero inside the training range."""
    g = np.clip(support.score(raw_mean), 0.0, 1.0)
    if agree_share is not None:
        g = g * np.clip(2.0 * np.asarray(agree_share, np.float64), 0.0, 1.0)
    return np.clip(g, 0.0, cap).astype(np.float32)


def blend_raw(p_calibrated: np.ndarray, p_raw: np.ndarray, g: np.ndarray) -> np.ndarray:
    """The final chances ``(N, K)``: a mixture of the calibrated and raw chances with weight ``g`` (N,) on the raw."""
    g = np.asarray(g, np.float32)[:, None]
    out = (1.0 - g) * p_calibrated + g * p_raw
    return np.minimum.accumulate(out, axis=1)


def agree_share_from(p_raw: np.ndarray, heavy_index: int) -> np.ndarray:
    """The share of the systems that call for heavy rain, read off the consensus chances: the consensus chance of the
    ``heavy_index`` threshold itself. Each system votes with the share of its members that reach the threshold, so
    this is exactly the share of systems calling for it when the systems are single runs or unanimous ensembles."""
    return np.clip(p_raw[:, heavy_index], 0.0, 1.0)


__all__ = ["agree_share_from", "blend_raw", "handover"]
