# Copyright 2026 Nathan. Apache-2.0.
"""
Why the forecast says what it says: occlusion attribution.

For a forecast and a set of points, each group of inputs is replaced in turn by a neutral value (zero after
normalisation, or "not observed" for observations) and the forecast is recomputed; the change in the chance of rain
in the next 24 hours is that group's contribution. Occlusion is slower than gradient methods but faithful for
groups of correlated inputs, and it answers the question people ask: "would the forecast be different without the
rain already on radar / without the front / without the mountains?".
"""

from __future__ import annotations

import numpy as np
import torch

from ..dataset import TARGET_DYNAMIC
from ..features import HOURLY_CHANNELS

GROUPS = {
    "driving model rain": {"hourly": ["tp"], "point": ["driving_rain"]},
    "rain already falling": {"point": ["memory_rain", "memory_known", "advected_rain", "advected_known"], "mesh": [0, 1]},
    "fronts and shear lines": {"hourly": ["theta_e850", "theta_e_grad", "frontness", "shear_line", "front_eta"]},
    "convective instability": {"hourly": ["k_index", "total_totals", "theta_e_diff", "lifted_index", "cape", "cin",
                                          "lcl_height"]},
    "column moisture": {"hourly": ["tcwv", "crh", "bretherton_rate", "vimfc", "rh700", "mfc850", "saharan"]},
    "tropical waves and cyclones": {"hourly": ["wave_trough", "wave_eta", "tc_rain", "tc_distance"]},
    "terrain and sea breeze": {"point": ["upslope_5km", "upslope_20km", "froude", "sea_breeze_pulse",
                                         "sea_breeze_since", "heat_1h", "heat_3h"]},
    "daily clock and season": {"point": ["cos_zenith", "lst_sin", "lst_cos", "insolation", "land_clock", "sea_clock",
                                         "doy_sin", "doy_cos", "cold_season"]},
    "climatology": {"point": ["clim_wet", "clim_rate"]},
}


def _any24(model, batch) -> np.ndarray:
    from ..products import _window_any, component

    with torch.no_grad():
        out = model(batch)
    comp = component(out, "", 0)
    k = comp["states"]
    return _window_any(comp["a"], comp["pi"], np.arange(k) > 0, 24)


def attribute(model, batch: dict, groups: dict | None = None) -> dict:
    """``{group: change in P(rain within 24 h) per target point (P,)}`` for the first forecast of ``batch``."""
    groups = groups or GROUPS
    model.eval()
    base = _any24(model, batch)
    out = {"baseline": base}
    for name, spec in groups.items():
        b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        for ch in spec.get("hourly", []):
            if ch in HOURLY_CHANNELS:
                b["hourly"][:, :, HOURLY_CHANNELS.index(ch)] = 0
        for ch in spec.get("point", []):
            if ch in TARGET_DYNAMIC:
                b["target_ctx"][..., TARGET_DYNAMIC.index(ch)] = 0
                if "station_ctx" in b:
                    b["station_ctx"][..., TARGET_DYNAMIC.index(ch)] = 0
        for ch in spec.get("mesh", []):
            b["mesh_ctx"][..., ch] = 0
        out[name] = base - _any24(model, b)
    return out


__all__ = ["GROUPS", "attribute"]
