# Copyright 2026 Nathan. Apache-2.0.
"""
NatureV1's own forecast as consensus members: the storm followed through the model's rollout, and steered by it.

The physics members start from a single analysis; their weakness is the flow they cannot forecast. A distilled
NatureV1 forecasts that flow. This module runs the full-state rollout once per initial time of a hindcast season
(:func:`naturev1.tracker.rollout_member_tracks`, one rollout for every storm active then) and turns the result
into two more members on the same footing as the physics ones -- the vortex tracked in the forecast pressure
field, and the storm carried by the forecast steering flow. :func:`hybrid_consensus` then blends them with the
physics members by along/cross-track weights fitted on past seasons: physics where it is strong (the first
hours, where persistence and the vorticity budget know more than any 1.5-degree field), the model's flow where
it is strong (days two to five).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def rollout_tracks(model, era5: str | Path, stepper, variables, cases, leads, precision: str = "bf16",
                   history: int | None = None, log=print) -> dict:
    """``{case key: {"rollout_vortex": (L, 2), "rollout_steering": (L, 2)}}`` lat/lon for every case."""
    from ..era5 import Normalizer
    from ..model import calendar_features
    from ..satdata import prepared_to_state
    from ..tracker import rollout_member_tracks

    era5 = Path(era5)
    meta = json.loads(era5.with_suffix(".json").read_text())
    normalizer = Normalizer.from_dict(meta["statistics"])
    times = np.asarray(meta["time"], np.int64)
    index = {int(t): i for i, t in enumerate(times)}
    values = np.load(era5, mmap_mode="r")
    history = history or model.config.history_frames
    device = next(model.parameters()).device
    by_time: dict[int, list] = {}
    for case in cases:
        by_time.setdefault(int(np.datetime64(case.time, "s").astype(np.int64)), []).append(case)
    out = {}
    for n, (t0, group) in enumerate(sorted(by_time.items()), start=1):
        i0 = index.get(t0)
        if i0 is None or i0 < history - 1:
            continue
        window = torch.stack([prepared_to_state(values[i0 - history + 1 + k], normalizer, model.config.analysis_channels)
                              for k in range(history)])[None].to(device)
        stamps = torch.tensor(times[i0 - history + 1:i0 + 1], dtype=torch.float64)[None]
        batch = {"analysis": window, "calendar": calendar_features(stamps).to(device), "valid_time": stamps[:, -1].to(device)}
        centres = torch.tensor([[c.lat, c.lon] for c in group], dtype=torch.float32, device=device)
        wind = torch.tensor([c.wind_kt if np.isfinite(c.wind_kt) else 50.0 for c in group], device=device)
        with torch.no_grad():
            members = rollout_member_tracks(model, batch, stepper, centres, leads, variables, precision, wind_kt=wind)
        for s, case in enumerate(group):
            entry = {}
            for name in ("rollout_vortex", "rollout_steering"):
                d = members[name][s].detach().cpu().numpy().astype(np.float64)        # (L, 2) degrees
                entry[name] = np.stack([case.lat + d[:, 0], ((case.lon + d[:, 1] + 180.0) % 360.0) - 180.0], -1)
            out[case.key] = entry
        if log and n % 100 == 0:
            log(f"[rollout-tracks] {n}/{len(by_time)} initial times")
    return out


def hybrid_consensus(cases, physics: dict, rollout: dict, weights: dict) -> np.ndarray:
    """``(N, L, 2)``: physics members and rollout members blended by ``weights`` (``fit_consensus`` output plus
    ``"members"``, the column order)."""
    from .consensus import apply_consensus

    nan = np.full((len(weights["along"]), 2), np.nan)
    columns = []
    for name in weights["members"]:
        source = rollout if name.startswith("rollout_") else physics
        columns.append(np.stack([source.get(c.key, {}).get(name, nan) for c in cases]))
    return apply_consensus(cases, np.stack(columns, 2), weights)


__all__ = ["hybrid_consensus", "rollout_tracks"]
