# Copyright 2026 Nathan. Apache-2.0.
"""
Verification the way forecast centres do it, with error bars the way statisticians would insist on.

The idea: "beats extrapolation" means nothing without a homogeneous sample (every model verified on exactly
the same cases at each lead) and a confidence interval that respects how correlated the cases are. Twelve
forecasts of one storm 12 hours apart are not twelve independent trials -- a model that happens to get one
long-lived storm right looks great on a naive standard error. So uncertainty here comes from a **block
bootstrap over storms**: storms are resampled with replacement and the skill recomputed, and a model only
"beats" a baseline at a lead when the whole 90% interval of its improvement is above zero.

Track errors are great-circle distances; the along/cross-track split (NHC convention) separates speed errors
from direction errors, which is how the physics members are diagnosed.
"""

from __future__ import annotations

import numpy as np

from .geometry import along_cross_track, great_circle_km


def track_errors(cases, forecast: np.ndarray) -> np.ndarray:
    """``(N, L)`` km between a forecast ``(N, L, 2)`` (lat, lon) and the verifying best-track positions."""
    truth_lat = np.stack([c.truth["lat"] for c in cases])
    truth_lon = np.stack([c.truth["lon"] for c in cases])
    valid = np.stack([c.truth["valid"] for c in cases])
    err = great_circle_km(truth_lat, truth_lon, forecast[..., 0], forecast[..., 1])
    return np.where(valid, err, np.nan)


def along_cross(cases, forecast: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat0 = np.array([c.lat for c in cases])[:, None]
    lon0 = np.array([c.lon for c in cases])[:, None]
    truth_lat = np.stack([c.truth["lat"] for c in cases])
    truth_lon = np.stack([c.truth["lon"] for c in cases])
    return along_cross_track(lat0, lon0, truth_lat, truth_lon, forecast[..., 0], forecast[..., 1])


def intensity_errors(cases, forecast_kt: np.ndarray) -> np.ndarray:
    """``(N, L)`` signed wind errors (forecast - truth), kt."""
    truth = np.stack([c.truth["wind"] for c in cases])
    valid = np.stack([c.truth["valid"] for c in cases]) & np.isfinite(truth)
    return np.where(valid, forecast_kt - truth, np.nan)


def homogeneous(errors: dict[str, np.ndarray]) -> np.ndarray:
    """``(N, L)`` mask where every member has a finite error."""
    mask = None
    for value in errors.values():
        ok = np.isfinite(value)
        mask = ok if mask is None else mask & ok
    return mask


def block_bootstrap(groups: np.ndarray, statistic, samples: int = 1000, seed: int = 0) -> np.ndarray:
    """
    ``statistic(index)`` over ``samples`` resamplings of whole groups (storms) with replacement. Returns the
    sampled statistics. ``groups``: ``(N,)`` group label per case.
    """
    rng = np.random.default_rng(seed)
    labels, inverse = np.unique(groups, return_inverse=True)
    members = [np.nonzero(inverse == g)[0] for g in range(len(labels))]
    out = []
    for _ in range(samples):
        chosen = rng.integers(0, len(labels), len(labels))
        index = np.concatenate([members[g] for g in chosen]) if len(chosen) else np.zeros(0, int)
        out.append(statistic(index))
    return np.asarray(out)


def compare(cases, errors: dict[str, np.ndarray], baseline: str, leads, samples: int = 1000,
            alpha: float = 0.10) -> dict:
    """
    Homogeneous comparison of every member against ``baseline``: mean error per lead, sample size, skill
    (% improvement) and its block-bootstrap ``1 - alpha`` interval, and whether the member beats the baseline
    (whole interval above zero).
    """
    mask = homogeneous(errors)
    groups = np.array([c.storm_id for c in cases])
    result = {"leads": list(leads), "count": mask.sum(0).tolist(), "storms": [], "members": {}}
    for k in range(len(leads)):
        result["storms"].append(int(len(set(groups[mask[:, k]]))))
    for name, err in errors.items():
        means, skills, lows, highs, beats = [], [], [], [], []
        for k in range(len(leads)):
            ok = mask[:, k]
            e, b = err[ok, k], errors[baseline][ok, k]
            g = groups[ok]
            if e.size == 0:
                means.append(np.nan); skills.append(np.nan); lows.append(np.nan); highs.append(np.nan); beats.append(False)
                continue
            means.append(float(e.mean()))
            skill = lambda idx: 100.0 * (1.0 - e[idx].mean() / max(b[idx].mean(), 1e-9)) if idx.size else np.nan
            skills.append(float(skill(np.arange(e.size))))
            if name == baseline:
                lows.append(0.0); highs.append(0.0); beats.append(False)
                continue
            boot = block_bootstrap(g, skill, samples=samples, seed=k)
            lo, hi = np.nanpercentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            lows.append(float(lo)); highs.append(float(hi)); beats.append(bool(lo > 0))
        result["members"][name] = {"mean_km": means, "skill": skills, "low": lows, "high": highs, "beats": beats}
    return result


def format_comparison(result: dict, title: str = "track error (km), homogeneous", unit: str = "km") -> str:
    leads = result["leads"]
    lines = [title, "  lead      " + "".join(f"{h:>8}h" for h in leads),
             "  cases     " + "".join(f"{n:>9}" for n in result["count"]),
             "  storms    " + "".join(f"{n:>9}" for n in result["storms"])]
    for name, row in result["members"].items():
        lines.append(f"  {name[:10]:<10}" + "".join(f"{v:>9.0f}" if np.isfinite(v) else "      nan" for v in row["mean_km"]))
        lines.append("   skill %  " + "".join(
            (f"{s:>+8.1f}{'*' if b else ' '}" if np.isfinite(s) else "      nan") for s, b in zip(row["skill"], row["beats"])))
    lines.append(f"  * the whole 90% storm-bootstrap interval of the improvement is above zero ({unit})")
    return "\n".join(lines)


__all__ = ["along_cross", "block_bootstrap", "compare", "format_comparison", "homogeneous", "intensity_errors",
           "track_errors"]
