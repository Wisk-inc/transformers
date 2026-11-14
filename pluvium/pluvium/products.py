# Copyright 2026 Nathan. Apache-2.0.
"""
From the heads' raw outputs to what a person asks: will it rain here, when will it start, how hard, will there be
thunder, how long will it last, how much will fall, and why.

Everything comes from one chain per place, so the answers agree with each other: the hourly chance of each intensity
(dry, light, moderate, heavy), the distribution of the hour rain *starts* (from a dry start), turns *heavy*, and
*stops* (from a wet start), the chance of any rain and of heavy rain in the next 6/12/24/48/72 hours, and the totals
and the wettest 1, 3 and 6 hours (sampled paths of the chain with amounts drawn from each state's distribution).
Thunder and drizzle/rain/showers come from their own heads.

Live, Pluvium runs once per driving model (GFS, ECMWF IFS, ECMWF AIFS) and the forecast is the **mixture of the
chains** (experimental): hourly chances are weighted means, start-time distributions are mixtures weighted by each
component's chance of being dry now, and totals pool every component's sampled paths. Disagreement between the
driving models therefore widens the start window and the totals honestly, instead of being averaged away.
"""

from __future__ import annotations

import numpy as np
import torch

from .chain import class_probs, first_entry, marginals_np, quantile_hour, sample_paths, state_means, torrential_given_heavy
from .features import REGIMES

WINDOWS = (6, 12, 24, 48, 72)
TOTAL_WINDOWS = (24, 48, 72)
TOTAL_THRESHOLDS = (10.0, 25.0, 50.0, 100.0)
BURSTS = (1, 3, 6)            # wettest 1, 3 and 6 hours within the first 24 (flash-flood drivers)


def _np(x) -> np.ndarray:
    return x.detach().float().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, np.float32)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def _sig(x):
    return 1 / (1 + np.exp(-x))


def component(out: dict, prefix: str = "", j: int = 0) -> dict:
    """One forecast's raw outputs as NumPy arrays with points first: ``a (P, T, K, K)`` transition probabilities,
    ``pi (P, K)``, amount parameters ``(P, T, ...)``, thunder ``(P, T)``, kind ``(P, T, 3)``, cause ``(Ff, P, R)``."""
    g = lambda k: _np(out[prefix + k])[j]                                  # noqa: E731
    trans = g("trans")                                                     # (T, P, K, K)
    comp = {"a": _softmax(np.moveaxis(trans, 0, 1)), "pi": _softmax(g("init"))}
    if prefix + "mix_logit" in out:
        comp.update(logit=np.moveaxis(g("mix_logit"), 0, 1), mu=np.moveaxis(g("mix_mu"), 0, 1),
                    logsig=np.moveaxis(g("mix_logsig"), 0, 1))
    else:
        comp.update(amt_mu=np.moveaxis(g("amt_mu"), 0, 1), amt_sig=np.moveaxis(g("amt_sig"), 0, 1))
    if prefix + "thunder" in out:
        comp["thunder"] = _sig(g("thunder").T)
    if prefix + "kind" in out:
        comp["kind"] = _softmax(np.moveaxis(g("kind"), 0, 1))
    comp["cause"] = _softmax(g("cause"))
    comp["states"] = comp["a"].shape[-1]
    return comp


def _window_any(a: np.ndarray, pi: np.ndarray, target: np.ndarray, hours: int) -> np.ndarray:
    """P(the state enters ``target`` at least once in hours 0..hours-1)."""
    start_in = (pi * target).sum(-1)
    later = first_entry(a[:, :hours], pi, target).sum(-1)
    return start_in + later


def summarize(out, t0: int, prefix: str = "", batch_index: int = 0, samples: int = 200, calibrator=None,
              weights=None, corrector=None, places=None, place_lon=None, keep_samples: bool = False) -> dict:
    """
    Per-point products for one forecast (arrays over points ``P`` and hours ``T``).

    ``out`` is the model's output dict, or a list of them (one per driving model) mixed with ``weights`` (equal by
    default). ``calibrator`` (isotonic, fitted on held-out years) and ``corrector`` (:mod:`pluvium.online`, learned
    from recent errors, keyed by ``places``) adjust the chances of rain.
    """
    outs = out if isinstance(out, (list, tuple)) else [out]
    w = np.ones(len(outs)) if weights is None else np.asarray(weights, np.float64)
    w = w / w.sum()
    comps = [component(o, prefix, batch_index) for o in outs]
    k = comps[0]["states"]
    p_state_k = [marginals_np(c["a"], c["pi"]) for c in comps]
    classes = sum(wk * class_probs(c, k, ps) for wk, c, ps in zip(w, comps, p_state_k))       # (P, T, 4)
    means = [state_means(c, k) for c in comps]
    expected = sum(wk * (ps * mk).sum(-1) for wk, ps, mk in zip(w, p_state_k, means))
    p_wet = 1.0 - classes[..., 0]
    torrential = sum(wk * ps[..., -1] * torrential_given_heavy(c, k) for wk, c, ps in zip(w, comps, p_state_k))
    t_hours = p_wet.shape[-1]
    dry = np.zeros(k, bool)
    dry[0] = True
    wet_target = ~dry
    heavy_target = np.zeros(k, bool)
    heavy_target[-1] = True
    # start (from dry now), heavy start, stop (from wet now)
    p0 = [c["pi"] for c in comps]
    dry_w = [wk * pk[:, 0] for wk, pk in zip(w, p0)]
    onset = sum(dw[:, None] * first_entry(c["a"], np.eye(k)[0][None].repeat(len(dw), 0), wet_target)
                for dw, c in zip(dry_w, comps)) / np.maximum(sum(dry_w), 1e-12)[:, None]
    heavy_now = [pk[:, -1] for pk in p0]
    nh_w = [wk * (1 - hn) for wk, hn in zip(w, heavy_now)]
    heavy_onset = sum(nw[:, None] * first_entry(c["a"], c["pi"] * (~heavy_target) / np.maximum(1 - hn, 1e-12)[:, None],
                                                heavy_target)
                      for nw, c, hn in zip(nh_w, comps, heavy_now)) / np.maximum(sum(nh_w), 1e-12)[:, None]
    wet_w = [wk * (1 - pk[:, 0]) for wk, pk in zip(w, p0)]
    stop = sum(ww[:, None] * first_entry(c["a"], c["pi"] * wet_target / np.maximum(1 - pk[:, :1], 1e-12), dry)
               for ww, c, pk in zip(wet_w, comps, p0)) / np.maximum(sum(wet_w), 1e-12)[:, None]
    any_rain, any_heavy = {}, {}
    for h in WINDOWS:
        if h > t_hours:
            continue
        any_rain[h] = sum(wk * _window_any(c["a"], c["pi"], wet_target, h) for wk, c in zip(w, comps))
        if k > 2:
            any_heavy[h] = sum(wk * _window_any(c["a"], c["pi"], heavy_target, h) for wk, c in zip(w, comps))
    if calibrator is not None:
        new = calibrator.apply_hourly(p_wet)
        scale = new / np.maximum(p_wet, 1e-9)
        classes = np.concatenate([1 - new[..., None], classes[..., 1:] * scale[..., None]], -1)
        expected = expected * scale
        p_wet = new
        any_rain = {h: calibrator.apply_window(v, h) for h, v in any_rain.items()}
    if corrector is not None and places is not None:
        new = corrector.apply_hourly(p_wet, places, place_lon, t0)
        scale = new / np.maximum(p_wet, 1e-9)
        classes = np.concatenate([1 - new[..., None], classes[..., 1:] * scale[..., None]], -1)
        expected = expected * scale * corrector.amount_factor(places)[:, None]
        p_wet = new
        any_rain = {h: corrector.apply_window(v, places, h) for h, v in any_rain.items()}
    thunder = sum(wk * c["thunder"] for wk, c in zip(w, comps)) if "thunder" in comps[0] else None
    kind = None
    if "kind" in comps[0]:
        wet_k = [wk * (1 - ps[..., 0]) for wk, ps in zip(w, p_state_k)]
        kind = sum(wk[..., None] * c["kind"] for wk, c in zip(wet_k, comps)) / np.maximum(sum(wet_k), 1e-12)[..., None]
    counts = np.maximum(1, np.round(w * samples)).astype(int)
    paths = np.concatenate([sample_paths(c, k, int(n), seed=i) for i, (c, n) in enumerate(zip(comps, counts))], 0)
    totals = {h: paths[:, :, :h].sum(-1) for h in TOTAL_WINDOWS if h <= t_hours}
    bursts = {}
    first = paths[:, :, : min(24, t_hours)]
    for b in BURSTS:
        if b <= first.shape[-1]:
            csum = np.cumsum(np.pad(first, ((0, 0), (0, 0), (1, 0))), -1)
            bursts[b] = (csum[..., b:] - csum[..., :-b]).max(-1)                # (S, P)
    cause = sum(wk * c["cause"] for wk, c in zip(w, comps))                  # (Ff, P, R)
    extra = {"samples24": totals[24]} if keep_samples and 24 in totals else {}
    return {**extra,
        "t0": int(t0), "states": k, "p_wet": p_wet.astype(np.float32), "classes": classes.astype(np.float32),
        "expected_mm": expected.astype(np.float32), "torrential": torrential.astype(np.float32),
        "p_now": (1 - sum(wk * pk[:, 0] for wk, pk in zip(w, p0))).astype(np.float32),
        "any_rain": {h: v.astype(np.float32) for h, v in any_rain.items()},
        "any_heavy": {h: v.astype(np.float32) for h, v in any_heavy.items()},
        "start_median": quantile_hour(onset, 0.5), "start_early": quantile_hour(onset, 0.1),
        "start_late": quantile_hour(onset, 0.9), "start_within": onset.sum(-1).astype(np.float32),
        # if rain starts in the next 24 hours: when (the "today" answer)
        "start_median_24": quantile_hour(onset[:, :24], 0.5), "start_early_24": quantile_hour(onset[:, :24], 0.1),
        "start_late_24": quantile_hour(onset[:, :24], 0.9), "start_within_24": onset[:, :24].sum(-1).astype(np.float32),
        "heavy_start_median": quantile_hour(heavy_onset, 0.5), "heavy_within": heavy_onset.sum(-1).astype(np.float32),
        "stop_median": quantile_hour(stop, 0.5),
        "thunder": None if thunder is None else thunder.astype(np.float32),
        "kind": None if kind is None else kind.astype(np.float32),
        "totals": {h: _distribution(v) for h, v in totals.items()},
        "bursts": {b: _distribution(v) for b, v in bursts.items()},
        "cause": cause,
    }


#: thresholds whose exceedance chances are kept for every total and burst (the flood rule reads 40, 60 and 75)
EXCEEDANCE = tuple(sorted(set(TOTAL_THRESHOLDS) | {40.0, 60.0, 75.0}))


def _distribution(v: np.ndarray) -> dict:
    return {"median": np.median(v, 0), "p10": np.quantile(v, 0.1, 0), "p90": np.quantile(v, 0.9, 0), "mean": v.mean(0),
            **{f"p_ge_{int(x)}": (v >= x).mean(0) for x in EXCEEDANCE}}


def dominant_cause(summary: dict, frames_for_hours: np.ndarray, point: int, window: int = 24) -> str:
    """The rain cause carrying the most expected rain in the first ``window`` hours at a point."""
    exp = summary["expected_mm"][point, :window]
    probs = summary["cause"][frames_for_hours[:window], point]
    weight = (probs * exp[:, None]).sum(0)
    if weight.sum() <= 1e-6:
        return "-"
    return REGIMES[int(weight.argmax())]


__all__ = ["BURSTS", "TOTAL_THRESHOLDS", "TOTAL_WINDOWS", "WINDOWS", "component", "dominant_cause", "summarize"]
