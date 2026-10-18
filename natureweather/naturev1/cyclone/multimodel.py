# Copyright 2026 Nathan. Apache-2.0.
"""
Several models at once -- the one thing in track forecasting that always helps.

NHC's best guidance is not any single model but their equal-weight average (TVCN, GUNA): the models' errors are
partly independent, so the average errs less than the best of them. Measured here on the held-out 2018 and
2020-2022 seasons, replacing HRES alone by the equal-weight mean of every archived model available for a forecast
(HRES, Pangu-Weather, GraphCast, FuXi, Aurora, FGN) cut the blended track error by 11% at 24 h and 19-23% at
48-120 h, with no weight fitted. Live, the models are GFS, ECMWF IFS, AIFS and GraphCast-GFS, read by
:mod:`naturev1.swarm.models`.

Two pieces live here:

* :func:`model_mean`: the equal-weight mean of model tracks in the storm-centred frame, lead by lead over whichever
  models still have the storm (a model that lost it drops out at that lead, nothing else changes). It is what the
  consensus is blended with (:func:`naturev1.cyclone.consensus.blend_with_model`), and the forecast-flow members of
  several models are averaged the same way.
* :func:`flow_dependent_cone`: the cone sized for *this* forecast rather than for the average one (Goerss's GPCE
  idea): a regression, fitted on the tuning seasons, of the consensus error on how far the physics members spread,
  how far the model mean sits from the consensus, how far the forecast-flow member sits from the model, the storm's
  speed, latitude and intensity. On the held-out seasons its radius ranks the actual errors with correlation
  0.17-0.38 (the fixed cone, by construction, 0) at the same 67% coverage.

Two things that did **not** help, measured the same way and left out: anchoring the model track to the observed
position (it makes the model's error correlated with the consensus's and loses the averaging), and shrinking the
model's weight when it disagrees with the forecast-flow member (when they disagree, the model is still the better
of the two).
"""

from __future__ import annotations

import numpy as np

from .geometry import from_local, to_local


def model_mean(case, tracks: dict | list | np.ndarray | None, leads) -> np.ndarray | None:
    """
    ``(L, 2)`` equal-weight mean of model tracks (``{name: (L, 2)}``, a list, or one ``(L, 2)`` array), in the
    storm-centred frame; NaN at a lead no model reaches. ``None`` when there is nothing.
    """
    if tracks is None:
        return None
    arrays = list(tracks.values()) if isinstance(tracks, dict) else ([tracks] if isinstance(tracks, np.ndarray) else list(tracks))
    arrays = [np.asarray(t, np.float64) for t in arrays if t is not None]
    if not arrays:
        return None
    if len(arrays) == 1:
        return arrays[0]
    stack = np.stack(arrays)                                               # (M, L, 2)
    east, north = to_local(case.lat, case.lon, stack[..., 0], stack[..., 1])
    ok = np.isfinite(east) & np.isfinite(north)
    count = ok.sum(0)
    with np.errstate(invalid="ignore"):
        e = np.where(ok, east, 0.0).sum(0) / np.where(count > 0, count, np.nan)
        n = np.where(ok, north, 0.0).sum(0) / np.where(count > 0, count, np.nan)
    lat, lon = from_local(case.lat, case.lon, e, n)
    out = np.stack([lat, lon], -1)
    out[count == 0] = np.nan
    return out


def model_spread_km(case, tracks: dict | list | None) -> np.ndarray | None:
    """``(L,)`` mean distance of the model tracks from their mean (NaN where fewer than two models remain)."""
    if tracks is None:
        return None
    arrays = [np.asarray(t, np.float64) for t in (tracks.values() if isinstance(tracks, dict) else tracks) if t is not None]
    if len(arrays) < 2:
        return None
    stack = np.stack(arrays)
    east, north = to_local(case.lat, case.lon, stack[..., 0], stack[..., 1])
    ok = np.isfinite(east) & np.isfinite(north)
    count = ok.sum(0)
    me = np.where(ok, east, 0.0).sum(0) / np.maximum(count, 1)
    mn = np.where(ok, north, 0.0).sum(0) / np.maximum(count, 1)
    d = np.where(ok, np.hypot(east - me, north - mn), 0.0).sum(0) / np.maximum(count, 1)
    return np.where(count >= 2, d, np.nan)


#: Predictors of :func:`flow_dependent_cone`, in order.
CONE_PREDICTORS = ("one", "member_spread_km", "model_consensus_km", "fflow_model_km", "speed_kmh", "abs_lat", "wind_kt")


def cone_predictors(case, consensus: np.ndarray, members: dict, member_names, model: np.ndarray | None,
                    fflow: np.ndarray | None, leads) -> np.ndarray:
    """``(L, 7)`` predictor rows for one forecast (missing pieces fall back to typical values via NaN -> 0 / medians)."""
    from .baselines import motion_back
    from .geometry import local_displacement_km

    L = len(leads)
    ce, cn = to_local(case.lat, case.lon, consensus[:, 0], consensus[:, 1])
    spread = np.full(L, np.nan)
    stack = [np.asarray(members[m], np.float64) for m in member_names if m in members]
    if stack:
        arr = np.stack(stack)                                               # (M, L, 2)
        e, n = to_local(case.lat, case.lon, arr[..., 0], arr[..., 1])
        d = np.hypot(e - ce[None], n - cn[None])
        spread = np.nanmean(d, axis=0)
    dmc = np.full(L, np.nan)
    if model is not None:
        me, mn = to_local(case.lat, case.lon, model[:, 0], model[:, 1])
        dmc = np.hypot(me - ce, mn - cn)
    dfm = np.full(L, np.nan)
    if model is not None and fflow is not None:
        fe, fn = to_local(case.lat, case.lon, fflow[:, 0], fflow[:, 1])
        dfm = np.hypot(fe - me, fn - mn)
    speed = np.nan
    rate = motion_back(case, 6)
    if rate is not None:
        e6, n6 = local_displacement_km(case.lat - rate[0] * 6, case.lon - rate[1] * 6, case.lat, case.lon)
        speed = float(np.hypot(e6, n6)) / 6.0
    rows = np.stack([np.ones(L), spread, dmc, dfm, np.full(L, speed), np.full(L, abs(case.lat)), np.full(L, case.wind_kt)], -1)
    return rows


def flow_dependent_cone(predictors: np.ndarray, table: dict, leads) -> np.ndarray:
    """``(L,)`` cone radius (km) from the fitted regression ``table`` (``coefficients.CONE_FLOW``)."""
    coef = np.asarray(table["coef"], np.float64)                            # (L_table, 7)
    scale = np.asarray(table["scale"], np.float64)
    medians = np.asarray(table.get("medians", np.zeros(coef.shape[1])), np.float64)
    x = np.where(np.isfinite(predictors), predictors, medians[None])
    hours = np.asarray(leads, np.float64)
    tl = np.asarray(table["leads"], np.float64)
    out = np.empty(len(hours))
    for k, h in enumerate(hours):
        j = int(np.clip(np.searchsorted(tl, h, side="right") - 1, 0, tl.size - 2))
        a = float(np.clip((h - tl[j]) / max(tl[j + 1] - tl[j], 1e-9), 0.0, 1.0))
        c = (1 - a) * coef[j] + a * coef[j + 1]
        s = (1 - a) * scale[j] + a * scale[j + 1]
        out[k] = max(float(x[k] @ c), 5.0) * s
    return out


__all__ = ["CONE_PREDICTORS", "cone_predictors", "flow_dependent_cone", "model_mean", "model_spread_km"]
