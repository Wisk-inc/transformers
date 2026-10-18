# Copyright 2026 Nathan. Apache-2.0.
"""
One call, the whole physics forecast: track consensus, its members and cone, the regime, the Dorian-type stall
warning, intensity and the environment that drives it -- from an analysis and the best track so far, with no
neural network and nothing fitted at run time.

:func:`physics_forecast` is what the rest of NatureV1 uses: the consensus track is a far stronger prior for the
learned track head than a frozen steering flow, and every diagnostic here is something a forecaster would want
next to the model's own answer.
"""

from __future__ import annotations

import numpy as np

from .data import LEADS


def physics_forecast(case, env, climatology=None, analogs=None, leads=LEADS, intensity: bool = True,
                     swarm: int = 0, swarm_fixes: list | None = None, swarm_device="cpu", forecast_flow=None,
                     model_track=None, model_extras=None, previous=None) -> dict:
    """
    Args:
        case: a :class:`naturev1.cyclone.data.Case` (position, intensity and the last 48 hours of best track; no
            ``truth`` needed).
        env: the analysis around the storm (:class:`naturev1.cyclone.fields.Environment`, with the 12-hour-earlier
            winds in ``previous`` if available).
        climatology: a :class:`naturev1.cyclone.climatology.Climatology` (members that need it are skipped
            without it, and the consensus renormalises over the rest).
        analogs: an :class:`naturev1.cyclone.analogs.AnalogDatabase`.
        previous: the previous cycle's result, ``{"time": its analysis time, "track": its (L, 2) track}``: its error
            for the time now analysed feeds back into the new track (:func:`naturev1.cyclone.consensus.apply_track_feedback`).
        swarm: if positive, also run the particle-filter sandbox (:mod:`naturev1.swarm`) with this many
            worlds, re-centred on the consensus track: scenario probabilities, a strike-probability map and a
            flow-dependent cone under ``result["swarm"]``.
        swarm_fixes: centre fixes (:mod:`naturev1.swarm.fixes`) to filter the swarm with instead of the 6-hourly
            history -- the satellite path, minute by minute.
        forecast_flow, model_track: dynamical models' forecast steering
            (:class:`naturev1.cyclone.forecastflow.ForecastFlow`, one or ``{name: flow}``) and their own tracks of the
            storm at ``leads`` (one ``(L, 2)`` or ``{name: track}``): they add the ``fflow`` and ``model`` members
            (equal-weight means over the models; the consensus fitted with them is blended with the models' mean
            track) and steer the swarm's worlds, each world following one model.
        model_extras: the models' ``extras`` from :func:`naturev1.swarm.models.fetch_models` (``{name: extras}``):
            their central-pressure deepening feeds the rapid-intensification probability and the model-trend
            intensity member.

    Returns:
        ``{"leads", "track" (L, 2), "members": {name: (L, 2)}, "cone_km" (L,), "regime", "stall", "ridge",
        "shear", "intensity_kt": {member: (L,)}, "intensity" (L,), "ri_probability" (P(+30 kt in 24 h)),
        "intensity_regime" ("rapid" when the rapid-intensification weights were used), "intensity_if_ri" (the path if
        the storm intensifies rapidly, to read with "ri_probability"), "peak_probability" (P(peak within the window >=
        64/96/113/137 kt)), "category_probability"
        (P(>= 64 / 96 / 113 / 137 kt) per lead), "cone_fixed_km", "model_spread_km" and "model_deepening_hpa" (with
        models: the climatological cone behind the flow-dependent one, the models' disagreement, their deepening)}``.
    """
    from .consensus import apply_consensus
    from .members import run_members
    from .regimes import classify
    from .ridge import ridge_diagnostics
    from .shear import shear_diagnostics
    from .stall import stall_diagnostics

    out = run_members(case, env, climatology, analogs, leads=leads, forecast_flow=forecast_flow,
                      model_track=model_track)
    ctx = out.pop("_context", None)
    ensemble = out.pop("_ensemble", None)
    table = consensus_table(out)
    nan = np.full((len(leads), 2), np.nan)
    stack = np.stack([out.get(m, nan) for m in table["members"]], 1)[None]
    physics_track = apply_consensus([case], stack, table)[0]
    track = physics_track
    if table.get("model_blend") is not None and out.get("model") is not None:
        from .consensus import blend_with_model

        n_models = (sum(1 for t in model_track.values() if t is not None and np.isfinite(np.asarray(t)[:, 0]).any())
                    if isinstance(model_track, dict) else 1)
        track = blend_with_model(case, physics_track, np.asarray(out["model"], np.float64), table["model_blend"], leads, n_models)
        result_models = n_models
    if previous is not None and isinstance(previous, dict) and previous.get("track") is not None:
        from .consensus import apply_track_feedback

        track = apply_track_feedback(case, track, previous["track"], previous.get("time"), leads)
    result = {"leads": list(leads), "track": track, "members": out, "consensus_table": table["name"],
              "n_models": locals().get("result_models", 0)}
    from . import coefficients

    # each consensus table has its own cone: the forecast-flow consensus errs less, so its cone is narrower
    cone_name = {"CONSENSUS_FF": "CONE_KM_FF", "CONSENSUS_BASIC_FF": "CONE_KM_BASIC_FF"}.get(table["name"], "CONE_KM")
    cone = getattr(coefficients, cone_name, None) or getattr(coefficients, "CONE_KM", None)
    if cone is not None:
        # The 67th percentile of the consensus's out-of-sample errors on past seasons -- NHC's cone construction --
        # widened where the physics ensemble disagrees more than usual for that lead.
        climatological = np.interp(np.asarray(leads, np.float64), cone["leads"], cone["radius_km"])
        radius = climatological
        if ensemble is not None:
            radius = np.maximum(radius, 1.2 * np.nan_to_num(ensemble["spread_km"], nan=0.0))
        result["cone_km"] = radius
        flow_table = getattr(coefficients, "CONE_FLOW", None)
        if flow_table is not None and out.get("model") is not None:
            # the cone sized for this forecast: member spread, model disagreement, speed, latitude, intensity
            # (fitted and checked without the ensemble-spread floor, so it stands on its own)
            from .multimodel import cone_predictors, flow_dependent_cone

            result["cone_fixed_km"] = climatological
            result["cone_predictors"] = cone_predictors(case, physics_track, out, table["members"], out.get("model"),
                                                        out.get("fflow"), leads)
            result["cone_km"] = flow_dependent_cone(result["cone_predictors"], flow_table, leads)
    if isinstance(model_track, dict):
        from .multimodel import model_spread_km

        result["model_spread_km"] = model_spread_km(case, model_track)
    if ctx is not None:
        stall = stall_diagnostics(ctx.flows["med"], np.where(np.isfinite(track), track, case.lat), leads, case.lat, case.lon)
        ridge = ridge_diagnostics(ctx.clean, case.lat, case.lon)
        result.update({"stall": {k: v for k, v in stall.items() if k != "cols"}, "cols": stall["cols"], "ridge": ridge,
                       "regime": classify(case, stall, ridge), "shear": shear_diagnostics(ctx.clean, case.lat, case.lon, ctx.motion),
                       "outer_strength_ms": ctx.outer_ms})
    if intensity:
        result.update(_intensity(case, env, ctx, track, leads, model_extras))
    if swarm and ctx is not None:
        from ..swarm.core import swarm_products

        # the swarm sized to this forecast's own cone where the flow-dependent cone could be made
        result["swarm"] = swarm_products(case, ctx, track, leads, particles=int(swarm), fixes=swarm_fixes,
                                         intensity_kt=result.get("intensity"), device=swarm_device,
                                         forecast_flow=forecast_flow,
                                         target_cone_km=result.get("cone_km") if "cone_predictors" in result else None)
    return result


def consensus_table(members: dict) -> dict:
    """
    The shipped consensus whose members are all present: the full table when a climatology and an analog
    database were given, else the basic one fitted for exactly the members that need neither -- renormalising
    the full table over half its members would lean on combinations it was never fitted to trust. With a dynamical
    model's forecast flow and track (the ``fflow`` and ``model`` members), the tables fitted with them come first.
    """
    from . import coefficients

    best, best_share = None, -1.0
    for name in ("CONSENSUS_FF", "CONSENSUS_BASIC_FF", "CONSENSUS", "CONSENSUS_BASIC"):
        table = getattr(coefficients, name, None)
        if table is None:
            continue
        have = [np.isfinite(np.asarray(members.get(m, [[np.nan]]))[0]).all() for m in table["members"]]
        if all(have):
            return {**table, "name": name}
        # none complete (say, no analysis 12 hours earlier): the table missing the smallest share of its members,
        # renormalised over the rest, in the same order of preference
        if np.mean(have) > best_share:
            best, best_share = name, float(np.mean(have))
    name = best or "CONSENSUS_BASIC"
    return {**getattr(coefficients, name), "name": name}


def _intensity(case, env, ctx, track, leads, model_extras=None) -> dict:
    from . import coefficients
    from .intensity import (category_probabilities, decay_along, environment_along_track, lgem_intensity,
                            model_deficit_change, model_trend_intensity, peak_probability, ri_probability, ri_scenario_intensity,
                            shifor_intensity, trend_intensity)

    members = {"persistence": np.full(len(leads), case.wind_kt)}
    shifor = shifor_intensity(case, leads)
    members["shifor"] = shifor
    if ctx is None or not np.isfinite(track[0, 0]):
        return {"intensity_kt": members, "intensity": shifor}
    filled = track.copy()
    good = np.isfinite(filled[:, 0])
    filled[~good] = filled[np.nonzero(good)[0][-1]]
    ocean = None
    if getattr(coefficients, "OCEAN", None):                 # the tables were fitted with the ocean climatology
        from .ocean import OceanClimatology

        ocean = OceanClimatology.default()
    try:
        along = environment_along_track(env, ctx.clean, case, filled, leads, np.nan_to_num(shifor, nan=case.wind_kt), ocean=ocean)
    except Exception:                                       # missing fields in a live analysis: degrade, don't fail
        return {"intensity_kt": members, "intensity": shifor}
    members["decay_shifor"] = decay_along(case, along, leads, np.nan_to_num(shifor, nan=case.wind_kt))
    members["lgem"] = lgem_intensity(case, along, leads)
    members["trend"] = trend_intensity(case, along, leads)
    model_dp = model_deficit_change(model_extras, leads)
    extra = {}
    if model_dp is not None and np.isfinite(model_dp).any():
        # the dynamical models' own deepening, as an intensity member and (below) as a rapid-intensification predictor
        members["mdl"] = model_trend_intensity(case, along, leads, model_dp)
        extra["model_deepening_hpa"] = model_dp
    try:
        from .coefficients import INTENSITY_CONSENSUS
        from .intensity import ships_intensity

        members["ships"] = decay_along(case, along, leads, ships_intensity(case, along, leads))
        table = INTENSITY_CONSENSUS
        regime = table.get("ri_regime")
        if regime is not None:
            # A storm primed for rapid intensification gets the weights fitted on such storms: they lean on LGEM and
            # the trend, and correct the consensus's habit of pulling an intensifying storm back toward the mean.
            p = ri_probability(case, along, model_dp=model_dp)
            extra.update({"ri_probability": p, "intensity_regime": "rapid" if p >= regime["cut"] else "normal"})
            if p >= regime["cut"]:
                table = {**table, "weights": regime["weights"], "bias": regime["bias"]}
        weights = np.asarray(table["weights"], np.float64)                      # (L, members)
        nan = np.full(len(leads), np.nan)
        stack = np.stack([members.get(n, nan) for n in table["members"]], -1)   # a member without its model: dropped
        blended = (np.nan_to_num(stack) * weights).sum(-1) / np.maximum((np.isfinite(stack) * weights).sum(-1), 1e-9)
        blended = blended + np.asarray(table.get("bias", np.zeros(len(leads))), np.float64)
        # a basin's own residual bias (wind conventions and storm sizes differ between basins), when the table has it
        blended = blended + np.asarray(INTENSITY_CONSENSUS.get("basin_bias", {}).get(case.basin, np.zeros(len(leads))), np.float64)
    except ImportError:
        blended = np.nanmean(np.stack([members["decay_shifor"], members["lgem"]]), 0)
    intensity = np.clip(blended, 15.0, 200.0)
    # the branch beside the expected value: the storm's path if it intensifies rapidly, with its probability
    if_ri = ri_scenario_intensity(case, along, leads)
    if np.isfinite(if_ri).any():
        extra["intensity_if_ri"] = np.clip(if_ri, 15.0, 200.0)
    categories = category_probabilities(intensity, extra.get("intensity_regime", "normal"), ri_probability=extra.get("ri_probability"),
                                        ri_intensity=extra.get("intensity_if_ri"))
    try:
        peak = peak_probability(case, along, intensity, extra.get("ri_probability"), model_dp)
    except Exception:
        peak = {}
    if peak:
        extra["peak_probability"] = peak
    if categories:
        extra["category_probability"] = categories
    return {"intensity_kt": members, "intensity": intensity, "environment": along, **extra}


__all__ = ["consensus_table", "physics_forecast"]
