# Copyright 2026 Nathan. Apache-2.0.
"""
The member suite: every physics and statistical track forecast for one storm, computed from one analysis.

Each member is a different answer to "where will it go", built from the modules of this package. They are
deliberately diverse -- persistence, climatology, analogs, steering trajectories through frozen, evolving and
climatology-relaxing flows, a barotropic model, an ensemble mean -- because a consensus gains most from members
whose errors differ (:mod:`naturev1.cyclone.consensus`). All of them start from the analysis at the initial
time and the best track up to it; nothing reads the future.

Shared ingredients are computed once in :func:`build_context`: the vortex removed from the analysis
(:mod:`.vortex`), the vortex's outer strength, the climatological environment for the date (:mod:`.climatology`),
the steering flows of three layers in both, beta drift (:mod:`.beta`), the observed motion and the vorticity-
budget motion (:mod:`.vorticity_advection`), and the 12-hour-earlier steering for the flow tendency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .baselines import xtrp_latlon
from .beta import BetaDrift, outer_strength
from .data import HISTORY_HOURS, LEADS, PREVIOUS_LEVELS
from .fields import Environment
from .geometry import from_local, to_local, velocity_from_positions
from .steering import SteeringFlow
from .trajectory import DecayingResidual, EvolvingFlow, combined, frozen, integrate
from .vortex import remove_vortex

#: The members the shipped consensus weighs (order matters: it is the column order of the weights).
DEFAULT_MEMBERS = ("xtrp6", "cliper", "analog", "budget_mix", "evo12", "med_anom48", "int_anom48", "budget_anom96",
                   "evo_anom48", "evo_anom96", "ens48_mean", "baro_res12", "baro")


def observed_motion(case) -> tuple[float, float] | None:
    """(east, north) m/s over the last 6 hours (else 12)."""
    for h in (6, 12):
        k = HISTORY_HOURS.index(-h)
        la, lo = case.history["lat"][k], case.history["lon"][k]
        if np.isfinite(la):
            return velocity_from_positions(la, lo, case.lat, case.lon, h)
    return None


def previous_environment(env: Environment) -> Environment | None:
    """The analysis 12 hours earlier, as an :class:`Environment` of its stored winds."""
    if "u" not in env.previous:
        return None
    return Environment(lat=env.lat, lon=env.lon, levels=PREVIOUS_LEVELS,
                       fields={"u": env.previous["u"], "v": env.previous["v"]})


@dataclass
class PhysicsContext:
    case: object
    env: Environment
    clean: Environment
    outer_ms: float
    motion: tuple[float, float]
    motion_mix: tuple[float, float]
    beta: BetaDrift
    flows: dict
    climate: Environment | None = None
    climate_flows: dict = field(default_factory=dict)
    before: SteeringFlow | None = None
    split: object = None

    def start_velocity(self, name: str) -> tuple[float, float]:
        u, v = self.flows[name].velocity(self.case.lat, self.case.lon)
        return float(u[0]), float(v[0])

    def beta0(self) -> tuple[float, float]:
        u, v = self.beta.velocity(self.case.lat, self.case.lon)
        return float(u[0]), float(v[0])

    def residual(self, name: str, motion=None, tau_h: float = 24.0) -> DecayingResidual:
        m = motion or self.motion
        u0, v0 = self.start_velocity(name)
        b0 = self.beta0()
        return DecayingResidual(m[0] - u0 - b0[0], m[1] - v0 - b0[1], tau_h)

    def beta_velocity(self):
        return lambda lat, lon, hours: self.beta.velocity(lat, lon)


def build_context(case, env: Environment, climatology=None) -> PhysicsContext | None:
    """Everything the members share; None when the storm has no motion history."""
    from .climatology import DampedAnomalyFlow  # noqa: F401  (imported for its side-effect-free availability)
    from .steering import intensity_layer
    from .vorticity_advection import budget_motion

    motion = observed_motion(case)
    if motion is None:
        return None
    split = remove_vortex(env, case.lat, case.lon)
    clean = split.environment
    ows = outer_strength(split)
    top, bottom = intensity_layer(case.wind_kt)
    layers = {"med": (400, 850), "dlm": (200, 850), "shal": (700, 850), "int": (top, bottom)}
    flows = {k: SteeringFlow(clean, *v) for k, v in layers.items()}
    climate, cflows = None, {}
    if climatology is not None:
        climate = climatology.environment(clean, case.time, shift_days=2)
        cflows = {k: SteeringFlow(climate, max(v[0], 200), min(v[1], 850)) for k, v in layers.items()}
    budget = budget_motion(env, case.lat, case.lon)
    mix = motion if budget is None else (0.5 * (budget["u"] + motion[0]), 0.5 * (budget["v"] + motion[1]))
    before = None
    k = HISTORY_HOURS.index(-12)
    la12, lo12 = case.history["lat"][k], case.history["lon"][k]
    prev = previous_environment(env)
    if prev is not None and np.isfinite(la12):
        before = SteeringFlow(remove_vortex(prev, la12, lo12).environment, 400, 850)
    return PhysicsContext(case, env, clean, ows, motion, mix, BetaDrift(clean, ows), flows, climate, cflows, before, split)


class _EvolvingDamped:
    """Evolving flow (12-hour tendency) whose departure from climatology then decays."""

    def __init__(self, now, before, climate, tau_evo: float, tau_anom: float) -> None:
        self.evo = EvolvingFlow(frozen(now), frozen(before), tau_evo)
        self.climate, self.tau = climate, tau_anom

    def __call__(self, lat, lon, hours):
        u, v = self.evo(lat, lon, hours)
        uc, vc = self.climate.velocity(lat, lon)
        m = np.exp(-np.asarray(hours, np.float64) / self.tau)
        return uc + m * (u - uc), vc + m * (v - vc)


def _anomaly_member(ctx: PhysicsContext, layer: str, tau: float, motion=None, leads=LEADS):
    from .climatology import DampedAnomalyFlow

    if layer not in ctx.climate_flows:
        return None
    flow = DampedAnomalyFlow(ctx.flows[layer], ctx.climate_flows[layer], tau)
    return integrate(ctx.case.lat, ctx.case.lon, combined(flow, ctx.beta_velocity(), ctx.residual(layer, motion)), leads)


def run_members(case, env: Environment, climatology=None, analogs=None, cliper_table: dict | None = None,
                names=DEFAULT_MEMBERS, leads=LEADS, ensemble_members: int = 24) -> dict:
    """
    ``{member: (L, 2) lat/lon}`` for every requested member that can be computed, plus ``"_context"`` (the shared
    :class:`PhysicsContext`) and ``"_ensemble"`` (members, spread) when the ensemble ran.
    """
    from .cliper import cliper_track

    out: dict = {}
    wanted = set(names)
    if "xtrp6" in wanted:
        out["xtrp6"] = xtrp_latlon(case, leads, 6)
    if "xtrp12" in wanted:
        out["xtrp12"] = xtrp_latlon(case, leads, 12)
    if "cliper" in wanted:
        out["cliper"] = cliper_track(case, leads, cliper_table)
    if "analog" in wanted and analogs is not None:
        out["analog"] = analogs.forecast(case, 100, 1.5)
    ctx = build_context(case, env, climatology)
    if ctx is None:
        return out
    out["_context"] = ctx
    lat0, lon0 = case.lat, case.lon
    beta = ctx.beta_velocity()
    if "steer_med" in wanted:
        out["steer_med"] = integrate(lat0, lon0, combined(frozen(ctx.flows["med"]), beta, ctx.residual("med")), leads)
    if "budget_mix" in wanted:
        out["budget_mix"] = integrate(lat0, lon0, combined(frozen(ctx.flows["med"]), beta, ctx.residual("med", ctx.motion_mix)), leads)
    if "evo12" in wanted and ctx.before is not None:
        out["evo12"] = integrate(lat0, lon0, combined(EvolvingFlow(frozen(ctx.flows["med"]), frozen(ctx.before), 12.0), beta,
                                                      ctx.residual("med")), leads)
    for name, (layer, tau, mot) in {"med_anom48": ("med", 48.0, None), "med_anom96": ("med", 96.0, None),
                                    "int_anom48": ("int", 48.0, None), "budget_anom48": ("med", 48.0, "mix"),
                                    "budget_anom96": ("med", 96.0, "mix")}.items():
        if name in wanted:
            track = _anomaly_member(ctx, layer, tau, ctx.motion_mix if mot == "mix" else None, leads)
            if track is not None:
                out[name] = track
    for name, tau in (("evo_anom48", 48.0), ("evo_anom96", 96.0)):
        if name in wanted and ctx.before is not None and "med" in ctx.climate_flows:
            flow = _EvolvingDamped(ctx.flows["med"], ctx.before, ctx.climate_flows["med"], 12.0, tau)
            out[name] = integrate(lat0, lon0, combined(flow, beta, ctx.residual("med", ctx.motion_mix)), leads)
    if {"ens48_mean", "ens96_mean"} & wanted:
        from .ensemble import physics_ensemble

        flows = {k: ctx.flows[k] for k in ("med", "dlm", "shal")}
        cflows = {k: ctx.climate_flows[k] for k in ("med", "dlm", "shal")} if ctx.climate_flows else None
        tau = 48.0 if "ens48_mean" in wanted else 96.0
        ens = physics_ensemble(case, ctx.clean, flows, ctx.beta, ctx.motion_mix, leads, climate=ctx.climate,
                               climate_flows=cflows, anomaly_tau_h=tau, members=ensemble_members)
        out["ens48_mean" if tau == 48.0 else "ens96_mean"] = ens["mean"]
        out["_ensemble"] = ens
    if {"baro_res24", "baro_res12", "baro"} & wanted:
        from .barotropic import BarotropicModel

        try:
            baro = BarotropicModel(ctx.clean, lat0, lon0, ctx.outer_ms, case.r34_nmi).run(leads)["track"]
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            baro = None
        if baro is not None:
            out["baro"] = baro
            first = list(leads).index(min(leads))
            x0, y0 = to_local(lat0, lon0, baro[first, 0], baro[first, 1])
            hours = float(leads[first])
            ru = ctx.motion_mix[0] - x0 / hours / 3.6
            rv = ctx.motion_mix[1] - y0 / hours / 3.6
            for name, tau in (("baro_res12", 12.0), ("baro_res24", 24.0)):
                if name in wanted:
                    shift = tau * 3.6 * (1 - np.exp(-np.asarray(leads, float) / tau))
                    bx, by = to_local(lat0, lon0, baro[:, 0], baro[:, 1])
                    la, lo = from_local(lat0, lon0, bx + ru * shift, by + rv * shift)
                    out[name] = np.stack([la, lo], -1)
    return out


__all__ = ["DEFAULT_MEMBERS", "PhysicsContext", "build_context", "observed_motion", "previous_environment", "run_members"]
