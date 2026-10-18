# Copyright 2026 Nathan. Apache-2.0.
"""
Steering that evolves the way a dynamical model forecasts it -- for the turns.

A storm turns when its steering changes: a trough digs in, a ridge erodes, a weakness opens. The analysis plus its
12-hour tendency (:class:`naturev1.cyclone.trajectory.EvolvingFlow`) cannot know about a trough that is still
2,000 km away and two days off. A dynamical model's forecast can. This is the idea of NHC's Beta and Advection Model:
carry the storm through the model's *forecast* layer-mean winds, plus beta drift.

:class:`ForecastFlow` holds those winds at forecast hours (every 12 h to 120 h) for three layers -- 850-700,
850-400 and 850-200 hPa -- with the model's own storm removed at each hour (Biot-Savart removal around where the
model has it, :func:`naturev1.cyclone.vortex.remove_vortex`), so a trajectory is steered by the environment, not by
the model's copy of the hurricane. Between forecast hours it interpolates linearly.

:func:`fflow_member` is the trajectory member; :mod:`naturev1.swarm` steers its worlds with the same flow. In
hindcasts the flow comes from the HRES forecasts of WeatherBench 2; live, from GFS
(:meth:`naturev1.swarm.analysis.GFSAnalysis.forecast_flow`). No neural network is involved: the model is the ECMWF or
NOAA physics model.
"""

from __future__ import annotations

import numpy as np

#: Layers stored, in channel order (u, v each): shallow, medium, deep.
LAYERS = (("shal", 700, 850), ("med", 400, 850), ("dlm", 200, 850))


class ForecastFlow:
    """Layer-mean steering at forecast hours: ``data`` ``(T, 6, H, W)`` (shal u, v, med u, v, dlm u, v) on the
    ``lat`` x ``lon`` box, at ``hours`` ``(T,)``."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray, hours, data: np.ndarray, source: str = "") -> None:
        self.lat = np.asarray(lat, np.float64)
        self.lon = np.asarray(lon, np.float64)
        self.hours = np.asarray(hours, np.float64)
        self.data = np.asarray(data, np.float32)
        self.source = source

    def _bracket(self, hours: float):
        h = float(np.clip(hours, self.hours[0], self.hours[-1]))
        k = int(np.clip(np.searchsorted(self.hours, h, side="right") - 1, 0, self.hours.size - 2))
        a = (h - self.hours[k]) / max(self.hours[k + 1] - self.hours[k], 1e-9)
        return k, a

    def field(self, hours: float, layer: str = "med") -> tuple[np.ndarray, np.ndarray]:
        """``(u, v)`` grids of ``layer`` at ``hours``."""
        j = [name for name, _, _ in LAYERS].index(layer)
        k, a = self._bracket(hours)
        u = (1 - a) * self.data[k, 2 * j] + a * self.data[k + 1, 2 * j]
        v = (1 - a) * self.data[k, 2 * j + 1] + a * self.data[k + 1, 2 * j + 1]
        return u.astype(np.float64), v.astype(np.float64)

    def velocity(self, env, lat, lon, hours: float, depth: float = 0.5):
        """Steering at points for a steering depth (0 shallow .. 0.5 medium .. 1 deep), interpolated in time."""
        w_sh, w_dp = max(1.0 - 2.0 * depth, 0.0), max(2.0 * depth - 1.0, 0.0)
        w_md = 1.0 - w_sh - w_dp
        u = np.zeros(np.shape(lat))
        v = np.zeros(np.shape(lat))
        for name, w in (("shal", w_sh), ("med", w_md), ("dlm", w_dp)):
            if w <= 0:
                continue
            gu, gv = self.field(hours, name)
            u = u + w * env.sample(gu, lat, lon)
            v = v + w * env.sample(gv, lat, lon)
        return u, v


def fflow_member(ctx, flow: ForecastFlow, leads, tau_h: float = 24.0, depth: float | None = 0.5) -> np.ndarray:
    """
    The storm carried through the forecast flow, plus beta drift, plus the part of its present motion the flow does
    not explain, fading with ``tau_h`` (``(L, 2)`` lat, lon). ``depth`` 0.5 is the 850-400 hPa layer, which beat the
    intensity-dependent depth (``None``) and the shallow and deep layers on the tuning seasons.
    """
    from ..swarm.core import depth_for_wind
    from .trajectory import DecayingResidual, combined, integrate

    case = ctx.case
    depth = depth_for_wind(case.wind_kt) if depth is None else float(depth)
    env = ctx.clean
    steer = lambda lat, lon, hours: flow.velocity(env, lat, lon, hours, depth)
    u0, v0 = steer(np.array([case.lat]), np.array([case.lon]), 0.0)
    b0 = ctx.beta0()
    residual = DecayingResidual(ctx.motion[0] - float(u0[0]) - b0[0], ctx.motion[1] - float(v0[0]) - b0[1], tau_h)
    return integrate(case.lat, case.lon, combined(steer, ctx.beta_velocity(), residual), leads)


__all__ = ["ForecastFlow", "LAYERS", "fflow_member"]
