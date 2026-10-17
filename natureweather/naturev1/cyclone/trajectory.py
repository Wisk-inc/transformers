# Copyright 2026 Nathan. Apache-2.0.
"""
Carrying the storm forward: fourth-order Runge-Kutta through a flow that is sampled where the storm *will* be.

The idea: steering is not a single vector computed at the initial position and extrapolated -- that is just
extrapolation with a different arrow. The storm is a parcel; at every hour it moves with the environmental
flow at its *current* forecast position, so a storm heading into the weakness between two ridges slows down
because the flow there is weak, and turns because the flow there turns. Every step is a great-circle move
(:func:`naturev1.cyclone.geometry.move`), so the integration is exact in geometry at any latitude and across
the dateline. The integrator is vectorised over ensemble members.

The velocity is a sum of parts, each a callable ``(lat, lon, hours) -> (u, v)`` m/s:

* the steering flow (frozen analysis, or one that evolves -- :class:`EvolvingFlow`);
* the beta drift (:class:`naturev1.cyclone.beta.BetaDrift`);
* the **initial-motion residual** (:class:`DecayingResidual`): what the storm is doing now that the parts
  above do not explain, carried on and faded with an e-folding time -- the part of persistence the physics
  cannot account for (steering errors in the analysis, propagation, the vortex's own asymmetries).
"""

from __future__ import annotations

import numpy as np

from .geometry import move


class DecayingResidual:
    """A constant (u, v) m/s that fades as ``exp(-t / tau_h)``."""

    def __init__(self, u: float, v: float, tau_h: float) -> None:
        self.u, self.v, self.tau_h = float(u), float(v), float(tau_h)

    def __call__(self, lat, lon, hours):
        factor = np.exp(-np.asarray(hours, np.float64) / max(self.tau_h, 1e-6))
        shape = np.shape(lat)
        return np.full(shape, self.u * factor), np.full(shape, self.v * factor)


class EvolvingFlow:
    """
    A steering flow that keeps changing the way it was changing: ``now + (now - before) * g(t)``, with the
    12-hour tendency's effect saturating at ``tau_h`` (``g(t) = tau (1 - exp(-t / tau)) / 12``), so a ridge that
    was building keeps building for a while, but never without limit.
    """

    def __init__(self, now, before, tau_h: float = 12.0, before_hours: float = 12.0) -> None:
        self.now, self.before, self.tau_h, self.before_hours = now, before, tau_h, before_hours

    def __call__(self, lat, lon, hours):
        u1, v1 = self.now(lat, lon, hours)
        if self.before is None or self.tau_h <= 0:
            return u1, v1
        u0, v0 = self.before(lat, lon, hours)
        g = self.tau_h * (1.0 - np.exp(-np.asarray(hours, np.float64) / self.tau_h)) / self.before_hours
        return u1 + (u1 - u0) * g, v1 + (v1 - v0) * g


def frozen(flow):
    """Adapt a ``velocity(lat, lon)`` object (e.g. :class:`naturev1.cyclone.steering.SteeringFlow`) to the
    ``(lat, lon, hours)`` signature."""
    return lambda lat, lon, hours: flow.velocity(lat, lon)


def combined(*parts):
    """Sum of velocity callables (None entries skipped)."""
    parts = [p for p in parts if p is not None]

    def velocity(lat, lon, hours):
        u = np.zeros(np.shape(lat))
        v = np.zeros(np.shape(lat))
        for part in parts:
            du, dv = part(lat, lon, hours)
            u = u + du
            v = v + dv
        return u, v

    return velocity


def integrate(lat0, lon0, velocity, leads, step_hours: float = 3.0, speed_cap_ms: float = 35.0) -> np.ndarray:
    """
    RK4 trajectories from ``(lat0, lon0)`` (scalars or ``(M,)`` arrays) under ``velocity(lat, lon, hours)``.

    Returns ``(L, 2)`` (or ``(M, L, 2)``) positions at ``leads`` hours. Speeds above ``speed_cap_ms`` are
    clipped -- no tropical cyclone has been carried faster, and a bad sample of the field should not throw one
    across an ocean.
    """
    scalar = np.ndim(lat0) == 0
    lat = np.atleast_1d(np.asarray(lat0, np.float64)).copy()
    lon = np.atleast_1d(np.asarray(lon0, np.float64)).copy()
    leads = [float(h) for h in leads]
    out = np.full((lat.size, len(leads), 2), np.nan)
    t, k = 0.0, 0
    end = max(leads) if leads else 0.0
    dt_s = step_hours * 3600.0

    def capped(u, v):
        speed = np.hypot(u, v)
        scale = np.where(speed > speed_cap_ms, speed_cap_ms / np.maximum(speed, 1e-9), 1.0)
        return u * scale, v * scale

    while k < len(leads) and abs(leads[k]) < 1e-9:
        out[:, k] = np.stack([lat, lon], -1)
        k += 1
    while t < end - 1e-9:
        h = min(step_hours, end - t)
        dt = h * 3600.0
        u1, v1 = capped(*velocity(lat, lon, t))
        la, lo = move(lat, lon, 0.5 * dt * u1 / 1000.0, 0.5 * dt * v1 / 1000.0)
        u2, v2 = capped(*velocity(la, lo, t + 0.5 * h))
        la, lo = move(lat, lon, 0.5 * dt * u2 / 1000.0, 0.5 * dt * v2 / 1000.0)
        u3, v3 = capped(*velocity(la, lo, t + 0.5 * h))
        la, lo = move(lat, lon, dt * u3 / 1000.0, dt * v3 / 1000.0)
        u4, v4 = capped(*velocity(la, lo, t + h))
        u = (u1 + 2 * u2 + 2 * u3 + u4) / 6.0
        v = (v1 + 2 * v2 + 2 * v3 + v4) / 6.0
        lat, lon = move(lat, lon, dt * u / 1000.0, dt * v / 1000.0)
        t += h
        while k < len(leads) and leads[k] <= t + 1e-9:
            out[:, k] = np.stack([lat, lon], -1)
            k += 1
    del dt_s
    return out[0] if scalar else out


__all__ = ["DecayingResidual", "EvolvingFlow", "combined", "frozen", "integrate"]
