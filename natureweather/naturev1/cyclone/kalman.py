# Copyright 2026 Nathan. Apache-2.0.
"""
How the storm is moving right now, estimated rather than read off two fixes.

The idea: extrapolation is nearly unbeatable at 6-12 hours because the current motion is the best
predictor there is -- but extrapolation *measures* that motion with two noisy best-track fixes. A best-track
position carries 20-50 km of uncertainty (more for a depression seen only by satellite), so a 12-hour motion
from two fixes is uncertain by several km/h before the forecast even starts. A Kalman filter over the last 48
hours of fixes, with a constant-acceleration motion model, estimates position, velocity and acceleration
jointly and weighs each fix by how much it can be trusted. Two refinements:

* **Robust updates** (Huber): a fix that disagrees with the motion by more than a few standard deviations
  -- a centre reformation, a relocated depression -- is down-weighted instead of dragging the velocity.
* **Damped acceleration** forecast (Gardner-McKenzie damped trend): the measured turn or speed-up carries
  on but fades with an e-folding time, because storms that are accelerating now rarely keep accelerating for
  days.

All in the storm-centred azimuthal-equidistant frame, km and hours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import from_local, to_local

#: Per-axis position uncertainty of a best-track fix, km, by status (after Landsea and Franklin 2013).
FIX_SIGMA_KM = {"HU": 15.0, "TS": 24.0, "SS": 30.0, "TD": 32.0}
#: White-jerk process noise, km^2 h^-5 (tuned on the 2016/2017/2019 seasons only).
PROCESS_NOISE = 2.0e-4
#: Huber threshold, standard deviations.
HUBER = 2.0
#: E-folding time of the measured acceleration in a forecast, hours.
ACCELERATION_TAU_H = 12.0


@dataclass
class MotionEstimate:
    position_km: np.ndarray        # (2,) filtered offset from the best-track fix now (x east, y north)
    velocity_kmh: np.ndarray       # (2,)
    acceleration_kmh2: np.ndarray  # (2,)
    covariance: np.ndarray         # (6, 6) state covariance
    fixes: int

    @property
    def velocity_ms(self) -> np.ndarray:
        return self.velocity_kmh / 3.6

    @property
    def speed_sigma_kmh(self) -> float:
        return float(np.sqrt(0.5 * (self.covariance[2, 2] + self.covariance[3, 3])))


def _transition(dt: float) -> np.ndarray:
    f = np.eye(6)
    for axis in range(2):
        p, v, a = axis, 2 + axis, 4 + axis
        f[p, v], f[p, a], f[v, a] = dt, 0.5 * dt * dt, dt
    return f


def _process(dt: float, q: float) -> np.ndarray:
    block = q * np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                          [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                          [dt ** 3 / 6, dt ** 2 / 2, dt]])
    out = np.zeros((6, 6))
    for axis in range(2):
        idx = [axis, 2 + axis, 4 + axis]
        out[np.ix_(idx, idx)] = block
    return out


def estimate_motion(case, process_noise: float | None = None, huber: float | None = None) -> MotionEstimate | None:
    """Filter the storm's last 48 hours of fixes into position, velocity and acceleration now."""
    process_noise = PROCESS_NOISE if process_noise is None else process_noise
    huber = HUBER if huber is None else huber
    from .data import HISTORY_HOURS

    hours = np.asarray(HISTORY_HOURS, np.float64)
    lat, lon = case.history["lat"], case.history["lon"]
    ok = np.isfinite(lat) & np.isfinite(lon)
    if ok.sum() < 2:
        return None
    x, y = to_local(case.lat, case.lon, lat[ok], lon[ok])
    t = hours[ok]
    sigma = FIX_SIGMA_KM.get(case.status, 28.0)
    r = np.eye(2) * sigma ** 2
    h = np.zeros((2, 6))
    h[0, 0] = h[1, 1] = 1.0
    # Start at the first fix with a vague velocity.
    state = np.array([x[0], y[0], 0.0, 0.0, 0.0, 0.0])
    cov = np.diag([sigma ** 2, sigma ** 2, 30.0 ** 2, 30.0 ** 2, 2.0 ** 2, 2.0 ** 2])
    if len(t) > 1:
        dt0 = t[1] - t[0]
        state[2:4] = [(x[1] - x[0]) / dt0, (y[1] - y[0]) / dt0]
    previous = t[0]
    for k in range(len(t)):
        dt = t[k] - previous
        if dt > 0:
            f = _transition(dt)
            state = f @ state
            cov = f @ cov @ f.T + _process(dt, process_noise)
        previous = t[k]
        innovation = np.array([x[k], y[k]]) - h @ state
        s = h @ cov @ h.T + r
        distance = float(np.sqrt(innovation @ np.linalg.solve(s, innovation)))
        inflate = max(1.0, distance / huber)
        s = h @ cov @ h.T + r * inflate
        gain = cov @ h.T @ np.linalg.inv(s)
        state = state + gain @ innovation
        cov = (np.eye(6) - gain @ h) @ cov
    return MotionEstimate(state[:2], state[2:4], state[4:6], cov, int(ok.sum()))


def damped_offsets(estimate: MotionEstimate, leads, tau_h: float | None = None,
                   from_fix: bool = True) -> np.ndarray:
    """``(L, 2)`` (x, y) km offsets at each lead, the acceleration fading with ``tau_h``."""
    tau_h = ACCELERATION_TAU_H if tau_h is None else tau_h
    t = np.asarray(leads, np.float64)[:, None]
    decay = tau_h * (t - tau_h * (1.0 - np.exp(-t / tau_h)))
    start = np.zeros(2) if from_fix else estimate.position_km
    return start + estimate.velocity_kmh * t + estimate.acceleration_kmh2 * decay


def kalman_track(case, leads, tau_h: float | None = None, from_fix: bool = True) -> np.ndarray:
    """``(L, 2)`` lat/lon: the filtered motion carried forward with damped acceleration."""
    estimate = estimate_motion(case)
    if estimate is None:
        return np.full((len(leads), 2), np.nan)
    offsets = damped_offsets(estimate, leads, tau_h, from_fix)
    lat, lon = from_local(case.lat, case.lon, offsets[:, 0], offsets[:, 1])
    return np.stack([lat, lon], -1)


__all__ = ["ACCELERATION_TAU_H", "FIX_SIGMA_KM", "MotionEstimate", "damped_offsets", "estimate_motion", "kalman_track"]
