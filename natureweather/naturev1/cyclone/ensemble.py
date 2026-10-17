# Copyright 2026 Nathan. Apache-2.0.
"""
An ensemble that knows where the analysis is uncertain -- the Bermuda High problem.

The idea: Dorian's turn depended on the exact strength and edge of the Bermuda High, which sits over open ocean
where no radiosondes are launched; models had to infer it from satellite retrievals. A single physics forecast
from that analysis inherits whatever error it holds there. An ensemble spreads the forecast over the
analysis's plausible errors -- and the ensemble *mean* is, on average, more accurate than any one member,
because it averages out the unpredictable part. Each member:

* adds a smooth, non-divergent perturbation to the steering flow: a few Gaussian streamfunction blobs
  (600-1100 km scale) scattered within 2,500 km of the storm, with amplitudes scaled to the climatological
  day-to-day variability of the 500 hPa wind at each blob (big where the weather is variable, small in the
  steady trades), **inflated over open ocean far from land** (the data voids where the analysis is least
  constrained), and growing with lead time as forecast errors do;
* draws its own steering layer (from the family in :data:`naturev1.cyclone.steering.LAYERS`), its own beta
  drift strength and its own memory of the initial-motion residual -- the structural uncertainty of the
  physics, not just of the initial state.

Perturbation winds are evaluated analytically at the members' positions (derivatives of Gaussians), so the
whole ensemble integrates as one vectorised trajectory.
"""

from __future__ import annotations

import numpy as np

from .geometry import destination, great_circle_km, to_local

MEMBERS = 24
BLOBS = 6
#: Perturbation wind as a fraction of the climatological 500 hPa variability, and the open-ocean inflation.
AMPLITUDE = 0.3
OCEAN_INFLATION = 1.35
#: Error growth: the perturbation's size doubles after this many hours.
DOUBLING_H = 60.0


class FlowPerturbations:
    """``(M, K)`` Gaussian streamfunction blobs; ``velocity(member_lat, member_lon, hours)`` -> ``(M,)`` (u, v)."""

    def __init__(self, lat0: float, lon0: float, env, climate=None, members: int = MEMBERS, blobs: int = BLOBS,
                 amplitude: float = AMPLITUDE, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        distance = 2500.0 * np.sqrt(rng.uniform(0.0, 1.0, (members, blobs)))
        bearing = rng.uniform(0.0, 360.0, (members, blobs))
        self.lat, self.lon = destination(lat0, lon0, bearing, distance)
        self.scale_km = rng.uniform(600.0, 1100.0, (members, blobs))
        sigma = np.full((members, blobs), 3.0)
        if climate is not None and "ustd500" in climate.fields:
            std = np.hypot(climate.sample(climate.fields["ustd500"], self.lat, self.lon),
                           climate.sample(climate.fields["vstd500"], self.lat, self.lon)) / np.sqrt(2.0)
            sigma = np.where(np.isfinite(std), std, 3.0)
        land = env.sample(env.fields["lsm"], self.lat, self.lon) if "lsm" in env.fields else np.zeros_like(sigma)
        sigma = sigma * np.where(land < 0.2, OCEAN_INFLATION, 1.0)
        # Peak wind of a Gaussian streamfunction blob is psi0 / (L sqrt(e)); choose psi0 for a peak of amplitude*sigma.
        wind = amplitude * sigma * rng.choice([-1.0, 1.0], (members, blobs))
        self.psi0 = wind * self.scale_km * 1000.0 * np.sqrt(np.e)

    def velocity(self, lat, lon, hours):
        lat = np.atleast_1d(lat)
        lon = np.atleast_1d(lon)
        x, y = to_local(self.lat, self.lon, lat[:, None], lon[:, None])          # member position relative to blobs
        x, y = x * 1000.0, y * 1000.0
        l2 = (self.scale_km * 1000.0) ** 2
        g = self.psi0 * np.exp(-0.5 * (x * x + y * y) / l2)
        u = (g * y / l2).sum(1)                                                   # u = -dpsi/dy
        v = (-g * x / l2).sum(1)                                                  # v =  dpsi/dx
        growth = 2.0 ** (np.asarray(hours, np.float64) / DOUBLING_H)
        return u * growth, v * growth


def ensemble_mean(tracks: np.ndarray) -> np.ndarray:
    """``(M, L, 2)`` -> ``(L, 2)``: the mean position of the members alive at each lead (local-frame average)."""
    from .geometry import from_local

    out = np.full(tracks.shape[1:], np.nan)
    for k in range(tracks.shape[1]):
        ok = np.isfinite(tracks[:, k, 0])
        if not ok.any():
            continue
        ref = tracks[ok, k][0]
        x, y = to_local(ref[0], ref[1], tracks[ok, k, 0], tracks[ok, k, 1])
        out[k] = np.stack(from_local(ref[0], ref[1], x.mean(), y.mean()), -1)
    return out


def ensemble_spread_km(tracks: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """``(L,)`` mean distance of the members from the ensemble mean."""
    return np.nanmean(great_circle_km(tracks[..., 0], tracks[..., 1], mean[None, :, 0], mean[None, :, 1]), axis=0)


def physics_ensemble(case, env_clean, flows: dict, beta, motion_ms, leads, climate=None, climate_flows=None,
                     members: int = MEMBERS, seed: int = 0, tau_range=(12.0, 48.0), beta_range=(0.6, 1.4),
                     anomaly_tau_h: float = 72.0) -> dict:
    """
    Run the ensemble. ``flows``: ``{layer name: SteeringFlow}`` of the vortex-removed analysis (members pick one
    each); ``climate_flows`` the matching climatological flows (then each member's steering anomaly decays with
    ``anomaly_tau_h``). Returns ``{"members": (M, L, 2), "mean": (L, 2), "spread_km": (L,)}``.
    """
    from .trajectory import integrate

    rng = np.random.default_rng(seed + 7919)
    names = list(flows)
    choice = rng.integers(0, len(names), members)
    tau = rng.uniform(*tau_range, members)
    beta_scale = rng.uniform(*beta_range, members)
    perturb = FlowPerturbations(case.lat, case.lon, env_clean, climate, members, seed=seed)
    start = {n: tuple(float(a[0]) for a in flows[n].velocity(case.lat, case.lon)) for n in names}
    b0 = tuple(float(a[0]) for a in beta.velocity(case.lat, case.lon))
    residual = np.array([[motion_ms[0] - start[names[c]][0] - beta_scale[m] * b0[0],
                          motion_ms[1] - start[names[c]][1] - beta_scale[m] * b0[1]] for m, c in enumerate(choice)])

    def velocity(lat, lon, hours):
        u = np.zeros(members)
        v = np.zeros(members)
        for k, name in enumerate(names):
            sel = choice == k
            if sel.any():
                us, vs = flows[name].velocity(lat[sel], lon[sel])
                if climate_flows is not None and name in climate_flows:
                    uc, vc = climate_flows[name].velocity(lat[sel], lon[sel])
                    memory = np.exp(-hours / anomaly_tau_h)
                    us, vs = uc + memory * (us - uc), vc + memory * (vs - vc)
                u[sel], v[sel] = us, vs
        bu, bv = beta.velocity(lat, lon)
        pu, pv = perturb.velocity(lat, lon, hours)
        decay = np.exp(-hours / tau)
        return (u + beta_scale * bu + pu + residual[:, 0] * decay, v + beta_scale * bv + pv + residual[:, 1] * decay)

    tracks = integrate(np.full(members, case.lat), np.full(members, case.lon), velocity, leads)
    mean = ensemble_mean(tracks)
    return {"members": tracks, "mean": mean, "spread_km": ensemble_spread_km(tracks, mean)}


__all__ = ["AMPLITUDE", "FlowPerturbations", "MEMBERS", "ensemble_mean", "ensemble_spread_km", "physics_ensemble"]
