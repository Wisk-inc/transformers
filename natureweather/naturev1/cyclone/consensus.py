# Copyright 2026 Nathan. Apache-2.0.
"""
A consensus that weighs speed and direction separately.

The idea: forecast members are good at different things. Extrapolation knows how fast the storm is going
right now; a steering trajectory knows where the flow will turn it; climatology knows that storms slow down
near the ridge. Averaging whole positions throws that away -- a member with excellent direction but poor
speed drags the consensus off in both. So each member's displacement (in the storm-centred frame, km) is
split into an **along-track** part (along the direction of a reference forecast) and a **cross-track** part,
and the two get their own weights at every lead, fitted on past seasons:

* cross-track weights are non-negative and sum to one (a convex blend of directions);
* along-track weights are non-negative but free in sum, which lets the consensus correct the systematic
  slowness or quickness that every physics member has (a scale of 1.05 is a 5% speed-up);
* members missing for a case drop out and the rest renormalise (cross-track) or rescale by the fitted
  total (along-track).

Weights are fitted by projected gradient descent on the squared error in km (convex; the projection is onto the
simplex for cross-track and onto the non-negative orthant for along-track). Then, optionally, a
**selective** step (Goerss's "selective consensus") drops members that disagree with the median by more
than ``k`` times the members' spread before blending.
"""

from __future__ import annotations

import numpy as np

from .geometry import from_local, to_local


def to_displacements(cases, forecasts: np.ndarray) -> np.ndarray:
    """``(N, L, M, 2)`` lat/lon -> ``(N, L, M, 2)`` (east, north) km from each case's initial position."""
    lat0 = np.array([c.lat for c in cases])[:, None, None]
    lon0 = np.array([c.lon for c in cases])[:, None, None]
    east, north = to_local(lat0, lon0, forecasts[..., 0], forecasts[..., 1])
    out = np.stack([east, north], -1)
    out[~np.isfinite(forecasts).all(-1)] = np.nan
    return out


def from_displacements(cases, displacement: np.ndarray) -> np.ndarray:
    """``(N, L, 2)`` km -> lat/lon."""
    lat0 = np.array([c.lat for c in cases])[:, None]
    lon0 = np.array([c.lon for c in cases])[:, None]
    lat, lon = from_local(lat0, lon0, displacement[..., 0], displacement[..., 1])
    out = np.stack([lat, lon], -1)
    out[~np.isfinite(displacement).all(-1)] = np.nan
    return out


def project_simplex(w: np.ndarray) -> np.ndarray:
    """Nearest point on the probability simplex (Duchi et al. 2008)."""
    u = np.sort(w)[::-1]
    css = np.cumsum(u) - 1.0
    k = np.arange(1, w.size + 1)
    rho = np.nonzero(u - css / k > 0)[0][-1]
    return np.maximum(w - css[rho] / (rho + 1.0), 0.0)


def _reference_direction(displacements: np.ndarray, reference: int) -> np.ndarray:
    """``(N, L, 2)`` unit vector of the reference member (falling back to the member mean)."""
    ref = displacements[:, :, reference]
    mean = np.nanmean(displacements, axis=2)
    ref = np.where(np.isfinite(ref), ref, mean)
    length = np.linalg.norm(ref, axis=-1, keepdims=True)
    unit = ref / np.maximum(length, 1e-6)
    unit[(length[..., 0] < 1e-6)] = np.array([0.0, 1.0])
    return unit


def _fit_one(a: np.ndarray, b: np.ndarray, simplex: bool, steps: int = 4000) -> np.ndarray:
    """
    Weights minimising the mean of ``(a w - b)^2`` (members missing from a case, NaN in ``a``, are renormalised
    away), by accelerated projected gradient (FISTA) with the exact Lipschitz step ``1 / lambda_max(a^T a / n)`` --
    nearly collinear members make a cruder step diverge.
    """
    n, m = a.shape
    ok = np.isfinite(a)
    filled = np.where(ok, a, 0.0)
    lipschitz = float(np.linalg.eigvalsh(filled.T @ filled / n)[-1])
    lr = 1.0 / max(lipschitz, 1e-12)
    project = project_simplex if simplex else (lambda x: np.maximum(x, 0.0))
    w = np.full(m, 1.0 / m)
    z, t = w.copy(), 1.0
    for _ in range(steps):
        active = ok @ z
        norm = np.where(active > 0, active / max(float(z.sum()), 1e-9), 1.0)
        resid = (filled @ z) / norm - b
        grad = (filled / norm[:, None]).T @ resid / n
        w_next = project(z - lr * grad)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = w_next + ((t - 1.0) / t_next) * (w_next - w)
        if not simplex:
            z = np.maximum(z, 0.0)
        w, t = w_next, t_next
    return w


def fit_consensus(cases, forecasts: np.ndarray, reference: int, valid_members: np.ndarray | None = None) -> dict:
    """
    Fit along- and cross-track weights per lead. ``forecasts``: ``(N, L, M, 2)`` lat/lon of every member.

    Returns ``{"along": (L, M), "cross": (L, M), "reference": int}``.
    """
    disp = to_displacements(cases, forecasts)
    truth = to_displacements(cases, np.stack([np.stack([c.truth["lat"], c.truth["lon"]], -1) for c in cases])[:, :, None])[:, :, 0]
    unit = _reference_direction(disp, reference)
    normal = np.stack([-unit[..., 1], unit[..., 0]], -1)
    along = (disp * unit[:, :, None]).sum(-1)            # (N, L, M)
    cross = (disp * normal[:, :, None]).sum(-1)
    t_along = (truth * unit).sum(-1)
    t_cross = (truth * normal).sum(-1)
    leads, members = disp.shape[1], disp.shape[2]
    wa, wc = np.zeros((leads, members)), np.zeros((leads, members))
    for k in range(leads):
        ok = np.isfinite(t_along[:, k]) & np.isfinite(along[:, k]).any(-1)
        if valid_members is not None:
            ok &= valid_members
        if ok.sum() < 20:
            wa[k] = wc[k] = 1.0 / members
            continue
        wa[k] = _fit_one(along[ok, k], t_along[ok, k], simplex=False)
        wc[k] = _fit_one(cross[ok, k], t_cross[ok, k], simplex=True)
    return {"along": wa, "cross": wc, "reference": int(reference)}


def apply_consensus(cases, forecasts: np.ndarray, weights: dict, selective_k: float | None = None) -> np.ndarray:
    """``(N, L, 2)`` lat/lon: the fitted blend of whichever members each case has."""
    disp = to_displacements(cases, forecasts)
    if selective_k is not None:
        median = np.nanmedian(disp, axis=2, keepdims=True)
        distance = np.linalg.norm(disp - median, axis=-1)
        spread = np.nanmedian(distance, axis=2, keepdims=True)
        drop = distance > selective_k * np.maximum(spread, 25.0)
        disp = np.where(drop[..., None], np.nan, disp)
    unit = _reference_direction(disp, weights["reference"])
    normal = np.stack([-unit[..., 1], unit[..., 0]], -1)
    along = (disp * unit[:, :, None]).sum(-1)
    cross = (disp * normal[:, :, None]).sum(-1)
    ok = np.isfinite(along)
    wa = np.asarray(weights["along"])[None]
    wc = np.asarray(weights["cross"])[None]
    total_a = wa.sum(-1)
    cover_a = (wa * ok).sum(-1)
    cover_c = (wc * ok).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = (np.where(ok, along, 0.0) * wa).sum(-1) * total_a / np.where(cover_a > 0, cover_a, np.nan)
        c = (np.where(ok, cross, 0.0) * wc).sum(-1) / np.where(cover_c > 0, cover_c, np.nan)
    blended = a[..., None] * unit + c[..., None] * normal
    return from_displacements(cases, blended)


def blend_with_model(case, track: np.ndarray, model_track: np.ndarray, spec: dict, leads) -> np.ndarray:
    """
    The consensus pulled toward a dynamical model's own track of the storm, by one weight per lead
    (``spec["alpha"]``, fitted on past seasons: a quarter at 6 h, nine tenths by day 5, where the model knows about
    the troughs and ridges the physics cannot see yet).

    The model's track first loses its starting offset, as NHC's "interpolated" model tracks do: where the model
    starts the storm, extrapolated back from its first two positions, is moved onto the observed position, the
    correction fading with ``spec["offset_tau_h"]``. Where the model has lost the storm, the consensus stands.
    """
    hours = np.asarray(leads, np.float64)
    alpha = np.interp(hours, spec.get("leads", hours), spec["alpha"])
    east, north = to_local(case.lat, case.lon, track[:, 0], track[:, 1])
    me, mn = to_local(case.lat, case.lon, model_track[:, 0], model_track[:, 1])
    tau = spec.get("offset_tau_h")
    if tau:
        f = hours[0] / max(hours[1] - hours[0], 1e-9)
        p0e, p0n = me[0] - (me[1] - me[0]) * f, mn[0] - (mn[1] - mn[0]) * f
        decay = np.exp(-hours / float(tau))
        me, mn = me - p0e * decay, mn - p0n * decay
    model_ok = np.isfinite(me) & np.isfinite(mn)
    e = np.where(model_ok, (1 - alpha) * east + alpha * me, east)
    n = np.where(model_ok, (1 - alpha) * north + alpha * mn, north)
    lat, lon = from_local(case.lat, case.lon, e, n)
    out = np.stack([lat, lon], -1)
    out[~np.isfinite(track).all(-1)] = np.nan
    return out


__all__ = ["apply_consensus", "blend_with_model", "fit_consensus", "from_displacements", "project_simplex",
           "to_displacements"]
