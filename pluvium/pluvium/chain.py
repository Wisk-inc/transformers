# Copyright 2026 Nathan. Apache-2.0.
"""
The intensity chain (experimental): rain at a place as a hidden Markov chain over *intensity states* -- dry, light,
moderate, heavy -- whose hour-by-hour transition matrix the network writes for every place.

Version 1 of Pluvium had two states (dry, wet). Four states let one chain answer "when does rain start", "when does
it turn heavy", "how long does the heavy part last" and "how much falls" consistently, and let training use what each
instrument actually says about intensity: the satellite's rate class, an airport's ``-RA`` / ``RA`` / ``+RA``. Each
instrument is a learned *confusion matrix* from the true state to the class it reports (the satellite smears an 8 km
box and misses light showers; an airport sees one spot), so training never takes an instrument's word as truth.

Amounts within a state are a lognormal truncated to the state's range (light 0.2-2.5 mm/h, moderate 2.5-7.6, heavy
7.6 and over), with its location and spread written by the network per hour; two-state models keep version 1's
free lognormal mixture for the wet state.

Torch functions train; NumPy functions make products.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .weather import EDGES, TORRENTIAL

#: Geometric centres of the wet states' ranges (mm/h), where the truncated lognormals start.
CENTRES = (math.sqrt(0.2 * 2.5), math.sqrt(2.5 * 7.6), 15.0)
#: Initial transition probabilities (from row state to column state) for a Caribbean hour, and initial-state odds.
PRIOR_TRANSITIONS = {
    2: ((0.95, 0.05), (0.45, 0.55)),
    4: ((0.95, 0.035, 0.01, 0.005), (0.45, 0.40, 0.12, 0.03), (0.20, 0.35, 0.30, 0.15), (0.10, 0.25, 0.30, 0.35)),
}
PRIOR_INITIAL = {2: (0.92, 0.08), 4: (0.92, 0.05, 0.02, 0.01)}


def bounds(states: int) -> list[tuple[float, float]]:
    """``(low, high)`` mm/h of each wet state."""
    if states == 2:
        return [(EDGES[0], math.inf)]
    edges = list(EDGES) + [math.inf]
    return [(edges[k], edges[k + 1]) for k in range(states - 1)]


def rate_classes(rate: torch.Tensor, states: int) -> torch.Tensor:
    """Class index of rates (mm/h); NaN -> -1."""
    edges = torch.tensor(EDGES[:1] if states == 2 else EDGES, device=rate.device, dtype=rate.dtype)
    cls = torch.bucketize(torch.nan_to_num(rate, nan=-1.0), edges, right=True)
    return torch.where(torch.isfinite(rate), cls, torch.full_like(cls, -1))


def class_mask_from_rate(rate: torch.Tensor, states: int) -> torch.Tensor:
    """``(..., states)`` bool: the observed class (all True where the rate is missing)."""
    cls = rate_classes(rate, states)
    one = F.one_hot(cls.clamp(min=0), states).bool()
    return torch.where((cls >= 0).unsqueeze(-1), one, torch.ones_like(one))


# ------------------------------------------------------------------------------------------- torch --

def hmm_nll(trans: torch.Tensor, init: torch.Tensor, mask: torch.Tensor, confusion: torch.Tensor):
    """
    Negative log-likelihood of observed classes under the chain, seen through an instrument.

    ``trans (B, T, P, K, K)`` transition logits into hour t (row: from), ``init (B, P, K)`` logits of the state in
    hour 0, ``mask (B, T, P, K)`` the classes each observation allows (all True: no observation), ``confusion (K, K)``
    probabilities ``P(reported class | true state)``. Returns ``(total NLL, number of observations)``.
    """
    m = mask.to(trans.dtype)
    emit = torch.log((m @ confusion.T.to(trans.dtype)).clamp(min=1e-12))         # (B, T, P, K): log P(obs | state)
    known = (~mask.all(-1)).to(trans.dtype)
    log_a = F.log_softmax(trans, dim=-1)
    alpha = F.log_softmax(init, dim=-1) + emit[:, 0]
    for t in range(1, trans.shape[1]):
        alpha = torch.logsumexp(alpha.unsqueeze(-1) + log_a[:, t], dim=-2) + emit[:, t]
    return -torch.logsumexp(alpha, dim=-1).sum(), known.sum()


def state_marginals(trans: torch.Tensor, init: torch.Tensor) -> torch.Tensor:
    """``(B, T, P, K)`` P(state) in every hour."""
    a = torch.softmax(trans, dim=-1)
    p = torch.softmax(init, dim=-1)
    out = [p]
    for t in range(1, trans.shape[1]):
        p = (p.unsqueeze(-2) @ a[:, t]).squeeze(-2)
        out.append(p)
    return torch.stack(out, 1)


def _log_norm_cdf_diff(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """``log(Phi(beta) - Phi(alpha))`` stably (beta may be +inf)."""
    upper = torch.isinf(beta)
    tail = torch.special.log_ndtr(-alpha)                                         # beta = inf
    hi = torch.special.log_ndtr(torch.where(upper, torch.zeros_like(beta), beta))
    lo = torch.special.log_ndtr(alpha)
    finite = hi + torch.log1p(-torch.exp((lo - hi).clamp(max=-1e-7)))
    return torch.where(upper, tail, finite)


def truncated_log_prob(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """log density of rates ``y`` under a lognormal(mu, sigma) truncated to ``[low, high)``."""
    ly = torch.log(y.clamp(min=1e-3))
    z = (ly - mu) / sigma
    a = (math.log(low) - mu) / sigma
    b = (math.log(high) - mu) / sigma if math.isfinite(high) else torch.full_like(mu, math.inf)
    return -0.5 * z**2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi) - ly - _log_norm_cdf_diff(a, b)


def wet_probability(out: dict) -> torch.Tensor:
    return 1.0 - state_marginals(out["trans"], out["init"])[..., 0]


# ------------------------------------------------------------------------------------------- numpy --

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def _ndtr(x):
    from scipy.special import ndtr

    return ndtr(x)


def marginals_np(a: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """``a (P, T, K, K)`` transition probabilities, ``pi (P, K)`` -> ``(P, T, K)``."""
    p = pi.astype(np.float64)
    out = np.empty(a.shape[:3], np.float64)
    out[:, 0] = p
    for t in range(1, a.shape[1]):
        p = np.einsum("pk,pkj->pj", p, a[:, t])
        out[:, t] = p
    return out


def first_entry(a: np.ndarray, start: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    ``P(first hour t >= 1 in which the state is in `target` (bool (K,)))`` starting from the distribution ``start
    (P, K)`` over states outside the target set at hour 0. Returns ``(P, T)`` (entry 0 = 0); the rest of the mass
    never enters within the window.
    """
    keep = ~target
    v = start * keep
    out = np.zeros(a.shape[:2])
    for t in range(1, a.shape[1]):
        step = np.einsum("pk,pkj->pj", v, a[:, t])
        out[:, t] = step[:, target].sum(-1)
        v = step * keep
    return out


def quantile_hour(dist: np.ndarray, q: float) -> np.ndarray:
    total = dist.sum(-1, keepdims=True)
    cum = np.cumsum(dist, -1) / np.maximum(total, 1e-12)
    hour = (cum >= q).argmax(-1).astype(np.float64)
    return np.where(total[..., 0] > 1e-6, hour, np.nan)


def amount_params(comp: dict, states: int):
    """Per-wet-state ``(mu, sigma)`` arrays ``(P, T, K-1)`` (K > 2) or the mixture (K = 2)."""
    if states == 2:
        return comp["logit"], comp["mu"], np.exp(comp["logsig"])
    return None, comp["amt_mu"], comp["amt_sig"]


def state_means(comp: dict, states: int) -> np.ndarray:
    """``E[rate | state]`` ``(P, T, K)`` (0 for dry)."""
    p, t = comp["a"].shape[:2]
    out = np.zeros((p, t, states))
    if states == 2:
        w = _softmax(comp["logit"])
        out[..., 1] = (w * np.exp(comp["mu"] + 0.5 * np.exp(2 * comp["logsig"]))).sum(-1)
        return out
    for k, (lo, hi) in enumerate(bounds(states)):
        mu, s = comp["amt_mu"][..., k], comp["amt_sig"][..., k]
        a = (np.log(lo) - mu) / s
        b = (np.log(hi) - mu) / s if np.isfinite(hi) else np.full_like(mu, np.inf)
        den = np.maximum(_ndtr(b) - _ndtr(a), 1e-12)
        out[..., k + 1] = np.exp(mu + 0.5 * s**2) * np.maximum(_ndtr(b - s) - _ndtr(a - s), 0) / den
    return out


def torrential_given_heavy(comp: dict, states: int) -> np.ndarray:
    """``P(rate >= 50 mm/h | heaviest state)`` ``(P, T)``."""
    if states == 2:
        w = _softmax(comp["logit"])
        z = (np.log(TORRENTIAL) - comp["mu"]) / np.exp(comp["logsig"])
        return (w * (1 - _ndtr(z))).sum(-1)
    mu, s = comp["amt_mu"][..., -1], comp["amt_sig"][..., -1]
    a = (np.log(EDGES[-1]) - mu) / s
    g = (np.log(TORRENTIAL) - mu) / s
    return (1 - _ndtr(g)) / np.maximum(1 - _ndtr(a), 1e-12)


def class_probs(comp: dict, states: int, p_state: np.ndarray) -> np.ndarray:
    """Probabilities of the four intensity classes ``(P, T, 4)`` from either kind of chain (two-state chains split
    their wet state with the amount mixture)."""
    if states == 4:
        return p_state
    w = _softmax(comp["logit"])
    sig = np.exp(comp["logsig"])
    cdf = lambda x: (w * _ndtr((np.log(x) - comp["mu"]) / sig)).sum(-1)                  # noqa: E731
    lo = cdf(EDGES[0])
    c1, c2 = cdf(EDGES[1]), cdf(EDGES[2])
    norm = np.maximum(1 - lo, 1e-9)
    wet = p_state[..., 1]
    light = np.clip((c1 - lo) / norm, 0, 1)
    moderate = np.clip((c2 - c1) / norm, 0, 1)
    heavy = np.clip(1 - light - moderate, 0, 1)
    return np.stack([p_state[..., 0], wet * light, wet * moderate, wet * heavy], -1)


def sample_paths(comp: dict, states: int, samples: int, seed: int = 0) -> np.ndarray:
    """``(samples, P, T)`` rain (mm) per hour: states from the chain, amounts from each state's distribution."""
    rng = np.random.default_rng(seed)
    a, pi = comp["a"], comp["pi"]
    p, t, k = a.shape[:3]
    cum0 = np.cumsum(pi, -1)
    state = (rng.random((samples, p, 1)) > cum0[None]).sum(-1).clip(max=k - 1)
    out = np.zeros((samples, p, t), np.float32)
    idx_p = np.arange(p)[None, :]
    for h in range(t):
        if h > 0:
            rows = a[idx_p, h, state]                                                      # (S, P, K)
            state = (rng.random((samples, p, 1)) > np.cumsum(rows, -1)).sum(-1).clip(max=k - 1)
        wet = state > 0
        if states == 2:
            w = _softmax(comp["logit"][:, h])                                              # (P, M)
            comp_idx = (rng.random((samples, p, 1)) > np.cumsum(w, -1)[None]).sum(-1).clip(max=w.shape[-1] - 1)
            mu = np.take_along_axis(np.broadcast_to(comp["mu"][None, :, h], (samples, p, w.shape[-1])),
                                    comp_idx[..., None], -1)[..., 0]
            s = np.take_along_axis(np.broadcast_to(np.exp(comp["logsig"][None, :, h]), (samples, p, w.shape[-1])),
                                   comp_idx[..., None], -1)[..., 0]
            amount = np.exp(mu + s * rng.standard_normal((samples, p)))
            out[:, :, h] = np.where(wet, np.clip(amount, EDGES[0], 300.0), 0.0)
            continue
        ws = np.clip(state - 1, 0, k - 2)
        mu = np.take_along_axis(np.broadcast_to(comp["amt_mu"][None, :, h], (samples, p, k - 1)), ws[..., None], -1)[..., 0]
        s = np.take_along_axis(np.broadcast_to(comp["amt_sig"][None, :, h], (samples, p, k - 1)), ws[..., None], -1)[..., 0]
        lo = np.log(np.array([b[0] for b in bounds(states)]))[ws]
        hi = np.log(np.array([b[1] for b in bounds(states)]))[ws]
        ua, ub = _ndtr((lo - mu) / s), _ndtr((hi - mu) / s)
        u = ua + (ub - ua) * rng.random((samples, p))
        from scipy.special import ndtri

        z = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
        out[:, :, h] = np.where(wet, np.clip(np.exp(mu + s * z), EDGES[0], 300.0), 0.0)
    return out


__all__ = ["CENTRES", "PRIOR_INITIAL", "PRIOR_TRANSITIONS", "bounds", "class_mask_from_rate", "class_probs",
           "first_entry", "hmm_nll", "marginals_np", "quantile_hour", "rate_classes", "sample_paths", "state_marginals",
           "state_means", "torrential_given_heavy", "truncated_log_prob", "wet_probability"]
