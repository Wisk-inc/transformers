# Copyright 2026 Nathan. Apache-2.0.
"""
Pluvium Blend: one calibrated forecast from every operational system and Pluvium's own network.

For each place and day ahead, the blend reads what each system says (its members' mean and spread, its chances of 1,
10 and 25 mm, and the same over everything within half a degree) together with the climatology of that place and
season, and writes the whole distribution of the day's rain: the chance of at least 0.2, 0.5, 1, 2 ... 150 mm. The
chances fall with the threshold by construction (cumulative softplus steps), so they never contradict each other.

It is trained on the sum of the Brier scores at those thresholds, each a proper score. Integrated over thresholds,
the Brier score is the CRPS, so the same training serves yes/no questions (Brier, CSI), amounts (RMSE, ACC, from the
distribution's mean) and the full distribution (CRPS, from quantiles). The same small network trained on one system's
inputs alone is that system's post-processed forecast: a fair rival, since every system then gets identical
treatment.

Experimental: learned multi-model blending in the style of the National Blend of Models and ecPoint, with one
distributional network instead of per-threshold regressions.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

#: thresholds of the predicted distribution, mm per day
THRESHOLDS = np.array([0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 35.0, 50.0, 75.0, 100.0, 150.0],
                      np.float32)
#: quantile levels used for CRPS and the mean
LEVELS = (np.arange(100) + 0.5) / 100
SCORED = (1.0, 10.0, 25.0, 50.0)
ENSEMBLE_FEATURES = ("mean", "spread", "p1", "p10", "p25", "nb_mean", "nb_p1", "nb_p10")
SINGLE_FEATURES = ("mean", "nb_mean", "nb_p1", "nb_p10")
CLIM_FEATURES = ("clim_p1", "clim_p10", "clim_p25", "clim_mean")


def system_features(members: np.ndarray, nb_members: np.ndarray | None = None) -> dict:
    """Features of one system at cells: ``members`` (M, N) daily rain at the cells; ``nb_members`` (M, K, N) the
    same at the K grid points within half a degree (or None). NaN members are ignored."""
    import warnings

    m = np.asarray(members, np.float32)
    valid = np.isfinite(m)
    n_valid = np.maximum(valid.sum(0), 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)            # cells where every member is missing stay NaN
        out = {"mean": np.log1p(np.nanmean(np.maximum(m, 0), 0))}
        if m.shape[0] > 1:
            out["spread"] = np.nanstd(np.log1p(np.maximum(m, 0)), 0)
            for t in (1, 10, 25):
                out[f"p{t}"] = ((m >= t) & valid).sum(0) / n_valid
        nb = m[:, None] if nb_members is None else np.asarray(nb_members, np.float32)
        nb_valid = np.isfinite(nb)
        out["nb_mean"] = np.log1p(np.nanmean(np.maximum(nb, 0), (0, 1)))
        for t in (1, 10):
            out[f"nb_p{t}"] = ((nb >= t) & nb_valid).sum((0, 1)) / np.maximum(nb_valid.sum((0, 1)), 1)
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


def feature_names(systems: list[str], ensembles: set) -> list[str]:
    names = []
    for s in systems:
        names += [f"{s}:{f}" for f in (ENSEMBLE_FEATURES if s in ensembles else SINGLE_FEATURES)] + [f"{s}:present"]
    return names + list(CLIM_FEATURES) + ["lead"]


def assemble(per_system: dict, systems: list[str], ensembles: set, clim: dict, lead: float, n: int) -> np.ndarray:
    """One block of rows ``(n, features)``: each system's features (zeros and ``present = 0`` where it is missing),
    the climatology and the lead (days / 14)."""
    cols = []
    for s in systems:
        feats = per_system.get(s)
        keys = ENSEMBLE_FEATURES if s in ensembles else SINGLE_FEATURES
        present = np.zeros(n, np.float32)
        if feats is not None:
            present = np.isfinite(feats["mean"]).astype(np.float32)
        for k in keys:
            v = feats[k] if feats is not None and k in feats else np.zeros(n, np.float32)
            cols.append(np.where(present > 0, np.nan_to_num(v), 0.0))
        cols.append(present)
    for k in CLIM_FEATURES:
        cols.append(np.nan_to_num(clim[k]).astype(np.float32))
    cols.append(np.full(n, lead / 14.0, np.float32))
    return np.stack(cols, 1).astype(np.float32)


class BlendNet(nn.Module):
    """Features -> P(rain >= each threshold), decreasing in the threshold by construction."""

    def __init__(self, n_in: int, width: int = 48, n_out: int = len(THRESHOLDS), dropout: float = 0.05) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, width),
                                  nn.SiLU(), nn.Linear(width, n_out))
        self.register_buffer("mu", torch.zeros(n_in))
        self.register_buffer("sd", torch.ones(n_in))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        z = self.body((x - self.mu) / self.sd)
        first = z[:, :1]
        steps = nn.functional.softplus(z[:, 1:]) + 1e-3
        return torch.cat([first, first - torch.cumsum(steps, 1)], 1)    # decreasing logits -> decreasing chances

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(x))


def fit(x: np.ndarray, y: np.ndarray, seed: int = 0, epochs: int = 6, max_rows: int = 400_000, batch: int = 2048,
        lr: float = 3e-3, width: int = 48, min_steps: int = 1200) -> BlendNet:
    """Train on rows ``x`` (N, F) and observed daily rain ``y`` (N,) mm; rows with NaN rain are dropped."""
    rng = np.random.default_rng(seed)
    keep = np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.shape[0] > max_rows:
        pick = rng.choice(x.shape[0], max_rows, replace=False)
        x, y = x[pick], y[pick]
    torch.manual_seed(seed)
    net = BlendNet(x.shape[1], width)
    net.mu.copy_(torch.from_numpy(x.mean(0)))
    net.sd.copy_(torch.from_numpy(np.maximum(x.std(0), 1e-3)))
    xt = torch.from_numpy(x)
    yt = torch.from_numpy((y[:, None] >= THRESHOLDS[None]).astype(np.float32))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    per_epoch = int(np.ceil(x.shape[0] / batch))
    steps = max(min_steps, epochs * per_epoch)                  # small samples get more passes, not fewer steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    net.train()
    done = 0
    while done < steps:
        order = torch.randperm(x.shape[0])
        for i in range(0, x.shape[0], batch):
            if done >= steps:
                break
            idx = order[i:i + batch]
            loss = nn.functional.binary_cross_entropy_with_logits(net.logits(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            done += 1
    net.eval()
    return net


@torch.no_grad()
def predict(net: BlendNet, x: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """P(rain >= THRESHOLDS) for rows ``x`` -> (N, len(THRESHOLDS))."""
    out = [net(torch.from_numpy(x[i:i + chunk])).numpy() for i in range(0, x.shape[0], chunk)]
    return np.concatenate(out, 0) if out else np.zeros((0, len(THRESHOLDS)), np.float32)


def exceed(p: np.ndarray, threshold: float) -> np.ndarray:
    """The chance of at least ``threshold`` mm from the predicted chances (interpolated in log-amount)."""
    k = int(np.searchsorted(THRESHOLDS, threshold))
    if k < len(THRESHOLDS) and THRESHOLDS[k] == threshold:
        return p[:, k]
    lo, hi = max(0, k - 1), min(len(THRESHOLDS) - 1, k)
    w = (np.log(threshold) - np.log(THRESHOLDS[lo])) / max(np.log(THRESHOLDS[hi]) - np.log(THRESHOLDS[lo]), 1e-9)
    return p[:, lo] * (1 - w) + p[:, hi] * w


def quantiles(p: np.ndarray, levels=LEVELS) -> np.ndarray:
    """Quantiles (N, L) of the predicted distribution: zero below the chance of rain, linear in log-amount between
    thresholds, an exponential tail above the last."""
    n = p.shape[0]
    cdf = 1.0 - p                                    # P(rain < threshold), increasing
    logt = np.log(THRESHOLDS)
    out = np.zeros((n, len(levels)), np.float32)
    for j, q in enumerate(levels):
        k = (cdf < q).sum(1)                         # thresholds the quantile lies above
        inside = (k > 0) & (k < len(THRESHOLDS))
        kk = np.clip(k, 1, len(THRESHOLDS) - 1)
        c0, c1 = cdf[np.arange(n), kk - 1], cdf[np.arange(n), kk]
        w = np.clip((q - c0) / np.maximum(c1 - c0, 1e-9), 0, 1)
        val = np.exp(logt[kk - 1] * (1 - w) + logt[kk] * w)
        tail = THRESHOLDS[-1] * (1.0 + np.log(np.maximum(1 - cdf[:, -1], 1e-9) / np.maximum(1 - q, 1e-9)))
        out[:, j] = np.where(k == 0, 0.0, np.where(inside, val, np.maximum(tail, THRESHOLDS[-1])))
    return out


__all__ = ["CLIM_FEATURES", "ENSEMBLE_FEATURES", "LEVELS", "SCORED", "SINGLE_FEATURES", "THRESHOLDS", "BlendNet",
           "assemble", "exceed", "feature_names", "fit", "predict", "quantiles", "system_features"]
