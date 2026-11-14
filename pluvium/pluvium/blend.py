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


class BlendBag(nn.Module):
    """Several BlendNets trained from different seeds on different subsamples; their chances are averaged (an
    average of decreasing curves is decreasing, so the chances still never contradict each other)."""

    def __init__(self, nets: list) -> None:
        super().__init__()
        self.nets = nn.ModuleList(nets)

    @property
    def mu(self) -> torch.Tensor:                        # the input size, as for a single net
        return self.nets[0].mu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([net(x) for net in self.nets]).mean(0)


def fit(x: np.ndarray, y: np.ndarray, seed: int = 0, epochs: int = 6, max_rows: int = 400_000, batch: int = 2048,
        lr: float = 3e-3, width: int = 48, min_steps: int = 1200, members: int = 1, groups: list | None = None,
        drop: float = 0.0) -> BlendNet | BlendBag:
    """Train on rows ``x`` (N, F) and observed daily rain ``y`` (N,) mm; rows with NaN rain are dropped. With
    ``members`` > 1, a bag of that many nets (seeds ``seed``, ``seed + 1`` ...), each on its own subsample.

    ``groups`` (column indices of each system, its ``present`` flag included) and ``drop`` hide each system from each
    training row with that chance, as if it had not arrived: the blend then cannot lean on one combination of systems
    and still forecasts well when one is missing."""
    one = lambda k: _fit_one(x, y, seed + k, epochs, max_rows, batch, lr, width, min_steps, groups, drop)   # noqa: E731
    return BlendBag([one(k) for k in range(members)]) if members > 1 else one(0)


def system_groups(systems: list[str], ensembles: set) -> list[np.ndarray]:
    """Column indices of each system's features in :func:`assemble`'s rows (for ``fit(groups=...)``)."""
    names = feature_names(systems, ensembles)
    return [np.array([i for i, n in enumerate(names) if n.split(":")[0] == s]) for s in systems]


def _fit_one(x: np.ndarray, y: np.ndarray, seed: int, epochs: int, max_rows: int, batch: int, lr: float, width: int,
             min_steps: int, groups: list | None = None, drop: float = 0.0) -> BlendNet:
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
            xb = xt[idx]
            if groups and drop > 0:                                  # hide whole systems, as if they had not arrived
                hide = torch.rand(xb.shape[0], len(groups)) < drop
                keep = torch.ones_like(xb)
                for g, cols in enumerate(groups):
                    keep[:, torch.as_tensor(cols)] = (~hide[:, g:g + 1]).float()
                xb = xb * keep
            loss = nn.functional.binary_cross_entropy_with_logits(net.logits(xb), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            done += 1
    net.eval()
    return net


class StackPool(nn.Module):
    """Pluvium Blend's last step: a weighted logarithmic pool of calibrated forecasts (each system post-processed on its
    own, and networks over several). Each component has a weight per band of thresholds (light, moderate and heavy
    rain), changing with lead; the pool averages the present components' log-odds with those weights, sharpens by
    their total, and adds an offset per threshold that also changes with lead. About 70 parameters, fitted on
    forecasts the components made for weeks they were not trained on: it cannot overfit, and it can always fall back
    to its best component."""

    #: the first threshold index of each band (0.2-3 mm, 5-25 mm, 35-150 mm)
    BANDS = (0, 5, 11)

    def __init__(self, n_comp: int, n_thr: int = len(THRESHOLDS), bands: tuple = BANDS) -> None:
        super().__init__()
        band = np.searchsorted(np.asarray(bands), np.arange(n_thr), side="right") - 1
        self.register_buffer("band", torch.as_tensor(band, dtype=torch.long))
        start = float(np.log(np.expm1(1.0 / n_comp)))                                     # weights start at 1/C
        self.a = nn.Parameter(torch.full((len(bands), n_comp), start))
        self.d = nn.Parameter(torch.zeros(len(bands), n_comp))
        self.b = nn.Parameter(torch.zeros(n_thr))
        self.e = nn.Parameter(torch.zeros(n_thr))

    def weights(self, lead: torch.Tensor) -> torch.Tensor:
        """(N, K, C): each component's weight at each threshold."""
        return nn.functional.softplus(self.a[self.band][None] + self.d[self.band][None] * lead[:, None, None])

    def logits(self, z: torch.Tensor, present: torch.Tensor, lead: torch.Tensor) -> torch.Tensor:
        w = self.weights(lead)
        wp = w * present[:, None, :]
        zt = z.transpose(1, 2)                                                              # (N, K, C)
        mean = (wp * zt).sum(-1) / wp.sum(-1).clamp_min(1e-6)
        return w.sum(-1) * mean + self.b[None] + self.e[None] * lead[:, None]

    def forward(self, z: torch.Tensor, present: torch.Tensor, lead: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(z, present, lead))


def to_logits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).astype(np.float32)


def fit_stack(z: np.ndarray, present: np.ndarray, lead: np.ndarray, y: np.ndarray, iters: int = 150) -> StackPool:
    """``z`` (N, C, K) components' log-odds of each threshold, ``present`` (N, C), ``lead`` (N,) days, ``y`` (N,) mm."""
    keep = np.isfinite(y)
    zt = torch.from_numpy(np.ascontiguousarray(z[keep], np.float32))
    pt = torch.from_numpy(present[keep].astype(np.float32))
    lt = torch.from_numpy((lead[keep] / 14.0).astype(np.float32))
    yt = torch.from_numpy((y[keep, None] >= THRESHOLDS[None]).astype(np.float32))
    pool = StackPool(z.shape[1], z.shape[2])
    opt = torch.optim.LBFGS(pool.parameters(), lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(pool.logits(zt, pt, lt), yt)
        loss.backward()
        return loss

    opt.step(closure)
    pool.eval()
    return pool


@torch.no_grad()
def predict_stack(pool: StackPool, z: np.ndarray, present: np.ndarray, lead) -> np.ndarray:
    """P(rain >= THRESHOLDS) from the components; never more likely at a higher threshold."""
    n = z.shape[0]
    lead = np.broadcast_to(np.asarray(lead, np.float32), (n,)) / 14.0
    p = pool(torch.from_numpy(np.ascontiguousarray(z, np.float32)), torch.from_numpy(present.astype(np.float32)),
             torch.from_numpy(np.ascontiguousarray(lead, np.float32))).numpy()
    return np.minimum.accumulate(p, axis=1)


@torch.no_grad()
def predict(net: BlendNet | BlendBag, x: np.ndarray, chunk: int = 200_000) -> np.ndarray:
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


__all__ = ["CLIM_FEATURES", "ENSEMBLE_FEATURES", "LEVELS", "SCORED", "SINGLE_FEATURES", "THRESHOLDS", "BlendBag",
           "BlendNet", "StackPool", "assemble", "exceed", "feature_names", "fit", "fit_stack", "predict",
           "predict_stack", "quantiles", "system_features", "system_groups", "to_logits"]
