# Copyright 2026 Nathan. Apache-2.0.
"""
What training maximises: the likelihood of what was observed, through what each observer is known to get wrong.

**Timing and intensity (hidden Markov, experimental).** The model's chain writes, for every hour and place, the
chance of moving between dry, light, moderate and heavy rain (two states, dry and wet, in version 1). The truth is a
sequence of classes -- but seen through an instrument. CMORPH misses some light showers, invents a few, and smears a
downpour over an 8 km box; an airport report is close to right about its own spot. So the observed sequence is scored
by the forward algorithm of a hidden Markov model whose emission probabilities are each source's own *confusion
matrix*, learned along with the network (with a weak prior at sensible values). A satellite miss is then not taught
to the model as "it was dry", and a satellite "moderate" over a mountain is not taught as the truth.

**Amount.** In the hours a source reports rain, its rate is scored under the distribution of the reported class
(version 1: the wet state's lognormal mixture), shifted by the source's learned bias (with a term in elevation:
satellite estimates read low over mountains) and widened by its learned noise.

**Thunder and kind of rain.** Where an airport reported, its thunder (``TS``, ``VCTS``, lightning in the remarks)
and its kind of precipitation (drizzle, rain, showers) are scored by the thunder and kind heads.

**Cause.** The weak rain-cause labels are scored with a small weight.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .chain import bounds, class_mask_from_rate, hmm_nll, state_marginals, truncated_log_prob
from .dataset import WET_MM

SOURCES = ("cmorph", "station")
#: Initial confusion matrices P(reported class | true state): rows are true states (dry first).
PRIOR_CONFUSION = {
    2: {"cmorph": ((0.98, 0.02), (0.25, 0.75)), "station": ((0.99, 0.01), (0.05, 0.95))},
    4: {"cmorph": ((0.97, 0.025, 0.004, 0.001), (0.30, 0.55, 0.12, 0.03), (0.10, 0.35, 0.40, 0.15),
                   (0.03, 0.12, 0.30, 0.55)),
        "station": ((0.98, 0.015, 0.004, 0.001), (0.15, 0.70, 0.13, 0.02), (0.05, 0.30, 0.50, 0.15),
                    (0.02, 0.08, 0.30, 0.60))},
}


class TruthSources(nn.Module):
    """Learned error model of each truth source: a confusion matrix over intensity classes, an amount bias (with
    an elevation term) and extra amount noise."""

    def __init__(self, states: int = 4) -> None:
        super().__init__()
        self.states = states
        self.confusion_logits = nn.ParameterDict({s: nn.Parameter(torch.tensor(PRIOR_CONFUSION[states][s]).log())
                                                  for s in SOURCES})
        for s in SOURCES:
            self.register_buffer(f"prior_{s}", torch.tensor(PRIOR_CONFUSION[states][s]).log(), persistent=False)
        self.bias = nn.ParameterDict({s: nn.Parameter(torch.tensor(0.0)) for s in SOURCES})
        self.elevation_bias = nn.ParameterDict({s: nn.Parameter(torch.tensor(0.0)) for s in SOURCES})
        self.log_noise = nn.ParameterDict({s: nn.Parameter(torch.tensor(math.log(0.3))) for s in SOURCES})

    def confusion(self, source: str) -> torch.Tensor:
        return torch.softmax(self.confusion_logits[source], dim=-1)

    def prior_penalty(self) -> torch.Tensor:
        total = 0.0
        for s in SOURCES:
            diff = self.confusion_logits[s] - getattr(self, f"prior_{s}")
            diff = diff - diff.mean(-1, keepdim=True)                     # softmax is shift-invariant per row
            total = total + (diff**2).sum()
        return 0.01 * total

    def describe(self) -> dict:
        out = {}
        with torch.no_grad():
            for s in SOURCES:
                c = self.confusion(s).cpu().numpy()
                out[s] = {"confusion": np.round(c, 3).tolist(), "miss": float(1 - c[1:, 1:].sum(-1).mean()),
                          "false_alarm": float(1 - c[0, 0]), "amount_bias": float(self.bias[s]),
                          "elevation_bias_per_km": float(self.elevation_bias[s]),
                          "amount_noise": float(self.log_noise[s].exp())}
        return out


def mixture_log_prob(y: torch.Tensor, logit: torch.Tensor, mu: torch.Tensor, logsig: torch.Tensor,
                     shift: torch.Tensor | float = 0.0, extra: torch.Tensor | float = 0.0) -> torch.Tensor:
    """log density of rates ``y`` (mm/h, > 0) under the lognormal mixture (shifted and widened per source)."""
    ly = torch.log(y.clamp(min=1e-3))
    sigma = torch.sqrt(torch.exp(2 * logsig) + torch.as_tensor(extra, device=y.device) ** 2)
    z = (ly.unsqueeze(-1) - mu - shift) / sigma
    comp = -0.5 * z**2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi) - ly.unsqueeze(-1)
    return torch.logsumexp(F.log_softmax(logit, -1) + comp, -1)


def mixture_mean(logit: torch.Tensor, mu: torch.Tensor, logsig: torch.Tensor) -> torch.Tensor:
    return (torch.softmax(logit, -1) * torch.exp(mu + 0.5 * torch.exp(2 * logsig))).sum(-1)


def station_masks(codes: torch.Tensor, states: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """From packed report codes ``(B, T, S)``: class masks ``(B, T, S, K)``, thunder ``(B, T, S)`` with its known
    mask, and kind (0 drizzle, 1 rain, 2 showers; -1 unknown)."""
    c = codes.to(torch.int16)
    known = c >= 0
    cc = torch.where(known, c, torch.zeros_like(c))
    intensity = cc & 3
    stated = ((cc >> 5) & 1).bool() & known
    wet = known & (intensity > 0)
    dry = known & ~wet
    m = torch.ones(codes.shape + (states,), dtype=torch.bool, device=codes.device)
    m[..., 0] = ~wet
    if states == 2:
        m[..., 1] = ~dry
    else:
        for k in range(1, states):
            m[..., k] = ~dry & (~stated | ~wet | (intensity == k))
    thunder = ((cc >> 2) & 1).bool()
    kind = ((cc >> 3) & 3).long() - 1
    kind = torch.where(wet & (kind >= 0), kind, torch.full_like(kind, -1))
    return m, thunder, known, kind


class PluviumLoss(nn.Module):
    """The training objective; ``forward(outputs, batch, elevation_km)`` returns ``(loss, metrics)``."""

    def __init__(self, states: int = 4, timing: float = 1.0, amount: float = 0.5, cause: float = 0.1,
                 station: float = 1.0, thunder: float = 0.5, kind: float = 0.2) -> None:
        super().__init__()
        self.states = states
        self.sources = TruthSources(states)
        self.w = {"timing": timing, "amount": amount, "cause": cause, "station": station, "thunder": thunder,
                  "kind": kind}

    def forward(self, out: dict, batch: dict, elevation_km: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        k = self.states
        truth = batch["truth"].to(out["trans"].dtype)                  # (B, T, P) mm/h, NaN missing
        known = torch.isfinite(truth)
        rate = torch.nan_to_num(truth, nan=0.0)
        mask = class_mask_from_rate(truth, k)
        nll, count = hmm_nll(out["trans"], out["init"], mask, self.sources.confusion("cmorph"))
        timing = nll / count.clamp(min=1)
        metrics = {"timing_nll": float(timing.detach())}
        # amounts where the source saw rain
        wet_obs = known & (rate >= WET_MM)
        amount = out["trans"].new_zeros(())
        if wet_obs.any():
            shift = self.sources.bias["cmorph"]
            if elevation_km is not None:
                shift = shift + self.sources.elevation_bias["cmorph"] * elevation_km.view(1, 1, -1)
            noise = self.sources.log_noise["cmorph"].exp()
            if k == 2:
                lp = mixture_log_prob(rate, out["mix_logit"], out["mix_mu"], out["mix_logsig"],
                                      shift.unsqueeze(-1) if torch.is_tensor(shift) and shift.dim() else shift, noise)
            else:
                lp = torch.zeros_like(rate)
                cls = mask.float().argmax(-1)
                for j, (lo, hi) in enumerate(bounds(k)):
                    sel = wet_obs & (cls == j + 1)
                    if sel.any():
                        sig = torch.sqrt(out["amt_sig"][..., j] ** 2 + noise**2)
                        lp_j = truncated_log_prob(rate.clamp(min=lo), out["amt_mu"][..., j] + shift, sig, lo, hi)
                        lp = torch.where(sel, lp_j, lp)
            amount = -(lp * wet_obs).sum() / wet_obs.sum()
            metrics["amount_nll"] = float(amount.detach())
        # cause
        cause = out["trans"].new_zeros(())
        if "regime" in batch:
            labels = batch["regime"]
            logits = out["cause"]
            if labels.shape[1] != logits.shape[1]:
                labels = labels[:, -logits.shape[1]:]
            cause = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1).long())
            metrics["cause_ce"] = float(cause.detach())
        loss = self.w["timing"] * timing + self.w["amount"] * amount + self.w["cause"] * cause
        # airports
        if "station_trans" in out and "station_truth" in batch:
            m_s, thunder, s_known, kind = station_masks(batch["station_truth"], k)
            if s_known.any():
                s_nll, s_count = hmm_nll(out["station_trans"], out["station_init"], m_s, self.sources.confusion("station"))
                station = s_nll / s_count.clamp(min=1)
                loss = loss + self.w["station"] * station
                metrics["station_nll"] = float(station.detach())
                if "station_thunder" in out:
                    bce = F.binary_cross_entropy_with_logits(out["station_thunder"], thunder.float(), reduction="none")
                    th = (bce * s_known).sum() / s_known.sum()
                    loss = loss + self.w["thunder"] * th
                    metrics["thunder_bce"] = float(th.detach())
                if "station_kind" in out and (kind >= 0).any():
                    sel = kind >= 0
                    ce = F.cross_entropy(out["station_kind"][sel], kind[sel])
                    loss = loss + self.w["kind"] * ce
                    metrics["kind_ce"] = float(ce.detach())
        loss = loss + self.sources.prior_penalty()
        with torch.no_grad():
            p = 1.0 - state_marginals(out["trans"], out["init"])[..., 0]
            wet = wet_obs.to(p.dtype)
            metrics["brier"] = float((((p - wet) ** 2) * known).sum() / known.sum().clamp(min=1))
            metrics["wet_rate"] = float((wet * known).sum() / known.sum().clamp(min=1))
        metrics["loss"] = float(loss.detach())
        return loss, metrics


__all__ = ["PRIOR_CONFUSION", "PluviumLoss", "SOURCES", "TruthSources", "mixture_log_prob", "mixture_mean",
           "station_masks"]
