# Copyright 2026 Nathan. Apache-2.0.
"""
Distillation: a small Pluvium that forecasts like the large one.

Pluvium-500 needs a GPU. For phones, laptops and offline use, a small student is trained on the same examples to
match the teacher's hourly intensity-state probabilities (Kullback-Leibler divergence of the chain marginals, Hinton
et al. 2015), together with its own likelihood of the observations, so it learns from both the truth and the
teacher's smoother, better-calibrated beliefs.
"""

from __future__ import annotations

import torch

from ..chain import state_marginals


def distill_step(student, teacher, loss_fn, batch: dict, elevation_km, optimizer, alpha: float = 0.5,
                 grad_clip: float = 1.0) -> dict:
    """One optimisation step of the student; returns metrics."""
    with torch.no_grad():
        t_out = teacher(batch)
        t_marg = state_marginals(t_out["trans"], t_out["init"]).clamp(min=1e-6)
    s_out = student(batch)
    data_loss, metrics = loss_fn(s_out, batch, elevation_km)
    s_marg = state_marginals(s_out["trans"], s_out["init"]).clamp(min=1e-6)
    if s_marg.shape[-1] != t_marg.shape[-1]:              # a two-state student learns the teacher's wet chance
        t_marg = torch.stack([t_marg[..., 0], 1 - t_marg[..., 0]], -1)
    kl = (t_marg * (t_marg.log() - s_marg.log())).sum(-1).mean()
    loss = (1 - alpha) * data_loss + alpha * kl
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
    optimizer.step()
    return {**metrics, "kl_to_teacher": float(kl.detach()), "distill_loss": float(loss.detach())}


__all__ = ["distill_step"]
