# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Frechet distance between two sets of embeddings, ``|mu_a - mu_b|^2 + tr(Sa + Sb - 2 (Sa Sb)^(1/2))``, as FAD
(Kilgour et al., 2019) computes it on VGGish frames. The default embedding here is the mel cepstrum of each 5 ms
frame ("FAD-MFCC"); pass ``embed`` for a learned one. Distances are comparable for one embedding only.
"""

from __future__ import annotations

import torch

from .mfcc import mel_cepstrum


def _sqrtm_psd(m: torch.Tensor) -> torch.Tensor:
    vals, vecs = torch.linalg.eigh((m + m.T) / 2)
    return vecs @ torch.diag(vals.clamp(min=0).sqrt()) @ vecs.T


def frechet_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Between ``a (N, D)`` and ``b (M, D)`` (rows are embeddings)."""
    a, b = a.double(), b.double()
    mu_a, mu_b = a.mean(0), b.mean(0)
    sa, sb = torch.cov(a.T).reshape(a.shape[1], a.shape[1]), torch.cov(b.T).reshape(b.shape[1], b.shape[1])
    root_a = _sqrtm_psd(sa)
    cross = _sqrtm_psd(root_a @ sb @ root_a)            # (Sa Sb)^(1/2) has the trace of this symmetric form
    return float((mu_a - mu_b).pow(2).sum() + torch.trace(sa) + torch.trace(sb) - 2 * torch.trace(cross))


def frame_embeddings(waves, sample_rate: int, embed=None, per_clip: int = 400) -> torch.Tensor:
    rows = []
    for w in waves:
        e = embed(w, sample_rate) if embed is not None else mel_cepstrum(w, sample_rate, n_coeffs=21)[:, 1:]
        if e.dim() == 1:
            e = e[None]
        if e.shape[0] > per_clip:
            e = e[torch.linspace(0, e.shape[0] - 1, per_clip).long()]
        rows.append(e.float())
    return torch.cat(rows) if rows else torch.zeros(0, 1)


def fad(generated, reference, sample_rate: int, embed=None) -> float:
    gen = frame_embeddings(generated, sample_rate, embed)
    return frechet_distance(gen, frame_embeddings(reference, sample_rate, embed))


__all__ = ["fad", "frame_embeddings", "frechet_distance"]
