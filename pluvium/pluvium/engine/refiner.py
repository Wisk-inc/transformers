# Copyright 2026 Nathan. Apache-2.0.
"""
Local attention among decoded points.

The decoder reads the mesh (about 11 km apart over land) from each town independently. Two villages a few kilometres
apart in the same valley can then get different start hours for the same shower. The refiner lets each decoded
point attend to its ``k`` nearest decoded neighbours (great-circle distance, with a learned distance bias per head),
so the field is coherent, and so contrasts finer than the mesh -- windward and lee sides of a ridge, a coast and the
hills behind it -- can be sharpened from the neighbours' own terrain and context.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def knn_on_sphere(lat: np.ndarray, lon: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """``(P, k)`` indices of each point's nearest points (itself first) and their distances in km."""
    from scipy.spatial import cKDTree

    la, lo = np.radians(np.asarray(lat, np.float64)), np.radians(np.asarray(lon, np.float64))
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    k = min(k, len(xyz))
    dist, idx = cKDTree(xyz).query(xyz, k=k)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    return idx.astype(np.int64), (dist * 6371.0).astype(np.float32)


class PointRefiner(nn.Module):
    """``layers`` rounds of neighbour attention over decoded points ``h (B, F, P, d)``."""

    def __init__(self, hidden: int, heads: int = 4, neighbours: int = 8, layers: int = 1, scale_km: float = 30.0) -> None:
        super().__init__()
        self.heads, self.k, self.scale = heads, neighbours, scale_km
        self.head_dim = hidden // heads
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.ModuleDict({
                "norm": nn.LayerNorm(hidden), "q": nn.Linear(hidden, hidden), "kv": nn.Linear(hidden, 2 * hidden),
                "out": nn.Linear(hidden, hidden), "bias": nn.Linear(2, heads), "ffn_norm": nn.LayerNorm(hidden),
                "ffn": nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))}))
        for layer in self.layers:
            nn.init.zeros_(layer["out"].weight)
            nn.init.zeros_(layer["out"].bias)

    def forward(self, h: torch.Tensor, neighbours: torch.Tensor, distance_km: torch.Tensor) -> torch.Tensor:
        b, f, p, d = h.shape
        idx = neighbours[:, : self.k]                                          # (P, k)
        dist = distance_km[:, : self.k].to(h.dtype) / self.scale
        bias_in = torch.stack([dist, torch.exp(-dist)], -1)                    # (P, k, 2)
        for layer in self.layers:
            x = layer["norm"](h)
            q = layer["q"](x).view(b, f, p, self.heads, self.head_dim)
            kv = layer["kv"](x)[:, :, idx]                                     # (B, F, P, k, 2d)
            kk, vv = kv.view(b, f, p, idx.shape[1], 2, self.heads, self.head_dim).unbind(4)
            score = torch.einsum("bfphc,bfpkhc->bfphk", q, kk) / math.sqrt(self.head_dim)
            score = score + layer["bias"](bias_in).permute(0, 2, 1)[None, None]   # (P, h, k)
            att = F.softmax(score, dim=-1)
            mixed = torch.einsum("bfphk,bfpkhc->bfphc", att, vv).reshape(b, f, p, d)
            h = h + layer["out"](mixed)
            h = h + layer["ffn"](layer["ffn_norm"](h))
        return h


__all__ = ["PointRefiner", "knn_on_sphere"]
