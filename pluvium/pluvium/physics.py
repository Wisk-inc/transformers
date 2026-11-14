# Copyright 2026 Nathan. Apache-2.0.
"""
Physics the network is given rather than asked to discover.

**Orographic rain as a trainable layer (experimental).** Smith and Barstad (2004) solved the linear problem of
moist air forced over terrain: the rain a mountain makes is the terrain's spectrum times a transfer function of the
wind, the stability and two time scales (cloud water forming, then falling out)::

    P^(k, l) = C_w i sigma h^(k, l) / [(1 - i m H_w)(1 + i sigma tau_c)(1 + i sigma tau_f)],  sigma = U k + V l

which puts rain on windward slopes, a shadow in the lee, and carries it downwind by the drift the time scales
allow. Here it runs per hour of the forecast, on 2 km tiles over every island with relief, driven by that hour's
low-level wind and water vapour, and its time scales, vapour scale height, stability and gain are *parameters
trained with the rest of the network*. The linear theory is from 2004; making it a differentiable layer inside a
neural forecaster, re-solved for every forecast hour, is what is new here, and it is untested on the Caribbean
until a training run measures it (``ablate_orography`` switches it off for exactly that comparison).

The layer's output is a signed rate (mm/h): positive upwind of ridges, negative in their lee. The network reads
it as an input feature at every point, and adds a learned share of its positive part to the rain it predicts.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .domain import KM_PER_DEG, OroTile


class OrographicRain(nn.Module):
    """
    Smith-Barstad linear upslope rain on a set of tiles, with learned physics.

    ``forward(u, v, pw)`` takes the tile-mean low-level wind (m/s, east and north) and precipitable water (kg/m2),
    each ``(B, F, T)`` for ``B`` samples, ``F`` frames and ``T`` tiles, and returns the rate field
    ``(B, F, T, S, S)`` in mm/h. :meth:`sample` reads it at points.
    """

    def __init__(self, tiles: list[OroTile], tau_c: float = 1000.0, tau_f: float = 1000.0, vapour_height: float = 2500.0,
                 stability: float = 0.005, gain: float = 1.0) -> None:
        super().__init__()
        self.num_tiles = len(tiles)
        size = tiles[0].size if tiles else 8
        dx_m = (tiles[0].dx_km if tiles else 2.0) * 1000.0
        self.size, self.dx_m = size, dx_m
        if tiles:
            h = torch.tensor(np.stack([t.elevation for t in tiles]), dtype=torch.float32)
        else:
            h = torch.zeros(0, size, size)
        self.register_buffer("h_hat", torch.fft.fft2(h), persistent=False)
        freq = torch.fft.fftfreq(size, d=dx_m) * 2 * math.pi
        self.register_buffer("k", freq.view(1, size), persistent=False)  # east (columns)
        self.register_buffer("l", freq.view(size, 1), persistent=False)  # north (rows)
        self.log_tau_c = nn.Parameter(torch.tensor(math.log(tau_c)))
        self.log_tau_f = nn.Parameter(torch.tensor(math.log(tau_f)))
        self.log_vapour_height = nn.Parameter(torch.tensor(math.log(vapour_height)))
        self.log_stability = nn.Parameter(torch.tensor(math.log(stability)))
        self.moist_stability = nn.Parameter(torch.tensor(0.0))  # moister air is less stable: N_m * exp(-a (pw - 45)/10)
        self.log_gain = nn.Parameter(torch.tensor(math.log(gain)))
        self.tiles = tiles

    def forward(self, u: torch.Tensor, v: torch.Tensor, pw: torch.Tensor) -> torch.Tensor:
        if self.num_tiles == 0:
            return u.new_zeros(*u.shape, 1, 1)
        u, v, pw = (x.float()[..., None, None] for x in (u, v, pw))   # (B, F, T, 1, 1)
        sigma = u * self.k + v * self.l                                 # (B, F, T, S, S) rad/s
        kappa = torch.sqrt(self.k**2 + self.l**2)
        n_m = self.log_stability.exp() * torch.exp(-self.moist_stability * (pw - 45.0) / 10.0)
        s2 = sigma**2
        ratio = (n_m**2 - s2) / (s2 + 1e-20)
        propagating = ratio > 0
        root = torch.sqrt(ratio.abs() + 1e-30) * kappa
        h_w = self.log_vapour_height.exp()
        # (1 - i m H): m real (sign of sigma) where waves propagate, m = i |m| where they are evanescent
        m_real = torch.where(propagating, torch.sign(sigma) * root, torch.zeros_like(root))
        m_imag = torch.where(propagating, torch.zeros_like(root), root)
        vertical = torch.complex(1.0 + m_imag * h_w, -m_real * h_w)
        tau_c, tau_f = self.log_tau_c.exp(), self.log_tau_f.exp()
        conversion = torch.complex(torch.ones_like(sigma), sigma * tau_c)
        fallout = torch.complex(torch.ones_like(sigma), sigma * tau_f)
        c_w = self.log_gain.exp() * 0.5 * pw / h_w                      # kg m^-3: uplift sensitivity
        numerator = torch.complex(torch.zeros_like(sigma), sigma) * c_w * self.h_hat
        spectrum = numerator / (vertical * conversion * fallout)
        spectrum = torch.where(s2 > 1e-24, spectrum, torch.zeros_like(spectrum))
        return torch.fft.ifft2(spectrum).real * 3600.0                  # kg m^-2 s^-1 -> mm/h

    # ------------------------------------------------------------------------------------- reading --
    def locate(self, lat: np.ndarray, lon: np.ndarray) -> dict:
        """Which tile owns each point and where in it: ``{"tile": (N,) int (-1 = none), "xy": (N, 2) in [-1, 1]}``,
        the normalised coordinates :func:`torch.nn.functional.grid_sample` wants (x east, y north)."""
        lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        tile = np.full(lat.shape, -1, np.int64)
        xy = np.zeros(lat.shape + (2,), np.float32)
        for t, spec in enumerate(self.tiles):
            own = spec.owns(lat, lon) & (tile < 0)
            if not own.any():
                continue
            half = (spec.size - 1) / 2.0
            col = (lon[own] - spec.lon0) * KM_PER_DEG * math.cos(math.radians(spec.lat0)) / spec.dx_km + half
            row = (lat[own] - spec.lat0) * KM_PER_DEG / spec.dx_km + half
            xy[own, 0] = 2.0 * col / (spec.size - 1) - 1.0
            xy[own, 1] = 2.0 * row / (spec.size - 1) - 1.0
            tile[own] = t
        return {"tile": tile, "xy": xy}

    def sample(self, field: torch.Tensor, where: dict) -> torch.Tensor:
        """``field (B, F, T, S, S)`` at located points -> ``(B, F, N)`` (zero where no tile owns the point)."""
        b, f = field.shape[:2]
        tile = torch.as_tensor(where["tile"], device=field.device)
        xy = torch.as_tensor(where["xy"], device=field.device, dtype=field.dtype)
        out = field.new_zeros(b, f, tile.numel())
        flat = field.reshape(b * f, field.shape[2], field.shape[3], field.shape[4])
        for t in torch.unique(tile).tolist():
            if t < 0:
                continue
            idx = torch.nonzero(tile == t).squeeze(-1)
            grid = xy[idx].view(1, 1, -1, 2).expand(b * f, -1, -1, -1)
            values = F.grid_sample(flat[:, t : t + 1], grid, mode="bilinear", align_corners=True)  # (BF, 1, 1, n)
            out[:, :, idx] = values.view(b, f, -1)
        return out

    def rate_at(self, u, v, pw, where: dict, chunk: int = 8) -> torch.Tensor:
        """Solve and read in chunks of frames (memory: one chunk of ``(B, chunk, T, S, S)`` complex fields at a time,
        recomputed in the backward pass)."""
        parts = []
        for start in range(0, u.shape[1], chunk):
            sl = slice(start, start + chunk)

            def solve(a, b_, c):
                return self.sample(self.forward(a, b_, c), where)

            if self.training and torch.is_grad_enabled():
                parts.append(torch.utils.checkpoint.checkpoint(solve, u[:, sl], v[:, sl], pw[:, sl], use_reentrant=False))
            else:
                parts.append(solve(u[:, sl], v[:, sl], pw[:, sl]))
        return torch.cat(parts, dim=1)

    def rates_at(self, u, v, pw, wheres: list, chunk: int = 8) -> list:
        """Like :meth:`rate_at` for several point sets at once: each chunk of frames is solved once and read at
        every set (the solve dominates the cost)."""
        parts = [[] for _ in wheres]
        for start in range(0, u.shape[1], chunk):
            sl = slice(start, start + chunk)

            def solve(a, b_, c):
                field = self.forward(a, b_, c)
                return tuple(self.sample(field, w) for w in wheres)

            if self.training and torch.is_grad_enabled():
                got = torch.utils.checkpoint.checkpoint(solve, u[:, sl], v[:, sl], pw[:, sl], use_reentrant=False)
            else:
                got = solve(u[:, sl], v[:, sl], pw[:, sl])
            for i, g in enumerate(got):
                parts[i].append(g)
        return [torch.cat(p, dim=1) for p in parts]

    def describe(self) -> dict:
        n_m = float(self.log_stability.exp())
        return {"tau_c_s": float(self.log_tau_c.exp()), "tau_f_s": float(self.log_tau_f.exp()),
                "vapour_height_m": float(self.log_vapour_height.exp()), "stability_per_s": n_m,
                "moist_stability": float(self.moist_stability), "gain": float(self.log_gain.exp()), "tiles": self.num_tiles}


def gaussian_hill(size: int = 128, dx_km: float = 2.0, height: float = 1000.0, half_width_km: float = 20.0,
                  lat0: float = 18.0, lon0: float = -77.0) -> OroTile:
    """A circular Gaussian mountain on one tile (tests and the documentation's sanity check)."""
    k = (np.arange(size) - (size - 1) / 2.0) * dx_km
    yy, xx = np.meshgrid(k, k, indexing="ij")
    h = height * np.exp(-(xx**2 + yy**2) / (2 * half_width_km**2))
    return OroTile(lat0, lon0, size * dx_km / KM_PER_DEG * 0.7, size, dx_km, h.astype(np.float32))


__all__ = ["OrographicRain", "gaussian_hill"]
