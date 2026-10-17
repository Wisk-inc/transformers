# Copyright 2026 Nathan. Apache-2.0.
"""
Breaking the 1.5-degree ceiling: decoding NatureV1's Fibonacci mesh onto a 0.25-degree global grid.

NatureV1's public weights read and write a 1.5-degree grid (~165 km), which caps how sharply a landfall point,
a rain band or a coastal wind maximum can be placed, whatever the model "knows". But the model's knowledge does
not live on the 1.5-degree grid -- it lives on the latent mesh, and the decoder is a cross-attention from any set
of query points to that mesh (:class:`ihelix.CrossAttention` through ``model.link``). Nothing ties the query
points to 1.5 degrees. So:

* :func:`highres_grid` builds the 0.25-degree (1440 x 721) query grid as a read-only source grid;
* :class:`HighResHead` turns decoded latent features at every 0.25-degree point into absolute fields, as a
  **residual on the 1.5-degree forecast upsampled bilinearly**, conditioned on what a coarse model cannot see but
  a fine grid can: 0.25-degree orography and land-sea mask (the coast a landfall crosses, the mountains that
  wring out rain) and the latitude. It starts at exactly zero, so an untrained head returns the upsampled
  forecast;
* :func:`decode_highres` evaluates it in chunks of points (a million query points with 24 neighbours each do
  not fit in one attention call), streaming results to the CPU;
* :func:`train_highres` fits the head (and fine-tunes the decoder) against ERA5 at 0.25 degrees from WeatherBench
  2 with the backbone frozen: the 1.5-degree analysis goes in, the 0.25-degree analysis of the same time comes
  out -- learned downscaling that then applies to every forecast step.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

ERA5_0P25 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
HIGHRES_FIELDS = ("10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "2m_temperature",
                  "total_precipitation_6hr")


def highres_coords(resolution: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(-90.0, 90.0, int(round(180.0 / resolution)) + 1)
    lon = np.arange(int(round(360.0 / resolution))) * resolution
    return lat, lon


def highres_grid(resolution: float = 0.25, rows: slice | None = None):
    """A read-only :class:`ihelix.FieldGrid` of the 0.25-degree grid (or a band of its rows)."""
    from ihelix import FieldGrid, Geometry

    from .corpora import latlon_to_coords
    from .era5 import equiangular_weights

    lat, lon = highres_coords(resolution)
    if rows is not None:
        lat = lat[rows]
    coords = latlon_to_coords(lat, lon)
    weights = torch.tensor(equiangular_weights(lat, len(lon)))
    return FieldGrid.source(coords, Geometry.globe(), weights=weights), lat, lon


def bilinear_upsample(field: torch.Tensor, src_lat: np.ndarray, src_lon: np.ndarray, dst_lat: np.ndarray,
                      dst_lon: np.ndarray) -> torch.Tensor:
    """``(B, H, W, C)`` on a regular grid -> ``(B, H', W', C)``, periodic in longitude."""
    b, h, w, c = field.shape
    fi = torch.as_tensor(np.clip((dst_lat - src_lat[0]) / (src_lat[1] - src_lat[0]), 0, h - 1.000001), dtype=field.dtype, device=field.device)
    fj = torch.as_tensor(((dst_lon - src_lon[0]) % 360.0) / (src_lon[1] - src_lon[0]), dtype=field.dtype, device=field.device)
    i0 = fi.floor().long()
    j0 = fj.floor().long() % w
    j1 = (j0 + 1) % w
    di = (fi - i0.to(field.dtype)).view(1, -1, 1, 1)
    dj = (fj - fj.floor()).view(1, 1, -1, 1)
    top = field[:, i0][:, :, j0] * (1 - dj) + field[:, i0][:, :, j1] * dj
    bottom = field[:, i0 + 1][:, :, j0] * (1 - dj) + field[:, i0 + 1][:, :, j1] * dj
    return top * (1 - di) + bottom * di


class HighResHead(nn.Module):
    """Decoded features + static 0.25-degree context -> residual on the upsampled 1.5-degree fields."""

    def __init__(self, hidden: int, fields: int, static: int = 3, width: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden + static + fields, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, fields * 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, decoded: torch.Tensor, static: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([decoded, static, base], -1))
        mean, log_var = out.chunk(2, -1)
        return base + mean, log_var.clamp(-10.0, 10.0)


def static_features(orography_m: np.ndarray, land: np.ndarray, lat: np.ndarray) -> torch.Tensor:
    """``(H*W, 3)``: orography (km), land fraction, sin(latitude) on the 0.25-degree grid."""
    h, w = orography_m.shape
    sinlat = np.repeat(np.sin(np.radians(lat))[:, None], w, 1)
    return torch.tensor(np.stack([orography_m / 1000.0, land, sinlat], -1).reshape(h * w, 3), dtype=torch.float32)


@torch.no_grad()
def decode_highres(model, head: HighResHead, current: torch.Tensor, coarse_fields: torch.Tensor, static: torch.Tensor,
                   resolution: float = 0.25, rows_per_chunk: int = 40, neighbours: int = 24) -> tuple[torch.Tensor, torch.Tensor]:
    """
    0.25-degree fields from the processed latent mesh ``current`` ``(B, P_latent, hidden)``.

    Args:
        coarse_fields: ``(B, 121, 240, C)`` the 1.5-degree forecast of the head's fields (normalised).
        static: ``(1440*721, 3)`` from :func:`static_features`.

    Returns:
        ``(mean, log_var)`` each ``(B, 721, 1440, C)`` on the CPU.
    """
    lat_hr, lon_hr = highres_coords(resolution)
    lat_lr, lon_lr = np.linspace(-90, 90, 121), np.arange(240) * 1.5
    b = current.shape[0]
    c = coarse_fields.shape[-1]
    means, log_vars = [], []
    for start in range(0, lat_hr.size, rows_per_chunk):
        rows = slice(start, min(start + rows_per_chunk, lat_hr.size))
        grid, lat, lon = highres_grid(resolution, rows)
        grid = model._aligned(grid)
        seed = model.mesh_seed.expand(b, grid.num_points, -1)
        decoded = model.decode_norm(model.decoder(seed, current, model.link(grid, model.latent_grid, neighbours)))
        base = bilinear_upsample(coarse_fields.to(decoded.dtype), lat_lr, lon_lr, lat, lon).reshape(b, -1, c)
        st = static[rows.start * lon_hr.size: rows.stop * lon_hr.size].to(decoded.device, decoded.dtype).expand(b, -1, -1)
        mean, log_var = head(decoded, st, base)
        means.append(mean.float().cpu().view(b, len(lat), len(lon), c))
        log_vars.append(log_var.float().cpu().view(b, len(lat), len(lon), c))
        model.clear_links()
    return torch.cat(means, 1), torch.cat(log_vars, 1)


def highres_loss(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
    """Area-weighted Gaussian NLL on the 0.25-degree points (NaN targets, e.g. SST over land, ignored)."""
    ok = torch.isfinite(target)
    t = torch.where(ok, target, torch.zeros_like(target))
    nll = 0.5 * (log_var + (t - mean) ** 2 / log_var.exp())
    w = area.view(1, -1, 1) * ok
    return (nll * w).sum() / w.sum().clamp_min(1.0)


def train_highres(model, head: HighResHead, batches, static: torch.Tensor, steps: int = 2000, lr: float = 3e-4,
                  points_per_step: int = 200_000, tune_decoder: bool = True, log=print) -> dict:
    """
    Fit the high-resolution head (and optionally the decoder) on ``batches`` -- an iterator of dicts with
    ``analysis`` (B, T, P, C), ``calendar``, ``coarse`` (B, 121, 240, F) and ``target`` (B, 721, 1440, F), the
    1.5-degree input and the 0.25-degree truth of the same time. Each step samples ``points_per_step`` random
    0.25-degree points (area-weighted), so a step costs a fraction of a full decode. The backbone is frozen.
    """
    from ihelix import FieldGrid, Geometry

    device = next(model.parameters()).device
    for p in model.parameters():
        p.requires_grad = False
    params = list(head.parameters())
    if tune_decoder:
        for p in list(model.decoder.parameters()) + list(model.decode_norm.parameters()):
            p.requires_grad = True
            params.append(p)
    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    lat_hr, lon_hr = highres_coords()
    lat_lr, lon_lr = np.linspace(-90, 90, 121), np.arange(240) * 1.5
    area = np.repeat(np.cos(np.radians(lat_hr))[:, None], lon_hr.size, 1).ravel()
    prob = area / area.sum()
    rng = np.random.default_rng(0)
    history = []
    for step, batch in zip(range(steps), batches):
        idx = np.sort(rng.choice(area.size, points_per_step, replace=False, p=prob))
        rows, cols = idx // lon_hr.size, idx % lon_hr.size
        coords = torch.tensor(np.stack([np.radians(lat_hr[rows]), np.radians(lon_hr[cols])], -1), dtype=torch.float64)
        grid = model._aligned(FieldGrid.source(coords, Geometry.globe()))
        with torch.no_grad():
            latent = model.encode(None, None, batch["analysis"].to(device), model._aligned(batch["grid"]), batch["calendar"].to(device))
            current = model.norm(model._run_blocks(latent, model.latent_grid))[:, -1]
        b = current.shape[0]
        seed = model.mesh_seed.expand(b, grid.num_points, -1)
        decoded = model.decode_norm(model.decoder(seed, current, model.link(grid, model.latent_grid, 24)))
        coarse = batch["coarse"].to(device)
        full = bilinear_upsample(coarse, lat_lr, lon_lr, lat_hr, lon_hr).reshape(b, -1, coarse.shape[-1])
        base = full[:, torch.as_tensor(idx, device=device)]
        target = batch["target"].to(device).reshape(b, -1, coarse.shape[-1])[:, torch.as_tensor(idx, device=device)]
        mean, log_var = head(decoded, static[torch.as_tensor(idx)].to(device).expand(b, -1, -1), base)
        loss = highres_loss(mean, log_var, target, torch.ones(len(idx), device=device))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimiser.step()
        model.clear_links()
        history.append(float(loss.detach()))
        if log and step % 100 == 0:
            log(f"[highres] step {step}: NLL {np.mean(history[-100:]):.4f}")
    return {"final_nll": float(np.mean(history[-100:])) if history else float("nan"), "steps": len(history)}


__all__ = ["ERA5_0P25", "HIGHRES_FIELDS", "HighResHead", "bilinear_upsample", "decode_highres", "highres_coords",
           "highres_grid", "highres_loss", "static_features", "train_highres"]
