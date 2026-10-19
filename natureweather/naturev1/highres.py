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
  out -- learned downscaling that then applies to every forecast step;
* ``python -m naturev1.highres`` (0.18) runs it as a stage: the 0.25-degree targets and static fields staged once
  to disk (resumable), then training on one GPU or many under ``torchrun`` that checkpoints every 20 minutes,
  pushes every checkpoint to the Hub and continues where it stopped -- on the same machine or a fresh one. The
  0.25-degree path keeps **its own copy** of the decoder (:class:`HighResDecoder`), so the backbone and its
  1.5-degree forecasts are never changed by it, and it is scored on held-out years against plain bilinear
  upsampling of the same forecast, field by field, over the globe and over land.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

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
def decode_highres(model, head, current: torch.Tensor, coarse_fields: torch.Tensor, static: torch.Tensor | None = None,
                   resolution: float = 0.25, rows_per_chunk: int = 40, neighbours: int = 24,
                   coarse_lat: np.ndarray | None = None, coarse_lon: np.ndarray | None = None
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    0.25-degree fields from the processed latent mesh ``current`` ``(B, P_latent, hidden)``.

    Args:
        head: a :class:`HighResHead` (decoded through the backbone's own decoder), or a trained
            :class:`HighResDecoder` from :func:`load_highres` (its own decoder, grid and static fields).
        coarse_fields: ``(B, H, W, C)`` the coarse forecast of the head's fields (normalised); 1.5 degrees,
            south to north, unless ``coarse_lat`` / ``coarse_lon`` say otherwise.
        static: ``(H'*W', 3)`` from :func:`static_features`; a :class:`HighResDecoder` carries its own.

    Returns:
        ``(mean, log_var)`` each ``(B, H', W', C)`` on the CPU.
    """
    from ihelix import FieldGrid, Geometry

    from .corpora import latlon_to_coords

    if isinstance(head, HighResDecoder):
        lat_hr, lon_hr = head.latitude, head.longitude
        static = head.static if static is None else static
        decoder, decode_norm, net = head.decoder, head.decode_norm, head.head
    else:
        lat_hr, lon_hr = highres_coords(resolution)
        decoder, decode_norm, net = model.decoder, model.decode_norm, head
    if static is None:
        raise ValueError("give the static fields (static_features) for a bare HighResHead")
    lat_lr = np.linspace(-90, 90, coarse_fields.shape[1]) if coarse_lat is None else np.asarray(coarse_lat)
    lon_lr = np.arange(coarse_fields.shape[2]) * 360.0 / coarse_fields.shape[2] if coarse_lon is None else np.asarray(coarse_lon)
    b = current.shape[0]
    c = coarse_fields.shape[-1]
    means, log_vars = [], []
    for start in range(0, lat_hr.size, rows_per_chunk):
        rows = slice(start, min(start + rows_per_chunk, lat_hr.size))
        lat = lat_hr[rows]
        grid = model._aligned(FieldGrid.source(latlon_to_coords(lat, lon_hr), Geometry.globe()))
        seed = model.mesh_seed.expand(b, grid.num_points, -1)
        decoded = decode_norm(decoder(seed, current, model.link(grid, model.latent_grid, neighbours)))
        base = bilinear_upsample(coarse_fields.to(decoded.device, decoded.dtype), lat_lr, lon_lr, lat, lon_hr).reshape(b, -1, c)
        st = static[rows.start * lon_hr.size: rows.stop * lon_hr.size].to(decoded.device, decoded.dtype).expand(b, -1, -1)
        mean, log_var = net(decoded, st, base)
        means.append(mean.float().cpu().view(b, len(lat), len(lon_hr), c))
        log_vars.append(log_var.float().cpu().view(b, len(lat), len(lon_hr), c))
        model.clear_links()
    return torch.cat(means, 1), torch.cat(log_vars, 1)


def highres_loss(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor, area: torch.Tensor,
                 beta: float = 0.5) -> torch.Tensor:
    """
    Area-weighted Gaussian NLL on the 0.25-degree points (NaN targets, e.g. SST over land, ignored), beta-weighted.

    Plain NLL lets the variance run ahead of the mean: shrinking ``log_var`` is the cheap way down, and once it has
    shrunk, every residual is amplified by ``1 / var`` and the mean oscillates instead of fitting (measured on the
    constructed mountain world: the learned output never beat bilinear upsampling in 300 steps). Weighting each
    point's NLL by ``var ** beta`` (detached; Seitzer et al. 2022) restores a mean gradient between NLL's and MSE's:
    with ``beta = 0.5`` the same run removes 99% of what bilinear upsampling misses. ``beta = 0`` is plain NLL.
    """
    ok = torch.isfinite(target)
    t = torch.where(ok, target, torch.zeros_like(target))
    nll = 0.5 * (log_var + (t - mean) ** 2 / log_var.exp())
    if beta:
        nll = nll * log_var.detach().exp() ** beta
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


# ------------------------------------------------------------------------- 0.18: the 0.25-degree stage --

def bilinear_points(field: torch.Tensor, src_lat: np.ndarray, src_lon: np.ndarray, lat: np.ndarray,
                    lon: np.ndarray) -> torch.Tensor:
    """``(B, H, W, C)`` on a regular grid (either latitude order) -> ``(B, N, C)`` at the points ``lat``/``lon``."""
    b, h, w, c = field.shape
    fi = np.clip((np.asarray(lat) - src_lat[0]) / (src_lat[1] - src_lat[0]), 0, h - 1.000001)
    fj = ((np.asarray(lon) - src_lon[0]) % 360.0) / (src_lon[1] - src_lon[0])
    i0 = np.floor(fi).astype(np.int64)
    j0 = np.floor(fj).astype(np.int64) % w
    j1 = (j0 + 1) % w

    def as_t(values):
        return torch.as_tensor(values, device=field.device)

    di = as_t(fi - i0).to(field.dtype).view(1, -1, 1)
    dj = as_t(fj - np.floor(fj)).to(field.dtype).view(1, -1, 1)
    i0, j0, j1 = as_t(i0), as_t(j0), as_t(j1)
    top = field[:, i0, j0] * (1 - dj) + field[:, i0, j1] * dj
    bottom = field[:, i0 + 1, j0] * (1 - dj) + field[:, i0 + 1, j1] * dj
    return top * (1 - di) + bottom * di


class HighResDecoder(nn.Module):
    """
    The 0.25-degree path as one module: **its own copy** of the backbone's decoder, and the head. The backbone is
    held by reference and is not a submodule, so a checkpoint holds this path alone (a few MB) and training it never
    changes the backbone's own 1.5-degree forecasts.
    """

    def __init__(self, model, fields: int, width: int = 256, latitude: np.ndarray | None = None,
                 longitude: np.ndarray | None = None, static: torch.Tensor | None = None) -> None:
        super().__init__()
        self.decoder = copy.deepcopy(model.decoder)
        self.decode_norm = copy.deepcopy(model.decode_norm)
        self.head = HighResHead(model.config.hidden_size, fields, width=width)
        object.__setattr__(self, "backbone", model)
        lat, lon = highres_coords()
        self.latitude = np.asarray(lat if latitude is None else latitude, np.float64)
        self.longitude = np.asarray(lon if longitude is None else longitude, np.float64)
        self.register_buffer("static", static if static is not None else torch.zeros(0, 3), persistent=False)

    def forward(self, current: torch.Tensor, grid, static: torch.Tensor, base: torch.Tensor,
                neighbours: int = 24) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.backbone
        seed = model.mesh_seed.expand(current.shape[0], grid.num_points, -1)
        decoded = self.decode_norm(self.decoder(seed, current, model.link(grid, model.latent_grid, neighbours)))
        return self.head(decoded, static.to(decoded.dtype), base.to(decoded.dtype))


def open_highres_store(store: str = ERA5_0P25):
    """WeatherBench 2's 0.25-degree ERA5 (anonymous read)."""
    import xarray as xr

    return xr.open_zarr(store, storage_options={"token": "anon"}, consolidated=True)


def _subset_normalizer(statistics: dict, fields) -> object:
    from .era5 import Normalizer

    full = Normalizer.from_dict(statistics)
    index = [full.variables.index(f) for f in fields]
    return Normalizer(list(fields), full.mean[index], full.std[index], np.zeros(len(fields), np.float32))


def _years(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def pair_times(era5_meta: dict, years, every_hours: int = 12) -> list[int]:
    """Valid times (unix seconds) of the staged 1.5-degree record in ``years``, every ``every_hours``."""
    times = np.asarray(era5_meta["time"], np.int64)
    year = times.astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
    keep = np.isin(year, list(years)) & ((times // 3600) % int(every_hours) == 0)
    return [int(t) for t in times[keep]]


def stage_highres(directory, times, statistics: dict, workers: int = 16, dataset=None,
                  fields=HIGHRES_FIELDS) -> tuple[Path, Path]:
    """
    Stage the 0.25-degree ERA5 at ``times`` (normalised with the backbone's own statistics, float16, ~10 MB a time)
    and the static fields (orography, land-sea mask). Resumable: an interrupted staging continues where it stopped,
    and a finished one is never downloaded again.

    Returns ``(targets.npy, static.npz)``.
    """
    from .era5 import materialise

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dataset = open_highres_store() if dataset is None else dataset
    stamps = dataset.time.values.astype("datetime64[s]").astype(np.int64)
    position = {int(t): i for i, t in enumerate(stamps)}
    indices = np.asarray(sorted(position[int(t)] for t in times if int(t) in position), np.int64)
    if not len(indices):
        raise ValueError("none of the asked-for times are in the 0.25-degree store")
    targets = materialise(dataset, indices, directory / "highres_targets.npy", variables=tuple(fields),
                          normalizer=_subset_normalizer(statistics, fields), levels=None, workers=workers, chunk=4)
    static = directory / "highres_static.npz"
    if not static.exists():
        lat = np.asarray(dataset["latitude"].values, np.float64)
        lon = np.asarray(dataset["longitude"].values, np.float64)

        def field(name):
            values = dataset[name]
            if "time" in values.dims:
                values = values.isel(time=0)
            return np.asarray(values.transpose("latitude", "longitude").values, np.float64)

        orography, land = field("geopotential_at_surface") / 9.80665, field("land_sea_mask")
        if lat[0] > lat[-1]:                                      # store north to south; keep south to north
            lat, orography, land = lat[::-1], orography[::-1], land[::-1]
        tmp = static.with_name(".tmp-" + static.name)
        with open(tmp, "wb") as handle:
            np.savez(handle, latitude=lat, longitude=lon, orography_m=orography.astype(np.float32),
                     land=land.astype(np.float32))
        tmp.replace(static)
    return targets, static


class HighResPairs(torch.utils.data.Dataset):
    """
    For each staged 0.25-degree time: the backbone's 1.5-degree analysis window ending then, its coarse fields now,
    and the 0.25-degree truth now (south to north). Memmaps open lazily, so DataLoader workers each open their own.
    """

    def __init__(self, era5, targets, channels: int, history: int, times=None, fields=HIGHRES_FIELDS) -> None:
        self.era5, self.targets = str(era5), str(targets)
        meta = json.loads(Path(era5).with_suffix(".json").read_text())
        tmeta = json.loads(Path(targets).with_suffix(".json").read_text())
        from .era5 import Normalizer

        self.normalizer = Normalizer.from_dict(meta["statistics"])
        variables = list(meta["variables"])
        self.field_index = [variables.index(f) for f in fields]
        self.channels, self.history = channels, history
        era5_times = np.asarray(meta["time"], np.int64)
        row = {int(t): i for i, t in enumerate(era5_times)}
        wanted = None if times is None else {int(t) for t in times}
        self.items = []
        for j, t in enumerate(np.asarray(tmeta["time"], np.int64)):
            i = row.get(int(t))
            if i is None or i < history - 1 or (wanted is not None and int(t) not in wanted):
                continue
            if int(era5_times[i] - era5_times[i - history + 1]) != (history - 1) * 21600:
                continue                                          # a gap in the record: no full window
            self.items.append((i, j, int(t)))
        self.flip = bool(tmeta["latitude"][0] > tmeta["latitude"][-1])
        self.latitude = np.asarray(tmeta["latitude"], np.float64)[::-1 if self.flip else 1].copy()
        self.longitude = np.asarray(tmeta["longitude"], np.float64)
        self.coarse_latitude = np.asarray(meta.get("latitude") or np.linspace(-90, 90, meta["shape"][1]), np.float64)
        self.coarse_longitude = np.asarray(meta.get("longitude") or np.arange(meta["shape"][2]) * 360.0 / meta["shape"][2],
                                           np.float64)
        self._values = self._targets = None

    def __len__(self) -> int:
        return len(self.items)

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_values"] = state["_targets"] = None
        return state

    def __getitem__(self, item: int) -> dict:
        from .model import calendar_features
        from .satdata import prepared_to_state

        if self._values is None:
            self._values = np.load(self.era5, mmap_mode="r")
            self._targets = np.load(self.targets, mmap_mode="r")
        i, j, t = self.items[item]
        frames = [np.asarray(self._values[k], np.float32) for k in range(i - self.history + 1, i + 1)]
        analysis = torch.stack([prepared_to_state(f, self.normalizer, self.channels) for f in frames])
        stamps = torch.tensor([t - 21600.0 * (self.history - 1 - k) for k in range(self.history)], dtype=torch.float64)
        coarse = np.nan_to_num(frames[-1][..., self.field_index])
        target = np.asarray(self._targets[j], np.float32)
        if self.flip:
            target = target[::-1]
        return {"analysis": analysis, "calendar": calendar_features(stamps),
                "coarse": torch.from_numpy(np.ascontiguousarray(coarse)),
                "target": torch.from_numpy(np.ascontiguousarray(target)).half(),
                "valid_time": torch.tensor(float(t), dtype=torch.float64)}


def sample_points(latitude: np.ndarray, width: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """``count`` area-weighted points of the ``len(latitude) x width`` grid (sorted flat indices, no repeats)."""
    weight = np.clip(np.cos(np.radians(latitude)), 1e-6, None)
    rows = rng.choice(latitude.size, size=count, p=weight / weight.sum())
    cols = rng.integers(0, width, size=count)
    return np.unique(rows * width + cols)


class HighResStep(nn.Module):
    """One training step of the 0.25-degree path as a module, so data parallelism sees one call of tensors."""

    def __init__(self, backbone, path: HighResDecoder, grid, coarse_lat, coarse_lon) -> None:
        super().__init__()
        self.path = path
        object.__setattr__(self, "backbone", backbone)
        self.grid = grid
        self.coarse_lat, self.coarse_lon = np.asarray(coarse_lat), np.asarray(coarse_lon)

    @torch.no_grad()
    def latent(self, analysis: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        model = self.backbone
        latent = model.encode(None, None, analysis, self.grid, calendar)
        return model.norm(model._run_blocks(latent, model.latent_grid))[:, -1]

    def predict(self, analysis, calendar, coarse, idx: np.ndarray):
        """``(mean, log_var, bilinear)`` at the flat 0.25-degree indices ``idx``, normalised."""
        from ihelix import FieldGrid, Geometry

        path = self.path
        width = path.longitude.size
        lat, lon = path.latitude[idx // width], path.longitude[idx % width]
        coords = torch.tensor(np.stack([np.radians(lat), np.radians(lon) % (2 * np.pi)], -1), dtype=torch.float64)
        grid = self.backbone._aligned(FieldGrid.source(coords, Geometry.globe()))
        current = self.latent(analysis, calendar)
        b = current.shape[0]
        base = bilinear_points(coarse.float(), self.coarse_lat, self.coarse_lon, lat, lon)
        static = path.static[torch.as_tensor(idx, device=path.static.device)].expand(b, -1, -1)
        mean, log_var = path(current, grid, static, base)
        self.backbone.clear_links()
        return mean.float(), log_var.float(), base

    def forward(self, analysis, calendar, coarse, target, idx: torch.Tensor):
        flat = idx.detach().cpu().numpy()
        mean, log_var, base = self.predict(analysis, calendar, coarse, flat)
        b, f = target.shape[0], target.shape[-1]
        truth = target.reshape(b, -1, f)[:, idx.to(target.device)].float()
        loss = highres_loss(mean, log_var, truth, torch.ones(len(flat), device=mean.device))
        with torch.no_grad():
            ok = torch.isfinite(truth)
            t = torch.where(ok, truth, torch.zeros_like(truth))
            n = ok.sum().clamp_min(1)
            parts = {"nll": loss.detach(), "mse": (((mean - t) ** 2) * ok).sum() / n,
                     "mse_bilinear": (((base - t) ** 2) * ok).sum() / n}
        return loss, parts


@torch.no_grad()
def evaluate_highres(step: HighResStep, loader, normalizer, fields, device, points: int = 200_000,
                     max_batches: int | None = None, seed: int = 1234) -> dict:
    """
    Held-out RMSE in physical units, field by field, of the learned 0.25-degree output and of bilinear upsampling of
    the same 1.5-degree fields -- over the globe and over land -- on area-weighted random points of every time.
    """
    from .engine import all_sum

    path = step.path
    land = path.static[:, 1] > 0.5
    sums: dict = {}
    for k, batch in enumerate(loader):
        if max_batches is not None and k >= max_batches:
            break
        idx = sample_points(path.latitude, path.longitude.size, points, np.random.default_rng([seed, k]))
        analysis, calendar = batch["analysis"].to(device), batch["calendar"].to(device)
        mean, _, base = step.predict(analysis, calendar, batch["coarse"].to(device), idx)
        b, f = mean.shape[0], mean.shape[-1]
        truth = batch["target"].to(device).reshape(b, -1, f)[:, torch.as_tensor(idx, device=device)].float()
        on_land = land[torch.as_tensor(idx, device=land.device)].to(device).view(1, -1)
        for c, name in enumerate(fields):
            t = normalizer.decode(truth[..., c], name)
            ok = torch.isfinite(t)
            t = torch.where(ok, t, torch.zeros_like(t))
            for label, guess in (("learned", mean[..., c]), ("bilinear", base[..., c])):
                err = (normalizer.decode(guess, name) - t) ** 2 * ok
                for where, mask in (("", ok), ("_land", ok & on_land)):
                    sums[f"{name}|{label}{where}"] = sums.get(f"{name}|{label}{where}", 0.0) + float((err * mask).sum())
                    sums[f"{name}|n{where}"] = sums.get(f"{name}|n{where}", 0.0) + (float(mask.sum()) if label == "learned" else 0.0)
        sums["times"] = sums.get("times", 0.0) + b
    sums = all_sum(sums, device) if sums else sums
    report = {"fields": {}, "times": int(sums.get("times", 0))}
    for name in fields:
        row = {}
        for where in ("", "_land"):
            n = max(sums.get(f"{name}|n{where}", 0.0), 1.0)
            for label in ("learned", "bilinear"):
                row[f"{label}{where}"] = math.sqrt(sums.get(f"{name}|{label}{where}", 0.0) / n)
        report["fields"][name] = row
    report["better_fields"] = int(sum(r["learned"] < r["bilinear"] for r in report["fields"].values()))
    report["better_fields_land"] = int(sum(r["learned_land"] < r["bilinear_land"] for r in report["fields"].values()))
    return report


UNITS = {"mean_sea_level_pressure": (0.01, "hPa"), "total_precipitation_6hr": (1000.0, "mm/6h"), "2m_temperature": (1.0, "K"),
         "10m_u_component_of_wind": (1.0, "m/s"), "10m_v_component_of_wind": (1.0, "m/s")}


def describe_highres(report: dict) -> str:
    lines = [f"{'field':26} {'0.25 learned':>13} {'bilinear':>10} {'gain':>7}   {'land: learned':>14} {'bilinear':>10}"]
    for name, row in report["fields"].items():
        scale, unit = UNITS.get(name, (1.0, ""))
        gain = 100 * (1 - row["learned"] / max(row["bilinear"], 1e-12))
        lines.append(f"{name:26} {row['learned'] * scale:>10.3f} {unit:<3}{row['bilinear'] * scale:>9.3f} {gain:>+6.1f}%   "
                     f"{row['learned_land'] * scale:>14.3f} {row['bilinear_land'] * scale:>10.3f}")
    lines.append(f"better than bilinear upsampling on {report['better_fields']} of {len(report['fields'])} fields "
                 f"({report['better_fields_land']} over land), {report['times']} held-out times")
    return "\n".join(lines)


def load_backbone(checkpoint: str):
    """A NatureV1 / NatureV2 model exactly as a checkpoint describes it (every path it was trained with)."""
    from ihelix import fibonacci_sphere

    from .model import NatureConfig, NatureV1

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = NatureConfig.from_dict(payload["config"])
    model = NatureV1(config, fibonacci_sphere(config.latent_points, num_neighbours=config.latent_neighbours,
                                              cluster_size=config.latent_cluster))
    model.load_state_dict(payload["model"])
    return model


def load_highres(path: str, model, device=None) -> HighResDecoder:
    """The trained 0.25-degree path (``highres.pt``) attached to ``model``; pass it to :func:`decode_highres`."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    module = HighResDecoder(model, len(payload["fields"]), payload["width"], np.asarray(payload["latitude"]),
                            np.asarray(payload["longitude"]), payload["static"].float())
    module.load_state_dict(payload["path"])
    module.fields, module.statistics, module.report = payload["fields"], payload["statistics"], payload.get("report")
    return module.to(device or next(model.parameters()).device)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="NatureV1 / NatureV2: the 0.25-degree output stage (stage, train, score)")
    parser.add_argument("--checkpoint", required=True, help="the backbone: the model whose forecasts are refined")
    parser.add_argument("--era5", required=True, help="the staged 1.5-degree record (naturev1.b300.stage_era5_record)")
    parser.add_argument("--data", required=True, help="where the 0.25-degree targets are staged (~10 MB a time)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-years", default="2014-2016")
    parser.add_argument("--val-years", default="2021")
    parser.add_argument("--every-hours", type=int, default=12, help="training times: one every N hours")
    parser.add_argument("--val-every-hours", type=int, default=72)
    parser.add_argument("--hours", type=float, default=4.0, help="wall-clock budget, counted across restarts")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--points", type=int, default=200_000, help="0.25-degree points per step")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stage-workers", type=int, default=16)
    parser.add_argument("--checkpoint-minutes", type=float, default=20.0)
    parser.add_argument("--eval-minutes", type=float, default=60.0)
    parser.add_argument("--hub-repo", default=None)
    parser.add_argument("--store", default=ERA5_0P25)
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    meta = json.loads(Path(args.era5).with_suffix(".json").read_text())
    train_times = pair_times(meta, _years(args.train_years), args.every_hours)
    val_times = pair_times(meta, _years(args.val_years), args.val_every_hours)
    data = Path(args.data)
    if not args.train_only:
        dataset = open_highres_store(args.store)
        print(f"[highres] staging {len(train_times) + len(val_times):,} times at 0.25 degrees "
              f"(~{(len(train_times) + len(val_times)) * 10.4 / 1000:.0f} GB) into {data}", flush=True)
        stage_highres(data, sorted(set(train_times) | set(val_times)), meta["statistics"], args.stage_workers, dataset)
        print("[highres] staged", flush=True)
    if args.stage_only:
        return

    from .engine import Engine, EngineSettings, is_main, setup_distributed
    from .safeio import atomic_torch_save
    from .sattrain import _keep_average, staged_grid

    rank, _, _, device = setup_distributed()
    backbone = load_backbone(args.checkpoint).to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    targets, static_file = data / "highres_targets.npy", data / "highres_static.npz"
    z = np.load(static_file)
    static = static_features(z["orography_m"], z["land"], z["latitude"])
    fields = list(HIGHRES_FIELDS)
    path = HighResDecoder(backbone, len(fields), args.width, z["latitude"], z["longitude"], static)
    for parameter in path.parameters():
        parameter.requires_grad = True
    history = backbone.config.history_frames
    train = HighResPairs(args.era5, targets, backbone.config.analysis_channels, history, train_times, fields)
    held = HighResPairs(args.era5, targets, backbone.config.analysis_channels, history, val_times, fields)
    if not np.allclose(train.latitude, path.latitude):
        raise ValueError("the staged targets and the static fields are on different grids")
    grid = backbone._aligned(staged_grid(meta))
    step = HighResStep(backbone, path, grid, train.coarse_latitude, train.coarse_longitude)
    settings = EngineSettings(run="highres", output=args.output, hours=args.hours, batch_size=args.batch_size,
                              learning_rate=args.lr, workers=args.workers, hub_repo=args.hub_repo,
                              max_steps=args.max_steps, warmup_steps=100, checkpoint_minutes=args.checkpoint_minutes,
                              eval_minutes=args.eval_minutes, log_every=25, weight_decay=0.01,
                              reserve_minutes=min(10.0, args.hours * 6.0))
    engine = Engine(settings, path, step, config={"fields": fields, "width": args.width, "backbone": str(args.checkpoint)})
    loader = engine.loader(train)
    val_loader = engine.loader(held, shuffle=False, drop_last=False)
    engine.resume()
    normalizer = _subset_normalizer(meta["statistics"], fields)
    if is_main():
        size = sum(p.numel() for p in path.parameters()) / 1e6
        print(f"[highres] {len(train):,} training times, {len(held):,} held out; 0.25-degree path {size:.2f} M parameters "
              f"(its own decoder copy + head; backbone frozen and unchanged); {engine.world} GPU(s), "
              f"{args.points:,} points a step", flush=True)
    report: dict = {}
    stop = len(train) == 0
    while not stop:
        engine.set_epoch(loader, engine.epoch)
        for batch in loader:
            batch = engine.to_device(batch)
            idx = sample_points(path.latitude, path.longitude.size, args.points,
                                np.random.default_rng([settings.seed, engine.step, rank]))
            with engine.autocast():
                loss, parts = engine.model(batch["analysis"], batch["calendar"], batch["coarse"], batch["target"],
                                           torch.as_tensor(idx, device=device))
            engine.backward_step(loss, {k: float(v) for k, v in parts.items()})
            stop, save, evaluate = engine.poll()
            if evaluate:
                with engine.averaged():
                    report = evaluate_highres(step, val_loader, normalizer, fields, device, args.points, max_batches=24)
                engine.last_eval = time.time()
                engine.history.append({"step": engine.step, **report})
                if is_main():
                    print(f"[highres] held-out at step {engine.step:,}:\n{describe_highres(report)}", flush=True)
            if save:
                engine.checkpoint(report)
            if stop:
                break
        engine.epoch += 1
    with engine.averaged():
        report = evaluate_highres(step, val_loader, normalizer, fields, device, args.points)
    if is_main():
        print(f"[highres] final held-out ({len(held):,} times):\n{describe_highres(report)}", flush=True)
    engine.history.append({"step": engine.step, "final": True, **report})
    engine.finish(report)
    _keep_average(engine)
    if is_main():
        final = Path(args.output) / "highres" / "highres.pt"
        atomic_torch_save({"path": path.state_dict(), "fields": fields, "width": args.width,
                           "latitude": path.latitude.tolist(), "longitude": path.longitude.tolist(),
                           "static": static.half(), "statistics": normalizer.to_dict(), "report": report,
                           "backbone": str(args.checkpoint), "steps": engine.step}, final)
        (final.parent / "report.json").write_text(json.dumps(report, indent=2))
        engine.pusher.submit([(final, "checkpoints/highres/highres.pt"),
                              (final.parent / "report.json", "checkpoints/highres/report.json")], "the 0.25-degree path")
        engine.pusher.wait(1800)
        print(f"[highres] finished: {final}", flush=True)


__all__ = ["ERA5_0P25", "HIGHRES_FIELDS", "HighResDecoder", "HighResHead", "HighResPairs", "HighResStep",
           "bilinear_points", "describe_highres", "evaluate_highres", "load_backbone", "load_highres", "open_highres_store",
           "pair_times", "sample_points", "stage_highres", "bilinear_upsample", "decode_highres", "highres_coords",
           "highres_grid", "highres_loss", "static_features", "train_highres"]


if __name__ == "__main__":
    main()
