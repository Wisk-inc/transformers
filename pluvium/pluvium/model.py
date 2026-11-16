# Copyright 2026 Nathan. Apache-2.0.
"""
Pluvium-500: rain, hour by hour, for every Caribbean town, on the NatureV1 / iHELIX engine.

Architecture
============

::

    ERA5 / GFS / ECMWF fields  --cross-attention-->  ┐
    (1 deg hourly, 1.5 deg aloft)                     │
    rain already falling, sun, island heating ------> ├─ variable-resolution Caribbean mesh (11 km over land,
    terrain descriptors, Smith-Barstad orographic --> ┘  80 km at sea), one frame per forecast step
                                                      │
                                         iHELIX blocks: geodesic attention (per-head radii 15-500 km),
                                         index attention (anything anywhere, content-addressed),
                                         gated delta-rule recurrence along the forecast's frames
                                                      │
                    ┌──────────────── decoder (cross-attention from each town or 8 km cell) ─────────────┐
               timing head                          amount head                              cause head
    neural non-homogeneous Markov chain      wet-hour rain: 3-component lognormal      cold front, tropical wave,
    P(dry->wet), P(wet->wet) every hour,     mixture, plus the orographic layer's      cyclone, afternoon storms,
    diurnal harmonics per point              upslope rate                              trade-wind showers, other

Version 2 (the default) adds, each behind a switch so version 1 can be rebuilt exactly for comparison:

* an **intensity chain** (:mod:`pluvium.chain`): four hidden states -- dry, light, moderate, heavy -- instead of two,
  with per-instrument confusion matrices, so timing and intensity come from one chain;
* **thunder** and **drizzle / rain / showers** heads, trained on what airports report;
* the **convective environment** (:mod:`pluvium.convection`) among the coarse inputs;
* **advected rain** (:mod:`pluvium.advection`): the last hour's rain carried along the steering wind, per point, for
  the first 12 hours;
* **airport assimilation**: the reports of the last three hours are read into the mesh's history frames by
  cross-attention (as MetNet-3 learns from station data), with dropout so the model works with or without them.

Version 1's experimental pieces (still in place):

* the **timing head** is a non-homogeneous two-state Markov chain whose transition probabilities the network writes
  for every hour and place (``P(rain starts | dry)``, ``P(rain continues | raining)``). From it the forecast reads,
  exactly, the chance of rain in each hour, the distribution of the hour rain *starts* and *stops*, and how long
  it lasts -- one consistent story instead of 72 independent probabilities. Training maximises the likelihood of the
  observed wet/dry sequences through a hidden-Markov forward pass in which each truth source (satellite, airport)
  has its own learned miss and false-alarm rates (:mod:`pluvium.losses`).
* **diurnal harmonics**: each point gets a learned daily cycle (two harmonics of local solar time) added to its
  transition logits, so "afternoon storms in the interior" is one smooth shape per place rather than something
  every hour must rediscover.
* the **orographic layer** (:class:`pluvium.physics.OrographicRain`) is read at every mesh point and decoder point
  as an input, and a learned share of its upslope rate feeds the rain amount directly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .chain import CENTRES, PRIOR_INITIAL, PRIOR_TRANSITIONS
from .dataset import ADVECTION, MESH_DYNAMIC, POINT_PHYSICS, TARGET_BASE
from .domain import STATIC_FEATURES, PointSet
from .features import ATMOS, CONVECTIVE, HOURLY_BASE, REGIMES, UPPER_CHANNELS
from .physics import OrographicRain
from .schedule import DEFAULT, Schedule
from .weather import OBS_FEATURES

#: switches that rebuild version 2 exactly (no theoretical physics inputs, no terrain patches, no refiner, temporal
#: mixer, lead modulation, experts or head dropout)
VERSION_2 = dict(physics_channels=False, point_physics=False, terrain_patch=False, refine_layers=0, temporal_layers=0,
                 lead_film=False, lead_shrink=False, experts=0, head_dropout=0.0, version=2)
#: and version 1 (two states; without thunder/kind heads, convective inputs, advection or airport assimilation)
VERSION_1 = dict(VERSION_2, states=2, thunder=False, precip_type=False, convective_channels=False, advection=False,
                 station_assimilation=False, version=1)
#: which output keys carry the point axis where (everything else: axis 2)
POINT_AXIS = {"init": 1}


@dataclass
class PluviumConfig:
    hidden: int = 512
    layers: int = 12
    heads: int = 8
    kv_heads: int = 4
    head_dim: int = 64
    intermediate: int = 2048
    latent_neighbours: int = 32
    min_radius_km: float = 15.0
    max_radius_km: float = 500.0
    index_layer_stride: int = 3
    recurrent_heads: int = 4
    recurrent_head_dim: int = 32
    encode_neighbours: int = 12
    hourly_radius_km: float = 110.0
    upper_radius_km: float = 170.0
    decode_neighbours: int = 16
    decode_radius_km: float = 22.0
    mixture: int = 3
    harmonics: int = 2
    orography: bool = True
    diurnal_harmonics: bool = True
    # version 2
    states: int = 4
    thunder: bool = True
    precip_type: bool = True
    convective_channels: bool = True
    advection: bool = True
    station_assimilation: bool = True
    station_radius_km: float = 150.0
    station_neighbours: int = 8
    # version 3
    physics_channels: bool = True
    point_physics: bool = True
    terrain_patch: bool = True
    refine_layers: int = 1
    refine_neighbours: int = 8
    temporal_layers: int = 1
    lead_film: bool = True
    lead_shrink: bool = True
    experts: int = 4
    head_dropout: float = 0.1
    version: int = 3
    name: str = "pluvium-500"

    @classmethod
    def named(cls, size: str, **overrides) -> PluviumConfig:
        rows = {
            "pluvium-tiny": dict(hidden=64, layers=2, heads=4, kv_heads=2, head_dim=16, intermediate=128,
                                 latent_neighbours=16, index_layer_stride=2, recurrent_heads=2, recurrent_head_dim=16,
                                 encode_neighbours=8, decode_neighbours=8, min_radius_km=20.0, max_radius_km=300.0,
                                 decode_radius_km=40.0),
            "pluvium-small": dict(hidden=96, layers=3, heads=4, kv_heads=2, head_dim=24, intermediate=256,
                                  latent_neighbours=16, index_layer_stride=3, recurrent_heads=2, recurrent_head_dim=24,
                                  encode_neighbours=8, decode_neighbours=8, min_radius_km=20.0, max_radius_km=400.0,
                                  decode_radius_km=45.0),
            "pluvium-base": dict(hidden=384, layers=10, heads=8, kv_heads=4, head_dim=48, intermediate=1536),
            "pluvium-500": {},
        }
        if size.endswith("-v1"):
            size, overrides = size[:-3], {**VERSION_1, **overrides}
        elif size.endswith("-v2"):
            size, overrides = size[:-3], {**VERSION_2, **overrides}
        if size not in rows:
            raise ValueError(f"unknown size {size!r}; choose from {', '.join(rows)} (add -v1 or -v2 for older versions)")
        name = size + {1: "-v1", 2: "-v2"}.get(overrides.get("version", 3), "")
        return cls(**{**rows[size], "name": name, **overrides})

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> PluviumConfig:
        known = {f.name for f in fields(cls)}
        values = dict(values)
        if "states" not in values:                       # a version 1 checkpoint
            values.update(VERSION_1)
        elif "physics_channels" not in values:           # a version 2 checkpoint
            values.update(VERSION_2)
        return cls(**{k: v for k, v in values.items() if k in known})

    def ihelix(self):
        from ihelix import IHelixConfig

        return IHelixConfig(
            in_channels=self.hidden, hidden_size=self.hidden, intermediate_size=self.intermediate,
            num_layers=self.layers, num_heads=self.heads, num_kv_heads=self.kv_heads, head_dim=self.head_dim,
            min_radius=self.min_radius_km, max_radius=self.max_radius_km, index_layer_stride=self.index_layer_stride,
            index_branching=4, index_beam_width=4, index_topk=4, landmark_dim=32, use_temporal=True,
            num_recurrent_heads=self.recurrent_heads, recurrent_head_dim=self.recurrent_head_dim,
            recurrent_value_head_dim=self.recurrent_head_dim, recurrent_chunk_size=16,
        )  # fmt: skip


class SubLink:
    """Rows of a precomputed :class:`ihelix.GridLink` (the decoder reads a different subset of targets each batch)."""

    def __init__(self, link, rows: torch.Tensor) -> None:
        self.neighbours = link.neighbours[rows]
        self.distances = link.distances[rows]
        self.offsets = link.offsets[rows]
        self.alignment = link.alignment[rows]
        self.weights = link.weights[rows]
        self.source_points = link.source_points


def _frame_embedding(schedule: Schedule) -> torch.Tensor:
    leads = torch.tensor(schedule.leads, dtype=torch.float32)
    spans = torch.tensor(schedule.spans, dtype=torch.float32)
    return torch.stack([leads / 72.0, torch.sin(2 * math.pi * leads / 24), torch.cos(2 * math.pi * leads / 24),
                        (leads < 0).float(), (spans - 1) / 2.0], -1)


class PluviumModel(nn.Module):
    """
    ``PluviumModel(config, geography, era5_grids)``; ``forward(batch)`` returns the heads' raw outputs for the batch's
    targets (and stations when present); :mod:`pluvium.losses` scores them and :mod:`pluvium.products` reads them.
    """

    def __init__(self, config: PluviumConfig, geo, grids: dict, schedule: Schedule = DEFAULT,
                 stations: PointSet | None = None) -> None:
        super().__init__()
        from ihelix import CrossAttention, GridLink, IHelixBlock
        from ihelix.model import IHelixRMSNorm

        self.config, self.schedule = config, schedule
        self.grids = {k: [float(x) for x in np.asarray(grids[k]).ravel()] for k in ("lat1", "lon1", "lat2", "lon2")}
        c, d = config, config.hidden
        k_states = c.states
        self.geo = geo
        mesh = geo.mesh
        self.latent_grid = mesh.grid(num_neighbours=c.latent_neighbours, cluster_size=64)
        lat1, lon1 = np.meshgrid(np.asarray(grids["lat1"]), np.asarray(grids["lon1"]), indexing="ij")
        lat2, lon2 = np.meshgrid(np.asarray(grids["lat2"]), np.asarray(grids["lon2"]), indexing="ij")
        self.grid_shapes = {"hourly": lat1.shape, "upper": lat2.shape}
        self.hourly_grid = PointSet(lat1.ravel(), lon1.ravel(), np.zeros((lat1.size, 1)), np.zeros(lat1.size, np.int8),
                                    np.cos(np.radians(lat1.ravel()))).source_grid()
        self.upper_grid = PointSet(lat2.ravel(), lon2.ravel(), np.zeros((lat2.size, 1)), np.zeros(lat2.size, np.int8),
                                   np.cos(np.radians(lat2.ravel()))).source_grid()
        self.link_hourly = GridLink(self.latent_grid, self.hourly_grid, min(c.encode_neighbours, lat1.size))
        self.link_upper = GridLink(self.latent_grid, self.upper_grid, min(c.encode_neighbours, lat2.size))
        targets = geo.targets
        self.link_targets = GridLink(targets.source_grid(), self.latent_grid, min(c.decode_neighbours, len(mesh)))
        self.stations = stations
        self.link_stations = (GridLink(stations.source_grid(), self.latent_grid, min(c.decode_neighbours, len(mesh)))
                              if stations is not None else None)
        n_static = len(STATIC_FEATURES) + 3
        self.register_buffer("mesh_static", self._static(mesh), persistent=False)
        self.register_buffer("target_static", self._static(targets), persistent=False)
        self.register_buffer("station_static", self._static(stations) if stations is not None else torch.zeros(0, n_static),
                             persistent=False)
        self.register_buffer("target_lon", torch.tensor(targets.lon, dtype=torch.float32), persistent=False)
        self.register_buffer("station_lon", torch.tensor(stations.lon if stations is not None else np.zeros(0),
                                                         dtype=torch.float32), persistent=False)
        self.register_buffer("frame_embed", _frame_embedding(schedule), persistent=False)
        hour_frame, hour_sub = schedule.hour_frame()
        future = schedule.future
        position = {int(k): i for i, k in enumerate(future)}
        self.register_buffer("hour_frame", torch.tensor([position[int(k)] for k in hour_frame]), persistent=False)
        self.register_buffer("hour_sub", torch.tensor(hour_sub), persistent=False)
        self.register_buffer("future", torch.tensor(future), persistent=False)
        self.register_buffer("history", torch.tensor(np.nonzero(schedule.leads < 0)[0]), persistent=False)
        nb, nc = len(HOURLY_BASE), len(CONVECTIVE)
        hourly_idx = list(range(nb)) + (list(range(nb, nb + nc)) if c.convective_channels else []) + (
            list(range(nb + nc, nb + nc + len(ATMOS))) if c.physics_channels else [])
        pb, pa = len(TARGET_BASE), len(ADVECTION)
        point_idx = list(range(pb)) + (list(range(pb, pb + pa)) if c.advection else []) + (
            list(range(pb + pa, pb + pa + len(POINT_PHYSICS))) if c.point_physics else [])
        n_hourly, n_point = len(hourly_idx), len(point_idx)
        self.register_buffer("hourly_idx", torch.tensor(hourly_idx), persistent=False)
        self.register_buffer("point_idx", torch.tensor(point_idx), persistent=False)

        # inputs
        self.hourly_in = nn.Linear(n_hourly, d)
        self.upper_in = nn.Linear(len(UPPER_CHANNELS), d)
        self.mesh_static_in = nn.Linear(n_static, d)
        self.mesh_dyn_in = nn.Linear(len(MESH_DYNAMIC) + 1, d)
        self.frame_in = nn.Linear(5, d)
        self.read_hourly = CrossAttention(d, c.heads, c.kv_heads, c.head_dim, 3, c.hourly_radius_km)
        self.read_upper = CrossAttention(d, c.heads, c.kv_heads, c.head_dim, 3, c.upper_radius_km)
        self.encode_norm = IHelixRMSNorm(d)
        self.read_obs = self.obs_in = self.link_obs = None
        if c.station_assimilation and stations is not None and len(stations):
            self.obs_in = nn.Linear(len(OBS_FEATURES), d)
            self.read_obs = CrossAttention(d, c.heads, c.kv_heads, c.head_dim, 3, c.station_radius_km)
            self.link_obs = GridLink(self.latent_grid, stations.source_grid(), min(c.station_neighbours, len(stations)))
        # processor
        icfg = c.ihelix()
        self.blocks = nn.ModuleList(IHelixBlock(icfg, self.latent_grid.geometry, i) for i in range(c.layers))
        self.norm = IHelixRMSNorm(d)
        self.gradient_checkpointing = False
        # decoder
        self.point_static_in = nn.Linear(n_static, d)
        self.point_dyn_in = nn.Linear(n_point + 1, d)
        self.decode = CrossAttention(d, c.heads, c.kv_heads, c.head_dim, 3, c.decode_radius_km)
        self.decode_norm = IHelixRMSNorm(d)
        self.decode_ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.head_norm = IHelixRMSNorm(d)
        # version 3 engine (pluvium.engine)
        from .engine import (
            HeadDropout,
            LeadFiLM,
            LeadShrink,
            PointRefiner,
            RegimeExperts,
            TemporalMixer,
            TerrainPatchEncoder,
            knn_on_sphere,
            terrain_patches,
        )

        self.patch_encoder = TerrainPatchEncoder(d) if c.terrain_patch else None
        if self.patch_encoder is not None:
            self.register_buffer("target_patches", torch.from_numpy(terrain_patches(geo.terrain, targets.lat, targets.lon)),
                                 persistent=False)
            self.register_buffer("station_patches", torch.from_numpy(
                terrain_patches(geo.terrain, stations.lat, stations.lon)) if stations is not None
                else torch.zeros(0, 2, 16, 16, dtype=torch.float16), persistent=False)
        self.film = LeadFiLM(d) if c.lead_film else None
        self.temporal = TemporalMixer(d, max(1, c.heads // 2), c.temporal_layers) if c.temporal_layers else None
        self.refiner = PointRefiner(d, max(1, c.heads // 2), c.refine_neighbours, c.refine_layers) if c.refine_layers else None
        self.experts = RegimeExperts(d, c.experts) if c.experts else None
        self.head_dropout = HeadDropout(c.head_dropout) if c.head_dropout > 0 else None
        self.shrink = (LeadShrink(torch.tensor(PRIOR_TRANSITIONS[c.states]).log()) if c.lead_shrink else None)
        self._knn = knn_on_sphere
        if self.refiner is not None and stations is not None:
            si, sd = knn_on_sphere(stations.lat, stations.lon, c.refine_neighbours)
            self.register_buffer("station_nbr", torch.from_numpy(si), persistent=False)
            self.register_buffer("station_nbr_km", torch.from_numpy(sd), persistent=False)
        # heads (per sub-hour of a frame: three)
        m = c.mixture
        self.timing = nn.Linear(d, 3 * k_states * k_states)
        self.initial = nn.Linear(d, k_states)
        self.amount = nn.Linear(d, 3 * 3 * m) if k_states == 2 else nn.Linear(d, 3 * (k_states - 1) * 2)
        self.thunder_head = nn.Linear(d, 3) if c.thunder else None
        self.kind_head = nn.Linear(d, 3 * 3) if c.precip_type else None
        self.cause = nn.Linear(d, len(REGIMES))
        self.n_waves = k_states * (k_states - 1)
        self.diurnal = nn.Linear(d, self.n_waves * 2 * c.harmonics)
        self.oro_gain = nn.Parameter(torch.tensor(0.0)) if k_states == 2 else nn.Parameter(torch.full((k_states - 1,), -1.0))
        self.orography = OrographicRain(geo.tiles) if (c.orography and geo.tiles) else None
        if self.orography is not None:
            self.mesh_where = self.orography.locate(mesh.lat, mesh.lon)
            self.target_where = self.orography.locate(targets.lat, targets.lon)
            self.station_where = (self.orography.locate(stations.lat, stations.lon) if stations is not None else None)
        self.apply(self._init)
        from ihelix.model import init_weights

        # iHELIX's own initialisation for its blocks (the temporal strand's decay and step parameters are created
        # empty and only set here; the attention biases start at zero)
        self.blocks.apply(lambda module: init_weights(module, icfg))
        for module in (self.read_hourly, self.read_upper, self.decode, self.read_obs):
            if module is not None:
                module.apply(lambda mod: init_weights(mod, icfg))
        self._init_heads()

    # ------------------------------------------------------------------------------------- set-up --
    @staticmethod
    def _static(points: PointSet) -> torch.Tensor:
        kind = np.eye(3, dtype=np.float32)[np.clip(points.kind, 0, 2)]
        return torch.tensor(np.concatenate([points.statics, kind], -1), dtype=torch.float32)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_heads(self) -> None:
        """Start at a sensible Caribbean climatology (transitions of :data:`pluvium.chain.PRIOR_TRANSITIONS`), with
        amounts centred in each state's range and an uninformative cause."""
        k = self.config.states
        with torch.no_grad():
            prior = torch.tensor(PRIOR_TRANSITIONS[k]).log()
            self.timing.bias.view(3, k, k).copy_(prior.expand(3, k, k))
            self.initial.bias.copy_(torch.tensor(PRIOR_INITIAL[k]).log())
            m = self.config.mixture
            if k == 2:
                a = self.amount.bias.view(3, 3, m)
                a[:, 0] = 0.0
                a[:, 1] = torch.linspace(-0.7, 1.6, m)
                a[:, 2] = math.log(0.7)
            else:
                a = self.amount.bias.view(3, k - 1, 2)
                a[..., 0] = 0.0
                a[..., 1] = math.log(math.expm1(0.45))       # sigma = 0.15 + softplus -> 0.6
            if self.thunder_head is not None:
                self.thunder_head.bias.fill_(math.log(0.02 / 0.98))
            self.diurnal.weight.mul_(0.1)

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        device = next(self.parameters()).device
        for name in ("latent_grid", "hourly_grid", "upper_grid"):
            grid = getattr(self, name)
            for key, value in list(grid.__dict__.items()):
                if isinstance(value, torch.Tensor):
                    setattr(grid, key, value.to(device))
        for name in ("link_hourly", "link_upper", "link_targets", "link_stations", "link_obs"):
            link = getattr(self, name, None)
            if link is not None:
                link.to(device)
        return out

    def gradient_checkpointing_enable(self, enable: bool = True) -> None:
        self.gradient_checkpointing = enable

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _maybe_checkpoint(self, fn, *args):
        """Run ``fn`` under activation checkpointing when gradient checkpointing is on (training only)."""
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    # ------------------------------------------------------------------------------------ forward --
    def _orography_for(self, batch: dict, wheres: list):
        """The orographic rate ``(B, F, N)`` at several located point sets (one solve), or Nones."""
        if self.orography is None or not any(w is not None for w in wheres):
            return [None] * len(wheres)
        tiles = batch["tiles"]
        live = [w for w in wheres if w is not None]
        got = iter(self.orography.rates_at(tiles[..., 0], tiles[..., 1], tiles[..., 2], live))
        return [next(got) if w is not None else None for w in wheres]

    def _where(self, name: str, rows=None):
        where = getattr(self, name, None)
        if where is None or rows is None:
            return where
        r = rows.cpu().numpy()
        return {"tile": where["tile"][r], "xy": where["xy"][r]}

    def encode(self, batch: dict, oro_mesh: torch.Tensor | None = None) -> torch.Tensor:
        """The processed mesh, ``(B, F, N_mesh, hidden)``."""
        hourly = batch["hourly"].index_select(2, self.hourly_idx)          # (B, F, C, H, W)
        b, f = hourly.shape[:2]
        n = self.latent_grid.num_points
        dtype = self.hourly_in.weight.dtype
        src1 = self.hourly_in(hourly.flatten(3).transpose(2, 3).to(dtype))      # (B, F, H*W, d)
        src2 = self.upper_in(batch["upper"].flatten(3).transpose(2, 3).to(dtype))
        if oro_mesh is None:
            oro_mesh = self._orography_for(batch, [getattr(self, "mesh_where", None)])[0]
        dyn = batch["mesh_ctx"].to(dtype)
        dyn = torch.cat([dyn, (oro_mesh if oro_mesh is not None else dyn.new_zeros(b, f, n)).unsqueeze(-1).to(dtype)], -1)
        x = (self.mesh_static_in(self.mesh_static)[None, None] + self.mesh_dyn_in(dyn)
             + self.frame_in(self.frame_embed)[None, :, None])
        flat = x.reshape(b * f, n, -1)
        flat = flat + self._maybe_checkpoint(self.read_hourly, flat, src1.reshape(b * f, src1.shape[2], -1),
                                             self.link_hourly)
        flat = flat + self._maybe_checkpoint(self.read_upper, flat, src2.reshape(b * f, src2.shape[2], -1),
                                             self.link_upper)
        x = flat.view(b, f, n, -1)
        if self.read_obs is not None and "station_obs" in batch:
            hist = self.history.numel()                                   # history frames come first
            obs = self.obs_in(batch["station_obs"].to(dtype))               # (B, H, S, d)
            xh = x[:, :hist].reshape(b * hist, n, -1)
            xh = xh + self._maybe_checkpoint(self.read_obs, xh, obs.reshape(b * hist, obs.shape[2], -1), self.link_obs)
            x = torch.cat([xh.view(b, hist, n, -1), x[:, hist:]], dim=1)
        x = self.encode_norm(x)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, self.latent_grid, use_reentrant=False)
            else:
                x = block(x, self.latent_grid)
        return self.norm(x)

    def _decode_body(self, q: torch.Tensor, latent: torch.Tensor, link) -> torch.Tensor:
        h = q + self.decode(q, latent, link)
        h = self.decode_norm(h)
        h = h + self.decode_ffn(h)
        return self.head_norm(h)

    def _refine(self, h: torch.Tensor, nbr: torch.Tensor, nbr_km: torch.Tensor) -> torch.Tensor:
        return self.refiner(h, nbr, nbr_km)

    def decode_points(self, latent: torch.Tensor, batch: dict, static: torch.Tensor, ctx: torch.Tensor, link,
                      lon: torch.Tensor, oro: torch.Tensor | None, patches: torch.Tensor | None = None,
                      neighbours: tuple | None = None) -> dict:
        """Heads at decoder points: ``static (P, S)``, ``ctx (B, F, P, D)`` per-frame context, ``lon (P,)``;
        ``patches (P, 2, 16, 16)`` terrain around the points and ``neighbours`` ``(index (P, k), km (P, k))`` for the
        version 3 engine."""
        c = self.config
        k_states = c.states
        b = latent.shape[0]
        fut = self.future
        lat_f = latent[:, fut]                                                   # (B, Ff, N, d)
        ff, p = lat_f.shape[1], static.shape[0]
        dtype = lat_f.dtype
        ctx_f = ctx[:, fut].index_select(-1, self.point_idx).to(dtype)
        oro_f = oro[:, fut].to(dtype) if oro is not None else ctx_f.new_zeros(b, ff, p)
        static_q = self.point_static_in(static.to(dtype))
        if self.patch_encoder is not None and patches is not None:
            static_q = static_q + self.patch_encoder(patches).to(dtype)
        q = (static_q[None, None] + self.point_dyn_in(torch.cat([ctx_f, oro_f.unsqueeze(-1)], -1))
             + self.frame_in(self.frame_embed[fut].to(dtype))[None, :, None])
        q = q.reshape(b * ff, p, -1)
        h = self._maybe_checkpoint(self._decode_body, q, lat_f.reshape(b * ff, lat_f.shape[2], -1), link)
        h = h.view(b, ff, p, -1)
        if self.film is not None:
            h = self.film(h, self.frame_embed[fut])
        if self.temporal is not None:
            h = self._maybe_checkpoint(self.temporal, h)
        if self.refiner is not None and neighbours is not None and p > 1:
            h = self._maybe_checkpoint(self._refine, h, neighbours[0], neighbours[1])
        if self.experts is not None:
            h = self._maybe_checkpoint(self.experts, h)
        if self.head_dropout is not None:
            h = self.head_dropout(h)
        hf, hs = self.hour_frame, self.hour_sub
        trans = self.timing(h).view(b, ff, p, 3, k_states, k_states).permute(0, 1, 3, 2, 4, 5)[:, hf, hs].float()
        wet_cols = torch.zeros_like(trans)
        if c.diurnal_harmonics:
            day0 = h[:, : min(24, ff)].mean(1)                                    # (B, P, d)
            coef = self.diurnal(day0).view(b, p, self.n_waves, 2, c.harmonics).float()
            t0 = batch["t0"].to(torch.float64).view(b, 1, 1)
            hours = torch.arange(self.schedule.num_hours, device=h.device, dtype=torch.float64).view(1, -1, 1)
            lst = ((t0 + hours + 0.5 + lon.view(1, 1, -1).double() / 15.0) % 24.0).float()   # (B, 72, P)
            kk = torch.arange(1, c.harmonics + 1, device=h.device).float()
            angle = 2 * math.pi * lst[..., None] * kk / 24.0                      # (B, 72, P, H)
            wave = (coef[:, None, :, :, 0, :] * torch.cos(angle)[:, :, :, None, :]
                    + coef[:, None, :, :, 1, :] * torch.sin(angle)[:, :, :, None, :]).sum(-1)   # (B, 72, P, waves)
            wet_cols = wet_cols + F.pad(wave.view(b, -1, p, k_states, k_states - 1), (1, 0))
        out = {}
        amount = None
        if k_states == 2:
            m = c.mixture
            amount = self.amount(h).view(b, ff, p, 3, 3, m).permute(0, 1, 3, 2, 4, 5)[:, hf, hs].float()
            mu = amount[..., 1, :]
            if oro is not None:
                up = F.relu(oro[:, fut][:, hf].float())
                mu = mu + F.softplus(self.oro_gain) * torch.log1p(up).unsqueeze(-1)
            out.update(mix_logit=amount[..., 0, :], mix_mu=mu, mix_logsig=amount[..., 2, :].clamp(-3.0, 2.0))
        else:
            raw = self.amount(h).view(b, ff, p, 3, k_states - 1, 2).permute(0, 1, 3, 2, 4, 5)[:, hf, hs].float()
            centres = torch.tensor(CENTRES[: k_states - 1], device=h.device).log()
            mu = raw[..., 0] + centres
            if oro is not None:
                up = torch.log1p(F.relu(oro[:, fut][:, hf].float()))                 # (B, 72, P)
                gain = F.softplus(self.oro_gain)                                    # per wet state
                wet_cols = wet_cols + F.pad((gain * up.unsqueeze(-1)).unsqueeze(-2).expand(-1, -1, -1, k_states, -1),
                                            (1, 0))
                mu = mu + 0.1 * gain * up.unsqueeze(-1)
            out.update(amt_mu=mu, amt_sig=0.15 + F.softplus(raw[..., 1]))
        out["trans"] = trans + wet_cols
        if self.shrink is not None:
            out["trans"] = self.shrink(out["trans"])
        out["init"] = self.initial(h[:, 0]).float()
        if self.thunder_head is not None:
            out["thunder"] = self.thunder_head(h).view(b, ff, p, 3).permute(0, 1, 3, 2)[:, hf, hs].float()
        if self.kind_head is not None:
            out["kind"] = self.kind_head(h).view(b, ff, p, 3, 3).permute(0, 1, 3, 2, 4)[:, hf, hs].float()
        out["cause"] = self.cause(h).float()
        return out

    def forward(self, batch: dict) -> dict:
        rows = batch["targets"]
        if rows.dim() > 1:
            rows = rows[0]
        with_stations = "station_ctx" in batch and self.link_stations is not None
        oro_mesh, oro_t, oro_s = self._orography_for(batch, [self._where("mesh_where"), self._where("target_where", rows),
                                                             self._where("station_where") if with_stations else None])
        latent = self.encode(batch, oro_mesh)
        link = SubLink(self.link_targets, rows)
        patches = self.target_patches[rows] if self.patch_encoder is not None else None
        nbr = None
        if self.refiner is not None:
            r = rows.cpu().numpy()
            i, km = self._knn(self.geo.targets.lat[r], self.geo.targets.lon[r], self.config.refine_neighbours)
            nbr = (torch.from_numpy(i).to(rows.device), torch.from_numpy(km).to(rows.device))
        out = self.decode_points(latent, batch, self.target_static[rows], batch["target_ctx"], link,
                                 self.target_lon[rows], oro_t, patches, nbr)
        if with_stations:
            st = self.decode_points(latent, batch, self.station_static, batch["station_ctx"], self.link_stations,
                                    self.station_lon, oro_s, self.station_patches if self.patch_encoder is not None else None,
                                    (self.station_nbr, self.station_nbr_km) if self.refiner is not None else None)
            out.update({f"station_{k}": v for k, v in st.items()})
        return out

    def forecast_points(self, batch: dict, points: PointSet, ctx: torch.Tensor, chunk: int = 4000,
                        latent: torch.Tensor | None = None) -> dict:
        """Heads at any points (towns): links and orography are built on the fly, in chunks."""
        from ihelix import GridLink

        latent = self.encode(batch) if latent is None else latent
        device = latent.device
        static = self._static(points).to(device)
        lon = torch.tensor(points.lon, dtype=torch.float32, device=device)
        parts = []
        for start in range(0, len(points), chunk):
            sub = points.subset(np.arange(start, min(len(points), start + chunk)))
            link = GridLink(sub.source_grid(), self.latent_grid, min(self.config.decode_neighbours,
                                                                      self.latent_grid.num_points)).to(device)
            oro = None
            if self.orography is not None:
                oro = self._orography_for(batch, [self.orography.locate(sub.lat, sub.lon)])[0]
            sl = slice(start, start + len(sub))
            patches = nbr = None
            if self.patch_encoder is not None:
                from .engine import terrain_patches

                patches = torch.from_numpy(terrain_patches(self.geo.terrain, sub.lat, sub.lon)).to(device)
            if self.refiner is not None:
                i, km = self._knn(sub.lat, sub.lon, self.config.refine_neighbours)
                nbr = (torch.from_numpy(i).to(device), torch.from_numpy(km).to(device))
            parts.append(self.decode_points(latent, batch, static[sl], ctx[:, :, sl], link, lon[sl], oro, patches, nbr))
        return {k: torch.cat([p[k] for p in parts], dim=POINT_AXIS.get(k, 2)) for k in parts[0]}


__all__ = ["POINT_AXIS", "PluviumConfig", "PluviumModel", "SubLink", "VERSION_1", "VERSION_2"]
