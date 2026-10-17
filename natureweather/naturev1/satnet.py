# Copyright 2026 Nathan. Apache-2.0.
"""
The satellite path, rebuilt around what went wrong.

The red team's findings, and what each network here is for:

* **Intensity was biased low by 10-18 kt for major hurricanes, and a 50-to-130 kt explosion (Otis) was
  called 45 kt.** A 1.5-degree reanalysis cannot see an eyewall. :class:`SatNetStorm` reads a four-frame
  loop of storm-centred 11-micron and 6.7-micron imagery and estimates intensity directly, the way the
  Dvorak technique and its successors do -- and is trained on every basin, about 230,000 fixes since
  1980, not on 16,000 Atlantic ones.
* **A storm-centre position was required as input.** :class:`SatNetStorm` is given only a first guess,
  deliberately wrong by up to 150 km in training, and has to say where the centre really is. Two passes
  (guess, refine, read again) take a rough position from a vortex tracker or an old advisory to a fix.
* **Rapid intensification was the weakest output.** RI is predicted here as an ordinal family -- the
  chance of gaining at least 25, 30, 35, 40 kt in 24 h always falls as the threshold rises -- alongside
  the 12-72 h intensity change, from inner-core structure in the imagery (a cold ring closing, an eye
  clearing, a burst over the centre) that the reanalysis does not have.
* **Southern-hemisphere storms were tracked worse than physics alone.** Everything storm-centred works in
  a hemisphere-normalised frame: southern images are flipped north-south (so every cyclone turns the same
  way), and every north-south output is flipped back. Both hemispheres then teach one model.
* **Satellites do not see the whole atmosphere.** :class:`SatelliteRetrieval` therefore never pretends
  to: it brings an aged state (6 to 48 hours old) up to date using a three-frame global loop of infrared
  and water vapour, cloud-motion vectors measured from the loop, and which satellite saw each pixel at
  what angle. Its output starts at exactly "the stale state, unchanged" and has to earn every change, with
  an uncertainty on each one.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn.functional as F
from torch import nn

from ihelix import FieldGrid, Geometry, GridToMesh, SphericalConv2d, latlon_channels

from .ibtracs import CHANGE_HOURS
from .satstage import GLOBAL_CHANNELS, SATELLITE_FAMILIES, band_axes, decode_global
from .stormsat import PATCH_DEGREES, decode_bt

KM_PER_DEGREE = 111.195
#: Rapid-intensification thresholds per window: (hours, (knots, ...)). Within a window they are ordinal.
#: These are the SHIPS-RII definitions -- 25/30/35/40 kt in 24 h, 45 in 36, 55 in 48, 65 in 72.
RI_WINDOWS = ((24, (25.0, 30.0, 35.0, 40.0)), (36, (45.0,)), (48, (55.0,)), (72, (65.0,)))
#: Global base rates (IBTrACS 1980-2024, tropical fixes), used to start every RI logit at climatology.
RI_PRIORS = {24: (0.099, 0.063, 0.041, 0.026), 36: (0.040,), 48: (0.035,), 72: (0.030,)}
#: Quadrant order NE, SE, SW, NW: what a north-south flip does to it.
FLIP_QUADRANTS = (1, 0, 3, 2)
WIND_ANCHOR_KT = (55.0, 30.0)
PRESSURE_ANCHOR = (990.0, 20.0)


def ri_threshold_list() -> list[tuple[int, float]]:
    return [(hours, kt) for hours, thresholds in RI_WINDOWS for kt in thresholds]


def ordinal_logits(level: torch.Tensor, gaps: torch.Tensor) -> torch.Tensor:
    """
    ``(B, windows)`` levels and ``(B, thresholds)`` raw gaps into ``(B, thresholds)`` RI logits that never
    rise with the threshold inside a window: P(gain >= 40 kt) <= P(>= 35) <= P(>= 30) <= P(>= 25).
    """
    steps = F.softplus(gaps)
    logits, start = [], 0
    for window, (_, thresholds) in enumerate(RI_WINDOWS):
        step = steps[:, start : start + len(thresholds)]
        step = torch.cat([torch.zeros_like(step[:, :1]), step[:, 1:]], -1)
        logits.append(level[:, window : window + 1] - step.cumsum(-1))
        start += len(thresholds)
    return torch.cat(logits, -1)


def init_ordinal(level: nn.Linear, gaps: nn.Linear, weight_scale: float = 0.1) -> None:
    """Start an ordinal RI head at the global base rates (:data:`RI_PRIORS`), with small weights."""
    with torch.no_grad():
        level.weight.mul_(weight_scale)
        gaps.weight.mul_(weight_scale)
        level.bias.zero_()
        gaps.bias.zero_()
        start = 0
        for window, (hours, thresholds) in enumerate(RI_WINDOWS):
            logits = [math.log(p / (1 - p)) for p in RI_PRIORS[hours]]
            level.bias[window] = logits[0]
            for k in range(1, len(thresholds)):
                gap = max(logits[k - 1] - logits[k], 1e-3)
                gaps.bias[start + k] = math.log(math.expm1(gap))
            start += len(thresholds)


@dataclass
class SatNetConfig:
    """Size of :class:`SatNetStorm`. The defaults are about 21M parameters."""

    frames: int = 4
    pixels: int = 256
    widths: tuple[int, ...] = (64, 128, 256, 384)
    depths: tuple[int, ...] = (2, 2, 3, 3)
    token_layers: int = 4
    heads: int = 8
    embed: int = 512
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> SatNetConfig:
        known = {f.name for f in fields(cls)}
        payload = {k: (tuple(v) if isinstance(v, list) else v) for k, v in values.items() if k in known}
        return cls(**payload)


class ResBlock(nn.Module):
    """Two 3x3 convolutions with a skip; the second starts at zero so a new block is an identity."""

    def __init__(self, channels: int, conv=nn.Conv2d) -> None:
        super().__init__()
        padding = {} if conv is not nn.Conv2d else {"padding": 1}
        self.body = nn.Sequential(
            nn.GroupNorm(8, channels), nn.GELU(), conv(channels, channels, 3, **padding),
            nn.GroupNorm(8, channels), nn.GELU(), conv(channels, channels, 3, **padding),
        )
        nn.init.zeros_(self.body[-1].weight)
        if self.body[-1].bias is not None:
            nn.init.zeros_(self.body[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


def _down(cin: int, cout: int, conv=nn.Conv2d) -> nn.Module:
    padding = {} if conv is not nn.Conv2d else {"padding": 1}
    return nn.Sequential(conv(cin, cout, 3, stride=2, **padding), nn.GroupNorm(8, cout), nn.GELU())


def hemisphere_flip(latitude: torch.Tensor) -> torch.Tensor:
    """``(B,)`` True for southern-hemisphere samples."""
    return latitude < 0


class SatNetStorm(nn.Module):
    """
    A storm-centred satellite loop in; intensity, centre, structure, RI and motion out.

    Inputs are ``(B, F, 2, P, P)`` uint8 brightness temperatures (11 um window, 6.7 um water vapour;
    :func:`naturev1.stormsat.encode_bt`) for frames now, -3, -6, -9 h, all cut around the same first
    guess of the centre, north up; ``(B, F)`` availability; ``(B,)`` first-guess latitude; ``(B, 4)``
    context (|lat| / 30, local solar time as sin/cos, share of frames present).

    How it reads them:

    * one convolutional encoder per frame (shared weights) down to 64 x 64,
    * the frames fused there -- now, and now minus each earlier frame, which is where motion and
      convective trends live -- and on down to a 16 x 16 grid,
    * a transformer over those 256 locations plus a summary token, so any part of the storm can be
      related to any other (an eye to its eyewall, a rainband to the shear direction),
    * the Dvorak measurements, fixed rather than learned: mean temperature and cold-cloud share in 25 km
      rings, eye-to-eyewall contrast, and the same in water vapour.

    Every output is in physical units with an uncertainty, and every north-south quantity is in the
    storm's own hemisphere frame on the inside and flipped back on the way out.
    """

    RING_KM = 25.0
    RINGS = 12

    def __init__(self, config: SatNetConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or SatNetConfig()
        widths, depths = config.widths, config.depths
        pixels = config.pixels
        offsets = (torch.arange(pixels, dtype=torch.float32) - (pixels - 1) / 2.0) * PATCH_DEGREES * KM_PER_DEGREE
        north, east = torch.meshgrid(offsets, offsets, indexing="ij")
        radius = torch.sqrt(north ** 2 + east ** 2)
        polar = torch.stack([radius / radius.max(), torch.atan2(north, east).cos(), torch.atan2(north, east).sin()])
        self.register_buffer("polar", polar, persistent=False)
        rings = torch.stack([(radius >= i * self.RING_KM) & (radius < (i + 1) * self.RING_KM)
                             for i in range(self.RINGS)]).float()
        self.register_buffer("rings", rings, persistent=False)
        self.register_buffer("eye", (radius < 20.0).float(), persistent=False)

        self.stem = nn.Sequential(nn.Conv2d(8, widths[0], 5, stride=2, padding=2), nn.GroupNorm(8, widths[0]), nn.GELU(),
                                  *[ResBlock(widths[0]) for _ in range(depths[0])])
        self.stage2 = nn.Sequential(_down(widths[0], widths[1]), *[ResBlock(widths[1]) for _ in range(depths[1])])
        fused = widths[1] * (2 * config.frames - 1) + config.frames
        self.fuse = nn.Sequential(nn.Conv2d(fused, widths[2], 1), nn.GroupNorm(8, widths[2]), nn.GELU())
        self.stage3 = nn.Sequential(_down(widths[2], widths[2]), *[ResBlock(widths[2]) for _ in range(depths[2])])
        self.stage4 = nn.Sequential(_down(widths[2], widths[3]), *[ResBlock(widths[3]) for _ in range(depths[3])])
        grid = pixels // 16
        self.position = nn.Parameter(torch.zeros(1, grid * grid + 1, widths[3]))
        self.summary_token = nn.Parameter(torch.zeros(1, 1, widths[3]))
        layer = nn.TransformerEncoderLayer(widths[3], config.heads, widths[3] * 4, config.dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.tokens = nn.TransformerEncoder(layer, config.token_layers, enable_nested_tensor=False)
        profile = 3 * self.RINGS + 2 + 4                  # rings, eye contrast, cold core, and the context
        self.trunk = nn.Sequential(nn.LayerNorm(widths[3] + profile), nn.Linear(widths[3] + profile, config.embed),
                                   nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.embed, config.embed), nn.GELU())
        e = config.embed
        self.intensity = nn.Linear(e, 4)                  # wind kt (mean, log var), pressure hPa (mean, log var)
        self.centre = nn.Linear(e, 4)                     # true centre - first guess, km (north, east), log var each
        self.radii = nn.Linear(e, 3 * 4 * 2)              # 34/50/64 kt x NE/SE/SW/NW, nmi, mean and log var
        self.rmw = nn.Linear(e, 2)
        self.change = nn.Linear(e, len(CHANGE_HOURS) * 2)  # wind change at +12..+72 h, kt
        self.ri_level = nn.Linear(e, len(RI_WINDOWS))
        self.ri_gaps = nn.Linear(e, sum(len(t) for _, t in RI_WINDOWS))
        self.motion = nn.Linear(e, 8)                     # past 6 h and next 12 h displacement, km, with log var
        self._init_heads()

    def _init_heads(self) -> None:
        with torch.no_grad():
            nn.init.trunc_normal_(self.position, std=0.02)
            nn.init.trunc_normal_(self.summary_token, std=0.02)
            for head in (self.intensity, self.centre, self.radii, self.rmw, self.change, self.motion):
                head.weight.mul_(0.1)
                head.bias.zero_()
            self.intensity.bias[1] = 2 * math.log(WIND_ANCHOR_KT[1])
            self.intensity.bias[3] = 2 * math.log(PRESSURE_ANCHOR[1])
            self.centre.bias[2:] = 2 * math.log(60.0)
            self.radii.bias[1::2] = 2 * math.log(60.0)
            self.rmw.bias[1] = 2 * math.log(15.0)
            self.change.bias[1::2] = 2 * math.log(15.0)
            self.motion.bias.view(2, 4)[:, 2:] = 2 * math.log(80.0)
        # The ordinal RI logits open at the global base rates: level - cumulative gaps.
        init_ordinal(self.ri_level, self.ri_gaps)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _inputs(self, codes: torch.Tensor, available: torch.Tensor, south: torch.Tensor):
        kelvin = decode_bt(codes)                                            # (B, F, 2, P, P), NaN missing
        flip = south.view(-1, 1, 1, 1, 1)
        kelvin = torch.where(flip, kelvin.flip(-2), kelvin)
        ir, wv = kelvin[:, :, 0], kelvin[:, :, 1]
        ok_ir, ok_wv = torch.isfinite(ir), torch.isfinite(wv)
        frame_ok = available.view(*available.shape, 1, 1).to(ir.device)
        channels = torch.stack([
            torch.nan_to_num((ir - 250.0) / 30.0), torch.nan_to_num((wv - 240.0) / 20.0),
            torch.where(ok_ir & ok_wv, (wv - ir) / 10.0, torch.zeros_like(ir)),
            (ok_ir & frame_ok).float(), (ok_wv & frame_ok).float(),
        ], 2) * frame_ok.unsqueeze(2).float()
        polar = self.polar.to(channels.dtype).view(1, 1, 3, *self.polar.shape[-2:]).expand(
            channels.shape[0], channels.shape[1], -1, -1, -1)
        return torch.cat([channels, polar], 2), ir[:, 0], wv[:, 0], ok_ir[:, 0], ok_wv[:, 0]

    def _profile(self, ir: torch.Tensor, wv: torch.Tensor, ok_ir: torch.Tensor, ok_wv: torch.Tensor) -> torch.Tensor:
        rings = self.rings.to(ir.dtype)
        weight = rings.unsqueeze(0) * ok_ir.unsqueeze(1).to(ir.dtype)
        count = weight.sum((-1, -2)).clamp_min(1.0)
        filled = torch.nan_to_num(ir).unsqueeze(1)
        ring_mean = (weight * filled).sum((-1, -2)) / count
        ring_cold = (weight * (filled < 213.15).to(ir.dtype)).sum((-1, -2)) / count
        weight_wv = rings.unsqueeze(0) * ok_wv.unsqueeze(1).to(ir.dtype)
        ring_wv = (weight_wv * torch.nan_to_num(wv).unsqueeze(1)).sum((-1, -2)) / weight_wv.sum((-1, -2)).clamp_min(1.0)
        eye = self.eye.to(ir.dtype) * ok_ir.to(ir.dtype)
        eye_temp = (eye * torch.nan_to_num(ir)).sum((-1, -2)) / eye.sum((-1, -2)).clamp_min(1.0)
        contrast = (eye_temp - ring_mean[:, 1:3].mean(-1)) / 30.0
        cold_core = ring_cold[:, :4].mean(-1)
        return torch.cat([(ring_mean - 250.0) / 30.0, ring_cold, (ring_wv - 240.0) / 20.0,
                          contrast.unsqueeze(-1), cold_core.unsqueeze(-1)], -1)

    def forward(self, codes: torch.Tensor, available: torch.Tensor, latitude: torch.Tensor,
                context: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch, frames = codes.shape[:2]
        south = hemisphere_flip(latitude.to(codes.device))
        available = available.to(codes.device).bool()
        x, ir, wv, ok_ir, ok_wv = self._inputs(codes, available, south)
        dtype = self.stem[0].weight.dtype
        x = x.to(dtype)
        features = self.stage2(self.stem(x.flatten(0, 1)))                    # (B*F, w1, 64, 64)
        features = features.view(batch, frames, *features.shape[1:])
        features = features * available.view(batch, frames, 1, 1, 1).to(features.dtype)
        now = features[:, 0]
        trends = [now - features[:, k] for k in range(1, frames)]
        flags = available.to(features.dtype).view(batch, frames, 1, 1).expand(-1, -1, *now.shape[-2:])
        fused = self.fuse(torch.cat([now, *[features[:, k] for k in range(1, frames)], *trends, flags], 1))
        grid = self.stage4(self.stage3(fused))                                # (B, w3, 16, 16)
        tokens = grid.flatten(2).transpose(1, 2)
        tokens = torch.cat([self.summary_token.expand(batch, -1, -1).to(tokens.dtype), tokens], 1)
        tokens = self.tokens(tokens + self.position.to(tokens.dtype))
        profile = self._profile(ir, wv, ok_ir, ok_wv).to(tokens.dtype)
        if context is None:
            context = torch.zeros(batch, 4, device=codes.device)
        h = self.trunk(torch.cat([tokens[:, 0], profile, context.to(tokens.dtype)], -1))
        return self._heads(h, south)

    def _heads(self, h: torch.Tensor, south: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = h.shape[0]
        sign = torch.where(south, -1.0, 1.0).to(h.dtype).view(-1, 1)
        intensity = self.intensity(h)
        centre = self.centre(h)
        centre_mean = centre[:, :2] * torch.cat([sign, torch.ones_like(sign)], -1) * 50.0
        radii = self.radii(h).view(batch, 3, 4, 2)
        flipped = radii[:, :, list(FLIP_QUADRANTS)]
        radii = torch.where(south.view(-1, 1, 1, 1), flipped, radii)
        rmw = self.rmw(h)
        change = self.change(h).view(batch, len(CHANGE_HOURS), 2)
        motion = self.motion(h).view(batch, 2, 4)
        motion_mean = motion[..., :2] * torch.cat([sign, torch.ones_like(sign)], -1).unsqueeze(1) * 100.0
        return {
            "embedding": h,
            "wind_kt": intensity[:, 0] * WIND_ANCHOR_KT[1] + WIND_ANCHOR_KT[0],
            "wind_log_var": intensity[:, 1].clamp(-6.0, 10.0),
            "pressure_hpa": intensity[:, 2] * PRESSURE_ANCHOR[1] + PRESSURE_ANCHOR[0],
            "pressure_log_var": intensity[:, 3].clamp(-6.0, 10.0),
            "centre_offset_km": centre_mean,
            "centre_log_var": centre[:, 2:].clamp(-2.0, 14.0),
            "radii_nmi": F.softplus(radii[..., 0] * 60.0 + 60.0),
            "radii_log_var": radii[..., 1].clamp(-4.0, 12.0),
            "rmw_nmi": F.softplus(rmw[:, 0] * 15.0 + 25.0) + 2.0,
            "rmw_log_var": rmw[:, 1].clamp(-4.0, 10.0),
            "wind_change_kt": change[..., 0] * 15.0 + 2.0,
            "wind_change_log_var": change[..., 1].clamp(-6.0, 10.0),
            "ri_logits": ordinal_logits(self.ri_level(h), self.ri_gaps(h)),
            "past_motion_km": motion_mean[:, 0],
            "future_motion_km": motion_mean[:, 1],
            "motion_log_var": motion[..., 2:].clamp(-2.0, 14.0),
        }


def _nll(mean, log_var, target, mask=None):
    observed = torch.isfinite(target)
    if mask is not None:
        observed = observed & mask
    if not bool(observed.any()):
        return mean.sum() * 0.0
    safe = torch.where(observed, target, mean.detach())
    loss = 0.5 * (log_var + (safe - mean) ** 2 / log_var.exp())
    return (loss * observed).sum() / observed.sum()


def satnet_targets_ri(change_kt: torch.Tensor, tropical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """RI labels ``(B, n)`` in :func:`ri_threshold_list` order from ``(B, len(CHANGE_HOURS))`` changes."""
    targets, masks = [], []
    for hours, threshold in ri_threshold_list():
        change = change_kt[:, CHANGE_HOURS.index(hours)]
        ok = torch.isfinite(change) & tropical.bool()
        targets.append(((change >= threshold) & ok).float())
        masks.append(ok.float())
    return torch.stack(targets, -1), torch.stack(masks, -1)


def satnet_loss(out: dict, batch: dict, weights: dict | None = None) -> tuple[torch.Tensor, dict]:
    """
    Every SatNet task, each a likelihood, each masked where IBTrACS has no label.

    RI uses focal cross-entropy (rare positives) and the rest Gaussian negative log-likelihoods, so an
    uncertain answer is cheap and a confident wrong one is expensive.
    """
    from .losses import focal_bce

    w = {"wind": 1.0, "pressure": 0.5, "centre": 1.0, "radii": 0.3, "rmw": 0.2, "change": 0.7, "ri": 2.0,
         "motion": 0.3, **(weights or {})}
    device = out["wind_kt"].device
    take = (lambda key: batch[key].to(device).float())
    parts = {}
    parts["wind"] = _nll(out["wind_kt"], out["wind_log_var"], take("wind_kt"))
    parts["pressure"] = _nll(out["pressure_hpa"], out["pressure_log_var"], take("pressure_hpa"))
    parts["centre"] = _nll(out["centre_offset_km"], out["centre_log_var"], take("centre_offset_km"))
    parts["radii"] = _nll(out["radii_nmi"], out["radii_log_var"], take("radii_nmi"))
    parts["rmw"] = _nll(out["rmw_nmi"], out["rmw_log_var"], take("rmw_nmi"))
    tropical = batch["tropical"].to(device).bool()
    parts["change"] = _nll(out["wind_change_kt"], out["wind_change_log_var"], take("wind_change_kt"),
                           tropical.view(-1, 1).expand(-1, len(CHANGE_HOURS)))
    target, mask = satnet_targets_ri(take("wind_change_kt"), tropical)
    parts["ri"] = focal_bce(out["ri_logits"].float(), target, mask=mask)
    motion = torch.stack([take("past_motion_km"), take("future_motion_km")], 1)
    parts["motion"] = _nll(torch.stack([out["past_motion_km"], out["future_motion_km"]], 1), out["motion_log_var"], motion)
    loss = sum(w[k] * v.float() for k, v in parts.items())
    return loss, {k: float(v.detach()) for k, v in parts.items()} | {"total": float(loss.detach())}


# ------------------------------------------------------------------------------------- cloud motion --

def cloud_motion(previous: torch.Tensor, current: torch.Tensor, hours: float, resolution_deg: float,
                 latitude: torch.Tensor, tile: int = 16, search: int = 16, stride: int = 2) -> torch.Tensor:
    """
    Cloud-motion vectors from two images, by normalised cross-correlation over tiles.

    Operational atmospheric motion vectors track features between images minutes apart; GridSat is
    three-hourly, so clouds change shape between frames and these are coarse -- a feature for the network
    to weigh, with the peak correlation as its own quality flag, not a wind observation.

    Args:
        previous, current: ``(B, H, W)`` fields (NaN missing) on a regular grid, north up.
        hours: time between them.
        latitude: ``(H,)`` row latitudes, degrees, for converting eastward pixels to metres.

    Returns:
        ``(B, 3, H, W)``: eastward and northward motion (m/s), and peak correlation in [-1, 1] (0 where
        there was nothing to correlate), constant over each tile.
    """
    batch, height, width = current.shape
    rows, cols = height // tile, width // tile
    h, w = rows * tile, cols * tile
    cur, prev = current[:, :h, :w], previous[:, :h, :w]
    ok_c, ok_p = torch.isfinite(cur), torch.isfinite(prev)
    fill_c = torch.where(ok_c, cur, cur.new_zeros(()))
    fill_p = torch.where(ok_p, prev, prev.new_zeros(()))
    pool = (lambda x: F.avg_pool2d(x.unsqueeze(1), tile).squeeze(1))
    mean_c = pool(fill_c)
    var_c = (pool(fill_c ** 2) - mean_c ** 2).clamp_min(0)
    best = torch.full((batch, rows, cols), -2.0, dtype=cur.dtype, device=cur.device)
    best_dy = torch.zeros_like(best)
    best_dx = torch.zeros_like(best)
    for dy in range(-search, search + 1, stride):
        for dx in range(-search, search + 1, stride):
            shifted = torch.roll(fill_p, shifts=(dy, dx), dims=(1, 2))
            shifted_ok = torch.roll(ok_p, shifts=(dy, dx), dims=(1, 2))
            if dy > 0:
                shifted_ok[:, :dy] = False
            elif dy < 0:
                shifted_ok[:, dy:] = False
            both = (ok_c & shifted_ok).to(cur.dtype)
            share = pool(both)
            p = shifted * both
            c = fill_c * both
            mean_p = pool(p) / share.clamp_min(1e-6)
            mean_cc = pool(c) / share.clamp_min(1e-6)
            cov = pool(c * p) / share.clamp_min(1e-6) - mean_cc * mean_p
            var_p = (pool(p ** 2) / share.clamp_min(1e-6) - mean_p ** 2).clamp_min(0)
            var_cc = (pool(c ** 2) / share.clamp_min(1e-6) - mean_cc ** 2).clamp_min(0)
            score = cov / (var_p * var_cc).sqrt().clamp_min(1e-6)
            score = torch.where((share > 0.5) & (var_c > 1.0), score, torch.full_like(score, -2.0))
            better = score > best
            best = torch.where(better, score, best)
            best_dy = torch.where(better, torch.full_like(best_dy, float(dy)), best_dy)
            best_dx = torch.where(better, torch.full_like(best_dx, float(dx)), best_dx)
    seconds = hours * 3600.0
    metres = resolution_deg * KM_PER_DEGREE * 1000.0
    tile_lat = latitude[: h].view(rows, tile).mean(-1).to(cur.device, cur.dtype)
    v = best_dy * metres / seconds
    u = best_dx * metres * torch.cos(torch.deg2rad(tile_lat)).view(1, -1, 1).clamp_min(0.05) / seconds
    quality = torch.where(best > -1.5, best, torch.zeros_like(best))
    u, v = torch.where(best > -1.5, u, torch.zeros_like(u)), torch.where(best > -1.5, v, torch.zeros_like(v))
    out = torch.stack([u, v, quality], 1)
    out = out.repeat_interleave(tile, -2).repeat_interleave(tile, -1)
    return F.pad(out, (0, width - w, 0, height - h))


# --------------------------------------------------------------------------------- global retrieval --

class SatelliteRetrieval(nn.Module):
    """
    An aged atmospheric state, brought up to date by a global satellite loop.

    Args:
        hidden: width on the mesh (the backbone's ``hidden_size``).
        state_channels: prognostic channels predicted (the backbone's ``state_channels``).
        analysis_channels: channels of the background and of the output state.
        blocks: the processor -- :class:`ihelix.IHelixBlock` modules built exactly like the backbone's,
            so :meth:`NatureV1.warm_start_retrieval` can start them from its first blocks.
        latent_grid: the mesh.
        resolution: degrees per pixel of the staged global frames.

    Inputs at forward: ``satellite`` ``(B, 3, 8, H, W)`` uint8 frames now / -3 h / -6 h, ``available``
    ``(B, 3)``, ``background`` ``(B, P, C)`` the stale state on the analysis grid, ``stale_hours`` ``(B,)``,
    ``calendar`` ``(B, 6)``, and the analysis grid. Output: the increment for every prognostic channel in
    units of its six-hour tendency spread times sqrt(stale / 6), and a log-variance for each.
    """

    def __init__(self, hidden: int, state_channels: int, analysis_channels: int, blocks: nn.ModuleList,
                 latent_grid: FieldGrid, resolution: float = 0.25, num_heads: int = 8, num_kv_heads: int = 4,
                 head_dim: int = 64, encode_radius_km: float = 240.0, stem_widths=(96, 192, 320)) -> None:
        super().__init__()
        from ihelix import CrossAttention
        from ihelix.kernels import IHelixRMSNorm

        self.resolution = resolution
        lat, lon = band_axes(resolution)
        if len(lat) % 8 or len(lon) % 8:
            raise ValueError(f"the staged band ({len(lat)} x {len(lon)} at {resolution} deg) must divide by 8: the "
                             "stem halves it three times")
        self.register_buffer("band_lat", torch.tensor(lat, dtype=torch.float32), persistent=False)
        self.register_buffer("latlon", latlon_channels(lat, lon), persistent=False)
        self.family = nn.Embedding(len(SATELLITE_FAMILIES), 4)
        per_frame = 12
        frames = 3
        cin = per_frame * frames + 4 + 4 + 6
        conv = (lambda a, b, k, stride=1: SphericalConv2d(a, b, k, stride=stride, latitude="band"))
        w = stem_widths
        self.stem = nn.Sequential(
            conv(cin, w[0], 3, 2), nn.GroupNorm(8, w[0]), nn.GELU(), ResBlock(w[0], _band_conv),
            conv(w[0], w[1], 3, 2), nn.GroupNorm(8, w[1]), nn.GELU(), ResBlock(w[1], _band_conv),
            conv(w[1], w[2], 3, 2), nn.GroupNorm(8, w[2]), nn.GELU(), ResBlock(w[2], _band_conv),
            ResBlock(w[2], _band_conv), nn.Conv2d(w[2], hidden, 1),
        )
        rows, cols = len(lat) // 8, len(lon) // 8
        token_lat = torch.tensor(lat[: rows * 8], dtype=torch.float64).view(rows, 8).mean(-1)
        token_lon = torch.tensor(lon[: cols * 8], dtype=torch.float64).view(cols, 8).mean(-1)
        mesh_lat, mesh_lon = torch.meshgrid(token_lat, token_lon, indexing="ij")
        coords = torch.stack([torch.deg2rad(mesh_lat).reshape(-1), torch.deg2rad(mesh_lon).reshape(-1)], -1)
        self.token_grid = FieldGrid.source(coords, Geometry.globe(), weights=torch.cos(coords[:, 0]))
        self.coverage_pool = GridToMesh(lat, lon, latent_grid)
        geometry = latent_grid.geometry
        self.read_satellite = CrossAttention(hidden, num_heads, num_kv_heads, head_dim, geometry.embed_dim,
                                             encode_radius_km * 2)
        self.read_background = CrossAttention(hidden, num_heads, num_kv_heads, head_dim, geometry.embed_dim,
                                              encode_radius_km * 3)
        self.background_embed = nn.Linear(analysis_channels, hidden)
        self.satellite_norm = IHelixRMSNorm(hidden)
        self.background_norm = IHelixRMSNorm(hidden)
        self.mesh_seed = nn.Parameter(torch.zeros(1, 1, hidden))
        self.stale_embed = nn.Sequential(nn.Linear(4, 128), nn.SiLU(), nn.Linear(128, hidden))
        self.time_embed = nn.Sequential(nn.Linear(6, 256), nn.SiLU(), nn.Linear(256, hidden))
        self.gate = nn.Linear(hidden + 1, hidden)
        self.blocks = blocks
        self.norm = IHelixRMSNorm(hidden)
        self.decoder = CrossAttention(hidden, num_heads, num_kv_heads, head_dim, geometry.embed_dim,
                                      encode_radius_km * 2)
        self.query_embed = nn.Linear(analysis_channels, hidden)
        self.decode_norm = IHelixRMSNorm(hidden)
        self.head = nn.Linear(hidden, state_channels * 2)
        with torch.no_grad():
            # Exactly "the stale state, unchanged" until trained; log-variance at one tendency spread.
            self.head.weight.zero_()
            self.head.bias.zero_()
        self.state_channels = state_channels

    def frame_features(self, satellite: torch.Tensor, available: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, cin, H, W)`` stem input and ``(B, H, W)`` current infrared coverage."""
        fields = decode_global(satellite)                                  # each (B, 3, H, W)
        ok = available.to(satellite.device).view(*available.shape, 1, 1).float()
        ir, ir_min, wv = fields["ir_mean"], fields["ir_min"], fields["wv_mean"]
        both = torch.isfinite(ir) & torch.isfinite(wv)
        family = self.family(fields["satellite_family"].clamp(0, len(SATELLITE_FAMILIES) - 1))   # (B,3,H,W,4)
        zenith = fields["view_zenith"]
        per_frame = torch.cat([
            torch.stack([torch.nan_to_num((ir - 250.0) / 30.0), torch.nan_to_num((ir_min - 250.0) / 30.0),
                         fields["ir_std"] / 10.0, torch.nan_to_num((wv - 240.0) / 20.0),
                         torch.where(both, (ir - wv) / 20.0, torch.zeros_like(ir)),
                         fields["ir_coverage"], fields["wv_coverage"],
                         torch.nan_to_num(torch.cos(torch.deg2rad(zenith)))], 2),
            family.permute(0, 1, 4, 2, 3),
        ], 2) * ok.unsqueeze(2)
        batch = satellite.shape[0]
        trends = []
        for name, scale in (("ir_mean", 10.0), ("wv_mean", 10.0)):
            field = fields[name]
            for k in (1, 2):
                delta = (field[:, 0] - field[:, k]) / scale
                good = torch.isfinite(delta) & (ok[:, 0] > 0) & (ok[:, k] > 0)
                trends.append(torch.where(good, delta, torch.zeros_like(delta)))
        motion = []
        with torch.no_grad():                                             # inputs, not parameters
            for name in ("wv_mean", "ir_mean"):
                pair_ok = (available[:, 0] & available[:, 1]).to(satellite.device).view(-1, 1, 1, 1).float()
                vectors = cloud_motion(fields[name][:, 1].float(), fields[name][:, 0].float(), 3.0,
                                       self.resolution, self.band_lat)
                motion.append(torch.cat([vectors[:, :2] / 20.0, vectors[:, 2:]], 1) * pair_ok)
        stacked = torch.cat([per_frame.flatten(1, 2), self.latlon.unsqueeze(0).expand(batch, -1, -1, -1).to(per_frame.dtype),
                             torch.stack(trends, 1), *motion], 1)
        return stacked, fields["ir_coverage"][:, 0] * ok[:, 0]

    def forward(self, satellite: torch.Tensor, available: torch.Tensor, background: torch.Tensor,
                stale_hours: torch.Tensor, calendar: torch.Tensor, analysis_grid: FieldGrid, latent: FieldGrid,
                link) -> dict:
        """``link(target_grid, source_grid, neighbours)`` is the owner's cached-correspondence function."""
        batch = background.shape[0]
        dtype = self.head.weight.dtype
        inputs, coverage = self.frame_features(satellite, available)
        tokens = self.stem(inputs.to(dtype))                                 # (B, hidden, rows, cols)
        tokens = tokens.flatten(2).transpose(1, 2)
        seed = self.mesh_seed.expand(batch, latent.num_points, -1).to(dtype)
        sat_mesh = self.satellite_norm(self.read_satellite(seed, tokens, link(latent, self.token_grid, 24)))
        bg_mesh = self.background_norm(self.read_background(seed, self.background_embed(background.to(dtype)),
                                                            link(latent, analysis_grid, 24)))
        mesh_cover = self.coverage_pool(coverage.float())["mean"].to(dtype).unsqueeze(-1)       # (B, P, 1)
        stale = stale_hours.to(dtype).view(-1, 1)
        stale_features = torch.cat([stale / 24.0, torch.log1p(stale) / 4.0, torch.sqrt(stale / 6.0),
                                    torch.ones_like(stale)], -1)
        gate = torch.sigmoid(self.gate(torch.cat([sat_mesh, mesh_cover], -1))) * mesh_cover
        x = bg_mesh + gate * sat_mesh + (self.stale_embed(stale_features) + self.time_embed(calendar.to(dtype))).unsqueeze(1)
        x = x.unsqueeze(1)                                                   # one frame for the blocks
        for block in self.blocks:
            x = block(x, latent)
        x = self.norm(x[:, 0])
        queries = self.query_embed(background.to(dtype))
        decoded = self.decode_norm(self.decoder(queries, x, link(analysis_grid, latent, 24)))
        out = self.head(decoded).view(batch, analysis_grid.num_points, self.state_channels, 2)
        return {"increment": out[..., 0], "increment_log_var": out[..., 1].clamp(-10.0, 10.0),
                "coverage": mesh_cover[..., 0]}


def _band_conv(cin: int, cout: int, kernel: int, **_) -> nn.Module:
    return SphericalConv2d(cin, cout, kernel, latitude="band")


def increment_scale(stepper, stale_hours: torch.Tensor) -> torch.Tensor:
    """``(B, 1, n_prog)``: the tendency spread of each prognostic channel, grown as sqrt(stale / 6 h)."""
    prognostic = stepper.roles.prognostic
    tendency = stepper.tendency[prognostic].to(stale_hours.device, torch.float32)
    return tendency.view(1, 1, -1) * torch.sqrt(stale_hours.float().clamp_min(1.0) / 6.0).view(-1, 1, 1)


def retrieval_targets(stepper, background: torch.Tensor, target: torch.Tensor, stale_hours: torch.Tensor):
    """The true increment in the units :class:`SatelliteRetrieval` predicts it in."""
    prognostic = stepper.roles.prognostic
    return (target[..., prognostic].float() - background[..., prognostic].float()) / increment_scale(stepper, stale_hours)


def observation_weights(stepper, target: torch.Tensor) -> torch.Tensor:
    """``(B, P, n_prog)`` 1 where a prognostic channel is observed, 0 where its mask says it is not (SST on land)."""
    normalizer = stepper.normalizer
    weights = torch.ones(*target.shape[:2], len(stepper.roles.prognostic), device=target.device)
    variables = len(stepper.roles.channels)
    for k, index in enumerate(getattr(normalizer, "masked", [])):
        if index in stepper.roles.prognostic and variables + k < target.shape[-1]:
            weights[..., stepper.roles.prognostic.index(index)] = (target[..., variables + k] > 0.5).float()
    return weights


def retrieval_loss(out: dict, stepper, background: torch.Tensor, target: torch.Tensor, stale_hours: torch.Tensor,
                   area: torch.Tensor, channel_weights: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """
    Gaussian negative log-likelihood of the true increment, area- and channel-weighted.

    Also reports, in the same units, the mean squared error of the retrieval and of the stale state
    left alone: ``skill = 1 - retrieval / stale`` is the share of the change since the background that the
    satellites recovered. That is the number that says whether this path works.
    """
    truth = retrieval_targets(stepper, background, target, stale_hours)
    mean = out["increment"].float()
    log_var = out["increment_log_var"].float()
    weight = observation_weights(stepper, target) * area.to(target.device).view(1, -1, 1) * channel_weights.to(
        target.device).view(1, 1, -1)
    nll = 0.5 * (log_var + (truth - mean) ** 2 / log_var.exp())
    total = weight.sum().clamp_min(1e-12)
    loss = (nll * weight).sum() / total
    with torch.no_grad():
        mse = (((truth - mean) ** 2) * weight).sum() / total
        stale = ((truth ** 2) * weight).sum() / total
    return loss, {"retrieval": float(loss.detach()), "mse": float(mse), "stale_mse": float(stale),
                  "skill": float(1.0 - mse / stale.clamp_min(1e-12))}


def apply_retrieval(stepper, background: torch.Tensor, out: dict, stale_hours: torch.Tensor,
                    valid_time: torch.Tensor, conserve: bool = True) -> torch.Tensor:
    """
    The retrieved state now: the stale state with the predicted increment in its prognostic channels,
    solar forcing recomputed for the valid time, everything static left as it was. ``(B, P, C)``.
    """
    from .state import conserve_mass, solar_forcing

    prognostic = stepper.roles.prognostic
    state = background.clone().float()
    change = out["increment"].float() * increment_scale(stepper, stale_hours.to(background.device))
    if conserve and stepper.prognostic_mass is not None:
        change = conserve_mass(change, stepper.prognostic_mass, stepper.grid.weights.to(change.device))
    state[..., prognostic] = state[..., prognostic] + change
    for index in stepper.roles.forcing:
        name = stepper.roles.channels[index]
        state[..., index] = solar_forcing(stepper.grid, valid_time.detach().cpu().double().numpy(), name,
                                          stepper.normalizer).to(state.device, state.dtype)
    return state


__all__ = ["RI_WINDOWS", "init_ordinal", "ordinal_logits", "SatNetConfig", "SatNetStorm", "SatelliteRetrieval", "apply_retrieval", "cloud_motion",
           "increment_scale", "observation_weights", "retrieval_loss", "retrieval_targets", "ri_threshold_list",
           "satnet_loss", "satnet_targets_ri", "GLOBAL_CHANNELS"]
