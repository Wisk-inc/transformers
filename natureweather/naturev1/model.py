# Copyright 2026 Nathan. Apache-2.0.
"""
NatureV1 -- a probabilistic weather model built on geolocated fields rather than pixel grids.

What it is not: a ViT or CNN front-end over a cropped image. Those inherit the projection's distortion
(a GOES pixel at the limb covers 60x the ground a nadir pixel does, and a convolution weights them the
same), they have no idea where any pixel is, and they cannot fuse two sources without resampling one
onto the other's grid and losing information in the process.

What it is instead: every input -- satellite scenes, analysis fields, station reports -- is a set of
samples with a latitude, a longitude, a physical value and a ground area. They are read onto one shared
global mesh by geometric cross-attention, processed there, and written back out to whatever points you
want a forecast at. Sources with completely different geometries fuse for free, because they were never
grids to begin with.

Every output is a distribution, not a number. A track comes out as a handful of weighted scenarios, each
with its own growing uncertainty ellipse; fields come out with per-point variance; landfall comes out as a
probability per lead time. A forecast that cannot say how sure it is, is not a forecast.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn.functional as F
from ihelix.kernels import IHelixRMSNorm
from torch import nn

from ihelix import (
    CrossAttention,
    FieldGrid,
    Geometry,
    GradientCheckpointing,
    GridBound,
    GridLink,
    IHelixBlock,
    IHelixConfig,
    grid_fingerprint,
)


#: Gridded fields the model predicts everywhere, with a variance for each.
SURFACE_FIELDS = (
    "t2m",          # 2 m temperature, K
    "mslp",         # mean sea level pressure, Pa
    "u10",          # 10 m eastward wind, m/s
    "v10",          # 10 m northward wind, m/s
    "rh",           # relative humidity, fraction
    "precip_rate",  # mm/hr
    "cloud",        # cloud fraction
)
#: Rapid-intensification thresholds, in knots gained over 24 hours. 30 kt is the National Hurricane
#: Center's definition; the others bracket it so the head learns a curve rather than one cut point.
#: Measured base rates in the Atlantic record: 6.65%, 4.11% and 2.24% respectively.
RI_THRESHOLDS_KT = (25.0, 30.0, 35.0)

#: Each extra member of the rotation ensemble turns the planet by the golden angle again: the angles never
#: repeat, never line up with the 1.5 degree analysis grid or the Fibonacci mesh, and the first k of them are
#: the same whatever the ensemble size -- so a bank built with more rotations serves a model using fewer.
ROTATION_STEP_DEG = 180.0 * (3.0 - math.sqrt(5.0))


def rotation_angles(count: int) -> list[float]:
    """The longitude turns (degrees) of members 1..count-1 of a rotation ensemble; member 0 is unturned."""
    return [(r * ROTATION_STEP_DEG) % 360.0 for r in range(1, max(int(count), 1))]


def turned_centre(storm_center: torch.Tensor, degrees: float) -> torch.Tensor:
    """Storm centres moved east by ``degrees`` with the planet, longitude back in [-180, 180)."""
    turned = storm_center.clone()
    turned[..., 1] = torch.remainder(turned[..., 1] + degrees + 180.0, 360.0) - 180.0
    return turned


#: Mean and log-variance outputs that merge by the law of total variance.
_SPREAD_PAIRS = {"intensity_log_var": "intensity_mean", "eyewall_peak_wind_log_var": "eyewall_peak_wind_kt",
                 "eyewall_rmw_log_var": "eyewall_rmw_nmi", "wind_radii_log_var": "wind_radii_nmi",
                 "ri_delta_log_var": "ri_delta_wind_kt"}


def merge_rotations(members: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    One set of storm outputs from the members of a rotation ensemble.

    Means average. A Gaussian's variance becomes the mixture's -- the members' mean variance plus the
    variance of their means, so disagreement between rotations widens the uncertainty rather than being
    averaged away -- and the track mixture's per-mode scale does the same with its displacements. Logits
    average in logit space (the log-linear opinion pool, which keeps a confident consensus confident).
    """
    if len(members) == 1:
        return dict(members[0])
    merged = {}
    for key, first in members[0].items():
        if not torch.is_floating_point(first):
            merged[key] = first
            continue
        stack = torch.stack([m[key].float() for m in members])
        if key in _SPREAD_PAIRS and _SPREAD_PAIRS[key] in members[0]:
            means = torch.stack([m[_SPREAD_PAIRS[key]].float() for m in members])
            total = stack.exp().mean(0) + means.var(0, unbiased=False)
            merged[key] = total.clamp_min(1e-12).log().clamp(-10.0, 10.0).to(first.dtype)
        elif key == "log_scale" and "displacement" in members[0]:
            spread = torch.stack([m["displacement"].float() for m in members]).var(0, unbiased=False)
            merged[key] = (0.5 * ((2.0 * stack).exp().mean(0) + spread).clamp_min(1e-12).log()).to(first.dtype)
        elif key.endswith("log_var"):
            merged[key] = stack.exp().mean(0).clamp_min(1e-12).log().to(first.dtype)
        else:
            merged[key] = stack.mean(0).to(first.dtype)
    return merged
#: Wind-radius thresholds (kt) reported per quadrant NE/SE/SW/NW -- the storm's actual wind footprint.
WIND_RADII_THRESHOLDS_KT = (34.0, 50.0, 64.0)

#: Climatological anchors for the heads that predict physical quantities in physical units.
#:
#: A linear head starts out predicting roughly zero. Asked for central pressure in hPa, that is an error
#: of about 1000, a squared error of a million, and a gradient norm in the hundreds of thousands -- the
#: intensity term then dwarfs every other head and the run spends its first thousands of steps learning
#: a constant the Atlantic already told us. Anchoring the output at the observed mean and scaling by the
#: observed spread starts the head at climatology, which is both the right prior and a loss that begins
#: near one instead of near a million. The numbers are Atlantic tropical-cyclone values from HURDAT2.
#:
#: These shift the *parameterization*, never the units: the head still reports m/s and absolute hPa, so
#: :func:`naturev1.forecast.decode_intensity` is unaffected.
WIND_ANCHOR_MS = (33.0, 15.0)        # mean, spread of maximum sustained wind
PRESSURE_ANCHOR_HPA = (985.0, 20.0)  # mean, spread of minimum central pressure
PEAK_WIND_ANCHOR_KT = (65.0, 30.0)   # mean, spread of peak eyewall wind
RMW_ANCHOR_NMI = (25.0, 15.0)        # mean, spread of radius of maximum wind
WIND_RADII_ANCHOR_NMI = (80.0, 60.0)  # mean, spread across thresholds and quadrants
RI_DELTA_ANCHOR_KT = (5.0, 15.0)     # mean, spread of the 24-hour wind change
#: Typical translation speed of a tropical cyclone, in degrees per hour -- about 11 knots.
TRACK_SPEED_DEG_PER_HOUR = 0.1
#: Measured base rates of rapid intensification in the Atlantic record, per threshold in
#: :data:`RI_THRESHOLDS_KT`. A classifier initialised at zero opens by saying every storm has a coin's
#: chance of rapid intensification, which for a 4% event is not a neutral prior -- it is a loud wrong
#: one, and the focal loss then spends its early gradient walking it back. Starting at the base rate
#: means the untrained model says "probably not, and here is the climatological rate", which is both
#: true and the right thing to update away from.
RI_BASE_RATES = (0.0665, 0.0411, 0.0224)

#: Every output and input path whose training is tracked. A path with a count of zero has never had
#: a gradient step, and :func:`naturev1.build_forecast` will not present its output as a forecast.
TRACKED_HEADS = (
    "field", "weather_type", "track", "landfall", "landfall_timing", "intensity", "ri", "ri_delta",
    "ri_onset", "eyewall_peak", "eyewall_rmw", "wind_radii", "enso", "state",
    "analysis_encoder", "satellite_encoder", "environment_encoder", "storm_position", "storm_state",
    "storm_environment", "storm_far", "track_prior", "storm_satellite",
    # 0.12: the satellite storm network, the global retrieval, RI 2.0, fusing the storm network into the
    # heads, and the backbone's last blocks when they are fine-tuned on storms.
    "satnet", "retrieval", "ri2", "satnet_fusion", "backbone_tail",
    # 0.12, physics: the thermodynamic and dynamic storm inputs.
    "storm_physics",
)
#: Parameters added after 0.7. A checkpoint written before them loads with these freshly initialised --
#: all of them start as exact no-ops, so the loaded model behaves exactly as it did when it was saved.
OPTIONAL_STATE = ("head_steps", "storm_position.", "storm_state_embed.", "state_head.", "noise_proj.",
                  "storm_environment_embed.", "storm_far_pool.", "track_consensus", "storm_satellite.",
                  "ri_calibration", "landfall_calibration", "feature_mean", "feature_std", "feature_threshold",
                  "feature_norm_bounds", "state_tendency",
                  "storm_satnet.", "satnet_fusion.", "retrieval.", "ri2_head.", "ri2_calibration",
                  "intensity_correction", "eyewall_correction", "storm_physics_embed.", "statistics.")

#: The storm's physical environment, computed from the state by :func:`naturev1.storm_environment`:
#: the SHIPS predictors. Deep-layer steering and 200-850 hPa shear over the 200-800 km ring around the
#: centre (tens of m/s), mid-level humidity over the inner 500 km (5 g/kg units), the ocean's distance
#: from 26.5 C (3 K units), and whether upper-level winds were available at all.
STORM_ENVIRONMENT_FEATURES = ("steering_u", "steering_v", "shear_u", "shear_v", "shear_speed",
                              "mid_level_humidity", "ocean_temperature", "observed")

#: The members of the track consensus, in the order of the ``track_consensus`` weights: the learned
#: head's most likely scenario, its probability-weighted mean, the steering-flow track, and straight
#: extrapolation of the last twelve hours. What the Hurricane Center calls a consensus is a weighted
#: blend of several models, and it beats every one of its members; this is that, within one model.
TRACK_MEMBERS = ("scenario", "mixture_mean", "steering", "extrapolation",
                 # 0.12: the storm carried by the steering flow of the model's own rollout -- the atmosphere as
                 # it will be, not as it is -- and the vortex followed through that rollout's pressure field;
                 # historical analogs; and the mean of the vortex tracked through every ensemble member.
                 "rollout_steering", "rollout_vortex", "analog", "rollout_ensemble")
#: Members every checkpoint before 0.12 was fitted with; later ones are NaN where not computed.
BASE_TRACK_MEMBERS = 4

#: What RI 2.0 is told besides the pooled atmosphere -- the SHIPS-RII predictors that carry the signal:
#: how far below its potential intensity the storm is, shear, mid-level moisture, ocean temperature,
#: twelve-hour persistence, latitude, and the satellite storm network's own reading of the core (whether
#: it ran, its seven RI logits, its five intensity-change estimates, and its current wind against the
#: best track's).
#: Then the thermodynamics (naturev1.thermo): potential intensity, the gap below it, ventilation, entropy deficit,
#: CAPE, 200 hPa divergence, mid-level humidity, the lower warm core, and whether any of it was computed.
RI2_PHYSICS = ("potential_intensity", "intensity_gap", "ventilation", "entropy_deficit", "cape", "divergence_200",
               "rh_mid", "phase_vtl", "observed")
RI2_INPUTS = 6 + 1 + 7 + 5 + 1 + len(RI2_PHYSICS)

#: What the storm heads are told about the storm itself, beyond the atmosphere around it -- the
#: predictors every operational intensity scheme (SHIPS, LGEM) starts from. A 1.5-degree reanalysis
#: smears a 30 km eyewall into a gentle low, so without being told, the heads cannot know whether they
#: are looking at a tropical storm or a category 5; and the last twelve hours of motion is the single
#: best predictor of the next twelve. Scaled to order one; see :func:`naturev1.storm_state_vector`.
STORM_STATE_FEATURES = ("wind", "pressure", "motion_lat_12h", "motion_lon_12h", "wind_change_12h",
                        "observed")

#: "Will it rain or be sunny" as a calibrated categorical, not a threshold on a regression.
WEATHER_TYPES = ("clear", "partly_cloudy", "overcast", "light_rain", "heavy_rain", "thunderstorm", "snow")


@dataclass
class NatureConfig:
    """
    Configuration for NatureV1. Defaults are the ~88M-parameter model.

    88M is a deliberate size. It trains from scratch on one card in a sane amount of time, leaves room
    for a large batch and long history on 96 GB, and -- since a weather model is bound far more by how
    much data you can feed it than by how many parameters it has -- spends the budget where it helps.

    Args:
        satellite_channels: input channels per satellite sample (ABI bands).
        analysis_channels: input channels per analysis sample (from GFS/ERA5, per level).
        hidden_size / num_layers / num_heads / num_kv_heads / head_dim: the processor.
        latent_points: samples on the shared global mesh everything is read onto.
        lead_times_hours: forecast lead times the heads predict at.
        environment_channels: channels of the slow environmental field -- sea-surface temperature, deep
            ocean heat content, vertical wind shear, mid-level humidity. These are the actual physical
            predictors of rapid intensification, so they are supplied explicitly rather than left to be
            inferred from imagery.
        track_modes: how many distinct scenarios the track head may propose. This is what lets the model
            say "most likely it recurves, but there is a 22% branch where it does not" rather than
            averaging the two into a track that goes somewhere neither would.
        history_frames: how many past timesteps the temporal strand sees.
        state_channels: prognostic channels to predict as a residual -- the full atmospheric state,
            not just the surface. ``0`` disables the head, which leaves every existing configuration
            unchanged. Set it from :func:`naturev1.state.channel_roles` so rollouts evolve the upper
            air rather than freezing it.
        noise_dim: width of the per-point noise injected for ensemble members. ``0`` disables it.
            Each member draws its own noise, the processor spreads it spatially, and a CRPS loss
            shapes the resulting spread into calibrated uncertainty.
        storm_pool_radius_km: when a storm centre is given, the storm heads read a Gaussian-weighted
            pool of the mesh around it with this radius, instead of a mean over the whole planet.
    """

    satellite_channels: int = 6
    analysis_channels: int = 24
    environment_channels: int = 8
    hidden_size: int = 512
    num_layers: int = 17
    num_heads: int = 8
    num_kv_heads: int = 4
    head_dim: int = 64
    intermediate_size: int = 2048

    latent_points: int = 4096
    latent_neighbours: int = 48
    latent_cluster: int = 64
    min_radius_km: float = 120.0
    max_radius_km: float = 1600.0
    encode_radius_km: float = 240.0
    index_layer_stride: int = 2
    index_topk: int = 8
    index_branching: int = 8
    index_beam_width: int = 4
    landmark_dim: int = 64

    history_frames: int = 6
    lead_times_hours: tuple[int, ...] = (6, 12, 18, 24, 36, 48, 72, 96, 120)
    track_modes: int = 6

    recurrent_heads: int = 4
    recurrent_head_dim: int = 64
    recurrent_chunk: int = 4

    state_channels: int = 0
    noise_dim: int = 0
    storm_pool_radius_km: float = 500.0

    # 0.12. Every one defaults to "absent", so an older configuration builds exactly the older model.
    satnet_embed: int = 0              # > 0 adds the satellite storm network (naturev1.satnet.SatNetStorm)
    satnet_widths: tuple[int, ...] = (64, 128, 256, 384)
    satnet_pixels: int = 256
    retrieval_layers: int = 0          # > 0 adds the global satellite retrieval with this many blocks
    retrieval_resolution: float = 0.25
    ri2: bool = False                  # the ordinal, predictor-fed rapid-intensification head
    hemisphere_frame: bool = False     # storm heads work in a poleward frame (set when trained that way)
    storm_rotations: int = 1           # test-time rotation ensemble for the storm outputs (1: off; chosen on
                                       # held-out seasons by naturev1.stormtrain.select_rotations)

    dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.track_modes < 1:
            raise ValueError("track_modes must be >= 1")

    @property
    def num_leads(self) -> int:
        return len(self.lead_times_hours)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> NatureConfig:
        known = {f.name for f in fields(cls)}
        payload = {k: v for k, v in values.items() if k in known}
        for key in ("lead_times_hours", "satnet_widths"):
            if key in payload:
                payload[key] = tuple(payload[key])
        return cls(**payload)

    def ihelix(self) -> IHelixConfig:
        return IHelixConfig(
            in_channels=self.hidden_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            min_radius=self.min_radius_km,
            max_radius=self.max_radius_km,
            encode_radius=self.encode_radius_km,
            index_layer_stride=self.index_layer_stride,
            index_topk=self.index_topk,
            index_branching=self.index_branching,
            index_beam_width=self.index_beam_width,
            landmark_dim=self.landmark_dim,
            use_temporal=True,
            num_recurrent_heads=self.recurrent_heads,
            recurrent_head_dim=self.recurrent_head_dim,
            recurrent_value_head_dim=self.recurrent_head_dim,
            recurrent_chunk_size=self.recurrent_chunk,
            rms_norm_eps=self.rms_norm_eps,
            initializer_range=self.initializer_range,
        )


class SourceEncoder(nn.Module):
    """
    Reads one data source onto the shared mesh.

    Each source keeps its own input projection and its own read radius, because a satellite scene at 2 km
    and an analysis field at 25 km are not the same kind of measurement and should not pretend to be. What
    they share is the mesh they are read onto, which is what makes fusing them a sum rather than a
    resampling.
    """

    def __init__(self, config: NatureConfig, in_channels: int, geometry: Geometry, radius_km: float) -> None:
        super().__init__()
        self.embed = nn.Linear(in_channels, config.hidden_size)
        self.read = CrossAttention(
            config.hidden_size, config.num_heads, config.num_kv_heads, config.head_dim,
            geometry.embed_dim, radius_km,
        )
        self.norm = IHelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # A learned "nothing here" vector, so a mesh point with no coverage -- a satellite that cannot see
        # the far side of the planet -- produces a clean absence instead of attending to noise.
        self.absent = nn.Parameter(torch.zeros(config.hidden_size))

    def forward(self, values: torch.Tensor, seed: torch.Tensor, link: GridLink) -> torch.Tensor:
        return self.norm(self.read(seed + self.absent, self.embed(values), link))


class TrackHead(nn.Module):
    """
    Where the storm goes, as a set of weighted scenarios rather than one line.

    Each mode predicts a displacement from the current centre at every lead time, plus the parameters of a
    2-D Gaussian around it -- two standard deviations and a correlation, so the uncertainty ellipse can
    tilt along the direction of travel the way a real one does. Mode weights are a categorical, so the
    output reads as "62% recurves offshore, 26% tracks the coast, 12% inland" instead of an average track
    that describes neither.
    """

    def __init__(self, config: NatureConfig) -> None:
        super().__init__()
        self.modes, self.leads = config.track_modes, config.num_leads
        self.mode_logits = nn.Linear(config.hidden_size, config.track_modes)
        # 5 numbers per mode per lead: d_lat, d_lon, log sigma_lat, log sigma_lon, atanh(rho)
        self.parameters_head = nn.Linear(config.hidden_size, config.track_modes * config.num_leads * 5)

    def forward(self, summary: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = summary.shape[0]
        raw = self.parameters_head(summary).view(batch, self.modes, self.leads, 5)
        return {
            "mode_logits": self.mode_logits(summary),
            "displacement": raw[..., :2],
            # Clamped so an early, confident-looking model cannot drive the variance to zero and make the
            # likelihood explode -- 1e-2 to ~150 degrees covers anything physical.
            "log_scale": raw[..., 2:4].clamp(-4.6, 5.0),
            "correlation": raw[..., 4].tanh() * 0.99,
        }


class EyewallHead(nn.Module):
    """
    The storm's core: peak eyewall wind, the radius it occurs at, and the wind footprint around it.

    Radius of maximum wind is the direct measure of eyewall size, and it is brutally scarce -- reported
    on under 5% of Atlantic best-track points, almost all of them after 2004. The wind radii (how far
    34, 50 and 64 kt winds extend into each quadrant) are far better populated and describe the same
    structure from outside in, so they are predicted jointly: they supervise the same representation and
    carry it when RMW is missing, which is most of the time.
    """

    def __init__(self, config: NatureConfig) -> None:
        super().__init__()
        self.leads = config.num_leads
        self.quadrants = 4
        self.thresholds = len(WIND_RADII_THRESHOLDS_KT)
        hidden = config.hidden_size
        self.peak_wind = nn.Linear(hidden, config.num_leads * 2)        # mean, log-variance
        self.rmw = nn.Linear(hidden, config.num_leads * 2)
        self.wind_radii = nn.Linear(hidden, config.num_leads * self.thresholds * self.quadrants * 2)

    def forward(self, summary: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = summary.shape[0]
        peak = self.peak_wind(summary).view(batch, self.leads, 2)
        rmw = self.rmw(summary).view(batch, self.leads, 2)
        radii = self.wind_radii(summary).view(batch, self.leads, self.thresholds, self.quadrants, 2)
        return {
            # Anchored at climatology so the head starts where the Atlantic already is; still knots.
            "eyewall_peak_wind_kt": peak[..., 0] * PEAK_WIND_ANCHOR_KT[1] + PEAK_WIND_ANCHOR_KT[0],
            "eyewall_peak_wind_log_var": peak[..., 1].clamp(-8.0, 8.0),
            # Softplus keeps a radius positive without a hard clamp that would kill its gradient.
            "eyewall_rmw_nmi": F.softplus(rmw[..., 0] * RMW_ANCHOR_NMI[1] + RMW_ANCHOR_NMI[0]) + 1.0,
            "eyewall_rmw_log_var": rmw[..., 1].clamp(-8.0, 8.0),
            "wind_radii_nmi": F.softplus(radii[..., 0] * WIND_RADII_ANCHOR_NMI[1] + WIND_RADII_ANCHOR_NMI[0]),
            "wind_radii_log_var": radii[..., 1].clamp(-8.0, 8.0),
        }


class RapidIntensificationHead(nn.Module):
    """
    Rapid intensification: will this storm gain 25/30/35 kt in the next 24 hours.

    RI fires on about 4% of eligible track points at the 30-knot threshold, which is the number that
    dictates everything about how this head is built and judged. A classifier that always answers "no"
    scores 96% accuracy and saves nobody, so the loss is weighted toward the positive class and the
    metrics that matter are precision, recall and Brier score, not accuracy.

    Alongside the classification it regresses the actual 24-hour intensity change, which is a denser
    signal than the rare binary and pulls the shared representation toward the physics -- shear,
    ocean heat, inner-core structure -- rather than toward the threshold itself.
    """

    def __init__(self, config: NatureConfig) -> None:
        super().__init__()
        self.thresholds = RI_THRESHOLDS_KT
        hidden = config.hidden_size
        self.trunk = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden // 2), nn.SiLU())
        self.classifier = nn.Linear(hidden // 2, len(RI_THRESHOLDS_KT))
        self.magnitude = nn.Linear(hidden // 2, 2)   # 24h wind change: mean and log-variance
        self.onset = nn.Linear(hidden // 2, config.num_leads)   # when in the window it is most likely

    def forward(self, summary: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.trunk(summary)
        magnitude = self.magnitude(features)
        return {
            "ri_logits": self.classifier(features),
            "ri_delta_wind_kt": magnitude[..., 0] * RI_DELTA_ANCHOR_KT[1] + RI_DELTA_ANCHOR_KT[0],
            "ri_delta_log_var": magnitude[..., 1].clamp(-8.0, 8.0),
            "ri_onset_logits": self.onset(features),
        }


class StormSatelliteEncoder(nn.Module):
    """
    What a forecaster reads in storm-centred infrared, as a vector the storm heads can add.

    Input is two storm-centred patches of 11-micron brightness temperature (see :mod:`naturev1.stormsat`):
    now, and six hours earlier. Two views of them go in:

    * a small convolutional network over the current image, its six-hour change and where data exists --
      spiral bands, a clearing eye, a central dense overcast, a convective burst;
    * a fixed radial profile, the Dvorak technique's own measurements: mean temperature and the share of
      cloud colder than -60 C in 25 km rings out to 300 km, and the eye's contrast with the eyewall.

    The output layer starts at exactly zero, so adding this branch to a trained model changes nothing
    until it is trained -- and a sample with no image always contributes nothing.
    """

    RING_KM = 25.0
    RINGS = 12

    def __init__(self, hidden: int, pixels: int = 256, degrees: float = 0.07) -> None:
        super().__init__()
        self.pixels = pixels
        widths = (32, 64, 128, 128, 192)
        layers, channels = [], 3
        for index, width in enumerate(widths):
            kernel = 5 if index == 0 else 3
            layers += [nn.Conv2d(channels, width, kernel, stride=2, padding=kernel // 2),
                       nn.GroupNorm(8, width), nn.GELU()]
            channels = width
        self.cnn = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        profile = 2 * self.RINGS + 1 + 2
        self.mlp = nn.Sequential(nn.Linear(widths[-1] + profile, hidden), nn.GELU())
        self.project = nn.Linear(hidden, hidden, bias=False)
        # Distance of every pixel from the centre, in km (8 km per pixel at 0.07 degrees), for the rings.
        offsets = (torch.arange(pixels, dtype=torch.float32) - (pixels - 1) / 2.0) * degrees * 111.2
        radius = torch.sqrt(offsets.view(-1, 1) ** 2 + offsets.view(1, -1) ** 2)
        rings = torch.stack([(radius >= i * self.RING_KM) & (radius < (i + 1) * self.RING_KM)
                             for i in range(self.RINGS)]).float()
        self.register_buffer("rings", rings, persistent=False)
        self.register_buffer("eye", (radius < 20.0).float(), persistent=False)

    def forward(self, codes: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codes: ``(B, 2, P, P)`` uint8 brightness temperatures from :func:`naturev1.stormsat.encode_bt`
                -- now, and six hours earlier.
            available: ``(B, 2)`` whether each image exists.

        Returns:
            ``(B, hidden)``, exactly zero for samples with no current image.
        """
        from .stormsat import decode_bt

        kelvin = decode_bt(codes.to(self.rings.device))                       # (B, 2, P, P), NaN missing
        available = available.to(kelvin.device).bool()
        now, before = kelvin[:, 0], kelvin[:, 1]
        valid_now = torch.isfinite(now)
        both = valid_now & torch.isfinite(before) & available[:, 1].view(-1, 1, 1)
        image = torch.nan_to_num((now - 250.0) / 30.0)
        change = torch.where(both, (now - before) / 10.0, torch.zeros_like(now))
        x = torch.stack([image, torch.nan_to_num(change), valid_now.float()], 1).to(self.rings.dtype)
        learned = self.cnn(x)

        weight = self.rings.unsqueeze(0) * valid_now.unsqueeze(1).to(self.rings.dtype)   # (B, R, P, P)
        count = weight.sum((-1, -2)).clamp_min(1.0)
        filled = torch.nan_to_num(now).unsqueeze(1)
        ring_mean = (weight * filled).sum((-1, -2)) / count
        ring_cold = (weight * (filled < 213.15).to(weight.dtype)).sum((-1, -2)) / count
        eye = self.eye * valid_now.to(self.eye.dtype)
        eye_temp = (eye * torch.nan_to_num(now)).sum((-1, -2)) / eye.sum((-1, -2)).clamp_min(1.0)
        contrast = (eye_temp - ring_mean[:, 1:3].mean(-1)) / 30.0
        profile = torch.cat([(ring_mean - 250.0) / 30.0, ring_cold, contrast.unsqueeze(-1),
                             available.to(ring_mean.dtype)], -1)
        out = self.project(self.mlp(torch.cat([learned, profile], -1)))
        return out * available[:, :1].to(out.dtype)


class RapidIntensification2(nn.Module):
    """
    RI 2.0: the SHIPS-RII thresholds (25/30/35/40 kt in 24 h, 45 in 36, 55 in 48, 65 in 72) as one ordinal
    family, from the pooled atmosphere *and* the predictors that actually carry RI signal.

    The first head learned RI from the pooled reanalysis alone and called a 50-to-130 kt explosion 45 kt.
    The explicit predictors -- above all the gap between a storm's current wind and the maximum the ocean
    beneath it can support -- and the satellite network's view of the inner core are what give an RI
    model its skill operationally; here they are inputs, not things to rediscover from 22,000 labels.
    Ordinal logits keep the probabilities consistent, and every logit opens at the global base rate.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        from .satnet import RI_WINDOWS, init_ordinal

        self.trunk = nn.Sequential(nn.Linear(hidden + RI2_INPUTS, hidden), nn.SiLU(), nn.Linear(hidden, hidden // 2),
                                   nn.SiLU())
        self.level = nn.Linear(hidden // 2, len(RI_WINDOWS))
        self.gaps = nn.Linear(hidden // 2, sum(len(t) for _, t in RI_WINDOWS))
        init_ordinal(self.level, self.gaps)

    def reset_priors(self) -> None:
        """Back to the global base rates -- needed after a blanket re-initialisation of the model."""
        from .satnet import init_ordinal

        init_ordinal(self.level, self.gaps)

    def forward(self, summary: torch.Tensor, predictors: torch.Tensor) -> torch.Tensor:
        from .satnet import ordinal_logits

        features = self.trunk(torch.cat([summary, predictors.to(summary.dtype)], -1))
        return ordinal_logits(self.level(features), self.gaps(features))


def potential_intensity_kt(ocean_feature: torch.Tensor) -> torch.Tensor:
    """
    Empirical maximum potential intensity (DeMaria and Kaplan 1994) from the ``ocean_temperature``
    environment feature: 28.2 + 55.8 exp(0.1813 (SST - 30 C)) m/s, in knots.
    """
    sst_c = ocean_feature.float() * 3.0 + 26.5
    return ((28.2 + 55.8 * torch.exp(0.1813 * (sst_c - 30.0))) / 0.514444).clamp(0.0, 200.0)


def ri2_predictors(batch: int, device, storm_center=None, storm_state=None, storm_environment=None,
                   satnet: dict | None = None, storm_physics: torch.Tensor | None = None) -> torch.Tensor:
    """The ``(B, RI2_INPUTS)`` predictor vector for :class:`RapidIntensification2`; zeros for what is missing."""
    out = torch.zeros(batch, RI2_INPUTS, device=device)
    if storm_physics is not None:
        from .thermo import STORM_PHYSICS_FEATURES

        physics = torch.nan_to_num(storm_physics.to(device).float())
        out[:, 20:] = physics[:, [STORM_PHYSICS_FEATURES.index(name) for name in RI2_PHYSICS]]
    wind = None
    if storm_state is not None:
        state = torch.nan_to_num(storm_state.to(device).float())
        observed = state[:, 5] > 0.5
        wind = torch.where(observed, state[:, 0] * PEAK_WIND_ANCHOR_KT[1] + PEAK_WIND_ANCHOR_KT[0],
                           torch.full_like(state[:, 0], 50.0))
        out[:, 4] = state[:, 4]
    if storm_environment is not None:
        env = torch.nan_to_num(storm_environment.to(device).float())
        potential = potential_intensity_kt(env[:, 6])
        out[:, 0] = (potential - (wind if wind is not None else 50.0)) / 50.0
        out[:, 1], out[:, 2], out[:, 3] = env[:, 4], env[:, 5], env[:, 6]
    if storm_center is not None:
        out[:, 5] = torch.nan_to_num(storm_center.to(device).float()[:, 0]).abs() / 30.0
    if satnet is not None and "ri_logits" in satnet:
        present = satnet.get("present")
        flag = (torch.ones(batch, device=device) if present is None
                else present.to(device).float().view(-1))
        out[:, 6] = flag
        out[:, 7:14] = satnet["ri_logits"].to(device).float().clamp(-10, 10) * flag.view(-1, 1)
        out[:, 14:19] = satnet["wind_change_kt"].to(device).float() / 15.0 * flag.view(-1, 1)
        if wind is not None:
            out[:, 19] = (satnet["wind_kt"].to(device).float() - wind) / 20.0 * flag
    return out


class NatureV1(GridBound, GradientCheckpointing, nn.Module):
    """
    The model. Reads any number of geolocated sources, forecasts fields and storm behaviour with
    uncertainty attached to everything.

    Call :meth:`gradient_checkpointing_enable` to recompute block activations instead of storing them.
    It costs roughly a third more time per step and buys back most of the activation memory, which on a
    large card is how spare VRAM becomes a bigger batch -- usually worth far more than the time.

    Args:
        config: the model configuration.
        latent_grid: the shared global mesh, normally a near-equal-area Fibonacci sphere so there is no
            pole to distort and no seam to cross. ``.to(device)`` carries it along with the weights, and
            any source or output grid you pass in is moved to meet them on first use.
    """

    def __init__(self, config: NatureConfig, latent_grid: FieldGrid) -> None:
        super().__init__()
        self.config = config
        self.latent_grid = latent_grid
        geometry = latent_grid.geometry
        inner = config.ihelix()

        self.satellite_encoder = SourceEncoder(config, config.satellite_channels, geometry, config.encode_radius_km)
        self.analysis_encoder = SourceEncoder(config, config.analysis_channels, geometry, config.encode_radius_km * 3)
        # The environment moves slowly and over long distances, so it is read with a much wider radius.
        self.environment_encoder = SourceEncoder(
            config, config.environment_channels, geometry, config.encode_radius_km * 6
        )
        self.mesh_seed = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # Calendar time matters physically -- solar angle, season -- so it is given, not inferred.
        self.time_embed = nn.Sequential(nn.Linear(6, 256), nn.SiLU(), nn.Linear(256, config.hidden_size))

        self.blocks = nn.ModuleList(IHelixBlock(inner, geometry, i) for i in range(config.num_layers))
        self.norm = IHelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.decoder = CrossAttention(
            config.hidden_size, config.num_heads, config.num_kv_heads, config.head_dim,
            geometry.embed_dim, config.encode_radius_km * 2,
        )
        self.decode_norm = IHelixRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        hidden, leads = config.hidden_size, config.num_leads
        # Gridded forecasts: a mean and a log-variance per field per lead time, everywhere.
        self.field_head = nn.Linear(hidden, leads * len(SURFACE_FIELDS) * 2)
        # The full atmospheric state, as a residual: every prognostic channel's change over one step.
        # Mean and log-variance per channel, in units of that channel's own tendency spread.
        self.state_head = nn.Linear(hidden, config.state_channels * 2) if config.state_channels else None
        # Ensemble noise, read in once before the processor; the geodesic blocks give it spatial
        # structure, so independent per-point draws become coherent perturbations.
        self.noise_proj = nn.Linear(config.noise_dim, hidden, bias=False) if config.noise_dim else None
        self.weather_type_head = nn.Linear(hidden, leads * len(WEATHER_TYPES))

        # Storm-scale outputs read from a pooled summary of the whole mesh.
        self.summary_norm = IHelixRMSNorm(hidden, eps=config.rms_norm_eps)
        self.summary_pool = nn.Linear(hidden, hidden)
        self.track_head = TrackHead(config)
        self.landfall_head = nn.Linear(hidden, leads * 2)      # logit, and log-variance of the timing
        self.intensity_head = nn.Linear(hidden, leads * 4)     # max wind, min pressure, each with log-variance
        self.enso_head = nn.Linear(hidden, 2)                  # Nino 3.4 index and its log-variance
        self.eyewall_head = EyewallHead(config)
        self.ri_head = RapidIntensificationHead(config)
        # Where the storm is, as a unit vector: latitude sets the Coriolis parameter and the beta drift
        # that pushes every tropical cyclone poleward and west, which no pooled field states directly.
        self.storm_position = nn.Linear(3, hidden, bias=False)
        # The storm's own state -- current intensity and recent motion. See STORM_STATE_FEATURES.
        self.storm_state_embed = nn.Linear(len(STORM_STATE_FEATURES), hidden, bias=False)
        # The storm's environment, computed from the state: steering, shear, moisture, ocean. See
        # STORM_ENVIRONMENT_FEATURES. The heads could in principle learn these from the pooled features;
        # given explicitly, 22,000 labels go on what the physics does not already say.
        self.storm_environment_embed = nn.Linear(len(STORM_ENVIRONMENT_FEATURES), hidden, bias=False)
        # A second, wider pool around the storm -- twice storm_pool_radius_km -- so the heads see the
        # core and the environment as two things: the ridge that steers a storm sits a thousand
        # kilometres from the eye, and a 500 km pool barely touches it.
        self.storm_far_pool = nn.Linear(hidden, hidden, bias=False)
        # Per-lead weights over TRACK_MEMBERS, fitted on held-out storms after the heads are trained.
        # All zero until fitted, which the forecast reads as "no consensus available".
        self.register_buffer("track_consensus", torch.zeros(leads, len(TRACK_MEMBERS)))
        # Training steps each output has actually received -- saved with the weights, so a forecast
        # built from a checkpoint knows which of its heads were ever trained.
        self.register_buffer("head_steps", torch.zeros(len(TRACKED_HEADS), dtype=torch.int64))
        # Storm-centred infrared (see StormSatelliteEncoder). Registered after every other parameter, so
        # the backbone fingerprint that keys the cached storm feature banks does not change.
        self.storm_satellite = StormSatelliteEncoder(hidden)
        # Probability calibration, fitted on held-out seasons after training: logit -> a * logit + b, per
        # landfall lead and per rapid-intensification threshold. (1, 0) is no change.
        self.register_buffer("landfall_calibration", torch.tensor([[1.0, 0.0]]).repeat(leads, 1))
        self.register_buffer("ri_calibration", torch.tensor([[1.0, 0.0]]).repeat(len(RI_BASE_RATES), 1))
        # What storm features looked like in training, to flag inputs unlike anything it learned from.
        # Unfitted while the threshold is zero.
        self.register_buffer("feature_mean", torch.zeros(hidden))
        self.register_buffer("feature_std", torch.ones(hidden))
        self.register_buffer("feature_threshold", torch.zeros(()))
        self.register_buffer("feature_norm_bounds", torch.tensor([0.0, float("inf")]))
        # Per-channel six-hour tendency spread the full-state rollout was trained with (StateStepper),
        # shipped with the weights so a rollout can be reproduced. Zero until recorded.
        self.register_buffer("state_tendency", torch.zeros(config.analysis_channels))

        self.apply(self._init)
        self._anchor_uncertainty()
        self._zero_init_residual_paths()
        self._links: OrderedDict = OrderedDict()
        self._build_satellite_paths(config, latent_grid)

    def _build_satellite_paths(self, config: NatureConfig, latent_grid: FieldGrid) -> None:
        """
        The 0.12 additions, built after everything else is initialised so the existing weights draw the
        same random numbers as before, and registered last so the backbone fingerprint does not move.
        Each one starts as an exact no-op; an older configuration builds none of them.
        """
        hidden, leads = config.hidden_size, config.num_leads
        # Post-hoc intensity bias correction per lead, (offset kt, slope per 30 kt around 65 kt): fitted
        # on held-out seasons. The red team measured major hurricanes 10-18 kt too weak; this is the fix
        # that does not need retraining. Zeros are no correction.
        self.register_buffer("intensity_correction", torch.zeros(leads, 2))
        self.register_buffer("eyewall_correction", torch.zeros(leads, 2))
        from .satnet import RI_WINDOWS

        thresholds = sum(len(t) for _, t in RI_WINDOWS)
        self.register_buffer("ri2_calibration", torch.tensor([[1.0, 0.0]]).repeat(thresholds, 1))
        # The storm's thermodynamics and dynamics (naturev1.thermo.storm_physics): potential intensity,
        # ventilation, phase space, divergence, humidity... Zero-initialised like every other storm input.
        from .statistical import StatisticalModels
        from .thermo import STORM_PHYSICS_FEATURES

        self.storm_physics_embed = nn.Linear(len(STORM_PHYSICS_FEATURES), hidden, bias=False)
        nn.init.zeros_(self.storm_physics_embed.weight)
        # SHIPS-style regression, logistic growth, climatology-persistence, the corrected intensity consensus,
        # track bias and predicted consensus error -- fitted after training, carried with the weights.
        self.statistics = StatisticalModels(config.lead_times_hours)
        self.ri2_head = RapidIntensification2(hidden) if config.ri2 else None
        if self.ri2_head is not None:
            self.ri2_head.reset_priors()
        if config.satnet_embed:
            from .satnet import SatNetConfig, SatNetStorm

            self.storm_satnet = SatNetStorm(SatNetConfig(embed=config.satnet_embed, widths=config.satnet_widths,
                                                         pixels=config.satnet_pixels))
            self.satnet_fusion = nn.Linear(config.satnet_embed, hidden, bias=False)
            nn.init.zeros_(self.satnet_fusion.weight)
        else:
            self.storm_satnet = self.satnet_fusion = None
        if config.retrieval_layers:
            from ihelix.model import init_weights

            from .satnet import SatelliteRetrieval

            inner = config.ihelix()
            blocks = nn.ModuleList(IHelixBlock(inner, latent_grid.geometry, i) for i in range(config.retrieval_layers))
            blocks.apply(lambda module: init_weights(module, inner))
            self.retrieval = SatelliteRetrieval(
                hidden, config.state_channels, config.analysis_channels, blocks, latent_grid,
                resolution=config.retrieval_resolution, num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads, head_dim=config.head_dim, encode_radius_km=config.encode_radius_km)
        else:
            self.retrieval = None

    def warm_start_retrieval(self) -> list[str]:
        """
        Start the retrieval from what the backbone already knows: its first blocks, its analysis reader,
        its decoder and its calendar embedding -- so it begins by understanding the atmosphere and only
        has to learn what the satellites add. Returns what was copied.
        """
        if self.retrieval is None:
            raise ValueError("this model has no retrieval (config.retrieval_layers is 0)")
        r, copied = self.retrieval, []
        pairs = [(r.read_background, self.analysis_encoder.read), (r.background_embed, self.analysis_encoder.embed),
                 (r.background_norm, self.analysis_encoder.norm), (r.decoder, self.decoder),
                 (r.decode_norm, self.decode_norm), (r.time_embed, self.time_embed)]
        pairs += [(r.blocks[i], self.blocks[i]) for i in range(min(len(r.blocks), len(self.blocks)))]
        for target, source in pairs:
            target.load_state_dict(source.state_dict())
            copied.append(type(source).__name__)
        with torch.no_grad():
            r.mesh_seed.copy_(self.mesh_seed)
        return copied

    def retrieve(self, satellite: torch.Tensor, available: torch.Tensor, background: torch.Tensor,
                 stale_hours: torch.Tensor, calendar: torch.Tensor, analysis_grid: FieldGrid) -> dict:
        """
        Run the satellite retrieval: ``(B, 3, 8, H, W)`` staged global frames, ``(B, 3)`` availability,
        ``(B, P, C)`` background state ``stale_hours`` old, ``(B, 6)`` calendar now. Returns the predicted
        increment and its log-variance; :func:`naturev1.satnet.apply_retrieval` turns that into a state.
        """
        if self.retrieval is None:
            raise ValueError("this model has no retrieval (config.retrieval_layers is 0)")
        return self.retrieval(satellite, available, background, stale_hours, calendar,
                              self._aligned(analysis_grid), self.latent_grid, self.link)

    def satnet_outputs(self, codes: torch.Tensor, available: torch.Tensor, latitude: torch.Tensor,
                       context: torch.Tensor | None = None) -> dict | None:
        """The satellite storm network's estimates, or None when this model has none."""
        if self.storm_satnet is None:
            return None
        return self.storm_satnet(codes.to(self.device), available.to(self.device), latitude.to(self.device),
                                 None if context is None else context.to(self.device))

    def satellite_parameters(self):
        """The satellite networks -- trained in their own stage, never by the storm-head fit."""
        for module in (self.storm_satnet, self.retrieval):
            if module is not None:
                yield from module.parameters()

    def _init(self, module: nn.Module) -> None:
        from ihelix.model import init_weights

        init_weights(module, self.config.ihelix())

    def _zero_init_residual_paths(self) -> None:
        """
        Start the residual head at exactly zero change, and the ensemble noise at exactly zero effect.

        A zero residual is persistence. So an untrained model with the state head *is* persistence to
        the last bit, and training can only move it from the strongest trivial baseline there is --
        rather than from a random field it first has to unlearn. The log-variance half is left at zero
        too, which is one tendency standard deviation: the honest prior, since the targets are scaled
        to unit spread.

        The noise projection starts at zero for the same reason from the other side: turning on
        ensembles cannot degrade a model, because at initialisation every member is identical. The
        CRPS loss then grows the spread only as far as the data justifies.

        One consequence worth knowing: a zero head passes no gradient *upstream* on the first step,
        since the derivative of its output with respect to its input is its weight. Step 0 trains the
        head alone; the backbone wakes from step 1. That is the ReZero / adaLN-Zero behaviour, and it
        means the backbone is never pushed around by a head that has not yet learned its job.
        """
        with torch.no_grad():
            if self.state_head is not None:
                self.state_head.weight.zero_()
                self.state_head.bias.zero_()
            if self.noise_proj is not None:
                self.noise_proj.weight.zero_()
            # Zero too: until trained, telling the heads where the storm is, or how strong, changes
            # nothing -- so a checkpoint from before these existed behaves exactly as it did.
            self.storm_position.weight.zero_()
            self.storm_state_embed.weight.zero_()
            self.storm_environment_embed.weight.zero_()
            self.storm_far_pool.weight.zero_()
            self.storm_satellite.project.weight.zero_()

    def _anchor_uncertainty(self) -> None:
        """
        Start every log-variance at the quantity's climatological spread instead of at one.

        A freshly initialised linear layer emits about zero, so a head that reports a log-variance
        begins by claiming a standard deviation of one -- one knot of uncertainty about peak wind, one
        hectopascal about central pressure. That is a wildly overconfident prior, and a Gaussian
        likelihood punishes it in proportion: the eyewall term alone opened at 438 against a total loss
        of 718, purely because the model was certain and wrong. Starting at the observed spread opens
        the same term near 5, and the head spends its gradient learning the storm rather than learning
        how unsure it should have been.

        Only biases are touched, so the weights still decide how the prediction varies with the input.
        """
        pairs = (
            (self.intensity_head, [WIND_ANCHOR_MS[1], PRESSURE_ANCHOR_HPA[1]]),
            (self.eyewall_head.peak_wind, [PEAK_WIND_ANCHOR_KT[1]]),
            (self.eyewall_head.rmw, [RMW_ANCHOR_NMI[1]]),
            (self.eyewall_head.wind_radii, [WIND_RADII_ANCHOR_NMI[1]]),
            (self.ri_head.magnitude, [RI_DELTA_ANCHOR_KT[1]]),
        )
        with torch.no_grad():
            for head, spreads in pairs:
                bias = head.bias
                # Every one of these layers lays its outputs out as (..., mean, log-variance), so the
                # log-variance slots are the odd indices.
                for index, spread in enumerate(spreads):
                    target = min(2.0 * math.log(spread), 8.0)
                    bias[index * 2 + 1 :: 2 * len(spreads)] = target

            # The rare-event heads open at the observed base rate rather than at even odds.
            for index, rate in enumerate(RI_BASE_RATES[: len(RI_THRESHOLDS_KT)]):
                self.ri_head.classifier.bias[index] = math.log(rate / (1.0 - rate))

            # The track head's spread is the one that must grow with lead time. Its mean starts at zero
            # displacement -- the right prior, the storm is where it is -- so the scale that goes with
            # that prior is how far a storm typically travels by then, not a constant. A tropical
            # cyclone moves on the order of 0.1 degrees an hour, so the anchor tracks the lead directly:
            # 0.6 degrees at +6 h out to 12 degrees at +120 h. Left at one degree for every lead, the
            # +120 h term opened at 192 against a total of 203.
            track_bias = self.track_head.parameters_head.bias.view(
                self.config.track_modes, self.config.num_leads, 5
            )
            for lead, hours in enumerate(self.config.lead_times_hours):
                track_bias[:, lead, 2:4] = math.log(max(TRACK_SPEED_DEG_PER_HOUR * hours, 1e-2))

    def link(self, target: FieldGrid, source: FieldGrid, num_neighbours: int) -> GridLink:
        """
        Cached geometric correspondence between two point sets, keyed on their content.

        Keyed on identity, as it used to be, every batch from a DataLoader worker -- which arrives with
        its own unpickled copy of the grid -- built and kept a new correspondence: two per step, about
        45 MB of GPU memory each step at ERA5 size, until the card ran out. See
        :func:`ihelix.grid_fingerprint`.
        """
        return self.cached_link(target, source, num_neighbours)

    def clear_links(self) -> None:
        """Drop cached correspondences, e.g. after swapping in a different input grid."""
        self._links.clear()

    def encode(
        self,
        satellite: torch.Tensor | None,
        satellite_grid: FieldGrid | None,
        analysis: torch.Tensor | None,
        analysis_grid: FieldGrid | None,
        calendar: torch.Tensor,
        environment: torch.Tensor | None = None,
        environment_grid: FieldGrid | None = None,
        neighbours: int = 24,
    ) -> torch.Tensor:
        """
        Read every available source onto the shared mesh and add them up.

        Sources are optional and independent: a scene with no satellite coverage, or an analysis-only
        step, simply contributes nothing rather than breaking the forward pass.
        """
        available = [x for x in (satellite, analysis, environment) if x is not None]
        if not available:
            raise ValueError("Give at least one of satellite, analysis or environment input.")
        reference = available[0]
        batch, steps = reference.shape[0], reference.shape[1]
        folded = batch * steps
        latent = self.latent_grid

        mesh = self.mesh_seed.expand(folded, latent.num_points, -1)
        total = torch.zeros_like(mesh)
        if satellite is not None:
            link = self.link(latent, satellite_grid, neighbours)
            total = total + self.satellite_encoder(
                satellite.reshape(folded, satellite_grid.num_points, -1), mesh, link
            )
        if analysis is not None:
            link = self.link(latent, analysis_grid, neighbours)
            total = total + self.analysis_encoder(
                analysis.reshape(folded, analysis_grid.num_points, -1), mesh, link
            )
        if environment is not None:
            link = self.link(latent, environment_grid, neighbours)
            total = total + self.environment_encoder(
                environment.reshape(folded, environment_grid.num_points, -1), mesh, link
            )
        total = total + self.time_embed(calendar).reshape(folded, 1, -1)
        return total.view(batch, steps, latent.num_points, -1)

    def forward(
        self,
        satellite: torch.Tensor | None = None,
        satellite_grid: FieldGrid | None = None,
        analysis: torch.Tensor | None = None,
        analysis_grid: FieldGrid | None = None,
        calendar: torch.Tensor | None = None,
        environment: torch.Tensor | None = None,
        environment_grid: FieldGrid | None = None,
        output_grid: FieldGrid | None = None,
        neighbours: int = 24,
        member_noise: torch.Tensor | None = None,
        storm_center: torch.Tensor | None = None,
        storm_state: torch.Tensor | None = None,
        storm_environment: torch.Tensor | None = None,
        track_prior: torch.Tensor | None = None,
        storm_satellite: tuple[torch.Tensor, torch.Tensor] | None = None,
        storm_satnet: dict | None = None,
        storm_physics: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            satellite: ``(B, T, N_sat, C_sat)`` normalized satellite samples, or None.
            satellite_grid: the :class:`FieldGrid` those samples live on.
            analysis: ``(B, T, N_ana, C_ana)`` analysis samples, or None.
            analysis_grid: its grid.
            calendar: ``(B, T, 6)`` cyclic time features from :func:`calendar_features`.
            output_grid: where to write gridded forecasts. Defaults to the shared mesh.
            neighbours: sources read per mesh point.
            member_noise: ``(B, P_latent, noise_dim)`` noise for one ensemble member, or None for a
                deterministic pass. Ignored unless the model was built with ``noise_dim > 0``.
            storm_center: ``(B, 2)`` latitude and longitude of the storm each sample is about, in
                degrees. The storm heads then read the atmosphere *around that storm*; without it they
                read a mean over the whole planet, which cannot tell two simultaneous storms apart.
                Rows of NaN fall back to the global pool.
            storm_state: ``(B, len(STORM_STATE_FEATURES))`` the storm's current intensity and recent
                motion, from :func:`naturev1.storm_state_vector`. Optional.
            storm_environment: ``(B, len(STORM_ENVIRONMENT_FEATURES))`` steering, shear, moisture and
                ocean around the storm, from :func:`naturev1.storm_environment`. Optional.
            track_prior: ``(B, leads, 2)`` a physics track in degrees, from
                :func:`naturev1.steering_track`; the track head then predicts the residual from it.
                Pass it only if the head was trained with one -- see :meth:`uses_track_prior`.
            storm_satellite: ``(codes, available)`` storm-centred infrared from
                :mod:`naturev1.stormsat` -- ``(B, 2, P, P)`` uint8 and ``(B, 2)`` bool. Optional.
            storm_satnet: the satellite storm network's outputs for these storms
                (:meth:`satnet_outputs`), fused into the heads and fed to RI 2.0. Optional.
            storm_physics: ``(B, len(STORM_PHYSICS_FEATURES))`` from :func:`naturev1.thermo.storm_physics`.
                Optional.

        Returns:
            A dict of forecasts. Gridded entries are ``(B, P, leads, ...)``; storm-scale entries are
            ``(B, ...)``. Every one carries an uncertainty.
        """
        latent = self.encode(satellite, satellite_grid, analysis, analysis_grid, calendar,
                             environment, environment_grid, neighbours)
        if self.noise_proj is not None and member_noise is not None:
            latent = latent + self.noise_proj(member_noise.to(latent.dtype)).unsqueeze(1)
        latent = self._run_blocks(latent, self.latent_grid)
        latent = self.norm(latent)

        # The last frame is "now"; the heads forecast forward from it.
        current = latent[:, -1]
        batch = current.shape[0]
        output_grid = output_grid or self.latent_grid
        if output_grid is self.latent_grid:
            decoded = current
        else:
            seed = self.mesh_seed.expand(batch, output_grid.num_points, -1)
            decoded = self.decode_norm(
                self.decoder(seed, current, self.link(output_grid, self.latent_grid, neighbours))
            )

        points, leads = decoded.shape[1], self.config.num_leads
        gridded = self.field_head(decoded).view(batch, points, leads, len(SURFACE_FIELDS), 2)

        # Area-weighted pooling: the mesh is near-equal-area, but weighting anyway keeps the summary
        # honest if someone swaps in a mesh that is not. ENSO is a basin-scale quantity and always
        # reads the whole planet; the storm heads read around the storm when told where it is.
        area = self.latent_grid.weights.to(current.dtype).view(1, -1, 1)
        global_summary = self.summary_pool(self.summary_norm((current * area).sum(1) / area.sum()))
        if (storm_center is None and storm_state is None and storm_environment is None and track_prior is None
                and storm_satellite is None and storm_satnet is None and storm_physics is None):
            storm = self._storm_outputs(global_summary)
        else:
            far = None if storm_center is None else self.storm_pool(
                current, storm_center, 2.0 * self.config.storm_pool_radius_km)
            storm = self.storm_heads(self.storm_pool(current, storm_center), storm_center, storm_state,
                                     storm_environment, track_prior, far, storm_satellite, storm_satnet, storm_physics)
            if self.config.storm_rotations > 1 and not self.training and storm_center is not None:
                sources = {"satellite": (satellite, satellite_grid), "analysis": (analysis, analysis_grid),
                           "environment": (environment, environment_grid)}
                members = [storm]
                for degrees in rotation_angles(self.config.storm_rotations):
                    near, wide = self.rotated_pools(sources, calendar, storm_center, degrees, neighbours, member_noise)
                    members.append(self.storm_heads(near, storm_center, storm_state, storm_environment, track_prior,
                                                    wide, storm_satellite, storm_satnet, storm_physics))
                storm = merge_rotations(members)
        enso = self.enso_head(global_summary)

        state = {}
        if self.state_head is not None:
            residual = self.state_head(decoded).view(batch, points, self.config.state_channels, 2)
            state = {"state_delta": residual[..., 0],
                     "state_delta_log_var": residual[..., 1].clamp(-10.0, 10.0)}

        return {
            **state,
            "field_mean": gridded[..., 0],
            "field_log_var": gridded[..., 1].clamp(-10.0, 10.0),
            "weather_type_logits": self.weather_type_head(decoded).view(batch, points, leads, len(WEATHER_TYPES)),
            "enso_mean": enso[..., 0],
            "enso_log_var": enso[..., 1].clamp(-10.0, 10.0),
            **storm,
        }

    # ------------------------------------------------------------------------------ storm path --
    #
    # Split in three so stage two can run the frozen backbone once per storm and train the heads on
    # the result: `storm_features` (backbone + pool, expensive, frozen), then `storm_heads` (cheap,
    # trained). `forward` goes through exactly the same two functions, so the fast path and the full
    # model cannot disagree.

    def rotated_grid(self, grid: FieldGrid, degrees: float) -> FieldGrid:
        """``grid`` turned east by ``degrees`` (a read-only source grid), built once per grid and angle."""
        from .corpora import rotate_longitude

        cache = self.__dict__.setdefault("_rotated_grids", OrderedDict())
        key = (grid_fingerprint(grid), round(float(degrees), 9))
        if key not in cache:
            coords = rotate_longitude(grid.coords, math.radians(degrees))
            cache[key] = FieldGrid.source(coords, grid.geometry, weights=grid.weights)
            while len(cache) > self.link_cache_size:
                cache.popitem(last=False)
        cache.move_to_end(key)
        return cache[key]

    def rotated_pools(self, sources: dict, calendar: torch.Tensor, storm_center: torch.Tensor, degrees: float,
                      neighbours: int = 24, member_noise: torch.Tensor | None = None):
        """
        The near and far storm pools with the whole planet turned east by ``degrees`` -- one member of the
        test-time rotation ensemble.

        Every source is read onto the mesh from a grid turned by the same angle, so the atmosphere lands
        on different mesh points and is processed by genuinely different arithmetic, while the storm, its
        latitude, its Coriolis parameter and everything around it are unchanged: the local strand reads
        neighbours in each point's own east/north frame, which turns with it. The storm centre turns too,
        and the heads still see the true centre, so nothing that depends on where the storm is moves.

        Args:
            sources: ``{"satellite"|"analysis"|"environment": (samples or None, grid)}``.
        """
        turned = {}
        for name in ("satellite", "analysis", "environment"):
            values, grid = sources.get(name, (None, None))
            turned[name] = (values, None if values is None else self.rotated_grid(self._aligned(grid), degrees))
        latent = self.encode(turned["satellite"][0], turned["satellite"][1], turned["analysis"][0],
                             turned["analysis"][1], calendar, turned["environment"][0], turned["environment"][1],
                             neighbours)
        if self.noise_proj is not None and member_noise is not None:
            latent = latent + self.noise_proj(member_noise.to(latent.dtype)).unsqueeze(1)
        current = self.norm(self._run_blocks(latent, self.latent_grid))[:, -1]
        centre = turned_centre(storm_center.to(current.device, torch.float32), degrees)
        if current.shape[0] == 1 and centre.shape[0] > 1:
            # One atmosphere, several storms in it: one backbone pass, one pool per storm.
            current = current.expand(centre.shape[0], -1, -1)
        return (self.storm_pool(current, centre),
                self.storm_pool(current, centre, 2.0 * self.config.storm_pool_radius_km))

    def storm_pool(self, current: torch.Tensor, storm_center: torch.Tensor | None,
                   radius_km: float | None = None) -> torch.Tensor:
        """
        Pool the mesh around each sample's storm, with a Gaussian of ``radius_km`` (``storm_pool_radius_km``).

        A mean over the whole planet -- which is what the storm heads read before this -- is the same
        number for every storm active at that hour. In a busy Atlantic September three or four storms
        share a timestamp, so the heads were being asked to give different answers to identical inputs,
        and had no way to know which storm a label was about. Pooling around the centre fixes both.

        Args:
            current: ``(B, P_latent, hidden)`` the processed mesh at "now".
            storm_center: ``(B, 2)`` degrees, or None / NaN rows for the global pool.

        Returns:
            ``(B, hidden)`` pooled features, before any trained layer.
        """
        latent = self.latent_grid
        weights = latent.weights.to(torch.float64).view(1, -1).expand(current.shape[0], -1)
        if storm_center is not None:
            centre = storm_center.to(latent.points.device, torch.float64)
            present = torch.isfinite(centre).all(-1)
            position = latent.geometry.embed(torch.deg2rad(torch.nan_to_num(centre)))       # (B, 3) km
            distance = torch.cdist(position, latent.points.to(torch.float64))               # chordal km
            local = torch.exp(-0.5 * (distance / (radius_km or self.config.storm_pool_radius_km)) ** 2)
            weights = torch.where(present.view(-1, 1), weights * local, weights)
        weights = (weights / weights.sum(-1, keepdim=True).clamp_min(1e-30)).to(current.dtype)
        return (current * weights.unsqueeze(-1)).sum(1)

    def storm_heads(self, pooled: torch.Tensor, storm_center: torch.Tensor | None = None,
                    storm_state: torch.Tensor | None = None, storm_environment: torch.Tensor | None = None,
                    track_prior: torch.Tensor | None = None, pooled_far: torch.Tensor | None = None,
                    storm_satellite: tuple[torch.Tensor, torch.Tensor] | None = None,
                    storm_satnet: dict | None = None, storm_physics: torch.Tensor | None = None,
                    rollout: dict | None = None,
                    ) -> dict[str, torch.Tensor]:
        """
        Every storm output, from pooled features plus where the storm is, how strong it is, what the
        atmosphere around it is doing (``storm_environment``), the wider pool (``pooled_far``), what it
        looks like in infrared (``storm_satellite``), and -- when a ``track_prior`` is given -- as a
        residual on that physics track, which is returned too.

        Once fitted, the landfall and rapid-intensification logits come back calibrated, and
        ``feature_score`` says how far the pooled features sit from what training looked like.

        With ``config.hemisphere_frame`` every north-south quantity is mirrored for southern-hemisphere
        storms on the way in and back on the way out, so the heads learn one poleward frame from both
        hemispheres. ``storm_satnet`` (the satellite storm network's outputs) is fused in and feeds RI 2.0.
        """
        dtype = pooled.dtype
        original_prior = track_prior
        sign = None
        if self.config.hemisphere_frame and storm_center is not None:
            from .ibtracs import hemisphere_sign

            sign = hemisphere_sign(torch.nan_to_num(storm_center[:, 0].to(pooled.device).float()))
            if storm_state is not None:
                storm_state = storm_state.clone()
                storm_state[:, 2] = storm_state[:, 2] * sign.to(storm_state.dtype)
            if storm_environment is not None:
                storm_environment = storm_environment.clone()
                for column in (1, 3):                                   # steering_v, shear_v
                    storm_environment[:, column] = storm_environment[:, column] * sign.to(storm_environment.dtype)
            if track_prior is not None:
                track_prior = track_prior.clone()
                track_prior[..., 0] = track_prior[..., 0] * sign.view(-1, 1).to(track_prior.dtype)
            if storm_physics is not None:
                from .thermo import STORM_PHYSICS_FEATURES

                storm_physics = storm_physics.clone()
                column = STORM_PHYSICS_FEATURES.index("steering_shear_v")
                storm_physics[:, column] = storm_physics[:, column] * sign.to(storm_physics.dtype)
        score = None
        if float(self.feature_threshold) > 0:
            features = pooled.to(self.feature_mean.dtype)
            score = ((features - self.feature_mean) / self.feature_std.clamp_min(1e-6)).pow(2).mean(-1)
            # A vector of the right shape but the wrong size (all zeros, say) can sit near the mean in every
            # dimension; its length cannot.
            norm = features.norm(dim=-1)
            outside = (norm < self.feature_norm_bounds[0]) | (norm > self.feature_norm_bounds[1])
            score = torch.where(outside, torch.full_like(score, float("inf")), score)
        if pooled_far is not None:
            pooled = pooled + self.storm_far_pool(pooled_far.to(pooled.device, dtype))
        if storm_center is not None:
            centre = storm_center.to(pooled.device, torch.float64)
            present = torch.isfinite(centre).all(-1, keepdim=True)
            if sign is not None:
                # In the poleward frame a storm at 20 S is told it is at 20 N: what latitude sets -- the
                # Coriolis parameter, the beta drift -- depends on its size, and the frame owns the sign.
                centre = torch.stack([centre[:, 0] * sign.to(centre.dtype), centre[:, 1]], -1)
            position = self.latent_grid.geometry.embed(torch.deg2rad(torch.nan_to_num(centre)))
            unit = position / position.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            pooled = pooled + self.storm_position(unit.to(dtype)) * present.to(dtype)
        if storm_state is not None:
            pooled = pooled + self.storm_state_embed(torch.nan_to_num(storm_state.to(pooled.device, dtype)))
        if storm_environment is not None:
            pooled = pooled + self.storm_environment_embed(
                torch.nan_to_num(storm_environment.to(pooled.device, dtype)))
        if storm_satellite is not None:
            codes, available = storm_satellite
            pooled = pooled + self.storm_satellite(codes.to(pooled.device), available.to(pooled.device)).to(dtype)
        if storm_physics is not None:
            pooled = pooled + self.storm_physics_embed(torch.nan_to_num(storm_physics.to(pooled.device, dtype)))
        if storm_satnet is not None and self.satnet_fusion is not None and "embedding" in storm_satnet:
            embedding = storm_satnet["embedding"].to(pooled.device, dtype)
            if "present" in storm_satnet:
                embedding = embedding * storm_satnet["present"].to(pooled.device, dtype).view(-1, 1)
            pooled = pooled + self.satnet_fusion(embedding)
        summary = self.summary_pool(self.summary_norm(pooled))
        outputs = self._storm_outputs(summary)
        landfall, ri = self.landfall_calibration.to(dtype), self.ri_calibration.to(dtype)
        outputs["landfall_logit"] = outputs["landfall_logit"] * landfall[:, 0] + landfall[:, 1]
        outputs["ri_logits"] = outputs["ri_logits"] * ri[:, 0] + ri[:, 1]
        if self.ri2_head is not None:
            predictors = ri2_predictors(summary.shape[0], summary.device, storm_center, storm_state,
                                        storm_environment, storm_satnet, storm_physics)
            calibration = self.ri2_calibration.to(dtype)
            outputs["ri2_logits"] = self.ri2_head(summary, predictors) * calibration[:, 0] + calibration[:, 1]
        if score is not None:
            outputs["feature_score"] = score
            outputs["feature_threshold"] = self.feature_threshold.expand_as(score)
        if track_prior is not None:
            prior = torch.nan_to_num(track_prior.to(outputs["displacement"].device, outputs["displacement"].dtype))
            outputs["displacement"] = outputs["displacement"] + prior.unsqueeze(1)    # every scenario
            outputs["track_prior"] = prior
        if sign is not None:
            flip = sign.to(outputs["displacement"].dtype)
            displacement = outputs["displacement"].clone()
            displacement[..., 0] = displacement[..., 0] * flip.view(-1, 1, 1)
            outputs["displacement"] = displacement
            outputs["correlation"] = outputs["correlation"] * flip.view(-1, 1, 1)
            south = (flip < 0).view(-1, 1, 1, 1)
            for key in ("wind_radii_nmi", "wind_radii_log_var"):
                value = outputs[key]
                outputs[key] = torch.where(south, value[..., [1, 0, 3, 2]], value)
            if original_prior is not None:
                outputs["track_prior"] = torch.nan_to_num(original_prior.to(displacement.device, displacement.dtype))
        self._apply_intensity_correction(outputs)
        if int(self.statistics.fitted[:4].sum()) > 0 and storm_state is not None:
            # The statistical-dynamical members and their corrected consensus (naturev1.statistical).
            from .statistical import intensity_members, intensity_predictors

            x = intensity_predictors(storm_state, storm_environment, storm_physics, storm_center, storm_satnet)
            members = intensity_members(self, outputs, x, storm_state, storm_environment, storm_physics, storm_satnet,
                                        None if rollout is None else rollout.get("rollout_wind_kt"),
                                        None if rollout is None else rollout.get("rollout_pressure_hpa"))
            outputs["intensity_members_kt"] = members
            if int(self.statistics.fitted[3]) > 0:
                outputs["intensity_consensus_kt"] = self.statistics.consensus(members)
            if int(self.statistics.fitted[7]) > 0:
                from .statistical import ri_consensus_probability, ri_member_logits

                wind_now = torch.nan_to_num(storm_state[:, 0].to(members.device).float()
                                            * PEAK_WIND_ANCHOR_KT[1] + PEAK_WIND_ANCHOR_KT[0])
                outputs["ri_consensus_probability"] = ri_consensus_probability(
                    self, ri_member_logits(self, outputs, x, wind_now, storm_satnet))
        if storm_satnet is not None:
            for key in ("wind_kt", "wind_log_var", "pressure_hpa", "centre_offset_km", "wind_change_kt", "ri_logits"):
                if key in storm_satnet:
                    outputs[f"satnet_{key}"] = storm_satnet[key].to(outputs["displacement"].device)
        return outputs

    def _apply_intensity_correction(self, outputs: dict) -> None:
        """Apply the fitted per-lead bias correction (offset + slope around 65 kt) to wind; identity at zero."""
        if float(self.intensity_correction.abs().sum()) > 0:
            correction = self.intensity_correction.to(outputs["intensity_mean"].dtype)
            wind_kt = outputs["intensity_mean"][..., 0] / 0.514444
            fixed = wind_kt + correction[:, 0] + correction[:, 1] * (wind_kt - 65.0) / 30.0
            mean = outputs["intensity_mean"].clone()
            mean[..., 0] = fixed.clamp_min(0.0) * 0.514444
            outputs["intensity_mean"] = mean
        if float(self.eyewall_correction.abs().sum()) > 0:
            correction = self.eyewall_correction.to(outputs["eyewall_peak_wind_kt"].dtype)
            peak = outputs["eyewall_peak_wind_kt"]
            outputs["eyewall_peak_wind_kt"] = (peak + correction[:, 0] + correction[:, 1] * (peak - 65.0) / 30.0).clamp_min(0.0)

    def uses_track_prior(self) -> bool:
        """Whether the track head was trained as a residual on a steering track -- and so needs one."""
        return int(self.head_steps[TRACKED_HEADS.index("track_prior")]) > 0

    def _storm_outputs(self, summary: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, leads = summary.shape[0], self.config.num_leads
        landfall = self.landfall_head(summary).view(batch, leads, 2)
        intensity = self.intensity_head(summary).view(batch, leads, 2, 2)
        # Wind in m/s and pressure in absolute hPa, anchored at climatology -- see the constants above.
        anchor_mean = torch.tensor([WIND_ANCHOR_MS[0], PRESSURE_ANCHOR_HPA[0]], device=summary.device)
        anchor_scale = torch.tensor([WIND_ANCHOR_MS[1], PRESSURE_ANCHOR_HPA[1]], device=summary.device)
        return {
            "landfall_logit": landfall[..., 0],
            "landfall_time_log_var": landfall[..., 1].clamp(-10.0, 10.0),
            "intensity_mean": intensity[..., 0] * anchor_scale + anchor_mean,
            "intensity_log_var": intensity[..., 1].clamp(-10.0, 10.0),
            **self.track_head(summary),
            **self.eyewall_head(summary),
            **self.ri_head(summary),
        }

    @torch.no_grad()
    def storm_features(self, analysis: torch.Tensor, analysis_grid: FieldGrid, calendar: torch.Tensor,
                       storm_center: torch.Tensor, neighbours: int = 24, far: bool = False):
        """
        The frozen half of the storm path: backbone once, pooled around each storm. ``(B, hidden)``.

        With the backbone frozen for stage two, this is a pure function of the reanalysis and the
        storm's position, so it is computed once per sample and cached -- see
        :func:`naturev1.storm_feature_bank`. Training the heads then costs milliseconds per epoch
        instead of an 89-channel ERA5 read per sample. With ``far=True`` the wider pool comes back as
        a second tensor.
        """
        latent = self.encode(None, None, analysis, analysis_grid, calendar, neighbours=neighbours)
        current = self.norm(self._run_blocks(latent, self.latent_grid))[:, -1]
        near = self.storm_pool(current, storm_center)
        if not far:
            return near
        return near, self.storm_pool(current, storm_center, 2.0 * self.config.storm_pool_radius_km)

    # ------------------------------------------------------------------- what has been trained --

    def mark_trained(self, parts: dict, batch: dict | None = None) -> None:
        """
        Record one optimizer step for every output that just received a gradient.

        Called by :class:`naturev1.Trainer` and :func:`naturev1.state_rollout_loss`. A head counts only if
        its loss term was present *and* its parameters were trainable -- a frozen field head does not
        become "trained" because a fine-tuning batch happened to carry field targets.
        """
        modules = {
            "field": self.field_head, "weather_type": self.weather_type_head,
            "track": self.track_head, "landfall": self.landfall_head,
            "intensity": self.intensity_head, "ri": self.ri_head.classifier,
            "ri_delta": self.ri_head.magnitude, "eyewall_peak": self.eyewall_head.peak_wind,
            "eyewall_rmw": self.eyewall_head.rmw, "wind_radii": self.eyewall_head.wind_radii,
            "enso": self.enso_head, "state": self.state_head,
            "satnet": self.storm_satnet, "retrieval": self.retrieval, "ri2": self.ri2_head,
            "backbone_tail": self.blocks,
        }
        touched = [name for name, value in parts.items()
                   if name in modules and value != 0.0 and modules[name] is not None]
        if batch is not None:
            for key, name in (("analysis", "analysis_encoder"), ("satellite", "satellite_encoder"),
                              ("environment", "environment_encoder"), ("storm_center", "storm_position"),
                              ("storm_state", "storm_state"), ("storm_environment", "storm_environment"),
                              ("pooled_far", "storm_far"), ("track_prior", "track_prior"),
                              ("storm_satellite", "storm_satellite"), ("storm_satnet", "satnet_fusion"),
                              ("storm_physics", "storm_physics")):
                if batch.get(key) is not None:
                    touched.append(name)
            modules.update({"analysis_encoder": self.analysis_encoder,
                            "satellite_encoder": self.satellite_encoder,
                            "environment_encoder": self.environment_encoder,
                            "storm_position": self.storm_position,
                            "storm_state": self.storm_state_embed,
                            "storm_environment": self.storm_environment_embed,
                            "storm_far": self.storm_far_pool,
                            # The prior has no parameters of its own: it is "trained" when the track
                            # head learns on top of it, which is what the record needs to say.
                            "track_prior": self.track_head,
                            "storm_satellite": self.storm_satellite,
                            "satnet_fusion": self.satnet_fusion,
                            "storm_physics": self.storm_physics_embed})
        with torch.no_grad():
            for name in set(touched):
                if any(p.requires_grad for p in modules[name].parameters()):
                    self.head_steps[TRACKED_HEADS.index(name)] += 1

    def trained_heads(self) -> dict[str, int]:
        """Optimizer steps each output has received. Zero means it is still at initialisation."""
        return {name: int(count) for name, count in zip(TRACKED_HEADS, self.head_steps.tolist())}

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """
        Load weights, accepting checkpoints written before the 0.8 additions.

        :data:`OPTIONAL_STATE` lists what is new; every one of those starts as an exact no-op, so a
        0.7 checkpoint loads into a 0.8 model and behaves exactly as it did. Anything else missing, any
        unexpected key and any shape mismatch is still an error.
        """
        steps = state_dict.get("head_steps")
        if steps is not None and steps.numel() < len(TRACKED_HEADS):
            # Written before some outputs existed: those were never trained, which is what zero says.
            state_dict = dict(state_dict)
            state_dict["head_steps"] = torch.cat([steps, steps.new_zeros(len(TRACKED_HEADS) - steps.numel())])
        weights = state_dict.get("track_consensus")
        if weights is not None and weights.shape[-1] < len(TRACK_MEMBERS):
            # Fitted over the four members before 0.12; the newer members get no weight until refitted.
            state_dict = dict(state_dict)
            state_dict["track_consensus"] = torch.cat(
                [weights, weights.new_zeros(*weights.shape[:-1], len(TRACK_MEMBERS) - weights.shape[-1])], -1)
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        missing = [key for key in result.missing_keys if not key.startswith(OPTIONAL_STATE)]
        fresh = [key for key in result.missing_keys if key.startswith(OPTIONAL_STATE)]
        if strict and (missing or result.unexpected_keys):
            raise RuntimeError(f"state dict does not match the model: missing {missing}, "
                               f"unexpected {result.unexpected_keys}")
        if fresh:
            print(f"[naturev1] checkpoint predates {sorted({k.split('.')[0] for k in fresh})}; "
                  "those start at their no-op initialisation", flush=True)
        return result

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def storm_head_parameters(self):
        """
        The heads that learn from best-track labels, which is the scarce data.

        Freezing everything else and training only these is what keeps an 88M network from memorising
        two thousand storms: the number of parameters actually fit to the small labels is this, not the
        whole model.
        """
        for module in (self.track_head, self.eyewall_head, self.ri_head, self.landfall_head,
                       self.intensity_head, self.summary_pool, self.summary_norm, self.storm_position,
                       self.storm_state_embed, self.storm_environment_embed, self.storm_far_pool,
                       self.storm_satellite, self.satnet_fusion, self.ri2_head, self.storm_physics_embed):
            if module is not None:
                yield from module.parameters()

    def freeze_backbone(self, frozen: bool = True) -> tuple[int, int]:
        """
        Freeze everything except the storm heads. Returns ``(trainable, total)`` parameter counts.

        Use after self-supervised pretraining on reanalysis, before fine-tuning on best tracks.
        """
        for parameter in self.parameters():
            parameter.requires_grad = not frozen
        for parameter in self.storm_head_parameters():
            parameter.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, self.num_parameters()


def calendar_features(timestamps: torch.Tensor) -> torch.Tensor:
    """
    Cyclic time features: hour of day and day of year as sines and cosines, plus a slow linear term.

    Given rather than inferred, because solar angle and season are physical drivers the model should not
    have to rediscover, and because sine/cosine pairs have no discontinuity at midnight or New Year.

    Args:
        timestamps: ``(..., )`` POSIX seconds, UTC.

    Returns:
        ``(..., 6)`` features.
    """
    seconds = timestamps.to(torch.float64)
    day = (seconds % 86400.0) / 86400.0
    year = (seconds % 31557600.0) / 31557600.0
    epoch = seconds / 3.15576e9  # centuries, a slow trend term
    two_pi = 2 * math.pi
    return torch.stack(
        [
            torch.sin(two_pi * day), torch.cos(two_pi * day),
            torch.sin(two_pi * year), torch.cos(two_pi * year),
            epoch, torch.zeros_like(epoch),
        ],
        dim=-1,
    ).to(torch.float32)
