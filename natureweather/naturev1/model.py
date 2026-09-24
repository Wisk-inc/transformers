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
)
#: Parameters added after 0.7. A checkpoint written before them loads with these freshly initialised --
#: all of them start as exact no-ops, so the loaded model behaves exactly as it did when it was saved.
OPTIONAL_STATE = ("head_steps", "storm_position.", "storm_state_embed.", "state_head.", "noise_proj.")

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
        if "lead_times_hours" in payload:
            payload["lead_times_hours"] = tuple(payload["lead_times_hours"])
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
        # Training steps each output has actually received -- saved with the weights, so a forecast
        # built from a checkpoint knows which of its heads were ever trained.
        self.register_buffer("head_steps", torch.zeros(len(TRACKED_HEADS), dtype=torch.int64))

        self.apply(self._init)
        self._anchor_uncertainty()
        self._zero_init_residual_paths()
        self._links: OrderedDict = OrderedDict()

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
        if storm_center is None and storm_state is None:
            storm = self._storm_outputs(global_summary)
        else:
            storm = self.storm_heads(self.storm_pool(current, storm_center), storm_center, storm_state)
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

    def storm_pool(self, current: torch.Tensor, storm_center: torch.Tensor | None) -> torch.Tensor:
        """
        Pool the mesh around each sample's storm, with a Gaussian of ``storm_pool_radius_km``.

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
            local = torch.exp(-0.5 * (distance / self.config.storm_pool_radius_km) ** 2)
            weights = torch.where(present.view(-1, 1), weights * local, weights)
        weights = (weights / weights.sum(-1, keepdim=True).clamp_min(1e-30)).to(current.dtype)
        return (current * weights.unsqueeze(-1)).sum(1)

    def storm_heads(self, pooled: torch.Tensor, storm_center: torch.Tensor | None = None,
                    storm_state: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Every storm output, from pooled features plus where the storm is and how strong it is."""
        dtype = pooled.dtype
        if storm_center is not None:
            centre = storm_center.to(pooled.device, torch.float64)
            present = torch.isfinite(centre).all(-1, keepdim=True)
            position = self.latent_grid.geometry.embed(torch.deg2rad(torch.nan_to_num(centre)))
            unit = position / position.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            pooled = pooled + self.storm_position(unit.to(dtype)) * present.to(dtype)
        if storm_state is not None:
            pooled = pooled + self.storm_state_embed(torch.nan_to_num(storm_state.to(pooled.device, dtype)))
        return self._storm_outputs(self.summary_pool(self.summary_norm(pooled)))

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
                       storm_center: torch.Tensor, neighbours: int = 24) -> torch.Tensor:
        """
        The frozen half of the storm path: backbone once, pooled around each storm. ``(B, hidden)``.

        With the backbone frozen for stage two, this is a pure function of the reanalysis and the
        storm's position, so it is computed once per sample and cached -- see
        :func:`naturev1.storm_feature_bank`. Training the heads then costs milliseconds per epoch
        instead of an 89-channel ERA5 read per sample.
        """
        latent = self.encode(None, None, analysis, analysis_grid, calendar, neighbours=neighbours)
        latent = self.norm(self._run_blocks(latent, self.latent_grid))
        return self.storm_pool(latent[:, -1], storm_center)

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
        }
        touched = [name for name, value in parts.items()
                   if name in modules and value != 0.0 and modules[name] is not None]
        if batch is not None:
            for key, name in (("analysis", "analysis_encoder"), ("satellite", "satellite_encoder"),
                              ("environment", "environment_encoder"), ("storm_center", "storm_position"),
                              ("storm_state", "storm_state")):
                if batch.get(key) is not None:
                    touched.append(name)
            modules.update({"analysis_encoder": self.analysis_encoder,
                            "satellite_encoder": self.satellite_encoder,
                            "environment_encoder": self.environment_encoder,
                            "storm_position": self.storm_position,
                            "storm_state": self.storm_state_embed})
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
                       self.storm_state_embed):
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
