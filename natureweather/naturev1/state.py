# Copyright 2026 Nathan. Apache-2.0.
"""
The full atmospheric state: predicting all of it, evolving all of it, and conserving what physics says
must be conserved.

This module exists because two things shipped separately and did not compose. The upper air added 89
input channels; the rollout fed forecasts back in. But the model only ever predicted seven surface
fields, so in a twelve-step rollout **84 of 89 channels were frozen at their initial value**. Z500 --
the headline WeatherBench number -- never changed, which made its "forecast" persistence by
construction. And incident solar radiation was frozen too, so the model saw the same sun position for
three days.

The fix is not a patch to the rollout; it is a change in what the model predicts. Four ideas, each from
the systems that currently lead the WeatherBench tables, each verified here:

**1. Predict the change, not the state** (GraphCast). Every prognostic channel gets a residual head:
the model outputs ``x(t+6h) - x(t)``. This makes persistence the zero output rather than something
the model has to learn to reproduce, and -- with the output projection initialised to zero -- an
untrained model *is* persistence exactly, and can only improve on it from there.

**2. Normalise the change by its own spread** (GraphCast). A six-hour change in temperature is a
fraction of a kelvin; the variance of temperature itself is twenty. Predicting the change in units of
the *state* standard deviation hands the loss numbers near 0.01 and makes every channel's gradient
depend on how volatile it happens to be. :func:`tendency_std` measures the spread of the change itself.

**3. Separate what is predicted from what is known.** Static fields (land-sea mask, orography) never
change. Solar radiation is a deterministic function of time and place -- computed here from orbital
geometry, it matches ERA5's own field with correlation 1.00000 and error under 1% of peak across thirty
years. Predicting either would be spending capacity to approximate something exact.

**4. Conserve mass.** The global area-weighted mean of surface pressure is the total mass of the
atmosphere, which does not change in six hours. Nothing in a neural network knows that. Unconstrained,
it drifts, and a drift in total mass is a drift in every pressure field on the planet. The projection in
:func:`conserve_mass` removes it exactly, at the cost of one weighted mean.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch


#: Fields that never change: geography, not weather. Carried forward exactly, never predicted.
STATIC_VARIABLES = frozenset({
    "land_sea_mask", "geopotential_at_surface", "slope_of_sub_gridscale_orography",
    "standard_deviation_of_orography", "standard_deviation_of_filtered_subgrid_orography",
    "angle_of_sub_gridscale_orography", "anisotropy_of_sub_gridscale_orography",
    "soil_type", "lake_cover", "lake_depth", "high_vegetation_cover", "low_vegetation_cover",
    "type_of_high_vegetation", "type_of_low_vegetation",
})

#: Fields known exactly in advance. Recomputed for each step's valid time, never predicted.
FORCING_VARIABLES = frozenset({
    "toa_incident_solar_radiation", "toa_incident_solar_radiation_6hr",
    "toa_incident_solar_radiation_12hr", "toa_incident_solar_radiation_24hr",
})

#: Accumulation window of each solar field, in hours.
SOLAR_WINDOW_HOURS = {
    "toa_incident_solar_radiation": 1.0, "toa_incident_solar_radiation_6hr": 6.0,
    "toa_incident_solar_radiation_12hr": 12.0, "toa_incident_solar_radiation_24hr": 24.0,
}

#: Total solar irradiance at 1 AU, W/m^2 (TSIS-1, Kopp 2021).
SOLAR_CONSTANT = 1361.0


def _base(channel: str) -> str:
    return channel.split("@", 1)[0]


def _level(channel: str) -> int | None:
    parts = channel.split("@", 1)
    return int(parts[1]) if len(parts) == 2 else None


@dataclass
class ChannelRoles:
    """
    What each input channel *is*, which decides what happens to it at every rollout step.

    * **prognostic** -- the weather. Predicted as a residual and evolved.
    * **forcing** -- known exactly in advance (solar). Recomputed for each step's valid time.
    * **static** -- geography, observation masks, and zero padding. Carried forward unchanged.

    Getting a channel's role wrong is silent. A static field marked prognostic lets the model slowly
    move a coastline; a prognostic field marked static freezes the upper atmosphere, which is exactly
    the bug this class replaced.
    """

    channels: list[str]
    prognostic: list[int] = field(default_factory=list)
    forcing: list[int] = field(default_factory=list)
    static: list[int] = field(default_factory=list)

    @property
    def num_prognostic(self) -> int:
        return len(self.prognostic)

    def describe(self) -> str:
        def names(indices, limit=6):
            shown = [self.channels[i] if i < len(self.channels) else f"<pad {i}>" for i in indices[:limit]]
            more = f" ... +{len(indices) - limit}" if len(indices) > limit else ""
            return ", ".join(shown) + more

        return "\n".join([
            f"  prognostic {len(self.prognostic):>4}  predicted and evolved      {names(self.prognostic)}",
            f"  forcing    {len(self.forcing):>4}  recomputed each step       {names(self.forcing)}",
            f"  static     {len(self.static):>4}  carried forward unchanged  {names(self.static)}",
        ])


def channel_roles(variables: list[str], input_channels: int) -> ChannelRoles:
    """
    Classify every input channel.

    The dataset lays channels out as ``[variables] + [observation masks] + [zero padding]``. Masks mark
    where a gappy variable was measured -- sea-surface temperature exists over ocean -- and the ocean
    does not move, so they are static. Padding is static by definition.

    Args:
        variables: the flat variable names, from the dataset's ``variables``.
        input_channels: the model's ``analysis_channels``, which may be wider than the variables.
    """
    roles = ChannelRoles(channels=list(variables))
    for index, name in enumerate(variables):
        base = _base(name)
        if base in STATIC_VARIABLES:
            roles.static.append(index)
        elif base in FORCING_VARIABLES:
            roles.forcing.append(index)
        else:
            roles.prognostic.append(index)
    roles.static.extend(range(len(variables), input_channels))
    return roles


# --------------------------------------------------------------------------------------------------
# Solar forcing, from orbital geometry
# --------------------------------------------------------------------------------------------------

def toa_flux(latitude_deg, longitude_deg, unix_seconds) -> np.ndarray:
    """
    Instantaneous top-of-atmosphere solar irradiance, W/m^2.

    Low-precision solar ephemeris (Astronomical Almanac), accurate to about 0.01 degree in the sun's
    position -- far below anything that matters at a 1.5-degree grid. Includes the Earth's orbital
    eccentricity, which moves the flux by 6.7% between perihelion and aphelion.
    """
    t = np.asarray(unix_seconds, dtype=np.float64)
    days = t / 86400.0 - 10957.5                                  # since J2000.0
    anomaly = np.radians((357.529 + 0.98560028 * days) % 360.0)
    longitude_mean = np.radians((280.459 + 0.98564736 * days) % 360.0)
    ecliptic = (longitude_mean + np.radians(1.915) * np.sin(anomaly)
                + np.radians(0.020) * np.sin(2 * anomaly))
    obliquity = np.radians(23.439 - 0.00000036 * days)
    declination = np.arcsin(np.sin(obliquity) * np.sin(ecliptic))
    right_ascension = np.arctan2(np.cos(obliquity) * np.sin(ecliptic), np.cos(ecliptic))
    distance_au = 1.00014 - 0.01671 * np.cos(anomaly) - 0.00014 * np.cos(2 * anomaly)
    sidereal = np.radians((280.46061837 + 360.98564736629 * days) % 360.0)
    hour_angle = sidereal + np.radians(longitude_deg) - right_ascension

    latitude = np.radians(latitude_deg)
    cos_zenith = (np.sin(latitude) * np.sin(declination)
                  + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle))
    return SOLAR_CONSTANT / distance_au**2 * np.clip(cos_zenith, 0.0, None)


def toa_accumulated(latitude_deg, longitude_deg, valid_unix, window_hours: float = 6.0,
                    substeps: int = 72) -> np.ndarray:
    """
    Solar energy accumulated over the window *ending* at the valid time, J/m^2.

    ERA5 accumulates over the preceding period, which was confirmed against the store rather than
    assumed: with this convention the field reproduces at correlation 1.00000 and error under 1% of
    peak from 1959 to 1989. The residual is conservative regridding -- WeatherBench area-averages each
    cell while this evaluates at the centre.
    """
    window = window_hours * 3600.0
    offsets = (np.arange(substeps) + 0.5) / substeps * window
    total = np.zeros(np.broadcast(np.asarray(latitude_deg), np.asarray(longitude_deg)).shape)
    for offset in offsets:
        total = total + toa_flux(latitude_deg, longitude_deg, valid_unix - window + offset)
    return total * (window / substeps)


def solar_forcing(grid, unix_seconds, channel: str, normalizer, roll_degrees=None) -> torch.Tensor:
    """
    The normalized solar channel for one valid time, on a grid -- ready to write into the input.

    Args:
        grid: any :class:`ihelix.FieldGrid` on the globe; coordinates are read from it.
        unix_seconds: ``(B,)`` valid times.
        channel: the solar channel's name, which fixes the accumulation window.
        normalizer: the dataset's normalizer, so the result is in the units the model was trained on.
        roll_degrees: ``(B,)`` how far each sample's globe was rotated east by augmentation. The
            rotated sample at longitude L holds the weather from L - roll, so its sun is computed there
            too. Ignoring it put the sun up to half a planet away from the weather it was lighting.
    """
    coords = grid.coords.detach().cpu().numpy()
    latitude = np.degrees(coords[:, 0])
    longitude = np.degrees(coords[:, 1])
    window = SOLAR_WINDOW_HOURS.get(_base(channel), 6.0)

    index = normalizer.variables.index(channel)
    mean, std = float(normalizer.mean[index]), float(normalizer.std[index])
    times = np.atleast_1d(np.asarray(unix_seconds, dtype=np.float64))
    rolls = (np.zeros(len(times)) if roll_degrees is None
             else np.broadcast_to(np.atleast_1d(np.asarray(roll_degrees, dtype=np.float64)), times.shape))
    fields = [(toa_accumulated(latitude, longitude - float(roll), float(t), window) - mean) / std
              for t, roll in zip(times, rolls)]
    return torch.tensor(np.stack(fields), dtype=torch.float32)


# --------------------------------------------------------------------------------------------------
# Residual statistics and loss weights
# --------------------------------------------------------------------------------------------------

def tendency_std(dataset, samples: int = 64, floor: float = 1e-3) -> torch.Tensor:
    """
    Per-channel standard deviation of the six-hour *change*, in normalized units.

    A residual predicted in units of the state's spread is a number near 0.01 for most channels, and
    its gradient then depends on how volatile a variable happens to be rather than on how wrong the
    prediction is. GraphCast normalises residuals by the spread of the residual itself; this is that.

    Measured on the upper-air store, the ratio runs from about 0.05 (temperature, geopotential --
    slow, large-scale) to about 1 (vertical velocity, precipitation -- nearly uncorrelated from one
    step to the next), which is the spread of scales the loss would otherwise be blind to.

    Args:
        dataset: any dataset whose items carry ``analysis`` with at least two history frames.
        floor: minimum returned, so a channel that never changes (padding) cannot divide by zero.
    """
    stride = max(len(dataset) // max(samples, 1), 1)
    total = None
    squares = None
    count = 0
    for position in range(0, len(dataset), stride):
        frames = dataset[position]["analysis"]
        if frames.shape[0] < 2:
            raise ValueError("tendency_std needs at least two history frames per item")
        delta = (frames[1:] - frames[:-1]).reshape(-1, frames.shape[-1]).double()
        total = delta.sum(0) if total is None else total + delta.sum(0)
        squares = (delta**2).sum(0) if squares is None else squares + (delta**2).sum(0)
        count += delta.shape[0]
    mean = total / count
    variance = (squares / count - mean**2).clamp_min(0.0)
    return variance.sqrt().clamp_min(floor).float()


def channel_loss_weights(channels: list[str], input_channels: int | None = None,
                         surface_weights: dict | None = None) -> torch.Tensor:
    """
    Per-channel loss weights, following GraphCast's recipe.

    Pressure levels are weighted in proportion to their pressure, normalised to mean one across the
    levels. That tracks the mass of air each level represents: 1000 hPa holds twenty times the air of
    50 hPa, and an error there matters to twenty times as much weather. Surface fields take fixed
    weights -- 2 m temperature at 1.0, which is what people feel, and the rest lower.

    Without this, 78 upper-air channels outvote 11 surface ones by sheer count, and the model spends
    its capacity on the stratosphere.
    """
    defaults = {"2m_temperature": 1.0, "10m_u_component_of_wind": 0.1, "10m_v_component_of_wind": 0.1,
                "mean_sea_level_pressure": 0.1, "total_precipitation_6hr": 0.1,
                "surface_pressure": 0.1, "total_column_water_vapour": 0.1,
                "sea_surface_temperature": 0.1}
    defaults.update(surface_weights or {})

    levels = sorted({lv for lv in (_level(c) for c in channels) if lv is not None})
    level_weight = {}
    if levels:
        raw = np.array(levels, dtype=np.float64)
        normalised = raw / raw.mean()
        level_weight = dict(zip(levels, normalised))

    width = input_channels if input_channels is not None else len(channels)
    weights = torch.zeros(width)
    for index, name in enumerate(channels):
        level = _level(name)
        if level is not None:
            weights[index] = float(level_weight[level])
        else:
            weights[index] = defaults.get(_base(name), 0.1)
    return weights


# --------------------------------------------------------------------------------------------------
# Physical constraints
# --------------------------------------------------------------------------------------------------

def conserve_mass(delta: torch.Tensor, channel: int, area_weights: torch.Tensor) -> torch.Tensor:
    """
    Remove any change in the atmosphere's total mass from a predicted surface-pressure tendency.

    Total mass is proportional to the global area-weighted mean of surface pressure, and it does not
    change on a six-hour timescale -- sources and sinks of air are negligible. A network has no reason
    to know this, and left alone its prediction carries a small spurious change every step; over a
    twelve-step rollout that becomes a drift in every pressure field on the planet.

    Subtracting the weighted mean of the tendency is the exact projection onto zero net change. It
    costs one weighted mean, touches only this channel, and leaves the spatial pattern -- which is the
    actual forecast -- intact.

    Args:
        delta: ``(B, P, C)`` predicted tendencies.
        channel: the surface-pressure channel.
        area_weights: ``(P,)`` quadrature weights of the grid the tendency lives on.
    """
    weights = area_weights.to(delta.dtype).view(1, -1)
    column = delta[..., channel]
    net = (column * weights).sum(-1, keepdim=True) / weights.sum()
    corrected = delta.clone()
    corrected[..., channel] = column - net
    return corrected


# --------------------------------------------------------------------------------------------------
# Probabilistic training: CRPS over a small ensemble
# --------------------------------------------------------------------------------------------------

def fair_crps(ensemble: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None,
              weights: torch.Tensor | None = None) -> torch.Tensor:
    """
    The fair continuous ranked probability score of an ensemble against an observation.

    Why this and not a Gaussian likelihood: a model trained on squared error, or on a single Gaussian,
    learns to predict the *conditional mean*. The mean of all the ways a hurricane could go is a weak,
    smeared vortex -- which is precisely the documented failure of GraphCast and every other
    deterministic AI weather model on tropical-cyclone intensity: low-biased by about the size of their
    mean error, worse than climatology-and-persistence. CRPS scores the whole distribution, so an
    ensemble trained on it keeps sharp individual members and puts the uncertainty in the spread
    between them. That is the approach behind ECMWF's AIFS-CRPS.

    The *fair* version subtracts the ensemble's own spread with an ``M(M-1)`` rather than ``M^2``
    denominator, which makes it unbiased for a finite ensemble: a two-member ensemble drawn from the
    true distribution scores the same, in expectation, as an infinite one. Without that correction the
    score rewards under-dispersion, and small ensembles learn to be overconfident.

        CRPS_fair = mean_i |x_i - y|  -  sum_{i != j} |x_i - x_j| / (2 M (M - 1))

    Args:
        ensemble: ``(M, ...)`` members.
        target: ``(...)`` the observation.
        mask: ``(...)`` 1 where the target was observed.
        weights: broadcastable per-channel weights, e.g. :func:`channel_loss_weights`.
    """
    members = ensemble.shape[0]
    skill = (ensemble - target.unsqueeze(0)).abs().mean(0)
    if members > 1:
        # Summed pair by pair rather than as one (M, M, ...) tensor, which at four members over the
        # full grid is 1.3 GB of activations before autograd keeps its own copy.
        spread = torch.zeros_like(skill)
        for i in range(members):
            for j in range(i + 1, members):
                spread = spread + 2.0 * (ensemble[i] - ensemble[j]).abs()
        score = skill - spread / (2 * members * (members - 1))
    else:
        score = skill                     # one member: CRPS reduces to absolute error

    if weights is not None:
        score = score * weights
    if mask is not None:
        return (score * mask).sum() / mask.sum().clamp_min(1.0)
    return score.mean()


def spread_skill_ratio(ensemble: torch.Tensor, target: torch.Tensor) -> float:
    """
    Ensemble spread over ensemble-mean error. Near 1 is calibrated; below 1 is overconfident.

    The diagnostic that says whether an ensemble's uncertainty means anything. An overconfident
    ensemble -- spread much smaller than its error -- is worse than useless for a hurricane cone,
    because it draws a narrow cone around the wrong track.
    """
    members = ensemble.shape[0]
    if members < 2:
        return float("nan")
    mean = ensemble.mean(0)
    rmse = float(((mean - target) ** 2).mean().sqrt())
    spread = float(ensemble.var(0, unbiased=True).mean().sqrt())
    # The (M+1)/M factor corrects the finite-ensemble bias in comparing spread to mean error.
    return spread * np.sqrt((members + 1) / members) / max(rmse, 1e-12)


# --------------------------------------------------------------------------------------------------
# Advancing the full state, and training on it
# --------------------------------------------------------------------------------------------------

@dataclass
class StateStepper:
    """
    Everything needed to advance the whole atmosphere one step, and nothing more.

    The replacement for :func:`naturev1.rollout.reinject`, which wrote back five surface channels and
    froze the other eighty-four. Every channel now has a rule: prognostic channels take the model's
    residual, forcings are recomputed for the new valid time, static channels stay where they are.

    Args:
        roles: from :func:`channel_roles`.
        tendency: ``(input_channels,)`` from :func:`tendency_std`.
        normalizer: the dataset's normalizer, for recomputing forcings in the model's units.
        grid: the grid the state lives on -- rollout requires the output grid to be the input grid.
        mass_channel: the surface-pressure channel, for :func:`conserve_mass`. None disables it.
        step_hours: model step length.
    """

    roles: ChannelRoles
    tendency: torch.Tensor
    normalizer: object
    grid: object
    mass_channel: int | None = None
    step_hours: float = 6.0

    @classmethod
    def build(cls, dataset, input_channels: int, grid, tendency: torch.Tensor | None = None,
              step_hours: float = 6.0, samples: int = 64) -> StateStepper:
        """Derive everything from a dataset, so nothing has to be restated by hand."""
        roles = channel_roles(dataset.variables, input_channels)
        if tendency is None:
            tendency = tendency_std(dataset, samples=samples)
        mass = dataset.variables.index("surface_pressure") if "surface_pressure" in dataset.variables else None
        return cls(roles, tendency, dataset.normalizer, grid, mass, step_hours)

    @property
    def prognostic_mass(self) -> int | None:
        """Position of surface pressure within the prognostic subset, which is where tendencies live."""
        if self.mass_channel is None or self.mass_channel not in self.roles.prognostic:
            return None
        return self.roles.prognostic.index(self.mass_channel)

    def to_state_delta(self, current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """True residual for the prognostic channels, in units of each channel's tendency spread."""
        prognostic = self.roles.prognostic
        scale = self.tendency[prognostic].to(current.device, current.dtype)
        return (target[..., prognostic] - current[..., prognostic]) / scale

    def advance(self, history: torch.Tensor, delta: torch.Tensor, valid_time: torch.Tensor,
                conserve: bool = True, roll_degrees: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One step: residual into the prognostic channels, fresh forcings, static left alone.

        Args:
            history: ``(B, T, P, C)`` input window.
            delta: ``(B, P, num_prognostic)`` predicted residual, in tendency-spread units.
            valid_time: ``(B,)`` unix seconds of the current "now".
            roll_degrees: ``(B,)`` augmentation rotation of each sample, from the batch.

        Returns:
            ``(history, valid_time)`` shifted one step forward.
        """
        if delta.shape[1] != history.shape[2]:
            raise ValueError(
                f"residual is on {delta.shape[1]} points and the state on {history.shape[2]}: a rollout "
                "needs output_grid to be the input grid, since the forecast is fed straight back in."
            )
        prognostic = self.roles.prognostic
        latest = history[:, -1].clone()

        scale = self.tendency[prognostic].to(delta.device, delta.dtype)
        change = delta * scale
        if conserve and self.prognostic_mass is not None:
            change = conserve_mass(change, self.prognostic_mass, self.grid.weights.to(change.device))
        latest[..., prognostic] = latest[..., prognostic] + change

        next_time = valid_time.to(torch.float64) + self.step_hours * 3600.0
        for index in self.roles.forcing:
            name = self.roles.channels[index]
            rolls = None if roll_degrees is None else roll_degrees.detach().cpu().numpy()
            latest[..., index] = solar_forcing(self.grid, next_time.cpu().numpy(), name,
                                               self.normalizer, rolls).to(latest.device, latest.dtype)
        return torch.cat([history[:, 1:], latest.unsqueeze(1)], dim=1), next_time


def _autocast(device, precision: str):
    """
    The mixed-precision context for a forward pass: a fresh one per call.

    ``"auto"`` is bf16 on CUDA and full precision elsewhere. An explicit ``"bf16"`` is honoured on any
    device -- which is what lets the GPU's mixed-precision path be exercised on a CPU. ``"fp16"`` is
    CUDA-only. Anything else is a plain no-op context rather than ``autocast(enabled=False)``, which
    would switch off an autocast the caller had already entered.
    """
    import contextlib

    device = torch.device(device)
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" else "fp32"
    if precision == "bf16":
        return torch.autocast(device.type, dtype=torch.bfloat16)
    if precision == "fp16" and device.type == "cuda":
        return torch.autocast(device.type, dtype=torch.float16)
    return contextlib.nullcontext()


def _advance_calendar(calendar: torch.Tensor, step_hours: float) -> torch.Tensor:
    from .rollout import advance_calendar

    shifted = advance_calendar(calendar[:, -1:], step_hours)
    return torch.cat([calendar[:, 1:], shifted], dim=1)


def state_rollout_loss(
    model,
    batch: dict,
    stepper: StateStepper,
    horizon: int,
    schedule=None,
    channel_weights: torch.Tensor | None = None,
    members: int = 1,
    accumulate: bool = True,
    rollout_member: int = 0,
    precision: str = "fp32",
    scaler=None,
) -> tuple[torch.Tensor, dict]:
    """
    Train on the whole atmosphere, rolled forward on the model's own forecasts.

    With ``members == 1`` each step is scored by a Gaussian likelihood on the residual. With
    ``members > 1`` each step draws that many noise samples, runs them, and is scored by
    :func:`fair_crps` -- which trains the ensemble to be sharp *and* calibrated rather than smooth. The
    trajectory continues from a single member, not the ensemble mean: the mean of a sharp ensemble is
    exactly the blurred field the ensemble exists to avoid, and feeding it back would reintroduce the
    blur one step later.

    Args:
        batch: needs ``analysis``, ``calendar``, ``valid_time`` and ``state_target`` -- build the
            dataset with ``state_steps >= horizon``.
        horizon: rollout length in steps.
        schedule: a :class:`naturev1.RolloutSchedule`, used only for its per-step weights.
        channel_weights: ``(input_channels,)`` from :func:`channel_loss_weights`.
        accumulate: backward each step and free its graph. Required for long horizons -- twelve graphs
            at once is roughly 115 GB at batch 16 on the full grid -- and exactly equivalent, since the
            trajectory is detached between steps.
        precision: ``"bf16"``, ``"fp16"`` or ``"auto"`` runs the forward passes under autocast -- the
            same precision as stage one. The loss and the trajectory stay in float32.
        scaler: a ``torch.amp.GradScaler`` for fp16, whose small gradients underflow without one.

    Returns:
        ``(loss, parts)``. With ``accumulate`` the loss is already backpropagated and comes back detached.
    """
    if "state_target" not in batch:
        raise KeyError("batch has no state_target: build the dataset with state_steps >= horizon")
    history = batch["analysis"]
    calendar = batch["calendar"]
    valid_time = batch["valid_time"]
    roll = batch.get("roll_degrees")
    futures = batch["state_target"]                     # (B, S, P, C)

    horizon = max(1, min(horizon, futures.shape[1]))
    if schedule is not None:
        step_weights = schedule.weights(horizon, device=history.device)
    else:
        step_weights = torch.full((horizon,), 1.0 / horizon, device=history.device)

    prognostic = stepper.roles.prognostic

    # Weight every point by the area it stands for. WeatherBench scores are latitude-weighted, and so
    # is naturev1.wb2 -- so an unweighted training loss optimises a different objective from the one
    # the model is judged on. On an equiangular grid the difference is not small: a row near the pole
    # holds as many points as the equator in a sliver of the area, and a plain mean over-weights the
    # polar caps by roughly six to one. Normalised to mean one, so the loss scale is unchanged.
    area = stepper.grid.weights.to(history.device, history.dtype)
    area = (area / area.mean()).view(1, -1, 1)
    weights = area
    if channel_weights is not None:
        weights = area * channel_weights[prognostic].to(history.device, history.dtype).view(1, 1, -1)

    noise_dim = getattr(model.config, "noise_dim", 0)
    latent_points = model.latent_grid.num_points
    total = torch.zeros((), device=history.device)
    parts: dict[str, float] = {}

    for step in range(horizon):
        current = history[:, -1]
        truth = stepper.to_state_delta(current, futures[:, step])

        if members > 1 and noise_dim:
            outputs = []
            for _ in range(members):
                noise = torch.randn(history.shape[0], latent_points, noise_dim, device=history.device)
                with _autocast(history.device, precision):
                    delta = model(analysis=history, analysis_grid=stepper.grid, calendar=calendar,
                                  output_grid=stepper.grid, member_noise=noise)["state_delta"]
                outputs.append(delta.float())
            ensemble = torch.stack(outputs)
            step_loss = fair_crps(ensemble, truth, weights=weights)
            forward_delta = ensemble[rollout_member]
            parts[f"spread_{step + 1}"] = float(ensemble.detach().std(0).mean())
        else:
            with _autocast(history.device, precision):
                out = model(analysis=history, analysis_grid=stepper.grid, calendar=calendar,
                            output_grid=stepper.grid)
            mean, log_var = out["state_delta"].float(), out["state_delta_log_var"].float()
            nll = 0.5 * (log_var + (truth - mean) ** 2 / log_var.exp())
            if weights is not None:
                nll = nll * weights
            step_loss = nll.mean()
            forward_delta = mean

        weighted = step_weights[step] * step_loss
        if accumulate:
            (scaler.scale(weighted) if scaler is not None else weighted).backward()
            total = total + weighted.detach()
        else:
            total = total + weighted
        parts[f"state_{step + 1}"] = float(step_loss.detach())

        if step + 1 < horizon:
            history, valid_time = stepper.advance(history, forward_delta.detach(), valid_time,
                                                  roll_degrees=roll)
            calendar = _advance_calendar(calendar, stepper.step_hours)

    parts["state_horizon"] = float(horizon)
    parts["state_drift"] = parts[f"state_{horizon}"] - parts["state_1"]
    return total, parts


@torch.no_grad()
def state_forecast(model, batch: dict, stepper: StateStepper, steps: int = 20, members: int = 1,
                   conserve: bool = True, precision: str = "fp32") -> torch.Tensor:
    """
    Run the full atmosphere forward: every prognostic channel evolves, forcings follow the sun.

    Returns:
        ``(members, steps, B, P, C)`` normalized states. Decode a channel with the dataset's
        ``denormalise``. With ``members > 1`` the spread across the first axis is the forecast
        uncertainty -- the thing a hurricane cone should be drawn from.
    """
    was_training = model.training
    model.eval()
    noise_dim = getattr(model.config, "noise_dim", 0)
    trajectories = []

    for _ in range(max(members, 1)):
        history = batch["analysis"]
        calendar = batch["calendar"]
        valid_time = batch["valid_time"]
        roll = batch.get("roll_degrees")
        states = []
        for _ in range(steps):
            noise = None
            if members > 1 and noise_dim:
                noise = torch.randn(history.shape[0], model.latent_grid.num_points, noise_dim,
                                    device=history.device)
            with _autocast(history.device, precision):
                delta = model(analysis=history, analysis_grid=stepper.grid, calendar=calendar,
                              output_grid=stepper.grid, member_noise=noise)["state_delta"].float()
            history, valid_time = stepper.advance(history, delta, valid_time, conserve=conserve,
                                                  roll_degrees=roll)
            calendar = _advance_calendar(calendar, stepper.step_hours)
            states.append(history[:, -1])
        trajectories.append(torch.stack(states))

    if was_training:
        model.train()
    return torch.stack(trajectories)


# --------------------------------------------------------------------------------------------------
# The rollout stage as a resumable run, and its scorecard
# --------------------------------------------------------------------------------------------------

def train_state_rollout(
    model,
    loader,
    stepper: StateStepper,
    max_steps: int,
    schedule=None,
    channel_weights: torch.Tensor | None = None,
    members: int = 1,
    learning_rate: float = 1e-4,
    warmup_steps: int = 200,
    weight_decay: float = 0.05,
    grad_clip: float = 1.0,
    precision: str = "auto",
    checkpoint_dir: str | None = None,
    hub_repo: str | None = None,
    checkpoint_seconds: float = 60.0,
    log_every: int = 25,
    device=None,
):
    """
    Full-state rollout training as a proper run: many epochs, a schedule, checkpoints, resume.

    The loop this replaces was a single ``for batch in loader`` -- one pass over the data. At five
    staged years and batch 8 that is about 900 steps, and the rollout curriculum does not *start*
    until step 2000, so the "twelve-step rollout" never rolled out: every step was a one-step step.
    It also saved nothing, so a Colab disconnect three hours in lost all three hours.

    Args:
        max_steps: optimizer steps in total, across restarts. The loader is cycled until they are done.
        schedule: a :class:`naturev1.RolloutSchedule`. Defaults to one sized to ``max_steps``: single
            steps for the first 10%, ramping to twelve by 60%, twelve for the rest.
        checkpoint_dir: where to save every ``checkpoint_seconds`` and resume from. Its last path
            component names the run on the Hub, so point it at e.g. ``.../rollout``.

    Returns:
        The :class:`naturev1.TrainingState` it finished in.
    """
    from .checkpoint import CheckpointManager, TrainingState
    from .guards import TrainingWatchdog
    from .rollout import RolloutSchedule
    from .train import TrainSettings, build_scheduler, move_batch

    device = torch.device(device or next(model.parameters()).device)
    if schedule is None:
        schedule = RolloutSchedule(start_step=max(max_steps // 10, 1), ramp_steps=max(max_steps // 2, 1),
                                   max_steps=12)
    full_at = schedule.start_step + schedule.ramp_steps
    if full_at > max_steps:
        print(f"[state] note: the schedule reaches {schedule.max_steps} steps at optimizer step "
              f"{full_at:,}, after this run ends at {max_steps:,}; the longest rollout trained will be "
              f"{schedule.horizon(max_steps - 1)} steps", flush=True)

    decay = [p for p in model.parameters() if p.requires_grad and p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=learning_rate, betas=(0.9, 0.95))
    # Warmup is capped at a tenth of the run, so a short run is not spent entirely warming up.
    warmup_steps = min(warmup_steps, max(max_steps // 10, 1))
    scheduler = build_scheduler(optimizer, TrainSettings(warmup_steps=warmup_steps, max_steps=max_steps))

    manager = state = None
    if checkpoint_dir is not None:
        manager = CheckpointManager(checkpoint_dir, every_seconds=checkpoint_seconds, repo_id=hub_repo)
        manager.fetch_from_hub()
        state = manager.load(model, optimizer, scheduler, map_location=str(device))
    state = state or TrainingState()
    if state.step >= max_steps:
        print(f"[state] already trained to step {state.step:,} of {max_steps:,}; nothing to do", flush=True)
        return state

    def save(force=False):
        if manager is not None:
            manager.save(model, optimizer, scheduler, None, state, config=model.config.to_dict(), force=force)

    watchdog = TrainingWatchdog(patience=200)
    params = [p for p in model.parameters() if p.requires_grad]
    # fp16 gradients underflow to zero without loss scaling; the scaler is a no-op for bf16 and fp32.
    scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16" and device.type == "cuda")
    model.train()
    started = time.time() - state.wall_seconds
    last_log = [time.time(), state.step]
    try:
        while state.step < max_steps:
            seen = 0
            for batch in loader:
                if state.step >= max_steps:
                    break
                seen += 1
                batch = move_batch(batch, device)
                horizon = schedule.horizon(state.step)
                loss, parts = state_rollout_loss(model, batch, stepper, horizon, schedule, channel_weights,
                                                 members=members, accumulate=True, precision=precision,
                                                 scaler=scaler)
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
                watchdog.observe(state.step, float(loss), float(norm))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                state.step += 1
                state.samples_seen += int(batch["analysis"].shape[0])
                state.wall_seconds = time.time() - started
                model.mark_trained({"state": float(loss)}, batch)
                if state.step % log_every == 0 or state.step == 1:
                    spread = parts.get("spread_1")
                    state.history.append({"step": state.step, **parts})
                    # Time per step grows with the horizon, so the estimate is for the current one.
                    now = time.time()
                    pace = (now - last_log[0]) / max(state.step - last_log[1], 1)
                    last_log[:] = [now, state.step]
                    left = pace * (max_steps - state.step) / 3600
                    print(f"[state] step {state.step:>7,}/{max_steps:,}  horizon {horizon:>2}  "
                          f"loss {float(loss):.4f}  drift {parts['state_drift']:+.4f}"
                          + (f"  spread {spread:.4f}" if spread is not None else "")
                          + f"  lr {scheduler.get_last_lr()[0]:.2e}  {pace:.1f} s/step  ~{left:.1f} h left",
                          flush=True)
                save()
            if seen == 0:
                raise ValueError("the loader produced no batches: the dataset is smaller than one batch "
                                 "(drop_last=True). Use a smaller batch or stage more data.")
            state.epoch += 1
    finally:
        save(force=True)
    print(f"[state] finished at step {state.step:,} after {state.wall_seconds / 60:.1f} min", flush=True)
    return state


@torch.no_grad()
def score_state(
    model,
    loader,
    stepper: StateStepper,
    fields: dict[str, str],
    steps: int = 12,
    climatology: torch.Tensor | None = None,
    max_batches: int = 16,
    members: int = 1,
    precision: str = "fp32",
    device=None,
):
    """
    Score the full-state rollout the way WeatherBench 2 does: Z500, T850 and friends, by lead.

    :func:`naturev1.score_model` scores the seven surface heads. The state head is what forecasts the
    upper air -- the thing medium-range skill actually lives in -- and until this function nothing
    measured it. Every number is area-weighted, in physical units, against persistence and climatology
    on the identical batches.

    Args:
        fields: label -> flat channel name, e.g. ``{"z500": "geopotential@500"}``. Labels missing from
            the dataset are skipped rather than failing.
        steps: rollout steps to score; lead ``k`` is ``k * step_hours``. The loader's dataset needs
            ``state_steps >= steps``.
        climatology: ``(points, channels)`` normalized mean state, from
            :func:`naturev1.build_climatology`. Enables the climatology baseline and ACC.

    Returns:
        A :class:`naturev1.Scorecard`.
    """
    from .train import move_batch
    from .wb2 import Score, Scorecard

    device = torch.device(device or next(model.parameters()).device)
    normalizer = stepper.normalizer
    wanted = {label: name for label, name in fields.items() if name in normalizer.variables}
    index = {label: stepper.roles.channels.index(name) for label, name in wanted.items()}
    area = stepper.grid.weights.to(device, torch.float64)
    area = (area / area.mean()).view(1, -1)
    totals: dict[tuple[str, int], dict[str, float]] = {}

    for number, batch in enumerate(loader):
        if number >= max_batches:
            break
        batch = move_batch(batch, device)
        if "state_target" not in batch:
            raise KeyError("score_state needs state_target: build the dataset with state_steps >= steps")
        horizon = min(steps, batch["state_target"].shape[1])
        forecast = state_forecast(model, batch, stepper, steps=horizon, members=members,
                                  precision=precision).mean(0)               # (steps, B, P, C)
        now = batch["analysis"][:, -1]
        for label, channel in index.items():
            name = wanted[label]

            def physical(values):
                return normalizer.decode(values.double(), name)

            persisted = physical(now[..., channel])
            reference = (physical(climatology[:, channel].to(device)).view(1, -1)
                         if climatology is not None else None)
            for step in range(horizon):
                predicted = physical(forecast[step, ..., channel])
                actual = physical(batch["state_target"][:, step, :, channel])
                bucket = totals.setdefault((label, int((step + 1) * stepper.step_hours)), dict.fromkeys(
                    ("model", "persist", "climo", "weight", "cov", "pp", "tt"), 0.0))
                weight = area.expand_as(actual)
                bucket["model"] += float(((predicted - actual) ** 2 * weight).sum())
                bucket["persist"] += float(((persisted - actual) ** 2 * weight).sum())
                bucket["weight"] += float(weight.sum())
                if reference is not None:
                    bucket["climo"] += float(((reference - actual) ** 2 * weight).sum())
                    a_pred, a_true = predicted - reference, actual - reference
                    bucket["cov"] += float((a_pred * a_true * weight).sum())
                    bucket["pp"] += float((a_pred**2 * weight).sum())
                    bucket["tt"] += float((a_true**2 * weight).sum())

    card = Scorecard()
    for (label, hours), bucket in sorted(totals.items()):
        count = max(bucket["weight"], 1e-12)
        acc = None
        if climatology is not None and bucket["pp"] > 0 and bucket["tt"] > 0:
            acc = bucket["cov"] / (bucket["pp"] * bucket["tt"]) ** 0.5
        card.add(Score(field=label, lead_hours=hours, rmse=(bucket["model"] / count) ** 0.5, acc=acc,
                       persistence_rmse=(bucket["persist"] / count) ** 0.5,
                       climatology_rmse=(bucket["climo"] / count) ** 0.5 if climatology is not None else None))
    return card


#: The WeatherBench 2 headline channels, by the flat names :func:`naturev1.expand_variables` produces.
HEADLINE_STATE_FIELDS = {
    "z500": "geopotential@500", "t850": "temperature@850", "t2m": "2m_temperature",
    "u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind", "msl": "mean_sea_level_pressure",
    "q700": "specific_humidity@700",
}
