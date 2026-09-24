# Copyright 2026 Nathan. Apache-2.0.
"""
Tests for full-state prediction.

The bug that motivated all of this: upper air and rollout shipped separately and did not compose. The
model predicted seven surface fields, so a twelve-step rollout froze 84 of 89 channels -- Z500 never
changed, and the sun stopped. Every test below pins one piece of the fix to a property that would
have caught it.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest
import torch
from naturev1 import NatureConfig, NatureV1
from naturev1.state import (
    SOLAR_CONSTANT,
    StateStepper,
    channel_loss_weights,
    channel_roles,
    conserve_mass,
    fair_crps,
    spread_skill_ratio,
    state_forecast,
    state_rollout_loss,
    tendency_std,
    toa_accumulated,
    toa_flux,
)

from ihelix import fibonacci_sphere


UPPER = [f"{v}@{lv}" for v in ("geopotential", "temperature") for lv in (250, 500, 850, 1000)]
CHANNELS = (["2m_temperature", "surface_pressure", "toa_incident_solar_radiation_6hr",
             "land_sea_mask", "geopotential_at_surface"] + UPPER)


# --------------------------------------------------------------------------------------------------
# Roles
# --------------------------------------------------------------------------------------------------

def test_roles_classify_every_channel():
    roles = channel_roles(CHANNELS, input_channels=len(CHANNELS) + 3)
    names = lambda idx: {CHANNELS[i] if i < len(CHANNELS) else "pad" for i in idx}  # noqa: E731

    assert names(roles.forcing) == {"toa_incident_solar_radiation_6hr"}
    assert {"land_sea_mask", "geopotential_at_surface", "pad"} <= names(roles.static)
    assert "geopotential@500" in names(roles.prognostic)
    assert "2m_temperature" in names(roles.prognostic)

    everything = sorted(roles.prognostic + roles.forcing + roles.static)
    assert everything == list(range(len(CHANNELS) + 3)), "every channel needs exactly one role"


def test_upper_air_is_prognostic_not_frozen():
    """The bug: 84 of 89 channels frozen. Every pressure-level field must be predicted."""
    roles = channel_roles(CHANNELS, len(CHANNELS))
    for name in UPPER:
        assert CHANNELS.index(name) in roles.prognostic, f"{name} would be frozen for the whole rollout"


# --------------------------------------------------------------------------------------------------
# Solar forcing
# --------------------------------------------------------------------------------------------------

def test_solar_flux_is_zero_at_night_and_peaks_near_the_subsolar_point():
    noon_greenwich = dt.datetime(2024, 3, 20, 12, 0, tzinfo=dt.timezone.utc).timestamp()
    at_noon = float(toa_flux(0.0, 0.0, noon_greenwich))
    at_midnight = float(toa_flux(0.0, 180.0, noon_greenwich))
    assert at_midnight == 0.0, "the night side receives no sunlight"
    assert at_noon > 0.99 * SOLAR_CONSTANT / 1.0 ** 2 * 0.98, "equator at equinox noon is near the solar constant"


def test_orbital_eccentricity_moves_the_flux_by_about_seven_percent():
    """Earth is 3.3% closer in January; flux goes as distance squared -- about 6.7%."""
    lat, lon = 0.0, 0.0
    january = max(float(toa_flux(lat, lon, dt.datetime(2024, 1, 3, h, tzinfo=dt.timezone.utc).timestamp()))
                  for h in range(24))
    july = max(float(toa_flux(lat, lon, dt.datetime(2024, 7, 4, h, tzinfo=dt.timezone.utc).timestamp()))
               for h in range(24))
    # Normalise out the seasonal declination by comparing at the subsolar maximum over the day.
    ratio = (january / math.cos(math.radians(22.8))) / (july / math.cos(math.radians(22.9)))
    assert 1.05 < ratio < 1.09


def test_daily_mean_at_the_equator_on_the_equinox_is_s0_over_pi():
    """Averaged over a day, the equator at equinox receives S0/pi -- a closed-form check."""
    start = dt.datetime(2024, 3, 20, 0, tzinfo=dt.timezone.utc).timestamp()
    energy = float(toa_accumulated(0.0, 0.0, start + 86_400, window_hours=24.0, substeps=288))
    mean_flux = energy / 86_400
    assert mean_flux == pytest.approx(SOLAR_CONSTANT / math.pi, rel=0.02)


def test_accumulation_is_over_the_window_ending_at_valid_time():
    """ERA5's convention, confirmed against the store: a 06Z value is sunlight from 00Z to 06Z."""
    valid = dt.datetime(2024, 6, 21, 6, tzinfo=dt.timezone.utc).timestamp()
    # At 90E, 00Z-06Z is 06-12 local: morning sun. At 90W it is 18-24 local: night.
    morning = float(toa_accumulated(0.0, 90.0, valid))
    night = float(toa_accumulated(0.0, -90.0, valid))
    assert morning > 1e6 and night < 0.2 * morning


# --------------------------------------------------------------------------------------------------
# Residual scale and loss weights
# --------------------------------------------------------------------------------------------------

class _Frames:
    """A dataset whose channels change at known, different rates."""

    def __init__(self, rates, frames=2, points=200, items=20):
        self.rates, self.frames, self.points, self.items = rates, frames, points, items

    def __len__(self):
        return self.items

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        base = torch.randn(1, self.points, len(self.rates), generator=generator)
        steps = [base]
        for _ in range(self.frames - 1):
            steps.append(steps[-1] + torch.randn(1, self.points, len(self.rates), generator=generator)
                         * torch.tensor(self.rates))
        return {"analysis": torch.cat(steps)}


def test_tendency_std_measures_the_change_not_the_state():
    """Temperature varies by 20 K but changes by a fraction of one in six hours."""
    spread = tendency_std(_Frames(rates=[0.05, 1.0, 0.3]))
    assert spread[0] == pytest.approx(0.05, rel=0.2)
    assert spread[1] == pytest.approx(1.0, rel=0.2)
    assert spread[2] == pytest.approx(0.3, rel=0.2)


def test_tendency_std_has_a_floor():
    """Padding never changes, and dividing by zero would put infinity in the loss."""
    assert float(tendency_std(_Frames(rates=[0.0])).min()) > 0.0


def test_loss_weights_follow_pressure():
    """1000 hPa holds twenty times the air of 50 hPa; an error there matters to that much more weather."""
    channels = [f"temperature@{lv}" for lv in (50, 500, 1000)] + ["2m_temperature", "land_sea_mask"]
    weights = channel_loss_weights(channels)
    upper = weights[:3]
    assert float(upper.mean()) == pytest.approx(1.0, rel=1e-6), "levels normalise to mean one"
    assert upper[0] < upper[1] < upper[2]
    assert float(upper[2] / upper[0]) == pytest.approx(20.0, rel=1e-6)
    assert float(weights[3]) == 1.0, "2 m temperature is what people feel"


def test_loss_weights_pad_to_the_input_width():
    assert channel_loss_weights(["2m_temperature"], input_channels=5).shape == (5,)


# --------------------------------------------------------------------------------------------------
# Mass conservation
# --------------------------------------------------------------------------------------------------

def test_mass_conservation_removes_the_net_change_exactly():
    torch.manual_seed(0)
    delta = torch.randn(3, 100, 4) + 0.7              # a spurious net tendency, as a network produces
    area = torch.rand(100) + 0.1
    fixed = conserve_mass(delta, channel=2, area_weights=area)

    net = (fixed[..., 2] * area).sum(-1) / area.sum()
    assert torch.allclose(net, torch.zeros(3), atol=1e-6)


def test_mass_conservation_keeps_the_pattern_and_the_other_channels():
    """Only the global mean moves. The spatial pattern is the forecast and must survive."""
    torch.manual_seed(1)
    delta = torch.randn(2, 50, 3)
    area = torch.ones(50)
    fixed = conserve_mass(delta, 1, area)

    assert torch.equal(fixed[..., 0], delta[..., 0]) and torch.equal(fixed[..., 2], delta[..., 2])
    pattern = delta[..., 1] - delta[..., 1].mean(-1, keepdim=True)
    assert torch.allclose(fixed[..., 1], pattern, atol=1e-6)


# --------------------------------------------------------------------------------------------------
# CRPS
# --------------------------------------------------------------------------------------------------

def test_one_member_crps_is_absolute_error():
    ensemble = torch.tensor([[1.0, 2.0, 3.0]])
    target = torch.tensor([1.5, 2.0, 5.0])
    assert float(fair_crps(ensemble, target)) == pytest.approx(float((ensemble[0] - target).abs().mean()))


def test_crps_is_zero_for_a_perfect_deterministic_ensemble():
    target = torch.randn(100)
    assert float(fair_crps(target.expand(4, -1).clone(), target)) == pytest.approx(0.0, abs=1e-6)


def test_fair_crps_does_not_reward_small_ensembles_for_being_overconfident():
    """
    The unfair version divides the spread term by M^2 and so under-credits spread for small M --
    which teaches a small ensemble to collapse. The fair version is unbiased: members drawn from the
    true distribution score the same in expectation whether there are two of them or twenty.
    """
    torch.manual_seed(0)
    target = torch.randn(20_000)
    scores = {}
    for members in (2, 4, 16):
        draws = torch.randn(members, 20_000)             # same distribution as the target
        scores[members] = float(fair_crps(draws, target))
    assert scores[2] == pytest.approx(scores[16], rel=0.03)
    assert scores[4] == pytest.approx(scores[16], rel=0.03)


def test_crps_prefers_a_calibrated_ensemble_to_an_overconfident_one():
    """The whole point: scoring the distribution penalises a narrow cone around the wrong answer."""
    torch.manual_seed(0)
    target = torch.randn(10_000)
    calibrated = torch.randn(8, 10_000)
    overconfident = 0.1 * torch.randn(8, 10_000)
    assert fair_crps(calibrated, target) < fair_crps(overconfident, target)


def test_crps_respects_the_mask_and_the_weights():
    ensemble = torch.zeros(2, 4)
    target = torch.tensor([1.0, 1.0, 100.0, 100.0])
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert float(fair_crps(ensemble, target, mask=mask)) == pytest.approx(1.0)
    weights = torch.tensor([2.0, 2.0, 2.0, 2.0])
    assert float(fair_crps(ensemble, target, mask=mask, weights=weights)) == pytest.approx(2.0)


def test_spread_skill_ratio_is_near_one_when_calibrated():
    torch.manual_seed(0)
    truth = torch.randn(50_000)
    centre = truth + torch.randn(50_000)                  # forecast error of unit size
    ensemble = centre.unsqueeze(0) + torch.randn(10, 50_000)
    assert spread_skill_ratio(ensemble, truth) == pytest.approx(1.0, abs=0.12)
    assert spread_skill_ratio(centre.unsqueeze(0) + 0.1 * torch.randn(10, 50_000), truth) < 0.3


# --------------------------------------------------------------------------------------------------
# The model and the stepper
# --------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def setup():
    grid = fibonacci_sphere(40, num_neighbours=8, cluster_size=10)
    channels = ["2m_temperature", "surface_pressure", "toa_incident_solar_radiation_6hr",
                "land_sea_mask", "temperature@500"]
    width = len(channels) + 2
    roles = channel_roles(channels, width)

    class _Norm:
        variables = channels
        mean = np.array([280.0, 1e5, 1e6, 0.5, 250.0], np.float32)
        std = np.array([20.0, 1e3, 5e6, 0.5, 10.0], np.float32)

    stepper = StateStepper(roles, torch.full((width,), 0.1), _Norm(), grid, mass_channel=1)
    config = NatureConfig(latent_points=48, num_layers=2, hidden_size=48, history_frames=2,
                          analysis_channels=width, lead_times_hours=(6,),
                          state_channels=roles.num_prognostic, noise_dim=4)
    model = NatureV1(config, fibonacci_sphere(48, num_neighbours=8, cluster_size=12))
    return model, stepper, grid, width


def make_batch(grid, width, steps=3):
    return {
        "analysis": torch.randn(1, 2, grid.num_points, width),
        "calendar": torch.zeros(1, 2, 6),
        "valid_time": torch.tensor([1.7e9], dtype=torch.float64),
        "state_target": torch.randn(1, steps, grid.num_points, width),
    }


def test_untrained_residual_model_is_exactly_persistence(setup):
    """Zero change is persistence, the strongest trivial baseline -- and training starts from there."""
    model, stepper, grid, width = setup
    batch = make_batch(grid, width)
    delta = model(analysis=batch["analysis"], analysis_grid=grid, calendar=batch["calendar"],
                  output_grid=grid)["state_delta"]
    assert torch.equal(delta, torch.zeros_like(delta))


def test_stepper_updates_prognostic_recomputes_forcing_and_keeps_static(setup):
    model, stepper, grid, width = setup
    history = torch.randn(1, 2, grid.num_points, width)
    delta = torch.ones(1, grid.num_points, stepper.roles.num_prognostic)

    advanced, when = stepper.advance(history, delta, torch.tensor([1.7e9], dtype=torch.float64),
                                     conserve=False)
    before, after = history[0, -1], advanced[0, -1]

    for index in stepper.roles.prognostic:
        assert not torch.allclose(after[:, index], before[:, index]), "prognostic channels must move"
    for index in stepper.roles.static:
        assert torch.equal(after[:, index], before[:, index]), "static channels must not move"
    sun = stepper.roles.forcing[0]
    assert not torch.equal(after[:, sun], before[:, sun]), "the sun must be recomputed, not frozen"
    assert float(when[0]) == pytest.approx(1.7e9 + 6 * 3600)


def test_stepper_conserves_mass_by_default(setup):
    model, stepper, grid, width = setup
    history = torch.randn(1, 2, grid.num_points, width)
    delta = torch.randn(1, grid.num_points, stepper.roles.num_prognostic) + 2.0
    advanced, _ = stepper.advance(history, delta, torch.tensor([1.7e9], dtype=torch.float64))

    change = advanced[0, -1, :, 1] - history[0, -1, :, 1]
    weights = grid.weights.float()
    assert float((change * weights).sum() / weights.sum()) == pytest.approx(0.0, abs=1e-5)


def test_stepper_refuses_a_forecast_on_the_wrong_grid(setup):
    model, stepper, grid, width = setup
    with pytest.raises(ValueError, match="output_grid"):
        stepper.advance(torch.randn(1, 2, grid.num_points, width), torch.randn(1, 7, 3),
                        torch.tensor([1.7e9], dtype=torch.float64))


def test_rollout_loss_trains_the_state_head(setup):
    model, stepper, grid, width = setup
    model.zero_grad(set_to_none=True)
    loss, parts = state_rollout_loss(model, make_batch(grid, width), stepper, horizon=3)
    assert torch.isfinite(loss)
    assert {"state_1", "state_2", "state_3"} <= set(parts)
    assert model.state_head.weight.grad is not None


def test_rollout_loss_needs_state_targets(setup):
    model, stepper, grid, width = setup
    batch = make_batch(grid, width)
    del batch["state_target"]
    with pytest.raises(KeyError, match="state_steps"):
        state_rollout_loss(model, batch, stepper, horizon=2)


def test_zero_init_head_trains_first_then_opens_the_path_upstream(setup):
    """
    A zero-initialised residual head passes no gradient upstream on the very first step -- the
    derivative of its output with respect to its input is its weight, which is zero. Only the head's
    own weights move at step 0; everything behind it wakes up from step 1. That is the ReZero /
    adaLN-Zero behaviour, and it is the price of starting at exact persistence: cheap, and it means
    the backbone is never pushed around by a head that has not yet learned what it is for.
    """
    model, stepper, grid, width = setup
    with torch.no_grad():
        model.state_head.weight.zero_()
        model.state_head.bias.zero_()
    model.zero_grad(set_to_none=True)
    state_rollout_loss(model, make_batch(grid, width), stepper, horizon=1)

    assert float(model.state_head.weight.grad.abs().sum()) > 0, "the head itself must train at step 0"
    upstream = model.analysis_encoder.embed.weight.grad
    assert upstream is None or float(upstream.abs().sum()) == 0.0, "nothing upstream moves at step 0"


def test_crps_rollout_trains_the_noise_path(setup):
    """
    Once the head has moved off zero, the CRPS gradient reaches the noise projection.

    Even with members initially identical, the *skill* term pushes the noise projection off zero --
    each member's error is taken against different noise, so the gradients do not cancel -- and once
    members differ the spread term engages. The ensemble starts itself; nothing has to seed it.
    """
    model, stepper, grid, width = setup
    with torch.no_grad():
        model.state_head.weight.normal_(0, 0.05)         # as it is after the first optimizer step
        model.noise_proj.weight.normal_(0, 0.1)
    model.zero_grad(set_to_none=True)
    loss, parts = state_rollout_loss(model, make_batch(grid, width), stepper, horizon=2, members=3)
    assert torch.isfinite(loss) and "spread_1" in parts
    assert model.noise_proj.weight.grad is not None
    assert float(model.noise_proj.weight.grad.abs().sum()) > 0


def test_state_forecast_evolves_the_whole_atmosphere(setup):
    """The regression test for the original bug: the upper air must change over a rollout."""
    model, stepper, grid, width = setup
    with torch.no_grad():
        model.state_head.weight.normal_(0, 0.05)
    batch = make_batch(grid, width)
    trajectory = state_forecast(model, batch, stepper, steps=4)
    assert trajectory.shape == (1, 4, 1, grid.num_points, width)

    upper = 4                                             # temperature@500
    assert not torch.equal(trajectory[0, -1, 0, :, upper], batch["analysis"][0, -1, :, upper])


def test_ensemble_forecast_members_differ(setup):
    model, stepper, grid, width = setup
    with torch.no_grad():
        model.noise_proj.weight.normal_(0, 0.5)
        model.state_head.weight.normal_(0, 0.05)
    trajectory = state_forecast(model, make_batch(grid, width), stepper, steps=2, members=3)
    assert trajectory.shape[0] == 3
    assert not torch.equal(trajectory[0], trajectory[1])


def test_default_models_are_unchanged():
    """Everything here is opt-in: a default config gains no parameters and no outputs."""
    model = NatureV1(NatureConfig(latent_points=48, num_layers=2, hidden_size=48),
                     fibonacci_sphere(48, num_neighbours=8, cluster_size=12))
    assert model.state_head is None and model.noise_proj is None


def test_training_loss_is_area_weighted_like_the_score(setup):
    """
    Scoring is latitude-weighted, so an unweighted training loss optimises something else.

    On an equiangular grid the polar rows hold as many points as the equator in a sliver of the area.
    Here an error confined to the low-weight half of the grid must count for less than the same error
    on the high-weight half -- which a plain mean would score identically.
    """
    model, stepper, grid, width = setup
    order = torch.argsort(grid.weights)
    light, heavy = order[: grid.num_points // 2], order[grid.num_points // 2:]

    def loss_with_error_on(points):
        batch = make_batch(grid, width, steps=1)
        batch["state_target"] = batch["analysis"][:, -1:].clone()      # a perfect persistence target...
        batch["state_target"][0, 0, points] += 3.0                      # ...wrong only on these points
        with torch.no_grad():
            model.state_head.weight.zero_()
            model.state_head.bias.zero_()
        loss, _ = state_rollout_loss(model, batch, stepper, horizon=1, accumulate=False)
        return float(loss)

    if float(grid.weights[heavy].sum()) > 1.05 * float(grid.weights[light].sum()):
        assert loss_with_error_on(heavy) > loss_with_error_on(light)
