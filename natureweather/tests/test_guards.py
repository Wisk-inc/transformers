# Copyright 2026 Nathan. Apache-2.0.
"""
Every guard here is named after a bug that already shipped in this codebase and looked healthy.

So each test reproduces the original failure and asserts the guard catches it -- and, just as
importantly, that the guard stays quiet on correct data. A check that fires on everything is worse than
no check, because people learn to ignore it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from naturev1 import (
    NatureConfig,
    NatureV1,
    PreflightError,
    TrainingWatchdog,
    check_device_agreement,
    check_forecast,
    check_grid_alignment,
    check_normalization,
    check_weights,
    preflight,
    responds_to_input,
)
from naturev1.era5 import equiangular_weights

from ihelix import fibonacci_sphere


# --------------------------------------------------------------------------------------------------
# The transposition bug: right point count, sane values, wrong continent
# --------------------------------------------------------------------------------------------------

def latlon_grid_and_temperature(transposed: bool):
    """A synthetic globe whose temperature is a clean function of latitude alone."""
    from naturev1.corpora import latlon_to_coords

    from ihelix import FieldGrid, Geometry

    lat = np.linspace(90, -90, 37)
    lon = np.linspace(0, 360, 72, endpoint=False)
    coords = latlon_to_coords(lat, lon)
    grid = FieldGrid.source(coords, Geometry.globe(),
                            weights=torch.tensor(equiangular_weights(lat, len(lon))))

    lat_mesh, _ = np.meshgrid(lat, lon, indexing="ij")
    field = 300.0 - 80.0 * np.sin(np.radians(np.abs(lat_mesh)))
    if transposed:
        field = field.T                       # the exact bug: longitude-major read, latitude-major mesh
    return grid, torch.tensor(field.ravel(), dtype=torch.float32)


def test_alignment_check_is_quiet_on_a_correct_read():
    grid, temperature = latlon_grid_and_temperature(transposed=False)
    assert check_grid_alignment(grid, temperature) == []


def test_alignment_check_catches_a_transposed_read():
    """
    The bug that passed every shape check: same point count, same values, every sample misplaced.

    Nothing but a physical probe finds this. Temperature against absolute latitude is the cheapest one.
    """
    grid, temperature = latlon_grid_and_temperature(transposed=True)
    problems = check_grid_alignment(grid, temperature)
    assert problems and "wrong coordinates" in problems[0]


# --------------------------------------------------------------------------------------------------
# Weights that delete the poles
# --------------------------------------------------------------------------------------------------

class _FakeGrid:
    def __init__(self, weights):
        self.weights = torch.as_tensor(weights, dtype=torch.float64)
        self.num_points = len(weights)


def test_weight_check_accepts_true_cell_area():
    latitudes = np.linspace(90, -90, 121)
    assert check_weights(_FakeGrid(equiangular_weights(latitudes, 240))) == []


def test_weight_check_catches_cosine_weights():
    """cos(latitude) gives the pole row 6.1e-17 -- zero in float32 -- and the Arctic silently vanishes."""
    latitudes = np.linspace(90, -90, 121)
    cosine = np.cos(np.radians(latitudes)).repeat(240)
    problems = check_weights(_FakeGrid(cosine / cosine.sum()))
    assert problems
    assert any("numerically absent" in p or "zero or negative" in p for p in problems)


def test_weight_check_catches_unnormalised_weights():
    assert any("sum to" in p for p in check_weights(_FakeGrid(np.full(100, 5.0))))


# --------------------------------------------------------------------------------------------------
# A channel that is a spike, not a distribution
# --------------------------------------------------------------------------------------------------

def test_normalization_check_accepts_a_normal_channel():
    assert check_normalization(torch.randn(10_000)) == []


def test_normalization_check_catches_a_raw_z_scored_precipitation_channel():
    """15% exact zeros and a 30-sigma maximum: the loss-optimal answer is zero rain, forever."""
    values = torch.zeros(10_000)
    values[:200] = torch.linspace(5, 30, 200)
    problems = check_normalization(values, "precip")
    assert problems and "standard deviations" in problems[0]


def test_normalization_check_catches_a_dead_channel():
    assert any("nearly constant" in p for p in check_normalization(torch.full((1000,), 0.5)))


def test_normalization_check_reports_nan_share():
    values = torch.randn(1000)
    values[:100] = float("nan")
    assert any("NaN" in p for p in check_normalization(values))


# --------------------------------------------------------------------------------------------------
# Device placement
# --------------------------------------------------------------------------------------------------

def test_device_check_catches_a_grid_left_behind():
    """The Colab crash: weights on cuda, grid on cpu, error raised four frames away in a Linear."""
    config = NatureConfig(latent_points=64, num_layers=2, hidden_size=64)
    model = NatureV1(config, fibonacci_sphere(64, num_neighbours=8, cluster_size=16)).to("meta")
    stray = fibonacci_sphere(32, num_neighbours=8, cluster_size=16)      # still on cpu

    problems = check_device_agreement(model, stray)
    assert problems and "cpu" in problems[0] and "meta" in problems[0]


def test_device_check_is_quiet_when_everything_agrees():
    config = NatureConfig(latent_points=64, num_layers=2, hidden_size=64)
    model = NatureV1(config, fibonacci_sphere(64, num_neighbours=8, cluster_size=16))
    assert check_device_agreement(model, model.latent_grid) == []


# --------------------------------------------------------------------------------------------------
# Forecasts that are not physically possible
# --------------------------------------------------------------------------------------------------

def good_forecast():
    return {
        "eyewall": [{"lead_hours": 24, "peak_wind_kt": 95.0, "rmw_nmi": 22.0,
                     "wind_radii_nmi": {"34kt": {"NE": 120.0, "SE": 100.0, "SW": 80.0, "NW": 90.0}}}],
        "intensity": [{"lead_hours": 24, "max_wind_ms": 49.0, "min_pressure_hpa": 950.0}],
        "rapid_intensification": {"probability_30kt": 0.31, "expected_change_kt": 18.0},
        "track_scenarios": [{"probability": 0.6, "track": [{"lead_hours": 24, "latitude": 25.0,
                                                            "longitude": -80.0, "cone_radius_km_95": 180.0}]},
                            {"probability": 0.4, "track": [{"lead_hours": 24, "latitude": 27.0,
                                                            "longitude": -78.0, "cone_radius_km_95": 200.0}]}],
        "enso": {"nino34_index": -0.8},
    }


def test_a_plausible_forecast_passes():
    assert check_forecast(good_forecast()) == []


@pytest.mark.parametrize("mutate,expect", [
    (lambda f: f["eyewall"][0].update(peak_wind_kt=420.0), "peak wind"),
    (lambda f: f["eyewall"][0].update(rmw_nmi=-5.0), "RMW"),
    (lambda f: f["intensity"][0].update(min_pressure_hpa=300.0), "pressure"),
    (lambda f: f["rapid_intensification"].update(probability_30kt=1.7), "probability"),
    (lambda f: f["rapid_intensification"].update(expected_change_kt=900.0), "24h change"),
    (lambda f: f["track_scenarios"][0].update(probability=0.99), "sum to"),
    (lambda f: f["track_scenarios"][0]["track"][0].update(latitude=141.0), "off the planet"),
    (lambda f: f["track_scenarios"][0]["track"][0].update(cone_radius_km_95=20_000.0), "wider than an ocean"),
    (lambda f: f["enso"].update(nino34_index=-9.0), "Nino"),
])
def test_impossible_forecasts_are_caught(mutate, expect):
    """
    Not because the model is likely to predict a 420 kt hurricane, but because if it ever does that
    must surface as an error rather than as a headline.
    """
    forecast = good_forecast()
    mutate(forecast)
    problems = check_forecast(forecast)
    assert problems, f"expected a complaint mentioning {expect!r}"
    assert any(expect in p for p in problems), f"{expect!r} not in {problems}"


def test_nan_in_a_forecast_is_caught():
    forecast = good_forecast()
    forecast["eyewall"][0]["peak_wind_kt"] = float("nan")
    assert any("NaN" in p for p in check_forecast(forecast))


# --------------------------------------------------------------------------------------------------
# The watchdog
# --------------------------------------------------------------------------------------------------

def test_watchdog_raises_on_a_non_finite_loss():
    """One NaN poisons every parameter, and the run keeps printing step numbers forever."""
    watchdog = TrainingWatchdog()
    watchdog.observe(1, 2.0)
    with pytest.raises(PreflightError, match="poisons"):
        watchdog.observe(2, float("nan"))


def test_watchdog_warns_on_divergence():
    watchdog = TrainingWatchdog(patience=5, divergence_factor=2.0)
    watchdog.observe(0, 1.0)
    for step in range(1, 12):
        watchdog.observe(step, 10.0)
    assert any("divergence" in w for w in watchdog.warnings)


def test_watchdog_warns_when_gradients_collapse():
    """A flat loss with no gradient is a model that stopped learning, not one that converged."""
    watchdog = TrainingWatchdog()
    watchdog.observe(1, 1.0, grad_norm=1e-12)
    assert any("stopped learning" in w for w in watchdog.warnings)


def test_watchdog_is_quiet_on_a_healthy_run():
    watchdog = TrainingWatchdog(patience=5)
    for step, loss in enumerate([2.0, 1.8, 1.7, 1.5, 1.45, 1.4, 1.42, 1.35]):
        watchdog.observe(step, loss, grad_norm=0.8)
    assert watchdog.warnings == []


# --------------------------------------------------------------------------------------------------
# Preflight as a whole
# --------------------------------------------------------------------------------------------------

def test_preflight_catches_a_fully_frozen_model():
    config = NatureConfig(latent_points=64, num_layers=2, hidden_size=64)
    model = NatureV1(config, fibonacci_sphere(64, num_neighbours=8, cluster_size=16))
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with pytest.raises(PreflightError, match="no trainable parameters"):
        preflight(model=model, strict=True)


def test_preflight_can_report_instead_of_raising():
    config = NatureConfig(latent_points=64, num_layers=2, hidden_size=64)
    model = NatureV1(config, fibonacci_sphere(64, num_neighbours=8, cluster_size=16))
    report = preflight(model=model, grid=model.latent_grid, strict=False)
    assert "problems" in report and "notes" in report
    assert any("parameters" in note for note in report["notes"])


def test_an_untrained_model_does_not_respond_to_the_weather():
    """
    The check that stops an untrained model being mistaken for a forecaster.

    Its heads are anchored at Atlantic climatology, so it answers 65 kt whatever it is shown -- which
    reads as a confident forecast and is a constant.
    """
    config = NatureConfig(latent_points=64, num_layers=2, hidden_size=64, history_frames=1)
    grid = fibonacci_sphere(48, num_neighbours=8, cluster_size=16)
    model = NatureV1(config, fibonacci_sphere(64, num_neighbours=8, cluster_size=16))

    # Freeze the encoder so the summary cannot vary: the pathological case, exactly.
    for parameter in model.parameters():
        parameter.data.zero_()

    def make_batch():
        return {"analysis": torch.randn(1, 1, grid.num_points, config.analysis_channels),
                "calendar": torch.zeros(1, 1, 6)}

    responds, spread = responds_to_input(model, make_batch, grid)
    assert not responds and spread < 1e-3
