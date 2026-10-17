# Copyright 2026 Nathan. Apache-2.0.
"""
Distillation alignment, multi-teacher targets and loss, data assimilation and the 0.25-degree decoder -- all
offline, on constructed fields with known answers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ------------------------------------------------------------------------------------------- align --

def test_conservative_regrid_preserves_area_means_and_handles_orientation():
    from naturev1.distill.align import NATURE_LAT, NATURE_LON, ConservativeRegrid

    lat = np.linspace(90, -90, 721)                       # descending, like many GRIB grids
    lon = np.arange(1440) * 0.25 - 180.0                  # -180..180 convention
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    field = 280.0 + 20.0 * np.cos(np.radians(lat_g)) + 5.0 * np.sin(np.radians(2 * lon_g))
    out = ConservativeRegrid(lat, lon)(field)
    assert out.shape == (121, 240)
    w_hi = np.cos(np.radians(lat_g))
    w_lo = np.cos(np.radians(np.meshgrid(NATURE_LAT, NATURE_LON, indexing="ij")[0]))
    assert (out * w_lo).sum() / w_lo.sum() == pytest.approx((field * w_hi).sum() / w_hi.sum(), rel=1e-3)
    j = int(np.argmin(np.abs(NATURE_LON - 45.0)))
    assert out[60, j] == pytest.approx(280 + 20 + 5 * np.sin(np.radians(90)), abs=0.1), "longitude convention handled"


def test_points_to_grid_from_a_reduced_gaussian_grid():
    from naturev1.distill.align import points_to_grid, reduced_gaussian_points

    lats = np.linspace(88.5, -88.5, 80)
    counts = [max(int(round(160 * np.cos(np.radians(p)))), 8) for p in lats]
    lat, lon, area = reduced_gaussian_points(counts, lats)
    values = 250.0 + 40.0 * np.cos(np.radians(lat))
    grid = points_to_grid(values, lat, lon, area)
    assert np.isfinite(grid).all()
    expected = 250.0 + 40.0 * np.cos(np.radians(np.linspace(-90, 90, 121)))
    assert np.abs(grid.mean(1) - expected)[5:-5].max() < 1.0


def test_vertical_and_time_alignment_never_extrapolate():
    from naturev1.distill.align import to_leads, to_levels

    levels = [1000, 850, 500, 250]
    data = np.array([290.0, 280.0, 250.0, 220.0])[:, None]
    out = to_levels(data, levels, [925, 700, 100])
    assert np.isfinite(out[0, 0]) and np.isfinite(out[1, 0]) and np.isnan(out[2, 0]), "100 hPa is outside the range"
    expected = 280.0 + (250.0 - 280.0) * (np.log(700) - np.log(850)) / (np.log(500) - np.log(850))
    assert out[1, 0] == pytest.approx(expected)
    series = np.array([1.0, 3.0, 5.0])[:, None]
    aligned, exact = to_leads(series, [12, 24, 36], [12, 18, 48])
    assert aligned[0, 0] == 1.0 and aligned[1, 0] == pytest.approx(2.0) and np.isnan(aligned[2, 0])
    assert exact.tolist() == [True, False, False]


def test_names_units_and_channels():
    from naturev1.distill.align import canonical_name, specific_from_relative, to_channels, to_si

    assert canonical_name("geopotential\t") == "geopotential", "Aurora's archive quirk"
    assert canonical_name("2t") == "2m_temperature" and canonical_name("prmsl") == "mean_sea_level_pressure"
    name, z = to_si("gh", np.array([5500.0]), "gpm")
    assert name == "geopotential" and z[0] == pytest.approx(5500 * 9.80665)
    assert to_si("msl", np.array([1000.0]), "hPa")[1][0] == 100000.0
    q = specific_from_relative(np.array([100.0]), np.array([300.0]), 1000.0)
    assert 0.02 < float(q[0]) < 0.025
    fields = {"temperature": {500: np.full((2, 3, 4), 250.0)}, "2m_temperature": np.full((2, 3, 4), 290.0)}
    values, provided = to_channels(fields, ["2m_temperature", "temperature@500", "temperature@850", "geopotential@500"])
    assert provided.tolist() == [True, True, False, False] and np.isnan(values[..., 2]).all()


def test_teacher_mixture_mean_variance_and_coverage():
    from naturev1.distill.targets import combine, skill_weights

    a = np.zeros((1, 2, 3))
    b = np.full((1, 2, 3), 2.0)
    b[..., 2] = np.nan                                          # teacher b does not provide channel 2
    mean, var, covered = combine({"a": a, "b": b}, {"b": np.full((1, 2, 3), 0.5)})
    assert mean[0, 0, 0] == pytest.approx(1.0) and var[0, 0, 0] == pytest.approx(1.0 + 0.25)
    assert mean[0, 0, 2] == 0.0 and var[0, 0, 2] == 0.0 and covered.all()
    w = skill_weights({"a": np.array([[1.0, np.nan]]), "b": np.array([[3.0, 2.0]])})
    assert w["a"][0, 0] == pytest.approx(0.75) and w["b"][0, 1] == pytest.approx(1.0)
    weighted, _, _ = combine({"a": a, "b": b}, None, {"a": np.full((1, 3), 0.75), "b": np.full((1, 3), 0.25)})
    assert weighted[0, 0, 0] == pytest.approx(0.5)


def test_distillation_curriculum_and_storm_focus():
    from naturev1.distill.train import horizon_for, storm_weight_map

    assert horizon_for(0, 1000, 8) == 1 and horizon_for(1000, 1000, 8) == 8
    assert all(horizon_for(s, 1000, 8) <= horizon_for(s + 50, 1000, 8) for s in range(0, 950, 50))
    lat, lon = np.meshgrid(np.linspace(-90, 90, 121), np.arange(240) * 1.5, indexing="ij")
    w = storm_weight_map(lat.ravel(), lon.ravel(), [(25.0, -75.0)], 800.0, 3.0)
    assert w.mean() == pytest.approx(1.0) and w.max() / w.min() == pytest.approx(4.0)


def test_distill_rollout_loss_prefers_the_teacher():
    from naturev1.distill.train import distill_rollout_loss

    class Stepper:
        def __init__(self):
            from types import SimpleNamespace

            self.roles = SimpleNamespace(prognostic=[0, 1])
            self.tendency = torch.ones(3)
            self.grid = SimpleNamespace(weights=torch.ones(5))
            self.step_hours = 6.0

        def to_state_delta(self, current, target):
            return target[..., [0, 1]] - current[..., [0, 1]]

        def advance(self, history, delta, valid_time):
            latest = history[:, -1].clone()
            latest[..., [0, 1]] += delta
            return torch.cat([history[:, 1:], latest.unsqueeze(1)], 1), valid_time + 21600

    class Model:
        def __init__(self, delta):
            self.delta = delta

        def __call__(self, **kw):
            b = kw["analysis"].shape[0]
            return {"state_delta": self.delta.expand(b, 5, 2), "state_delta_log_var": torch.zeros(b, 5, 2)}

    batch = {"analysis": torch.zeros(1, 2, 5, 3), "calendar": torch.zeros(1, 2, 6), "valid_time": torch.zeros(1),
             "teacher_mean": torch.ones(1, 2, 5, 2), "teacher_var": torch.zeros(1, 2, 5, 2),
             "teacher_covered": torch.ones(1, 2, 5, 2, dtype=torch.bool), "state_target": torch.zeros(1, 2, 5, 3),
             "truth_ok": torch.zeros(1, 2, dtype=torch.bool), "storm_weight": torch.ones(1, 5)}
    good, _ = distill_rollout_loss(Model(torch.tensor([1.0, 1.0])), batch, Stepper(), 1, alpha=0.0, precision="fp32")
    bad, _ = distill_rollout_loss(Model(torch.tensor([0.0, 0.0])), batch, Stepper(), 1, alpha=0.0, precision="fp32")
    assert float(good) < float(bad)


# ------------------------------------------------------------------------------------ assimilation --

def test_assimilation_corrects_bias_rejects_bad_data_and_improves_the_state():
    from naturev1.assimilation import Assimilator, Observations, interpolate_background

    lat, lon = np.linspace(-90, 90, 121), np.arange(240) * 1.5
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    truth = 280 + 20 * np.cos(np.radians(la)) + 3 * np.sin(np.radians(3 * lo))
    error = 1.5 * np.sin(np.radians(2 * lo + 30)) * np.cos(np.radians(2 * la))
    background, sigma = (truth + error).ravel(), np.full(truth.size, 1.5)
    rng = np.random.default_rng(0)
    olat, olon = rng.uniform(-50, 50, 1500), rng.uniform(0, 360, 1500)
    obs = interpolate_background(lat, lon, truth.ravel(), olat, olon) + 0.5 * rng.standard_normal(1500) + 0.8
    obs[:15] += 30.0
    da = Assimilator(["temperature@850"], lat, lon)
    for _ in range(2):
        out, _, report = da.cycle(background[:, None], sigma[:, None],
                                  {"temperature@850": [("other_centre", Observations(olat, olon, obs, 0.5))]})
    counts = report["other_centre|temperature@850"]
    assert counts["background"] + counts["gross"] + counts["buddy"] >= 15
    band = (np.abs(la) < 45).ravel()
    rmse = lambda x: np.sqrt(np.mean((x - truth.ravel())[band] ** 2))
    assert rmse(out[:, 0]) < 0.75 * rmse(background)
    assert np.median(da.bias.bias["other_centre|temperature@850"][3:9]) == pytest.approx(0.8, abs=0.15)


def test_superobs_reduce_count_and_keep_means():
    from naturev1.assimilation import Observations, superobs

    rng = np.random.default_rng(0)
    obs = Observations(rng.uniform(10, 11, 400), rng.uniform(20, 21, 400), np.full(400, 5.0), 1.0)
    s = superobs(obs, 1.5)
    assert len(s) < 10 and np.allclose(s.value, 5.0) and (s.sigma < 1.0).all()


# --------------------------------------------------------------------------------------- high-res --

def test_highres_head_starts_as_the_upsampled_forecast_and_upsampling_is_exact_for_smooth_fields():
    from naturev1.highres import HighResHead, bilinear_upsample, highres_coords

    head = HighResHead(hidden=8, fields=2)
    base = torch.randn(1, 10, 2)
    mean, log_var = head(torch.randn(1, 10, 8), torch.randn(1, 10, 3), base)
    assert torch.equal(mean, base) and torch.equal(log_var, torch.zeros_like(log_var))
    lat_lr, lon_lr = np.linspace(-90, 90, 121), np.arange(240) * 1.5
    lat_hr, lon_hr = highres_coords(0.25)
    field = torch.tensor(np.meshgrid(lat_lr, lon_lr, indexing="ij")[0], dtype=torch.float64)[None, :, :, None]
    up = bilinear_upsample(field, lat_lr, lon_lr, lat_hr, lon_hr)
    assert up.shape == (1, 721, 1440, 1)
    assert torch.allclose(up[0, :, 0, 0], torch.tensor(lat_hr), atol=1e-6), "linear in latitude is reproduced exactly"
