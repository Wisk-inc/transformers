# Copyright 2026 Nathan. Apache-2.0.
"""
0.12 physics: each piece checked against what the physics says it must do, on constructed atmospheres and storms.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest
import torch
from ihelix import FieldGrid, Geometry, fibonacci_sphere

from naturev1.era5 import Normalizer
from naturev1.thermo import (
    STORM_PHYSICS_FEATURES, cape, entropy_deficit, genesis_map, genesis_potential_index, mixing_ratio,
    potential_intensity, storm_physics,
)
from naturev1.windfield import (
    fit_b, holland_profile, land_decay_kt, radius_of, rcliper_rate_mm_h, wind_field, wind_hazard,
)

LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
#: A mean hurricane-season tropical sounding (Jordan 1958-like): temperature C and specific humidity g/kg.
SOUNDING_T = [26.3, 21.8, 17.6, 9.5, 2.6, -5.6, -16.2, -31.4, -40.9, -53.4, -66.1, -73.5]
SOUNDING_Q = [18.0, 15.0, 12.0, 6.8, 4.3, 2.4, 1.1, 0.33, 0.13, 0.05, 0.02, 0.01]


def sounding(batch=1):
    t = (torch.tensor(SOUNDING_T, dtype=torch.float64) + 273.15).view(1, -1).repeat(batch, 1)
    q = (torch.tensor(SOUNDING_Q, dtype=torch.float64) / 1000.0).view(1, -1).repeat(batch, 1)
    return torch.tensor(LEVELS, dtype=torch.float64), t, q


# ---------------------------------------------------------------------------------------- thermo --

def test_potential_intensity_rises_with_the_ocean_and_vanishes_over_cold_water():
    levels, t, q = sounding(5)
    sst = torch.tensor([20.0, 26.0, 28.0, 29.0, 31.0], dtype=torch.float64) + 273.15
    pi = potential_intensity(sst, torch.full((5,), 1013.0, dtype=torch.float64), levels, t, q)
    v = pi["vmax_ms"]
    assert v[0] == 0.0, "an ocean colder than the air above it has nothing to give"
    assert (v[1:].diff() > 0).all(), "warmer ocean, higher ceiling"
    assert 65.0 < float(v[3]) < 100.0, "29 C under a mean tropical sounding: a category 4-5 ceiling"
    assert 880.0 < float(pi["pmin_hpa"][3]) < 960.0 and bool(pi["converged"][1:].all())
    assert 190.0 < float(pi["outflow_k"][3]) < 215.0, "outflow near the tropical tropopause"


def test_cape_is_zero_for_a_stable_column_and_positive_for_the_tropics():
    levels, t, q = sounding(2)
    t[1] = 300.0                                                       # isothermal: absolutely stable aloft
    q[1] = 0.001
    value, outflow, _ = cape(t[:, 0], mixing_ratio(q[:, 0]), torch.full((2,), 1000.0, dtype=torch.float64),
                             levels, t, mixing_ratio(q))
    assert 800.0 < float(value[0]) < 3500.0 and float(value[1]) == 0.0 and math.isnan(float(outflow[1]))


def test_entropy_deficit_is_zero_when_saturated_and_grows_with_dry_air():
    t_mid = torch.tensor([275.0, 275.0], dtype=torch.float64)
    from naturev1.thermo import saturation_mixing_ratio

    saturated = saturation_mixing_ratio(t_mid, torch.tensor(600.0, dtype=torch.float64))
    r_mid = torch.stack([saturated[0], saturated[1] * 0.3])
    chi = entropy_deficit(torch.full((2,), 302.0, dtype=torch.float64), torch.full((2,), 1012.0, dtype=torch.float64),
                          t_mid, r_mid, torch.full((2,), 600.0, dtype=torch.float64),
                          torch.full((2,), 298.0, dtype=torch.float64), torch.full((2,), 0.016, dtype=torch.float64),
                          torch.full((2,), 1000.0, dtype=torch.float64))
    assert float(chi[0]) < 0.05 < float(chi[1])


def test_genesis_potential_follows_its_ingredients():
    base = genesis_potential_index(torch.tensor(5e-5), torch.tensor(60.0), torch.tensor(70.0), torch.tensor(5.0))
    assert genesis_potential_index(torch.tensor(1e-4), torch.tensor(60.0), torch.tensor(70.0), torch.tensor(5.0)) > base
    assert genesis_potential_index(torch.tensor(5e-5), torch.tensor(40.0), torch.tensor(70.0), torch.tensor(5.0)) < base
    assert genesis_potential_index(torch.tensor(5e-5), torch.tensor(60.0), torch.tensor(70.0), torch.tensor(25.0)) < base


VARIABLES = (["mean_sea_level_pressure", "sea_surface_temperature", "2m_temperature", "land_sea_mask",
              "10m_u_component_of_wind", "10m_v_component_of_wind"]
             + [f"temperature@{lv}" for lv in LEVELS] + [f"specific_humidity@{lv}" for lv in LEVELS]
             + [f"geopotential@{lv}" for lv in LEVELS]
             + [f"u_component_of_wind@{lv}" for lv in (850, 700, 500, 400, 300, 250, 200)]
             + [f"v_component_of_wind@{lv}" for lv in (850, 700, 500, 400, 300, 250, 200)])
NORMALIZER = Normalizer(VARIABLES, np.zeros(len(VARIABLES), np.float32), np.ones(len(VARIABLES), np.float32),
                        np.zeros(len(VARIABLES), np.float32))
LAT = torch.linspace(-90.0, 90.0, 121, dtype=torch.float64)
LON = torch.arange(240, dtype=torch.float64) * 1.5


def regular_grid():
    lat, lon = torch.meshgrid(torch.deg2rad(LAT), torch.deg2rad(LON), indexing="ij")
    coords = torch.stack([lat.reshape(-1), lon.reshape(-1)], -1)
    return FieldGrid.source(coords, Geometry.globe(), weights=torch.cos(coords[:, 0]).clamp_min(1e-4))


#: Height deficit (m) of a warm-core vortex at each level: largest at the bottom, fading upward.
WARM_CORE = {1000: 110, 925: 100, 850: 90, 700: 70, 600: 55, 500: 40, 400: 25, 300: 12, 250: 6, 200: 3, 150: 1, 100: 0}


def storm_state_grid(clat=18.0, clon=300.0, sst_c=29.5, core=WARM_CORE, cyclonic=True, land=False):
    lat = LAT.view(-1, 1).expand(-1, 240)
    lon = LON.view(1, -1).expand(121, -1)
    north = (lat - clat) * 111.2
    east = ((lon - clon + 180) % 360 - 180) * 111.2 * math.cos(math.radians(clat))
    bump = torch.exp(-(north ** 2 + east ** 2) / (2 * 300.0 ** 2))
    sense = (1.0 if clat >= 0 else -1.0) * (1.0 if cyclonic else -1.0)
    fields = {"mean_sea_level_pressure": 101200.0 - 3000.0 * bump,
              "sea_surface_temperature": torch.full_like(lat, sst_c + 273.15),
              "2m_temperature": torch.full_like(lat, sst_c + 272.15),
              "land_sea_mask": torch.full_like(lat, 1.0 if land else 0.0),
              "10m_u_component_of_wind": -sense * 0.1 * north * bump, "10m_v_component_of_wind": sense * 0.1 * east * bump}
    standard = {1000: 110, 925: 760, 850: 1480, 700: 3100, 600: 4370, 500: 5850, 400: 7570, 300: 9670, 250: 10920,
                200: 12400, 150: 14200, 100: 16600}
    for i, lv in enumerate(LEVELS):
        fields[f"temperature@{lv}"] = torch.full_like(lat, SOUNDING_T[i] + 273.15)
        fields[f"specific_humidity@{lv}"] = torch.full_like(lat, SOUNDING_Q[i] / 1000.0)
        fields[f"geopotential@{lv}"] = (standard[lv] - core.get(lv, 0) * bump) * 9.80665
    for lv in (850, 700, 500, 400, 300, 250, 200):
        strength = {850: 0.05, 700: 0.045, 500: 0.03, 400: 0.02, 300: 0.01, 250: 0.0, 200: -0.01}[lv]
        fields[f"u_component_of_wind@{lv}"] = -sense * strength * north * bump + (5.0 if lv <= 300 else -3.0)
        fields[f"v_component_of_wind@{lv}"] = sense * strength * east * bump
    return torch.stack([fields[v] for v in VARIABLES], -1).reshape(-1, len(VARIABLES)).float()


def test_storm_physics_reads_a_warm_core_tropical_cyclone():
    grid = regular_grid()
    state = torch.stack([storm_state_grid(), storm_state_grid(clat=-15.0, clon=150.0)])
    centre = torch.tensor([[18.0, -60.0], [-15.0, 150.0]])
    physics = storm_physics(state, VARIABLES, NORMALIZER, grid, centre)
    assert physics.shape == (2, len(STORM_PHYSICS_FEATURES)) and torch.isfinite(physics).all()
    f = {name: physics[:, i] for i, name in enumerate(STORM_PHYSICS_FEATURES)}
    assert (f["observed"] == 1).all()
    assert (f["phase_vtl"] > 0).all() and (f["phase_vtu"] > 0).all(), "warm core below and above 600 hPa"
    assert (f["phase_b"].abs() < 0.5).all(), "a symmetric vortex has no thermal asymmetry"
    assert (f["vorticity_850"] > 0).all(), "cyclonic reads positive in both hemispheres"
    assert (f["potential_intensity"] * 50 > 50).all(), "29.5 C ocean: a high ceiling"
    assert torch.allclose(f["potential_intensity"][0], f["potential_intensity"][1], rtol=0.05)


def test_genesis_map_is_zero_over_land_and_cold_water():
    grid = regular_grid()
    warm = genesis_map(storm_state_grid(), VARIABLES, NORMALIZER, grid)
    cold = genesis_map(storm_state_grid(sst_c=22.0), VARIABLES, NORMALIZER, grid)
    dry_land = genesis_map(storm_state_grid(land=True), VARIABLES, NORMALIZER, grid)
    assert warm.shape == (121 * 240,) and float(warm.max()) > 0
    assert float(cold.max()) == 0 and float(dry_land.max()) == 0


# ------------------------------------------------------------------------------------------- wind --

def test_profile_hits_the_gale_radius_and_inverts():
    vmax, rmw = torch.tensor([110.0]), torch.tensor([25.0])
    b = fit_b(vmax, rmw, torch.tensor([220.0]), torch.tensor([34.0]))
    assert abs(float(holland_profile(torch.tensor([220.0]), vmax, rmw, b)) - 34.0) < 0.05
    assert abs(float(radius_of(torch.tensor([34.0]), vmax, rmw, b)) - 220.0) < 0.5
    assert float(holland_profile(rmw, vmax, rmw, b)) == pytest.approx(110.0, rel=1e-6), "the peak is at Rmax"


def test_quadrant_radii_are_reproduced():
    radii_km = torch.tensor([[250.0, 200.0, 120.0, 160.0]])                   # NE SE SW NW
    centre = (20.0, -70.0)
    bearings = torch.deg2rad(torch.tensor([45.0, 135.0, 225.0, 315.0]))
    lat = centre[0] + radii_km[0] * torch.cos(bearings) / 111.195
    lon = centre[1] + radii_km[0] * torch.sin(bearings) / (111.195 * math.cos(math.radians(centre[0])))
    field = wind_field(lat, lon, torch.tensor([centre[0]]), torch.tensor([centre[1]]), torch.tensor([100.0]),
                       torch.tensor([30.0]), radii_km)
    assert torch.allclose(field["speed_kt"][0].float(), torch.full((4,), 34.0), atol=1.5)


def test_motion_strengthens_the_side_where_rotation_and_motion_add():
    # North: anticlockwise, moving north -> strongest on the right (east). South: clockwise, moving south ->
    # strongest on the left of the motion -- which is again the east side.
    for lat0, right_east in ((20.0, True), (-20.0, True)):
        points_lat = torch.tensor([lat0, lat0])
        points_lon = torch.tensor([-69.4, -70.6])                            # east, west, 60 km
        field = wind_field(points_lat, points_lon, torch.tensor([lat0]), torch.tensor([-70.0]), torch.tensor([90.0]),
                           torch.tensor([30.0]), None, torch.tensor([[0.0, 7.0 if lat0 > 0 else -7.0]]))
        east, west = field["speed_kt"][0].tolist()
        assert (east > west) == right_east, "the side where the cyclonic wind and the motion add"


def test_land_roughness_gusts_and_inland_decay():
    over_sea = wind_field(torch.tensor([20.5]), torch.tensor([-70.0]), torch.tensor([20.0]), torch.tensor([-70.0]),
                          torch.tensor([100.0]), torch.tensor([40.0]))
    over_land = wind_field(torch.tensor([20.5]), torch.tensor([-70.0]), torch.tensor([20.0]), torch.tensor([-70.0]),
                           torch.tensor([100.0]), torch.tensor([40.0]), land=torch.tensor([1.0]))
    assert float(over_land["speed_kt"]) == pytest.approx(0.8 * float(over_sea["speed_kt"]), rel=1e-6)
    assert float(over_land["gust_kt"] / over_land["speed_kt"]) > float(over_sea["gust_kt"] / over_sea["speed_kt"])
    decayed = land_decay_kt(torch.tensor([100.0]), torch.tensor([24.0]))
    assert float(decayed) == pytest.approx(26.7 + (90.0 - 26.7) * math.exp(-0.095 * 24), rel=1e-6)
    assert float(land_decay_kt(torch.tensor([100.0]), torch.tensor([0.0]))) == 100.0


def test_rcliper_peaks_at_its_radius_and_grows_with_intensity():
    r = torch.linspace(0, 500, 501)
    strong, weak = rcliper_rate_mm_h(r, torch.tensor(120.0)), rcliper_rate_mm_h(r, torch.tensor(40.0))
    assert float(strong.max()) > float(weak.max())
    # Tuleya et al.: the rain-rate peak contracts toward the centre as the storm intensifies.
    assert 5 < int(strong.argmax()) < int(weak.argmax()) < 80


def _deterministic_outputs(leads, lat_step=0.0, lon_step=-1.0, wind_kt=100.0):
    """A single-scenario forecast moving due west one degree per lead step, with tiny uncertainty."""
    count = len(leads)
    displacement = torch.zeros(1, 1, count, 2)
    for i, hours in enumerate(leads):
        displacement[0, 0, i] = torch.tensor([lat_step * hours / 6, lon_step * hours / 6])
    return {"mode_logits": torch.zeros(1, 1), "displacement": displacement,
            "log_scale": torch.full((1, 1, count, 2), -6.0), "correlation": torch.zeros(1, 1, count),
            "intensity_mean": torch.full((1, count, 2), 0.0).index_fill_(2, torch.tensor([0]), wind_kt * 0.514444),
            "intensity_log_var": torch.full((1, count, 2), -8.0)}


def test_wind_hazard_is_certain_under_the_track_and_zero_far_away():
    leads = (6, 12, 24, 48)
    outputs = _deterministic_outputs(leads)
    hazard = wind_hazard(outputs, 0, (20.0, -60.0), leads, [20.0, 35.0], [-64.0, -60.0], samples=40, trained={})
    p = hazard["probability"]
    assert float(p[0, -1, 2]) == 1.0, "the track passes over this point at hurricane strength"
    assert float(p[1, -1, 0]) == 0.0, "1,600 km north never sees a gale"
    assert (p[0].diff(dim=0) >= 0).all(), "a cumulative probability never falls with lead"
    assert (p[0, :, 0] >= p[0, :, 1]).all() and (p[0, :, 1] >= p[0, :, 2]).all()
    earliest, likely = hazard["arrival_hours"][0].tolist()
    assert 0 <= earliest <= likely < 24, "gales arrive well before the centre reaches it at +24 h"
    assert float(hazard["rain_mm"][0, 0]) > 50.0 and float(hazard["rain_mm"][1, 0]) < 0.5


def test_landfall_weakens_the_wind_inland():
    leads = (6, 12, 24, 48)
    outputs = _deterministic_outputs(leads)
    coast = (lambda lat, lon: (torch.as_tensor(lon) < -63.0).double())          # land west of 63 W
    sea = wind_hazard(outputs, 0, (20.0, -60.0), leads, [20.0], [-70.0], samples=20, trained={})
    land = wind_hazard(outputs, 0, (20.0, -60.0), leads, [20.0], [-70.0], samples=20, trained={}, land_fn=coast)
    assert float(land["peak_kt"][0]) < 0.7 * float(sea["peak_kt"][0])


# ------------------------------------------------------------------------------------ statistics --

def _bank(n=400, leads=(6, 12, 24, 48), seed=0):
    from naturev1.model import STORM_ENVIRONMENT_FEATURES

    g = torch.Generator().manual_seed(seed)
    count = len(leads)
    wind_now = 40 + 80 * torch.rand(n, generator=g)
    shear = torch.rand(n, generator=g) * 3
    physics = torch.zeros(n, len(STORM_PHYSICS_FEATURES))
    physics[:, 0] = 1.6                                                         # 80 m/s potential intensity
    physics[:, -1] = 1.0
    env = torch.zeros(n, len(STORM_ENVIRONMENT_FEATURES))
    env[:, 4] = shear
    env[:, 6] = 1.0
    env[:, 7] = 1.0
    state = torch.zeros(n, 6)
    state[:, 0] = (wind_now - 65.0) / 30.0
    state[:, 5] = 1.0
    hours = torch.tensor(leads, dtype=torch.float32)
    change = (12.0 - 6.0 * shear).view(-1, 1) * hours.view(1, -1) / 24.0         # shear stops intensification
    truth = (wind_now.view(-1, 1) + change + torch.randn(n, count, generator=g)).clamp_min(15.0)
    intensity_target = torch.zeros(n, count, 2)
    intensity_target[..., 0] = truth * 0.514444
    return {"targets": {"current_wind_kt": wind_now, "storm_state": state, "storm_environment": env,
                        "storm_physics": physics, "storm_center": torch.tensor([[20.0, -60.0]]).repeat(n, 1),
                        "intensity_target": intensity_target, "intensity_mask": torch.ones(n, count, 2)}}, truth


class _Stub(torch.nn.Module):
    def __init__(self, leads):
        super().__init__()
        from naturev1.statistical import StatisticalModels

        self.statistics = StatisticalModels(leads)
        self.config = type("C", (), {"lead_times_hours": leads})()


def test_regression_and_logistic_models_learn_what_shear_does():
    from naturev1.statistical import (
        fit_climatology, fit_logistic, fit_regression, intensity_predictors,
    )

    leads = (6, 12, 24, 48)
    bank, truth = _bank(leads=leads)
    model = _Stub(leads)
    fit_regression(model, bank)
    fit_logistic(model, bank, steps=200)
    fit_climatology(model, bank)
    t = bank["targets"]
    x = intensity_predictors(t["storm_state"], t["storm_environment"], t["storm_physics"], t["storm_center"])
    regression = model.statistics.regression_member(x, t["current_wind_kt"])
    persistence = (t["current_wind_kt"].view(-1, 1) - truth).abs().mean()
    assert float((regression - truth).abs().mean()) < 0.3 * float(persistence)
    logistic = model.statistics.logistic_member(x, t["current_wind_kt"], torch.full((x.shape[0],), 155.0))
    assert float((logistic - truth).abs().mean()) < float(persistence)
    climatology = model.statistics.climatology_member(t["current_wind_kt"], torch.zeros_like(t["current_wind_kt"]))
    assert torch.isfinite(climatology).all()


def test_intensity_consensus_finds_the_good_member_and_removes_bias():
    from naturev1.statistical import INTENSITY_MEMBERS, fit_intensity_consensus

    leads = (6, 12)
    bank, truth = _bank(leads=leads)
    model = _Stub(leads)
    n = truth.shape[0]
    members = torch.full((n, len(leads), len(INTENSITY_MEMBERS)), float("nan"))
    members[..., 0] = truth + 10.0                                              # biased but right in shape
    members[..., 2] = truth + torch.randn(n, len(leads)) * 25.0                 # noisy
    fit_intensity_consensus(model, bank, members)
    weights = model.statistics.intensity_weights
    assert (weights[:, 0] > 0.9).all() and (weights[:, 1:].sum(-1)[:, None] >= 0).all()
    blended = model.statistics.consensus(members)
    assert float((blended - truth).abs().mean()) < 2.0, "the +10 kt bias is corrected away"


def test_track_bias_and_predicted_error_cone():
    from naturev1.statistical import consensus_radius_km, corrected_track, fit_consensus_error, fit_track_bias, member_spread_km
    from naturev1.steering import displacement_km

    leads = (12, 24)
    g = torch.Generator().manual_seed(0)
    n = 2000
    truth = torch.randn(n, len(leads), 2, generator=g)
    spread_scale = torch.rand(n, 1, 1, generator=g) * 2
    members = truth.unsqueeze(2) + torch.randn(n, len(leads), 8, 2, generator=g) * spread_scale.unsqueeze(-1)
    members = members + torch.tensor([-0.3, 0.0]).view(1, 1, 1, 2)            # everything slow toward the pole
    model = _Stub(leads)
    model.track_consensus = torch.full((len(leads), 8), 1.0 / 8)
    valid = torch.ones(n, len(leads), dtype=torch.bool)
    latitude = torch.full((n,), 20.0)
    fit_track_bias(model, members, truth, valid, latitude, shrink=1.0)
    assert torch.allclose(model.statistics.track_bias[:, 0], torch.full((2,), 0.3), atol=0.05)
    fit_consensus_error(model, members, truth, valid, latitude, coverage=0.67)
    corrected = corrected_track(model, members, latitude)
    radius = consensus_radius_km(model, member_spread_km(members, corrected, latitude))
    error = displacement_km(truth - corrected, latitude.view(-1, 1)).norm(dim=-1)
    coverage = float((error <= radius).float().mean())
    assert 0.6 < coverage < 0.74
    assert float(model.statistics.consensus_error[:, 1].min()) > 0, "more disagreement, wider cone"


# --------------------------------------------------------------------------------------- tracking --

def test_intensity_dependent_steering_layer_changes_the_track():
    from naturev1.steering import physics_steering_track

    grid = regular_grid()
    state = storm_state_grid()                                                   # easterly below, westerly aloft
    centre = torch.tensor([[18.0, -60.0]])
    weak = physics_steering_track(state.unsqueeze(0), VARIABLES, NORMALIZER, grid, centre, (24,),
                                  wind_kt=torch.tensor([35.0]))
    strong = physics_steering_track(state.unsqueeze(0), VARIABLES, NORMALIZER, grid, centre, (24,),
                                    wind_kt=torch.tensor([130.0]))
    assert float(weak[0, 0, 1]) < float(strong[0, 0, 1]), "a deep storm feels the westerlies aloft"


def test_multi_field_fix_and_rollout_tracking_read_intensity_and_core():
    from naturev1.tracker import multi_fix_centre, track_storm_in_rollout

    grid = regular_grid()
    fix = multi_fix_centre(storm_state_grid().unsqueeze(0), VARIABLES, NORMALIZER, grid, torch.tensor([[17.0, -61.0]]))
    assert bool(fix["found"][0]) and int(fix["fixes"][0]) >= 2
    assert abs(float(fix["centre"][0, 0]) - 18.0) < 1.0 and abs(float(fix["centre"][0, 1]) + 60.0) < 1.0
    states = torch.stack([storm_state_grid(clat=18.0 + 0.5 * k, clon=300.0 - 1.0 * k) for k in range(1, 5)]).unsqueeze(1)
    tracked = track_storm_in_rollout(states, VARIABLES, NORMALIZER, grid, torch.tensor([[18.0, -60.0]]), (6, 12, 24))
    assert abs(float(tracked["displacement"][0, 2, 0]) - 2.0) < 0.8 and abs(float(tracked["displacement"][0, 2, 1]) + 4.0) < 0.8
    assert (tracked["pressure_hpa"][0] < 1000).all() and (tracked["phase"][0, :, 1] > 0).all()


def test_analogs_move_like_their_neighbours_in_both_hemispheres():
    from naturev1.besttrack import Track
    from naturev1.tracker import AnalogForecaster

    tracks = []
    for k in range(40):
        times = np.datetime64("2001-08-01T00", "s") + np.arange(30) * np.timedelta64(6, "h") + np.timedelta64(k * 30, "D")
        lat0, lon0 = 12.0 + (k % 5), -40.0 - (k % 7)
        tracks.append(Track(storm_id=f"S{k}", name="A", time=times, latitude=(lat0 + 0.3 * np.arange(30)).astype(np.float32),
                            longitude=(lon0 - 0.8 * np.arange(30)).astype(np.float32),
                            max_wind_kt=np.full(30, 60.0, np.float32), min_pressure_hpa=np.full(30, 990.0, np.float32),
                            status=np.array(["TS"] * 30), rmw_nmi=np.full(30, np.nan, np.float32),
                            wind_radii_nmi=np.full((30, 3, 4), np.nan, np.float32), record=np.array([""] * 30)))
    analog = AnalogForecaster.from_tracks(tracks, (24, 48))
    state = torch.tensor([[0.0, 0.0, 0.6, -1.6, 0.0, 1.0], [0.0, 0.0, -0.6, -1.6, 0.0, 1.0]])
    out = analog.predict(torch.tensor([[14.0, -45.0], [-14.0, -45.0]]), state, torch.tensor([60.0, 60.0]),
                         torch.tensor([0.6, 0.6]))
    north, south = out["displacement"]
    assert float(north[0, 0]) > 0.8 and float(north[0, 1]) < -2.0, "poleward and westward, like its analogs"
    assert float(south[0, 0]) < -0.8, "mirrored into the southern hemisphere"
    restored = AnalogForecaster.from_state_dict(analog.state_dict())
    assert torch.allclose(restored.predict(torch.tensor([[14.0, -45.0]]), state[:1], torch.tensor([60.0]),
                                           torch.tensor([0.6]))["displacement"], out["displacement"][:1], atol=1e-5)


# ------------------------------------------------------------------------------------ integration --

def _tiny_model(**over):
    from naturev1.model import NatureConfig, NatureV1

    torch.manual_seed(0)
    config = NatureConfig(analysis_channels=8, state_channels=3, hidden_size=32, num_layers=2, num_heads=2, num_kv_heads=1,
                          head_dim=16, intermediate_size=64, latent_points=96, latent_neighbours=12, latent_cluster=16,
                          landmark_dim=16, index_branching=2, index_beam_width=2, index_topk=2, recurrent_heads=2,
                          recurrent_head_dim=16, history_frames=2, min_radius_km=600.0, max_radius_km=2000.0,
                          encode_radius_km=900.0, lead_times_hours=(6, 12, 24, 48), **over)
    return NatureV1(config, fibonacci_sphere(96, num_neighbours=12, cluster_size=16))


def test_physics_input_changes_nothing_until_trained_and_reaches_ri2():
    model = _tiny_model(ri2=True)
    pooled = torch.randn(2, 32)
    centre = torch.tensor([[20.0, -60.0], [25.0, -70.0]])
    physics = torch.randn(2, len(STORM_PHYSICS_FEATURES))
    a, b = model.storm_heads(pooled, centre), model.storm_heads(pooled, centre, storm_physics=physics)
    for key in a:
        if key != "ri2_logits":
            assert torch.equal(a[key], b[key]), key
    assert not torch.equal(a["ri2_logits"], b["ri2_logits"]), "RI 2.0 reads the thermodynamics from the start"


def test_forecast_carries_physics_consensus_and_wind_hazard():
    from naturev1 import build_forecast, check_forecast
    from naturev1.statistical import fit_climatology, fit_intensity_consensus, fit_regression

    model = _tiny_model()
    leads = model.config.lead_times_hours
    bank, _ = _bank(n=300, leads=leads)
    fit_regression(model, bank)
    fit_climatology(model, bank)
    t = bank["targets"]
    with torch.no_grad():
        for p in model.storm_head_parameters():
            p.normal_(0, 0.01)
        out = model.storm_heads(torch.randn(300, 32), t["storm_center"], t["storm_state"], t["storm_environment"],
                                storm_physics=t["storm_physics"])
    bank["features"] = torch.zeros(300, 32)
    members = out["intensity_members_kt"]
    fit_intensity_consensus(model, bank, members)
    with torch.no_grad():
        single = model.storm_heads(torch.randn(1, 32), t["storm_center"][:1], t["storm_state"][:1],
                                   t["storm_environment"][:1], storm_physics=t["storm_physics"][:1])
    physics = storm_physics(torch.stack([storm_state_grid()]), VARIABLES, NORMALIZER, regular_grid(),
                            torch.tensor([[18.0, -60.0]]))
    bundle = build_forecast(single, leads, dt.datetime(2020, 9, 1), storm_center=(20.0, -60.0),
                            point_of_interest=(20.2, -61.0), trained=None, storm_physics=physics, model=model,
                            hazard_samples=50)
    assert len(bundle["intensity_consensus"]["by_lead"]) == len(leads)
    assert "label" in bundle["physics"]["phase"] and bundle["physics"]["potential_intensity_kt"] > 60
    hazard = bundle["wind_hazard"]
    assert len(hazard["by_lead"]) == len(leads) and 0.0 <= hazard["by_lead"][-1]["p_34kt"] <= 1.0
    assert not [p for p in check_forecast(bundle) if "off the planet" in p]
