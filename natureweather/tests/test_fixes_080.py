# Copyright 2026 Nathan. Apache-2.0.
"""
Regression tests for the 0.8.0 audit.

Each test pins one bug found in a full read of the code to the property that would have caught it.
The critical ones first: a GPU memory leak on every training step, hurricane labels at the wrong valid
time, a rollout that never rolled out, a sun on the wrong side of the planet, a publish gate that could
never open, and outputs that had never been trained printed as forecasts.
"""

from __future__ import annotations

import json
import math
import signal

import numpy as np
import pytest
import torch
from naturev1 import (
    CachedERA5,
    CheckpointManager,
    NatureConfig,
    NatureV1,
    RolloutSchedule,
    StateStepper,
    Trainer,
    TrainSettings,
    build_climatology,
    build_forecast,
    check_forecast,
    era5_loader,
    lead_track_points,
    responds_to_input,
    score_state,
    solar_forcing,
    total_loss,
    train_state_rollout,
)
from naturev1.besttrack import Track
from naturev1.corpora import latlon_to_coords
from naturev1.era5 import GridCollate, Normalizer, equiangular_weights, materialise
from naturev1.guards import PreflightError
from naturev1.model import SURFACE_FIELDS
from naturev1.storms import StormTargets

from ihelix import FieldGrid, Geometry, fibonacci_sphere, grid_fingerprint


LAT = np.linspace(-90.0, 90.0, 13)
LON = np.arange(24) * 15.0
VARIABLES = ["2m_temperature", "surface_pressure", "toa_incident_solar_radiation_6hr",
             "land_sea_mask", "temperature@500", "geopotential@500"]
WIDTH = len(VARIABLES) + 2


def _grid():
    return FieldGrid.source(latlon_to_coords(LAT, LON), Geometry.globe(),
                            weights=torch.tensor(equiangular_weights(LAT, len(LON))))


def _normalizer():
    return Normalizer(list(VARIABLES), np.array([280, 1e5, 1e6, 0.5, 250, 5.5e4], np.float32),
                      np.array([20, 1e3, 5e6, 0.5, 10, 3e3], np.float32), np.zeros(len(VARIABLES), np.float32))


@pytest.fixture()
def staged(tmp_path):
    """A small staged corpus on disk, exactly what materialise() writes."""
    steps = 40
    rng = np.random.default_rng(0)
    values = rng.normal(size=(steps, len(LAT), len(LON), len(VARIABLES))).astype(np.float16)
    path = tmp_path / "era5.npy"
    np.save(path, values)
    times = (1.6e9 + np.arange(steps) * 21600).astype(np.int64)
    path.with_suffix(".json").write_text(json.dumps({
        "variables": VARIABLES, "time": times.tolist(), "shape": list(values.shape),
        "statistics": _normalizer().to_dict()}))
    return path


def _model(state_channels=0, noise_dim=0, channels=WIDTH):
    config = NatureConfig(analysis_channels=channels, hidden_size=32, num_layers=2, num_heads=2,
                          num_kv_heads=1, head_dim=16, intermediate_size=64, latent_points=48,
                          history_frames=2, lead_times_hours=(6, 12), track_modes=2,
                          state_channels=state_channels, noise_dim=noise_dim)
    return NatureV1(config, fibonacci_sphere(48, num_neighbours=8, cluster_size=12))


# ------------------------------------------------------------------------------------------ critical --

def test_worker_batches_do_not_grow_the_link_cache(staged):
    """Every worker batch is a fresh copy of the grid; keyed on identity, each built and kept a link."""
    grid = _grid()
    model = _model()
    dataset = CachedERA5(staged, history=2, lead_steps=(1, 2), channels=WIDTH, augment=False)
    loader = era5_loader(dataset, batch_size=2, num_workers=2, analysis_grid=grid, output_grid=grid)
    with torch.no_grad():
        for _, batch in zip(range(6), loader):
            assert batch["analysis_grid"] is not grid
            model(analysis=batch["analysis"], analysis_grid=batch["analysis_grid"],
                  calendar=batch["calendar"], output_grid=batch["output_grid"])
    assert len(model._links) == 2, f"expected one encode and one decode link, found {len(model._links)}"


def test_collate_keeps_a_private_cpu_copy():
    """A GPU grid in a forked worker cannot be pickled back; the collate must hold a CPU copy."""
    grid = _grid()
    collate = GridCollate(grid, grid)
    assert collate.analysis_grid is not grid
    assert collate.analysis_grid.coords.device.type == "cpu"
    assert grid_fingerprint(collate.analysis_grid) == grid_fingerprint(grid)


def _track_with_landfall_special():
    """Six-hourly fixes plus one landfall row at 14:30 -- HURDAT2's convention."""
    base = np.datetime64("2022-09-26T00:00:00")
    hours = [0, 6, 12, 14.5, 18, 24, 30, 36]
    times = np.array([base + np.timedelta64(int(h * 3600), "s") for h in hours])
    n = len(times)
    record = np.array(["", "", "", "L", "", "", "", ""])
    return Track("AL092022", "IAN", times, np.linspace(20, 27, n).astype(np.float32),
                 np.linspace(-80, -82, n).astype(np.float32), np.linspace(100, 130, n).astype(np.float32),
                 np.full(n, 950.0, np.float32), np.array(["HU"] * n), np.full(n, np.nan, np.float32),
                 np.full((n, 3, 4), np.nan, np.float32), record)


def test_storm_leads_are_found_by_time_not_by_row():
    """Indexed by row, the landfall special made every later lead 6 h short -- for landfalling storms."""
    track = _track_with_landfall_special()
    points = lead_track_points(track, 0, (1, 2, 3, 4, 6))
    hours = [(track.time[p] - track.time[0]) / np.timedelta64(1, "h") if p >= 0 else None for p in points]
    assert hours == [6, 12, 18, 24, 36]

    built = StormTargets(track, 0, np.array([1, 2, 3, 4, 6]), np.full(len(track), np.nan)).build()
    # +18 h is the 18:00 fix, not the 14:30 landfall row that sits fourth in the table.
    assert float(built["track_target"][2, 0]) == pytest.approx(float(track.latitude[4] - track.latitude[0]))
    assert float(built["landfall_target"][2]) == 1.0, "landfall happened before +18 h"
    assert float(built["landfall_target"][1]) == 0.0, "but not before +12 h"


def test_storm_samples_carry_their_centre():
    built = StormTargets(_track_with_landfall_special(), 2, np.array([1]), np.full(8, np.nan)).build()
    assert built["storm_center"].tolist() == pytest.approx([float(np.linspace(20, 27, 8)[2]),
                                                            float(np.linspace(-80, -82, 8)[2])])


def test_storm_heads_see_which_storm():
    """Two storms at the same hour: identical inputs, different centres, must give different answers."""
    torch.manual_seed(0)
    model = _model().eval()
    grid = _grid()
    analysis = torch.randn(1, 2, grid.num_points, WIDTH).expand(2, -1, -1, -1)
    calendar = torch.zeros(2, 2, 6)
    centres = torch.tensor([[15.0, -45.0], [25.0, 130.0]])
    with torch.no_grad():
        pooled = model(analysis=analysis, analysis_grid=grid, calendar=calendar, output_grid=grid,
                       storm_center=centres)
        globally = model(analysis=analysis, analysis_grid=grid, calendar=calendar, output_grid=grid)
        fallback = model(analysis=analysis, analysis_grid=grid, calendar=calendar, output_grid=grid,
                         storm_center=torch.full((2, 2), float("nan")))
    assert not torch.allclose(pooled["displacement"][0], pooled["displacement"][1])
    assert torch.allclose(globally["displacement"][0], globally["displacement"][1])
    assert torch.allclose(fallback["displacement"], globally["displacement"])
    assert torch.allclose(pooled["enso_mean"], globally["enso_mean"]), "ENSO stays a global quantity"


def test_rollout_training_actually_rolls_out_and_resumes(tmp_path, staged):
    """The old loop was one epoch -- 900 steps -- with a curriculum that started at step 2000."""
    grid = _grid()
    dataset = CachedERA5(staged, history=2, lead_steps=1, channels=WIDTH, augment=True, state_steps=4)
    stepper = StateStepper.build(dataset, WIDTH, grid, samples=8)
    model = _model(state_channels=stepper.roles.num_prognostic)
    loader = era5_loader(dataset, batch_size=4, num_workers=0, analysis_grid=grid, output_grid=grid)
    schedule = RolloutSchedule(start_step=2, ramp_steps=4, max_steps=4)

    state = train_state_rollout(model, loader, stepper, max_steps=14, schedule=schedule, precision="fp32",
                                checkpoint_dir=str(tmp_path / "rollout"), log_every=1)
    assert state.step == 14, "must cycle the loader until max_steps, not stop after one epoch"
    assert state.epoch >= 1
    assert max(entry["state_horizon"] for entry in state.history) == 4, "the horizon must ramp"
    assert model.trained_heads()["state"] == 14

    again = _model(state_channels=stepper.roles.num_prognostic)
    resumed = train_state_rollout(again, loader, stepper, max_steps=16, schedule=schedule,
                                  precision="fp32", checkpoint_dir=str(tmp_path / "rollout"))
    assert resumed.step == 16
    assert again.trained_heads()["state"] == 16, "the training record travels with the checkpoint"


def test_sun_rotates_with_an_augmented_globe():
    """A sample rolled east by k cells holds the weather from k cells west, and its sun must too."""
    grid = _grid()
    normalizer = _normalizer()
    when = 1.6e9 + 5 * 3600
    plain = solar_forcing(grid, [when], "toa_incident_solar_radiation_6hr", normalizer)[0]
    rolled = solar_forcing(grid, [when], "toa_incident_solar_radiation_6hr", normalizer,
                           roll_degrees=[5 * 15.0])[0]
    expected = np.roll(plain.view(len(LAT), len(LON)).numpy(), 5, axis=1)
    assert np.allclose(rolled.view(len(LAT), len(LON)).numpy(), expected, atol=1e-5)


def test_items_report_their_rotation(staged):
    item = CachedERA5(staged, history=2, lead_steps=1, channels=WIDTH, augment=True)[3]
    assert "roll_degrees" in item and float(item["roll_degrees"]) % 15.0 == 0.0
    assert float(CachedERA5(staged, history=2, lead_steps=1, channels=WIDTH, augment=False)[3]
                 ["roll_degrees"]) == 0.0


def test_response_check_refuses_the_same_batch_three_times(staged):
    """Fed next(iter(unshuffled_loader)) three times, it refused every model ever trained."""
    grid = _grid()
    model = _model()
    loader = era5_loader(CachedERA5(staged, history=2, lead_steps=1, channels=WIDTH, augment=False),
                         batch_size=1, num_workers=0, shuffle=False)
    with pytest.raises(PreflightError, match="same batch"):
        responds_to_input(model, lambda: next(iter(loader)), grid)

    stream = iter(loader)
    responds, spread = responds_to_input(model, lambda: next(stream), grid)
    assert isinstance(responds, bool) and spread >= 0.0


def test_untrained_outputs_are_not_presented_as_forecasts():
    """ENSO, weather type and RI onset have no labels anywhere in the pipeline; they must say so."""
    model = _model().eval()
    grid = _grid()
    batch = {"analysis": torch.randn(1, 2, grid.num_points, WIDTH), "calendar": torch.zeros(1, 2, 6)}
    with torch.no_grad():
        outputs = model(analysis=batch["analysis"], analysis_grid=grid, calendar=batch["calendar"],
                        output_grid=grid)

    model.mark_trained({"field": 1.0, "track": 1.0, "ri": 1.0, "intensity": 1.0, "landfall": 1.0,
                        "eyewall_peak": 1.0}, batch)
    trained = model.trained_heads()
    assert trained["enso"] == 0 and trained["weather_type"] == 0 and trained["field"] == 1

    import datetime as dt

    bundle = build_forecast(outputs, (6, 12), dt.datetime(2024, 9, 1), storm_center=(25.0, -80.0),
                            output_grid=grid, point_of_interest=(25.8, -80.2), normalizer=_normalizer(),
                            trained=trained, inputs=("analysis",))
    assert bundle["enso"]["available"] is False
    assert bundle["point_forecast"]["forecast"][0]["weather"]["available"] is False
    assert "likeliest_onset_hours" not in bundle["rapid_intensification"]
    assert bundle["trustworthy"] is True
    assert check_forecast(bundle) == [] or all("not" not in p for p in check_forecast(bundle))

    satellite = build_forecast(outputs, (6, 12), dt.datetime(2024, 9, 1), trained=trained,
                               inputs=("satellite",))
    assert satellite["trustworthy"] is False, "the satellite encoder has never been trained"
    assert check_forecast(satellite), "an untrustworthy bundle must fail the forecast check"


def test_point_forecast_is_in_physical_units():
    """Raw heads are in standard deviations: 285 K printed as 0.35."""
    model = _model().eval()
    grid = _grid()
    with torch.no_grad():
        outputs = model(analysis=torch.zeros(1, 2, grid.num_points, WIDTH), analysis_grid=grid,
                        calendar=torch.zeros(1, 2, 6), output_grid=grid)
    outputs["field_mean"] = torch.zeros_like(outputs["field_mean"])

    import datetime as dt

    bundle = build_forecast(outputs, (6, 12), dt.datetime(2024, 9, 1), output_grid=grid,
                            point_of_interest=(0.0, 0.0), normalizer=_normalizer())
    t2m = bundle["point_forecast"]["forecast"][0]["fields"]["t2m"]
    assert t2m["units"] == "K" and t2m["value"] == pytest.approx(280.0)
    assert bundle["point_forecast"]["forecast"][0]["fields"]["rh"]["available"] is False


def test_untrained_state_head_scores_exactly_as_persistence(staged):
    """Zero residual is persistence; the state scorecard must say skill 0.0 to the last digit."""
    grid = _grid()
    dataset = CachedERA5(staged, history=2, lead_steps=1, channels=WIDTH, augment=False, state_steps=3)
    stepper = StateStepper.build(dataset, WIDTH, grid, samples=8)
    model = _model(state_channels=stepper.roles.num_prognostic)
    loader = era5_loader(dataset, batch_size=2, num_workers=0, shuffle=False,
                         analysis_grid=grid, output_grid=grid)
    climatology = build_climatology(dataset, np.arange(len(dataset)), WIDTH, grid.num_points)
    card = score_state(model, loader, stepper, {"z500": "geopotential@500", "t500": "temperature@500",
                                                "absent": "not_a_channel"},
                       steps=3, climatology=climatology, max_batches=3)
    labels = {score.field for score in card.scores}
    assert labels == {"z500", "t500"}
    for score in card.scores:
        assert score.skill_vs_persistence == pytest.approx(0.0, abs=1e-9)
        assert score.acc is not None and -1.0 <= score.acc <= 1.0
    assert sorted({score.lead_hours for score in card.scores}) == [6, 12, 18]


def test_hub_checkpoints_are_namespaced_per_stage(tmp_path):
    """One shared latest.pt let stage two resume from stage one's step count and train nothing."""
    assert CheckpointManager(tmp_path / "stage1").hub_path == "checkpoints/stage1/latest.pt"
    assert CheckpointManager(tmp_path / "stage2").hub_path == "checkpoints/stage2/latest.pt"


def test_staging_twice_is_free(tmp_path):
    """The cell skipped staging whenever the file existed -- including after a crash at 4%."""
    xr = pytest.importorskip("xarray")
    times = np.array([np.datetime64("2020-01-01T00") + np.timedelta64(6 * i, "h") for i in range(12)])
    data = np.random.default_rng(0).normal(280, 5, size=(12, len(LON), len(LAT))).astype(np.float32)
    store = xr.Dataset({"2m_temperature": (("time", "longitude", "latitude"), data)},
                       coords={"time": times, "longitude": LON, "latitude": LAT})
    normalizer = Normalizer(["2m_temperature"], np.array([280.0], np.float32), np.array([5.0], np.float32),
                            np.zeros(1, np.float32))
    path = tmp_path / "era5.npy"
    materialise(store, np.arange(12), path, variables=("2m_temperature",), normalizer=normalizer,
                workers=2, progress=False)
    stamp = path.stat().st_mtime_ns
    materialise(store, np.arange(12), path, variables=("2m_temperature",), normalizer=normalizer,
                workers=2, progress=False)
    assert path.stat().st_mtime_ns == stamp, "a finished staging must not be redone"


# --------------------------------------------------------------------------------------------- major --

def test_trainer_gives_the_interrupt_key_back(staged):
    """Installed for good, the trainer's handler swallowed every later Stop in the notebook."""
    before = signal.getsignal(signal.SIGINT)
    grid = _grid()
    model = _model()
    loader = era5_loader(CachedERA5(staged, history=2, lead_steps=(1, 2), channels=WIDTH, augment=False),
                         batch_size=2, num_workers=0, analysis_grid=grid, output_grid=grid)
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        trainer = Trainer(model, TrainSettings(max_steps=1, checkpoint_dir=directory, precision="fp32",
                                               warmup_steps=1, log_every=1), device="cpu")
        assert signal.getsignal(signal.SIGINT) is before, "constructing a trainer must not take the key"
        trainer.fit(loader, epochs=1)
    assert signal.getsignal(signal.SIGINT) is before
    assert model.trained_heads()["field"] == 1 and model.trained_heads()["analysis_encoder"] == 1


def test_field_loss_is_area_weighted():
    """Unweighted, the pole rows -- a sliver of the planet -- count as much as the equator."""
    grid = _grid()
    points = grid.num_points
    outputs = {"field_mean": torch.zeros(1, points, 1, len(SURFACE_FIELDS)),
               "field_log_var": torch.zeros(1, points, 1, len(SURFACE_FIELDS))}
    target = torch.zeros(1, points, 1, len(SURFACE_FIELDS))
    polar = torch.tensor(np.abs(np.repeat(LAT, len(LON))) > 80)
    target[0, polar] = 5.0
    batch = {"field_target": target, "field_mask": torch.ones_like(target)}
    plain, _ = total_loss(outputs, batch, SURFACE_FIELDS)
    weighted, _ = total_loss(outputs, {**batch, "output_grid": grid}, SURFACE_FIELDS)
    assert float(weighted) < 0.5 * float(plain), "polar errors must count for the area they cover"


def test_climatology_is_unrotated_and_uses_the_indices(staged):
    dataset = CachedERA5(staged, history=2, lead_steps=1, channels=WIDTH, augment=True)
    one = build_climatology(dataset, np.array([5]), WIDTH, len(LAT) * len(LON), samples=1)
    dataset.augment = False
    assert torch.allclose(one, dataset[5]["analysis"][-1]), "index 5, unrotated"
    dataset.augment = True
    build_climatology(dataset, np.arange(4), WIDTH, len(LAT) * len(LON))
    assert dataset.augment is True, "augmentation is restored afterwards"


def test_old_checkpoints_load_and_mismatched_ones_are_set_aside(tmp_path):
    model = _model()
    legacy = {k: v for k, v in model.state_dict().items()
              if not k.startswith(("storm_position", "head_steps"))}
    _model().load_state_dict(legacy)                      # a 0.7 checkpoint loads cleanly

    manager = CheckpointManager(tmp_path / "stage1")
    manager.save(_model(channels=WIDTH), None, force=True)
    wider = _model(channels=WIDTH + 4)
    assert manager.load(wider) is None
    assert not (tmp_path / "stage1" / "latest.pt").exists()
    assert list((tmp_path / "stage1" / "incompatible").glob("*latest.pt")), "kept, not deleted"


def test_cached_dataset_keeps_its_last_window(staged):
    dataset = CachedERA5(staged, history=2, lead_steps=1, channels=WIDTH, augment=False)
    assert len(dataset) == 40 - 3 + 1
    assert dataset[len(dataset) - 1]["field_target"].shape[0] == len(LAT) * len(LON)


# ------------------------------------------------------------------------------ stage two, fast path --

def test_storm_state_reads_the_last_twelve_hours_by_time():
    from naturev1 import storm_state_vector

    track = _track_with_landfall_special()
    state = storm_state_vector(track, 4)                       # 18:00, twelve hours after 06:00
    assert state[2] == pytest.approx(float(track.latitude[4] - track.latitude[1]))
    assert state[4] == pytest.approx(float(track.max_wind_kt[4] - track.max_wind_kt[1]) / 15.0)
    assert state[5] == 1.0
    assert storm_state_vector(track, 0)[2] == 0.0, "the first fix has no history"


def test_fast_storm_path_is_the_forward_pass():
    """Stage two trains on storm_heads(storm_features(...)); it must be the model, not an imitation."""
    torch.manual_seed(0)
    model = _model().eval()
    with torch.no_grad():
        for module in (model.storm_position, model.storm_state_embed):
            module.weight.normal_(0, 0.1)
    grid = _grid()
    analysis = torch.randn(2, 2, grid.num_points, WIDTH)
    calendar = torch.randn(2, 2, 6)
    centre = torch.tensor([[15.0, -45.0], [25.0, 130.0]])
    state = torch.randn(2, 6)
    with torch.no_grad():
        full = model(analysis=analysis, analysis_grid=grid, calendar=calendar, output_grid=grid,
                     storm_center=centre, storm_state=state)
        fast = model.storm_heads(model.storm_features(analysis, grid, calendar, centre), centre, state)
    for key in ("displacement", "intensity_mean", "ri_logits", "eyewall_peak_wind_kt", "landfall_logit"):
        assert torch.allclose(full[key], fast[key], atol=1e-5), key


def _store(steps=48):
    xr = pytest.importorskip("xarray")
    rng = np.random.default_rng(1)
    times = np.array([np.datetime64("2005-08-20T00") + np.timedelta64(6 * i, "h") for i in range(steps)])
    shape = (steps, len(LON), len(LAT))
    return xr.Dataset(
        {"2m_temperature": (("time", "longitude", "latitude"), rng.normal(285, 10, shape).astype(np.float32)),
         "surface_pressure": (("time", "longitude", "latitude"), rng.normal(1e5, 900, shape).astype(np.float32)),
         "toa_incident_solar_radiation_6hr": (("time", "longitude", "latitude"),
                                              rng.uniform(0, 2e7, shape).astype(np.float32)),
         "land_sea_mask": (("longitude", "latitude"), rng.integers(0, 2, shape[1:]).astype(np.float32)),
         "temperature": (("time", "level", "longitude", "latitude"),
                         rng.normal(250, 8, (steps, 1, *shape[1:])).astype(np.float32)),
         "geopotential": (("time", "level", "longitude", "latitude"),
                          rng.normal(5.5e4, 2e3, (steps, 1, *shape[1:])).astype(np.float32))},
        coords={"time": times, "longitude": LON, "latitude": LAT, "level": [500]})


def _storm(storm_id, first_hour, points, lat0):
    base = np.datetime64("2005-08-20T00:00:00") + np.timedelta64(first_hour * 3600, "s")
    times = np.array([base + np.timedelta64(6 * 3600 * i, "s") for i in range(points)])
    return Track(storm_id, storm_id, times, np.linspace(lat0, lat0 + 8, points).astype(np.float32),
                 np.linspace(-60, -80, points).astype(np.float32), np.linspace(40, 120, points).astype(np.float32),
                 np.linspace(1000, 940, points).astype(np.float32), np.array(["HU"] * points),
                 np.full(points, np.nan, np.float32), np.full((points, 3, 4), np.nan, np.float32),
                 np.array([""] * points))


def test_feature_bank_trains_the_heads_and_is_reused(tmp_path):
    from naturev1 import (
        ERA5Window,
        pair_tracks_with_reanalysis,
        storm_feature_bank,
        storm_scorecard,
        train_storm_heads,
    )

    store = _store()
    variables = ("2m_temperature", "surface_pressure", "toa_incident_solar_radiation_6hr", "land_sea_mask",
                 "temperature", "geopotential")
    tracks = [_storm("AL012005", 24, 20, 15.0), _storm("AL022005", 48, 16, 22.0)]  # overlapping in time
    store_times = store.time.values.astype("datetime64[s]").astype(np.int64)
    starts, targets, report = pair_tracks_with_reanalysis(tracks, store_times, (1, 2), history=2)
    assert report["paired"] > 20 and report["storms"] == 2
    base = ERA5Window(store, indices=starts[:1], history=2, lead_steps=1, variables=variables, levels=[500],
                      channels=WIDTH, augment=False)
    grid = _grid()
    model = _model()
    model.config.lead_times_hours = (6, 12)

    cache = str(tmp_path / "bank.pt")
    bank = storm_feature_bank(model, store, base, starts, targets, grid, cache=cache, precision="fp32",
                              progress=False)
    assert bank["features"].shape == (len(starts), model.config.hidden_size)
    assert torch.isfinite(bank["features"]).all() and bank["features"].abs().sum() > 0

    # Each sample's features are exactly what the full model would compute from the streamed window.
    window = ERA5Window(store, indices=starts[5:6], history=2, lead_steps=1, variables=variables,
                        levels=[500], channels=WIDTH, augment=False)[0]
    direct = model.storm_features(window["analysis"][None], grid, window["calendar"][None],
                                  targets[5].build()["storm_center"][None])
    assert torch.allclose(direct[0], bank["features"][5], atol=1e-4)

    again = storm_feature_bank(model, store, base, starts, targets, grid, cache=cache, progress=False)
    assert again["fingerprint"] == bank["fingerprint"]

    history = train_storm_heads(model, bank, bank, epochs=4, batch_size=8, patience=10,
                                checkpoint_dir=str(tmp_path / "stage2"), log_every=100)
    assert len(history) == 4 and all(math.isfinite(h["val"]) for h in history)
    trained = model.trained_heads()
    assert trained["track"] > 0 and trained["intensity"] > 0 and trained["storm_state"] > 0
    assert trained["enso"] == 0
    assert all(not p.requires_grad for p in model.blocks.parameters()), "backbone stays frozen"
    table = storm_scorecard(model, bank, (6, 12))
    assert "track km" in table and "6h" in table


def test_every_advertised_name_exists():
    """An export edit that silently missed has broken a published release before; this pins it."""
    import naturev1

    import ihelix

    for package in (naturev1, ihelix):
        missing = [name for name in package.__all__ if not hasattr(package, name)]
        assert not missing, f"{package.__name__}.__all__ lists names it does not define: {missing}"


def test_a_tie_with_persistence_is_not_skill():
    """234.954 against 234.956 is a tie; the verdict used to call it genuine skill."""
    from naturev1.wb2 import Score, Scorecard

    card = Scorecard([Score("z500", 6, 234.954, 0.97, 234.956, 1064.0),
                      Score("z500", 12, 361.709, 0.94, 361.714, 1066.0)])
    verdict = card.verdict()
    assert "not a forecast yet" in verdict and "ties it on 2" in verdict
    assert not any(score.beats_persistence for score in card.scores)

    better = Scorecard([Score("z500", 24, 300.0, 0.9, 400.0, 1000.0)])
    assert "skill against both baselines out to +24h (z500)" in better.verdict()
