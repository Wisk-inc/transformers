# Copyright 2026 Nathan. Apache-2.0.
"""
NatureV2 on a live storm (0.19): GFS into the training state, NHC's best track, the GOES storm inputs, and the trained
model run on all of it -- here on constructed GFS fields and a small model, with the downloads replaced.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from naturev1.era5 import Normalizer
from naturev1.model import NatureConfig, NatureV1
from naturev1.swarm import naturev2 as live
from naturev1.swarm.analysis import LAT_AXIS, LON_AXIS, G

from ihelix import fibonacci_sphere


VARIABLES = ["2m_temperature", "mean_sea_level_pressure", "surface_pressure", "u_component_of_wind@850",
             "v_component_of_wind@850", "u_component_of_wind@200", "v_component_of_wind@200",
             "specific_humidity@700", "toa_incident_solar_radiation_6hr", "land_sea_mask"]
MEANS = np.array([288, 101300, 98000, 0, 0, 0, 0, 0.004, 1.1e6, 0.3], np.float32)
STDS = np.array([12, 900, 3000, 8, 8, 15, 15, 0.003, 1.0e6, 0.4], np.float32)
WHEN = np.datetime64("2025-10-27T00", "h")
LAND = (np.abs(LAT_AXIS)[:, None] > 60) | (LON_AXIS[None, :] < 30)          # a crude continent


def fake_reader(calls=None):
    """Smooth GFS-like fields on the 1.5-degree nodes, with a cyclone at 17N 77W in the winds and pressure."""
    lat, lon = np.meshgrid(LAT_AXIS, LON_AXIS, indexing="ij")
    d2 = (lat - 17.0) ** 2 + ((lon - 283.0 + 180) % 360 - 180) ** 2
    vortex = np.exp(-d2 / 8.0)

    def reader(when, lead, keys):
        if calls is not None:
            calls.append((str(when), lead, len(keys)))
        out = {}
        for var, level in keys:
            p = float(level.split()[0]) if level.endswith(" mb") else 1000.0
            base = {"HGT": 7.0 * np.log(1013.0 / p) * 1000.0, "TMP": 230.0 + 0.06 * p, "SPFH": 1e-5 * p,
                    "VVEL": 0.0, "PRMSL": 101300.0 - 4000.0 * vortex, "PRES": 98000.0 - 3500.0 * vortex,
                    "PWAT": 40.0 + 20.0 * vortex, "LAND": LAND.astype(float), "APCP": 12.0 * vortex}.get(var, 0.0)
            if var == "UGRD":
                base = 5.0 - 30.0 * vortex * (lat - 17.0) / 3.0
            if var == "VGRD":
                base = 30.0 * vortex * ((lon - 283.0 + 180) % 360 - 180) / 3.0
            if (var, level) == ("TMP", "surface"):
                base = 302.0 - 0.2 * np.abs(lat)
            if (var, level) == ("HGT", "surface"):
                base = 500.0 * LAND
            if (var, level) == ("TMP", "2 m above ground"):
                base = 300.0 - 0.3 * np.abs(lat)
            out[(var, level)] = np.broadcast_to(np.asarray(base, np.float64), lat.shape).copy()
        return out

    return reader


def normalizer():
    return Normalizer(VARIABLES, MEANS, STDS, np.zeros(len(VARIABLES), np.float32))


def test_gfs_becomes_the_training_state_in_eras_units_and_layout(tmp_path):
    calls = []
    names = VARIABLES + ["geopotential@500", "sea_surface_temperature", "total_precipitation_6hr", "geopotential_at_surface",
                         "vertical_velocity@500"]
    state = live.gfs_state(WHEN, names, tmp_path, reader=fake_reader(calls))
    assert state.shape == (121, 240, len(names)) and state.dtype == np.float32
    get = lambda name: state[..., names.index(name)]  # noqa: E731
    assert np.allclose(get("geopotential@500"), 7.0 * np.log(1013.0 / 500.0) * 1000.0 * G, rtol=1e-6), "HGT x g"
    assert np.isnan(get("sea_surface_temperature")[LAND]).all() and np.isfinite(get("sea_surface_temperature")[~LAND]).all()
    i, j = int(np.argmin(np.abs(LAT_AXIS - 16.5))), int(np.argmin(np.abs(LON_AXIS - 283.5)))
    expected = 12.0 * np.exp(-((16.5 - 17.0) ** 2 + (283.5 - 283.0) ** 2) / 8.0) / 1000.0
    assert get("total_precipitation_6hr")[i, j] == pytest.approx(expected, rel=1e-5), "mm -> m, from the previous cycle"
    assert ("2025-10-26T18", 6, 1) in calls, "the 0-6 h rain of the cycle six hours earlier"
    sun = get("toa_incident_solar_radiation_6hr")
    assert sun.min() >= 0 and sun.max() > 5e6 and (sun == 0).mean() > 0.2, "day side lit, night side dark"
    assert get("geopotential_at_surface")[LAND].mean() == pytest.approx(500.0 * G, rel=1e-6)
    # ERA5's own constants override GFS's; the download is cached.
    era5_static = {"land_sea_mask": np.zeros((121, 240)), "geopotential_at_surface": np.full((121, 240), 9.0)}
    again = live.gfs_state(WHEN, names, tmp_path, static=era5_static, reader=fake_reader(calls))
    assert len(calls) == 2, "second call read the cache"
    assert np.isfinite(again[..., names.index("sea_surface_temperature")]).all(), "ERA5's mask says all sea here"
    assert (again[..., names.index("geopotential_at_surface")] == 9.0).all()


def test_the_best_track_keeps_wind_pressure_radii_and_stops_at_now():
    lines = []
    for t, (lat, lon, wind, mslp) in {"2025102618": ("170N", "0760W", 120, 950), "2025102700": ("172N", "0770W", 140, 930),
                                      "2025102706": ("175N", "0780W", 150, 915)}.items():
        for rad, quads in ((34, (150, 130, 90, 120)), (50, (80, 70, 50, 60)), (64, (40, 35, 25, 30))):
            lines.append(f"AL, 13, {t},   , BEST,   0, {lat}, {lon}, {wind}, {mslp}, HU, {rad}, NEQ, "
                         f"{quads[0]:4d}, {quads[1]:4d}, {quads[2]:4d}, {quads[3]:4d}, 1006,  250,  10,")
    lines.append("AL, 13, 2025102803,   , BEST,   0, 180N, 0790W, 130, 940, HU, 34, NEQ,  100, 100, 100, 100,")  # landfall fix
    track = live.bdeck_track("\n".join(lines), "al132025", "MELISSA", until="2025-10-27T00", basin="NA")
    assert len(track) == 2 and str(track.time[-1])[:13] == "2025-10-27T00"
    assert track.max_wind_kt[-1] == 140 and track.min_pressure_hpa[-1] == 930 and track.rmw_nmi[-1] == 10
    assert track.longitude[-1] == pytest.approx(-77.0) and track.latitude[-1] == pytest.approx(17.2)
    assert track.wind_radii_nmi[-1].tolist() == [[150, 130, 90, 120], [80, 70, 50, 60], [40, 35, 25, 30]]
    everything = live.bdeck_track("\n".join(lines))
    assert len(everything) == 3, "the 03 UTC landfall fix is not synoptic"


def test_goes_sees_only_where_a_goes_imager_looked():
    assert live.goes_source("2025-10-27", -77.0) == ("noaa-goes19", "east")
    assert live.goes_source("2021-09-01", -77.0) == ("noaa-goes16", "east")
    assert live.goes_source("2021-09-01", -150.0) == ("noaa-goes17", "west")
    assert live.goes_source("2024-08-01", -150.0) == ("noaa-goes18", "west")
    assert live.goes_source("2015-09-01", -77.0) is None, "GOES-13 had no ABI"
    assert live.goes_source("2025-10-27", 140.0) is None, "the western Pacific is Himawari's"


def _eye(pixels=256):
    r = np.hypot(*np.meshgrid(np.arange(pixels) - pixels / 2, np.arange(pixels) - pixels / 2)) * 0.07 * 111
    return np.where(r < 20, 280.0, np.where(r < 200, 195.0, 255.0))


def test_the_storm_inputs_are_the_patches_the_satellite_paths_were_trained_on():
    def fetch(t, band):
        if t < WHEN - np.timedelta64(8, "h"):
            return None                                            # the 9-hour-old images are missing
        return _eye() + (0.0 if band == 13 else 25.0)

    inputs = live.goes_storm_inputs(WHEN, (17.2, -77.0), fetch=fetch)
    codes, seen = inputs["satellite"]
    assert codes.shape == (1, 2, 256, 256) and codes.dtype == torch.uint8 and seen.tolist() == [[True, True]]
    loop = inputs["satnet"]
    assert loop["codes"].shape == (1, 4, 2, 256, 256) and loop["available"].tolist() == [[True, True, True, False]]
    assert loop["context"][0, 3] == pytest.approx(0.75) and loop["latitude"].item() == pytest.approx(17.2)
    from naturev1.stormsat import decode_bt

    assert float(np.nanmin(decode_bt(codes[0, 0].numpy()))) == pytest.approx(195.0, abs=1.0)


TINY = {"analysis_channels": 12, "state_channels": 8, "hidden_size": 32, "num_layers": 2, "num_heads": 2,
        "num_kv_heads": 1, "head_dim": 16, "intermediate_size": 64, "latent_points": 96, "latent_neighbours": 12,
        "latent_cluster": 16, "landmark_dim": 16, "index_branching": 2, "index_beam_width": 2, "index_topk": 2,
        "recurrent_heads": 2, "recurrent_head_dim": 16, "history_frames": 2, "min_radius_km": 600.0,
        "max_radius_km": 2000.0, "encode_radius_km": 900.0, "lead_times_hours": (6, 12, 24)}


def test_naturev2_forecasts_a_live_storm_from_gfs_and_goes(tmp_path):
    from naturev1.sattrain import staged_grid
    from naturev1.state import StateStepper, channel_roles

    torch.manual_seed(0)
    model = NatureV1(NatureConfig(**TINY, satnet_embed=64, ri2=True, hemisphere_frame=True, storm_ocean=True,
                                  storm_ir=True), fibonacci_sphere(96, num_neighbours=12, cluster_size=16)).eval()
    model.mark_trained(dict.fromkeys(("track", "intensity", "ri", "landfall", "state", "satnet", "ri2"), 1.0))
    norm = normalizer()
    window = live.gfs_window(WHEN, VARIABLES, norm, 12, history=2, cache=tmp_path, reader=fake_reader())
    assert window["analysis"].shape == (1, 2, 121 * 240, 12) and window["calendar"].shape[:2] == (1, 2)
    assert torch.isfinite(window["analysis"]).all()
    grid = model._aligned(staged_grid({"shape": [1, 121, 240, len(VARIABLES)]}))
    stepper = StateStepper(channel_roles(VARIABLES, 12), torch.full((12,), 0.5), norm, grid,
                           VARIABLES.index("surface_pressure"))
    lines = [f"AL, 13, {t},   , BEST,   0, {lat}, {lon}, 140, 930, HU, 34, NEQ, 150, 130, 90, 120, 1006, 250, 10,"
             for t, lat, lon in (("2025102612", "168N", "0750W"), ("2025102618", "170N", "0760W"),
                                 ("2025102700", "172N", "0770W"))]
    track = live.bdeck_track("\n".join(lines), "al132025", "MELISSA", until=WHEN)
    satellite = live.goes_storm_inputs(WHEN, (17.2, -77.0), fetch=lambda t, band: _eye())
    result = live.naturev2_forecast(model, window, grid, VARIABLES, norm, stepper, track, WHEN, satellite,
                                    precision="fp32")
    bundle = result["bundle"]
    assert isinstance(bundle["track_scenarios"], list) and isinstance(bundle["intensity"], list)
    assert result["inputs"] == {"satellite": True, "satnet": True, "ocean": True}
    assert result["rollout"]["rollout_vortex"].shape == (3, 2) and np.isfinite(result["rollout"]["rollout_vortex"]).all()
    track_now = live.naturev2_track(result)
    assert track_now.shape == (3, 2)
    text = live.describe_naturev2(result, swarm_track=np.tile([[17.5, -78.0]], (2, 1)), swarm_leads=(6, 24))
    assert "NatureV2 track" in text and "full-state rollout" in text and "vs the swarm" in text
    gaps = text.split("vs the swarm consensus (km):** ")[1].split("  \n")[0]
    assert gaps.startswith("6 h ") and " 24 h " in gaps and "12 h" not in gaps, "matched by forecast hour"


def test_compare_to_era5_is_zero_for_the_same_state_and_sees_a_bias(tmp_path):
    norm = normalizer()
    era5 = live.gfs_state(WHEN, VARIABLES, tmp_path, reader=fake_reader())
    same = live.compare_to_era5(era5, era5, norm)
    assert same["median"]["rmse"] == pytest.approx(0.0, abs=1e-6) and same["median"]["correlation"] == pytest.approx(1.0)
    warm = era5.copy()
    warm[..., VARIABLES.index("2m_temperature")] += 1.2
    scored = live.compare_to_era5(warm, era5, norm)["2m_temperature"]
    assert scored["rmse"] == pytest.approx(0.1, rel=0.01) and scored["correlation"] == pytest.approx(1.0)
