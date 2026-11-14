# Copyright 2026 Nathan. Apache-2.0.
"""The vortex detector and tracker on synthetic pressure fields with a known storm."""

import numpy as np

from pluvium.cyclone.detect import candidates, depth, find_centre, refine
from pluvium.cyclone.geodesy import distance_km
from pluvium.cyclone.tracker import TrackerConfig, track_member, track_members
from pluvium.cyclone.tracks import TrackSet

LAT = np.arange(5.0, 45.01, 0.25)
LON = np.arange(-100.0, -40.0 + 0.01, 0.25)


def low(clat, clon, deep=35.0, radius_km=180.0, background=1012.0):
    d = distance_km(clat, clon, LAT[:, None], LON[None, :])
    return background - deep * np.exp(-0.5 * (d / radius_km) ** 2)


def path(k, steps=10):
    """A storm moving west-north-west then recurving: positions at 12-hour steps."""
    t = np.arange(1, steps + 1)
    return 15.0 + 1.1 * t + 0.03 * t ** 2, -60.0 - 2.2 * t + 0.25 * t ** 2 + 0.0 * k


def test_centre_is_found_below_the_grid_spacing():
    field = low(17.37, -71.62)
    found = find_centre(field, LAT, LON, 17.0, -71.0, radius_km=300.0)
    assert found is not None
    assert distance_km(found["lat"], found["lon"], 17.37, -71.62) < 8.0          # the grid spacing is about 27 km
    assert abs(found["pressure"] - (1012.0 - 35.0)) < 0.5 and found["depth"] > 20.0
    i, j = int(np.argmin(np.abs(LAT - 17.25))), int(np.argmin(np.abs(LON + 71.5)))
    lat, lon, p = refine(field, LAT, LON, i, j)
    assert abs(lat - 17.37) < 0.1 and abs(lon + 71.62) < 0.1 and p <= field[i, j] + 1e-9


def test_the_nearest_closed_low_wins_over_a_deeper_one_and_ripples_are_ignored():
    field = np.minimum(low(18.0, -70.0, deep=20.0), low(24.0, -66.0, deep=45.0))    # a deeper low 800 km away
    field = field - 0.4 * np.cos(np.radians(LON))[None, :]                           # a gentle gradient
    found = candidates(field, LAT, LON, 18.3, -70.4, radius_km=1200.0)
    assert len(found) == 2 and distance_km(found[0]["lat"], found[0]["lon"], 18.0, -70.0) < 15.0
    flat = 1012.0 - 0.5 * np.exp(-0.5 * (distance_km(18.0, -70.0, LAT[:, None], LON[None, :]) / 60.0) ** 2)
    assert find_centre(flat, LAT, LON, 18.0, -70.0, radius_km=300.0, min_depth=1.5) is None
    assert depth(low(18.0, -70.0), LAT, LON, 18.0, -70.0, 977.0) > 25.0


def test_a_storm_is_followed_and_a_dissipating_one_ends():
    lats, lons = path(0)
    msl = np.stack([low(a, b) for a, b in zip(lats, lons)])
    steps = 12 * np.arange(1, 11)
    start = {"lat": 15.0, "lon": -60.0, "east_kmh": -18.0, "north_kmh": 9.0}
    track = track_member(msl, LAT, LON, steps, start)
    err = distance_km(track[:, 0], track[:, 1], lats, lons)
    assert np.isfinite(track).all() and err.max() < 12.0
    msl[6:] = 1012.0                                                    # the model loses the storm after step 6
    track = track_member(msl, LAT, LON, steps, start, TrackerConfig(bridge=1))
    assert np.isfinite(track[:6, 0]).all() and np.isnan(track[6:, 0]).all()


def test_one_missing_step_is_bridged():
    lats, lons = path(0)
    msl = np.stack([low(a, b) for a, b in zip(lats, lons)])
    msl[4] = 1012.0
    track = track_member(msl, LAT, LON, 12 * np.arange(1, 11), {"lat": 15.0, "lon": -60.0, "east_kmh": -18.0,
                                                                 "north_kmh": 9.0})
    assert np.isnan(track[4, 0]) and np.isfinite(track[5:, 0]).all()
    assert distance_km(track[9, 0], track[9, 1], lats[9], lons[9]) < 15.0


def test_members_and_their_mean():
    rng = np.random.default_rng(0)
    lats, lons = path(0)
    members = []
    for _ in range(7):
        dlat, dlon = rng.normal(0, 0.5, 10).cumsum() * 0.3, rng.normal(0, 0.5, 10).cumsum() * 0.3
        members.append(np.stack([low(a, b) for a, b in zip(lats + dlat, lons + dlon)]))
    msl = np.stack(members)
    msl[3, 5:] = np.nan                                                # a member whose fields stop early
    steps = 12 * np.arange(1, 11)
    tr = track_members(msl, LAT, LON, steps, {"lat": 15.0, "lon": -60.0, "east_kmh": -18.0, "north_kmh": 9.0})
    assert tr.shape == (7, 10, 3) and np.isnan(tr[3, 5:, 0]).all() and np.isfinite(tr[3, :5, 0]).all()
    ts = TrackSet("AL992025", 0, "ens", steps, tr[..., 0], tr[..., 1], tr[..., 2])
    mlat, mlon = ts.mean()
    assert distance_km(mlat[-1], mlon[-1], lats[-1], lons[-1]) < 120.0
    assert np.isfinite(ts.spread_km()).all() and ts.spread_km()[-1] > ts.spread_km()[0]
    assert set(ts.points()) == set(int(s) for s in steps)
    few = TrackSet("AL992025", 0, "ens", steps, np.where(np.arange(7)[:, None] < 2, tr[..., 0], np.nan), tr[..., 1], tr[..., 2])
    assert np.isnan(few.mean()[0]).all()                              # two members of seven are not a mean
