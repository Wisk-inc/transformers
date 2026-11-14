# Copyright 2026 Nathan. Apache-2.0.
"""Geometry on the sphere for storm tracks: distances, bearings, displacements and means."""

import numpy as np

from pluvium.cyclone.geodesy import (
    KM_PER_NM,
    along_cross,
    bearing_deg,
    destination,
    distance_km,
    from_along_cross,
    mean_position,
    offset_km,
    shift,
)


def test_distance_and_bearing_match_known_values():
    assert abs(distance_km(0, 0, 0, 1) - 111.195) < 0.01                    # one degree of longitude at the equator
    assert abs(distance_km(18.0, -76.8, 25.76, -80.19) - 931.5) < 5.0       # Kingston to Miami
    assert abs(bearing_deg(0, 0, 1, 0)) < 1e-6 and abs(bearing_deg(0, 0, 0, 1) - 90.0) < 1e-6
    assert abs(KM_PER_NM - 1.852) < 1e-12


def test_destination_inverts_distance_and_bearing():
    lat, lon = destination(17.0, -77.0, 37.0, 500.0)
    assert abs(distance_km(17.0, -77.0, lat, lon) - 500.0) < 1e-6
    assert abs(bearing_deg(17.0, -77.0, lat, lon) - 37.0) < 1e-6


def test_shift_inverts_offset():
    east, north = offset_km(20.0, -60.0, 23.5, -64.0)
    lat, lon = shift(20.0, -60.0, east, north)
    assert abs(lat - 23.5) < 1e-6 and abs(lon + 64.0) < 1e-6


def test_mean_position_handles_the_date_line_and_missing_members():
    lat, lon = mean_position(np.array([10.0, 10.0]), np.array([179.0, -179.0]))
    assert abs(lat - 10.0) < 0.01 and abs(abs(lon) - 180.0) < 1e-6          # not longitude 0
    lat, lon = mean_position(np.array([10.0, np.nan, 12.0]), np.array([-60.0, np.nan, -60.0]))
    assert abs(lat - 11.0) < 0.01 and abs(lon + 60.0) < 1e-9
    lat, lon = mean_position(np.array([np.nan]), np.array([np.nan]))
    assert np.isnan(lat) and np.isnan(lon)
    lat, _ = mean_position(np.array([10.0, 20.0]), np.array([-60.0, -60.0]), weights=np.array([3.0, 1.0]))
    assert abs(lat - 12.5) < 0.05


def test_along_and_cross_track_errors_have_the_right_signs():
    # a storm heading north (0 degrees): a forecast further north is ahead, one further east is to the right
    a, c = along_cross(21.0, -70.0, 20.0, -70.0, 0.0)
    assert abs(a - 111.2) < 0.5 and abs(c) < 0.5
    a, c = along_cross(20.0, -69.0, 20.0, -70.0, 0.0)
    assert abs(a) < 2.0 and c > 100.0
    lat, lon = from_along_cross(20.0, -70.0, 150.0, -80.0, 45.0)
    a, c = along_cross(lat, lon, 20.0, -70.0, 45.0)
    assert abs(a - 150.0) < 1.0 and abs(c + 80.0) < 1.0
