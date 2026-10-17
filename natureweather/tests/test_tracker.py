# Copyright 2026 Nathan. Apache-2.0.
"""The vortex tracker: finds a low to a fraction of a grid cell, follows it, lets go when it fills."""

from __future__ import annotations

import math

import numpy as np
import torch
from ihelix import FieldGrid, Geometry

from naturev1.era5 import Normalizer
from naturev1.tracker import find_storms, track_vortex, vortex_centre

VARIABLES = ["mean_sea_level_pressure", "u_component_of_wind@850", "v_component_of_wind@850"]
NORMALIZER = Normalizer(VARIABLES, np.zeros(3, np.float32), np.ones(3, np.float32), np.zeros(3, np.float32))
LAT = torch.linspace(-90.0, 90.0, 121, dtype=torch.float64)
LON = torch.arange(240, dtype=torch.float64) * 1.5


def grid():
    lat, lon = torch.meshgrid(torch.deg2rad(LAT), torch.deg2rad(LON), indexing="ij")
    coords = torch.stack([lat.reshape(-1), lon.reshape(-1)], -1)
    return FieldGrid.source(coords, Geometry.globe(), weights=torch.cos(coords[:, 0]).clamp_min(1e-4))


def state_with_lows(lows):
    """Sea-level pressure (Pa) with Gaussian lows ``(lat, lon, depth_hpa)``, and a cyclonic 850 hPa vortex."""
    lat = LAT.view(-1, 1).expand(-1, 240)
    lon = LON.view(1, -1).expand(121, -1)
    pressure = torch.full((121, 240), 101300.0, dtype=torch.float64)
    u = torch.zeros_like(pressure)
    v = torch.zeros_like(pressure)
    for clat, clon, depth in lows:
        north = (lat - clat) * 111.2
        east = ((lon - clon + 180) % 360 - 180) * 111.2 * math.cos(math.radians(clat))
        r2 = north ** 2 + east ** 2
        bump = torch.exp(-r2 / (2 * 350.0 ** 2))
        pressure = pressure - depth * 100.0 * bump
        sense = 1.0 if clat >= 0 else -1.0                         # anticlockwise north, clockwise south
        u = u - sense * 0.05 * north * bump
        v = v + sense * 0.05 * east * bump
    return torch.stack([pressure, u, v], -1).reshape(-1, 3).float()


def test_finds_the_low_to_a_fraction_of_a_cell():
    state = state_with_lows([(18.3, 291.7, 20.0)]).unsqueeze(0)
    centre, depth, found = vortex_centre(state, VARIABLES, NORMALIZER, grid(), torch.tensor([[17.0, -70.0]]))
    assert bool(found[0]) and float(depth[0]) > 10
    assert abs(float(centre[0, 0]) - 18.3) < 0.3 and abs(float(centre[0, 1]) - (291.7 - 360)) < 0.3


def test_no_low_in_reach_is_not_a_low():
    state = state_with_lows([(18.0, 290.0, 20.0)]).unsqueeze(0)
    _, _, found = vortex_centre(state, VARIABLES, NORMALIZER, grid(), torch.tensor([[-30.0, 100.0]]))
    assert not bool(found[0])


def test_follows_a_moving_low_and_lets_go_when_it_fills():
    track = [(15.0 + 0.6 * k, 300.0 - 1.0 * k, 20.0 if k < 6 else 0.0) for k in range(1, 9)]
    states = torch.stack([state_with_lows([low]) for low in track]).unsqueeze(1)          # (S, 1, P, C)
    out = track_vortex(states, VARIABLES, NORMALIZER, grid(), torch.tensor([[15.0, -60.0]]), (6, 18, 30, 48))
    assert abs(float(out[0, 0, 0]) - 0.6) < 0.3 and abs(float(out[0, 0, 1]) + 1.0) < 0.3
    assert abs(float(out[0, 2, 0]) - 3.0) < 0.3 and abs(float(out[0, 2, 1]) + 5.0) < 0.3
    assert torch.isnan(out[0, 3]).all(), "the low filled at +36 h: no position after that"


def test_find_storms_sees_both_hemispheres_and_ignores_the_extratropics():
    state = state_with_lows([(15.0, 300.0, 15.0), (-14.0, 160.0, 12.0), (55.0, 330.0, 25.0)])
    found = find_storms(state, VARIABLES, NORMALIZER, grid())
    places = sorted((round(c["latitude"]), round(c["longitude"])) for c in found)
    assert places == [(-14, 160), (15, -60)], found
