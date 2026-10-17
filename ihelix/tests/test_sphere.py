# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
0.6: regular latitude-longitude arrays. Each test is a property that is either true of the arithmetic or not.
"""

import math

import pytest
import torch

from ihelix import (
    ConservativeRegrid,
    FieldGrid,
    Geometry,
    GridToMesh,
    IHelixConfig,
    IHelixField,
    SphericalConv2d,
    cell_edges,
    fibonacci_sphere,
    latlon_channels,
    sphere_pad,
)

DT = torch.float64


def axes(step, south=-90.0, north=90.0):
    lat = torch.arange(south + step / 2, north, step, dtype=DT)
    lon = torch.arange(-180.0 + step / 2, 180.0, step, dtype=DT)
    return lat, lon


def area_mean(field, lat, mask=None):
    weights = torch.cos(torch.deg2rad(lat)).view(-1, 1).expand_as(field)
    if mask is not None:
        weights = weights * mask
    return float((field * weights).sum() / weights.sum())


def smooth(lat, lon):
    la, lo = torch.deg2rad(lat).view(-1, 1), torch.deg2rad(lon).view(1, -1)
    return 250.0 + 20.0 * torch.cos(la) ** 2 + 5.0 * torch.sin(3 * lo) * torch.cos(la)


def test_cell_edges_clamp_at_the_pole_and_follow_descending_axes():
    edges = cell_edges(torch.tensor([90.0, 88.5, 87.0]), -90.0, 90.0)
    assert edges[0] == 90.0, "a pole row gets a half-width cap, not a cell past the pole"
    assert torch.allclose(edges[1:], torch.tensor([89.25, 87.75, 86.25], dtype=DT))


def test_conservative_regrid_preserves_the_global_area_mean():
    fine_lat, fine_lon = axes(0.5)
    coarse_lat, coarse_lon = axes(1.5)
    torch.manual_seed(0)
    field = torch.randn(fine_lat.numel(), fine_lon.numel(), dtype=DT) * 10 + 270
    regrid = ConservativeRegrid(fine_lat, fine_lon, coarse_lat, coarse_lon).to(DT)
    regrid.lat_matrix, regrid.lon_matrix = regrid.lat_matrix.double(), regrid.lon_matrix.double()
    out, coverage = regrid(field)
    assert out.shape == (coarse_lat.numel(), coarse_lon.numel())
    assert torch.allclose(coverage, torch.ones_like(coverage))
    assert abs(area_mean(out, coarse_lat) - area_mean(field, fine_lat)) < 1e-9, "conservative means conserved"


def test_non_integer_ratio_and_a_smooth_field_is_reproduced():
    fine_lat, fine_lon = axes(0.07 * 5, -70.0, 70.0)        # 0.35 degrees, band like GridSat
    coarse_lat, coarse_lon = axes(1.0, -60.0, 60.0)
    regrid = ConservativeRegrid(fine_lat, fine_lon, coarse_lat, coarse_lon)
    out, coverage = regrid(smooth(fine_lat, fine_lon).float())
    expected = smooth(coarse_lat, coarse_lon).float()
    assert (coverage > 0.999).all()
    assert (out - expected).abs().max() < 0.2, "a cell average of a smooth field is close to its centre value"


def test_missing_pixels_lower_coverage_not_the_value():
    fine_lat, fine_lon = axes(0.5)
    coarse_lat, coarse_lon = axes(2.0)
    field = torch.full((fine_lat.numel(), fine_lon.numel()), 280.0)
    field[: fine_lat.numel() // 2, :50] = float("nan")              # a hole
    out, coverage = ConservativeRegrid(fine_lat, fine_lon, coarse_lat, coarse_lon)(field)
    covered = coverage > 0
    assert torch.allclose(out[covered], torch.full_like(out[covered], 280.0)), "no bias toward zero in a gap"
    assert (coverage < 1).any() and torch.isnan(out[~covered]).all()


def test_a_band_source_leaves_the_poles_uncovered():
    band_lat, band_lon = axes(0.5, -70.0, 70.0)
    out_lat, out_lon = axes(5.0)
    _, coverage = ConservativeRegrid(band_lat, band_lon, out_lat, out_lon)(torch.ones(band_lat.numel(), band_lon.numel()))
    assert coverage[out_lat.abs() > 72.5].max() == 0 and coverage[out_lat.abs() < 67.5].min() > 0.999


def test_longitude_wraps_whatever_the_convention():
    lat = torch.arange(-89.5, 90.0, 1.0, dtype=DT)
    lon_east = torch.arange(0.5, 360.0, 1.0, dtype=DT)                 # 0..360
    lon_west = torch.arange(-179.0, 180.0, 2.0, dtype=DT)              # -180..180, coarser
    field = torch.cos(torch.deg2rad(lon_east)).view(1, -1).expand(lat.numel(), -1).float()
    out, coverage = ConservativeRegrid(lat, lon_east, lat, lon_west)(field)
    assert (coverage > 0.999).all()
    expected = torch.cos(torch.deg2rad(lon_west)).view(1, -1).expand_as(out)
    assert (out - expected).abs().max() < 1e-3


def test_moments_and_extremes_see_inside_the_cell():
    fine_lat, fine_lon = axes(0.25, -10.0, 10.0)
    coarse_lat, coarse_lon = axes(1.0, -10.0, 10.0)
    field = torch.full((fine_lat.numel(), fine_lon.numel()), 290.0)
    field[40, 400] = 190.0                                              # one overshooting top
    regrid = ConservativeRegrid(fine_lat, fine_lon, coarse_lat, coarse_lon)
    mean, spread, _ = regrid.moments(field)
    low, high = regrid.extremes(field)
    row, col = 10, 100
    assert low[row, col] == 190.0 and high[row, col] == 290.0
    assert spread[row, col] > 20.0 and spread[0, 0] < 1e-3
    assert 280.0 < mean[row, col] < 290.0
    assert (low[torch.arange(low.shape[0]) != row] == 290.0).all()


def test_sphere_pad_wraps_longitude_and_folds_across_the_pole():
    lat, lon = axes(30.0)                                                # 6 x 12
    field = torch.arange(lat.numel() * lon.numel(), dtype=DT).view(lat.numel(), lon.numel())
    padded = sphere_pad(field, 1, 1, "global")
    assert padded.shape == (8, 14)
    assert torch.equal(padded[1:-1, 0], field[:, -1]) and torch.equal(padded[1:-1, -1], field[:, 0])
    # Beyond the north pole (last row): the same row, half-way round.
    assert torch.equal(padded[-1, 1:-1], field[-1].roll(6))
    assert torch.equal(padded[0, 1:-1], field[0].roll(6))
    with_poles = torch.arange(5 * 4, dtype=DT).view(5, 4)
    folded = sphere_pad(with_poles, 1, 0, "poles")
    assert torch.equal(folded[0], with_poles[1].roll(2)), "the pole row is not repeated"


def test_spherical_conv_has_no_seam_at_the_date_line():
    torch.manual_seed(0)
    conv = SphericalConv2d(1, 3, 3, latitude="global").to(DT)
    lat, lon = axes(10.0)
    field = smooth(lat, lon).unsqueeze(0).unsqueeze(0)
    base = conv(field)
    shifted = conv(field.roll(7, -1))
    assert torch.allclose(shifted, base.roll(7, -1), atol=1e-12), "rolling longitude commutes with the conv"
    strided = SphericalConv2d(1, 2, 3, stride=2, latitude="band").to(DT)
    assert strided(field).shape == (1, 2, 9, 18)


def test_latlon_channels():
    channels = latlon_channels(torch.tensor([0.0, 90.0]), torch.tensor([0.0, 90.0, 180.0]))
    assert channels.shape == (4, 2, 3)
    assert torch.allclose(channels[0, 1], torch.ones(3)) and torch.allclose(channels[3, 0], torch.tensor([1.0, 0.0, -1.0]), atol=1e-6)


def test_grid_to_mesh_pools_every_pixel_once_and_reports_coverage():
    mesh = fibonacci_sphere(256, num_neighbours=8, cluster_size=32)
    lat, lon = axes(2.0, -70.0, 70.0)
    pool = GridToMesh(lat, lon, mesh)
    field = torch.full((lat.numel(), lon.numel()), 3.0)
    out = pool(field, extremes=True)
    covered = out["coverage"] > 0
    assert torch.allclose(out["mean"][covered], torch.full_like(out["mean"][covered], 3.0))
    polar = mesh.coords[:, 0].abs() > math.radians(80)
    assert (out["coverage"][polar] == 0).all(), "a band image says nothing about the poles"
    assert (out["coverage"][mesh.coords[:, 0].abs() < math.radians(50)] > 0.999).all()
    # Batch dimensions pass through, and a masked half lowers coverage without biasing the mean.
    batch = torch.stack([field, field * 2])
    mask = torch.ones_like(batch, dtype=torch.bool)
    mask[1, :, : lon.numel() // 2] = False
    two = pool(batch, mask=mask)
    assert two["mean"].shape == (2, 256)
    seen = two["coverage"][1] > 0
    assert torch.allclose(two["mean"][1][seen], torch.full_like(two["mean"][1][seen], 6.0))
    assert two["coverage"][1].sum() < two["coverage"][0].sum()


def test_index_attention_write_back_is_unchanged_and_compiles_without_a_mask():
    """The integer write-back reproduces the old boolean-mask one exactly, including padded clusters."""
    torch.manual_seed(0)
    coords = torch.rand(200, 2, dtype=DT)
    from ihelix import Axis

    grid = FieldGrid.from_points(coords, Geometry([Axis("linear"), Axis("linear")]), num_neighbours=8, cluster_size=32)
    assert grid.cluster_valid.logical_not().any(), "the case with padding is the one that matters"
    keep = grid.cluster_valid.reshape(-1)
    assert torch.equal(grid.scatter_points, grid.order[keep])
    model = IHelixField(IHelixConfig(in_channels=3, hidden_size=32, num_layers=2, num_heads=2, num_kv_heads=1,
                                     head_dim=16, min_radius=0.05, max_radius=0.2, landmark_dim=16,
                                     index_branching=2, index_beam_width=2, index_topk=2, use_temporal=False),
                        grid.geometry).to(DT).eval()
    values = torch.randn(1, 200, 3, dtype=DT)
    first = model(values, grid)
    # A grid saved before 0.6 has no plan; it is rebuilt on first use and gives the same answer.
    grid.scatter_points = grid.scatter_slots = None
    assert torch.equal(model(values, grid), first)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_regrid_runs_on_the_device_it_is_moved_to(device):
    lat, lon = axes(1.0, -20.0, 20.0)
    regrid = ConservativeRegrid(lat, lon, *axes(4.0, -20.0, 20.0)).to(device)
    out, _ = regrid(torch.ones(lat.numel(), lon.numel(), device=device))
    assert out.device.type == device
