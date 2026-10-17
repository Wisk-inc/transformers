# Copyright 2026 Nathan. Apache-2.0.
"""
Storm-centred satellite patches: every source must land the same storm on the same pixels.

Offline: the HURSAT tarball, the GridSat field and the GOES scene are synthetic, written in the real
formats, with a temperature that is a known function of position -- so a patch that is shifted, flipped
or wrapped the wrong way across the date line shows up as a wrong number, not as a plausible picture.
"""

from __future__ import annotations

import tarfile

import numpy as np
import pytest
import torch

xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")

from naturev1.besttrack import Track  # noqa: E402
from naturev1.satellite import fixed_grid_to_latlon  # noqa: E402
from naturev1.stormsat import (  # noqa: E402
    PATCH_PIXELS, BT_RANGE_K, HursatArchive, crop_regular, decode_bt, encode_bt, goes_patch,
    latlon_to_fixed_grid, patch_axes, satellite_bank,
)


def field_of(lat, lon):
    """A brightness temperature that encodes where it is: 250 K + 1 K per degree north + 0.1 K per degree east."""
    return 250.0 + lat + 0.1 * ((lon + 180.0) % 360.0 - 180.0)


def test_patch_axes_are_centred_and_wrap():
    lats, lons = patch_axes((20.0, 179.95))
    assert lats.size == PATCH_PIXELS and abs(lats.mean() - 20.0) < 1e-9
    assert lons.min() >= -180.0 and lons.max() < 180.0, "longitudes wrap into -180..180"


def test_crop_a_global_grid_across_the_date_line():
    lat = np.arange(-70.0, 70.0, 0.07)
    lon = np.arange(-180.0, 180.0, 0.07)
    grid_lon, grid_lat = np.meshgrid(lon, lat)
    patch = crop_regular(lat, lon, field_of(grid_lat, grid_lon).astype(np.float32), (15.0, 179.9))
    want_lat, want_lon = patch_axes((15.0, 179.9))
    expected = field_of(want_lat[:, None], want_lon[None, :])
    assert np.isfinite(patch).all()
    assert np.nanmax(np.abs(patch - expected)) < 0.1, "every pixel must come from its own place"


def test_crop_outside_a_regional_grid_is_missing_not_clamped():
    lat = np.arange(10.0, 31.0, 0.07)
    lon = np.arange(-100.0, -79.0, 0.07)
    field = np.full((lat.size, lon.size), 260.0, np.float32)
    patch = crop_regular(lat, lon, field, (29.0, -80.0))          # the storm near the grid's corner
    assert np.isnan(patch).any() and np.isfinite(patch).any()


def test_byte_encoding_keeps_two_thirds_of_a_kelvin_and_marks_missing():
    values = np.array([170.0, 200.5, 299.9, np.nan, 400.0, 100.0], np.float32)
    back = decode_bt(encode_bt(values))
    assert np.allclose(back[:3], values[:3], atol=0.34)
    assert np.isnan(back[3]), "missing stays missing"
    assert back[4] == pytest.approx(BT_RANGE_K[1], abs=0.5) and back[5] == pytest.approx(BT_RANGE_K[0], abs=0.5)
    tensor = decode_bt(torch.from_numpy(encode_bt(values)))
    assert torch.isnan(tensor[3]) and abs(float(tensor[1]) - 200.5) < 0.34


def test_goes_forward_projection_inverts_the_existing_one():
    lat = np.array([25.0, 10.0, -20.0, 45.0, 0.0])
    lon = np.array([-77.0, -40.0, -60.0, -100.0, -75.0])
    x, y = latlon_to_fixed_grid(lat, lon, -75.0)
    back_lat, back_lon = fixed_grid_to_latlon(x, y, -75.0)
    assert np.allclose(back_lat, lat, atol=1e-9) and np.allclose(back_lon, lon, atol=1e-9)
    hidden_x, _ = latlon_to_fixed_grid(np.array([0.0]), np.array([105.0]), -75.0)
    assert np.isnan(hidden_x[0]), "the far side of the Earth is not visible"


def _goes_file(path):
    x = np.linspace(-0.151844, 0.151844, 1200)
    y = np.linspace(0.151844, -0.151844, 1200)
    mesh_x, mesh_y = np.meshgrid(x, y)
    lat, lon = fixed_grid_to_latlon(mesh_x, mesh_y, -75.0)
    cmi = np.where(np.isfinite(lat), field_of(np.nan_to_num(lat), np.nan_to_num(lon)), np.nan).astype(np.float32)
    projection = xr.DataArray(0, attrs={"longitude_of_projection_origin": -75.0, "perspective_point_height": 35786023.0,
                                        "semi_major_axis": 6378137.0, "semi_minor_axis": 6356752.31414})
    xr.Dataset({"CMI": (("y", "x"), cmi), "goes_imager_projection": projection},
               coords={"x": x, "y": y}).to_netcdf(path)


def test_goes_scene_resampled_onto_the_storm_patch(tmp_path):
    path = tmp_path / "C13.nc"
    _goes_file(path)
    patch = goes_patch(path, (26.5, -77.0))
    want_lat, want_lon = patch_axes((26.5, -77.0))
    expected = field_of(want_lat[:, None], want_lon[None, :])
    good = np.isfinite(patch)
    assert good.mean() > 0.99
    # 1200 pixels across the disk is ~10 km each; the patch is 8 km, so allow about one source pixel.
    assert np.median(np.abs(patch[good] - expected[good])) < 0.15


def _hursat_tarball(directory, sid="2005236N23285", name="KATRINA"):
    lat = np.arange(19.0, 40.0, 0.07).astype(np.float32)
    lon = np.arange(-100.1, -79.1, 0.07).astype(np.float32)
    grid_lon, grid_lat = np.meshgrid(lon, lat)
    members = []
    for angle, satellite, offset in ((38, "GOE-12", 0.0), (60, "GOE-10", 5.0)):
        ir = (field_of(grid_lat, grid_lon) + offset)[None].astype(np.float32)
        file = directory / f"{sid}.{name}.2005.08.29.1200.{angle}.{satellite}.097.hursat-b1.v06.nc"
        xr.Dataset({"IRWIN": (("htime", "lat", "lon"), ir)}, coords={"lat": lat, "lon": lon}).to_netcdf(file)
        members.append(file)
    archive = directory / f"{sid}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for file in members:
            tar.add(file, arcname=file.name)
    return archive


def test_hursat_picks_the_most_direct_view_and_crops_on_the_storm(tmp_path):
    _hursat_tarball(tmp_path)
    archive = HursatArchive(tmp_path, {"AL122005": "2005236N23285"})
    patch = archive.patch("AL122005", np.datetime64("2005-08-29T12:00"), (29.5, -89.6))
    want_lat, want_lon = patch_axes((29.5, -89.6))
    expected = field_of(want_lat[:, None], want_lon[None, :])
    good = np.isfinite(patch)
    assert good.mean() > 0.95
    assert np.nanmax(np.abs(patch[good] - expected[good])) < 0.1, "the 38-degree view, not the 60-degree one (+5 K)"
    assert archive.patch("AL122005", np.datetime64("2005-08-30T12:00"), (29.5, -89.6)) is None, "no image then"
    assert archive.patch("AL992005", np.datetime64("2005-08-29T12:00"), (29.5, -89.6)) is None, "unknown storm"
    many = archive.storm_patches("AL122005", [(np.datetime64("2005-08-29T12:00"), (29.5, -89.6)),
                                              (np.datetime64("2005-08-29T13:00"), (30.0, -89.0)),
                                              (np.datetime64("2005-09-02T12:00"), (35.0, -80.0))])
    assert many[0] is not None and many[1] is not None and many[2] is None, "one pass, nearest image within 90 min"


class _Target:
    def __init__(self, track, point):
        self.track, self.point = track, point


def _track(storm_id, start, positions):
    times = np.array([np.datetime64(start, "s") + np.timedelta64(6 * i, "h") for i in range(len(positions))])
    n = len(positions)
    return Track(storm_id=storm_id, name="TEST", time=times,
                 latitude=np.array([p[0] for p in positions], np.float32),
                 longitude=np.array([p[1] for p in positions], np.float32),
                 max_wind_kt=np.full(n, 100.0, np.float32), min_pressure_hpa=np.full(n, 950.0, np.float32),
                 status=np.array(["HU"] * n), rmw_nmi=np.full(n, np.nan, np.float32),
                 wind_radii_nmi=np.full((n, 3, 4), np.nan, np.float32), record=np.array([""] * n))


class _FakeHursat:
    def __init__(self):
        self.calls = []

    def storm_patches(self, storm, requests):
        out = []
        for when, centre in requests:
            self.calls.append((storm, str(when), centre))
            lats, lons = patch_axes(centre)
            out.append(field_of(lats[:, None], lons[None, :]).astype(np.float32))
        return out


class _FakeGridsat:
    def patches(self, requests, progress=True):
        return [None if str(when).startswith("2017-09-01T00") else
                field_of(patch_axes(c)[0][:, None], patch_axes(c)[1][None, :]).astype(np.float32)
                for when, c in requests]


def test_satellite_bank_aligns_with_the_samples_and_is_reused(tmp_path):
    old = _track("AL122005", "2005-08-29T00:00", [(26.0, -88.0), (27.0, -89.0), (28.0, -89.5)])
    new = _track("AL152017", "2017-09-01T00:00", [(15.0, -50.0), (15.5, -52.0)])
    targets = [_Target(old, 0), _Target(old, 2), _Target(new, 1), _Target(new, 0)]
    hursat = _FakeHursat()
    bank = satellite_bank(targets, tmp_path / "sat.pt", hursat=hursat, gridsat=_FakeGridsat(), progress=False)
    assert bank["ir"].shape == (4, 2, PATCH_PIXELS, PATCH_PIXELS) and bank["ir"].dtype == torch.uint8
    available = bank["available"].tolist()
    assert available[0] == [True, False], "a storm's first fix has no image six hours earlier"
    assert available[1] == [True, True]
    assert available[2] == [True, False], "GridSat had no file for the earlier time"
    assert available[3] == [False, False]
    assert bank["source"][:3] == ["HURSAT-B1", "HURSAT-B1", "GridSat-B1"]
    # the six-hours-earlier image is centred on where the storm was then, not where it is now
    earlier = decode_bt(bank["ir"][1, 1].numpy())
    lats, lons = patch_axes((27.0, -89.0))
    assert np.nanmax(np.abs(earlier - field_of(lats[:, None], lons[None, :]))) < 0.4
    calls = len(hursat.calls)
    again = satellite_bank(targets, tmp_path / "sat.pt", hursat=hursat, gridsat=_FakeGridsat(), progress=False)
    assert len(hursat.calls) == calls and torch.equal(again["ir"], bank["ir"]), "same samples: reused"
    satellite_bank(targets[:3], tmp_path / "sat.pt", hursat=hursat, gridsat=_FakeGridsat(), progress=False)
    assert len(hursat.calls) > calls, "different samples: rebuilt"
