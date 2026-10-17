# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Regular latitude-longitude arrays, handled the way the sphere wants them handled.

The rest of this package never sees an array: it sees points, distances and weights. But the data does
not arrive that way. Satellite archives and reanalyses come as regular latitude-longitude arrays -- a
global infrared mosaic is 2000 x 5143 pixels every three hours -- and a model that reads them has to do
three things to them before any attention runs. Each has a standard wrong way to do it:

* **Change resolution.** Bilinear interpolation from 0.07 degrees to 0.25 samples one source pixel in
  thirteen and throws the rest away, which aliases every convective cell smaller than a grid box into
  noise. :class:`ConservativeRegrid` averages *every* source pixel into the target cells it overlaps,
  weighted by the area of the overlap, so the coarse field has the same area-mean as the fine one -- and
  it carries a mask through, so a gap in the coverage lowers a cell's coverage instead of dragging its
  value toward zero. It also reports what an average hides: the coldest and warmest pixel in each cell,
  and the spread, which for infrared is the difference between a field of cumulus and one overshooting
  thunderstorm top.
* **Convolve.** ``nn.Conv2d`` with zero padding puts a wall at the date line and another at each pole.
  :class:`SphericalConv2d` pads by wrapping longitude around and by folding across the pole, where the
  row beyond the north pole is the row just below it on the opposite meridian. A storm at 179.9 E and
  one at 179.9 W are then the same storm, as they are in reality.
* **Get onto the mesh.** :class:`GridToMesh` assigns every pixel to its nearest mesh point once, and from
  then on pools an image onto the mesh with one indexed sum -- mean, extremes and coverage per mesh
  point -- at the cost of reading the array, no matter how fine it is.

All three are static: the geometry is worked out at construction, and the forward pass is matrix
products and indexed reductions with fixed shapes, which is what a compiled graph on a GPU wants.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def cell_edges(centres, lower: float | None = None, upper: float | None = None) -> torch.Tensor:
    """
    ``(n + 1,)`` cell boundaries for ``(n,)`` regularly or irregularly spaced cell centres, in degrees.

    Interior edges sit halfway between neighbours; the two outer edges sit half a spacing beyond the end
    centres, then clamped to ``[lower, upper]`` -- which is how a latitude grid whose first row is the
    pole itself gets a half-width polar cap instead of a cell that pokes past the pole. Works for
    descending centres too (ERA5 stores latitude north to south).
    """
    centres = torch.as_tensor(centres, dtype=torch.float64).reshape(-1)
    if centres.numel() == 1:
        raise ValueError("a single centre has no spacing to infer edges from")
    middle = (centres[1:] + centres[:-1]) / 2
    first = centres[0] - (middle[0] - centres[0])
    last = centres[-1] + (centres[-1] - middle[-1])
    edges = torch.cat([first.view(1), middle, last.view(1)])
    if lower is not None or upper is not None:
        edges = edges.clamp(lower if lower is not None else -math.inf, upper if upper is not None else math.inf)
    return edges


def _intervals(edges: torch.Tensor) -> torch.Tensor:
    """``(n, 2)`` (low, high) per cell, whichever way the edges run."""
    return torch.stack([torch.minimum(edges[:-1], edges[1:]), torch.maximum(edges[:-1], edges[1:])], -1)


def _overlap(target: torch.Tensor, source: torch.Tensor, period: float | None = None) -> torch.Tensor:
    """
    ``(T, S)`` length of the overlap of every target interval with every source interval.

    With a ``period`` the line is a circle: a source cell at 359.9 overlaps a target cell at -0.1. Three
    shifts suffice because both sets of intervals are reduced to one turn first.
    """
    if period is None:
        shifts = (0.0,)
    else:
        offset = torch.floor(source[:1, 0] / period) * period - torch.floor(target[:1, 0] / period) * period
        source = source - offset
        shifts = (-period, 0.0, period)
    total = torch.zeros(target.shape[0], source.shape[0], dtype=torch.float64)
    for shift in shifts:
        low = torch.maximum(target[:, None, 0], source[None, :, 0] + shift)
        high = torch.minimum(target[:, None, 1], source[None, :, 1] + shift)
        total = total + (high - low).clamp_min(0.0)
    return total


def _nearest_cell(centres: torch.Tensor, intervals: torch.Tensor, period: float | None = None) -> torch.Tensor:
    """For every source centre, the target cell containing it (or the nearest, past the ends)."""
    middle = intervals.mean(-1)
    distance = (centres[:, None] - middle[None, :]).abs()
    if period is not None:
        distance = torch.minimum(distance % period, period - distance % period)
    return distance.argmin(-1)


class ConservativeRegrid(nn.Module):
    """
    Area-conservative regridding between two regular latitude-longitude grids, with a mask carried through.

    A regular grid's cells are rectangles in (sin latitude, longitude), so the area of the overlap of two
    cells factorises into a latitude part and a longitude part, and so does the whole regridding:

        target = A_lat  @  source  @  A_lon^T

    where ``A_lat[t, s]`` is the share of target row ``t``'s area (in ``sin(latitude)``, which is what
    area is proportional to) covered by source row ``s``, and ``A_lon`` the same in longitude, which wraps.
    Two small dense matrices instead of one enormous sparse one: 0.07 to 0.25 degrees globally is a
    560 x 2000 and a 1440 x 5143 matrix, and the whole operation is two matrix products.

    Missing data is handled the conservative way. The masked field and the mask itself are both
    regridded; their ratio is the mean over the part of each target cell that was observed, and the
    mask's regridded value is the observed share of the cell -- the ``coverage``. A satellite gap then
    shows up as coverage below one, never as a cold or warm bias.

    Args:
        source_lat, source_lon: centres of the source rows and columns, degrees.
        target_lat, target_lon: centres of the target rows and columns, degrees.
        periodic: whether longitude wraps. True for global grids; a regional source simply leaves the
            target cells outside it with zero coverage either way.
    """

    def __init__(self, source_lat, source_lon, target_lat, target_lon, periodic: bool = True) -> None:
        super().__init__()
        source_lat_edges = cell_edges(source_lat, -90.0, 90.0)
        target_lat_edges = cell_edges(target_lat, -90.0, 90.0)
        source_lon_edges, target_lon_edges = cell_edges(source_lon), cell_edges(target_lon)
        sine = (lambda edges: torch.sin(torch.deg2rad(edges)))
        target_rows, source_rows = _intervals(sine(target_lat_edges)), _intervals(sine(source_lat_edges))
        target_cols, source_cols = _intervals(target_lon_edges), _intervals(source_lon_edges)
        period = 360.0 if periodic else None
        lat = _overlap(target_rows, source_rows) / (target_rows[:, 1] - target_rows[:, 0]).clamp_min(1e-15)[:, None]
        lon = _overlap(target_cols, source_cols, period) / (target_cols[:, 1] - target_cols[:, 0]).clamp_min(1e-15)[:, None]
        # Persistent=False: these are a pure function of the axes, rebuilt in microseconds, and a 30 MB
        # matrix has no business inside a checkpoint.
        self.register_buffer("lat_matrix", lat.float(), persistent=False)
        self.register_buffer("lon_matrix", lon.float(), persistent=False)
        # Which target cell each source pixel's centre falls in -- for extremes, which do not average.
        source_lat_t = torch.as_tensor(source_lat, dtype=torch.float64).reshape(-1)
        source_lon_t = torch.as_tensor(source_lon, dtype=torch.float64).reshape(-1)
        lat_index = _nearest_cell(torch.sin(torch.deg2rad(source_lat_t)), target_rows)
        lon_index = _nearest_cell(source_lon_t, target_cols, period)
        # A source pixel that lies wholly outside every target cell (a band source read onto a smaller
        # region, say) must not be pinned to the edge cell.
        inside_lat = (lat[lat_index, torch.arange(lat.shape[1])] > 0)
        inside_lon = (lon[lon_index, torch.arange(lon.shape[1])] > 0)
        self.register_buffer("lat_index", torch.where(inside_lat, lat_index, -1), persistent=False)
        self.register_buffer("lon_index", torch.where(inside_lon, lon_index, -1), persistent=False)
        self.source_shape = (lat.shape[1], lon.shape[1])
        self.target_shape = (lat.shape[0], lon.shape[0])

    def extra_repr(self) -> str:
        return f"{self.source_shape[0]}x{self.source_shape[1]} -> {self.target_shape[0]}x{self.target_shape[1]}"

    def _check(self, values: torch.Tensor) -> None:
        if tuple(values.shape[-2:]) != self.source_shape:
            raise ValueError(f"expected (..., {self.source_shape[0]}, {self.source_shape[1]}), "
                             f"got {tuple(values.shape)}")

    def _apply_matrices(self, field: torch.Tensor) -> torch.Tensor:
        lat = self.lat_matrix.to(field.dtype)
        lon = self.lon_matrix.to(field.dtype)
        return torch.matmul(torch.matmul(lat, field), lon.transpose(0, 1))

    def _prepare(self, values: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        self._check(values)
        values = values.to(self.lat_matrix.device)
        compute = values.dtype if values.is_floating_point() and values.dtype != torch.float16 else torch.float32
        values = values.to(compute)
        valid = torch.isfinite(values)
        if mask is not None:
            valid = valid & mask.to(values.device).bool()
        return torch.where(valid, values, torch.zeros_like(values)), valid.to(compute)

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None = None,
                min_coverage: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Regrid ``(..., H, W)`` to ``(..., H', W')``.

        Args:
            values: source field; NaN counts as missing.
            mask: optional, True where observed.
            min_coverage: target cells observed over less than this share of their area come back NaN.

        Returns:
            ``(mean, coverage)``: the area mean over the observed part of each target cell, and the
            observed share of each target cell in ``[0, 1]``.
        """
        filled, valid = self._prepare(values, mask)
        total = self._apply_matrices(filled)
        coverage = self._apply_matrices(valid)
        mean = total / coverage.clamp_min(1e-12)
        mean = torch.where(coverage >= min_coverage, mean, torch.full_like(mean, float("nan")))
        return mean, coverage.clamp(0.0, 1.0)

    def moments(self, values: torch.Tensor, mask: torch.Tensor | None = None,
                min_coverage: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(mean, standard deviation, coverage)`` per target cell -- the spread is sub-cell texture."""
        filled, valid = self._prepare(values, mask)
        coverage = self._apply_matrices(valid)
        safe = coverage.clamp_min(1e-12)
        mean = self._apply_matrices(filled) / safe
        # Centred on a rough per-field offset first: E[x^2] - E[x]^2 in float32 on 250 K brightness
        # temperatures cancels catastrophically, and a 1 K spread vanishes into round-off.
        reference = (filled.sum((-1, -2), keepdim=True) / valid.sum((-1, -2), keepdim=True).clamp_min(1.0))
        shifted = torch.where(valid > 0, filled - reference, torch.zeros_like(filled))
        second = self._apply_matrices(shifted * shifted) / safe
        centred = mean - reference
        spread = (second - centred * centred).clamp_min(0.0).sqrt()
        missing = coverage < min_coverage
        nan = torch.full_like(mean, float("nan"))
        return torch.where(missing, nan, mean), torch.where(missing, nan, spread), coverage.clamp(0.0, 1.0)

    def extremes(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ``(minimum, maximum)`` over the source pixels whose centres fall in each target cell; NaN if none.

        Done in two separable passes -- over columns, then rows -- so the index is a pair of short vectors
        rather than one the size of the source image.
        """
        filled, valid = self._prepare(values, mask)
        observed = valid > 0
        lows = torch.where(observed, filled, torch.full_like(filled, math.inf))
        highs = torch.where(observed, filled, torch.full_like(filled, -math.inf))
        out = []
        for field, how, empty in ((lows, "amin", math.inf), (highs, "amax", -math.inf)):
            out.append(self._reduce(field, how, empty))
        low, high = out
        nan = torch.full_like(low, float("nan"))
        return torch.where(torch.isfinite(low), low, nan), torch.where(torch.isfinite(high), high, nan)

    def _reduce(self, field: torch.Tensor, how: str, empty: float) -> torch.Tensor:
        rows, cols = self.target_shape
        lead = field.shape[:-2]
        lon_index, lat_index = self.lon_index, self.lat_index
        # Pixels outside every target cell go to a spill slot that is cut off afterwards.
        lon_slot = torch.where(lon_index >= 0, lon_index, cols)
        lat_slot = torch.where(lat_index >= 0, lat_index, rows)
        first = torch.full((*lead, field.shape[-2], cols + 1), empty, dtype=field.dtype, device=field.device)
        first = first.scatter_reduce(-1, lon_slot.expand(*lead, field.shape[-2], -1), field, how, include_self=True)
        first = first[..., :cols].transpose(-1, -2)                                     # (..., cols, H)
        second = torch.full((*lead, cols, rows + 1), empty, dtype=field.dtype, device=field.device)
        second = second.scatter_reduce(-1, lat_slot.expand(*lead, cols, -1), first, how, include_self=True)
        return second[..., :rows].transpose(-1, -2)


def sphere_pad(x: torch.Tensor, pad_lat: int, pad_lon: int, latitude: str = "global") -> torch.Tensor:
    """
    Pad ``(..., H, W)`` equirectangular data the way the sphere continues it.

    Longitude wraps. Across a pole, the continuation of a meridian is the opposite meridian running back
    down: the row ``k`` beyond the north pole is the row ``k`` below it, half-way round the globe. So the
    padding is the edge rows, mirrored and rolled by ``W / 2``.

    Args:
        latitude: ``"global"`` for cell-centred rows that do not include the poles (0.25 degrees from
            -89.875 to 89.875); ``"poles"`` for rows that include them (ERA5's 121 rows from -90 to 90),
            where the pole row is not repeated; ``"band"`` for a grid that stops short of the poles
            (geostationary imagery, 70S to 70N), which is edge-replicated -- there is nothing beyond it.
    """
    if pad_lat:
        width = x.shape[-1]
        if latitude in ("global", "poles"):
            if width % 2:
                raise ValueError("folding across the pole needs an even number of longitudes")
            skip = 1 if latitude == "poles" else 0
            if x.shape[-2] < pad_lat + skip:
                raise ValueError("too few rows to fold across the pole")
            top = x[..., skip : skip + pad_lat, :].flip(-2).roll(width // 2, -1)
            bottom = x[..., x.shape[-2] - skip - pad_lat : x.shape[-2] - skip, :].flip(-2).roll(width // 2, -1)
            x = torch.cat([top, x, bottom], -2)
        elif latitude == "band":
            x = torch.cat([x[..., :1, :].expand(*x.shape[:-2], pad_lat, -1), x,
                           x[..., -1:, :].expand(*x.shape[:-2], pad_lat, -1)], -2)
        else:
            raise ValueError(f"unknown latitude mode {latitude!r}")
    if pad_lon:
        x = torch.cat([x[..., -pad_lon:], x, x[..., :pad_lon]], -1)
    return x


class SphericalConv2d(nn.Conv2d):
    """
    ``nn.Conv2d`` on an equirectangular grid, with the padding the sphere implies: no wall at the date line,
    no wall at the poles. See :func:`sphere_pad`. Everything else -- weights, strides, groups -- is Conv2d.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 latitude: str = "band", bias: bool = True, groups: int = 1) -> None:
        if kernel_size % 2 == 0:
            raise ValueError("use an odd kernel so the padding is symmetric")
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias, groups=groups)
        self.latitude = latitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.kernel_size[0] // 2
        return super().forward(sphere_pad(x, half, half, self.latitude))


def latlon_channels(lat, lon, dtype=torch.float32) -> torch.Tensor:
    """
    ``(4, H, W)`` sin/cos of latitude and longitude, to append to an image so a convolution knows where it is.

    A convolution is translation-invariant by construction, and on an equirectangular grid that is the
    wrong invariance: a 3 x 3 kernel covers 80 km at the equator and 27 km at 70 degrees. Telling it the
    latitude lets it learn the difference; the longitude pair lets it learn which satellite is looking.
    """
    lat = torch.deg2rad(torch.as_tensor(lat, dtype=torch.float64).reshape(-1, 1))
    lon = torch.deg2rad(torch.as_tensor(lon, dtype=torch.float64).reshape(1, -1))
    shape = (lat.shape[0], lon.shape[1])
    return torch.stack([torch.sin(lat).expand(shape), torch.cos(lat).expand(shape),
                        torch.sin(lon).expand(shape), torch.cos(lon).expand(shape)]).to(dtype)


def _unit_vectors(lat_deg: torch.Tensor, lon_deg: torch.Tensor) -> torch.Tensor:
    lat, lon = torch.deg2rad(lat_deg.to(torch.float64)), torch.deg2rad(lon_deg.to(torch.float64))
    return torch.stack([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], -1)


class GridToMesh(nn.Module):
    """
    Pool a regular latitude-longitude image onto a mesh: every pixel goes to its nearest mesh point.

    The assignment is a Voronoi partition of the sphere by the mesh, computed once. Pooling is then one
    indexed sum per statistic -- the area-weighted mean over the observed pixels of each mesh cell, the
    coldest and warmest pixel, and the observed share of the cell -- so a quarter-degree global image
    (800,000 pixels) lands on a 4096-point mesh in one pass, at any resolution, with no attention.

    Mesh points no pixel falls in -- the polar caps under a geostationary band -- come back with
    coverage zero and zeros elsewhere, and a model reading them should be told the coverage.

    Args:
        lat, lon: the image's row and column centres, degrees.
        mesh: an :class:`ihelix.FieldGrid` on the globe (coords in radians, latitude first).
    """

    def __init__(self, lat, lon, mesh, chunk: int = 65536) -> None:
        super().__init__()
        lat = torch.as_tensor(lat, dtype=torch.float64).reshape(-1)
        lon = torch.as_tensor(lon, dtype=torch.float64).reshape(-1)
        grid_lat, grid_lon = torch.meshgrid(lat, lon, indexing="ij")
        pixels = _unit_vectors(grid_lat.reshape(-1), grid_lon.reshape(-1))
        coords = mesh.coords[:, :2].to("cpu", torch.float64)
        targets = _unit_vectors(torch.rad2deg(coords[:, 0]), torch.rad2deg(coords[:, 1]))
        index = torch.empty(pixels.shape[0], dtype=torch.long)
        for start in range(0, pixels.shape[0], chunk):
            index[start : start + chunk] = (pixels[start : start + chunk] @ targets.T).argmax(-1)
        # Pixel area on the sphere, proportional to the cosine of latitude for a regular grid.
        area = grid_lat.reshape(-1).deg2rad().cos().clamp_min(0.0)
        cell_area = torch.zeros(targets.shape[0], dtype=torch.float64).index_add_(0, index, area)
        self.register_buffer("index", index, persistent=False)
        self.register_buffer("area", area.float(), persistent=False)
        self.register_buffer("cell_area", cell_area.float(), persistent=False)
        self.num_points, self.image_shape = targets.shape[0], (lat.numel(), lon.numel())

    def extra_repr(self) -> str:
        return f"{self.image_shape[0]}x{self.image_shape[1]} -> {self.num_points} mesh points"

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None = None,
                extremes: bool = False) -> dict[str, torch.Tensor]:
        """
        Args:
            values: ``(..., H, W)`` or ``(..., H * W)``; NaN counts as missing.
            mask: optional, True where observed.
            extremes: also return per-cell ``min`` and ``max``.

        Returns:
            ``{"mean": (..., P), "coverage": (..., P)}`` and, with ``extremes``, ``"min"`` and ``"max"``.
            Unobserved cells are 0 in every statistic, with coverage 0 saying so.
        """
        flat = values.reshape(*values.shape[: -2 if values.shape[-2:] == self.image_shape else -1], -1)
        if flat.shape[-1] != self.index.numel():
            raise ValueError(f"expected {self.index.numel()} pixels, got {flat.shape[-1]}")
        flat = flat.to(self.area.device)
        compute = flat.dtype if flat.is_floating_point() and flat.dtype != torch.float16 else torch.float32
        flat = flat.to(compute)
        valid = torch.isfinite(flat)
        if mask is not None:
            valid = valid & mask.reshape(flat.shape).to(flat.device).bool()
        weight = self.area.to(compute) * valid.to(compute)
        lead = flat.shape[:-1]
        index = self.index.expand(*lead, -1)
        filled = torch.where(valid, flat, torch.zeros_like(flat))
        total = torch.zeros(*lead, self.num_points, dtype=compute, device=flat.device).scatter_add(-1, index, filled * weight)
        observed = torch.zeros(*lead, self.num_points, dtype=compute, device=flat.device).scatter_add(-1, index, weight)
        out = {"mean": total / observed.clamp_min(1e-12),
               "coverage": (observed / self.cell_area.to(compute).clamp_min(1e-12)).clamp(0.0, 1.0)}
        if extremes:
            for name, how, empty in (("min", "amin", math.inf), ("max", "amax", -math.inf)):
                field = torch.where(valid, flat, torch.full_like(flat, empty))
                reduced = torch.full((*lead, self.num_points), empty, dtype=compute, device=flat.device)
                reduced = reduced.scatter_reduce(-1, index, field, how, include_self=True)
                out[name] = torch.where(torch.isfinite(reduced), reduced, torch.zeros_like(reduced))
        return out


__all__ = ["ConservativeRegrid", "GridToMesh", "SphericalConv2d", "cell_edges", "latlon_channels", "sphere_pad"]
