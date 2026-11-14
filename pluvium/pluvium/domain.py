# Copyright 2026 Nathan. Apache-2.0.
"""
The ground Pluvium forecasts for: the Caribbean as points the model thinks on, the cells it learns from, the towns
it speaks about, and what the terrain looks like under each of them.

* **The latent mesh** is variable-resolution, because rain over Jamaica is not decided at the scale of rain over
  open sea: about 11 km over land, 22 km over the coastal sea within 60 km of a shore, 80 km over open ocean. The
  iHELIX processor runs on it; its quadrature weights say how much ground each point stands for, so the attention
  integrates over the Caribbean rather than over however many points happen to sit on an island.
* **The targets** are CMORPH's own 8 km cells (the truth the model is trained against): every cell over land or
  within 30 km of it, plus a fixed scatter of open-sea cells so the model still sees rain systems arrive.
* **The towns** come from GeoNames: every populated place in every Caribbean country and territory, with its
  first-level unit (Jamaica's parishes, Cuba's provinces, ...) and its own time zone.
* **The terrain** comes from :class:`naturev1.swarm.geography.RegionMap` (geoBoundaries units, Terrarium elevation),
  and every point gets the same 13 descriptors: elevation (mean, max, roughness), slope, how much land surrounds it
  at 5, 15 and 40 km, how far it is from the coast, and which way the sea lies.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .util import fetch, log

#: Box the predictors are read over (south, north, west, east): the Caribbean plus room for systems to arrive from.
DOMAIN = (8.0, 29.0, -91.0, -57.0)
#: Box the forecasts are made for: every Caribbean country and territory (naturev1's CARIBBEAN_BOX).
TARGET_BOX = (9.5, 27.6, -89.6, -59.2)
#: CMORPH CDR 8 km grid (centres), from the files' own metadata.
CMORPH_LAT0, CMORPH_LAT1, CMORPH_NLAT = -59.963614312, 59.963613336, 1649
CMORPH_LON0, CMORPH_LON1, CMORPH_NLON = 0.0363783345, 359.9636198775, 4948

#: GeoNames country codes of the Caribbean (ISO 3166-1 alpha-2), with Belize on the Caribbean coast.
CARIBBEAN_ISO2 = ("JM", "CU", "HT", "DO", "PR", "BS", "KY", "TC", "VG", "VI", "AI", "AG", "KN", "MS", "DM", "LC", "VC",
                  "GD", "BB", "TT", "AW", "CW", "SX", "MF", "BL", "GP", "MQ", "BQ", "BZ")
GEONAMES = "https://download.geonames.org/export/dump/{name}"
#: Populated-place feature codes that are not places anyone lives now.
GONE = {"PPLQ", "PPLH", "PPLW", "PPLCH"}

STATIC_FEATURES = ("elev_mean", "elev_max", "elev_std", "slope_x", "slope_y", "land_5km", "land_15km", "land_40km",
                   "coast_km", "sea_dir_x", "sea_dir_y", "lat_n", "lon_n")
KM_PER_DEG = 111.195


def cmorph_axes() -> tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(CMORPH_LAT0, CMORPH_LAT1, CMORPH_NLAT)
    lon = np.linspace(CMORPH_LON0, CMORPH_LON1, CMORPH_NLON)
    return lat, lon


def signed_lon(lon):
    """Longitudes on -180..180 (CMORPH and some archives count 0..360)."""
    return (np.asarray(lon, np.float64) + 180.0) % 360.0 - 180.0


def cmorph_window(box=DOMAIN):
    """Rows and columns of the CMORPH 8 km grid covering ``box``, west to east. The columns are a slice, or an index
    array when the box crosses the prime meridian (CMORPH counts longitude 0..360). Boxes crossing the date line are
    not supported."""
    lat, lon = cmorph_axes()
    s, n, w, e = box
    if e - w >= 360 or w < -180 or e > 180:
        raise ValueError(f"box {box}: longitudes must lie within -180..180 without crossing the date line")
    rows = np.nonzero((lat >= s) & (lat <= n))[0]
    signed = signed_lon(lon)
    cols = np.nonzero((signed >= w) & (signed <= e))[0]
    cols = cols[np.argsort(signed[cols], kind="stable")]
    if cols.size and np.all(np.diff(cols) == 1):
        cols = slice(int(cols[0]), int(cols[-1]) + 1)
    return slice(int(rows[0]), int(rows[-1]) + 1), cols


def window_lon(cols) -> np.ndarray:
    """Signed longitudes of a CMORPH window's columns (see :func:`cmorph_window`), ascending."""
    return signed_lon(cmorph_axes()[1][cols])


# ------------------------------------------------------------------------------------------------ terrain --

@dataclass
class Terrain:
    """Rasters derived once from a RegionMap: everything a point's descriptors are sampled from."""

    lat: np.ndarray
    lon: np.ndarray
    land: np.ndarray        # bool (H, W)
    elevation: np.ndarray   # m, 0 at sea
    smooth: dict            # name -> (H, W) float32 rasters
    coast_km: np.ndarray    # + inland, - offshore
    sea_dir: np.ndarray     # (2, H, W) unit vector (east, north) towards the sea (land) or towards land (sea)

    @classmethod
    def from_region(cls, region) -> Terrain:
        from scipy import ndimage

        lat, lon = np.asarray(region.lat, np.float64), np.asarray(region.lon, np.float64)
        land = np.asarray(region.unit) >= 0
        elev = np.where(land, np.asarray(region.elevation, np.float32), 0.0).astype(np.float32)
        dlat_km = float(lat[1] - lat[0]) * KM_PER_DEG
        dlon_km = float(lon[1] - lon[0]) * KM_PER_DEG * math.cos(math.radians(float(lat.mean())))
        cell_km = math.sqrt(dlat_km * dlon_km)

        def gauss(field, km):
            return ndimage.gaussian_filter(field.astype(np.float32), (km / dlat_km, km / dlon_km), mode="constant")

        smooth = {
            "land_5km": gauss(land, 5.0), "land_15km": gauss(land, 15.0), "land_40km": gauss(land, 40.0),
            "elev_mean": gauss(elev, 4.0) / 1000.0,
            "elev_max": ndimage.maximum_filter(elev, size=max(3, int(round(10.0 / cell_km)) | 1)) / 1000.0,
        }  # fmt: skip
        smooth["elev_std"] = np.sqrt(np.clip(gauss(elev**2 / 1e6, 8.0) - gauss(elev / 1000.0, 8.0) ** 2, 0, None))
        inside = ndimage.distance_transform_edt(land, sampling=(dlat_km, dlon_km))
        outside = ndimage.distance_transform_edt(~land, sampling=(dlat_km, dlon_km))
        coast = np.where(land, inside, -outside).astype(np.float32)
        gy, gx = np.gradient(gauss(coast, 6.0), dlat_km, dlon_km)
        norm = np.hypot(gx, gy) + 1e-6
        # down the inland-distance gradient is towards the sea on land and away from land at sea
        sea_dir = np.stack([-gx / norm, -gy / norm]).astype(np.float32)
        return cls(lat, lon, land, elev, smooth, coast, sea_dir)

    def sample(self, field: np.ndarray, lat, lon) -> np.ndarray:
        from scipy import ndimage

        lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        i = (lat - self.lat[0]) / (self.lat[1] - self.lat[0])
        j = (lon - self.lon[0]) / (self.lon[1] - self.lon[0])
        return ndimage.map_coordinates(field, [i, j], order=1, mode="nearest").astype(np.float32)

    def descriptors(self, lat, lon) -> np.ndarray:
        """``(N, 13)`` STATIC_FEATURES at points, in order-one units (km for heights, 100 km for coast distance)."""
        lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        elev = self.elevation / 1000.0
        dy = (self.lat[1] - self.lat[0]) * KM_PER_DEG * 1000.0
        dx = (self.lon[1] - self.lon[0]) * KM_PER_DEG * 1000.0 * np.cos(np.radians(self.lat))[:, None]
        gy, gx = np.gradient(self.elevation.astype(np.float64), axis=(0, 1))
        cols = [
            self.sample(self.smooth["elev_mean"], lat, lon), self.sample(self.smooth["elev_max"], lat, lon),
            self.sample(self.smooth["elev_std"], lat, lon),
            np.clip(self.sample((gx / dx).astype(np.float32), lat, lon) * 10.0, -3, 3),
            np.clip(self.sample((gy / dy).astype(np.float32), lat, lon) * 10.0, -3, 3),
            self.sample(self.smooth["land_5km"], lat, lon), self.sample(self.smooth["land_15km"], lat, lon),
            self.sample(self.smooth["land_40km"], lat, lon), self.sample(self.coast_km, lat, lon) / 100.0,
            self.sample(self.sea_dir[0], lat, lon), self.sample(self.sea_dir[1], lat, lon),
            ((lat - 18.0) / 10.0).astype(np.float32), ((lon + 74.0) / 20.0).astype(np.float32),
        ]  # fmt: skip
        del elev
        return np.stack(cols, axis=-1).astype(np.float32)

    def land_at(self, lat, lon) -> np.ndarray:
        return self.sample(self.land.astype(np.float32), lat, lon) >= 0.5

    def save(self, path) -> None:
        np.savez_compressed(path, lat=self.lat, lon=self.lon, land=self.land, elevation=self.elevation,
                            coast_km=self.coast_km, sea_dir=self.sea_dir, **{f"s_{k}": v for k, v in self.smooth.items()})

    @classmethod
    def load(cls, path) -> Terrain:
        d = np.load(path)
        smooth = {k[2:]: d[k] for k in d.files if k.startswith("s_")}
        return cls(d["lat"], d["lon"], d["land"], d["elevation"], smooth, d["coast_km"], d["sea_dir"])


# --------------------------------------------------------------------------------------------- the mesh --

@dataclass
class PointSet:
    """Named points with their static descriptors (mesh points, CMORPH targets, stations or towns)."""

    lat: np.ndarray
    lon: np.ndarray
    statics: np.ndarray                       # (N, len(STATIC_FEATURES))
    kind: np.ndarray                          # int8: 0 land, 1 coastal sea, 2 open sea (mesh/targets); 0 for towns
    weights: np.ndarray | None = None         # quadrature (share of the domain), mesh only
    extra: dict = field(default_factory=dict)  # e.g. CMORPH rows/cols, station ids, town names

    def __len__(self) -> int:
        return int(self.lat.size)

    def grid(self, num_neighbours: int = 32, cluster_size: int = 64):
        """The iHELIX FieldGrid over these points (coordinates in radians on the globe, distances in km)."""
        import torch
        from ihelix import FieldGrid, Geometry

        coords = torch.tensor(np.radians(np.stack([self.lat, self.lon], -1)), dtype=torch.float64)
        weights = None if self.weights is None else torch.tensor(self.weights, dtype=torch.float64)
        return FieldGrid.from_points(coords, Geometry.globe(), num_neighbours=min(num_neighbours, len(self)),
                                     cluster_size=cluster_size, weights=weights)

    def source_grid(self):
        """A read-only iHELIX grid (no neighbour graph): what decoders and encoders link to."""
        import torch
        from ihelix import FieldGrid, Geometry

        coords = torch.tensor(np.radians(np.stack([self.lat, self.lon], -1)), dtype=torch.float64)
        weights = None if self.weights is None else torch.tensor(self.weights, dtype=torch.float64)
        return FieldGrid.source(coords, Geometry.globe(), weights)

    def subset(self, index) -> PointSet:
        index = np.asarray(index)
        extra = {k: (v[index] if isinstance(v, np.ndarray) and v.shape[:1] == self.lat.shape else v)
                 for k, v in self.extra.items()}
        return PointSet(self.lat[index], self.lon[index], self.statics[index], self.kind[index],
                        None if self.weights is None else self.weights[index], extra)

    def save(self, path) -> None:
        payload = {"lat": self.lat, "lon": self.lon, "statics": self.statics, "kind": self.kind}
        if self.weights is not None:
            payload["weights"] = self.weights
        for k, v in self.extra.items():
            payload[f"x_{k}"] = np.asarray(v)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path) -> PointSet:
        d = np.load(path, allow_pickle=False)
        extra = {k[2:]: d[k] for k in d.files if k.startswith("x_")}
        return cls(d["lat"], d["lon"], d["statics"], d["kind"], d["weights"] if "weights" in d.files else None, extra)


def _bin_points(lat, lon, step: float, s: float, w: float) -> dict:
    """Group points into ``step``-degree bins: ``{(i, j): indices}``."""
    i = np.floor((np.asarray(lat) - s) / step).astype(np.int64)
    j = np.floor((np.asarray(lon) - w) / step).astype(np.int64)
    key = i * 100000 + j
    order = np.argsort(key, kind="stable")
    keys, starts = np.unique(key[order], return_index=True)
    groups = np.split(order, starts[1:])
    return {(int(k // 100000), int(k % 100000)): g for k, g in zip(keys, groups)}


def build_mesh(terrain: Terrain, box=DOMAIN, land_step: float = 0.1, coast_step: float = 0.2, ocean_step: float = 0.75,
               coast_km: float = 60.0) -> PointSet:
    """
    The variable-resolution latent mesh: one point per ``land_step`` bin holding land (at the bin's land centroid,
    so a small island keeps its point), coastal-sea points every ``coast_step`` within ``coast_km`` of a shore, open
    sea every ``ocean_step``. Quadrature weights are each point's share of the box's area.
    """
    s, n, w, e = box
    la, lo = np.meshgrid(terrain.lat, terrain.lon, indexing="ij")
    inside = (la >= s) & (la <= n) & (lo >= w) & (lo <= e)
    pts_lat, pts_lon, kinds, area = [], [], [], []
    land = terrain.land & inside
    for (i, j), idx in _bin_points(la[land], lo[land], land_step, s, w).items():
        pts_lat.append(float(la[land][idx].mean()))
        pts_lon.append(float(lo[land][idx].mean()))
        kinds.append(0)
        area.append(land_step**2)
    have = np.array([pts_lat, pts_lon]).T if pts_lat else np.zeros((0, 2))
    lat_c = np.arange(s + coast_step / 2, n, coast_step)
    lon_c = np.arange(w + coast_step / 2, e, coast_step)
    cla, clo = np.meshgrid(lat_c, lon_c, indexing="ij")
    cla, clo = cla.ravel(), clo.ravel()
    coast_dist = terrain.sample(terrain.coast_km, cla, clo)
    near = (coast_dist < 0) & (coast_dist > -coast_km)
    from scipy.spatial import cKDTree

    tree = cKDTree(have) if len(have) else None
    for a, b in zip(cla[near], clo[near]):
        if tree is not None and tree.query([a, b])[0] < coast_step * 0.6:
            continue
        pts_lat.append(float(a))
        pts_lon.append(float(b))
        kinds.append(1)
        area.append(coast_step**2)
    have = np.array([pts_lat, pts_lon]).T
    tree = cKDTree(have)
    lat_o = np.arange(s + ocean_step / 2, n, ocean_step)
    lon_o = np.arange(w + ocean_step / 2, e, ocean_step)
    for a in lat_o:
        for b in lon_o:
            if tree.query([a, b])[0] < ocean_step * 0.5:
                continue
            pts_lat.append(float(a))
            pts_lon.append(float(b))
            kinds.append(2)
            area.append(ocean_step**2)
    lat, lon = np.array(pts_lat), np.array(pts_lon)
    weights = np.array(area) * np.cos(np.radians(lat))
    weights = weights / weights.sum()
    return PointSet(lat, lon, terrain.descriptors(lat, lon), np.array(kinds, np.int8), weights)


def build_targets(terrain: Terrain, box=TARGET_BOX, coast_km: float = 30.0, ocean_cells: int = 4000,
                  seed: int = 0, cmorph_box=DOMAIN) -> PointSet:
    """CMORPH 8 km cells to learn from: all over land or within ``coast_km`` of it, plus ``ocean_cells`` open-sea
    cells drawn once. ``extra["row"]``/``extra["col"]`` index the cell in the CMORPH window over ``cmorph_box``."""
    lat, _ = cmorph_axes()
    rows_d, cols_d = cmorph_window(cmorph_box)
    s, n, w, e = box
    la, lo = np.meshgrid(lat[rows_d], window_lon(cols_d), indexing="ij")
    inside = (la >= s) & (la <= n) & (lo >= w) & (lo <= e)
    coast = terrain.sample(terrain.coast_km, la.ravel(), lo.ravel()).reshape(la.shape)
    near = inside & (coast > -coast_km)
    far = inside & ~near
    rng = np.random.default_rng(seed)
    far_idx = np.flatnonzero(far)
    pick = rng.choice(far_idx, size=min(ocean_cells, far_idx.size), replace=False) if far_idx.size else far_idx
    flat = np.concatenate([np.flatnonzero(near), np.sort(pick)])
    r, c = np.unravel_index(flat, la.shape)
    tlat, tlon = la.ravel()[flat], lo.ravel()[flat]
    cdist = coast.ravel()[flat]
    kind = np.where(cdist >= 0, 0, np.where(cdist > -coast_km, 1, 2)).astype(np.int8)
    return PointSet(tlat, tlon, terrain.descriptors(tlat, tlon), kind,
                    extra={"row": r.astype(np.int32), "col": c.astype(np.int32)})


# --------------------------------------------------------------------------------------------- the towns --

@dataclass
class Gazetteer:
    """Every populated place: name, coordinates, country, first-level unit (parish/province), population, zone."""

    points: PointSet

    @property
    def names(self) -> np.ndarray:
        return self.points.extra["name"]

    def find(self, query: str, country: str | None = None) -> np.ndarray:
        """Indices of places whose name matches ``query`` (case-insensitive): exact names first, then names that start
        with it, then names that contain it; most populous first within each."""
        q = query.strip().lower()
        names = np.char.lower(self.points.extra["name"].astype(str))
        rank = np.where(names == q, 0, np.where(np.char.startswith(names, q), 1,
                                                np.where(np.char.find(names, q) >= 0, 2, 3)))
        if country:
            rank = np.where(self.points.extra["country"].astype(str) == country.upper(), rank, 3)
        idx = np.flatnonzero(rank < 3)
        population = self.points.extra["population"][idx].astype(np.int64)
        return idx[np.lexsort((-population, rank[idx]))]

    def label(self, k: int) -> str:
        x = self.points.extra
        unit = str(x["unit"][k])
        return f"{x['name'][k]}, {unit + ', ' if unit else ''}{x['country'][k]}"


def _read_admin1(cache: Path) -> dict:
    path = cache / "admin1CodesASCII.txt"
    if not path.exists():
        path.write_bytes(fetch(GEONAMES.format(name="admin1CodesASCII.txt")))
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def parse_geonames(text: str, admin1: dict, box=TARGET_BOX, min_population: int = 0) -> list[dict]:
    """GeoNames dump rows (tab separated) -> place dicts for populated places inside ``box``."""
    s, n, w, e = box
    rows = []
    for line in text.splitlines():
        p = line.split("\t")
        if len(p) < 18 or p[6] != "P" or p[7] in GONE:
            continue
        lat, lon = float(p[4]), float(p[5])
        if not (s <= lat <= n and w <= lon <= e):
            continue
        population = int(p[14] or 0)
        if population < min_population:
            continue
        rows.append({"name": p[1], "lat": lat, "lon": lon, "country": p[8], "unit": admin1.get(f"{p[8]}.{p[10]}", ""),
                     "population": population, "feature": p[7], "timezone": p[17] or "UTC", "id": int(p[0])})
    return rows


def build_gazetteer(terrain: Terrain, countries=CARIBBEAN_ISO2, cache="~/.cache/pluvium/geonames", box=TARGET_BOX,
                    min_population: int = 0, verbose: bool = True) -> Gazetteer:
    """Download (once) and parse GeoNames for ``countries``; every populated place becomes a forecast point."""
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    admin1 = _read_admin1(cache)
    rows = []
    for cc in countries:
        path = cache / f"{cc}.txt"
        if not path.exists():
            try:
                blob = fetch(GEONAMES.format(name=f"{cc}.zip"))
                with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                    path.write_bytes(zf.read(f"{cc}.txt"))
            except Exception as error:
                if verbose:
                    log(f"  GeoNames {cc}: not fetched ({error})")
                continue
        got = parse_geonames(path.read_text(encoding="utf-8"), admin1, box, min_population)
        rows += got
        if verbose:
            log(f"  GeoNames {cc}: {len(got):,} places")
    return gazetteer_from_rows(terrain, rows)


def gazetteer_from_rows(terrain: Terrain, rows: list[dict]) -> Gazetteer:
    lat = np.array([r["lat"] for r in rows], np.float64)
    lon = np.array([r["lon"] for r in rows], np.float64)
    extra = {k: np.array([r[k] for r in rows]) for k in ("name", "country", "unit", "population", "feature",
                                                          "timezone", "id")}
    statics = terrain.descriptors(lat, lon) if len(rows) else np.zeros((0, len(STATIC_FEATURES)), np.float32)
    return Gazetteer(PointSet(lat, lon, statics, np.zeros(len(rows), np.int8), None, extra))


# ------------------------------------------------------------------------------------- orographic tiles --

@dataclass
class OroTile:
    """One square of the orographic layer: a ``size`` x ``size`` grid at ``dx_km`` centred on land, whose central
    ``core_deg`` box is the area it answers for (the rest is a margin of sea and land so the FFT does not wrap)."""

    lat0: float
    lon0: float
    core_deg: float
    size: int
    dx_km: float
    elevation: np.ndarray  # (size, size) m, rows south -> north, columns west -> east

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        half = (self.size - 1) / 2.0
        k = np.arange(self.size) - half
        lat = self.lat0 + k * self.dx_km / KM_PER_DEG
        lon = self.lon0 + k * self.dx_km / (KM_PER_DEG * math.cos(math.radians(self.lat0)))
        return lat, lon

    def owns(self, lat, lon) -> np.ndarray:
        h = self.core_deg / 2.0
        lat, lon = np.asarray(lat), np.asarray(lon)
        return (np.abs(lat - self.lat0) <= h) & (np.abs(lon - self.lon0) <= h)


def build_tiles(terrain: Terrain, box=TARGET_BOX, core_deg: float = 3.5, size: int = 256, dx_km: float = 2.0,
                min_relief_m: float = 50.0) -> list[OroTile]:
    """A lattice of ``core_deg`` boxes over ``box``; each one whose core holds land higher than ``min_relief_m``
    becomes a tile, its elevation resampled from the terrain raster onto a ``dx_km`` grid."""
    s, n, w, e = box
    tiles = []
    for lat0 in np.arange(s + core_deg / 2, n + core_deg / 2, core_deg):
        for lon0 in np.arange(w + core_deg / 2, e + core_deg / 2, core_deg):
            tile = OroTile(float(lat0), float(lon0), core_deg, size, dx_km, np.zeros((size, size), np.float32))
            lat, lon = tile.axes()
            la, lo = np.meshgrid(lat, lon, indexing="ij")
            elev = terrain.sample(terrain.elevation, la.ravel(), lo.ravel()).reshape(la.shape)
            core = tile.owns(la, lo)
            if not (core & (elev > min_relief_m)).any():
                continue
            tile.elevation = np.clip(elev, 0.0, None).astype(np.float32)
            tiles.append(tile)
    return tiles


def save_tiles(tiles: list[OroTile], path) -> None:
    np.savez_compressed(path, meta=np.array([[t.lat0, t.lon0, t.core_deg, t.size, t.dx_km] for t in tiles], np.float64),
                        elevation=np.stack([t.elevation for t in tiles]) if tiles else np.zeros((0, 1, 1), np.float32))


def load_tiles(path) -> list[OroTile]:
    d = np.load(path)
    return [OroTile(float(m[0]), float(m[1]), float(m[2]), int(m[3]), float(m[4]), d["elevation"][k])
            for k, m in enumerate(d["meta"])]


# ----------------------------------------------------------------------------------------- the whole --

@dataclass
class Geography:
    """Everything spatial a run needs, built once (``build``), cached as files, loaded in seconds (``load``)."""

    terrain: Terrain
    mesh: PointSet
    targets: PointSet
    tiles: list
    attribution: str = ""
    #: the CMORPH window the targets' ``row``/``col`` index (geographies saved before areas existed used DOMAIN)
    cmorph_box: tuple = DOMAIN
    target_box: tuple = TARGET_BOX
    mesh_box: tuple = DOMAIN
    #: which countries this geography serves (:class:`pluvium.area.Area`); None means the original Caribbean
    area: object = None

    @classmethod
    def build(cls, cache="~/.cache/pluvium/geography", step_deg: float = 0.02, region=None, verbose: bool = True,
              mesh_box=None, target_box=None, ocean_cells: int = 4000, area=None, **mesh_kwargs) -> Geography:
        """Rasterise the area's countries (``naturev1.swarm.geography.RegionMap``: geoBoundaries + Terrarium), then
        derive the mesh, the CMORPH targets and the orographic tiles. Without ``area`` it is the Caribbean.
        ``mesh_box``/``target_box`` override the area's boxes (sub-domains); ``region`` may be given (tests use a
        synthetic island)."""
        cache = Path(cache).expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        countries = area.countries if area is not None else None
        target_box = tuple(target_box or (area.target_box if area is not None else TARGET_BOX))
        mesh_box = tuple(mesh_box or (area.domain if area is not None else DOMAIN))
        cmorph_box = tuple(area.domain) if area is not None else DOMAIN
        if region is None:
            from naturev1.swarm.geography import CARIBBEAN, RegionMap

            tag = "" if area is None else f"_{area.name}"
            region_path = cache / f"region{tag}_{step_deg:.3f}.npz"
            if region_path.exists():
                region = RegionMap.load(region_path)
            else:
                region = RegionMap.build(countries or CARIBBEAN, box=area.target_box if area is not None else TARGET_BOX,
                                         step_deg=step_deg, cache=cache / "naturev1", verbose=verbose)
                region.save(region_path)
        terrain = Terrain.from_region(region)
        mesh = build_mesh(terrain, box=mesh_box, **mesh_kwargs)
        targets = build_targets(terrain, box=target_box, ocean_cells=ocean_cells, cmorph_box=cmorph_box)
        tiles = build_tiles(terrain, box=target_box)
        if verbose:
            log(f"geography{'' if area is None else ' for ' + area.name}: mesh {len(mesh):,} points "
                f"({int((mesh.kind == 0).sum()):,} land), {len(targets):,} CMORPH target cells, {len(tiles)} orographic tiles")
        return cls(terrain, mesh, targets, tiles, getattr(region, "attribution", ""), cmorph_box, target_box, mesh_box, area)

    def save(self, directory) -> Path:
        d = Path(directory).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        self.terrain.save(d / "terrain.npz")
        self.mesh.save(d / "mesh.npz")
        self.targets.save(d / "targets.npz")
        save_tiles(self.tiles, d / "tiles.npz")
        meta = {"attribution": self.attribution, "domain": list(self.cmorph_box), "cmorph_box": list(self.cmorph_box),
                "target_box": list(self.target_box), "mesh_box": list(self.mesh_box)}
        if self.area is not None:
            meta["area"] = self.area.to_json()
        (d / "geography.json").write_text(json.dumps(meta, indent=1))
        return d

    @classmethod
    def load(cls, directory) -> Geography:
        d = Path(directory).expanduser()
        meta = json.loads((d / "geography.json").read_text()) if (d / "geography.json").exists() else {}
        area = None
        if meta.get("area"):
            from .area import Area

            area = Area.from_json(meta["area"])
        cmorph_box = tuple(meta.get("cmorph_box", meta.get("domain", DOMAIN)))      # older files: the Caribbean window
        return cls(Terrain.load(d / "terrain.npz"), PointSet.load(d / "mesh.npz"), PointSet.load(d / "targets.npz"),
                   load_tiles(d / "tiles.npz"), meta.get("attribution", ""), cmorph_box,
                   tuple(meta.get("target_box", TARGET_BOX)), tuple(meta.get("mesh_box", cmorph_box)), area)


__all__ = ["CARIBBEAN_ISO2", "DOMAIN", "STATIC_FEATURES", "TARGET_BOX", "Gazetteer", "Geography", "OroTile", "PointSet",
           "Terrain", "build_gazetteer", "build_mesh", "build_targets", "build_tiles", "cmorph_axes", "cmorph_window",
           "signed_lon", "window_lon",
           "gazetteer_from_rows", "load_tiles", "parse_geonames", "save_tiles"]
