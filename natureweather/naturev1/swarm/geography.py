# Copyright 2026 Nathan. Apache-2.0.
"""
The ground the swarm lands on: every country and territory of the Caribbean as a raster of its own administrative
units (Jamaica's parishes, Cuba's provinces, Haiti's departments, the Dominican Republic's provinces, the Bahamas'
districts, the parishes of the Lesser Antilles, ...) with the terrain under them.

* **Boundaries** come from geoBoundaries (gbOpen, mostly OpenStreetMap-derived; attribution is kept with the map).
  Where a country has no first-level units the country itself is one unit.
* **Terrain** comes from the open Terrarium elevation tiles (AWS Open Data; SRTM, GMTED and others), averaged onto
  the map's grid, with its slope -- the slope is what turns wind into forced ascent and rain into run-off.
* Every unit gets at least one cell, so a 13 km^2 island is not lost on a 3 km grid.

``RegionMap.build(CARIBBEAN)`` downloads once and caches; ``RegionMap.load`` reads the cache. Any ISO-3166 code that
geoBoundaries knows can be added.
"""

from __future__ import annotations

import io
import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Caribbean countries and territories (ISO 3166-1 alpha-3), with Belize on the Caribbean coast.
CARIBBEAN = {
    "JAM": "Jamaica", "CUB": "Cuba", "HTI": "Haiti", "DOM": "Dominican Republic", "PRI": "Puerto Rico",
    "BHS": "Bahamas", "CYM": "Cayman Islands", "TCA": "Turks and Caicos Islands", "VGB": "British Virgin Islands",
    "VIR": "US Virgin Islands", "AIA": "Anguilla", "ATG": "Antigua and Barbuda", "KNA": "Saint Kitts and Nevis",
    "MSR": "Montserrat", "DMA": "Dominica", "LCA": "Saint Lucia", "VCT": "Saint Vincent and the Grenadines",
    "GRD": "Grenada", "BRB": "Barbados", "TTO": "Trinidad and Tobago", "ABW": "Aruba", "CUW": "Curacao",
    "SXM": "Sint Maarten", "MAF": "Saint Martin", "BLM": "Saint Barthelemy", "GLP": "Guadeloupe",
    "MTQ": "Martinique", "BES": "Caribbean Netherlands", "BLZ": "Belize",
}
#: Map extent (south, north, west, east) covering all of them.
CARIBBEAN_BOX = (9.5, 27.6, -89.6, -59.2)

GEOBOUNDARIES = "https://www.geoboundaries.org/api/current/{release}/{iso}/{level}/"
#: Release types tried in order: the open release, then the humanitarian one (which has, e.g., Puerto Rico's
#: municipios and Sint Maarten).
RELEASES = ("gbOpen", "gbHumanitarian")
TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def _get(url: str, timeout: float = 120.0, attempts: int = 4) -> bytes:
    last = None
    for k in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except Exception as error:                        # network hiccups: retry with backoff
            last = error
            import time

            time.sleep(2 ** k)
    raise RuntimeError(f"could not fetch {url}: {last}")


def _get_geojson(url: str) -> bytes:
    """geoBoundaries files live in Git LFS: ask the LFS media host first (it also works where github.com itself is
    filtered), then the URL as given."""
    import re

    m = re.match(r"https://github.com/([^/]+)/([^/]+)/raw/([^/]+)/(.+)", url)
    candidates = ([f"https://media.githubusercontent.com/media/{m[1]}/{m[2]}/{m[3]}/{m[4]}"] if m else []) + [url]
    last = None
    for candidate in candidates:
        try:
            data = _get(candidate, attempts=2)
            if data.lstrip()[:1] == b"{":
                return data
        except Exception as error:
            last = error
    raise RuntimeError(f"could not fetch {url}: {last}")


def fetch_units(iso: str, cache: Path, level: str = "ADM1") -> tuple[list[dict], dict]:
    """``([{"name", "rings": [(exterior, [holes])...]}], meta)`` for one country, from geoBoundaries (cached): its
    first-level units from the open or the humanitarian release, else the country as a whole."""
    path = cache / f"{iso}_{level}.geojson"
    meta_path = cache / f"{iso}_{level}.json"
    if not path.exists():
        meta = None
        for release in RELEASES:
            try:
                meta = json.loads(_get(GEOBOUNDARIES.format(release=release, iso=iso, level=level), attempts=1))
                break
            except Exception:
                continue
        if meta is None:
            if level == "ADM1":
                return fetch_units(iso, cache, "ADM0")
            raise RuntimeError(f"geoBoundaries has no {iso}")
        url = meta.get("simplifiedGeometryGeoJSON") or meta["gjDownloadURL"]
        path.write_bytes(_get_geojson(url))
        meta_path.write_text(json.dumps({k: meta.get(k) for k in ("boundaryName", "boundaryType", "boundaryCanonical",
                                                                 "boundarySource", "boundaryLicense",
                                                                 "boundaryYearRepresented")}))
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    units = []
    for feature in json.loads(path.read_text())["features"]:
        geom = feature["geometry"]
        props = feature.get("properties", {})
        name = props.get("shapeName") or props.get("shapeGroup") or iso
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        rings = [(np.asarray(p[0], np.float64), [np.asarray(h, np.float64) for h in p[1:]]) for p in polys]
        units.append({"name": name, "rings": rings})
    return units, meta


def _tile_xy(lat: float, lon: float, z: int) -> tuple[float, float]:
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def terrain(lat: np.ndarray, lon: np.ndarray, cache: Path, zoom: int = 8, boxes=None) -> np.ndarray:
    """Elevation (m) on the ``lat`` x ``lon`` grid from Terrarium tiles at ``zoom`` (block-averaged); only tiles that
    intersect ``boxes`` ((south, north, west, east) each) are fetched, the rest of the grid is sea level."""
    from PIL import Image

    out = np.zeros((lat.size, lon.size), np.float32)
    count = np.zeros_like(out)
    tiles = set()
    for s, n, w, e in boxes or [(lat[0], lat[-1], lon[0], lon[-1])]:
        x0, y0 = _tile_xy(n, w, zoom)
        x1, y1 = _tile_xy(s, e, zoom)
        tiles |= {(x, y) for x in range(int(x0), int(x1) + 1) for y in range(int(y0), int(y1) + 1)}
    step_lat, step_lon = lat[1] - lat[0], lon[1] - lon[0]
    for x, y in sorted(tiles):
        path = cache / f"terrarium_{zoom}_{x}_{y}.png"
        if not path.exists():
            path.write_bytes(_get(TERRARIUM.format(z=zoom, x=x, y=y)))
        rgb = np.asarray(Image.open(io.BytesIO(path.read_bytes())).convert("RGB"), np.float32)
        h = rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0
        px = (np.arange(256) + 0.5) / 256.0
        plon = (x + px) / 2 ** zoom * 360.0 - 180.0
        plat = np.degrees(np.arctan(np.sinh(math.pi * (1.0 - 2.0 * (y + px) / 2 ** zoom))))
        i = np.round((plat - lat[0]) / step_lat).astype(int)
        j = np.round((plon - lon[0]) / step_lon).astype(int)
        ii, jj = np.meshgrid(i, j, indexing="ij")
        ok = (ii >= 0) & (ii < lat.size) & (jj >= 0) & (jj < lon.size)
        np.add.at(out, (ii[ok], jj[ok]), h[ok])
        np.add.at(count, (ii[ok], jj[ok]), 1.0)
    return np.where(count > 0, out / np.maximum(count, 1.0), 0.0)


def _ring_rows(ring: np.ndarray, lat: np.ndarray, lon: np.ndarray, out: np.ndarray) -> None:
    """Toggle (even-odd) the cells whose centres lie inside one closed ring, by scanline."""
    x, y = ring[:, 0], ring[:, 1]
    if x[0] != x[-1] or y[0] != y[-1]:
        x, y = np.append(x, x[0]), np.append(y, y[0])
    x1, y1, x2, y2 = x[:-1], y[:-1], x[1:], y[1:]
    rows = np.nonzero((lat >= y.min()) & (lat <= y.max()))[0]
    step = lon[1] - lon[0]
    for i in rows:
        yr = lat[i]
        cross = ((y1 <= yr) & (y2 > yr)) | ((y2 <= yr) & (y1 > yr))
        if not cross.any():
            continue
        xs = np.sort(x1[cross] + (yr - y1[cross]) * (x2[cross] - x1[cross]) / (y2[cross] - y1[cross]))
        for a, b in zip(xs[0::2], xs[1::2]):
            j0 = int(np.ceil((a - lon[0]) / step - 1e-9))
            j1 = int(np.floor((b - lon[0]) / step + 1e-9))
            j0, j1 = max(j0, 0), min(j1, lon.size - 1)
            if j1 >= j0:
                out[i, j0:j1 + 1] ^= True


def _rasterise(rings, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Boolean ``(H, W)`` mask of grid-cell centres inside the polygons (even-odd, so holes come out)."""
    mask = np.zeros((lat.size, lon.size), bool)
    for exterior, holes in rings:
        poly = np.zeros_like(mask)
        _ring_rows(exterior, lat, lon, poly)
        for hole in holes:
            _ring_rows(hole, lat, lon, poly)
        mask |= poly
    return mask


@dataclass
class RegionMap:
    """A raster of administrative units with terrain: ``unit`` is ``(H, W)`` (-1 = sea), ``elevation`` m."""

    lat: np.ndarray
    lon: np.ndarray
    unit: np.ndarray
    elevation: np.ndarray
    names: list
    countries: list
    attribution: str = ""

    @property
    def land(self) -> np.ndarray:
        return self.unit >= 0

    def slope(self) -> tuple[np.ndarray, np.ndarray]:
        """(dh/dx, dh/dy), metres per metre (east, north)."""
        dy = (self.lat[1] - self.lat[0]) * 111195.0
        dx = (self.lon[1] - self.lon[0]) * 111195.0 * np.cos(np.radians(self.lat))[:, None]
        gy, gx = np.gradient(self.elevation.astype(np.float64))
        return gx / dx, gy / dy

    def cells(self) -> dict:
        """The land cells as flat arrays: lat, lon, unit, elevation, slope (east, north)."""
        i, j = np.nonzero(self.land)
        gx, gy = self.slope()
        return {"lat": self.lat[i], "lon": self.lon[j], "unit": self.unit[i, j], "elevation": self.elevation[i, j],
                "slope_x": gx[i, j], "slope_y": gy[i, j], "i": i, "j": j}

    def unit_of(self, lat, lon) -> np.ndarray:
        """Unit index under points (-1 at sea or off the map)."""
        lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        lon = (lon + 180.0) % 360.0 - 180.0
        i = np.round((lat - self.lat[0]) / (self.lat[1] - self.lat[0])).astype(int)
        j = np.round((lon - self.lon[0]) / (self.lon[1] - self.lon[0])).astype(int)
        ok = (i >= 0) & (i < self.lat.size) & (j >= 0) & (j < self.lon.size)
        out = np.full(lat.shape, -1, int)
        out[ok] = self.unit[i[ok], j[ok]]
        return out

    def label(self, k: int) -> str:
        return f"{self.names[k]}, {self.countries[k]}"

    def save(self, path) -> None:
        np.savez_compressed(path, lat=self.lat, lon=self.lon, unit=self.unit, elevation=self.elevation,
                            names=np.array(self.names), countries=np.array(self.countries),
                            attribution=np.array(self.attribution))

    @classmethod
    def load(cls, path) -> "RegionMap":
        d = np.load(path, allow_pickle=False)
        return cls(d["lat"], d["lon"], d["unit"], d["elevation"], d["names"].tolist(), d["countries"].tolist(),
                   str(d["attribution"]))

    @classmethod
    def build(cls, countries: dict | list = CARIBBEAN, box: tuple = CARIBBEAN_BOX, step_deg: float = 0.03,
              cache: str | Path = "~/.cache/naturev1/region", zoom: int | None = None, verbose: bool = True) -> "RegionMap":
        """Download, rasterise and cache the map of ``countries`` (ISO-3 codes, or ``{code: name}``) over ``box``."""
        cache = Path(cache).expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        if isinstance(countries, (list, tuple)):
            countries = {c: c for c in countries}
        s, n, w, e = box
        lat = np.arange(s, n + 1e-9, step_deg)
        lon = np.arange(w, e + 1e-9, step_deg)
        unit = np.full((lat.size, lon.size), -1, np.int32)
        names, owners, boxes, credits = [], [], [], set()
        for iso, country in countries.items():
            try:
                units, meta = fetch_units(iso, cache)
            except Exception as error:
                if verbose:
                    print(f"  {iso}: no boundaries ({error})")
                continue
            credits.add(f"{meta.get('boundarySource', 'geoBoundaries')} ({meta.get('boundaryLicense', 'see source')})")
            for u in units:
                mask = _rasterise(u["rings"], lat, lon) & (unit < 0)
                k = len(names)
                if not mask.any():                     # smaller than a cell: its nearest cell to the centroid
                    pts = np.concatenate([r[0] for r in u["rings"]])
                    cy, cx = pts[:, 1].mean(), pts[:, 0].mean()
                    i, j = int(round((cy - s) / step_deg)), int(round((cx - w) / step_deg))
                    if not (0 <= i < lat.size and 0 <= j < lon.size) or unit[i, j] >= 0:
                        continue
                    mask = np.zeros_like(mask)
                    mask[i, j] = True
                unit[mask] = k
                names.append(u["name"])
                owners.append(country)
                pts = np.concatenate([r[0] for r in u["rings"]])
                boxes.append((pts[:, 1].min() - 0.1, pts[:, 1].max() + 0.1, pts[:, 0].min() - 0.1, pts[:, 0].max() + 0.1))
            if verbose:
                print(f"  {iso}: {len(units)} units")
        zoom = zoom or int(np.clip(round(math.log2(360.0 / (256 * step_deg / 3.0))), 5, 11))
        elevation = terrain(lat, lon, cache, zoom, boxes)
        elevation = np.where(unit >= 0, np.maximum(elevation, 0.0), 0.0).astype(np.float32)
        attribution = ("Boundaries: geoBoundaries gbOpen (Runfola et al. 2020) -- " + "; ".join(sorted(credits))
                       + ". Terrain: Terrarium tiles, AWS Open Data (SRTM, GMTED, NED, ETOPO1 and others).")
        return cls(lat, lon, unit, elevation, names, owners, attribution)


def synthetic_island(lat0: float = 18.1, lon0: float = -77.3, half_lat: float = 0.45, half_lon: float = 1.2,
                     step: float = 0.02, parishes: int = 4, peak_m: float = 2000.0) -> RegionMap:
    """An elliptical island split into ``parishes`` west-to-east strips with a ridge along its spine, for tests."""
    lat = np.arange(lat0 - 1.5, lat0 + 1.5 + 1e-9, step)
    lon = np.arange(lon0 - 2.5, lon0 + 2.5 + 1e-9, step)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    inside = ((la - lat0) / half_lat) ** 2 + ((lo - lon0) / half_lon) ** 2 <= 1.0
    strip = np.clip(((lo - (lon0 - half_lon)) / (2 * half_lon) * parishes).astype(int), 0, parishes - 1)
    unit = np.where(inside, strip, -1).astype(np.int32)
    ridge = peak_m * np.exp(-((la - lat0) / (0.4 * half_lat)) ** 2) * np.clip(1 - ((lo - lon0) / half_lon) ** 2, 0, 1)
    elevation = np.where(inside, ridge, 0.0).astype(np.float32)
    names = [f"Parish {k + 1}" for k in range(parishes)]
    return RegionMap(lat, lon, unit, elevation, names, ["Testland"] * parishes, "synthetic")


__all__ = ["CARIBBEAN", "CARIBBEAN_BOX", "RegionMap", "fetch_units", "synthetic_island", "terrain"]
