# Copyright 2026 Nathan. Apache-2.0.
"""
Where Pluvium forecasts: one country, several, or any box between 60 S and 60 N.

Nothing in the engine is tied to Jamaica or the Caribbean. The mesh is built on the sphere from whatever land the
area holds. Terrain (Terrarium), borders and first-level divisions (geoBoundaries), towns (GeoNames), satellite rain
(CMORPH), airports (ISD, METAR) and the driving models (GFS, ECMWF) are all global. An :class:`Area` says which part of
the world a deployment covers:

* ``countries``: ISO 3166 codes (borders, parishes or provinces, towns);
* ``target_box``: where forecasts are made (the countries' extent plus a margin);
* ``domain``: where the mesh and the driving fields reach. It is wider, so weather has room to arrive from upwind;
* ``languages``: the bulletin languages for each country, written with fixed phrasing (never generated);
* ``satellite``: the live "rain now" source. GOES-East covers the Americas from 135 W to 15 W; GOES-West covers the
  Pacific side. Elsewhere Pluvium forecasts without it, and it was trained to: a third of training forecasts have no
  rain-now input.

``Area.for_countries(["TT"])`` builds one from country codes. ``Area.caribbean()`` is the original deployment, and
geographies saved before areas existed load as it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .util import fetch, log

COUNTRY_INFO = "https://download.geonames.org/export/dump/countryInfo.txt"
#: CMORPH, the truth Pluvium learns from, covers 60 S to 60 N.
LAT_LIMIT = 60.0
#: Languages the bulletins are written in (fixed phrasing).
WRITTEN = ("en", "es", "fr", "jam")
#: Languages added for some countries beyond their official ones.
EXTRA_LANGUAGES = {"JM": ("jam",)}
#: The largest area one deployment should cover; split bigger countries by province (``--box``).
MAX_SPAN_DEG = 30.0


@dataclass
class Area:
    name: str
    countries: dict                                   # ISO3 -> name
    iso2: tuple                                       # the same countries' ISO2 codes (GeoNames)
    target_box: tuple                                 # south, north, west, east
    domain: tuple
    languages: dict = field(default_factory=dict)     # ISO2 -> ("en", "jam")

    # ----------------------------------------------------------------------------------- building --
    @classmethod
    def caribbean(cls) -> Area:
        from naturev1.swarm.geography import CARIBBEAN

        from .domain import CARIBBEAN_ISO2, DOMAIN, TARGET_BOX

        info = country_info()
        langs = {cc: languages_for(cc, info) for cc in CARIBBEAN_ISO2} if info else {"JM": ("en", "jam")}
        return cls("caribbean", dict(CARIBBEAN), tuple(CARIBBEAN_ISO2), tuple(TARGET_BOX), tuple(DOMAIN), langs)

    @classmethod
    def for_countries(cls, codes, name: str | None = None, margin_deg: float = 0.3, upstream_deg: float = 6.0,
                      box=None, cache="~/.cache/pluvium/areas") -> Area:
        """An area from ISO2 or ISO3 codes. Its extent comes from geoBoundaries' national outline unless ``box`` is
        given; the domain adds ``upstream_deg`` on every side."""
        info = country_info(cache)
        iso2, countries = [], {}
        for code in codes:
            row = _lookup(code, info)
            iso2.append(row["iso2"])
            countries[row["iso3"]] = row["name"]
        if box is None:
            boxes = [national_box(iso3, cache) for iso3 in countries]
            box = (min(b[0] for b in boxes) - margin_deg, max(b[1] for b in boxes) + margin_deg,
                   min(b[2] for b in boxes) - margin_deg, max(b[3] for b in boxes) + margin_deg)
        s, n, w, e = (float(x) for x in box)
        if n - s > MAX_SPAN_DEG or e - w > MAX_SPAN_DEG:
            raise ValueError(f"{'/'.join(iso2)} spans {n - s:.0f} x {e - w:.0f} degrees; give a smaller box "
                             "(one deployment per region of a large country)")
        if s < -LAT_LIMIT or n > LAT_LIMIT:
            raise ValueError(f"the area reaches {max(abs(s), abs(n)):.0f} degrees of latitude; CMORPH, the truth "
                             f"Pluvium learns from, stops at {LAT_LIMIT:.0f}")
        domain = (max(-LAT_LIMIT, s - upstream_deg), min(LAT_LIMIT, n + upstream_deg), w - upstream_deg, e + upstream_deg)
        langs = {cc: languages_for(cc, info) for cc in iso2}
        return cls(name or "-".join(c.lower() for c in iso2), countries, tuple(iso2), (s, n, w, e), domain, langs)

    # --------------------------------------------------------------------------------- properties --
    @property
    def centre(self) -> tuple[float, float]:
        s, n, w, e = self.target_box
        return 0.5 * (s + n), 0.5 * (w + e)

    @property
    def satellite(self) -> str:
        """``goes-east``, ``goes-west`` or ``none``: the geostationary view of the area for live "rain now"."""
        lat, lon = self.centre
        if abs(lat) > 55:
            return "none"
        if -135.0 <= lon <= -15.0:
            return "goes-east"
        if lon < -135.0 or lon >= 150.0:
            return "goes-west"
        return "none"

    def bulletin_languages(self, iso2: str) -> tuple:
        return tuple(self.languages.get(iso2, ("en",)))

    # ---------------------------------------------------------------------------------------- io --
    def to_json(self) -> dict:
        d = asdict(self)
        d["languages"] = {k: list(v) for k, v in self.languages.items()}
        return d

    @classmethod
    def from_json(cls, d: dict) -> Area:
        return cls(d["name"], dict(d["countries"]), tuple(d["iso2"]), tuple(d["target_box"]), tuple(d["domain"]),
                   {k: tuple(v) for k, v in d.get("languages", {}).items()})

    def save(self, path) -> Path:
        path = Path(path).expanduser()
        path.write_text(json.dumps(self.to_json(), indent=1))
        return path

    @classmethod
    def load(cls, path) -> Area:
        return cls.from_json(json.loads(Path(path).expanduser().read_text()))


# ------------------------------------------------------------------------------------------- lookups --

def parse_country_info(text: str) -> dict:
    """GeoNames' countryInfo.txt -> {ISO2: {"iso2", "iso3", "name", "languages"}}."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split("\t")
        if len(p) < 16:
            continue
        out[p[0]] = {"iso2": p[0], "iso3": p[1], "name": p[4], "languages": p[15]}
    return out


def country_info(cache="~/.cache/pluvium/areas") -> dict:
    path = Path(cache).expanduser() / "countryInfo.txt"
    if not path.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(fetch(COUNTRY_INFO, timeout=60.0, attempts=4))
        except Exception as error:
            log(f"country list not available ({type(error).__name__}); using defaults")
            return {}
    return parse_country_info(path.read_text(encoding="utf-8", errors="replace"))


def _lookup(code: str, info: dict) -> dict:
    code = code.strip().upper()
    if code in info:
        return info[code]
    for row in info.values():
        if row["iso3"] == code:
            return row
    raise ValueError(f"unknown country code {code!r}")


def languages_for(iso2: str, info: dict) -> tuple:
    """The bulletin languages for a country: its official languages that Pluvium writes (first listed first), plus
    any in :data:`EXTRA_LANGUAGES`; English when none is written."""
    listed = [lang.split("-")[0] for lang in info.get(iso2, {}).get("languages", "").split(",") if lang]
    langs = [lang for lang in listed if lang in WRITTEN]
    langs = list(dict.fromkeys(langs)) or ["en"]
    return tuple(langs + [x for x in EXTRA_LANGUAGES.get(iso2, ()) if x not in langs])


def national_box(iso3: str, cache="~/.cache/pluvium/areas") -> tuple:
    """``(south, north, west, east)`` of a country's outline from geoBoundaries (cached)."""
    from naturev1.swarm.geography import fetch_units

    folder = Path(cache).expanduser() / "boundaries"
    folder.mkdir(parents=True, exist_ok=True)
    units, _ = fetch_units(iso3, folder, "ADM0")
    lats, lons = [], []
    for unit in units:
        for exterior, _holes in unit["rings"]:
            lons.append(exterior[:, 0])
            lats.append(exterior[:, 1])
    lat, lon = np.concatenate(lats), np.concatenate(lons)
    if lon.max() - lon.min() > 180:                       # straddles the date line: work in 0-360
        lon = lon % 360.0
    return float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())


__all__ = ["Area", "LAT_LIMIT", "WRITTEN", "country_info", "languages_for", "national_box", "parse_country_info"]
