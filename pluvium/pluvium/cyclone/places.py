# Copyright 2026 Nathan. Apache-2.0.
"""
Places a storm may pass: coastal cities and islands of the hurricane basins.

A short gazetteer for passage chances (:mod:`.approach`): about forty places around the North Atlantic and twenty
around the eastern North Pacific. Coordinates are to a tenth of a degree, enough for "within 150 km". Add your own
with :func:`register`.
"""

from __future__ import annotations

import numpy as np

from .geodesy import distance_km

PLACES = {
    "natl": {
        "Kingston": (18.0, -76.8), "Montego Bay": (18.5, -77.9), "Santiago de Cuba": (20.0, -75.8),
        "Havana": (23.1, -82.4), "Port-au-Prince": (18.5, -72.3), "Santo Domingo": (18.5, -69.9),
        "San Juan": (18.5, -66.1), "Philipsburg": (18.0, -63.0), "Pointe-a-Pitre": (16.2, -61.5),
        "Roseau": (15.3, -61.4), "Bridgetown": (13.1, -59.6), "Port of Spain": (10.7, -61.5),
        "Willemstad": (12.1, -68.9), "George Town": (19.3, -81.4), "Nassau": (25.0, -77.3),
        "Cockburn Town": (21.5, -71.1), "Hamilton": (32.3, -64.8), "Belize City": (17.5, -88.2),
        "Cancun": (21.2, -86.8), "Puerto Cabezas": (14.0, -83.4), "Bluefields": (12.0, -83.8),
        "San Andres": (12.6, -81.7), "Veracruz": (19.2, -96.1), "Tampico": (22.2, -97.9),
        "Corpus Christi": (27.8, -97.4), "Houston": (29.8, -95.4), "New Orleans": (30.0, -90.1),
        "Mobile": (30.7, -88.0), "Tampa": (27.9, -82.5), "Miami": (25.8, -80.2), "Key West": (24.6, -81.8),
        "Jacksonville": (30.3, -81.7), "Charleston": (32.8, -79.9), "Wilmington": (34.2, -77.9),
        "Cape Hatteras": (35.2, -75.5), "Norfolk": (36.9, -76.3), "New York": (40.7, -74.0),
        "Boston": (42.4, -71.1), "Halifax": (44.6, -63.6), "St. John's": (47.6, -52.7),
        "Praia": (14.9, -23.5), "Ponta Delgada": (37.7, -25.7),
    },
    "epac": {
        "Tapachula": (14.9, -92.3), "Salina Cruz": (16.2, -95.2), "Puerto Escondido": (15.9, -97.1),
        "Acapulco": (16.9, -99.9), "Zihuatanejo": (17.6, -101.6), "Lazaro Cardenas": (18.0, -102.2),
        "Manzanillo": (19.1, -104.3), "Puerto Vallarta": (20.7, -105.2), "Mazatlan": (23.2, -106.4),
        "Los Mochis": (25.8, -109.0), "Guaymas": (27.9, -110.9), "Cabo San Lucas": (22.9, -109.9),
        "La Paz": (24.1, -110.3), "Ensenada": (31.9, -116.6), "San Diego": (32.7, -117.2),
        "Socorro Island": (18.8, -111.0), "Clarion Island": (18.4, -114.7), "Acajutla": (13.6, -89.8),
        "Hilo": (19.7, -155.1), "Honolulu": (21.3, -157.9), "Lihue": (22.0, -159.4),
    },
}


def register(basin: str, name: str, lat: float, lon: float) -> None:
    """Add a place to a basin's list."""
    PLACES.setdefault(basin, {})[name] = (float(lat), float(lon))


def near(basin: str, lat: float, lon: float, within_km: float = 2500.0) -> dict:
    """The basin's places within ``within_km`` of a position: ``{name: (lat, lon)}``."""
    out = {}
    for name, (plat, plon) in PLACES.get(basin, {}).items():
        if float(distance_km(lat, lon, plat, plon)) <= within_km:
            out[name] = (plat, plon)
    return out


def nearest(basin: str, lat: float, lon: float) -> tuple:
    """``(name, distance km)`` of the basin's place nearest a position."""
    names = list(PLACES.get(basin, {}))
    if not names:
        return None, float("nan")
    d = [float(distance_km(lat, lon, *PLACES[basin][n])) for n in names]
    k = int(np.argmin(d))
    return names[k], d[k]


__all__ = ["PLACES", "near", "nearest", "register"]
