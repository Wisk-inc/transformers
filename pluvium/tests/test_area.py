# Copyright 2026 Nathan. Apache-2.0.
"""Pluvium for any country: CMORPH windows anywhere, areas, languages, bulletins and geographies that remember them."""

import json

import numpy as np
import pytest

from pluvium.area import Area, languages_for, parse_country_info
from pluvium.domain import DOMAIN, cmorph_axes, cmorph_window, signed_lon, window_lon
from pluvium.render.bulletin import bulletins
from pluvium.sources.cmorph import read_window

INFO = parse_country_info("\n".join([
    "#ISO\tISO3\tISO-Numeric\tfips\tCountry\tCapital\tArea\tPopulation\tContinent\ttld\tCurrencyCode\tCurrencyName\tPhone\t"
    "Postal\tRegex\tLanguages\tgeonameid\tneighbours\tEquivalentFipsCode",
    "JM\tJAM\t388\tJM\tJamaica\tKingston\t10991\t2934855\tNA\t.jm\tJMD\tDollar\t+1-876\t\t\ten-JM\t3489940\t\t",
    "HT\tHTI\t332\tHA\tHaiti\tPort-au-Prince\t27750\t11123176\tNA\t.ht\tHTG\tGourde\t509\t\t\tht,fr-HT\t3723988\tDO\t",
    "DO\tDOM\t214\tDR\tDominican Republic\tSanto Domingo\t48730\t10627165\tNA\t.do\tDOP\tPeso\t+1-809\t\t\tes-DO\t3508796\tHT\t",
    "LK\tLKA\t144\tCE\tSri Lanka\tColombo\t65610\t21670000\tAS\t.lk\tLKR\tRupee\t94\t\t\tsi,ta,en\t1227603\t\t",
]))


def test_cmorph_window_unchanged_for_the_caribbean():
    rows, cols = cmorph_window(DOMAIN)
    lat, lon = cmorph_axes()
    old_cols = np.nonzero((lon >= DOMAIN[2] % 360) & (lon <= DOMAIN[3] % 360))[0]
    assert isinstance(cols, slice) and (cols.start, cols.stop) == (old_cols[0], old_cols[-1] + 1)
    np.testing.assert_allclose(window_lon(cols), lon[old_cols] - 360.0)


def test_cmorph_window_east_and_across_the_prime_meridian():
    _, cols = cmorph_window((5.0, 10.0, 79.0, 82.0))                 # Sri Lanka: eastern hemisphere
    lons = window_lon(cols)
    assert isinstance(cols, slice) and 79.0 <= lons.min() and lons.max() <= 82.0 and np.all(np.diff(lons) > 0)
    _, cols = cmorph_window((4.5, 11.5, -3.5, 1.5))                  # Ghana: crosses longitude 0
    lons = window_lon(cols)
    assert not isinstance(cols, slice) and lons.min() < -3.4 and lons.max() > 1.4 and np.all(np.diff(lons) > 0)
    data = np.arange(2 * 3 * 4948).reshape(2, 3, 4948)
    got = read_window(data, slice(0, 3), cols)
    np.testing.assert_array_equal(got, data[:, :, cols])
    with pytest.raises(ValueError):
        cmorph_window((0.0, 5.0, 175.0, 185.0))                       # the date line


def test_signed_lon():
    np.testing.assert_allclose(signed_lon([0.0, 179.0, 181.0, 359.5, -91.0]), [0.0, 179.0, -179.0, -0.5, -91.0])


def test_languages_from_geonames():
    assert languages_for("JM", INFO) == ("en", "jam")
    assert languages_for("HT", INFO) == ("fr",)
    assert languages_for("DO", INFO) == ("es",)
    assert languages_for("LK", INFO) == ("en",)


def test_area_for_countries(monkeypatch):
    import pluvium.area as area

    monkeypatch.setattr(area, "country_info", lambda cache=None: INFO)
    boxes = {"HTI": (18.0, 20.1, -74.5, -71.6), "DOM": (17.5, 19.95, -72.0, -68.3), "LKA": (5.9, 9.8, 79.5, 81.9)}
    monkeypatch.setattr(area, "national_box", lambda iso3, cache=None: boxes[iso3])
    a = Area.for_countries(["DO", "HTI"])
    assert a.iso2 == ("DO", "HT") and set(a.countries) == {"DOM", "HTI"}
    assert a.target_box == pytest.approx((17.2, 20.4, -74.8, -68.0))
    assert a.domain == pytest.approx((11.2, 26.4, -80.8, -62.0))
    assert a.satellite == "goes-east" and a.bulletin_languages("HT") == ("fr",)
    lk = Area.for_countries(["LK"])
    assert lk.satellite == "none" and lk.domain[2] > 70
    assert Area.from_json(json.loads(json.dumps(a.to_json()))) == a
    with pytest.raises(ValueError):
        Area.for_countries(["LK"], box=(40.0, 65.0, 10.0, 20.0))       # beyond CMORPH's 60 degrees


def test_bulletins_in_each_language():
    rows = [{"country": "DO", "parish": "Santiago", "any_rain_24h_typical": 0.7, "thunder_24h_highest": 0.4,
             "flood_highest": "low", "earliest_likely_start": "Sat 2 pm", "rain_24h_mm_p90_highest": 35.0,
             "summary": "Showers likely."}]
    out = bulletins(rows, "DO", "2026-11-14 06:00", ("es", "en", "fr"))
    assert set(out) == {"bulletin_DO.txt", "bulletin_DO_en.txt", "bulletin_DO_fr.txt"}
    assert "Boletín de lluvia" in out["bulletin_DO.txt"] and "probable" in out["bulletin_DO.txt"]
    assert "Bulletin de pluie" in out["bulletin_DO_fr.txt"] and "35 mm" in out["bulletin_DO_fr.txt"]
    jm = bulletins([dict(rows[0], country="JM", parish="Portland")], "JM", "x", ("en", "jam"))
    assert set(jm) == {"bulletin_JM.txt", "bulletin_JM_patois.txt"}


def test_geography_keeps_its_area(tmp_path):
    from pluvium.domain import Geography
    from pluvium.synthetic import synthetic_geography

    geo = synthetic_geography(verbose=False)
    geo.area = Area("test", {"JAM": "Jamaica"}, ("JM",), (16.0, 20.5, -80.0, -74.5), (10.0, 26.0, -86.0, -68.0),
                    {"JM": ("en", "jam")})
    geo.save(tmp_path / "g")
    back = Geography.load(tmp_path / "g")
    assert back.area == geo.area and back.cmorph_box == geo.cmorph_box and back.target_box == geo.target_box
    meta = json.loads((tmp_path / "g" / "geography.json").read_text())
    for key in ("area", "cmorph_box", "mesh_box"):
        meta.pop(key)
    (tmp_path / "g" / "geography.json").write_text(json.dumps(meta))
    old = Geography.load(tmp_path / "g")                               # files from before areas existed
    assert old.area is None and old.cmorph_box == tuple(meta["domain"])
