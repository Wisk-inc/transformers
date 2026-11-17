# Copyright 2026 Nathan. Apache-2.0.
import numpy as np

from pluvium.sources.stations import parse_hourly
from pluvium.util import hour_of, year_hours
from pluvium.weather import class_mask, code_from_metar, decode, describe, metar_weather, rate_class


def test_metar_groups():
    w = metar_weather("METAR MKJP 030000Z 03009KT 8000 -SHRA BKN018 26/24 Q1015=")
    assert w["wet"] and w["intensity"] == 1 and w["kind"] == 3 and not w["thunder"]
    w = metar_weather("METAR TJSJ 141200Z 09012KT 3000 +TSRA BKN020CB 25/23 A2990")
    assert w["intensity"] == 3 and w["thunder"]
    w = metar_weather("METAR MKJS 140000Z 21003KT 9999 VCTS FEW020CB 28/26 Q1009")
    assert not w["wet"] and w["thunder"]
    w = metar_weather("METAR MKJP 022300Z 01013KT 8000 VCSH BKN018 27/23 Q1014=")
    assert not w["wet"]
    assert metar_weather("SPECI MDSD 141220Z 10004KT 4000 -DZ BR OVC008 24/23 Q1013")["kind"] == 1
    assert code_from_metar("METAR MKJP 140000Z 16006KT //// ////// 29/27 Q1009")[0] == -1
    assert code_from_metar("METAR MKJP 141300Z 10010KT 9999 FEW020 30/22 Q1012 RERA")[1] >= 0


def test_codes_and_masks():
    assert list(rate_class(np.array([0.0, 0.5, 3.0, 9.0, np.nan]))) == [0, 1, 2, 3, -1]
    codes = np.array([-1, 32, 33, 35, 1])
    m = class_mask(codes, 4)
    assert m[0].all() and m[1].tolist() == [True, False, False, False]
    assert m[2].tolist() == [False, True, False, False] and m[3].tolist() == [False, False, False, True]
    assert m[4].tolist() == [False, True, True, True]                  # wet, intensity not said
    assert decode(np.array([39]))["thunder"][0]
    assert describe(np.array([0.2, 0.2, 0.3, 0.3]), 0.6) .endswith("thunderstorms")


def test_isd_parsing_assigns_report_to_previous_hour():
    header = "STATION,DATE,REPORT_TYPE,MW1,AW1,AA1,REM\n"
    rows = ['"1","2021-03-01T14:00:00","FM-15","","","","MET01METAR MKJP 011400Z 10010KT 9999 -RA FEW020 30/22 Q1012"',
            '"1","2021-03-01T15:00:00","FM-15","","","","MET01METAR MKJP 011500Z 10010KT 9999 FEW020 30/22 Q1012"']
    out = parse_hourly(header + "\n".join(rows), 2021)
    first, _ = year_hours(2021)
    h13 = hour_of("2021-03-01T13") - first
    d = decode(out)
    assert d["wet"][h13] and d["intensity"][h13] == 1
    assert d["known"][h13 + 1] and not d["wet"][h13 + 1]


def test_the_airport_box_is_sent_with_two_decimals_and_rounded_outwards():
    """The Aviation Weather Center answers HTTP 400 to a box with all the decimals of a country's own extent; every
    live forecast for an area set by its countries ran without airport reports until 0.5.7."""
    from pluvium.sources.metar import API, bbox

    box = (16.4012383430461, 19.825031116881735, -79.66832347975166, -74.88458317352918)       # south, north, west, east
    got = bbox(box)
    assert got == {"s": "16.40", "w": "-79.67", "n": "19.83", "e": "-74.88"}
    assert float(got["s"]) <= box[0] and float(got["n"]) >= box[1] and float(got["w"]) <= box[2] and float(got["e"]) >= box[3]
    assert API.format(**bbox((10, 25, -90, -60)), hours=3).endswith("bbox=10.00,-90.00,25.00,-60.00&format=json&hours=3")
