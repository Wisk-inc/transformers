# Copyright 2026 Nathan. Apache-2.0.
import numpy as np

from pluvium.atmos.cyclones import find_cyclones, parametric_rain
from pluvium.atmos.diurnal import diurnal_features
from pluvium.atmos.moisture import bretherton_rate, column_relative_humidity
from pluvium.atmos.saharan import saharan_index
from pluvium.atmos.sea_breeze import sea_breeze_arrival
from pluvium.atmos.thermo import cape_cin, saturation_vapour_pressure, theta_e_bolton
from pluvium.util import hour_of


def tropical_profile(moist=1.0, warm_mid=0.0):
    t = {850: np.array([[292.0]]), 700: np.array([[283.0 + warm_mid]]), 500: np.array([[267.0]]),
         200: np.array([[221.0]])}
    q = {850: np.array([[0.014 * moist]]), 700: np.array([[0.008 * moist]]), 500: np.array([[0.003 * moist]]),
         200: np.array([[0.0001]])}
    return t, q


def test_thermodynamics_reference_values():
    assert abs(saturation_vapour_pressure(np.array(293.15)) - 23.37) < 0.1          # hPa at 20 C
    the = theta_e_bolton(np.array(295.0), np.array(0.015), 850.0)
    assert 340 < the < 360


def test_cape_grows_with_moisture_and_falls_with_mid_level_warmth():
    c_dry, _, _ = cape_cin(*tropical_profile(moist=0.6))
    c_wet, _, lcl_wet = cape_cin(*tropical_profile(moist=1.1))
    c_cap, _, _ = cape_cin(*tropical_profile(moist=1.1, warm_mid=4.0))
    assert c_wet.item() > c_dry.item() and c_wet.item() > c_cap.item() and lcl_wet.item() < 1500


def test_bretherton_law_and_crh():
    t, q = tropical_profile()
    _, _, crh = column_relative_humidity(t, q)
    assert 0.3 < crh.item() < 1.0
    assert bretherton_rate(np.array(0.9)) > 10 * bretherton_rate(np.array(0.7))


def test_saharan_index_only_in_season_and_when_dry():
    up = {"t700": np.full((2, 1, 1), 287.0), "q700": np.full((2, 1, 1), 0.002), "t500": np.full((2, 1, 1), 268.0),
          "q500": np.full((2, 1, 1), 0.0005), "u700": np.full((2, 1, 1), -14.0)}
    hours = np.array([hour_of("2021-07-01T12"), hour_of("2021-01-01T12")])
    s = saharan_index(up, hours)
    assert s[0, 0, 0] > 0.6 and s[1, 0, 0] == 0


def test_cyclone_detection_and_rain_profile():
    lat, lon = np.arange(10.0, 26.0, 1.0), np.arange(-85.0, -60.0, 1.0)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    r = np.hypot(la - 17.0, (lo + 75.0) * 0.95)
    msl = (101000 - 4000 * np.exp(-(r / 2.0) ** 2))[None]
    ang = np.arctan2(la - 17.0, lo + 75.0)
    speed = 30 * (r / 1.0) * np.exp(1 - r / 1.0)
    u, v = (-speed * np.sin(ang))[None], (speed * np.cos(ang))[None]
    found = find_cyclones(msl, u, v, lat, lon)[0]
    assert found and abs(found[0][0] - 17.0) <= 1 and abs(found[0][1] + 75.0) <= 1
    rain = parametric_rain(np.array([0.0, 50.0, 200.0, 600.0]), vmax=45.0)
    assert rain[1] > rain[2] > rain[3] and rain.max() < 60


def test_sea_breeze_reaches_inland_in_the_afternoon():
    hours = hour_of("2021-07-01T00") + np.arange(24)
    pulse, since = sea_breeze_arrival(np.array([15.0]), np.array([1.0]), np.array([0.0]), np.zeros((24, 1)),
                                      np.zeros((24, 1)), np.array([-77.0]), hours, np.array([True]))
    lst_peak = (hours[int(np.argmax(pulse[:, 0]))] + 0.5 - 77.0 / 15.0) % 24
    assert 10 <= lst_peak <= 18


def test_diurnal_clocks():
    hours = hour_of("2021-07-01T00") + np.arange(24)
    f = diurnal_features(np.array([18.0, 18.0]), np.array([-77.0, -77.0]), hours, np.array([1.0, 0.0]))
    lst = (hours + 0.5 - 77.0 / 15.0) % 24
    assert abs(lst[int(np.argmax(f[:, 0, 1]))] - 15.5) < 1.5          # land clock peaks mid-afternoon
    assert abs(lst[int(np.argmax(f[:, 1, 2]))] - 5.5) < 1.5           # sea clock before dawn
