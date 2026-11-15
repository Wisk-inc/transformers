# Copyright 2026 Nathan. Apache-2.0.
"""The live tool's pieces that need no network: cycles, the shipped prior, state round trips and the bulletin."""

import json

import numpy as np

from pluvium.cyclone import live
from pluvium.cyclone.cone import ConeLearner
from pluvium.cyclone.fields import SYSTEMS
from pluvium.cyclone.learner import SeasonLearner
from pluvium.cyclone.products import Forecaster, Guidance
from pluvium.cyclone.report import bulletin, compass, motion, position, stamp, to_json, track_csv
from pluvium.util import hour_of


def test_latest_cycle_waits_for_the_fields():
    assert live.latest_cycle(hour_of("2025-10-25T22")) == hour_of("2025-10-25T12")
    assert live.latest_cycle(hour_of("2025-10-25T20")) == hour_of("2025-10-25T00")       # 12 UTC is not out yet
    assert live.latest_cycle(hour_of("2025-10-25T09")) == hour_of("2025-10-25T00")
    assert live.latest_cycle(hour_of("2025-10-25T08")) == hour_of("2025-10-24T12")


def test_recent_starts_include_a_storm_with_no_future_yet():
    class Track:
        hours = hour_of("2025-10-20T00") + 6 * np.arange(24)                             # six days of fixes

        def tropical(self, hour):
            return hour >= hour_of("2025-10-21T00")

    class Storm:
        track = Track()

    cycle = hour_of("2025-10-25T12")
    starts = live.recent_starts([Storm()], cycle, replay_days=3)
    assert starts[-1] == cycle and starts[0] == hour_of("2025-10-22T12") and np.all(np.diff(starts) == 12)
    assert live.recent_starts([Storm()], hour_of("2025-10-20T12"), 3).size == 0          # not a cyclone yet


def test_the_shipped_prior_loads_into_a_forecaster():
    state = live.prior_state()
    assert state is not None and state["learner"]["systems"] == list(SYSTEMS)
    f = Forecaster(list(SYSTEMS))
    assert abs(f.learner.at(96)["aifs"] - 1 / 6) < 1e-9 and f.cone.rows == {}
    f.load_state(state)
    w = f.learner.at(96)
    assert abs(sum(w.values()) - 1.0) < 1e-9 and w["aifs"] > 0.25 > w["gfs"]             # what two seasons taught
    assert 60 < f.cone.base(72) / 1.852 < 110 and len(f.cone.rows[12]) > 100             # nautical miles at 72 hours
    again = Forecaster(list(SYSTEMS))
    again.load_state(json.loads(json.dumps(f.state())))                                  # survives a JSON round trip
    assert again.learner.at(96) == f.learner.at(96) and abs(again.cone.base(72) - f.cone.base(72)) < 1e-9


def test_state_moves_between_learners_with_different_systems():
    rng = np.random.default_rng(0)
    a = SeasonLearner(["x", "y", "z"])
    for hour in range(60):
        a._update(hour * 12, 48, {"x": rng.normal(0, 30, 2), "y": rng.normal(0, 90, 2), "z": rng.normal(0, 60, 2)})
    b = SeasonLearner(["y", "new", "x"])
    b.load_state(a.state())
    w = b.at(48)
    assert w["x"] > w["new"] > w["y"]                                                    # the newcomer is average
    cone = ConeLearner()
    for e in rng.rayleigh(50.0, 80):
        cone.add(24, e, float("nan") if rng.random() < 0.2 else rng.lognormal(4, 0.3))
    other = ConeLearner()
    other.load_state(json.loads(json.dumps(cone.state())))
    assert abs(other.base(24) - cone.base(24)) < 1e-9 and abs(other.slope() - cone.slope()) < 1e-9


def test_bulletin_says_where_when_and_how_sure(tmp_path):
    init = hour_of("2025-10-25T12")
    track = {lead: (16.3 + 0.012 * lead, -75.0 - 0.02 * lead) for lead in range(12, 121, 12)}
    g = Guidance("AL132025", init, (16.3, -75.0), track, {k: 60.0 for k in track}, {k: 50.0 + k for k in track},
                 {k: 50.0 + k for k in track})
    g.places = {"Kingston": {"chance": 0.88, "raw_chance": 0.8, "track_km": 47.0, "track_hour": 75.0},
                "Nassau": {"chance": 0.01, "raw_chance": 0.0, "track_km": float("nan"), "track_hour": float("nan")}}
    text = bulletin(g, "Melissa")
    assert "PLUVIUM CYCLONE: Melissa" in text and "2025-10-25 12 UTC" in text and "16.3N 75.0W" in text
    assert "Kingston" in text and "88%" in text and "Nassau" not in text and "within 47 km at 75 h" in text
    assert "moving W" in text and "not a forecast of wind" in text and text.count("\n   ") >= 5
    assert "nearly stationary" in motion(Guidance("S", init, (10.0, -50.0), {24: (10.05, -50.05)}, {}, {}, {}))
    assert compass(0) == "N" and compass(95) == "E" and compass(359) == "N" and compass(202.5) == "SSW"
    assert position(-12.34, 170.0) == "12.3S 170.0E" and position(20.0, 285.0) == "20.0N 75.0W"
    assert stamp(init) == "2025-10-25 12 UTC"
    data = to_json(g, "Melissa")
    assert json.loads(json.dumps(data))["places"]["Nassau"]["track_km"] is None and data["cone_km"]["72"] == 122.0
    path = track_csv(g, tmp_path / "track.csv")
    rows = path.read_text().strip().splitlines()
    assert rows[0] == "lead_hours,valid_utc,lat,lon,cone_km" and len(rows) == 11 and rows[1].startswith("12,2025-10-26 00 UTC")
    quiet = Guidance("S", init, None, track, {}, {k: 50.0 for k in track}, {})
    quiet.places = {"Nassau": g.places["Nassau"]}
    assert "No listed place" in bulletin(quiet)
