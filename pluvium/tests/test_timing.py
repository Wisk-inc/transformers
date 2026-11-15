# Copyright 2026 Nathan. Apache-2.0.
"""When in the day: six-hour periods from made-up runs where the right answer is known."""

import numpy as np
import pytest

from pluvium import timing as T
from pluvium.util import hour_of

DAY0 = hour_of("2026-11-15T00")


def run(init=DAY0, members=3):
    """A run on a 2 x 2 grid. Member 0: 3 mm in the last period of each day. Member 1: drizzle, 0.8 mm a day,
    never a wet period nor a wet day. Member 2: 2 mm in every period."""
    steps = np.array(T.steps_for(init, DAY0))
    values = np.zeros((members, steps.size, 2, 2), np.float32)
    for k, step in enumerate(steps):
        periods = (step - (DAY0 - init)) // T.HOURS                    # whole periods since the first day began
        values[0, k] = 3.0 * (periods // T.PERIODS)
        values[1, k] = 0.2 * periods
        values[2, k] = 2.0 * periods
    return {"values": values, "steps": steps, "lat": np.array([18.0, 18.25]), "lon": np.array([-77.5, -77.25])}


def test_steps_and_periods_follow_the_run():
    assert T.steps_for(DAY0, DAY0) == list(range(6, 73, 6))
    assert T.steps_for(DAY0 - 24, DAY0) == list(range(24, 97, 6))        # yesterday's run: a day further in
    with pytest.raises(ValueError):
        T.steps_for(DAY0 + 24, DAY0)                                     # a run from the future
    amounts = T.period_amounts(run(), DAY0, DAY0)
    assert amounts.shape == (3, 12, 2, 2)
    assert amounts[0, :4, 0, 0].tolist() == [0.0, 0.0, 0.0, 3.0] and np.allclose(amounts[2], 2.0)
    late = run(DAY0 - 24)
    late["values"] += 5.0                                                # rain that fell before the first day
    assert np.allclose(T.period_amounts(late, DAY0 - 24, DAY0)[2], 2.0)  # ... is not counted in it
    short = {**run(), "steps": run()["steps"][:6], "values": run()["values"][:, :6]}
    cut = T.period_amounts(short, DAY0, DAY0)
    assert np.isfinite(cut[:, :6]).all() and np.isnan(cut[:, 6:]).all()  # what the run does not reach is unknown
    dipped = run()
    dipped["values"][2, 5] -= 13.0                                       # packed files can dip between steps
    assert T.period_amounts(dipped, DAY0, DAY0).min() == 0.0


def test_chance_of_a_period_is_the_calibrated_day_spread_by_the_members():
    z = run()
    at = T.at_places(T.period_amounts(z, DAY0, DAY0), z, np.array([18.1, 18.2]), np.array([-77.4, -77.3]))
    assert at.shape == (3, 12, 2)
    shares = T.member_shares({"toy": at})
    assert np.allclose(shares["wet_day"], 2 / 3)                         # the drizzle member has no wet day
    assert np.allclose(shares["wet_period"][0, :, 0], [1 / 3, 1 / 3, 1 / 3, 2 / 3]) and shares["systems"].tolist() == [1, 1, 1]
    given = T.conditional(shares)
    assert np.allclose(given[0, :, 0], [0.5, 0.5, 0.5, 1.0])             # of the wet-day members, how many are wet then
    chance = T.chances(np.full((3, 2), 0.6), given)
    assert np.allclose(chance[0, :, 0], [0.3, 0.3, 0.3, 0.6])
    assert (chance <= 0.6 + 1e-9).all()                                  # a period is never likelier than its day
    # two systems count the same, whatever their number of members
    dry = np.zeros((40, 12, 2), np.float32)
    both = T.member_shares({"toy": at, "dry": dry})
    assert np.allclose(both["wet_day"], 1 / 3) and both["systems"].tolist() == [2, 2, 2]
    # a system missing a period of a day is left out of that day only
    gap = at.copy()
    gap[:, 5] = np.nan
    part = T.member_shares({"toy": gap})
    assert part["systems"].tolist() == [1, 0, 1] and np.allclose(part["wet_day"][1], 0.0)
    assert T.member_shares({})["systems"].tolist() == [0, 0, 0]


def test_where_no_member_is_wet_the_prior_decides_and_usual_timing_is_learned_from_truth():
    shares = {"wet_day": np.zeros((3, 2)), "wet_period": np.zeros((3, 4, 2))}
    assert np.allclose(T.conditional(shares), 0.25)                      # nothing to go on: each period alike
    prior = np.broadcast_to(np.array([0.1, 0.1, 0.2, 0.7])[None, :, None], (3, 4, 2))
    assert np.allclose(T.conditional(shares, prior)[0, :, 0], [0.1, 0.1, 0.2, 0.7])
    sure = {"wet_day": np.ones((3, 2)), "wet_period": np.zeros((3, 4, 2))}
    assert T.conditional(sure, prior)[0, 3, 0] < 0.1                     # members that are sure outweigh the prior
    # truth: at cell 0 it always rains in the last period, at cell 1 in the first; cell 2 is never wet
    days = 60
    rain = np.zeros((days, 4, 3), np.float32)
    rain[:, 3, 0] = 5.0
    rain[:, 0, 1] = 5.0
    months = np.full(days, 7)
    usual = T.usual_timing(rain, months)
    assert usual.shape == (12, 4, 3)
    assert usual[6, :, 0].tolist() == [0, 0, 0, 1] and usual[6, :, 1].tolist() == [1, 0, 0, 0]
    assert np.allclose(usual[6, :, 2], [0.5, 0, 0, 0.5])                 # no wet day of its own: the cells' common answer
    assert np.allclose(usual[0], 0.25)                                   # a month with no truth: each period alike
    assert np.allclose(T.mixed(np.ones((3, 4, 2)), np.zeros((3, 4, 2)), (0.25, 0.5, 1.0))[:, 0, 0], [0.25, 0.5, 1.0])
    assert T.mixed(np.ones((3, 4, 2)), None) is not None


def test_periods_are_named_in_the_place_s_own_clock():
    names = T.labels(DAY0, "America/Jamaica")
    assert len(names) == 12 and names[:4] == ["Sat 7 pm to 1 am", "Sun 1 am to 7 am", "Sun 7 am to 1 pm", "Sun 1 pm to 7 pm"]
    assert T.labels(DAY0, "Asia/Kolkata")[0] == "Sun 5:30 am to 11:30 am"
    assert T.labels(DAY0, "no/such-zone")[0] == "Sun 12 am to 6 am"      # an unknown zone: UTC
