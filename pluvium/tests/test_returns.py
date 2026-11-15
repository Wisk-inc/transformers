# Copyright 2026 Nathan. Apache-2.0.
"""Return periods: the generalised Pareto fit recovers a known tail, and the answers respect the record's limits."""

import numpy as np

from pluvium.tails import gpd
from pluvium.tails.returns import DAYS_PER_YEAR, ReturnLevels

SHAPE, SCALE, WET = 0.15, 9.0, 0.3                                        # the truth the synthetic record is drawn from


def record(years=8.0, n=150, seed=0):
    """Daily rain at ``n`` places: wet on 30% of days, wet-day amounts generalised Pareto; wetter towards the end of
    the list of places (a rain shadow at one end, a windward coast at the other)."""
    rng = np.random.default_rng(seed)
    days = int(years * DAYS_PER_YEAR)
    wet = rng.random((days, n)) < WET
    factor = np.linspace(0.6, 1.6, n)
    amounts = gpd.quantile(rng.random((days, n)), SHAPE, SCALE) * factor
    return np.where(wet, amounts, 0.0).astype(np.float32), factor


def true_level(years: float, factor: np.ndarray) -> np.ndarray:
    """The amount exceeded once in ``years`` years on average, from the generating distribution."""
    p_day = 1.0 / (years * DAYS_PER_YEAR)                                 # chance on a day
    return gpd.quantile(1.0 - p_day / WET, SHAPE, SCALE) * factor


def test_pwm_recovers_shape_and_scale_and_the_functions_are_inverses():
    rng = np.random.default_rng(1)
    for shape, scale in ((0.0, 5.0), (0.2, 12.0), (-0.1, 8.0)):
        x = gpd.quantile(rng.random(40_000), shape, scale)
        s, c = gpd.fit_pwm(x)
        assert abs(s - shape) < 0.03 and abs(c / scale - 1) < 0.05
        p = np.array([0.1, 0.5, 0.9, 0.99])
        np.testing.assert_allclose(1.0 - gpd.survival(gpd.quantile(p, shape, scale), shape, scale), p, atol=1e-9)
        assert abs(gpd.mean_excess(shape, scale) - x.mean()) < 0.05 * x.mean()
    assert gpd.fit_pwm(np.array([3.0]))[0] == 0.0 and np.isnan(gpd.fit_pwm(np.array([]))[1])
    assert gpd.survival(1e6, -0.2, 5.0) == 0.0                            # a bounded tail ends
    assert gpd.fit_pwm(np.r_[np.full(50, 1e-3), 1e6])[0] == gpd.SHAPE_RANGE[1]   # clipped, not wild


def test_return_levels_match_the_truth_within_the_record():
    daily, factor = record()
    daily[:40, :5] = np.nan                                               # a gap in the record
    levels = ReturnLevels.fit(daily)
    assert abs(levels.shape - SHAPE) < 0.08 and abs(levels.years - 8.0) < 0.2
    assert np.all(levels.threshold >= 5.0) and levels.threshold[-1] > levels.threshold[0]
    for years in (1.0, 5.0):
        got, want = levels.level(years), true_level(years, factor)
        assert abs(np.mean(got / want) - 1.0) < 0.08                      # right on average over the area
        assert np.mean(np.abs(got / want - 1.0)) < 0.2                    # and not far off at any one place
        back = levels.period(got)
        np.testing.assert_allclose(back, years, rtol=1e-4)                # period and level are inverses
    wetter = levels.period(np.full(150, 60.0))
    assert wetter[0] > 3 * wetter[-1]                                     # 60 mm is far rarer in the rain shadow
    assert np.all(levels.period(np.full(150, 1.0)) == 0.0)                # an ordinary amount: no period
    assert levels.period(np.array([60.0, 60.0]), index=np.array([0, 149])).shape == (2,)


def test_wording_states_the_limit_of_the_record_and_levels_round_trip(tmp_path):
    daily, _ = record(years=2.0, n=40, seed=3)
    levels = ReturnLevels.fit(daily)
    assert abs(levels.horizon - 8.0) < 0.5
    assert levels.wording(0.1) == "" and levels.wording(float("nan")) == ""
    assert levels.wording(0.5) == "a few times a year" and levels.wording(1.0) == "about once a year"
    assert levels.wording(3.2) == "about once in 3 years"
    assert levels.wording(500.0) == "rarer than once in 8 years"          # no false precision beyond the record
    back = ReturnLevels.load(levels.save(tmp_path / "returns.npz"))
    np.testing.assert_allclose(back.level(2.0), levels.level(2.0))
    assert back.shape == levels.shape and back.years == levels.years
    dry = ReturnLevels.fit(np.zeros((400, 6), np.float32))               # a place where it never rains
    assert np.all(dry.threshold == 5.0) and np.all(np.isfinite(dry.level(2.0))) and np.all(dry.period(np.full(6, 2.0)) == 0)
