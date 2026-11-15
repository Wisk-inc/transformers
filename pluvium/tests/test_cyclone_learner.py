# Copyright 2026 Nathan. Apache-2.0.
"""The season learner: weights from verified forecasts only, a fair start for new systems, and the selection rule."""

import numpy as np

from pluvium.benchmarks.cyclone_bench import Season, choose, deck_points, ratio_to, tally
from pluvium.cyclone.geodesy import distance_km, shift
from pluvium.cyclone.interp import extend, interpolate
from pluvium.cyclone.learner import SeasonLearner, typical_km
from pluvium.cyclone.minvar import RunningCovariance, solve_weights

LEADS = (12, 24, 36, 48, 60, 72, 96, 120)


def storm(start_hour, days=8, lat0=15.0, lon0=-40.0):
    """A storm moving west-north-west: real-time fixes ``{hour: {0: (lat, lon, wind)}}`` every 12 hours."""
    return {start_hour + 12 * k: {0: (lat0 + 0.35 * k, lon0 - 1.1 * k, 60.0)} for k in range(2 * days + 1)}


def forecast(carq, init, noise_km, rng, shared=None):
    """A system's forecast from ``init``: the truth plus an error that grows with lead (km per sqrt(12 h))."""
    out = {}
    for lead in LEADS:
        fix = carq.get(init + lead, {}).get(0)
        if fix is None:
            continue
        grow = np.sqrt(lead / 12.0)
        east, north = (rng.normal(0, noise_km, 2) if shared is None else shared[lead]) * grow
        lat, lon = shift(fix[0], fix[1], east, north)
        out[lead] = (float(lat), float(lon))
    return out


def season(learner, noises, rng, storms=6, same=()):
    """Run a learner through a season; returns the mean error (km) of its consensus and of the equal-weight mean."""
    from pluvium.cyclone.consensus import consensus

    mine, equal = [], []
    for s in range(storms):
        carq = storm(1000 + 400 * s)
        for init in sorted(carq)[:-2]:
            shared = {lead: rng.normal(0, noises[same[0]], 2) for lead in LEADS} if same else None
            points = {name: forecast(carq, init, km, rng, shared if name in same else None)
                      for name, km in noises.items()}
            learner.learn_until(init)
            got = learner.predict(f"S{s}", init, points, carq)
            plain = consensus(points)
            for lead, pos in got.items():
                fix = carq[init + lead][0]
                mine.append(float(distance_km(pos[0], pos[1], fix[0], fix[1])))
                equal.append(float(distance_km(plain[lead][0], plain[lead][1], fix[0], fix[1])))
    return float(np.mean(mine)), float(np.mean(equal))


def test_weights_start_equal_and_learning_waits_for_verification():
    rng = np.random.default_rng(0)
    learner = SeasonLearner(["a", "b", "c"])
    assert all(abs(w - 1 / 3) < 1e-9 for w in learner.at(24).values())
    carq = storm(1000)
    points = {name: forecast(carq, 1000, 30.0, rng) for name in ("a", "b", "c")}
    learner.predict("S", 1000, points, carq)
    assert learner.learn_until(1000 + 6) == 0 and learner.learned == 0             # nothing has verified yet
    assert learner.learn_until(1000 + 12) == 1                                     # the 12-hour lead, and only that
    assert learner.learn_until(1000 + 48) == 3 and len(learner.pending) == 4
    assert abs(sum(learner.at(24).values()) - 1.0) < 1e-9


def test_minimum_variance_weights_leave_a_poor_system_behind():
    rng = np.random.default_rng(1)
    noises = {"good": 25.0, "fair": 35.0, "poor": 110.0}
    learner = SeasonLearner(list(noises))
    mine, equal = season(learner, noises, rng)
    w = learner.at(48)
    assert w["poor"] < 0.15 < w["fair"] < w["good"]
    assert mine < 0.85 * equal                                                      # and the consensus is better for it
    eg = SeasonLearner(list(noises), method="eg", eta=0.1)
    mine_eg, equal_eg = season(eg, noises, np.random.default_rng(1))
    assert eg.at(48)["poor"] < 0.2 and mine_eg < 0.92 * equal_eg
    fixed = SeasonLearner(list(noises), method="equal")
    mine_eq, equal_eq = season(fixed, noises, np.random.default_rng(1))
    assert abs(mine_eq - equal_eq) < 1e-6


def test_systems_that_repeat_each_other_share_a_vote():
    rng = np.random.default_rng(2)
    noises = {"twin1": 40.0, "twin2": 40.0, "other": 40.0}
    learner = SeasonLearner(list(noises), shrink=0.0, to_equal=0.0)
    season(learner, noises, rng, storms=10, same=("twin1", "twin2"))
    w = learner.at(48)
    assert w["other"] > 0.42 and abs(w["twin1"] - w["twin2"]) < 0.15                # not one third each


def test_a_new_system_starts_as_an_average_member():
    rng = np.random.default_rng(3)
    cov = RunningCovariance(["a", "b", "new"])
    for hour in range(40):
        cov.add(hour * 12, {"a": rng.normal(0, 0.6, 2), "b": rng.normal(0, 1.2, 2)})
    w = dict(zip(cov.systems, cov.weights()))
    assert w["a"] > w["new"] > w["b"]                                               # neither trusted nor written off
    for hour in range(40, 80):
        cov.add(hour * 12, {"a": rng.normal(0, 0.6, 2), "b": rng.normal(0, 1.2, 2), "new": rng.normal(0, 0.2, 2)})
    assert dict(zip(cov.systems, cov.weights()))["new"] > 0.5                       # its record now speaks
    assert np.isnan(RunningCovariance(["x"]).rms()["x"])
    np.testing.assert_allclose(solve_weights(np.eye(4), 0.0), 0.25, atol=1e-6)
    lopsided = solve_weights(np.diag([1.0, 100.0]), 0.25)
    assert lopsided[0] > 0.8 and lopsided[1] >= 0.125 - 1e-9                        # the floor from equal weights


def test_old_evidence_fades():
    rng = np.random.default_rng(4)
    cov = RunningCovariance(["a", "b"], halflife_days=30.0, to_equal=0.0)
    for k in range(60):                                                             # a was poor two months ago
        cov.add(k * 12, {"a": rng.normal(0, 1.5, 2), "b": rng.normal(0, 0.5, 2)})
    early = cov.weights()[0]
    for k in range(60, 300):                                                        # and has been good since
        cov.add(k * 12, {"a": rng.normal(0, 0.5, 2), "b": rng.normal(0, 1.5, 2)})
    assert early < 0.25 and cov.weights()[0] > 0.75


def test_early_guidance_reaches_the_last_lead():
    track = {lead: (15.0 + lead / 40.0, -50.0 - lead / 12.0) for lead in range(12, 121, 12)}
    longer = extend(track, 12)
    assert 132 in longer and len(longer) == len(track) + 1
    step = distance_km(*track[108], *track[120])
    assert abs(distance_km(*track[120], *longer[132]) - step) < 0.01 * step         # the same speed
    early = interpolate(track, (15.5, -51.5), 12, extend_hours=12)
    assert max(early) == 120 and early[0] == (15.5, -51.5)
    assert max(interpolate(track, (15.5, -51.5), 12)) == 108
    assert extend({12: (10.0, -40.0)}, 12) == {12: (10.0, -40.0)}                   # nothing to extrapolate from
    assert typical_km(12) < typical_km(48) < typical_km(120) and abs(typical_km(84) - 212.5) < 1e-6


def test_the_selection_rule_makes_complexity_pay():
    assert choose({"simple": 0.995, "fancy": 0.985}) == "equal"                     # not two percent better
    assert choose({"simple": 0.962, "fancy": 0.960}) == "simple"                    # a tie goes to the simpler
    assert choose({"simple": 0.99, "fancy": 0.93}) == "fancy"
    assert abs(ratio_to({12: 50.0, 24: 90.0}, {12: 100.0, 24: 100.0, 36: 1.0}) - 0.7) < 1e-9
    aids = {"GDMN": {100: {0: (10.0, -40.0, 50), 12: (10.5, -41.0, 55), 132: (14.0, -50.0, 60)}}}
    assert deck_points(aids, "GDMN", 100) == {12: (10.5, -41.0), 132: (14.0, -50.0)}
    assert deck_points(aids, "GDMN", 100, {12, 24}) == {12: (10.5, -41.0)} and deck_points(aids, "EMX", 100) == {}
    block = {"OFCL": {"12": {"delta_km": -3.0, "ci95": [-5.0, -1.0]}, "24": {"delta_km": 2.0, "ci95": [-1.0, 5.0]}}}
    assert tally(block)["OFCL"] == {"leads": 2, "lower": 1, "better95": 1, "worse95": 0}
    assert Season("ep", 2025).label == "East Pacific 2025" and Season("al", 2024).archive == "natl"
