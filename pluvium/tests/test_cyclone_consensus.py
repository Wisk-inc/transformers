# Copyright 2026 Nathan. Apache-2.0.
"""Bias correction, weights, consensus, the interpolator and the in-storm memory, on tracks with known errors."""

import numpy as np

from pluvium.cyclone.bias import TrackBias, headings
from pluvium.cyclone.consensus import consensus
from pluvium.cyclone.geodesy import distance_km, from_along_cross, shift
from pluvium.cyclone.interp import interpolate
from pluvium.cyclone.online import StormMemory
from pluvium.cyclone.weights import ConsensusWeights, error_vectors

LEADS = (12, 24, 36, 48, 60, 72, 96, 120)


def truth(seed):
    rng = np.random.default_rng(seed)
    lat0, lon0 = 14.0 + rng.random() * 6, -55.0 - rng.random() * 20
    out, lat, lon = {}, lat0, lon0
    last = 0
    for lead in LEADS:
        dt = lead - last
        lat, lon = shift(lat, lon, -17.0 * dt + rng.normal(0, 2) * dt, 8.0 * dt + rng.normal(0, 2) * dt)
        out[lead], last = (float(lat), float(lon)), lead
    return out


def forecast(obs, slow_km_per_lead, noise_km, rng, common=None):
    """The truth with an along-track lag growing with lead, plus noise (``common`` is noise shared by systems)."""
    head = headings(obs)
    out = {}
    for lead, (lat, lon) in obs.items():
        e, n = rng.normal(0, noise_km, 2) * np.sqrt(lead / 24.0)
        if common is not None:
            e, n = e + common[lead][0], n + common[lead][1]
        a, b = from_along_cross(lat, lon, -slow_km_per_lead * lead / 12.0, 0.0, head[lead])
        out[lead] = tuple(float(v) for v in shift(a, b, e, n))
    return out


def test_bias_is_learned_and_removed():
    rng = np.random.default_rng(1)
    bias = TrackBias(shrink=5.0)
    for seed in range(60):
        obs = truth(seed)
        bias.learn("slow", forecast(obs, 12.0, 15.0, rng), obs)
    along, cross = bias.bias("slow", 72)
    assert -80.0 < along < -55.0 and abs(cross) < 15.0                 # 12 km per 12 h lag = 72 km at 72 h, shrunk
    obs = truth(999)
    raw = forecast(obs, 12.0, 0.0, rng)
    fixed = bias.correct("slow", raw)
    assert distance_km(*fixed[72], *obs[72]) < 0.35 * distance_km(*raw[72], *obs[72])
    assert bias.bias("never seen", 72) == (0.0, 0.0) and bias.correct("never seen", raw) == raw


def test_weights_favour_the_accurate_and_discount_the_redundant():
    rng = np.random.default_rng(2)
    w = ConsensusWeights(["good", "twin_a", "twin_b", "poor"], shrink=0.1, to_equal=0.0)
    for seed in range(120):
        obs = truth(seed)
        shared = {lead: rng.normal(0, 40.0, 2) * np.sqrt(lead / 24.0) for lead in LEADS}
        tracks = {"good": forecast(obs, 0.0, 25.0, rng), "twin_a": forecast(obs, 0.0, 8.0, rng, shared),
                  "twin_b": forecast(obs, 0.0, 8.0, rng, shared), "poor": forecast(obs, 0.0, 90.0, rng)}
        for name, pts in tracks.items():
            w.add(seed, name, error_vectors(pts, obs))
    w.fit()
    short = w.at(24)
    assert abs(sum(short.values()) - 1.0) < 1e-9 and min(short.values()) >= 0.0
    assert short["good"] > short["twin_a"] + 0.1 and short["good"] > 0.4          # twins share their vote
    assert short["poor"] < 0.1
    assert w.at(96) == w.weights["long"] and w.rms["poor"]["short"] > w.rms["good"]["short"]


def test_consensus_beats_its_members_and_survives_a_missing_system():
    rng = np.random.default_rng(3)
    errs = {"consensus": [], "a": [], "b": [], "c": []}
    for seed in range(80):
        obs = truth(seed)
        tracks = {s: forecast(obs, 0.0, 40.0, rng) for s in "abc"}
        if seed % 5 == 0:
            tracks["c"] = {k: v for k, v in tracks["c"].items() if k <= 48}        # c loses the storm
        cons = consensus(tracks)
        assert set(cons) == set(LEADS)
        errs["consensus"].append(distance_km(*cons[72], *obs[72]))
        for s in "ab":
            errs[s].append(distance_km(*tracks[s][72], *obs[72]))
    assert np.mean(errs["consensus"]) < 0.8 * np.mean(errs["a"])
    weighted = consensus({"a": {12: (10.0, -50.0)}, "b": {12: (12.0, -50.0)}}, scale={"a": 3.0, "b": 1.0})
    assert abs(weighted[12][0] - 10.5) < 0.05


def test_the_interpolator_slides_an_old_run_to_the_observed_position():
    obs = truth(7)
    old = {lead: tuple(float(v) for v in shift(lat, lon, 60.0, -40.0)) for lead, (lat, lon) in obs.items()}
    early = interpolate(old, obs[12], 12)
    assert early[0] == obs[12] and set(early) == {0, 12, 24, 36, 48, 60, 84, 108}
    assert distance_km(*early[12], *obs[24]) < 3.0                     # a constant offset is removed entirely
    faded = interpolate(old, obs[12], 12, taper_hours=48.0)
    assert distance_km(*faded[60], *old[72]) < 1.0                     # past the taper the old run stands
    assert interpolate({24: (10.0, -50.0)}, (10.0, -50.0), 12) == {}


def test_storm_memory_downweights_the_system_that_has_this_storm_wrong():
    obs = truth(11)
    hours = {lead: 1000 + lead for lead in LEADS}
    carq = {1000 + lead: {0: (obs[lead][0], obs[lead][1], 60.0)} for lead in LEADS}
    right = dict(obs)
    wrong = {lead: tuple(float(v) for v in shift(lat, lon, 150.0, 0.0)) for lead, (lat, lon) in obs.items()}
    mem = StormMemory(gain=1.0)
    assert mem.scale(1000, ["right", "wrong"]) == {"right": 1.0, "wrong": 1.0}
    mem.remember(1000, {"right": right, "wrong": wrong}, carq)
    assert mem.scale(1006, ["right", "wrong"]) == {"right": 1.0, "wrong": 1.0}     # nothing has verified yet
    scale = mem.scale(hours[24], ["right", "wrong", "new"])
    assert scale["right"] > 1.3 and scale["wrong"] < 0.6 and scale["new"] == 1.0
