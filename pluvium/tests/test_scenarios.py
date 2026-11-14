# Copyright 2026 Nathan. Apache-2.0.
"""Scenarios: coupling keeps both the calibrated marginals and the ensemble's structure; the scores can tell."""

import numpy as np

from pluvium.blend import THRESHOLDS, exceed, quantile_at, quantiles
from pluvium.scenarios import ecc, independent, medoid, probability_matched_mean, ranks, schaake
from pluvium.scenarios.areal import areal_chance, areal_mean, areal_summary, overconfidence
from pluvium.scenarios.fss import FSSAccumulator, fss
from pluvium.scenarios.schaake import pick_days
from pluvium.scenarios.scores import (
    areal_crps,
    box_groups,
    crps_ensemble,
    energy_score,
    member_sharpness,
    nearby_pairs,
    variogram_score,
)
from pluvium.scenarios.spectra import SpectrumAccumulator, radial_spectrum
from pluvium.util import hour_of


def storm_fields(members, n, rng, width=12.0, vary=0.0):
    """Members with one storm each at a different place along a line of ``n`` places (coherent in space). With
    ``vary`` the storms differ in strength, so the total over the line varies from member to member."""
    x = np.arange(n)
    centres = rng.uniform(0.25 * n, 0.75 * n, members)
    amp = 60.0 * rng.lognormal(0.0, vary, members) if vary else np.full(members, 60.0)
    return amp[:, None] * np.exp(-0.5 * ((x[None] - centres[:, None]) / width) ** 2)


def chances_from(samples):
    """Calibrated-looking chances (N, K) from a large sample (S, N)."""
    return np.stack([(samples >= t).mean(0) for t in THRESHOLDS], 1).astype(np.float32)


def test_quantile_at_generalises_quantiles():
    rng = np.random.default_rng(0)
    p = np.sort(rng.random((50, len(THRESHOLDS))), 1)[:, ::-1].astype(np.float32)
    levels = np.array([0.05, 0.5, 0.95])
    np.testing.assert_allclose(quantile_at(p, np.tile(levels, (50, 1))), quantiles(p, levels), rtol=1e-6)
    u = rng.random((50, 7))
    q = quantile_at(p, u)
    order = np.argsort(u, 1)
    assert np.all(np.diff(np.take_along_axis(q, order, 1), axis=1) >= -1e-6)      # monotone in the level, row by row


def test_ranks_break_ties_by_the_neighbourhood():
    members = np.array([[0.0, 5.0], [0.0, 1.0], [0.0, 9.0]])
    tiebreak = np.array([[0.2, 0.0], [0.9, 0.0], [0.1, 0.0]])
    r = ranks(members, tiebreak)
    np.testing.assert_array_equal(r[:, 0], [1, 2, 0])                   # all dry: the one with rain nearby ranks top
    np.testing.assert_array_equal(r[:, 1], [1, 0, 2])
    assert sorted(ranks(np.zeros((5, 3)))[:, 0].tolist()) == [0, 1, 2, 3, 4]


def test_coupling_keeps_marginals_and_structure():
    rng = np.random.default_rng(1)
    n, m = 120, 40
    climate = storm_fields(4000, n, rng)                                 # what the calibrated chances stand for
    p = chances_from(climate)
    template = storm_fields(m, n, rng)
    coupled = ecc(p, template, rng=rng)
    shuffled = independent(p, m, rng)
    assert coupled.shape == (m, n)
    for t in (1.0, 10.0, 25.0):                                         # the same marginals, either way
        assert np.abs((coupled >= t).mean(0) - exceed(p, t)).max() < 1.5 / m + 0.02
        assert np.abs((shuffled >= t).mean(0) - exceed(p, t)).max() < 1.5 / m + 0.02
    rough = lambda s: np.abs(np.diff(s, axis=1)).mean()                 # noqa: E731
    assert rough(coupled) < 0.35 * rough(shuffled)                      # coupled scenarios are smooth storms
    k = int(np.argmax(template[0]))
    assert coupled[0, k] > np.median(coupled[:, k])                     # member 0 is wet where its template was
    obs = storm_fields(1, n, rng)[0]
    pairs = nearby_pairs(np.zeros(n), np.arange(n) * 0.1, max_km=40.0)
    assert variogram_score(coupled, obs, pairs) < variogram_score(shuffled, obs, pairs)
    assert np.isfinite(energy_score(coupled, obs)) and energy_score(coupled, obs) > 0


def test_areal_totals_need_coupling():
    rng = np.random.default_rng(2)
    n, m = 100, 60
    p = chances_from(storm_fields(4000, n, rng, width=25.0, vary=0.8))
    template = storm_fields(m, n, rng, width=25.0, vary=0.8)
    coupled, shuffled = ecc(p, template, rng=rng), independent(p, m, rng)
    lat = np.zeros(n)
    assert abs(areal_mean(coupled, lat).mean() - areal_mean(shuffled, lat).mean()) < 1.5      # same mean
    assert overconfidence(coupled, shuffled, lat) > 2.0                 # independent places are far too certain
    hit, miss = [], []
    groups = np.zeros(n, np.int64)
    for _ in range(40):
        obs = storm_fields(1, n, rng, width=25.0, vary=0.8)[0]
        hit.append(areal_crps(coupled, obs, groups)[0])
        miss.append(areal_crps(shuffled, obs, groups)[0])
    assert np.mean(hit) < 0.9 * np.mean(miss)
    chance = areal_chance(coupled, lat, (5.0, 30.0))
    assert 0.0 <= chance[30.0] <= chance[5.0] <= 1.0
    summary = areal_summary(coupled, lat, mask=np.arange(n) < 50)
    assert summary["p10_mm"] <= summary["median_mm"] <= summary["p90_mm"]


def test_one_map_from_many():
    rng = np.random.default_rng(3)
    s = storm_fields(30, 80, rng)
    pmm = probability_matched_mean(s)
    mean = s.mean(0)
    assert pmm.max() > 1.3 * mean.max()                                 # the peak is restored
    assert np.argmax(pmm) == np.argmax(mean)                            # where the mean has it
    assert abs(pmm.mean() - mean.mean()) < 0.05 * mean.mean() + 1e-6
    k = medoid(s)
    centres = s.argmax(1)
    assert abs(centres[k] - np.median(centres)) <= 6                    # the most central storm


def test_schaake_uses_past_days_from_the_same_season():
    days = hour_of("2020-01-01T00") + 24 * np.arange(4 * 365)
    target = hour_of("2024-08-15T00")
    idx = pick_days(days, target, 20)
    assert idx.size == 20 and np.all(days[idx] < target)
    months = (days[idx].astype("datetime64[h]").astype("datetime64[M]").astype(int) % 12) + 1
    assert set(months.tolist()) <= {7, 8, 9}
    rng = np.random.default_rng(4)
    history = storm_fields(20, 60, rng)
    p = chances_from(storm_fields(3000, 60, rng))
    out = schaake(p, history, rng=rng)
    assert out.shape == (20, 60) and np.abs((out >= 10).mean(0) - exceed(p, 10.0)).max() < 0.1


def test_scores_have_known_values():
    ens = np.array([[0.0], [2.0], [4.0]])
    assert abs(crps_ensemble(ens, np.array([2.0]))[0] - (4 / 3 - 8 / 6)) < 1e-9          # fair CRPS: mean |x-y| - spread
    assert crps_ensemble(np.full((5, 1), 3.0), np.array([3.0]))[0] == 0.0
    v, o = member_sharpness(np.array([[0.0] * 10 + [10.0] * 10] * 3), np.array([0.0] * 10 + [10.0] * 10))
    assert abs(v - o) < 1e-9
    g = box_groups(np.array([0.0, 0.5, 1.0, 1.5, 5.0]), np.array([0.0, 0.5, 1.0, 1.5, 5.0]), 2.0, min_places=2)
    assert g[0] == g[1] == g[2] == g[3] >= 0 and g[4] == -1


def test_fss_rises_with_scale_and_spectra_show_blur():
    rng = np.random.default_rng(5)
    obs = np.zeros((60, 60))
    obs[20:28, 20:28] = 30.0
    shifted = np.roll(obs, 5, axis=1)                                   # the storm, five cells to the east
    assert fss(shifted, obs, 10.0, 1) < 0.5 < fss(shifted, obs, 10.0, 15)
    assert fss(obs, obs, 10.0, 1) == 1.0
    acc = FSSAccumulator(thresholds=(10.0,), sizes=(1, 5, 9, 15))
    acc.add(shifted, obs)
    s = acc.scores()[0]
    assert np.all(np.diff(s) > 0) and 25.0 < acc.useful_scale(25.0)[10.0] <= 15 * 25.0
    from scipy.ndimage import gaussian_filter

    sharp = rng.gamma(0.5, 5.0, (64, 64))
    wl, ps = radial_spectrum(sharp, 25.0)
    assert wl[0] > wl[-1] and ps.shape == wl.shape
    spec = SpectrumAccumulator(25.0)
    for _ in range(5):
        field = rng.gamma(0.5, 5.0, (64, 64))
        spec.add(gaussian_filter(field, 2.0), field)
    ratio = spec.ratio()
    assert ratio[-1] < 0.05 < ratio[0] and spec.effective_resolution() > 150.0     # small scales are gone
