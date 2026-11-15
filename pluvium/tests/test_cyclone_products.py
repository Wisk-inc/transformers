# Copyright 2026 Nathan. Apache-2.0.
"""Cone, cloud, passage chances and scenarios on synthetic storms where the right answer is known."""

import numpy as np

from pluvium.benchmarks.cyclone_products import brier_skill, reliability
from pluvium.cyclone.approach import closest_approach, fill, member_approach, passage_chance
from pluvium.cyclone.cloud import Cloud, build_cloud, dress
from pluvium.cyclone.clusters import k_medoids, scenarios, silhouette, track_distances
from pluvium.cyclone.cone import ConeLearner, coverage
from pluvium.cyclone.geodesy import distance_km, shift
from pluvium.cyclone.learner import typical_km
from pluvium.cyclone.places import PLACES, near, nearest, register
from pluvium.cyclone.products import Forecaster, system_spread
from pluvium.cyclone.tracks import TrackSet

LEADS = np.arange(12, 121, 12)


def straight(lat0=15.0, lon0=-50.0, dlat=0.3, dlon=-1.0):
    """A track moving steadily: positions at the leads."""
    k = LEADS / 12.0
    return lat0 + dlat * k, lon0 + dlon * k


def cloud_of(tracks, weights=None):
    lat = np.array([t[0] for t in tracks])
    lon = np.array([t[1] for t in tracks])
    w = np.ones(lat.shape) if weights is None else np.asarray(weights, float)[:, None] * np.ones(lat.shape)
    return Cloud(LEADS, lat, lon, w / w.sum(0, keepdims=True), ["x"] * lat.shape[0])


def test_cone_learns_its_size_and_follows_spread():
    rng = np.random.default_rng(0)
    cone = ConeLearner()
    assert abs(cone.radius(48) - 1.1 * typical_km(48)) < 1e-9                    # the prior, before anything verifies
    spreads = rng.lognormal(np.log(60.0), 0.5, 600)
    errors = spreads * rng.rayleigh(1.0, 600)                                     # error grows with spread
    for e, s in zip(errors[:400], spreads[:400]):
        cone.add(48, e, s)
    assert 0.5 < cone.slope() <= 1.0
    static = np.full(200, cone.base(48))
    follows = np.array([cone.radius(48, s) for s in spreads[400:]])
    assert abs(coverage(static, errors[400:]) - 2 / 3) < 0.1
    assert abs(coverage(follows, errors[400:]) - 2 / 3) < 0.1
    wide, narrow = spreads[400:] > np.median(spreads), spreads[400:] <= np.median(spreads)
    gap = lambda r: abs(coverage(r[wide], errors[400:][wide]) - coverage(r[narrow], errors[400:][narrow]))   # noqa: E731
    assert gap(follows) < 0.5 * gap(static)                                       # honest in easy and hard cases alike
    flat = ConeLearner()
    for e in rng.rayleigh(80.0, 300):
        flat.add(24, e, rng.lognormal(4.0, 0.5))                                  # error unrelated to spread
    assert flat.slope() < 0.15 and abs(flat.rms(24) - 80.0 * np.sqrt(2)) < 12.0


def test_cone_waits_for_the_real_time_fix():
    cone = ConeLearner()
    carq = {124: {0: (16.0, -52.0, 50)}}
    cone.queue(100, {24: (16.0, -51.0), 48: (17.0, -53.0)}, {24: 40.0, 48: 70.0}, carq)
    assert cone.learn_until(120) == 0 and len(cone.pending) == 2
    assert cone.learn_until(124) == 1 and len(cone.rows[24]) == 1
    assert abs(cone.rows[24][0][0] - float(distance_km(16.0, -51.0, 16.0, -52.0))) < 1e-6
    assert cone.learn_until(200) == 0 and not cone.pending                        # no fix at hour 148: dropped


def test_dressing_centres_and_sizes_the_cloud_but_keeps_its_shape():
    rng = np.random.default_rng(1)
    base = straight()
    tracks = [(base[0] + rng.normal(0, 0.1) * LEADS / 60.0, base[1] + rng.normal(0, 0.6) * LEADS / 60.0)
              for _ in range(30)]                                                 # six times wider east-west
    cloud = cloud_of(tracks)
    centre = {int(lead): (20.0 + 0.2 * k, -60.0 - 0.8 * k) for k, lead in enumerate(LEADS)}
    target = {int(lead): 3.0 * cloud.spread_km()[k] for k, lead in enumerate(LEADS)}
    out = dress(cloud, centre, target)
    np.testing.assert_allclose(out.spread_km(), [target[int(k)] for k in LEADS], rtol=0.02)
    k = -1
    assert abs(np.average(out.lat[:, k], weights=out.weight[:, k]) - centre[120][0]) < 0.05
    assert np.std(out.lon[:, k]) * np.cos(np.radians(21.8)) > 3 * np.std(out.lat[:, k])   # still stretched east-west
    moved = dress(cloud, centre)                                                  # centred only
    np.testing.assert_allclose(moved.spread_km(), cloud.spread_km(), rtol=0.02)
    assert abs(cloud.member_weight().sum() - 1.0) < 1e-9


def test_a_system_counts_once_however_many_members_it_has():
    lat, lon = straight()
    many = TrackSet("S", 0, "ens", LEADS, np.tile(lat, (20, 1)), np.tile(lon, (20, 1)), np.full((20, 10), 990.0))
    one = TrackSet("S", 0, "ifs", LEADS, lat[None] + 2.0, lon[None], np.full((1, 10), 990.0))
    cloud = build_cloud({"ens": many, "ifs": one})
    w = cloud.member_weight()
    assert cloud.members == 21 and abs(w[:20].sum() - 0.5) < 1e-9 and abs(w[20] - 0.5) < 1e-9

    class Weights:
        def at(self, lead):
            return {"ens": 0.2, "ifs": 0.8}

    assert abs(build_cloud({"ens": many, "ifs": one}, Weights()).member_weight()[20] - 0.8) < 1e-9


def test_mixture_gives_a_single_run_the_ensembles_shape():
    from pluvium.cyclone.cloud import mixture_cloud, perturbations
    from pluvium.cyclone.geodesy import mean_position

    rng = np.random.default_rng(5)
    lat, lon = straight()
    ens = TrackSet("S", 0, "ens", LEADS, lat[None] + rng.normal(0, 0.5, (20, 1)) * LEADS[None] / 60.0,
                   lon[None] + rng.normal(0, 0.5, (20, 1)) * LEADS[None] / 60.0, np.full((20, 10), 990.0))
    run = TrackSet("S", 0, "ifs", LEADS, lat[None] + 3.0, lon[None], np.full((1, 10), 990.0))
    east, north = perturbations({"ens": ens, "ifs": run}, LEADS)
    assert east.shape == (20, 10) and abs(np.nanmean(east[:, -1])) < 5.0          # offsets from the system's own mean
    plain = build_cloud({"ens": ens, "ifs": run})
    mix = mixture_cloud({"ens": ens, "ifs": run})
    assert mix.members == 40 and mix.system.count("ifs") == 20
    k = -1
    own = np.array(mix.system) == "ifs"
    assert np.std(mix.lat[own, k]) > 0.3 and np.std(plain.lat[20:, k]) == 0.0     # the single run is a cloud now
    assert abs(np.mean(mix.lat[own, k]) - (lat[k] + 3.0)) < 0.05                  # centred on its own track
    w = mix.weight[:, k]
    mlat, _ = mean_position(mix.lat[:, k], mix.lon[:, k], w)
    assert abs(mlat - (lat[k] + 1.5)) < 0.15                                      # halfway: the two systems count alike

    class Weights:
        def at(self, lead):
            return {"ens": 0.25, "ifs": 0.75}

    heavy = mixture_cloud({"ens": ens, "ifs": run}, Weights())
    assert abs(heavy.member_weight()[np.array(heavy.system) == "ifs"].sum() - 0.75) < 1e-9
    alone = mixture_cloud({"ifs": run})                                           # no ensemble to borrow from
    assert alone.members == 1


def test_closest_approach_and_passage_chance():
    lat, lon = straight()
    points = {int(k): (float(a), float(b)) for k, a, b in zip(LEADS, lat, lon)}
    place = (float(lat[3]) + 0.5, float(lon[3]) - 0.5)                            # between the 48 and 60 hour positions
    km, hour = closest_approach(points, *place, start=(15.0, -50.0))
    assert km < 60.0 and 48 <= hour <= 60 and hour % 3 == 0
    assert closest_approach(points, *place, start=(15.0, -50.0), horizon=24)[0] > 200.0
    assert np.isnan(closest_approach({}, 0.0, 0.0)[0])
    hours, tlat, tlon = fill(LEADS, lat, lon, start=(15.0, -50.0))
    assert hours[0] == 0 and hours[-1] == 120 and tlat.shape == (1, 41) and abs(tlon[0, 2] - (-50.5)) < 1e-9
    rng = np.random.default_rng(2)
    near_tracks = [(lat + rng.normal(0, 0.2), lon + rng.normal(0, 0.2)) for _ in range(12)]
    far_tracks = [(lat + 6.0, lon) for _ in range(4)]
    cloud = cloud_of(near_tracks + far_tracks)
    assert abs(passage_chance(cloud, *place, radius_km=150.0, start=(15.0, -50.0)) - 0.75) < 1e-9
    assert passage_chance(cloud, 40.0, -20.0, radius_km=150.0) == 0.0
    lost = cloud_of(near_tracks[:2])
    lost.lat[1, 2:] = np.nan                                                      # one member loses the storm at 36 h
    lost.lon[1, 2:] = np.nan
    d = member_approach(lost, *place)
    assert d[0] < 150.0 < d[1]                                                    # a storm that is gone passes nothing


def test_scenarios_find_a_real_split_and_ignore_noise():
    rng = np.random.default_rng(3)
    lat, lon = straight()
    bend = np.maximum(LEADS - 48, 0) / 12.0
    west = [(lat + rng.normal(0, 0.15, 10), lon + rng.normal(0, 0.15, 10)) for _ in range(14)]
    north = [(lat + 0.9 * bend + rng.normal(0, 0.15, 10), lon + 0.8 * bend + rng.normal(0, 0.15, 10)) for _ in range(6)]
    cloud = cloud_of(west + north)
    found = scenarios(cloud)
    assert len(found) == 2 and abs(found[0].weight - 0.7) < 1e-9 and abs(found[1].weight - 0.3) < 1e-9
    assert found[0].medoid < 14 <= found[1].medoid and set(found[1].members.tolist()) == set(range(14, 20))
    assert found[1].points[120][0] > found[0].points[120][0] + 3.0                # the second one recurved
    one = scenarios(cloud_of(west))
    assert len(one) == 1 and abs(one[0].weight - 1.0) < 1e-9
    d = track_distances(cloud)
    medoids, label = k_medoids(d, 2, cloud.member_weight())
    assert silhouette(d, label, cloud.member_weight()) > 0.7 and len(set(label[:14])) == 1
    assert scenarios(Cloud(LEADS, np.zeros((0, 10)), np.zeros((0, 10)), np.zeros((0, 10)), [])) == []


def test_places_and_scores():
    assert "Kingston" in near("natl", 17.0, -75.0, 500.0) and "Boston" not in near("natl", 17.0, -75.0, 500.0)
    assert nearest("natl", 18.1, -76.7)[0] == "Kingston" and nearest("nowhere", 0.0, 0.0)[0] is None
    register("natl", "Test Cay", 12.0, -70.0)
    assert PLACES["natl"].pop("Test Cay") == (12.0, -70.0)
    assert len(PLACES["natl"]) >= 40 and len(PLACES["epac"]) >= 20
    y = np.array([1.0, 0.0, 0.0, 0.0])
    assert brier_skill(y, y) == 1.0 and abs(brier_skill(np.full(4, 0.25), y)) < 1e-9
    assert np.isnan(brier_skill(np.zeros(3), np.zeros(3)))
    rel = reliability(np.array([0.0, 0.3, 0.3, 0.9]), np.array([0.0, 0.0, 1.0, 1.0]))
    assert [r["n"] for r in rel] == [1, 2, 1] and rel[1]["observed"] == 0.5
    assert np.isnan(system_spread({"a": {12: (10.0, -50.0)}, "b": {12: (10.5, -50.0)}}, 12))


def test_forecaster_steps_causally_through_a_storm():
    from pluvium.cyclone.season import Case

    class Track:
        storm, name = "AL992099", "TEST"

    class Storm:
        id, name, track = "AL992099", "TEST", Track()

    rng = np.random.default_rng(4)
    carq = {1000 + 12 * k: {0: (15.0 + 0.3 * k, -50.0 - 1.0 * k, 60.0)} for k in range(30)}
    forecaster = Forecaster(["ens", "ifs", "gfs"])
    radii = []
    for step in range(12):
        init = 1000 + 12 * step
        sets = {}
        for name, members, noise in (("ens", 12, 0.3), ("ifs", 1, 0.2), ("gfs", 1, 1.2)):
            lat = np.array([[carq[init + int(k)][0][0] + rng.normal(0, noise) * np.sqrt(k / 12.0) for k in LEADS]
                            for _ in range(members)])
            lon = np.array([[carq[init + int(k)][0][1] + rng.normal(0, noise) * np.sqrt(k / 12.0) for k in LEADS]
                            for _ in range(members)])
            sets[name] = TrackSet("AL992099", init, name, LEADS, lat, lon, np.full(lat.shape, 990.0))
        case = Case(Storm(), init, sets, {s: t.points() for s, t in sets.items()}, carq=carq)
        g = forecaster.step(case, "natl")
        radii.append(g.cone[48])
        assert set(g.track) == set(int(k) for k in LEADS) and g.start == carq[init][0][:2]
        assert g.raw_cloud.members == 14 and g.cloud.members == 36                # three systems, twelve perturbations
        assert len(g.scenarios) >= 1 and g.early(carq[init + 12][0][:2])[0] == carq[init + 12][0][:2]
        assert set(g.places["Bridgetown"]) >= {"chance", "mixture_chance", "sized_chance", "raw_chance", "track_km"}
    assert radii[0] == 1.1 * typical_km(48) and radii[-1] != radii[0]             # the prior at first, then learned
    assert forecaster.learner.learned > 0 and len(forecaster.cone.rows[12]) >= 10
    assert forecaster.learner.at(48)["gfs"] < forecaster.learner.at(48)["ifs"]
    here = shift(carq[1000 + 12 * 11][0][0], carq[1000 + 12 * 11][0][1], 0.0, 0.0)
    assert np.isfinite(here[0])
