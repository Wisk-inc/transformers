# Copyright 2026 Nathan. Apache-2.0.
"""Several models at once: the equal-weight mean, the flow-dependent cone, the model readers' grid handling and the
swarm's steering-source latent -- on constructed inputs where the answer is known."""

from __future__ import annotations

import numpy as np

from naturev1.cyclone import geometry as geo
from naturev1.cyclone.data import LEADS
from naturev1.cyclone.members import build_context
from naturev1.cyclone.multimodel import cone_predictors, flow_dependent_cone, model_mean, model_spread_km
from naturev1.swarm.analysis import LAT_AXIS, LON_AXIS, _coarsen
from naturev1.swarm.core import Swarm, SwarmFields
from naturev1.swarm.models import coarsen_to_nodes

from test_cyclone import _case, _vortex_env
from test_forecastflow import _flow


def test_model_mean_averages_in_the_storm_frame_and_drops_models_that_lost_the_storm():
    case = _case()
    hours = np.asarray(LEADS, float)
    west = np.stack(geo.move(case.lat, case.lon, -20.0 * hours, 0.0 * hours), -1)
    north = np.stack(geo.move(case.lat, case.lon, 0.0 * hours, 20.0 * hours), -1)
    north[-2:] = np.nan                                       # this model lost the storm after 72 h
    mean = model_mean(case, {"a": west, "b": north}, LEADS)
    k = LEADS.index(24)
    e, n = geo.local_displacement_km(case.lat, case.lon, *mean[k])
    assert abs(e + 10.0 * 24) < 1.0 and abs(n - 10.0 * 24) < 1.0, "halfway between the two"
    assert np.allclose(mean[-2:], west[-2:]), "only the model that still has the storm"
    spread = model_spread_km(case, {"a": west, "b": north})
    assert abs(spread[k] - 20.0 * 24 * np.sqrt(2) / 2) < 2.0 and np.isnan(spread[-1])
    assert model_mean(case, None, LEADS) is None and np.allclose(model_mean(case, west, LEADS), west)


def test_flow_dependent_cone_widens_with_spread_and_interpolates_leads():
    case = _case()
    track = np.stack(geo.move(case.lat, case.lon, -20.0 * np.asarray(LEADS, float), 0.0 * np.asarray(LEADS, float)), -1)
    members = {"xtrp6": track, "cliper": np.stack(geo.move(case.lat, case.lon, -20.0 * np.asarray(LEADS, float), 5.0 * np.asarray(LEADS, float)), -1)}
    x = cone_predictors(case, track, members, ["xtrp6", "cliper"], track, track, LEADS)
    assert x.shape == (len(LEADS), 7) and np.all(x[:, 0] == 1.0) and x[LEADS.index(24), 1] > x[LEADS.index(6), 1]
    table = {"leads": [6, 120], "coef": [[10.0, 0.5, 0, 0, 0, 0, 0], [100.0, 0.5, 0, 0, 0, 0, 0]], "scale": [1.0, 1.0],
             "medians": [1, 50, 50, 50, 15, 20, 60]}
    radius = flow_dependent_cone(x, table, LEADS)
    assert radius[0] < radius[-1] and abs(radius[-1] - (100.0 + 0.5 * x[-1, 1])) < 1e-6
    wide = x.copy()
    wide[:, 1] *= 3
    assert np.all(flow_dependent_cone(wide, table, LEADS) > radius)


def test_coarsening_any_grid_matches_the_quarter_degree_coarsening_and_handles_half_degree():
    rng = np.random.default_rng(0)
    lat_q = np.linspace(90, -90, 721)
    lon_q = np.arange(0, 360, 0.25)
    field = np.sin(np.radians(lat_q))[:, None] * 10.0 + np.cos(np.radians(lon_q))[None, :] * 3.0 + rng.normal(size=(721, 1440))
    a = _coarsen(field)
    b = coarsen_to_nodes(field, 90.0, 0.25, 0.0, 0.25)
    assert a.shape == b.shape == (LAT_AXIS.size, LON_AXIS.size)
    assert np.nanmax(np.abs(a - b)) < 1e-9, "identical on the quarter-degree grid"
    lat_h = np.linspace(90, -90, 361)
    lon_h = (np.arange(0, 360, 0.5) + 180.0) % 360.0 - 180.0                           # -180 .. 179.5, like ECMWF
    smooth = np.sin(np.radians(lat_h))[:, None] * 10.0 + np.cos(np.radians(np.sort(lon_h)))[None, :] * 3.0
    c = coarsen_to_nodes(smooth, 90.0, 0.5, -180.0, 0.5)
    truth = np.sin(np.radians(LAT_AXIS))[:, None] * 10.0 + np.cos(np.radians(LON_AXIS))[None, :] * 3.0
    assert np.isfinite(c).all() and np.abs(c - truth).max() < 0.2


def test_each_world_follows_one_model_and_the_swarm_spreads_between_them():
    case = _case(motion_kmh=(-18.0, 0.0))
    env = _vortex_env(u=-5.0, v=0.0)
    ctx = build_context(case, env)
    flows = {"west": _flow(env), "turning": _flow(env, turn_hour=48)}
    fields = SwarmFields.from_context(ctx, forecast=flows)
    assert fields.n_sources == 2 and fields.source_names == ["west", "turning"]
    sw = Swarm.from_context(case, ctx, 3000, seed=0, fields=fields)
    src = sw.params["source"]
    assert src.min() == 0 and src.max() == 1 and 0.4 < float((src == 1).float().mean()) < 0.6
    sw.assimilate_history(case)
    fc = sw.forecast(LEADS, case.lat, case.lon)
    k120 = LEADS.index(120)
    north = geo.local_displacement_km(case.lat, case.lon, fc.positions[:, k120, 0], fc.positions[:, k120, 1])[1]
    assert np.quantile(north, 0.9) - np.quantile(north, 0.1) > 500.0, "the two models' futures both survive"
    one = Swarm.from_context(case, ctx, 3000, seed=0, fields=SwarmFields.from_context(ctx, forecast=flows["west"]))
    one.assimilate_history(case)
    assert np.nanmean(north) > np.nanmean(geo.local_displacement_km(case.lat, case.lon, one.forecast(LEADS, case.lat, case.lon).positions[:, k120, 0],
                                                                    one.forecast(LEADS, case.lat, case.lon).positions[:, k120, 1])[1]) + 200.0


def test_model_indexes_parse_both_bucket_formats_and_a_run_is_skipped_cleanly(monkeypatch, tmp_path):
    from naturev1.swarm import models

    ncep = "1:0:d=2025102700:PRMSL:mean sea level:24 hour fcst:\n2:1000:d=2025102700:UGRD:850 mb:24 hour fcst:\n3:2500:d=2025102700:VGRD:850 mb:24 hour fcst:\n"
    idx = models._index_ncep(ncep)
    assert idx[("PRMSL", "mean sea level")] == (0, 999) and idx[("VGRD", "850 mb")] == (2500, None)
    ecmwf = ('{"param": "msl", "levtype": "sfc", "_offset": 10, "_length": 5}\n'
             '{"param": "u", "levtype": "pl", "levelist": "850", "_offset": 15, "_length": 7}\n')
    idx = models._index_ecmwf(ecmwf)
    assert idx[("msl", "sfc")] == (10, 14) and idx[("u", "850")] == (15, 21)
    # a bucket that is down: the model is reported and skipped, the others go on
    monkeypatch.setattr(models, "_get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("503")))
    case = _case()
    notes = []
    out = models.fetch_models(case, "2025-10-27T00", ("gfs", "ifs"), cache=tmp_path, log=notes.append)
    assert out == {} and len(notes) == 2 and all("not used" in n for n in notes)
    assert not models.LiveModel(models.MODELS["gefs"], tmp_path).available("2025-10-27T00")
