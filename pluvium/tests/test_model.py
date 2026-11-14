# Copyright 2026 Nathan. Apache-2.0.
import numpy as np
import pytest
import torch

from pluvium.dataset import SampleBuilder, collate
from pluvium.losses import PluviumLoss
from pluvium.synthetic import SyntheticWorld, synthetic_geography
from pluvium.train import build_model, grids_of, station_points


@pytest.fixture(scope="module")
def world():
    geo = synthetic_geography()
    return geo, SyntheticWorld(geo, days=12)


@pytest.mark.parametrize("size", ["pluvium-tiny", "pluvium-tiny-v1"])
def test_forward_backward(world, size):
    torch.manual_seed(0)
    geo, w = world
    st = w.station_cache()
    builder = SampleBuilder(geo, w, w, st, ocean_targets=20, max_land_targets=80)
    starts = builder.valid_starts(w.hours()[::24])
    targets = builder.choose_targets(True)
    batch = collate([builder.example(int(t), True, targets) for t in starts[:2]])
    model = build_model(geo, grids_of(w), size, station_points(geo, st))
    loss_fn = PluviumLoss(states=model.config.states)
    out = model(batch)
    loss, metrics = loss_fn(out, batch, torch.tensor(geo.targets.statics[targets, 0]))
    loss.backward()
    assert np.isfinite(metrics["loss"])
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    if size.endswith("v1"):
        assert "thunder" not in out and out["trans"].shape[-1] == 2
    else:
        assert out["trans"].shape[-1] == 4 and "station_thunder" in out
