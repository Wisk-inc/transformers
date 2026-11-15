# Copyright 2026 Nathan. Apache-2.0.
"""Sizing a training step to the GPU's memory: the search on made-up cards, and a real (tiny) training run on the
CPU whose memory is reported by a stand-in meter."""

import math

import numpy as np
import pytest
import torch

from pluvium.dataset import SampleBuilder
from pluvium.losses import PluviumLoss
from pluvium.synthetic import SyntheticWorld, synthetic_geography
from pluvium.train import TrainConfig, Trainer, build_model, grids_of, station_points
from pluvium.training.fit import Probe, fill, largest_batch, loader_plan, lr_scale, top_up

GB = 2.0**30


class Card:
    """A made-up GPU: memory is a base, plus so much per forecast, plus so much per forecast for every extra share
    of target cells. Checkpointing makes a forecast cheaper in memory and slower. Past ``capacity`` a step runs out
    of memory. ``bend`` makes memory grow faster than a straight line, as fragmentation does."""

    def __init__(self, capacity=90.0, base=2.0, forecast=11.0, cells=3.0, saved=0.45, bend=0.0, jitter=0.0, seed=0):
        self.capacity, self.base, self.forecast, self.cells = capacity * GB, base * GB, forecast * GB, cells * GB
        self.saved, self.bend, self.jitter = saved, bend, jitter
        self.rng = np.random.default_rng(seed)
        self.calls = []

    def peak(self, batch, scale, checkpointing):
        per = self.forecast * (self.saved if checkpointing else 1.0) + self.cells * (scale - 1.0)
        return self.base + batch * per * (1.0 + self.bend * batch) * (1.0 + self.jitter * self.rng.uniform(-1, 1))

    def measure(self, batch, scale, checkpointing):
        self.calls.append((batch, round(scale, 3), checkpointing))
        peak = self.peak(batch, scale, checkpointing)
        rate = (3.0 if checkpointing else 4.0) / (1.0 + 0.3 * (scale - 1.0))      # forecasts per second
        return Probe(batch, scale, checkpointing, peak if peak <= self.capacity else None, rate)


@pytest.mark.parametrize("card", [
    Card(),                                              # a 90 GB card and a large forecast: a handful per step
    Card(forecast=0.9, cells=0.3),                       # a small forecast: about a hundred per step
    Card(forecast=30.0, cells=6.0),                      # so large that few fit and the rest goes to more cells
    Card(bend=0.004),                                    # memory grows faster than the line through 1 and 2
    Card(jitter=0.01, seed=3),                           # measurements that wobble
    Card(capacity=24.0, forecast=9.0, cells=2.0),        # a 24 GB card
])
def test_fill_comes_close_to_the_budget_and_never_passes_it(card):
    budget = 0.95 * card.capacity
    fit = fill(card.measure, budget, max_scale=2.5)
    assert fit.peak <= budget and fit.batch >= 1 and 1.0 <= fit.scale <= 2.5
    assert fit.share >= 0.90, f"only {fit.share:.0%} of the budget is used: {fit.describe()}"
    assert len(card.calls) <= 30                         # a few measurements, not a sweep
    assert fit.to_json()["batch"] == fit.batch and "probes" not in fit.to_json()
    assert "forecasts/s" in fit.describe()


def test_fill_keeps_the_faster_way_and_checkpoints_only_when_it_must():
    roomy = Card()                                       # without checkpointing: faster, and it fits
    assert fill(roomy.measure, 0.95 * roomy.capacity).checkpointing is False
    tight = Card(capacity=24.0, forecast=40.0)           # one forecast needs 42 GB without it, 20 GB with it
    fit = fill(tight.measure, 0.95 * tight.capacity)
    assert fit.checkpointing is True and fit.batch == 1
    hopeless = Card(capacity=8.0, forecast=40.0)
    with pytest.raises(MemoryError, match="one forecast does not fit"):
        fill(hopeless.measure, 0.95 * hopeless.capacity)
    # one forecast takes half the card and a second does not fit: the faster way would leave 48% unused, so the
    # way that fits four is kept, because filling the card is the purpose
    half = Card(capacity=69.0, forecast=32.0, cells=6.0)
    fit = fill(half.measure, 0.95 * half.capacity)
    assert fit.checkpointing is True and fit.batch == 4 and fit.share > 0.9
    # ... unless the area has more cells to score, which fills it the faster way
    fit = fill(half.measure, 0.95 * half.capacity, max_scale=8.0)
    assert fit.checkpointing is False and fit.batch == 1 and fit.scale > 5.0 and fit.share > 0.9


def test_largest_batch_and_top_up_are_what_the_card_allows():
    card = Card(forecast=11.0, cells=3.0)
    budget = 0.95 * card.capacity                        # 85.5 GB: 2 + 7 x 11 = 79, and 8 would need 90
    best = largest_batch(card.measure, budget, False)
    assert best.batch == 7 and best.within(budget)
    more = top_up(card.measure, budget, best, max_scale=3.0)
    # 6.5 GB are left: at 21 GB per unit of scale (7 forecasts x 3 GB) that is 0.31 more
    assert more.batch == 7 and 1.25 <= more.scale <= 1.31 and more.within(budget) and more.peak > 0.99 * budget
    assert top_up(card.measure, budget, best, max_scale=1.0) is best          # an area with no more cells to give
    full = Probe(7, 1.0, False, 0.97 * budget, 4.0)
    assert top_up(card.measure, budget, full, max_scale=3.0) is full          # nothing worth topping up


def test_learning_rate_follows_the_batch_and_the_loader_stays_within_ram():
    assert lr_scale(1) == lr_scale(8) == 1.0             # as every earlier run was trained
    assert lr_scale(18) == pytest.approx(1.5) and lr_scale(32) == pytest.approx(2.0)
    assert lr_scale(4096) == 2.0 and lr_scale(4096, cap=3.0) == 3.0          # capped
    # 30 workers, 4 batches ready each, 0.56 GB a batch: 84 GB waiting. On a 64 GB machine that cannot be.
    workers, ready = loader_plan(30, 0.56 * GB, 64 * GB)
    assert workers * (ready + 1) * 0.56 * GB <= 0.3 * 64 * GB and workers >= 1 and ready >= 1
    assert loader_plan(30, 0.01 * GB, 64 * GB) == (30, 4)                     # small batches: nothing to change
    assert loader_plan(0, 5 * GB, 64 * GB) == (0, 4)                          # no workers: nothing to plan
    assert loader_plan(16, 50 * GB, 64 * GB) == (1, 1)                        # the least there can be


# ------------------------------------------------------------------ a real training run, a stand-in meter --

class Meter:
    """Stands in for :class:`pluvium.train.CudaMeter` on a machine with no GPU: reports the memory a made-up card
    would need for the setting just measured, and runs out of memory when the card would."""

    def __init__(self, card: Card) -> None:
        self.card, self.now, self.measured = card, (0, 1.0, False), 0

    def usable(self) -> float:
        return self.card.capacity

    def start(self, batch=0, scale=1.0, checkpointing=False) -> None:
        self.now, self.measured = (batch, scale, checkpointing), self.measured + 1
        if self.card.peak(*self.now) > self.card.capacity:
            raise torch.OutOfMemoryError("made-up card: out of memory")

    def sync(self) -> None:
        pass

    def peak(self) -> float:
        return self.card.peak(*self.now)

    def used(self) -> tuple:
        return self.card.peak(*self.now), self.card.capacity

    def release(self) -> None:
        pass


@pytest.fixture(scope="module")
def world():
    geo = synthetic_geography()
    w = SyntheticWorld(geo, days=12)
    st = w.station_cache()
    return geo, w, st


def make_trainer(world, out, meter=None, **cfg):
    geo, w, st = world
    torch.manual_seed(0)
    builder = SampleBuilder(geo, w, w, st, ocean_targets=20, max_land_targets=40)
    starts = builder.valid_starts(w.hours()[::24])
    model = build_model(geo, grids_of(w), "pluvium-tiny", station_points(geo, st))
    settings = dict(size="pluvium-tiny", hours=1.0, workers=0, probe_steps=1, max_steps=2, eval_forecasts=2,
                    warmup_steps=1, log_every=1, memory_share=0.9)
    settings.update(cfg)
    trainer = Trainer(model, PluviumLoss(states=model.config.states), builder, starts[:6], starts[6:8], out,
                      TrainConfig(**settings), device="cpu", meter=meter)
    return trainer, builder


def test_training_is_sized_to_the_card_resumes_at_that_size_and_survives_running_out(world, tmp_path, monkeypatch):
    # a 10 GB card on which 2 forecasts fit (1 + 2 x 3.4 = 7.8 of the 9 GB budget) and a third does not. Here
    # checkpointing saves nothing, so whichever way the timing of this machine prefers, the size is the same
    card = Card(capacity=10.0, base=1.0, forecast=3.4, cells=1.0, saved=1.0)
    meter = Meter(card)
    trainer, builder = make_trainer(world, tmp_path / "a", meter)
    assert builder.land_idx.size > 40, "the synthetic island must have more land cells than a forecast is scored at"
    result = trainer.run()
    fit = trainer.fit
    assert result["step"] == 2 and fit is not None and fit["policy"] == "fill"
    budget = 0.9 * card.capacity
    assert fit["peak"] <= budget and fit["peak"] >= 0.9 * budget              # the card is filled, and not overfilled
    assert trainer.cfg.batch_size == fit["batch"] == 2
    assert fit["scale"] > 1.0                                                 # the rest went to more target cells
    assert trainer.lr_scale == lr_scale(fit["batch"]) == 1.0
    assert (builder.max_land, builder.ocean_targets) == (40, 20)              # handed back as it was given
    assert trainer._yardstick_targets().size <= 40 + 20                       # scored at the usual cells
    # a second session on the same card: the size is kept, nothing is measured again
    again_meter = Meter(card)
    again, builder2 = make_trainer(world, tmp_path / "a", again_meter, max_steps=3)
    assert again.step == 2 and again.cfg.batch_size == fit["batch"] and again.scale == pytest.approx(fit["scale"])
    assert builder2.max_land > 40 and again_meter.measured == 0
    # ... and a step that runs out of memory makes the step smaller instead of stopping the run
    real, state = again.train_step, {"left": 1}

    def step_once_out_of_memory(batch):
        if state["left"]:
            state["left"] -= 1
            raise torch.OutOfMemoryError("made-up card: out of memory")
        return real(batch)

    monkeypatch.setattr(again, "train_step", step_once_out_of_memory)
    assert again.run()["step"] == 3
    assert again.scale < fit["scale"] and again.cfg.batch_size == fit["batch"]   # the extra cells went first
    # a session on a different card measures again
    other = Meter(Card(capacity=20.0, base=1.0, forecast=3.4, cells=1.0, saved=1.0))
    moved, _ = make_trainer(world, tmp_path / "a", other, max_steps=3)
    assert moved.fit is None and moved.cfg.batch_size == 0


def test_a_batch_given_by_hand_and_the_old_rule_are_left_alone(world, tmp_path):
    meter = Meter(Card(capacity=10.0, base=1.0, forecast=3.4))
    trainer, builder = make_trainer(world, tmp_path / "b", meter, batch_size=2)
    assert trainer.pick_batch_size() == 2 and meter.measured == 0 and trainer.fit is None
    assert (builder.max_land, builder.ocean_targets) == (40, 20) and trainer.lr_scale == 1.0
    no_gpu, _ = make_trainer(world, tmp_path / "c")                           # no GPU and no meter: as before
    assert no_gpu.meter is None and no_gpu.pick_batch_size() == 2
    # a card on which nothing fits: sizing gives up, says so, and the run still gets a batch
    hopeless = Meter(Card(capacity=1.5, base=1.0, forecast=3.4))
    stuck, _ = make_trainer(world, tmp_path / "d", hopeless)
    assert stuck.pick_batch_size() == 2 and stuck.fit is None and math.isclose(stuck.scale, 1.0)
