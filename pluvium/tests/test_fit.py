# Copyright 2026 Nathan. Apache-2.0.
"""Sizing a training step to the GPU's memory: the search on made-up cards, and a real (tiny) training run on the
CPU whose memory is reported by a stand-in meter."""

import json
import math
import sys
import time

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
    # one failure: memory is freed and the same step is tried again, nothing is given up
    assert again.scale == pytest.approx(fit["scale"]) and again.cfg.batch_size == fit["batch"]
    # a second failure in a row gives up the extra cells and keeps both forecasts
    state["left"] = 2
    twice, _ = make_trainer(world, tmp_path / "a", Meter(card), max_steps=4)
    real = twice.train_step
    monkeypatch.setattr(twice, "train_step", step_once_out_of_memory)
    assert twice.run()["step"] == 4
    assert twice.scale == 1.0 and twice.cfg.batch_size == fit["batch"]
    # a session on a different card measures again
    other = Meter(Card(capacity=20.0, base=1.0, forecast=3.4, cells=1.0, saved=1.0))
    moved, _ = make_trainer(world, tmp_path / "a", other, max_steps=3)
    assert moved.fit is None and moved.cfg.batch_size == 0


def test_steps_are_sized_on_the_rainiest_issue_time_and_every_kind_of_out_of_memory_shrinks_them(world, tmp_path,
                                                                                                  monkeypatch):
    # The loss needs more memory when it rained, so a step sized on a dry day ran out on a stormy one. Here the
    # fourth issue time is the only one whose three days are all heavy rain.
    from pluvium.train import out_of_memory

    card = Card(capacity=10.0, base=1.0, forecast=3.4, cells=1.0, saved=1.0)
    trainer, builder = make_trainer(world, tmp_path / "e", Meter(card))
    wettest = int(trainer.train_starts[3])
    real_targets = builder.cmorph.targets

    def rain(hours, columns=None):
        x = np.array(real_targets(hours, columns), dtype=np.float32)
        x[(np.asarray(hours) >= wettest) & (np.asarray(hours) < wettest + 72)] = 9.0
        return x

    monkeypatch.setattr(builder.cmorph, "targets", rain)
    seen, real_example = [], builder.example
    monkeypatch.setattr(builder, "example", lambda start, *a, **k: seen.append(int(start)) or real_example(start, *a, **k))
    trainer.pick_batch_size()
    assert trainer._probe_start() == wettest and set(seen) == {wettest}
    assert trainer.fit["probe"] == "rainiest" and trainer.fit["probe_start"] == wettest
    # the CUDA libraries report running out of memory in their own words, and those count; nothing else does
    assert out_of_memory(torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert out_of_memory(RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`"))
    assert out_of_memory(RuntimeError("CUDA error: out of memory"))
    assert not out_of_memory(RuntimeError("CUDA error: an illegal memory access was encountered"))
    assert not out_of_memory(ValueError("out of memory"))
    # a step that runs out in the encoder: giving up target cells alone would not help, so forecasts go too
    trainer._adopt(10, 1.5, False, trainer.fit)
    assert trainer._shrink(1) and (trainer.cfg.batch_size, trainer.scale) == (10, 1.5)      # freed, tried once more
    assert trainer._shrink(2) and (trainer.cfg.batch_size, trainer.scale) == (10, 1.0)      # the extra cells go first
    assert trainer._shrink(3) and trainer.cfg.batch_size == 8 and 1.0 < trainer.scale <= 1.25   # then forecasts,
    trainer._adopt(2, 1.0, False, trainer.fit)                                # and the room they leave goes to cells
    assert trainer._shrink(2) and trainer.cfg.batch_size == 1 and trainer.scale > 1.0
    trainer._adopt(1, 1.0, False, trainer.fit)
    assert trainer._shrink(1) and not trainer._shrink(2)                      # nothing left to give up


def test_a_run_that_stops_is_started_again_with_smaller_steps_then_the_old_rule(tmp_path):
    from pluvium.program import supervise

    log = tmp_path / "tries.txt"

    def child(succeed_at: int) -> list:
        return [sys.executable, "-c",
                "import os, sys; open(sys.argv[1], 'a').write(os.environ['PLUVIUM_VRAM'] + ' ' + "
                "os.environ['PLUVIUM_VRAM_SHARE'] + ' ' + os.environ.get('PLUVIUM_BATCH', '-') + ' ' + "
                f"' '.join(sys.argv[2:]) + '\\n'); sys.exit(0 if os.environ['PLUVIUM_ATTEMPT'] == '{succeed_at}' else 1)",
                str(log)]

    assert supervise(["--hours", "1"], 0.95, command=child(3)) == 0
    assert log.read_text().splitlines() == ["fill 0.95 - --hours 1 --vram fill --vram-share 0.95",
                                            "fill 0.85 - --hours 1 --vram fill --vram-share 0.85",
                                            "fastest 0.95 - --hours 1 --vram fastest --vram-share 0.95"]
    log.unlink()
    # a run started with the old rule is tried with it, then with one forecast per step
    assert supervise(["--hours", "1"], 0.95, command=child(9), vram="fastest") == 1
    assert log.read_text().splitlines() == ["fastest 0.95 - --hours 1 --vram fastest --vram-share 0.95",
                                            "fastest 0.95 1 --hours 1 --vram fastest --vram-share 0.95 --batch 1"]
    once = [sys.executable, "-c", "import sys; sys.exit(0)"]
    assert supervise([], 0.95, command=once) == 0                              # a run that works is run once
    always = [sys.executable, "-c", "import sys; sys.exit(3)"]
    assert supervise([], 0.95, command=always) == 3                            # four tries, then its error


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


def test_a_killed_or_silent_run_is_started_again_and_its_last_lines_are_kept(tmp_path):
    """The first GPU runs stopped with an error whose last line nobody saw. Now the end of a stopped run's output is
    kept on disk and on the Hub, and a run that is killed or goes silent is started again like one that fails."""
    from pluvium.program import supervise

    class Api:
        def __init__(self):
            self.files = {}

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
            self.files[path_in_repo] = open(path_or_fileobj).read()

    def child(body: str) -> list:
        return [sys.executable, "-u", "-c", "import os, signal, sys, time\nfirst = os.environ['PLUVIUM_ATTEMPT'] == '1'\n" + body]

    # killed outright on the first try (as when the machine runs out of RAM), fine on the second
    api = Api()
    killed = child("print('step 1', flush=True)\nif first:\n    os.kill(os.getpid(), signal.SIGKILL)\nprint('done')")
    assert supervise([], 0.95, command=killed, root=tmp_path / "a", repo="org/repo", api=api) == 0
    assert "was killed (signal 9)" in api.files["logs/attempt-1.txt"] and "step 1" in api.files["logs/attempt-1.txt"]
    assert (tmp_path / "a" / "logs" / "attempt-1.txt").exists() and len(api.files) == 1
    # silent for longer than allowed on the first try: stopped, and started again
    silent = child("print('started', flush=True)\nif first:\n    time.sleep(120)\nprint('done')")
    began = time.time()
    assert supervise([], 0.95, command=silent, root=tmp_path / "b", stall_minutes=0.03) == 0
    assert time.time() - began < 60
    assert "printed nothing" in (tmp_path / "b" / "logs" / "attempt-1.txt").read_text()
    # a fault that comes back every time: every try is kept, the error can be read, and a token never gets out
    api = Api()
    token = "hf_" + "a1B2" * 9
    broken = child(f"print('RuntimeError: CUDA error: device-side assert {token}', file=sys.stderr)\nsys.exit(3)")
    assert supervise([], 0.95, command=broken, root=tmp_path / "c", repo="org/repo", api=api) == 3
    assert sorted(api.files) == [f"logs/attempt-{k}.txt" for k in (1, 2, 3, 4)]
    last = api.files["logs/attempt-4.txt"]
    assert "attempt 4 of 4 (fastest, share 0.95, batch 1)" in last and "RuntimeError: CUDA error: device-side assert" in last
    assert token not in last and "[token]" in last


def test_a_size_that_stops_a_run_is_taken_as_too_large_and_not_tried_again(world, tmp_path):
    # a card on which 3 forecasts would fit by memory, but a step of 3 fails in a CUDA library (not out of memory)
    card = Card(capacity=20.0, base=1.0, forecast=3.4, cells=1.0, saved=1.0)

    class Fussy(Meter):
        def start(self, batch=0, scale=1.0, checkpointing=False):
            super().start(batch, scale, checkpointing)
            if batch >= 3:
                raise RuntimeError("CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemmStridedBatched`")

    trainer, _ = make_trainer(world, tmp_path / "s", Fussy(card))
    assert trainer.pick_batch_size() == 2 and trainer.fit["policy"] == "fill"       # sized below what failed
    note = json.loads((tmp_path / "s" / "sizes.json").read_text())
    assert any(size[0] >= 3 for size in note["crashed"]) and note.get("trying") is None
    # the run started after it is told: it does not try that size again
    again_meter = Fussy(card)
    again, _ = make_trainer(world, tmp_path / "s", again_meter)
    tried = []
    real = again_meter.start
    again_meter.start = lambda batch=0, scale=1.0, checkpointing=False: tried.append(batch) or real(batch, scale, checkpointing)
    assert again.pick_batch_size() == 2 and all(b < 3 for b in tried)
    # a process that died in the middle of a measurement left the size behind: it is skipped too
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "sizes.json").write_text(json.dumps({"trying": [2, 1.0, False]}))
    dead, _ = make_trainer(world, tmp_path / "t", Meter(card))
    probe = dead._measure(2, 1.0, False)
    assert probe.peak is None and [2, 1.0, False] in json.loads((tmp_path / "t" / "sizes.json").read_text())["crashed"]
    # the smallest step failing for another reason than memory is a real fault, and is raised
    hopeless = Fussy(card)
    hopeless.start = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("CUDA error: device-side assert triggered"))
    broken, _ = make_trainer(world, tmp_path / "u", hopeless)
    with pytest.raises(RuntimeError, match="device-side assert"):
        broken._measure(1, 1.0, False)


def test_the_old_batch_rule_never_tries_a_batch_that_cannot_fit(world, tmp_path, monkeypatch):
    """On the 96 GB card one forecast took 38.6 GB. The old rule went on to try three, moved that batch to the GPU
    outside the lines that catch running out of memory, and the run ended there."""
    import pluvium.train as T

    gb = 2**30
    trainer, _ = make_trainer(world, tmp_path / "f", memory="fastest")
    moved = []
    real_to_device = T.to_device

    def to_device(batch, device):
        b = int(batch["truth"].shape[0])
        moved.append(b)
        if b >= 3:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return real_to_device(batch, device)

    monkeypatch.setattr(T, "to_device", to_device)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: type("P", (), {"total_memory": 96 * gb})())
    for name in ("reset_peak_memory_stats", "synchronize", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 38.6 * gb * moved[-1])
    assert trainer._fastest_batch() in (1, 2) and moved == [1, 2]                 # three was not even moved there
    # a card where the estimate says three fits and moving it there runs out: handled, the run goes on
    moved.clear()
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 10 * gb * moved[-1])
    assert trainer._fastest_batch() in (1, 2) and moved == [1, 2, 3]


def test_one_batch_that_cannot_be_trained_on_is_left_out_and_five_in_a_row_stop_the_run(world, tmp_path, monkeypatch):
    trainer, _ = make_trainer(world, tmp_path / "g", batch_size=2, max_steps=4)
    real, calls = trainer.train_step, {"n": 0}

    def sometimes(batch):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("a batch with something wrong in it")
        return real(batch)

    monkeypatch.setattr(trainer, "train_step", sometimes)
    assert trainer.run()["step"] == 4 and calls["n"] == 4                         # the run went on past it
    always, _ = make_trainer(world, tmp_path / "h", batch_size=2, max_steps=20)
    monkeypatch.setattr(always, "train_step", lambda batch: (_ for _ in ()).throw(ValueError("every batch fails")))
    with pytest.raises(ValueError, match="every batch fails"):
        always.run()
    assert always.step == 4                                                       # four left out, the fifth stopped it
