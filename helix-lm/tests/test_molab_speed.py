# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The molab program keeps the GPU fed: batches are built by many CPU threads with their own random streams,
the batch size is the fastest one for the card (not simply the largest), stale batches never fool the
measurement, and an out-of-memory goes back to the last size that fit."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from helix_lm.scripts import tristream_molab as molab


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(molab.time, "time", c)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (40e9, 100e9))
    return c


def run(sizer, clock, clips_per_second) -> None:
    """Steps at the sizer's current size, each taking size / speed(size) seconds."""
    for _ in range(40):
        size = sizer.size
        clock.now += size / clips_per_second(size)
        sizer.tick(size)
        if sizer.size != size or sizer.done:
            return


def test_batch_grows_while_it_is_faster_and_keeps_the_fastest(clock):
    speed = {16: 100.0, 32: 180.0, 64: 300.0, 128: 310.0, 256: 305.0}  # saturates at 64
    lines = []
    sizer = molab.BatchSizer(16, "train", SimpleNamespace(type="cuda"), log=lines.append)
    while not sizer.done:
        run(sizer, clock, lambda s: speed[s])
    assert sizer.size == 64
    assert "larger was not faster" in lines[-1] and "GPU memory in use 60 of 100 GB" in lines[-1]


def test_stale_batches_from_before_a_resize_are_ignored(clock):
    sizer = molab.BatchSizer(16, "codec", SimpleNamespace(type="cuda"), log=lambda *_: None)
    run(sizer, clock, lambda s: 100.0 * s / 16)  # perfect scaling
    assert sizer.size == 32
    for _ in range(50):  # old-size batches still in the queue: they must not count as the 32 measurement
        clock.now += 16 / 100.0
        sizer.tick(16)
    assert sizer.size == 32 and not sizer.done and sizer.steps == 0


def test_out_of_memory_after_a_doubling_goes_back_to_the_last_size_that_fit(clock):
    sizer = molab.BatchSizer(16, "refiner", SimpleNamespace(type="cuda"), log=lambda *_: None)
    run(sizer, clock, lambda s: 100.0 * s / 16)
    assert sizer.size == 32
    assert sizer.out_of_memory() == 16 and sizer.done
    assert sizer.out_of_memory() == 12  # a later out-of-memory (the discriminator joins) shrinks by a quarter


def test_a_fixed_batch_and_the_cpu_never_resize(clock):
    for sizer in (
        molab.BatchSizer(24, "train", SimpleNamespace(type="cuda"), fixed=True),
        molab.BatchSizer(24, "train", SimpleNamespace(type="cpu")),
    ):
        run(sizer, clock, lambda s: 1000.0 * s)
        assert sizer.size == 24 and sizer.done


def ints(gen, n):
    return torch.randint(0, 1 << 30, (n,), generator=gen)


@pytest.mark.parametrize("mode", ["thread-forced", "process"])
def test_workers_have_their_own_random_streams_and_batches_have_the_asked_size(mode):
    size = {"n": 12}
    prefetch = molab.Prefetcher(ints, lambda: size["n"], chunk=5, workers=3, seed=7, mode=mode)
    try:
        batches = [prefetch.get() for _ in range(6)]
        assert all(b.shape == (12,) for b in batches)  # three chunks of five, cut to twelve
        values = torch.cat(batches).tolist()
        assert len(set(values)) == len(values)  # no worker repeats another
        size["n"] = 3  # a resize takes effect on the very next batch
        assert prefetch.get().shape == (3,)
    finally:
        prefetch.close()
    assert 0.0 <= prefetch.wait_share() <= 1.0


def test_train_shaped_chunks_are_joined_field_by_field():
    def make(gen, n):
        return torch.ones(n, 4), torch.zeros(n, 4), [torch.ones(3)] * n, [1] * n, [2] * n, n, torch.full((n,), 0.5)

    for mode in ("thread-forced", "process"):  # the loader turns tuples into lists: the fields must survive
        prefetch = molab.Prefetcher(make, lambda: 10, chunk=4, workers=2, mode=mode)
        try:
            clean, dirty, f0s, accents, singers, rows, weights = prefetch.get()
        finally:
            prefetch.close()
        assert clean.shape == (10, 4) and dirty.shape == (10, 4) and weights.shape == (10,)
        assert len(f0s) == len(accents) == len(singers) == 10 and rows == 10


def test_a_pinned_chunk_keeps_its_tuple():
    chunk = molab._Chunk((torch.ones(2), [torch.zeros(1)], 3))
    if torch.cuda.is_available():
        chunk = chunk.pin_memory()
    a, b, n = chunk.data
    assert isinstance(chunk.data, tuple) and a.shape == (2,) and isinstance(b, list) and n == 3


def test_a_worker_error_reaches_the_training_loop():
    def broken(gen, n):
        raise ValueError("bad clip")

    for mode in ("thread-forced", "process"):
        prefetch = molab.Prefetcher(broken, lambda: 4, workers=2, mode=mode)
        try:
            with pytest.raises(ValueError, match="bad clip"):
                prefetch.get()
        finally:
            prefetch.close()


def test_a_dead_worker_process_falls_back_to_threads():
    import os

    def dies_in_a_worker(gen, n):
        if torch.utils.data.get_worker_info() is not None:
            os._exit(1)  # what a bus error from a tiny /dev/shm looks like to the loader
        return ints(gen, n)

    lines = []
    prefetch = molab.Prefetcher(dies_in_a_worker, lambda: 6, chunk=3, workers=2, mode="process", log=lines.append)
    try:
        assert prefetch.get().shape == (6,)
        assert prefetch.mode == "thread" and any("switching to threads" in line for line in lines)
    finally:
        prefetch.close()


def test_small_shared_memory_means_threads(monkeypatch):
    monkeypatch.setattr(molab, "shared_memory_gb", lambda: 0.06)
    lines = []
    prefetch = molab.Prefetcher(ints, lambda: 4, workers=4, device=SimpleNamespace(type="cuda"), log=lines.append)
    try:
        assert prefetch.mode == "thread" and "--shm-size" in lines[0]
    finally:
        prefetch.close()


@pytest.mark.parametrize("mode", ["thread-forced", "process"])
def test_the_program_exits_cleanly_with_workers_mid_batch(mode, tmp_path):
    import subprocess
    import sys

    script = tmp_path / "run.py"
    script.write_text(
        "import torch\n"
        "from helix_lm.scripts import tristream_molab as molab\n"
        "def slow(gen, n):\n"
        "    x = torch.randn(n, 400_000, generator=gen)\n"
        "    for _ in range(20):\n"
        "        x = torch.fft.irfft(torch.fft.rfft(x), n=400_000)\n"
        "    return x\n"
        f"p = molab.Prefetcher(slow, lambda: 2, chunk=2, workers=3, mode='{mode}')\n"
        "p.get()\n"
        "p.close()\n"
        "print('done')\n"
    )
    import os

    env = dict(os.environ, PYTHONPATH=str(Path(molab.__file__).parents[2]))
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, result.stderr[-500:]
    assert "done" in result.stdout and "terminate called" not in result.stderr


def test_loader_workers_use_every_core_but_two(monkeypatch):
    monkeypatch.setattr(molab.os, "cpu_count", lambda: 32)
    assert molab.loader_workers() == 30
    assert molab.loader_workers(5) == 5
    monkeypatch.setattr(molab.os, "cpu_count", lambda: 2)
    assert molab.loader_workers() == 2


def test_more_worker_processes_build_batches_faster():
    from helix_lm.audio.singing import estimate_f0
    from helix_lm.music.regularize import augment

    def make(gen, n):
        waves = []
        for _ in range(n):
            wave = torch.randn(48000, generator=gen) * 0.1
            wave, _ = augment(wave, 24000, seed=int(torch.randint(0, 1 << 30, (), generator=gen)))
            waves.append(wave)
        waves = torch.stack(waves)
        return waves, estimate_f0(waves, 24000, frame_rate=100.0)

    def rate(workers: int) -> float:
        prefetch = molab.Prefetcher(make, lambda: 4, chunk=4, workers=workers, mode="process")
        try:
            prefetch.get()
            begin = time.perf_counter()
            for _ in range(16):
                prefetch.get()
            return 16 / (time.perf_counter() - begin)
        finally:
            prefetch.close()

    one, three = rate(1), rate(3)
    print(f"batches/s: 1 worker process {one:.1f}, 3 worker processes {three:.1f}")
    assert three > 1.6 * one


def test_raising_the_hours_reopens_a_finished_stage(tmp_path):
    import json
    import os

    args = SimpleNamespace(size="ts-tiny", codec_hours=10.0, refiner_hours=2.0, train_hours=8.0, air_hours=0.0)
    (tmp_path / "markers").mkdir()
    for stage, path in (("codec", "codec/codec.pt"), ("train", "runs/ts-tiny/latest.pt")):
        (tmp_path / "markers" / stage).write_text("x")
        ckpt = tmp_path / path
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"weights")
        ckpt.with_suffix(".json").write_text(json.dumps({"step": 100, "hours": 10.0 if stage == "codec" else 8.0}))
    assert molab.complete(tmp_path, "codec", args)
    args.codec_hours = 14.0  # rented more time: the codec trains its 4 extra hours
    assert not molab.complete(tmp_path, "codec", args)
    assert molab.complete(tmp_path, "train", args)
    (tmp_path / "markers" / "air").write_text("x")
    assert molab.complete(tmp_path, "air", args)  # zero hours, ran once: nothing to do
    # the scores are measured again after the model trained more
    (tmp_path / "markers" / "eval").write_text("x")
    summary = tmp_path / "eval" / "summary.json"
    summary.parent.mkdir()
    summary.write_text("{}")
    os.utime(summary, (2_000_000_000, 2_000_000_000))
    assert molab.complete(tmp_path, "eval", args)
    os.utime(tmp_path / "runs" / "ts-tiny" / "latest.pt", (2_000_000_100, 2_000_000_100))
    assert not molab.complete(tmp_path, "eval", args)
