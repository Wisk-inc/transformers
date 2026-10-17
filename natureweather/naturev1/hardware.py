# Copyright 2026 Nathan. Apache-2.0.
"""
Before spending three hours of eight B300s: check the machine, the software, the disks, the network, and
measure what a step actually costs -- then say how many passes over the data the budget buys.

Nothing here changes anything. Every check prints one line, marked ``ok``, ``!!`` (will not work) or
``..`` (works, but worth knowing), and :func:`hardware_preflight` returns whether anything was ``!!``.
"""

from __future__ import annotations

import os
import shutil
import time
import urllib.request
from pathlib import Path

import torch


def _line(mark: str, text: str, lines: list[str]) -> None:
    lines.append(f"  {mark} {text}")


def gpu_report(lines: list[str]) -> bool:
    if not torch.cuda.is_available():
        _line("!!", "no CUDA GPU visible to PyTorch", lines)
        return False
    ok = True
    count = torch.cuda.device_count()
    arch = torch.cuda.get_arch_list()
    _line("ok", f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {count} GPU(s); built for {' '.join(arch)}", lines)
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        major, minor = props.major, props.minor
        # A cubin built for X.y runs on X.z for z >= y; Blackwell data-centre parts (B200 10.0, B300 10.3) need
        # a build that carries sm_100 (CUDA 12.8 or later) -- or PTX it can JIT from.
        supported = any(a in (f"sm_{major}{minor}", f"sm_{major}0", f"compute_{major}0", f"compute_{major}{minor}")
                        for a in arch)
        mark = "ok" if supported else "!!"
        ok = ok and supported
        _line(mark, f"GPU {index}: {props.name}, {props.total_memory / 1e9:.0f} GB, compute capability {major}.{minor}"
              + ("" if supported else " -- this PyTorch has no kernels for it: install a CUDA 12.8+ build "
                                      "(pip install torch --index-url https://download.pytorch.org/whl/cu128 or newer)"),
              lines)
    if ok:
        for index in range(count):
            a = torch.randn(4096, 4096, device=f"cuda:{index}", dtype=torch.bfloat16)
            torch.cuda.synchronize(index)
            began = time.time()
            for _ in range(20):
                b = a @ a
            torch.cuda.synchronize(index)
            seconds = (time.time() - began) / 20
            tflops = 2 * 4096 ** 3 / seconds / 1e12
            finite = bool(torch.isfinite(b).all())
            _line("ok" if finite else "!!", f"GPU {index}: bf16 matmul {tflops:,.0f} TFLOP/s (4096^3, untuned)"
                  + ("" if finite else " -- NON-FINITE RESULT"), lines)
            ok = ok and finite
    return ok


def disk_report(data_dir: str | Path, need_tb: float, lines: list[str]) -> bool:
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free / 1e12
    mark = "ok" if free >= need_tb else "!!"
    _line(mark, f"{path}: {free:.2f} TB free, {need_tb:.2f} TB needed for staging", lines)
    try:
        test = path / ".preflight.bin"
        block = os.urandom(1 << 26)
        began = time.time()
        with open(test, "wb") as handle:
            for _ in range(16):
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        write = 1.0 / max(time.time() - began, 1e-9)
        test.unlink()
        _line("ok" if write > 0.5 else "..", f"{path}: sequential write {write:.1f} GB/s", lines)
    except OSError as error:
        _line("!!", f"{path}: cannot write ({error})", lines)
        return False
    return mark == "ok"


def memory_report(lines: list[str]) -> None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        _line("ok", f"host: {os.cpu_count()} CPUs, {pages / 1e9:.0f} GB RAM (the page cache holds the hot data)", lines)
    except (ValueError, OSError):
        _line("..", f"host: {os.cpu_count()} CPUs", lines)


def network_report(hub_repo: str | None, lines: list[str]) -> bool:
    ok = True
    probes = (("GridSat-B1 on AWS", "https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com/data/2005/GRIDSAT-B1.2005.08.29.12.v02r01.nc"),
              ("IBTrACS at NCEI", "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/"),
              ("WeatherBench2 on GCS", "https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr/.zmetadata"))
    for name, url in probes:
        try:
            request = urllib.request.Request(url, method="HEAD")
            began = time.time()
            with urllib.request.urlopen(request, timeout=30) as response:
                code = response.status
            _line("ok", f"{name}: reachable ({code}, {1000 * (time.time() - began):.0f} ms)", lines)
        except Exception as error:
            _line("!!", f"{name}: {type(error).__name__}: {error}", lines)
            ok = False
    if hub_repo:
        try:
            from huggingface_hub import HfApi

            user = HfApi(token=os.environ.get("HF_TOKEN") or None).whoami()
            _line("ok", f"Hugging Face: logged in as {user.get('name')}; every checkpoint will go to {hub_repo}", lines)
        except Exception as error:
            _line("!!", f"Hugging Face: no usable token ({type(error).__name__}) -- checkpoints would stay local", lines)
            ok = False
    return ok


def throughput_report(hours: float, satnet_samples: int, retrieval_samples: int, gpus_each: int,
                      lines: list[str]) -> None:
    """Time a few full-size training steps of each network on GPU 0 and project passes within ``hours``."""
    if not torch.cuda.is_available():
        return
    from ihelix import fibonacci_sphere

    from .model import NatureConfig, NatureV1

    device = torch.device("cuda", 0)
    config = NatureConfig(analysis_channels=96, state_channels=86, satnet_embed=512, retrieval_layers=6)
    model = NatureV1(config, fibonacci_sphere(4096, num_neighbours=48, cluster_size=64)).to(device)
    for name, batch, run in (("storm network", 128, "satnet"), ("retrieval", 8, "retrieval")):
        module = model.storm_satnet if run == "satnet" else model.retrieval
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4)
        if run == "satnet":
            inputs = (torch.randint(1, 255, (batch, 4, 2, 256, 256), dtype=torch.uint8, device=device),
                      torch.ones(batch, 4, dtype=torch.bool, device=device), torch.zeros(batch, device=device),
                      torch.zeros(batch, 4, device=device))
            step = (lambda: sum(v.float().mean() for v in module(*inputs).values()))
        else:
            from ihelix import FieldGrid, Geometry

            from .corpora import latlon_to_coords
            import numpy as np

            coords = latlon_to_coords(np.linspace(-90, 90, 121), np.arange(240) * 1.5)
            grid = model._aligned(FieldGrid.source(coords, Geometry.globe(), weights=torch.ones(coords.shape[0])))
            sat = torch.randint(1, 255, (batch, 3, 8, 560, 1440), dtype=torch.uint8, device=device)
            sat[:, :, 7] = 1
            inputs = (sat, torch.ones(batch, 3, dtype=torch.bool, device=device),
                      torch.randn(batch, grid.num_points, 96, device=device) * 0.1,
                      torch.full((batch,), 12.0, device=device), torch.zeros(batch, 6, device=device),
                      grid, model.latent_grid, model.link)
            step = (lambda: module(*inputs)["increment"].float().pow(2).mean())
        times = []
        for index in range(6):
            torch.cuda.synchronize(device)
            began = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = step()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            if index >= 2:
                times.append(time.time() - began)
        per_step = sum(times) / len(times)
        rate = batch / per_step * gpus_each
        samples = satnet_samples if run == "satnet" else retrieval_samples
        passes = rate * hours * 3600 / max(samples, 1)
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        _line("ok", f"{name}: {per_step * 1000:.0f} ms per step of {batch} on one GPU ({peak:.0f} GB peak) -> "
              f"~{rate:,.0f} samples/s on {gpus_each} GPUs, ~{passes:.0f} passes over {samples:,} samples in "
              f"{hours:g} h (if the disks keep up)", lines)
        torch.cuda.reset_peak_memory_stats(device)
    del model
    torch.cuda.empty_cache()


def hardware_preflight(data_dir: str | Path, hub_repo: str | None = None, need_tb: float = 1.6, hours: float = 3.0,
              satnet_samples: int = 200_000, retrieval_samples: int = 52_000, gpus_each: int = 4,
              measure: bool = True) -> bool:
    """Run every check, print the report, return True when nothing would stop the run."""
    lines: list[str] = ["PREFLIGHT"]
    ok = gpu_report(lines)
    memory_report(lines)
    ok = disk_report(data_dir, need_tb, lines) and ok
    ok = network_report(hub_repo, lines) and ok
    if ok and measure:
        try:
            throughput_report(hours, satnet_samples, retrieval_samples, gpus_each, lines)
        except Exception as error:                                   # a measurement must not block the run
            _line("..", f"throughput probe failed: {type(error).__name__}: {error}", lines)
    lines.append("  => " + ("ready" if ok else "fix the !! lines first"))
    print("\n".join(lines), flush=True)
    return ok


__all__ = ["hardware_preflight"]
