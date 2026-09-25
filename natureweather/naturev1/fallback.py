# Copyright 2026 Nathan. Apache-2.0.
"""
Fallbacks: degrade to something honest instead of failing, and say which path was taken.

A pipeline that only works when everything is available is a pipeline that works on the machine it was
written on. The store may not carry the levels you asked for; the disk may not hold the years you
wanted to stage; the card may not do bf16; the network may drop halfway through a download.

Every fallback here follows the same two rules, because a silent fallback is worse than a crash:

* **it says what it did**, in the line where it did it, so a surprising result later can be traced to
  the moment the run stopped doing what you asked;
* **it never fabricates data**. Falling back to fewer variables is fine. Filling a missing variable
  with zeros, or with the mean, is not -- that is inventing an observation, and the model cannot tell
  the difference between a measurement and a guess unless you tell it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


@dataclass
class Choice:
    """One decision the pipeline made for you, and why."""

    what: str
    took: str
    instead_of: str
    because: str

    def __repr__(self) -> str:
        return f"{self.what}: {self.took} instead of {self.instead_of} ({self.because})"


@dataclass
class Plan:
    """The configuration a run actually ended up with, with every substitution recorded."""

    choices: list[Choice] = field(default_factory=list)
    values: dict = field(default_factory=dict)

    def note(self, what: str, took: str, instead_of: str, because: str) -> None:
        self.choices.append(Choice(what, took, instead_of, because))
        print(f"[fallback] {self.choices[-1]}", flush=True)

    def __repr__(self) -> str:
        if not self.choices:
            return "Plan(no substitutions -- running exactly as asked)"
        return "Plan(\n  " + "\n  ".join(repr(c) for c in self.choices) + "\n)"

    def summary(self) -> str:
        lines = ["configuration"]
        for key, value in self.values.items():
            lines.append(f"  {key:22} {value}")
        if self.choices:
            lines.append("\nsubstitutions made")
            for choice in self.choices:
                lines.append(f"  {choice.what:22} {choice.took}  (wanted {choice.instead_of}: {choice.because})")
        else:
            lines.append("\n  no substitutions -- running exactly as asked")
        return "\n".join(lines)


def resolve_variables(dataset, wanted, levels=None, plan: Plan | None = None):
    """
    Keep what the store actually carries, and name what it does not.

    Silently dropping a variable is how a run ends up training on eleven channels while the log says
    eighty-nine. Anything missing is reported by name.

    Returns:
        ``(kept, levels, plan)``.
    """
    plan = plan or Plan()
    kept = [name for name in wanted if name in dataset]
    missing = [name for name in wanted if name not in dataset]
    if missing:
        plan.note("variables", f"{len(kept)} of {len(wanted)}", "all requested",
                  f"not in this store: {', '.join(missing)}")

    if levels is not None and "level" in dataset.dims:
        available = {int(v) for v in dataset.level.values}
        keep_levels = [lv for lv in levels if lv in available]
        dropped = [lv for lv in levels if lv not in available]
        if dropped:
            plan.note("levels", f"{len(keep_levels)} levels", f"{len(levels)} levels",
                      f"store lacks {dropped} hPa")
        levels = keep_levels
    elif levels is not None:
        plan.note("levels", "surface only", f"{len(levels)} levels",
                  "this store has no pressure levels -- medium-range skill lives in Z500, so expect "
                  "little beyond persistence")
        levels = None

    if not kept:
        raise ValueError(f"none of {list(wanted)} are in this store; nothing to train on")
    return kept, levels, plan


def resolve_staging(years: int, channels: int, points: int, path: str = ".",
                    headroom: float = 1.3, plan: Plan | None = None, staged=()):
    """
    Stage as many years as the disk will actually hold, rather than filling it and dying.

    ``staged`` names the staging files this run would reuse or replace. Their size counts as available:
    measured against free space alone, a finished staging made the next run's plan smaller -- 45 GB
    of ERA5 on disk is 45 GB less free -- the smaller plan did not match the file, and the file was
    downloaded all over again. With it, the same request gives the same plan whatever is on disk.

    Returns:
        ``(years, streaming, plan)``. ``streaming`` is True when nothing fits and the run must read
        from the network, which is roughly 100x slower per window but always works.
    """
    plan = plan or Plan()
    per_year_gb = points * channels * 2 * 1460 / 1e9

    try:
        usage = os.statvfs(path)
        free_gb = usage.f_bavail * usage.f_frsize / 1e9
    except OSError:
        plan.note("staging", "as asked", "a disk check", f"cannot stat {path}")
        return years, False, plan
    from pathlib import Path

    free_gb += sum(Path(name).stat().st_size for name in staged if Path(name).exists()) / 1e9

    affordable = int(free_gb / (per_year_gb * headroom))
    if affordable >= years:
        plan.values["staging"] = f"{years} years, {years * per_year_gb:.1f} GB of {free_gb:.0f} GB free"
        return years, False, plan

    if affordable < 1:
        plan.note("staging", "streaming from the network", f"{years} years on disk",
                  f"{free_gb:.0f} GB free, one year needs {per_year_gb:.1f} GB. "
                  "Expect ~100x slower per window; use fewer channels or a bigger disk.")
        return 0, True, plan

    plan.note("staging", f"{affordable} years", f"{years} years",
              f"{free_gb:.0f} GB free at {per_year_gb:.1f} GB/year")
    return affordable, False, plan


def resolve_precision(requested: str = "bf16", plan: Plan | None = None) -> tuple[str, Plan]:
    """
    bf16 on hardware that has it, fp16 with loss scaling where it does not, fp32 on CPU.

    Getting this wrong is not a slowdown, it is silent NaN: fp16 without loss scaling underflows small
    gradients to zero and the run learns nothing while the loss curve looks ordinary.
    """
    import torch

    plan = plan or Plan()
    if not torch.cuda.is_available():
        if requested != "fp32":
            plan.note("precision", "fp32", requested, "no CUDA device; half precision on CPU is slower")
        return "fp32", plan
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        plan.note("precision", "fp16", "bf16",
                  "this card has no bf16; fp16 needs loss scaling, which the Trainer enables")
        return "fp16", plan
    plan.values["precision"] = requested
    return requested, plan


def resolve_batch(requested: int, points: int, channels: int, layers: int,
                  vram_gb: float | None = None, plan: Plan | None = None) -> tuple[int, Plan]:
    """
    A batch that fits, estimated from activation size, for when autotuning is not worth the minutes.

    Deliberately conservative: an out-of-memory error two hours into a run costs more than a batch
    that is 20% smaller than it could have been.
    """
    import torch

    plan = plan or Plan()
    if vram_gb is None:
        if not torch.cuda.is_available():
            if requested > 1:
                plan.note("batch", "1", str(requested), "running on CPU")
            return 1, plan
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    # Activations dominate: roughly points x channels x layers x a few tensors, in bf16, with
    # gradient checkpointing assumed on.
    per_sample_gb = points * max(channels, 64) * layers * 6 * 2 / 1e9
    affordable = max(1, int(vram_gb * 0.7 / max(per_sample_gb, 1e-6)))
    if affordable < requested:
        plan.note("batch", str(affordable), str(requested),
                  f"~{per_sample_gb:.2f} GB per sample against {vram_gb:.0f} GB")
        return affordable, plan
    plan.values["batch"] = requested
    return requested, plan


def with_retry(call, attempts: int = 4, delay: float = 2.0, what: str = "request"):
    """
    Retry a network read with exponential backoff, then give up loudly.

    Cloud stores drop connections. A staging run that dies at 80% because one chunk timed out has
    wasted an hour for no reason, and the fix is four lines.
    """
    last = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as error:               # noqa: BLE001 -- deliberately broad: any transport error
            last = error
            if attempt == attempts - 1:
                break
            wait = delay * 2**attempt
            print(f"[retry] {what} failed ({type(error).__name__}: {error}); "
                  f"retrying in {wait:.0f}s ({attempt + 1}/{attempts - 1})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{what} failed after {attempts} attempts: {last}") from last


def resolve_all(dataset, wanted, levels, years: int, points: int, layers: int,
                batch: int, precision: str = "bf16", path: str = ".", staged=()) -> Plan:
    """
    Every fallback in one call, returning the configuration a run will actually use.

    Print :meth:`Plan.summary` before training. If it says something you did not intend, that is the
    moment to stop -- not three hours in, wondering why the loss will not fall.
    """
    plan = Plan()
    kept, levels, plan = resolve_variables(dataset, wanted, levels, plan)

    from .era5 import expand_variables

    channels = len(expand_variables(dataset, kept, levels))
    years, streaming, plan = resolve_staging(years, channels, points, path, plan=plan, staged=staged)
    precision, plan = resolve_precision(precision, plan)
    batch, plan = resolve_batch(batch, points, channels, layers, plan=plan)

    plan.values.update({
        "variables": f"{len(kept)} -> {channels} channels",
        "levels": len(levels) if levels else "surface only",
        "staging": "stream" if streaming else f"{years} years",
        "precision": precision,
        "batch": batch,
    })
    plan.resolved = {"variables": kept, "levels": levels, "channels": channels, "years": years,
                     "streaming": streaming, "precision": precision, "batch": batch}
    return plan


# --------------------------------------------------------------------------------------------------
# Installing, which is the first thing that can go wrong and the one with the worst error message
# --------------------------------------------------------------------------------------------------

def install_packages(requirements, quiet: bool = True) -> tuple[bool, list[str]]:
    """
    Install these requirements, trying every reasonable strategy before giving up.

    Returns:
        ``(ok, attempts)``. On failure the caller gets the commands that were tried, so the message
        can say what to run by hand instead of printing a subprocess traceback.
    """
    import shutil
    import subprocess
    import sys

    requirements = list(requirements)
    flags = ["-q"] if quiet else []
    commands = []

    if shutil.which("uv"):
        # --python names the interpreter explicitly: uv otherwise looks for VIRTUAL_ENV, which a
        # notebook kernel started outside the shell that made the venv does not always have set.
        commands.append(("uv", ["uv", "pip", "install", "--python", sys.executable,
                                "--upgrade", *requirements]))
    base = [sys.executable, "-m", "pip", "install", *flags, "--upgrade"]
    commands += [
        ("pip, fresh index", [*base, "--no-cache-dir", "--index-url", "https://pypi.org/simple",
                              *requirements]),
        ("pip, no cache", [*base, "--no-cache-dir", *requirements]),
        ("pip", [*base, *requirements]),
    ]

    attempted = []
    for label, command in commands:
        attempted.append(" ".join(command))
        try:
            subprocess.check_call(command)
            print(f"[install] ok via {label}", flush=True)
            return True, attempted
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            reason = getattr(error, "returncode", type(error).__name__)
            print(f"[install] {label} failed ({reason}), trying the next approach", flush=True)
    return False, attempted


def ensure_packages(needed: dict, quiet: bool = True) -> bool:
    """
    Import-and-version check, then install only what is missing or out of date.

    Args:
        needed: ``{import_name: (requirement, minimum_version_tuple)}``.

    Returns:
        True if anything was installed, in which case the caller should restart the interpreter --
        a package upgraded underneath an already-imported one is not reliably picked up.
    """
    import importlib
    import importlib.util

    def stale(module: str, minimum: tuple) -> bool:
        try:
            version = importlib.import_module(module).__version__
            return tuple(int(part) for part in version.split(".")[:3]) < minimum
        except Exception:
            return True

    missing = [requirement for module, (requirement, minimum) in needed.items()
               if importlib.util.find_spec(module) is None or stale(module, minimum)]
    if not missing:
        return False

    print(f"[install] needed: {', '.join(missing)}", flush=True)
    ok, attempted = install_packages(missing, quiet=quiet)
    if not ok:
        raise RuntimeError(
            "could not install " + ", ".join(missing) + ".\n\nTried:\n  "
            + "\n  ".join(attempted)
            + "\n\nIf the package exists on PyPI but pip cannot see it, the index is being cached "
              "somewhere. Run this by hand and then restart:\n"
              "  !pip install --no-cache-dir --index-url https://pypi.org/simple --upgrade "
              + " ".join(f'"{r}"' for r in missing)
        )
    importlib.invalidate_caches()
    return True
