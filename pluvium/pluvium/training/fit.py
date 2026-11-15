# Copyright 2026 Nathan. Apache-2.0.
"""
Sizing a training step to the GPU's memory.

The first rule kept the batch that trained fastest per forecast. Speed per forecast stops rising at a small batch, so
on a large card most of the memory stayed empty: a 90 GB card could sit at 20 GB. This module fills the card instead.
It is given one thing, a way to run a few training steps at a setting and read back how much memory they needed, and
it looks for the setting that comes closest to a memory budget without passing it.

Three things are set:

* **gradient checkpointing**, on or off. Checkpointing saves memory by computing parts of the network twice. Both
  ways are sized, and the one that trains more forecasts per second is kept, unless it would leave more than 15% of
  the budget unused and the other would not;
* **the batch**: as many forecasts per step as fit. Memory is close to a straight line in the batch, so two small
  measurements predict the largest batch, and that prediction is then measured and corrected;
* **target cells per forecast**: a whole extra forecast often does not fit in what is left. Scoring each forecast at
  more of the area's cells does, and gives every forecast more to learn from.

Nothing here touches the GPU. :class:`pluvium.train.Trainer` supplies the measurements, and the tests supply
measurements from a made-up card, which is how this logic is checked on a machine with no GPU.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field


@dataclass
class Probe:
    """A few training steps at one setting. ``peak`` is the memory they needed in bytes, or None when they ran out;
    ``rate`` is forecasts per second."""

    batch: int
    scale: float                       # target cells per forecast, as a multiple of the usual number
    checkpointing: bool
    peak: float | None
    rate: float = 0.0

    @property
    def fits(self) -> bool:
        return self.peak is not None

    def within(self, budget: float) -> bool:
        return self.peak is not None and self.peak <= budget


@dataclass
class Fit:
    """The setting kept, what it measured and what was tried on the way."""

    batch: int
    scale: float
    checkpointing: bool
    peak: float
    budget: float
    rate: float
    probes: list = field(default_factory=list)

    @property
    def share(self) -> float:
        """How much of the budget the kept setting uses."""
        return self.peak / self.budget if self.budget > 0 else 0.0

    def to_json(self) -> dict:
        out = asdict(self)
        out.pop("probes")
        return out

    def describe(self) -> str:
        gb = 2.0**30
        return (f"batch {self.batch}, {self.scale:.2f} times the usual target cells, gradient checkpointing "
                f"{'on' if self.checkpointing else 'off'}: {self.peak / gb:.1f} of {self.budget / gb:.1f} GB "
                f"({self.share:.0%} of the budget), {self.rate:.2f} forecasts/s")


def largest_batch(measure, budget: float, checkpointing: bool, max_batch: int = 1024, close: float = 0.94,
                  tries: int = 8) -> Probe | None:
    """The largest batch whose steps stay within ``budget`` bytes with the usual target cells, or None when not even
    one forecast fits. ``measure(batch, scale, checkpointing) -> Probe``.

    Memory is taken to be a straight line in the batch. Batches 1 and 2 give its slope; the batch it predicts is
    measured, and the line is drawn again through what was measured until a batch within ``close`` of the budget is
    found or nothing larger fits.
    """
    one = measure(1, 1.0, checkpointing)
    if not one.within(budget):
        return None
    if max_batch < 2:
        return one
    two = measure(2, 1.0, checkpointing)
    if not two.within(budget):
        return one
    best = two
    per = max(two.peak - one.peak, 0.02 * one.peak)          # bytes one more forecast costs; noise must not make it 0
    base = two.peak - 2 * per
    too_big, failures = None, 0                               # the smallest batch known not to fit
    for _ in range(tries):
        if best.peak >= close * budget or best.batch >= max_batch:
            break
        guess = min(int((budget - base) // per), max_batch)
        if too_big is not None:
            guess = min(guess, too_big - 1)
            if failures >= 2:                                 # the line keeps promising too much: halve the gap
                guess = min(guess, (best.batch + too_big) // 2)
        if guess <= best.batch:
            break
        got = measure(guess, 1.0, checkpointing)
        if got.within(budget):
            per = max((got.peak - best.peak) / (got.batch - best.batch), 0.25 * per)
            base = got.peak - got.batch * per
            best = got
        else:
            too_big, failures = guess, failures + 1
            if got.fits:                                      # it ran, over the budget: that still measures the slope
                per = max((got.peak - best.peak) / (got.batch - best.batch), per)
            else:
                per *= 1.15                                   # out of memory: a forecast costs more than was thought
            base = best.peak - best.batch * per
    return best


def top_up(measure, budget: float, best: Probe, max_scale: float, close: float = 0.94, tries: int = 3) -> Probe:
    """Spend what is left of the budget on more target cells per forecast, up to ``max_scale`` times the usual
    number. Returns ``best`` unchanged when there is nothing to gain or nothing fits."""
    if max_scale <= 1.02 or best.peak >= close * budget:
        return best
    first = min(max_scale, 1.5)
    got = measure(best.batch, first, best.checkpointing)
    if not got.fits:
        got = measure(best.batch, 1.0 + (first - 1.0) / 3.0, best.checkpointing)
        return got if got.within(budget) else best
    per = (got.peak - best.peak) / (got.scale - 1.0)          # bytes per unit of scale
    if per <= 0:                                              # more cells cost nothing measurable: take them all
        wanted = max_scale
    else:
        wanted = min(max_scale, 1.0 + (budget - best.peak) / per)
    kept = got if got.within(budget) else best
    if got.within(budget) and wanted <= got.scale + 0.02:
        return kept
    if not got.within(budget):
        wanted = 1.0 + 0.95 * (wanted - 1.0)                  # a little short of what the line says
        if wanted < 1.03:
            return best
    low = kept.scale
    for _ in range(tries):
        trial = measure(best.batch, wanted, best.checkpointing)
        if trial.within(budget):
            return trial
        wanted = (low + wanted) / 2.0                         # too much: half way back to what is known to fit
        if wanted <= low + 0.02:
            break
    return kept


def fill(measure, budget: float, max_scale: float = 1.0, modes=(False, True), max_batch: int = 1024,
         close: float = 0.94, enough: float = 0.85, say=None) -> Fit:
    """The setting that fills ``budget`` bytes: gradient checkpointing on or off, the largest batch that way, and
    more target cells per forecast with what is left.

    Of the two ways, the one that trains more forecasts per second at its largest batch is tried first. If it ends
    below ``enough`` of the budget (one forecast can take half a card, and a second does not fit), the other way is
    sized too and the fuller of the two is kept: the purpose here is to fill the card.

    ``measure(batch, scale, checkpointing) -> Probe`` runs a few training steps. Raises ``MemoryError`` when one
    forecast does not fit either way.
    """
    probes = []

    def measured(batch: int, scale: float, checkpointing: bool) -> Probe:
        probe = measure(batch, scale, checkpointing)
        probes.append(probe)
        if say is not None:
            gb = "out of memory" if probe.peak is None else f"{probe.peak / 2**30:.1f} GB"
            say(f"  batch {batch:>3}, cells x{scale:.2f}, checkpointing {'on ' if checkpointing else 'off'}: {gb}"
                + (f", {probe.rate:.2f} forecasts/s" if probe.peak is not None else ""))
        return probe

    found = [best for mode in modes if (best := largest_batch(measured, budget, mode, max_batch, close)) is not None]
    if not found:
        raise MemoryError(f"one forecast does not fit in {budget / 2**30:.1f} GB, with or without gradient checkpointing")
    fastest = max(p.rate for p in found)                      # within 5% of each other the speeds are a tie,
    first = max((p for p in found if p.rate >= 0.95 * fastest), key=lambda p: p.peak)     # and the fuller one wins
    chosen = top_up(measured, budget, first, max_scale, close)
    if chosen.peak < enough * budget:                         # the faster way leaves too much of the card unused
        for other in found:
            if other is not first:
                fuller = top_up(measured, budget, other, max_scale, close)
                if fuller.peak > chosen.peak:
                    chosen = fuller
    return Fit(chosen.batch, chosen.scale, chosen.checkpointing, float(chosen.peak), float(budget), chosen.rate, probes)


def lr_scale(batch: int, reference: int = 8, cap: float = 2.0) -> float:
    """What to multiply the learning rate by at ``batch``. A larger batch means fewer, steadier updates in the same
    time, so each may be larger: the rate grows with the square root of the batch above ``reference``, up to
    ``cap``. At or below ``reference`` it is left alone, which is how every earlier run was trained."""
    if batch <= reference:
        return 1.0
    return float(min(cap, math.sqrt(batch / reference)))


def ram_bytes(default: float = 32 * 2.0**30) -> float:
    """The machine's memory in bytes, or ``default`` where it cannot be asked."""
    try:
        return float(os.sysconf("SC_PAGE_SIZE")) * float(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return default


def loader_plan(workers: int, batch_bytes: float, ram: float, share: float = 0.3, prefetch: int = 4) -> tuple:
    """``(workers, batches each keeps ready)`` so that the finished batches waiting in the machine's memory stay
    within ``share`` of it. A batch that fills a large GPU is large in RAM too, and every loader worker holds
    several: with nothing done about it, the loader and not the GPU is what runs out of memory."""
    if workers <= 0 or batch_bytes <= 0:
        return workers, prefetch
    budget = share * ram
    held = lambda w, p: w * (p + 1) * batch_bytes             # noqa: E731  (ready batches, and the one being built)
    while prefetch > 1 and held(workers, prefetch) > budget:
        prefetch -= 1
    while workers > 1 and held(workers, prefetch) > budget:
        workers -= 1
    return workers, prefetch


__all__ = ["Fit", "Probe", "fill", "largest_batch", "loader_plan", "lr_scale", "ram_bytes", "top_up"]
