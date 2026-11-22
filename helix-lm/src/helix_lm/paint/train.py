# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Training Pinxit, in two stages.

**Practice sheets** (:func:`train_sheets`): windows made of strokes whose every stroke is known
(:mod:`helix_lm.paint.synth`). The model learns the craft: to see which strokes are missing and to make them
with the right gesture, brush and pigments. No data is needed.

**Pictures** (:func:`train_pictures`): real photos, each with a plan made by the mind for a style and a feeling
(:func:`helix_lm.paint.mind.make_plan`). For a pass of that plan the model lays its strokes on a canvas that
looks the way earlier passes would have left it, and is judged on three things: how close the canvas comes to
the plan where it matters; whether its strokes tend to be as fast, heavy, long and wide as the untrained hand's
rules say they should be here (those rules are where the gestures come from); and whether it made as many
strokes as the mind allowed. The eye's adapters are trained in this stage too. What the model writes to its
memory in a coarse pass is read by the cells of the next one.

Both stages run for a number of steps or of hours, save every few minutes (the model, a running average of its
weights, the optimizer, and how long it has trained), and go on from a checkpoint.

**Memory.** The Helix trunk keeps a great deal per token while it trains (about 150 MB for one canvas cell at
the small size), so both stages have the trunk recompute its activations in the backward pass, which cuts that
to a quarter for a third more time, and both size their steps to the GPU: as many practice sheets, or as many
cells of real pictures, as fit with a little to spare (:func:`largest_batch`). A picture step takes the passes
of several pictures at once; when they hold more cells than fit, the rest go through the model without
gradients. A step that still runs out of memory is dropped and the next ones are made smaller.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .color import linear_to_oklab, srgb_to_linear
from .engine import StudioHand, lay, open_level, plan_levels
from .eye import load_lora, lora_state
from .io import resize
from .model import Pinxit, feeling_vector
from .strokes import PHASES, Strokes
from .synth import practice_sheet, sheet_loss


class Average:
    """A running average of a model's weights (the ones used for painting). It lives wherever the model does."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                if self.shadow[k].device != v.device:
                    self.shadow[k] = self.shadow[k].to(v.device)
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict) -> None:
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k] = v.to(self.shadow[k].device)

    def full_state(self, model: torch.nn.Module) -> dict:
        """The model's state with the averaged weights in place of the current ones."""
        return {k: self.shadow.get(k, v).cpu() for k, v in model.state_dict().items()}


def save_state(path, model: Pinxit, average: Average | None, optimizer, step: int, hours: float, stage: str,
               eye=None, progress: dict | None = None) -> None:
    """Write a checkpoint through a temporary file, with a small ``.json`` beside it (step, hours, stage).
    ``progress`` is how far the other stages have come (``{stage: {"step", "hours"}}``): each stage keeps its
    own count, so asking for more hours of one continues that one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict

    progress = {**(progress or {}), stage: {"step": int(step), "hours": float(hours)}}
    state = {"config": asdict(model.config), "model": {k: v.cpu() for k, v in model.state_dict().items()},
             "ema": average.full_state(model) if average is not None else None,
             "optimizer": optimizer.state_dict() if optimizer is not None else None, "step": int(step),
             "hours": float(hours), "stage": stage, "progress": progress}
    if eye is not None:                                  # which eye, and what its adapters learned
        state["eye"], state["eye_lora"] = eye.name, lora_state(eye)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    side = path.with_suffix(".json")
    side_tmp = side.with_name(f"{side.name}.{os.getpid()}.tmp")
    side_tmp.write_text(json.dumps({"step": int(step), "hours": float(hours), "stage": stage,
                                    "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "progress": progress,
                                    "total_hours": sum(p["hours"] for p in progress.values())}))
    os.replace(side_tmp, side)


def load_state(path, model: Pinxit, average: Average | None = None, optimizer=None, eye=None, device=None) -> dict:
    """Take a checkpoint up again. Returns ``{"step", "hours", "stage"}`` of the stage that saved it and
    ``"progress"``, the same for every stage that has trained."""
    state = torch.load(Path(path), map_location=device or "cpu", weights_only=True)
    model.load_state_dict(state["model"])
    if average is not None and state.get("ema"):
        average.load_state_dict(state["ema"])
    if optimizer is not None and state.get("optimizer"):
        with contextlib.suppress(ValueError):        # another stage's optimizer (other parameters): start it fresh
            optimizer.load_state_dict(state["optimizer"])
    if eye is not None and state.get("eye_lora"):
        load_lora(eye, state["eye_lora"])
    last = {"step": int(state.get("step", 0)), "hours": float(state.get("hours", 0.0)),
            "stage": state.get("stage", "")}
    progress = state.get("progress") or ({last["stage"]: {"step": last["step"], "hours": last["hours"]}}
                                         if last["stage"] else {})
    return {**last, "progress": progress}


def _autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


@contextlib.contextmanager
def training(model: Pinxit):
    """The model in training mode with the trunk recomputing its activations in the backward pass (a quarter of
    the memory for a third more time, and exactly the same gradients); as it was before on the way out."""
    config, was_training = model.trunk.config, model.training
    before = config.activation_checkpointing
    config.activation_checkpointing = True
    model.train()
    try:
        yield model
    finally:
        config.activation_checkpointing = before
        model.train(was_training)


def _show(name: str, value: float) -> str:
    return f"{name} {value:.1f}" if name in ("pictures", "cells") else f"{name} {value:.4f}"


def _run(step_fn, model, parameters, *, hours, steps, lr, warmup, checkpoint, stage, log, keep_going, save_minutes,
         on_save, eye, shrink=None) -> dict:
    """The loop both stages share: step, clip, average, log, save, stop at the budget. ``shrink()`` makes the
    next steps smaller and says how, after one has run out of GPU memory."""
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    average = Average(model)
    step, done_hours, progress = 0, 0.0, {}
    if checkpoint is not None and Path(checkpoint).exists():
        got = load_state(checkpoint, model, average, optimizer, eye, device)
        progress = got["progress"]
        if stage in progress:
            step, done_hours = progress[stage]["step"], progress[stage]["hours"]
            log(f"[{stage}] went on from step {step} ({done_hours:.2f} h of this stage so far)")
        else:
            log(f"[{stage}] starts from the model as {got['stage'] or 'training'} left it "
                f"(step {got['step']}, {got['hours']:.2f} h)")
    began, last_save, last_log, shown, first, tight = time.time(), time.time(), 0.0, {}, True, 0

    def elapsed() -> float:
        return done_hours + (time.time() - began) / 3600.0

    def save() -> None:
        if checkpoint is not None:
            save_state(checkpoint, model, average, optimizer, step, elapsed(), stage, eye, progress)
            if on_save:
                on_save(step, elapsed())

    while (steps is None or step < steps) and (hours is None or elapsed() < hours) and (keep_going is None
                                                                                        or keep_going()):
        for group in optimizer.param_groups:
            group["lr"] = lr * min(1.0, (step + 1) / max(1, warmup))
        loss, out_of_memory = None, False
        try:
            loss = step_fn(step)
            if loss is not None:
                optimizer.zero_grad(set_to_none=True)
                loss["total"].backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
        except torch.cuda.OutOfMemoryError:
            out_of_memory = True                              # handled below, once the failed step is let go of
        if out_of_memory:
            loss = None
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            tight += 1
            if shrink is None or tight > 8:
                raise RuntimeError(f"[{stage}] out of GPU memory {tight} steps in a row: use a smaller --size")
            log(f"[{stage}] step {step} ran out of GPU memory and was dropped; {shrink()}")
            continue
        step += 1
        if loss is None:
            continue
        tight = 0
        average.update(model)
        for k, v in loss.items():
            v = float(v.detach()) if torch.is_tensor(v) else float(v)
            shown[k] = 0.95 * shown.get(k, v) + 0.05 * v
        if time.time() - last_log > 60 or first:
            memory = ""
            if device.type == "cuda":
                total = torch.cuda.get_device_properties(device).total_memory
                memory = f", GPU memory {torch.cuda.max_memory_allocated(device) / total:.0%} at most"
            log(f"[{stage}] step {step}, {elapsed():.2f} h: " + ", ".join(_show(k, v) for k, v in shown.items())
                + memory)
            last_log, first = time.time(), False
        if time.time() - last_save > save_minutes * 60:
            save()
            last_save = time.time()
    save()
    return {"step": step, "hours": elapsed(), "loss": shown, "average": average}


def largest_batch(try_step, start: int, most: int, device: torch.device, log=print, least: int = 8,
                  share: float = 0.88, measure=None, total: int | None = None, what: str = "batch") -> int:
    """
    How much fits in one step on the GPU, using ``share`` of its memory. ``try_step(size)`` runs a whole step
    (forward and backward) at that size and may return how large it really was. The size is halved from
    ``start`` until a step fits, the memory of a second, larger step gives what one more costs, and the size
    that should fill the GPU is then tried and taken back by a tenth at a time if it does not fit. On a CPU the
    answer is ``start``. ``measure(size) -> (peak bytes or None for out of memory, real size)`` and ``total``
    stand in for the GPU in tests.
    """
    if measure is None:
        if device.type != "cuda":
            return start
        total = torch.cuda.get_device_properties(device).total_memory

        def measure(size: int):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            real = None
            try:
                real = try_step(size)
            except torch.cuda.OutOfMemoryError:
                real = -1
            if real == -1:
                gc.collect()
                torch.cuda.empty_cache()
                return None, size
            return torch.cuda.max_memory_allocated(device), (size if real is None else int(real))

    size = max(least, min(start, most))
    peak, real = measure(size)
    while peak is None or peak > share * total:
        if size <= least:
            raise RuntimeError(f"a step of {least} does not fit in this GPU's memory: use a smaller model size")
        size = max(least, size // 2)
        peak, real = measure(size)
    best = min(size, real)
    if real >= size and size < most:                          # there may be room for more
        peak2, real2 = measure(min(most, 2 * size))
        if peak2 is not None and peak2 <= share * total and real2 > real:
            best = real2
            each = max((peak2 - peak) / (real2 - real), 1.0)
            goal = min(most, int(real2 + (share * total - peak2) / each)) // 8 * 8
            while goal > best:
                peak3, real3 = measure(goal)
                if peak3 is not None and peak3 <= min(0.95, share + 0.05) * total:
                    best = min(goal, real3)
                    break
                goal = int(goal * 0.9) // 8 * 8
    log(f"{what} {best}: what fits in {share:.0%} of the GPU's memory")
    return best


def train_sheets(model: Pinxit, *, hours: float | None = None, steps: int | None = None, batch: int = 0,
                 lr: float = 3e-4, checkpoint=None, log=print, keep_going=None, save_minutes: float = 20.0,
                 on_save=None, seed: int = 0) -> dict:
    """The first stage (see the module's notes). ``batch`` 0 takes as many sheets as fit the GPU (64 on a
    CPU)."""
    device = next(model.parameters()).device
    c = model.config
    generator = torch.Generator().manual_seed(seed + int(time.time()) % 100000)
    from .emotion import NEUTRAL
    from .style import StyleCard

    style, feel = StyleCard().vector().to(device), feeling_vector(NEUTRAL).to(device)
    size = {"batch": batch}

    def one(count: int) -> dict:
        sheet = practice_sheet(count, c.strokes, c.palette, generator, device, c.window)
        with _autocast(device):
            out = model(sheet["target"], sheet["canvas"], sheet["brief"], sheet["palette"], style, feel)
        return sheet_loss(out, sheet, c.window)

    def shrink() -> str:
        size["batch"] = max(8, int(size["batch"] * 0.85))
        return f"the batch is now {size['batch']}"

    with training(model):
        if batch <= 0:
            size["batch"] = 64
            if device.type == "cuda":
                size["batch"] = largest_batch(lambda b: one(b)["total"].backward() and None, 256, 16384, device, log)
                model.zero_grad(set_to_none=True)
        return _run(lambda step: one(size["batch"]), model, list(model.parameters()), hours=hours, steps=steps,
                    lr=lr, warmup=500, checkpoint=checkpoint, stage="sheets", log=log, keep_going=keep_going,
                    save_minutes=save_minutes, on_save=on_save, eye=None, shrink=shrink)


# ------------------------------------------------------------------------------------------- pictures --
class PlanPool:
    """
    Plans kept ready for the picture stage. Making a plan takes a CPU core about a third of a second and a step
    on a large GPU wants dozens, so a plan is not used once: it stays in the pool and is met about ``reuse``
    times, each time at another pass and on another canvas, before a new one takes the place of the oldest.
    When the makers still cannot keep up (more than a quarter of the time is spent waiting for them) ``reuse``
    is doubled, up to 32.
    """

    def __init__(self, plans, size: int = 256, reuse: int = 4, seed: int = 0) -> None:
        self.stream, self.size, self.reuse = iter(plans), max(1, size), max(1, reuse)
        self.pool, self.oldest, self.since = [], 0, 0
        self.made = self.drawn = 0
        self.waited, self.began = 0.0, time.time()
        self.rng = random.Random(seed)

    def _more(self) -> None:
        if self.stream is None:
            return
        began = time.time()
        try:
            plan = next(self.stream)
        except StopIteration:                                 # a list of plans, used up: go round what there is
            self.stream = None
            return
        self.waited += time.time() - began
        if len(self.pool) < self.size:
            self.pool.append(plan)
        else:
            self.pool[self.oldest] = plan
            self.oldest = (self.oldest + 1) % self.size
        self.made, self.since = self.made + 1, 0
        if self.made % 200 == 0:
            if self.waited > 0.25 * (time.time() - self.began) and self.reuse < 32:
                self.reuse *= 2
            self.waited, self.began = 0.0, time.time()

    def draw(self):
        if not self.pool or self.since >= self.reuse:
            self._more()
        if not self.pool:
            raise ValueError("no plans to train on")
        self.since += 1
        self.drawn += 1
        return self.pool[self.rng.randrange(len(self.pool))]

    def note(self) -> str:
        return (f"{self.made} plans made so far, each met about {self.drawn / max(1, self.made):.1f} times "
                f"(a new one every {self.reuse} draws)")


@dataclass
class Pass:
    """One pass of one picture's plan, opened for a training step."""

    plan: object
    levels: list
    index: int
    canvas: torch.Tensor          # the canvas as earlier passes would have left it
    opened: dict                  # what :func:`helix_lm.paint.engine.open_level` gave for this pass
    parent: dict | None = None    # the same for the pass before, when what it writes down is to be carried

    @property
    def cells(self) -> int:
        before = int(self.parent["view"].cy.shape[0]) if self.parent is not None else 0
        return int(self.opened["view"].cy.shape[0]) + before


def _proxy_canvas(plan, level, generator, device) -> torch.Tensor:
    """A canvas as earlier passes would have left it: the plan at half this pass's sharpness, part-way from the
    bare ground."""
    hc, wc = level.size
    target = resize(plan.target.to(device), (hc, wc))
    coarse = resize(resize(target, (max(1, hc // 2), max(1, wc // 2))), (hc, wc))
    done = 0.6 + 0.4 * float(torch.rand((), generator=generator))
    return srgb_to_linear(coarse) * done + plan.ground.to(device)[:, None, None] * (1.0 - done)


def open_pass(plan, generator: torch.Generator, device, index: int | None = None) -> Pass | None:
    """Choose a pass of ``plan`` (later ones more often: they hold most of a painting's strokes) and open it on
    a canvas as the passes before would have left it. One time in two the pass before is opened as well, so
    that what the model writes to its memory there is read here. None when no cell of the pass needs work."""
    levels = plan_levels(*plan.target.shape[-2:])
    if index is None:
        weights = torch.tensor([float(lv.gy * lv.gx) ** 0.5 for lv in levels])
        index = int(torch.multinomial(weights, 1, generator=generator))
    index = min(index, len(levels) - 1)
    canvas = _proxy_canvas(plan, levels[index], generator, device)
    opened = open_level(plan, levels[index], len(levels), canvas, generator)
    if opened is None:
        return None
    parent = None
    if index > 0 and float(torch.rand((), generator=generator)) < 0.5:
        before = levels[index - 1]
        parent = open_level(plan, before, len(levels), _proxy_canvas(plan, before, generator, device), generator)
    return Pass(plan, levels, index, canvas, opened, parent)


def _eye_cells(dense: torch.Tensor, view) -> torch.Tensor:
    grid = torch.stack([(view.cx.float() + 0.5) / view.level.gx * 2 - 1,
                        (view.cy.float() + 0.5) / view.level.gy * 2 - 1], -1)[None, :, None].to(dense.device)
    return F.grid_sample(dense[None].float(), grid, mode="bilinear", padding_mode="border",
                         align_corners=False)[0, :, :, 0].T


def _cells(model: Pinxit, views: list, dense: list | None, memories: list | None) -> dict:
    """The cells of several pictures' views as one batch for the model."""
    c = model.config
    device = next(model.parameters()).device
    parts: dict = {k: [] for k in ("target", "canvas", "brief", "palette", "style", "feeling", "memory",
                                   "has_memory", "eye")}
    for i, view in enumerate(views):
        a, plan = view.cy.shape[0], view.plan
        palette = plan.palette[: c.palette]
        if palette.shape[0] < c.palette:
            palette = torch.cat([palette, palette[-1:].expand(c.palette - palette.shape[0], 3)])
        remembered = None if memories is None else memories[i]
        parts["target"].append(view.target)
        parts["canvas"].append(view.canvas)
        parts["brief"].append(view.brief)
        parts["palette"].append(palette.to(device)[None].expand(a, -1, -1))
        parts["style"].append(plan.card.vector().to(device)[None].expand(a, -1))
        parts["feeling"].append(feeling_vector(plan.feeling).to(device)[None].expand(a, -1))
        parts["memory"].append(view.brief.new_zeros(a, c.memory_dim) if remembered is None else remembered)
        parts["has_memory"].append(torch.full((a,), remembered is not None, device=device))
        if dense is not None:
            parts["eye"].append(_eye_cells(dense[i], view))
    return {k: torch.cat(v) for k, v in parts.items() if v}


_FIELDS = ("geom", "brush", "color", "color2", "streak", "keep", "phase")


def _model(model: Pinxit, cells: dict, grad: torch.Tensor | None) -> dict:
    """The model on a batch of cells. Where ``grad (A,)`` is False the cells go through without gradients: that
    is how a step stays inside its memory whatever the pictures in it."""
    device = next(model.parameters()).device

    def run(rows) -> dict:
        part = cells if rows is None else {k: v[rows] for k, v in cells.items()}
        with _autocast(device):
            return model(part["target"], part["canvas"], part["brief"], part["palette"], part["style"],
                         part["feeling"], part["memory"], part.get("eye"), has_memory=part["has_memory"])

    if grad is None or bool(grad.all()):
        return run(None)
    yes, no = grad.nonzero().flatten(), (~grad).nonzero().flatten()
    with torch.no_grad():
        quiet = run(no)
    if yes.numel() == 0:
        return quiet
    learned = run(yes)
    back = torch.argsort(torch.cat([yes, no]))

    def join(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b.to(a.dtype)])[back]

    strokes = Strokes(*(join(getattr(learned["strokes"], f), getattr(quiet["strokes"], f)) for f in _FIELDS))
    return {"strokes": strokes, **{k: join(learned[k], quiet[k]) for k in ("keep_logit", "brush_logit", "mix",
                                                                           "memory")}}


def picture_batch(model: Pinxit, passes: list, generator: torch.Generator, eye=None,
                  cells: int | None = None) -> dict | None:
    """
    The loss of one step over several pictures' passes (from :func:`open_pass`). The cells of all of them go
    through the model as one batch, the eye sees all the pictures at once, and each picture is then judged on
    its own canvas. ``cells`` is how many cells may carry gradients: when the passes hold more, that many are
    drawn at random and the rest are run without. None when there is nothing to do.
    """
    passes = [p for p in passes if p is not None]
    if not passes:
        return None
    device = next(model.parameters()).device
    c = model.config
    dense = None
    if eye is not None and model.eye_in is not None:
        dense = eye.features_many([p.plan.source.to(device) for p in passes])["dense"]
    before = [i for i, p in enumerate(passes) if p.parent is not None]
    n_before = sum(int(passes[i].parent["view"].cy.shape[0]) for i in before)
    n_now = sum(int(p.opened["view"].cy.shape[0]) for p in passes)
    grad = None
    if cells is not None and n_before + n_now > cells:
        grad = torch.zeros(n_before + n_now, dtype=torch.bool)
        grad[torch.randperm(n_before + n_now, generator=generator)[: max(1, cells)]] = True
        grad = grad.to(device)
    memories: list = [None] * len(passes)
    if before:                                                # what the pass before would have written down
        views = [passes[i].parent["view"] for i in before]
        wrote = _model(model, _cells(model, views, None if dense is None else [dense[i] for i in before], None),
                       None if grad is None else grad[:n_before])["memory"]
        at = 0
        for i in before:
            p = passes[i]
            was, level, a = p.levels[p.index - 1], p.levels[p.index], int(p.parent["view"].cy.shape[0])
            store = wrote.new_zeros(was.gy * was.gx, c.memory_dim).index_put((p.parent["active"],), wrote[at:at + a])
            carried = store.reshape(was.gy, was.gx, -1).repeat_interleave(2, 0).repeat_interleave(2, 1)
            memories[i] = carried.reshape(level.gy * level.gx, -1)[p.opened["active"]]
            at += a
    views = [p.opened["view"] for p in passes]
    out = _model(model, _cells(model, views, dense, memories), None if grad is None else grad[n_before:])
    s = out["strokes"]
    terms: dict = {"picture": [], "habit": [], "exists": []}
    at = 0
    for p, view in zip(passes, views, strict=True):
        a, level = int(view.cy.shape[0]), p.levels[p.index]
        rows, n = slice(at, at + a), min(level.strokes, c.strokes)
        at += a
        allowed = (torch.arange(n, device=device)[None] < view.counts[:, None]).float()
        phase = torch.rand(a, n, PHASES, generator=generator).to(device)
        strokes = Strokes(s.geom[rows, :n], s.brush[rows, :n], s.color[rows, :n], s.color2[rows, :n],
                          s.streak[rows, :n], s.keep[rows, :n] * allowed, phase)
        laid, _, _ = lay(view, strokes, p.canvas)
        miss = (linear_to_oklab(laid.permute(1, 2, 0)) - p.opened["target_lab"]).abs().sum(-1)
        terms["picture"].append((p.opened["weight"] * miss).mean())
        with torch.no_grad():
            rule = StudioHand().propose(view).geom[..., [2, 3, 8, 9]]
        share = allowed / allowed.sum(1, keepdim=True).clamp(min=1.0)
        terms["habit"].append((((s.geom[rows, :n][..., [2, 3, 8, 9]] - rule) * share[..., None]).sum(1) ** 2).mean())
        terms["exists"].append(F.binary_cross_entropy_with_logits(out["keep_logit"][rows, :n], allowed))
    picture, habit, exists = (torch.stack(terms[k]).mean() for k in ("picture", "habit", "exists"))
    return {"total": 10.0 * picture + 0.5 * habit + 0.3 * exists, "picture": picture, "habit": habit,
            "exists": exists, "pictures": float(len(passes)), "cells": float(n_before + n_now)}


def picture_step(model: Pinxit, plan, levels: list, index: int, generator: torch.Generator, eye=None) -> dict | None:
    """The loss for one pass (``levels[index]``) of one picture's plan; None when no cell of it needs work."""
    device = next(model.parameters()).device
    return picture_batch(model, [open_pass(plan, generator, device, index)], generator, eye)


def train_pictures(model: Pinxit, plans, *, eye=None, hours: float | None = None, steps: int | None = None,
                   lr: float = 1e-4, cells: int = 0, pictures: int = 48, pool: int = 256, reuse: int = 4,
                   checkpoint=None, log=print, keep_going=None, save_minutes: float = 20.0, on_save=None,
                   seed: int = 0) -> dict:
    """
    The second stage. ``plans`` yields :class:`helix_lm.paint.mind.Plan` objects (see
    :func:`helix_lm.paint.data.plan_stream`); ``eye`` is the pretrained eye with its adapters. A step takes the
    passes of up to ``pictures`` pictures, until they hold ``cells`` cells; ``cells`` 0 takes as many as fit the
    GPU (256 on a CPU). ``pool`` and ``reuse`` are the :class:`PlanPool`'s.
    """
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed + 17)
    ready = PlanPool(plans, pool, reuse, seed)
    parameters = list(model.parameters())
    if eye is not None:
        parameters += [p for p in eye.parameters() if p.requires_grad]
    size = {"cells": cells}

    def gather(limit: int) -> list:
        passes, held = [], 0
        while held < limit and len(passes) < pictures:
            one = open_pass(ready.draw(), generator, device)
            if one is not None:
                passes.append(one)
                held += one.cells
        return passes

    def step(number: int) -> dict | None:
        if number and number % 500 == 0:
            log(f"[pictures] {ready.note()}")
        return picture_batch(model, gather(size["cells"]), generator, eye, size["cells"])

    def attempt(limit: int) -> int:
        passes = gather(limit)
        picture_batch(model, passes, generator, eye, limit)["total"].backward()
        return min(limit, sum(p.cells for p in passes))

    def shrink() -> str:
        size["cells"] = max(16, int(size["cells"] * 0.85))
        return f"a step now takes {size['cells']} cells"

    with training(model):
        if cells <= 0:
            size["cells"] = 256
            if device.type == "cuda":
                size["cells"] = largest_batch(attempt, 256, 1 << 16, device, log, least=16, what="cells in a step")
                model.zero_grad(set_to_none=True)
                if eye is not None:
                    eye.zero_grad(set_to_none=True)
        return _run(step, model, parameters, hours=hours, steps=steps, lr=lr, warmup=200, checkpoint=checkpoint,
                    stage="pictures", log=log, keep_going=keep_going, save_minutes=save_minutes, on_save=on_save,
                    eye=eye, shrink=shrink)


__all__ = ["Average", "Pass", "PlanPool", "largest_batch", "load_state", "open_pass", "picture_batch", "picture_step",
           "save_state", "train_pictures", "train_sheets", "training"]
