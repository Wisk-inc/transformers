# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Painting: from a plan to strokes, broad to fine.

The canvas is covered by a grid of cells; each pass halves the cells. In a pass every cell that still needs paint
gets a few strokes, planned in the cell's window (twice the cell, so strokes cross into the neighbours), by a
**hand**:

* :class:`StudioHand` needs no trained weights. It puts strokes where the canvas differs most from the plan,
  runs them along the edges (or along the plan's flow where there are none), takes their colours from the plan,
  and gives each a pressure and a speed: heavy in the darks and the first pass, fast along clear directions,
  slow and careful at the subject.
* :class:`helix_lm.paint.model.ModelHand` asks the trained Pinxit model for the strokes.

The strokes of a pass are then refined together by gradient descent through the renderer, so that the canvas
comes closer to the plan; the trained model needs few such steps or none.

**Restraint** is built into the loop, not added after it. A cell is worked in a pass only if the plan's
``finish`` allows it: the first two passes cover the whole canvas, later ones reach less and less of it, and the
last reaches only the subject. A part the plan leaves unfinished but that is still far from right gets one fast,
long stroke that suggests it. A stroke that changes almost nothing is not made, and the painting has a budget
of strokes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .brushes import BRUSHES, one_hot
from .color import delta_e, linear_to_oklab, linear_to_srgb, snap_to_palette, srgb_to_linear
from .io import resize
from .mind import Plan
from .render import composite_level, cut_windows, render_painting, render_windows
from .see import orientation_field
from .strokes import ABS, HALF_LENGTH, PHASES, Strokes, inverse_geometry, to_absolute

CELL = 16
WINDOW = 2 * CELL
COUNTS = (14, 12, 10, 8, 6, 5, 4)             # strokes a cell may get, from the first pass to the last
STEPS = (48, 40, 30, 24, 20, 16, 12)          # refinement steps
SEGMENTS = (10, 8, 6, 6, 4, 4, 4)             # pieces a stroke's curve is cut into
BRIEF = 12                                    # numbers the model is told about a cell


@dataclass
class Level:
    index: int
    gy: int
    gx: int
    strokes: int
    segments: int
    steps: int

    @property
    def size(self) -> tuple[int, int]:
        return self.gy * CELL, self.gx * CELL


def plan_levels(height: int, width: int, detail: int | None = None, steps: float = 1.0) -> list[Level]:
    """The passes for a canvas of ``height x width`` pixels (multiples of the cell): from the coarsest grid that
    still has whole cells to the grid of single cells."""
    gy, gx = height // CELL, width // CELL
    grids = [(gy, gx)]
    while gy % 2 == 0 and gx % 2 == 0:
        gy, gx = gy // 2, gx // 2
        grids.append((gy, gx))
    grids = grids[::-1][: detail or len(grids)]
    last = len(COUNTS) - 1
    return [Level(i, a, b, COUNTS[min(i, last)], SEGMENTS[min(i, last)], round(STEPS[min(i, last)] * steps))
            for i, (a, b) in enumerate(grids)]


@dataclass
class View:
    """What a hand is shown for one pass. ``A`` is the number of cells being worked."""

    level: Level
    levels: int
    plan: Plan
    target: torch.Tensor          # (A, 3, R, R) linear RGB: the plan in each cell's window
    canvas: torch.Tensor          # (A, 3, R, R) linear RGB: the canvas there now
    error: torch.Tensor           # (A, R, R): how far the canvas is from the plan
    cy: torch.Tensor              # (A,) the cells' rows
    cx: torch.Tensor              # (A,) and columns
    counts: torch.Tensor          # (A,) strokes each cell is to get
    gesture: torch.Tensor         # (A,) bool: an unfinished part to be suggested with one stroke
    maps: torch.Tensor            # (9, H, W): direction (cos2, sin2, coherence), flow (cos2, sin2, weight), focus,
                                  # finish, lightness of the plan, at this pass's canvas size
    brief: torch.Tensor           # (A, BRIEF): the same in a few numbers per cell, for the model
    generator: torch.Generator
    memory: torch.Tensor | None = None      # (A, C): what the model remembers of these places


def sample_maps(maps: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``maps (C, H, W)`` read at pixel positions ``x, y (M,)`` -> ``(M, C)``."""
    h, w = maps.shape[-2:]
    grid = torch.stack([x / w * 2 - 1, y / h * 2 - 1], -1)[None, :, None]
    return F.grid_sample(maps[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, :, 0].T


class StudioHand:
    """The hand that needs no training (see the module's notes)."""

    name = "studio"

    def propose(self, view: View) -> Strokes:
        a, n, r = view.cy.shape[0], view.level.strokes, WINDOW
        gen, f, card = view.generator, view.plan.feeling, view.plan.card
        dev = view.target.device

        def rand(*shape):
            return torch.rand(*shape, generator=gen).to(dev)

        def randn(*shape):
            return torch.randn(*shape, generator=gen).to(dev)

        lo, hi = round(0.3 * r), round(0.7 * r)
        side = hi - lo
        chance = (view.error[:, lo:hi, lo:hi].reshape(a, -1) + 1e-3) ** 1.5
        pick = torch.multinomial(chance.cpu(), n, replacement=True, generator=gen).to(dev)
        cxw = ((lo + pick % side).float() + rand(a, n)) / r
        cyw = ((lo + pick // side).float() + rand(a, n)) / r
        x = view.cx[:, None].float() * CELL - CELL / 2 + cxw * r
        y = view.cy[:, None].float() * CELL - CELL / 2 + cyw * r
        cos2, sin2, coh, fcos, fsin, fw, focus, _, light = sample_maps(
            view.maps, x.flatten(), y.flatten()).reshape(a, n, 9).unbind(-1)
        frac = view.level.index / max(1, view.levels - 1)
        coh = coh.clamp(0.0, 1.0)
        dx, dy = coh * cos2 + (1 - coh) * fw * fcos, coh * sin2 + (1 - coh) * fw * fsin
        theta = torch.where(dx.abs() + dy.abs() < 0.05, rand(a, n) * math.pi, 0.5 * torch.atan2(dy, dx))
        theta = theta + 0.5 * (f.jitter + card.jitter) * randn(a, n) * (1.2 - coh)
        theta = theta + math.pi * (rand(a, n) < 0.5)
        # the gesture: fast along clear directions, slow at the subject; heavy in the darks and the first pass
        speed = (f.speed + 0.35 * (coh - 0.5) - 0.45 * focus * frac + 0.12 * randn(a, n)).clamp(0.03, 0.97)
        pressure = (f.pressure + 0.3 * (0.5 - light) + 0.15 * (1 - frac) + 0.10 * randn(a, n)).clamp(0.03, 0.97)
        half = 0.125 * f.length * card.length * (0.75 + 0.5 * speed) * (0.6 + 0.8 * coh)
        radius = 0.05 * f.width * card.width * (1.15 - 0.6 * coh) * (1.0 - 0.35 * focus * frac)
        half, radius = half * torch.exp(0.3 * randn(a, n)), radius * torch.exp(0.3 * randn(a, n))   # no two alike
        curl = min(1.0, 0.8 * (f.bend + card.bend))
        bend = 0.5 + (rand(a, n) - 0.5) * 2 * curl * (1 - 0.6 * coh) * (1.1 - speed)
        weights = torch.tensor(card.brushes, device=dev) * torch.tensor(f.brushes, device=dev)
        coarse = torch.tensor((1.5, 0.6, 1.0, 0.6, 1.0, 1.6), device=dev)
        fine = torch.tensor((0.8, 1.6, 1.3, 0.9, 0.7, 0.3), device=dev)
        weights = (weights * (coarse * (1 - frac) + fine * frac)).clamp(min=1e-4)[None, None].repeat(a, n, 1)
        weights[..., 3] *= 0.5 + speed                       # a fast stroke is more often a dry one
        weights[..., 4] *= 0.5 + pressure                    # a heavy one more often the knife
        brush = torch.multinomial(weights.reshape(-1, len(BRUSHES)).cpu(), 1, generator=gen).to(dev).reshape(a, n)
        order = radius.argsort(dim=1, descending=True)       # broad strokes first, then narrow ones

        def by(v):
            return torch.gather(v, 1, order)

        cxw, cyw, half, radius, theta, bend, speed, pressure, brush = (
            by(v) for v in (cxw, cyw, half, radius, theta, bend, speed, pressure, brush))
        if bool(view.gesture.any()):                         # an unfinished part: one long, fast, dry stroke
            g = view.gesture
            half[g, 0], speed[g, 0], pressure[g, 0], brush[g, 0] = HALF_LENGTH[1], 0.92, 0.45, 3
        geom = inverse_geometry(cxw, cyw, half, radius, theta, bend, 0.5 + 0.45 * rand(a, n),
                                card.opacity * (0.85 + 0.15 * pressure), pressure, speed).to(dev)
        normal = torch.stack([-torch.sin(theta), torch.cos(theta)], -1)
        centre = torch.stack([cxw, cyw], -1)
        points = torch.stack([centre, centre + 0.9 * radius[..., None] * normal], 2) * 2 - 1       # (A, N, 2, 2)
        seen = F.grid_sample(view.target, points, mode="bilinear", padding_mode="border", align_corners=False)
        keep = (torch.arange(n, device=dev)[None] < view.counts[:, None]).float()
        return Strokes(geom, one_hot(brush), seen[..., 0].permute(0, 2, 1), seen[..., 1].permute(0, 2, 1),
                       (0.15 + 0.5 * card.texture) * rand(a, n), keep, rand(a, n, PHASES))


def lay(view: View, strokes: Strokes, canvas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The pass's strokes laid on ``canvas (3, H, W)``: ``(new canvas, alpha (A, N, R, R), rgb (A, N, 3, R, R))``."""
    alpha, rgb = render_windows(strokes, WINDOW, view.level.segments)
    n = alpha.shape[1]
    # only the part of the canvas the worked cells can reach is laid: their rows and columns and one more on
    # each side, starting on an even row and column so the order windows are laid in stays the same
    y0 = max(0, int(view.cy.min()) - 1) // 2 * 2
    x0 = max(0, int(view.cx.min()) - 1) // 2 * 2
    y1, x1 = min(view.level.gy, int(view.cy.max()) + 2), min(view.level.gx, int(view.cx.max()) + 2)
    gy, gx = y1 - y0, x1 - x0
    where = (view.cy - y0) * gx + (view.cx - x0)
    full_a = alpha.new_zeros(gy * gx, n, WINDOW, WINDOW).index_put((where,), alpha)
    full_c = rgb.new_zeros(gy * gx, n, 3, WINDOW, WINDOW).index_put((where,), rgb)
    part = (slice(None), slice(y0 * CELL, y1 * CELL), slice(x0 * CELL, x1 * CELL))
    laid = composite_level(canvas[part][None], full_a.reshape(1, gy, gx, n, WINDOW, WINDOW),
                           full_c.reshape(1, gy, gx, n, 3, WINDOW, WINDOW))[0]
    if gy == view.level.gy and gx == view.level.gx:
        return laid, alpha, rgb
    out = F.pad(laid, (x0 * CELL, canvas.shape[-1] - x1 * CELL, y0 * CELL, canvas.shape[-2] - y1 * CELL))
    inside = torch.zeros_like(canvas[:1])
    inside[part] = 1.0
    return out + canvas * (1.0 - inside), alpha, rgb


def gradients():
    """What the fitting loops run inside. Fitting a stroke needs gradients, so they are switched on here even
    when the caller painted under ``torch.no_grad()``. ``torch.inference_mode()`` cannot be undone from inside
    it, so that is said plainly instead of failing deep in a backward pass."""
    if torch.is_inference_mode_enabled():
        raise RuntimeError("the painter fits its strokes with gradients: call it outside torch.inference_mode() "
                           "(torch.no_grad() is fine)")
    return torch.enable_grad()


def refine(strokes: Strokes, view: View, canvas: torch.Tensor, target_lab: torch.Tensor, weight: torch.Tensor,
           steps: int, palette_strength: float, lr: float = 0.04) -> Strokes:
    """Move, turn, resize and recolour the pass's strokes together so the canvas comes closer to the plan
    (weighted by how much each part matters). Their length, width, pressure and speed are held near what the
    hand gave them: the fit must not flatten the gesture."""
    palette = view.plan.palette.to(canvas.device)

    def logit(v):
        return torch.logit(v.clamp(1e-4, 1 - 1e-4))

    strokes = strokes.detach()
    g = logit(strokes.geom).requires_grad_(steps > 0)
    c = logit(linear_to_srgb(strokes.color)).requires_grad_(steps > 0)
    c2 = logit(linear_to_srgb(strokes.color2)).requires_grad_(steps > 0)
    s = logit(strokes.streak).requires_grad_(steps > 0)
    first = g.detach().clone()

    def build() -> Strokes:
        return Strokes(torch.sigmoid(g), strokes.brush,
                       snap_to_palette(srgb_to_linear(torch.sigmoid(c)), palette, palette_strength),
                       snap_to_palette(srgb_to_linear(torch.sigmoid(c2)), palette, palette_strength),
                       torch.sigmoid(s), strokes.keep, strokes.phase)

    if steps > 0:
        optimizer = torch.optim.Adam([g, c, c2, s], lr=lr)
        held = torch.tensor([2, 3, 8, 9], device=g.device)
        with gradients():
            for _ in range(steps):
                out, _, _ = lay(view, build(), canvas)
                miss = (linear_to_oklab(out.permute(1, 2, 0)) - target_lab).abs().sum(-1)
                loss = (weight * miss).mean() + 0.02 * ((g - first)[..., held] ** 2
                                                        * strokes.keep[..., None]).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    with torch.no_grad():
        return build().detach()


def open_level(plan: Plan, level: Level, levels: int, canvas: torch.Tensor, generator: torch.Generator,
               region: torch.Tensor | None = None, memory: torch.Tensor | None = None,
               everything: bool = False) -> dict | None:
    """Look at the canvas ``(3, H, W)`` (linear RGB, this pass's size) against the plan and decide which cells
    are worked and how many strokes each may get. Returns the :class:`View` for the hand with the pass's
    ``weight`` map, the plan in OKLab (``target_lab``) and ``notes``, or None when no cell needs work.
    ``everything`` works every cell (of the ``region``) with a full count of strokes, however close the canvas
    already is to the plan: for painting in something the plan does not hold yet."""
    hc, wc = level.size
    dev = canvas.device
    target = resize(plan.target.to(dev), (hc, wc))
    target_lin = srgb_to_linear(target)
    focus, finish = resize(plan.focus.to(dev)[None], (hc, wc))[0], resize(plan.finish.to(dev)[None], (hc, wc))[0]
    flow = resize(plan.flow.to(dev), (hc, wc))
    field = orientation_field(target, 1.0, 2.0)
    target_lab = linear_to_oklab(target_lin.permute(1, 2, 0))
    maps = torch.stack([field["cos2"], field["sin2"], field["coherence"], flow[0], flow[1], flow[2], focus, finish,
                        target_lab[..., 0]])
    error = delta_e(linear_to_srgb(canvas), target)
    block_in = level.index <= 1
    last = level.index >= levels - (2 if levels >= 6 else 1)        # the finest strokes: only for the subject
    weight = torch.ones_like(finish) if block_in else 0.25 + 0.75 * finish
    if region is not None:
        mask = resize(region.to(dev)[None], (hc, wc))[0]
        weight = weight * mask
    restraint = plan.restraint
    # how finished a cell must be to be worked in this pass: anything in the first two, nearly anything in the
    # middle ones (with fewer strokes where less is finished), only the subject in the last
    reach = 0.0 if block_in else (0.45 * restraint + 0.05 if last else 0.12 * restraint)

    def per_cell(m):
        return F.avg_pool2d(m[None, None], CELL)[0, 0]

    cell_error, cell_finish, cell_focus = per_cell(error * (0.5 + focus)), per_cell(finish), per_cell(focus)
    need = ((cell_error - 0.018) / (0.085 - 0.018)).clamp(0.0, 1.0)
    allowed = (cell_finish >= reach).float()
    if block_in or everything:                                # the whole canvas is covered
        counts = torch.full_like(cell_finish, level.strokes).long()
    else:
        counts = torch.ceil(level.strokes * need * allowed * (0.35 + 0.65 * cell_finish)).long()
    gesture = (allowed == 0) & (cell_error > 0.11) if not (block_in or everything) else \
        torch.zeros_like(allowed, dtype=torch.bool)
    counts = torch.where(gesture, torch.ones_like(counts), counts)
    if region is not None:
        inside = per_cell(mask) > 0.05
        counts, gesture = counts * inside, gesture & inside
    counts = counts.clamp(0, level.strokes)
    active = counts.flatten().nonzero().flatten()
    notes = {"cells": int(active.numel()), "of": level.gy * level.gx, "proposed": int(counts.sum()), "kept": 0}
    if active.numel() == 0:
        return None
    cy, cx = active // level.gx, active % level.gx
    flow_c = torch.stack([per_cell(m) for m in (flow[0], flow[1], flow[2], field["coherence"], target_lab[..., 0])])
    f = plan.feeling
    brief = torch.stack([torch.full_like(cell_focus, level.index / max(1, levels - 1)), cell_focus, cell_finish,
                         counts.float() / level.strokes, flow_c[0], flow_c[1], flow_c[2],
                         torch.full_like(cell_focus, f.speed), torch.full_like(cell_focus, f.pressure),
                         torch.full_like(cell_focus, restraint), flow_c[3], flow_c[4]], -1).reshape(-1, BRIEF)
    view = View(level, levels, plan, cut_windows(target_lin[None], CELL)[0].reshape(-1, 3, WINDOW, WINDOW)[active],
                cut_windows(canvas[None], CELL)[0].reshape(-1, 3, WINDOW, WINDOW)[active],
                cut_windows(error[None, None], CELL)[0].reshape(-1, WINDOW, WINDOW)[active], cy, cx,
                counts.flatten()[active], gesture.flatten()[active], maps, brief[active], generator,
                None if memory is None else memory[active])
    return {"view": view, "weight": weight, "target_lab": target_lab, "notes": notes, "active": active,
            "block_in": block_in}


def paint_level(plan: Plan, level: Level, levels: int, rows: torch.Tensor, hand, generator: torch.Generator,
                region: torch.Tensor | None = None, memory: torch.Tensor | None = None, budget: int | None = None,
                log=None, canvas: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """One pass. ``rows`` are the painting's strokes so far; returns the new strokes ``(K, ABS)`` and notes
    (cells worked, strokes proposed and kept, what the model remembers). ``region (H, W)`` limits the pass to a
    part of the canvas (for changes to a finished painting). ``canvas`` is the painting so far as a picture
    (linear RGB, any size), when the caller keeps one; without it the strokes are rendered here."""
    hc, wc = level.size
    dev = rows.device
    with torch.no_grad():
        if canvas is not None:
            canvas = resize(canvas.to(dev), (hc, wc))
        elif rows.shape[0]:
            canvas, _ = render_painting(rows, wc, hc, plan.ground.to(dev), segments=level.segments + 2, relief=0.0,
                                        tremor=False)
        else:
            canvas = plan.ground.to(dev)[:, None, None].expand(3, hc, wc).clone()
    opened = open_level(plan, level, levels, canvas, generator, region, memory)
    if opened is None:
        return rows.new_zeros(0, ABS), {"cells": 0, "of": level.gy * level.gx, "proposed": 0, "kept": 0}
    view, weight, target_lab, notes = opened["view"], opened["weight"], opened["target_lab"], opened["notes"]
    active, block_in, cy, cx = opened["active"], opened["block_in"], view.cy, view.cx
    proposal = hand.propose(view)
    remembered = getattr(hand, "remembered", None)
    steps = level.steps if getattr(hand, "refines", True) else round(level.steps * getattr(hand, "steps_share", 0.0))
    strokes = refine(proposal, view, canvas, target_lab, weight, steps, plan.settings.palette_strength)
    with torch.no_grad():                                     # a stroke that changes almost nothing is not made
        _, alpha, rgb = lay(view, strokes, canvas)
        # judged as the eye sees it (square roots: a small step in the darks counts as much as a large one in
        # the lights); the first two passes keep every stroke, because their job is to cover the canvas
        seen = (rgb.clamp(min=0).sqrt() - view.canvas[:, None].clamp(min=0).sqrt()).abs().sum(2)
        change = (alpha * seen).mean((-1, -2))
        keep = strokes.keep if block_in else strokes.keep * (change > 1.2e-3)
        if budget is not None and int(keep.sum()) > budget:
            cut = torch.sort((change * keep).flatten(), descending=True).values[max(budget - 1, 0)]
            keep = keep * (change >= cut)
        strokes.keep = keep
    new = to_absolute(strokes, cy, cx, CELL, wc, level.index)
    notes["kept"] = int(new.shape[0])
    if remembered is not None:
        notes["memory"], notes["cells_index"] = remembered, active
    if log:
        log(f"pass {level.index + 1} of {levels} ({level.gy}x{level.gx} cells): worked {notes['cells']} of "
            f"{notes['of']} cells, {notes['kept']} strokes")
    return new, notes


__all__ = ["BRIEF", "CELL", "COUNTS", "Level", "StudioHand", "View", "WINDOW", "lay", "open_level", "paint_level",
           "plan_levels", "refine", "sample_maps"]
