# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Painting something in from words. Experimental.

:meth:`helix_lm.paint.painting.Painting.add` with ``image=`` needs no model: the picture is fitted into the
painting and painted in its palette. With ``text=`` there is nothing to copy, so the strokes are found by asking
an eye that reads text (SigLIP 2 or CLIP): strokes in the painting's palette are laid in the box and moved,
turned and recoloured until the eye says the box looks like the words (the idea of CLIPDraw, Frans et al. 2021,
with this renderer and brushes). What comes out is loose and suggestive, in keeping with the rest; it is only as
good as the eye's sense of the words, and with the untrained test eye it means nothing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .color import linear_to_srgb, snap_to_palette, srgb_to_linear
from .engine import StudioHand, gradients, lay, open_level
from .io import resize
from .render import render_painting
from .strokes import Strokes, to_absolute

NOT_THIS = "a blurry mess of random brush strokes"


def add_text(painting, text: str, box=(0.3, 0.3, 0.7, 0.7), eye=None, steps: int = 120, views: int = 6,
             log=None):
    """Paint ``text`` into ``box = (left, top, right, bottom)`` of ``painting``. Returns the painting.

    The strokes are small enough for the box to hold about six cells across, so the eye is shown something it
    can read. Each step the box is seen ``views`` ways (shifted and scaled a little) and the strokes move
    towards what the eye calls ``text`` and away from what it calls a mess; a stroke that does not help costs a
    little for being there, so it fades instead of staying as scatter."""
    if eye is None or not getattr(eye, "has_text", False):
        raise ValueError("painting from words needs an eye that reads text, for example "
                         "load_eye('google/siglip2-base-patch16-256'); or give image= to paint a picture in")
    plan, levels = painting.plan, painting._levels()
    h, w = plan.target.shape[-2:]
    mask = painting._box_mask(box, feather=0.1)
    extent = max(float(mask.sum(0).gt(0.5).sum()), float(mask.sum(1).gt(0.5).sum()))
    level = min(levels, key=lambda lv: abs(h / lv.gy - extent / 6.0))         # cells about a sixth of the box
    hc, wc = level.size
    generator = torch.Generator().manual_seed(plan.settings.seed + 7 + len(painting.history))
    with torch.no_grad():
        canvas, _ = render_painting(painting.rows, wc, hc, plan.ground, relief=0.0, tremor=False)
    with painting._change(mask):          # undone exactly by Painting.undo; put back if this fails
        plan.finish, plan.focus = torch.maximum(plan.finish, mask), torch.maximum(plan.focus, 0.8 * mask)
        # the plan does not hold the thing yet, so the canvas there looks finished: every cell is worked anyway
        opened = open_level(plan, level, len(levels), canvas, generator, region=mask, everything=True)
        if opened is None:
            raise ValueError("the box is too small to paint in")
        view = opened["view"]
        start = StudioHand().propose(view)
        pick = torch.randint(0, plan.palette.shape[0], (2, *start.keep.shape), generator=generator)
        palette = plan.palette

        def logit(v):
            return torch.logit(v.clamp(1e-4, 1 - 1e-4))

        g = logit(start.geom).requires_grad_(True)
        c = logit(linear_to_srgb(palette[pick[0]])).requires_grad_(True)
        c2 = logit(linear_to_srgb(palette[pick[1]])).requires_grad_(True)
        keep = torch.ones_like(start.keep)

        def build() -> Strokes:
            return Strokes(torch.sigmoid(g), start.brush,
                           snap_to_palette(srgb_to_linear(torch.sigmoid(c)), palette, plan.settings.palette_strength),
                           snap_to_palette(srgb_to_linear(torch.sigmoid(c2)), palette, plan.settings.palette_strength),
                           start.streak, keep, start.phase)

        words, mess = eye.text([text, NOT_THIS])
        x0, y0, x1, y1 = box
        optimizer = torch.optim.Adam([g, c, c2], lr=0.05)
        first = last = None
        with gradients():
            for _ in range(steps):
                strokes = build()
                out, _, _ = lay(view, strokes, canvas)
                picture = linear_to_srgb(out)
                crops = []
                for _ in range(views):                                # the box seen a few slightly different ways
                    jx, jy = (torch.rand(2, generator=generator) - 0.5) * 0.24
                    scale = 0.95 + 0.4 * float(torch.rand((), generator=generator))
                    cx, cy, bw, bh = (x0 + x1) / 2 + float(jx) * (x1 - x0), (y0 + y1) / 2 + float(jy) * (y1 - y0), \
                        (x1 - x0) * scale, (y1 - y0) * scale
                    ys = torch.linspace(cy - bh / 2, cy + bh / 2, 128) * 2 - 1
                    xs = torch.linspace(cx - bw / 2, cx + bw / 2, 128) * 2 - 1
                    grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij")[::-1], -1)[None]
                    crops.append(F.grid_sample(picture[None], grid, mode="bilinear", padding_mode="border",
                                               align_corners=False)[0])
                crops = torch.stack(crops)
                score = eye.score(crops, words).mean()
                there = (strokes.geom[..., 7] * strokes.keep).mean()          # how much paint the strokes lay
                loss = -score + 0.3 * eye.score(crops, mess).mean() + 0.03 * there
                first = float(score.detach()) if first is None else first
                last = float(score.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            strokes = build().detach()
            out, _, _ = lay(view, strokes, canvas)
            new = to_absolute(strokes, view.cy, view.cx, hc // level.gy, wc, level.index)
            painted = resize(linear_to_srgb(out), (h, w))
            plan.target = plan.target * (1 - mask) + painted * mask       # the plan now holds what was added
        before = len(painting)
        painting.rows = torch.cat([painting.rows, new])
        painting.history.append({"op": "add", "text": text, "box": list(box), "from": before, "strokes": len(new),
                                 "eye_score": [round(first, 4), round(last, 4)]})
    if log:
        log(f"added '{text}': {len(new)} strokes; the eye's score for the words went from {first:.3f} to {last:.3f}")
    return painting


__all__ = ["add_text"]
