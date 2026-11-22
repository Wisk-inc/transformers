# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Practice sheets: training pictures made of strokes whose every stroke is known.

One sheet is a window with some strokes already on it (the canvas) and the same window with up to ``N`` more
(the target). The model is shown both and has to say which strokes turn the first into the second: where, how
long, which brush, which pigments, how hard and how fast, and how many. The strokes come in a painter's order,
broad ones first. Nothing has to be downloaded for this stage, and it never runs out.
"""

from __future__ import annotations

import math

import torch

from .color import linear_to_oklab, oklab_to_linear
from .engine import BRIEF
from .render import render_windows
from .strokes import random_strokes


def random_palette(batch: int, pigments: int, generator: torch.Generator, device=None) -> torch.Tensor:
    """``(batch, pigments, 3)`` linear RGB: every lightness, weak and strong colours of every hue."""
    r = torch.rand(batch, pigments, 3, generator=generator).to(device)
    light, chroma, hue = 0.12 + 0.85 * r[..., 0], 0.22 * r[..., 1] ** 1.5, 2 * math.pi * r[..., 2]
    lab = torch.stack([light, chroma * torch.cos(hue), chroma * torch.sin(hue)], -1)
    return oklab_to_linear(lab).clamp(0.0, 1.0)


def paint_over(canvas: torch.Tensor, alpha: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """Lay strokes ``alpha (B, N, R, R)``, ``rgb (B, N, 3, R, R)`` one after another on ``canvas (B, 3, R, R)``."""
    for k in range(alpha.shape[1]):
        a = alpha[:, k : k + 1]
        canvas = canvas * (1.0 - a) + rgb[:, k] * a
    return canvas


def _pick(palette: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return torch.gather(palette, 1, index[..., None].expand(*index.shape, 3))


def practice_sheet(batch: int, strokes: int, pigments: int, generator: torch.Generator, device=None,
                   window: int = 32, segments: int = 6) -> dict:
    """``batch`` sheets on ``device``. Returns ``target, canvas (B, 3, R, R)`` linear RGB, ``brief (B, BRIEF)``,
    ``palette (B, P, 3)``, the strokes to find ``truth`` (``keep`` marks the ones that exist), and the pigments
    they use ``pick (B, N, 2)``."""
    def rand(*shape):
        return torch.rand(*shape, generator=generator).to(device)

    def randint(high, shape):
        return torch.randint(0, high, shape, generator=generator).to(device)

    palette = random_palette(batch, pigments, generator, device)
    ground = _pick(palette, randint(pigments, (batch, 1)))[:, 0]
    canvas = ground[:, :, None, None].expand(batch, 3, window, window)
    under = random_strokes((batch, strokes), generator, device)
    pick_under = randint(pigments, (batch, strokes, 2))
    under.color, under.color2 = _pick(palette, pick_under[..., 0]), _pick(palette, pick_under[..., 1])
    under.keep = (torch.arange(strokes, device=device)[None] < randint(strokes + 1, (batch, 1))).float()
    with torch.no_grad():
        canvas = paint_over(canvas, *render_windows(under, window, segments))
    truth = random_strokes((batch, strokes), generator, device)
    order = truth.geom[..., 3].argsort(dim=1, descending=True)                   # broad strokes first
    truth.geom = torch.gather(truth.geom, 1, order[..., None].expand_as(truth.geom))
    truth.brush = torch.gather(truth.brush, 1, order[..., None].expand_as(truth.brush))
    pick = randint(pigments, (batch, strokes, 2))
    truth.color, truth.color2 = _pick(palette, pick[..., 0]), _pick(palette, pick[..., 1])
    count = (rand(batch) ** 0.7 * (strokes + 1)).long().clamp(max=strokes)
    truth.keep = (torch.arange(strokes, device=device)[None] < count[:, None]).float()
    with torch.no_grad():
        target = paint_over(canvas, *render_windows(truth, window, segments))
    # the brief the mind would have given: how many strokes, which way they run, how fast and how hard
    keep = truth.keep
    total = keep.sum(1).clamp(min=1.0)
    angle = 2 * math.pi * truth.geom[..., 4]
    c2, s2 = (torch.cos(2 * angle) * keep).sum(1) / total, (torch.sin(2 * angle) * keep).sum(1) / total
    together = torch.sqrt(c2 ** 2 + s2 ** 2)
    brief = torch.stack([rand(batch), rand(batch), rand(batch), count.float() / strokes,
                         c2 / together.clamp(min=1e-6), s2 / together.clamp(min=1e-6), together,
                         (truth.geom[..., 9] * keep).sum(1) / total, (truth.geom[..., 8] * keep).sum(1) / total,
                         rand(batch), together, linear_to_oklab(target.mean((-1, -2)))[:, 0]], -1)
    assert brief.shape[-1] == BRIEF
    return {"target": target, "canvas": canvas, "brief": brief, "palette": palette, "truth": truth, "pick": pick}


def sheet_loss(out: dict, sheet: dict, window: int = 32, segments: int = 6) -> dict:
    """How wrong the model's answer to a practice sheet is: the picture its strokes make against the target, and
    each stroke against the one it should have been (its gesture, its brush, its pigments, whether it exists)."""
    pred, truth = out["strokes"], sheet["truth"]
    mask = truth.keep
    n = mask.sum().clamp(min=1.0)
    made = paint_over(sheet["canvas"], *render_windows(pred, window, segments))
    picture = (linear_to_oklab(made.permute(0, 2, 3, 1)) - linear_to_oklab(sheet["target"].permute(0, 2, 3, 1))
               ).abs().sum(-1).mean()
    diff = (pred.geom - truth.geom).abs()
    turn = 0.5 * (1.0 - torch.cos(2 * math.pi * (pred.geom[..., 4] - truth.geom[..., 4])))    # an angle wraps round
    gesture = ((diff.sum(-1) - diff[..., 4] + 2.0 * turn) * mask).sum() / n
    lp = torch.log_softmax(out["brush_logit"], -1)
    brush = -((lp * truth.brush).sum(-1) * mask).sum() / n
    lm = torch.log_softmax(out["mix"], -1)                                                     # (B, N, 2, P)
    pigment = -(torch.gather(lm, -1, sheet["pick"][..., None])[..., 0].sum(-1) * mask).sum() / n
    streak = ((pred.streak - truth.streak).abs() * mask).sum() / n
    exists = torch.nn.functional.binary_cross_entropy_with_logits(out["keep_logit"], mask)
    total = 10.0 * picture + 2.0 * gesture + 0.5 * brush + 0.5 * pigment + 0.5 * streak + exists
    return {"total": total, "picture": picture, "gesture": gesture, "brush": brush, "pigment": pigment,
            "exists": exists}


__all__ = ["paint_over", "practice_sheet", "random_palette", "sheet_loss"]
