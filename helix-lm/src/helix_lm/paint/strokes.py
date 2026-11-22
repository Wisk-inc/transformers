# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
What a stroke is.

In a window (the square piece of canvas a group of strokes is planned in) a stroke is ten numbers between 0 and
1: where its middle is, how long and how wide it is, which way it goes, how much it bends, how much thinner it
gets toward its end, how opaque the paint is, and the two numbers that make it a gesture and not a shape: how
hard the brush was pressed and how fast it moved. A pressed stroke is wider, denser and thicker, with paint pooled
where the brush landed. A fast stroke runs dry, breaks into bristle streaks and ends in a flick; a slow one is
heavy and a little unsteady. With them go the brush weights, two colours (a brush can carry two, which show as
streaks), how strongly the second colour shows, whether the stroke is made at all, and five random phases that
place its bristle marks and its tremor.

A stroke always stays inside its window: its middle lies in the central 40%, and its length and width are limited
so that no end can cross the edge. Windows overlap by half, so strokes still cross from one cell into the next.

On a finished painting a stroke is one row of ``ABS`` numbers in units of the canvas width, so a painting can be
rendered at any size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .brushes import BRUSHES, FIELDS, brush_profile

GEOM = 10                                  # cx, cy, length, width, angle, bend, taper, opacity, pressure, speed
PHASES = 5
CENTER = (0.30, 0.40)                      # the middle lies in [0.30, 0.70] of the window
HALF_LENGTH = (0.02, 0.19)                 # half the length, in window units
RADIUS = (0.012, 0.073)                    # half the width
BEND = 0.6                                 # the control point moves at most this share of the half length sideways
END_WIDTH = (0.35, 0.65)                   # the end is 35% to 100% as wide as the start
OPACITY = (0.05, 0.95)
ABS = 6 + 2 + 1 + len(FIELDS) + 3 + 3 + 1 + PHASES + 2 + 2      # 33 numbers a stroke on a painting
# columns of a painting's rows
A_PTS, A_R0, A_R1, A_ALPHA, A_PROFILE = slice(0, 6), 6, 7, 8, slice(9, 9 + len(FIELDS))
A_COLOR, A_COLOR2, A_STREAK, A_PHASE = slice(17, 20), slice(20, 23), 23, slice(24, 29)
A_PRESSURE, A_SPEED, A_LEVEL, A_BRUSH = 29, 30, 31, 32


@dataclass
class Strokes:
    """Strokes planned in windows. Every tensor starts with the same leading shape ``(..., N)``."""

    geom: torch.Tensor        # (..., N, 10) in 0..1
    brush: torch.Tensor       # (..., N, brushes) weights summing to 1
    color: torch.Tensor       # (..., N, 3) linear RGB
    color2: torch.Tensor      # (..., N, 3) linear RGB, the second colour on the brush
    streak: torch.Tensor      # (..., N) 0..1 how much of the second colour shows
    keep: torch.Tensor        # (..., N) 0..1 whether the stroke is made
    phase: torch.Tensor       # (..., N, 5) in 0..1, places the bristle marks and the tremor

    def flat(self) -> Strokes:
        n = math.prod(self.geom.shape[:-1])
        return Strokes(self.geom.reshape(n, GEOM), self.brush.reshape(n, -1), self.color.reshape(n, 3),
                       self.color2.reshape(n, 3), self.streak.reshape(n), self.keep.reshape(n),
                       self.phase.reshape(n, PHASES))

    def index(self, idx) -> Strokes:
        return Strokes(self.geom[idx], self.brush[idx], self.color[idx], self.color2[idx], self.streak[idx],
                       self.keep[idx], self.phase[idx])

    def detach(self) -> Strokes:
        return Strokes(*(getattr(self, f).detach() for f in ("geom", "brush", "color", "color2", "streak", "keep",
                                                             "phase")))


def geometry(geom: torch.Tensor) -> dict:
    """``(..., 10)`` in 0..1 -> the curve in window units: ``p0, p1, p2 (..., 2)``, half widths ``r0, r1``,
    ``opacity``, ``pressure`` and ``speed``. The curve is a quadratic Bezier from ``p0`` through the pull of
    ``p1`` to ``p2``."""
    cx = CENTER[0] + CENTER[1] * geom[..., 0]
    cy = CENTER[0] + CENTER[1] * geom[..., 1]
    half = HALF_LENGTH[0] + (HALF_LENGTH[1] - HALF_LENGTH[0]) * geom[..., 2]
    radius = RADIUS[0] + (RADIUS[1] - RADIUS[0]) * geom[..., 3]
    angle = 2.0 * math.pi * geom[..., 4]
    dx, dy = torch.cos(angle), torch.sin(angle)
    side = (2.0 * geom[..., 5] - 1.0) * BEND * half
    c = torch.stack([cx, cy], -1)
    d = torch.stack([dx, dy], -1)
    n = torch.stack([-dy, dx], -1)
    return {"p0": c - half[..., None] * d, "p1": c + side[..., None] * n, "p2": c + half[..., None] * d,
            "r0": radius, "r1": radius * (END_WIDTH[0] + END_WIDTH[1] * geom[..., 6]),
            "opacity": OPACITY[0] + OPACITY[1] * geom[..., 7], "pressure": geom[..., 8], "speed": geom[..., 9]}


def inverse_geometry(cx, cy, half, radius, angle, bend=0.5, taper=0.7, opacity=0.9, pressure=0.5,
                     speed=0.5) -> torch.Tensor:
    """The ten numbers for a stroke described in window units (the reverse of :func:`geometry`), clamped to
    what a stroke may be. ``angle`` in radians; ``bend``, ``taper``, ``pressure`` and ``speed`` already in 0..1."""
    def unit(v, lo, span):
        return ((torch.as_tensor(v, dtype=torch.float32) - lo) / span).clamp(0.0, 1.0)

    cx = torch.as_tensor(cx, dtype=torch.float32)
    parts = [unit(cx, CENTER[0], CENTER[1]), unit(cy, CENTER[0], CENTER[1]),
             unit(half, HALF_LENGTH[0], HALF_LENGTH[1] - HALF_LENGTH[0]),
             unit(radius, RADIUS[0], RADIUS[1] - RADIUS[0]),
             torch.remainder(torch.as_tensor(angle, dtype=torch.float32), 2.0 * math.pi) / (2.0 * math.pi),
             torch.as_tensor(bend, dtype=torch.float32).clamp(0, 1).expand_as(cx),
             torch.as_tensor(taper, dtype=torch.float32).clamp(0, 1).expand_as(cx),
             unit(opacity, OPACITY[0], OPACITY[1]).expand_as(cx),
             torch.as_tensor(pressure, dtype=torch.float32).clamp(0, 1).expand_as(cx),
             torch.as_tensor(speed, dtype=torch.float32).clamp(0, 1).expand_as(cx)]
    return torch.stack([p.expand_as(cx) for p in parts], -1)


def to_absolute(strokes: Strokes, cell_y: torch.Tensor, cell_x: torch.Tensor, cell: float, width: float,
                level: int) -> torch.Tensor:
    """Window strokes ``(M, N)`` -> rows of a painting ``(M * N kept, ABS)`` in units of the canvas width.
    ``cell_y, cell_x (M,)`` say which cell each window belongs to, ``cell`` is the cell's size in pixels of this
    level's canvas and ``width`` that canvas's width in pixels. The rows come out in painting order: stroke
    number first, then the four groups of windows that do not touch each other, then the cell."""
    g = geometry(strokes.geom)
    m, n = strokes.keep.shape
    origin = torch.stack([cell_x.float() * cell - cell / 2.0, cell_y.float() * cell - cell / 2.0], -1)   # (M, 2)
    span = 2.0 * cell

    def place(p):
        return (origin[:, None] + p * span) / width

    profile = brush_profile(strokes.brush)
    rows = torch.cat([place(g["p0"]), place(g["p1"]), place(g["p2"]),
                      (g["r0"] * span / width)[..., None], (g["r1"] * span / width)[..., None],
                      (g["opacity"] * strokes.keep)[..., None], profile, strokes.color, strokes.color2,
                      strokes.streak[..., None], strokes.phase, g["pressure"][..., None], g["speed"][..., None],
                      torch.full((m, n, 1), float(level), device=strokes.geom.device),
                      strokes.brush.argmax(-1, keepdim=True).float()], -1)
    group = (cell_y % 2) * 2 + (cell_x % 2)                                    # (M,)
    order = torch.arange(n, device=rows.device)[None].expand(m, n) * 4 + group[:, None]
    order = order.reshape(-1).float() * (m + 1) + torch.arange(m, device=rows.device).repeat_interleave(n)
    rows = rows.reshape(m * n, ABS)[order.argsort()]
    return rows[rows[:, A_ALPHA] > 0.02]


def random_strokes(shape: tuple, generator: torch.Generator | None = None, device=None,
                   palette: torch.Tensor | None = None) -> Strokes:
    """Random strokes of every kind, for training data and tests. ``shape`` is the leading shape ``(..., N)``.
    With a ``palette (..., P, 3)`` (linear RGB, leading shape without ``N``) the colours are pigments from it."""
    def rand(*s):
        return torch.rand(*s, generator=generator, device=generator.device if generator is not None else device)

    geom = rand(*shape, GEOM)
    geom[..., 7] = 0.6 + 0.4 * geom[..., 7]
    brush = torch.nn.functional.one_hot(
        torch.randint(0, len(BRUSHES), shape, generator=generator,
                      device=generator.device if generator is not None else device), len(BRUSHES)).float()
    if palette is None:
        color, color2 = rand(*shape, 3), rand(*shape, 3)
    else:
        p = palette.shape[-2]
        pick = torch.randint(0, p, (2, *shape), generator=generator,
                             device=generator.device if generator is not None else device)
        pal = palette.to(geom.device)
        color = torch.gather(pal, -2, pick[0][..., None].expand(*shape, 3))
        color2 = torch.gather(pal, -2, pick[1][..., None].expand(*shape, 3))
    out = Strokes(geom, brush, color, color2, rand(*shape) * 0.6, torch.ones(shape, device=geom.device),
                  rand(*shape, PHASES))
    return out if device is None else Strokes(*(getattr(out, f).to(device) for f in (
        "geom", "brush", "color", "color2", "streak", "keep", "phase")))


__all__ = ["ABS", "GEOM", "PHASES", "Strokes", "geometry", "inverse_geometry", "random_strokes", "to_absolute"]
