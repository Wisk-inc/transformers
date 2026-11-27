# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The making-of studio: a video of an animation being drawn by hand, frame after frame.

What is seen is a desk from above. The board (a whiteboard, or a sheet of paper) is the largest thing and sits on
the left; the tools lie in a tray on the right, each where it was put down, and the one in hand is missing from
it. A hand holds the tool and draws every stroke of every frame, over a faint copy of the frame before (the
lightbox). Along the top runs a slim exposure sheet: one cell for each frame of the animation, a thumbnail in the
cells already drawn, a line through the cells a drawing is held for, the frame being drawn marked. At the end the
finished animation is played.

The design rules, and where they come from:

1. Nothing a generated interface puts in by default: no purple or blue gradients, no glass panels, no glow, no
   heavy drop shadows, no emoji or sparkle icons, no rounded cards, no decoration that means nothing, no layout
   centred out of habit, no single generic font for everything. These are the tells listed in practitioners'
   catalogues of "AI slop" design: the anti-slop ban list of the ux-skill wiki (Laith0003 on GitHub), Kosta
   Canatselis, "Spot the slop: a UI designer's guide to fixing AI defaults" (world.hey.com), and the 925 Studios
   and DEV Community write-ups on the purple gradient and Inter defaults. They are opinions of working designers,
   not studies, but they agree with each other.
2. The work is the hero, the interface stays out of its way: neutral greys, one accent colour (the red of a
   playhead) used for one thing only (where the drawing is now), crisp 1 px rules, square corners, everything on
   an 8 px grid (at 1280 wide) and aligned to its edges.
3. The strip along the top is an exposure sheet, as in Toon Boom Harmony's Xsheet view ("designed to work like
   a traditional paper exposure sheet": the drawing in its cell, a held drawing shown by a line through the
   cells it is held for, empty cells where nothing is exposed) and the instances and exposure cells of TVPaint's
   timeline (doc.tvpaint.com). A background drawn once has its own cell, as a background has its own column on
   an x-sheet. Cels are numbered as Clip Studio Paint's timeline numbers them.
4. Two typefaces with two jobs: a plain grotesque for the few labels (short labels in capitals, as on a control
   panel), and a monospaced face for numbers, so frame counts and timings do not jitter as they change.
5. The physical things look physical, not drawn as icons: pens, markers, the cel brush and paint pots are
   shaded objects in their real colours, with soft contact shadows; the light comes from the top left for all of
   them, the hand included.

The hand is a procedural 3D model, not a photograph and not a capture of a real hand: a right hand in a dynamic
tripod grip (the tool pinched between the pads of the thumb and the index finger and resting on the side of the
middle finger, the ring and little fingers curled under, as occupational therapists describe it). It is made of
smooth signed-distance shapes, round cones for the forearm and the three segments of each finger and ellipsoids
for the back of the hand, the ball of the thumb and the web between thumb and index, blended with a smooth
maximum (Inigo Quilez, "smooth minimum" and "distance functions", iquilezles.org). Seen straight from above, each
shape gives its height and its normal at every pixel of the hand's own box only, so it is fast. The skin is lit
with wrapped diffuse light and a little red where the light turns to shadow, the cheap stand-in for light
scattered under the skin (Simon Green, "Real-time approximations to subsurface scattering", GPU Gems, chapter
16), a soft broad sheen, and darker creases where the shapes meet. Knuckles have creases, fingers have nails.
The hand casts a soft shadow on the board, further off and softer the higher each part of it is, so the shadow
moves away from the pen when it lifts.

The motion: the point of the tool is exactly on the stroke being drawn, always. The hand turns about it and the
forearm swings after it on damped springs, so they lag a little and overshoot when a stroke turns; the wrist turns
a little with the direction of the stroke. A hand slows on a curve (:func:`helix_lm.paint.draw.timing`). Between
strokes the tool lifts and travels in a low arc; to change tools the hand goes to the tray, puts one down and
takes the next (a twin marker is only turned round for its other nib, a cel brush is dipped in the next pot).

Use :func:`making_of` for the video, or :func:`shots` for its pictures one by one.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .animate import VideoWriter
from .color import linear_to_srgb, srgb_to_linear
from .draw import CELS, TOOLS, Board, Line, fill_rings, timing
from .see import gaussian_blur

# ------------------------------------------------------------------------------------------------ light --
LIGHT = torch.tensor([-0.42, -0.58, 0.70])
LIGHT = LIGHT / LIGHT.norm()                 # from the top left and above; z points up out of the board
HALF = (LIGHT + torch.tensor([0.0, 0.0, 1.0])) / (LIGHT + torch.tensor([0.0, 0.0, 1.0])).norm()
SUN = (1.0, 0.985, 0.96)                     # a warm white lamp
SKY = (0.16, 0.165, 0.175)                   # light from the room, from every side

# materials: wrap (how far light reaches round), scatter (red at the terminator), shine, sharpness of the shine
SKIN, NAIL, PLASTIC, MATTE, METAL, CLOTH, LACQUER = range(7)
MATERIALS = torch.tensor([
    [0.30, 1.00, 0.055, 14.0],      # skin
    [0.30, 0.45, 0.28, 60.0],       # nail
    [0.04, 0.00, 0.30, 70.0],       # plastic
    [0.10, 0.00, 0.02, 6.0],        # felt, bristles, paint
    [0.00, 0.00, 0.90, 38.0],       # metal
    [0.55, 0.00, 0.00, 1.0],        # cloth
    [0.05, 0.00, 0.45, 90.0],       # lacquered wood
])
SKIN_TONE = (0.86, 0.66, 0.54)       # sRGB; any other can be given


def _lin(c) -> torch.Tensor:
    return srgb_to_linear(torch.tensor(c, dtype=torch.float32))


# ------------------------------------------------------------------------------------------------ shapes --
@dataclass
class Cone:
    """A round cone from ``a`` to ``b`` (pixels, x y and z up), radii ``ra`` and ``rb``: a finger segment, the
    forearm, a part of a pen. ``group`` shapes of the same group melt into each other over ``k`` pixels."""

    a: tuple
    b: tuple
    ra: float
    rb: float
    colour: torch.Tensor
    material: int = SKIN
    group: int = 0
    k: float = 0.0
    nail: tuple | None = None        # (where along it 0..1, half length 0..1, up (x, y, z)): a nail on top
    creases: float = 0.0             # creases across the joint at its start, this dark


@dataclass
class Blob:
    """A smooth body seen from above: an ellipsoid of half axes ``axes`` (along ``angle``, across, up) around
    ``c``, sheared so that its top rises ``slope`` (along, across) per pixel."""

    c: tuple
    axes: tuple
    angle: float
    colour: torch.Tensor
    slope: tuple = (0.0, 0.0)
    material: int = SKIN
    group: int = 0
    k: float = 0.0


def _box(shape, aa: float = 1.0) -> tuple:
    if isinstance(shape, Cone):
        r = max(shape.ra, shape.rb) + aa + 1.0
        xs, ys = (shape.a[0], shape.b[0]), (shape.a[1], shape.b[1])
        return min(xs) - r, min(ys) - r, max(xs) + r, max(ys) + r
    big = max(shape.axes[0], shape.axes[1]) + aa + 1.0
    return shape.c[0] - big, shape.c[1] - big, shape.c[0] + big, shape.c[1] + big


def _cone(s: Cone, px: torch.Tensor, py: torch.Tensor, aa: float):
    """Coverage, height and normal of a round cone at the pixels ``px (1, w)``, ``py (h, 1)``."""
    ax, ay, az = s.a
    bx, by, bz = s.b
    dx, dy, dz = bx - ax, by - ay, bz - az
    length = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
    flat = math.sqrt(dx * dx + dy * dy)
    qx, qy = px - ax, py - ay
    # the outline seen from above: a 2D round cone
    t2 = ((qx * dx + qy * dy) / max(flat * flat, 1e-6)).clamp(0.0, 1.0) if flat > 1e-3 else torch.zeros_like(qx + qy)
    r2d = s.ra + (s.rb - s.ra) * t2
    ex, ey = qx - t2 * dx, qy - t2 * dy
    sdf = (ex * ex + ey * ey).sqrt() - r2d
    cov = (0.5 - sdf / aa).clamp(0.0, 1.0)
    # the top of the surface: the slanted cylinder between the end spheres, or one of them
    top = torch.full_like(sdf, -1e9)
    c = flat / length
    if c > 0.03:
        wx, wy = dx / flat, dy / flat
        along = qx * wx + qy * wy
        across = qx * -wy + qy * wx
        r = s.ra + (s.rb - s.ra) * t2
        room = r * r - across * across
        z = az + (dz / flat) * along + room.clamp(min=0.0).sqrt() / c
        tau = c * along + (dz / length) * (z - az)
        top = torch.where((room >= 0) & (tau >= 0) & (tau <= length), z, top)
    for (cx, cy, cz), rad in (((ax, ay, az), s.ra), ((bx, by, bz), s.rb)):
        room = rad * rad - (px - cx) ** 2 - (py - cy) ** 2
        top = torch.maximum(top, torch.where(room >= 0, cz + room.clamp(min=0.0).sqrt(), torch.full_like(room, -1e9)))
    edge = az + t2 * dz                               # just outside: the height of its middle
    top = torch.where(top < -1e8, edge.expand_as(top), top)
    # the normal: from the nearest point of the axis to the surface
    tau = (((px - ax) * dx + (py - ay) * dy + (top - az) * dz) / (length * length)).clamp(0.0, 1.0)
    nx, ny, nz = px - (ax + tau * dx), py - (ay + tau * dy), top - (az + tau * dz)
    norm = (nx * nx + ny * ny + nz * nz).sqrt().clamp(min=1e-6)
    normal = torch.stack([nx / norm, ny / norm, nz / norm])
    return cov, top, normal, tau


def _blob(s: Blob, px: torch.Tensor, py: torch.Tensor, aa: float):
    cx, cy, cz = s.c
    big_a, big_b, big_c = s.axes
    ca, sa = math.cos(s.angle), math.sin(s.angle)
    u = (px - cx) * ca + (py - cy) * sa
    v = -(px - cx) * sa + (py - cy) * ca
    k0 = ((u / big_a) ** 2 + (v / big_b) ** 2).sqrt()
    k1 = ((u / big_a ** 2) ** 2 + (v / big_b ** 2) ** 2).sqrt().clamp(min=1e-6)
    sdf = k0 * (k0 - 1.0) / k1
    cov = (0.5 - sdf / aa).clamp(0.0, 1.0)
    q = (1.0 - k0 * k0).clamp(min=0.0)
    root = q.sqrt()
    top = cz + s.slope[0] * u + s.slope[1] * v + big_c * root
    soft = root.clamp(min=0.04)
    gu = s.slope[0] - big_c * (u / big_a ** 2) / soft
    gv = s.slope[1] - big_c * (v / big_b ** 2) / soft
    gx, gy = gu * ca - gv * sa, gu * sa + gv * ca
    norm = (gx * gx + gy * gy + 1.0).sqrt()
    normal = torch.stack([-gx / norm, -gy / norm, 1.0 / norm])
    return cov, top, normal


def render(shapes: list, x0: int, y0: int, w: int, h: int, aa: float = 1.0) -> dict:
    """Shapes seen from above, in the box ``x0, y0, w, h`` (pixels): their coverage ``(h, w)``, height above the
    board ``(h, w)``, normal ``(3, h, w)``, colour (linear RGB, unlit) ``(3, h, w)`` and material ``(4, h, w)``.
    Each shape is worked out in its own box only; the shapes of one group melt into each other (a smooth
    maximum of their heights), the others lie on top of or under each other."""
    height = torch.full((h, w), -1e9)
    normal = torch.zeros(3, h, w)
    normal[2] = 1.0
    albedo = torch.zeros(3, h, w)
    params = torch.zeros(4, h, w)
    group = torch.full((h, w), -1, dtype=torch.int8)
    cover = torch.zeros(h, w)
    for s in shapes:
        bx0, by0, bx1, by1 = _box(s, aa)
        ix0, iy0 = max(x0, int(math.floor(bx0))), max(y0, int(math.floor(by0)))
        ix1, iy1 = min(x0 + w, int(math.ceil(bx1)) + 1), min(y0 + h, int(math.ceil(by1)) + 1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        px = torch.arange(ix0, ix1, dtype=torch.float32)[None, :] + 0.5
        py = torch.arange(iy0, iy1, dtype=torch.float32)[:, None] + 0.5
        if isinstance(s, Cone):
            cov, top, nrm, tau = _cone(s, px, py, aa)
        else:
            (cov, top, nrm), tau = _blob(s, px, py, aa), None
        has = cov > 0
        if not bool(has.any()):
            continue
        sl = (slice(iy0 - y0, iy1 - y0), slice(ix0 - x0, ix1 - x0))
        every = (slice(None), *sl)
        colour, mat = s.colour[:, None, None], MATERIALS[s.material][:, None, None]
        if isinstance(s, Cone) and (s.nail is not None or s.creases > 0):
            colour, mat = colour.expand(3, *cov.shape).clone(), mat.expand(4, *cov.shape).clone()
            _decals(s, tau, nrm, colour, mat)
        old_h, old_g = height[sl], group[sl]
        empty = old_h < -1e8
        same = (old_g == s.group) & ~empty if s.k > 0 else torch.zeros_like(empty)
        k = max(s.k, 1e-3)
        blend = (0.5 + 0.5 * (top - old_h) / k).clamp(0.0, 1.0)
        mix = torch.where(same, blend, (top > old_h).float() * cov)       # another group: on top or under
        mix = torch.where(empty, 1.0, mix) * has
        new_h = torch.where(same, old_h + (top - old_h) * blend + k * blend * (1.0 - blend), torch.maximum(top, old_h))
        height[sl] = torch.where(has, torch.where(empty, top, new_h), old_h)
        n = normal[every].lerp_(nrm, mix)
        n.div_(n.norm(dim=0, keepdim=True).clamp(min=1e-6))
        albedo[every].lerp_(colour.expand_as(albedo[every]), mix)
        params[every].lerp_(mat.expand_as(params[every]), mix)
        group[sl] = torch.where(mix > 0.5, s.group, old_g).to(torch.int8)
        torch.maximum(cover[sl], cov, out=cover[sl])
    return {"cover": cover, "height": height.clamp(min=0.0), "normal": normal, "albedo": albedo, "params": params}


def _decals(s: Cone, tau: torch.Tensor, nrm: torch.Tensor, colour: torch.Tensor, mat: torch.Tensor) -> None:
    """A nail on top of a finger's last segment, creases across a joint: colour and material, in place."""
    length = math.dist(s.a, s.b)
    axis = torch.tensor([s.b[0] - s.a[0], s.b[1] - s.a[1], s.b[2] - s.a[2]]) / max(length, 1e-6)
    if s.creases > 0:
        r = s.ra
        along = tau * length
        lines = sum(torch.exp(-(((along - off * r) / (0.07 * r + 0.35)) ** 2)) for off in (0.15, 0.42, 0.7))
        facing = (nrm[2] * 1.2 - 0.1).clamp(0.0, 1.0)
        colour.mul_(1.0 - s.creases * lines * facing)
    if s.nail is not None:
        where, half, up = s.nail
        up = torch.tensor(up, dtype=torch.float32)
        up = up - axis * float(up @ axis)
        up = up / up.norm().clamp(min=1e-6)
        side = (nrm * up[:, None, None]).sum(0)                     # 1 straight on top
        along = (tau - where) / half
        inside = (1.0 - along * along) * 2.5 + (side - 0.62) * 9.0
        nail = (inside.clamp(-1.0, 1.0) * 0.5 + 0.5) * (side > 0.2)
        rim = torch.exp(-((along + 1.0) / 0.22) ** 2) * nail          # the skin over its root
        tint = _lin((0.93, 0.74, 0.70))
        tip = ((along - 0.55) / 0.45).clamp(0.0, 1.0)                 # the free edge, paler
        nail_colour = tint[:, None, None] * (1.0 + 0.10 * tip) * (1.0 - 0.25 * rim)
        colour.mul_(1.0 - nail).add_(nail_colour * nail)
        mat.mul_(1.0 - nail).add_(MATERIALS[NAIL][:, None, None] * nail)


def shade(g: dict, unit: float, lift: float = 0.0) -> torch.Tensor:
    """The rendered shapes lit (linear RGB ``(3, h, w)``): wrapped diffuse light with a red band where it turns
    to shadow on skin, light from the room, a soft sheen, shadows the shapes cast on each other, darker where
    they crowd each other. Only the pixels they cover are lit."""
    cover = g["cover"]
    h, w = cover.shape
    height = g["height"] - lift
    # what needs the neighbours (crowding, shadows), worked out at half the size
    small = F.avg_pool2d(height[None, None], 2, ceil_mode=True)[0, 0] if min(h, w) > 8 else height
    near = gaussian_blur(small, max(0.5, 0.025 * unit))
    crowd = (1.0 - 2.6 * (near - small).clamp(min=0.0) / unit).clamp(0.4, 1.0)
    dark = _self_shadow(small, unit / (2.0 if small is not height else 1.0))
    if small is not height:
        both = F.interpolate(torch.stack([crowd, dark])[None], scale_factor=2, mode="bilinear",
                             align_corners=False)[0, :, :h, :w]
        crowd, dark = both[0], both[1]
    where = (cover > 0).flatten().nonzero().flatten()
    n = g["normal"].reshape(3, -1)[:, where]
    albedo = g["albedo"].reshape(3, -1)[:, where]
    p = g["params"].reshape(4, -1)[:, where]
    crowd, dark = crowd.flatten()[where], dark.flatten()[where]
    wrap, scatter, shine, sharp = p[0], p[1], p[2], p[3].clamp(min=1.0)
    ndl = (n * LIGHT[:, None]).sum(0)
    diffuse = ((ndl + wrap) / (1.0 + wrap)).clamp(min=0.0) * (1.0 - 0.8 * dark)
    band = _smooth(0.0, 0.32, diffuse) * _smooth(0.64, 0.32, diffuse) * scatter
    edge = (1.0 - n[2]).clamp(min=0.0) ** 2 * scatter                  # thin skin at the edges looks redder
    red = torch.tensor([0.22, 0.05, 0.015])[:, None] * band + torch.tensor([0.06, 0.0, -0.01])[:, None] * edge
    ndh = (n * HALF[:, None]).sum(0).clamp(min=0.0)
    gloss = shine * ndh ** sharp * crowd * (1.0 - 0.7 * dark)
    sun = torch.tensor(SUN)[:, None]
    sky = torch.tensor(SKY)[:, None] * (0.55 + 0.45 * n[2]) * crowd
    metal = (p[2] > 0.7).float()                                       # metal shows the room more than the lamp
    lit = albedo * (sun * diffuse * (0.9 + 0.1 * crowd) + sky) + albedo * red + sun * gloss
    lit = lit + metal * 0.25 * albedo * (0.5 - 0.5 * n[1]) * crowd
    out = torch.zeros(3, h * w)
    out[:, where] = lit
    return out.reshape(3, h, w)


def _self_shadow(height: torch.Tensor, unit: float) -> torch.Tensor:
    """Where a part of the shapes is in the shadow of another (the pen on a finger, a finger on the back of the
    hand): looked for by walking from each pixel toward the light over the heights, 0..1 ``(h, w)``."""
    h, w = height.shape
    away = LIGHT[:2] / LIGHT[:2].norm()
    rise = float(LIGHT[2] / LIGHT[:2].norm())
    soft = 0.012 * unit + 0.5
    worst = torch.zeros_like(height)
    step, seen = 1.0, set()
    while step < 0.4 * unit:
        dx, dy = round(float(away[0]) * step), round(float(away[1]) * step)
        step *= 1.45
        if (dx, dy) in seen or abs(dx) >= w or abs(dy) >= h:
            continue
        seen.add((dx, dy))
        moved = torch.zeros_like(height)
        ys, yd = slice(max(0, dy), h + min(0, dy)), slice(max(0, -dy), h + min(0, -dy))
        xs, xd = slice(max(0, dx), w + min(0, dx)), slice(max(0, -dx), w + min(0, -dx))
        moved[yd, xd] = height[ys, xs]
        worst = torch.maximum(worst, ((moved - math.hypot(dx, dy) * rise - height) / soft).clamp(0.0, 1.0))
    return gaussian_blur(worst, 0.5 + 0.004 * unit)


def _smooth(a: float, b: float, x: torch.Tensor) -> torch.Tensor:
    t = ((x - a) / (b - a)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def cast_shadow(cover: torch.Tensor, height: torch.Tensor, unit: float, pad: int) -> torch.Tensor:
    """The shadow the shapes cast on the board, ``(h + 2 pad, w + 2 pad)`` in 0..1: each part of them is cast
    away from the light by its height, and blurred more the higher it is (a lamp is not a point)."""
    h, w = cover.shape
    down = max(1, min(4, round(unit / 40)))
    big = F.pad(torch.stack([cover, height * (cover > 0)]), (pad, pad, pad, pad))
    hh, ww = big.shape[-2] // down * down, big.shape[-1] // down * down
    small = F.avg_pool2d(big[None, :, :hh, :ww], down)[0]
    cov_s, hgt_s = small[0], small[1] / small[0].clamp(min=1e-4)
    away = (-LIGHT[:2] / LIGHT[2]).tolist()
    total = torch.ones_like(cov_s)
    sh, sw = cov_s.shape
    for level in (0.015, 0.09, 0.22, 0.45, 0.9, 1.6):
        z = level * unit
        width = 0.45 * z + 0.6 * unit * 0.02
        weight = (1.0 - (hgt_s - z).abs() / width).clamp(0.0, 1.0) * cov_s
        if float(weight.max()) <= 0:
            continue
        ox, oy = away[0] * z / down, away[1] * z / down
        theta = torch.tensor([[[1.0, 0.0, -2.0 * ox / sw], [0.0, 1.0, -2.0 * oy / sh]]])
        grid = F.affine_grid(theta, (1, 1, sh, sw), align_corners=False)
        moved = F.grid_sample(weight[None, None], grid, align_corners=False)[0, 0]
        sigma = (0.6 + 0.10 * z) / down
        soft = gaussian_blur(moved, sigma)
        strength = max(0.16, 0.42 - 0.16 * z / unit)
        total = total * (1.0 - strength * soft.clamp(0.0, 1.0))
    shadow = F.interpolate((1.0 - total)[None, None], size=(hh, ww), mode="bilinear", align_corners=False)[0, 0]
    return F.pad(shadow, (0, big.shape[-1] - ww, 0, big.shape[-2] - hh))


# ------------------------------------------------------------------------------------------------- tools --
# a tool along its axis from the point, in hand lengths: (from, to, radius there, radius here, material, what)
PEN_BODY, INK, CAP, STEEL = "body", "ink", "cap", "steel"
FELT, BRISTLE, PAINT, WOOD = "felt", "bristle", "paint", "wood"


def tool_parts(tool: str, posted: bool = True) -> list:
    """The parts of a tool along its axis from the point, in hand lengths. ``posted``: the cap is on the back
    end (in hand); else on the point (in the tray)."""
    if tool in ("pen", "brush pen"):
        point = [(0.0, 0.010, 0.0016, 0.0022, METAL, STEEL), (0.010, 0.045, 0.004, 0.015, PLASTIC, CAP)] \
            if tool == "pen" else [(0.0, 0.055, 0.0015, 0.011, MATTE, INK), (0.055, 0.07, 0.012, 0.016, PLASTIC, CAP)]
        body = [(0.045, 0.10, 0.017, 0.021, PLASTIC, PEN_BODY), (0.10, 0.62, 0.022, 0.022, PLASTIC, PEN_BODY)]
        if posted:
            return point + body + [(0.56, 0.76, 0.026, 0.026, PLASTIC, CAP), ("clip", 0.60, 0.74, 0.006, CAP)]
        return [(0.0, 0.19, 0.026, 0.026, PLASTIC, CAP), ("clip", 0.04, 0.17, 0.006, CAP)] + body[1:] + \
            [(0.62, 0.64, 0.021, 0.018, PLASTIC, PEN_BODY)]
    if tool == "pencil":
        return [(0.0, 0.012, 0.001, 0.0045, MATTE, INK), (0.012, 0.075, 0.005, 0.020, MATTE, WOOD),
                (0.075, 0.66, 0.020, 0.020, LACQUER, PEN_BODY), (0.66, 0.70, 0.021, 0.021, METAL, STEEL),
                (0.70, 0.73, 0.019, 0.018, MATTE, PAINT)]
    if tool in ("marker", "fine marker"):
        nib = [(0.0, 0.035, 0.007, 0.016, MATTE, FELT)] if tool == "marker" else [(0.0, 0.03, 0.003, 0.007, MATTE,
                                                                                            FELT)]
        collar = [(0.03, 0.075, 0.024, 0.032, PLASTIC, PEN_BODY)]
        barrel = [(0.075, 0.70, 0.040, 0.040, PLASTIC, PEN_BODY), (0.58, 0.64, 0.041, 0.041, PLASTIC, CAP)]
        if posted:
            return nib + collar + barrel + [(0.64, 0.84, 0.044, 0.042, PLASTIC, CAP)]
        return [(0.0, 0.16, 0.044, 0.043, PLASTIC, CAP)] + barrel + [(0.64, 0.80, 0.044, 0.042, PLASTIC, CAP)]
    # the cel brush: bristles wet with paint, a steel ferrule, a lacquered handle
    return [(0.0, 0.045, 0.0015, 0.013, MATTE, PAINT), (0.045, 0.085, 0.013, 0.016, MATTE, BRISTLE),
            (0.085, 0.17, 0.016, 0.018, METAL, STEEL), (0.17, 0.42, 0.020, 0.024, LACQUER, WOOD),
            (0.42, 0.88, 0.024, 0.012, LACQUER, WOOD)]


def _tool_colours(tool: str, colour) -> dict:
    ink = _lin(colour)
    grey = 0.5 * (ink + _lin((0.16, 0.16, 0.17)))
    body = {"pen": (0.86, 0.86, 0.84), "brush pen": (0.20, 0.20, 0.22), "pencil": (0.18, 0.30, 0.24),
            "marker": (0.93, 0.93, 0.91), "fine marker": (0.93, 0.93, 0.91), "brush": (0.12, 0.12, 0.13)}[tool]
    return {PEN_BODY: _lin(body), INK: grey if tool == "pencil" else ink, CAP: ink if tool != "pencil" else grey,
            STEEL: _lin((0.62, 0.63, 0.65)), FELT: ink, BRISTLE: _lin((0.80, 0.66, 0.45)),
            PAINT: ink if tool == "brush" else _lin((0.85, 0.55, 0.55)), WOOD: _lin((0.42, 0.12, 0.08))
            if tool == "brush" else _lin((0.80, 0.62, 0.42))}


def tool_shapes(tool: str, colour, start, axis, up, unit: float, posted: bool = True, group: int = 1) -> list:
    """The shapes of a tool whose point is at ``start`` (pixels, with height), pointing back along ``axis`` (a
    unit 3D vector), its clip on the ``up`` side."""
    colours = _tool_colours(tool, colour)
    out = []
    sx, sy, sz = start
    for part in tool_parts(tool, posted):
        if part[0] == "clip":
            _, s0, s1, r, what = part
            off = 0.028 * unit
            a = tuple(start[i] + axis[i] * s0 * unit + up[i] * off for i in range(3))
            b = tuple(start[i] + axis[i] * s1 * unit + up[i] * (off + 0.004 * unit) for i in range(3))
            out.append(Cone(a, b, r * unit, r * unit, colours[what], METAL, group))
            continue
        s0, s1, r0, r1, material, what = part
        a = (sx + axis[0] * s0 * unit, sy + axis[1] * s0 * unit, sz + axis[2] * s0 * unit)
        b = (sx + axis[0] * s1 * unit, sy + axis[1] * s1 * unit, sz + axis[2] * s1 * unit)
        out.append(Cone(a, b, r0 * unit, r1 * unit, colours[what], material, group))
    return out


# -------------------------------------------------------------------------------------------------- hand --
# The hand in its own frame: f forward (toward the finger tips), l toward the thumb, z up from the board, in hand
# lengths (wrist crease to the tip of the middle finger, some 19 cm); the point of the tool at the origin.
TILT, TURN = math.radians(48.0), math.radians(20.0)
PEN_AXIS = (-math.cos(TURN) * math.cos(TILT), math.sin(TURN) * math.cos(TILT), math.sin(TILT))
PEN_UP = (math.cos(TURN) * math.sin(TILT), -math.sin(TURN) * math.sin(TILT), math.cos(TILT))

# joints (f, l, z) from the knuckle to the tip, placed round the pen: the index pad on top of it a quarter of a
# hand length up from the point, the thumb pad against its side a little further up, the middle finger under it
JOINTS = {
    "index": [(-0.430, 0.000, 0.300), (-0.261, 0.027, 0.368), (-0.196, 0.036, 0.275), (-0.124, 0.045, 0.231)],
    "middle": [(-0.440, -0.095, 0.282), (-0.282, -0.058, 0.208), (-0.214, -0.008, 0.136), (-0.150, 0.014, 0.096)],
    "ring": [(-0.455, -0.180, 0.244), (-0.362, -0.162, 0.122), (-0.392, -0.140, 0.042), (-0.448, -0.124, 0.034)],
    "little": [(-0.470, -0.255, 0.196), (-0.388, -0.236, 0.086), (-0.412, -0.218, 0.032), (-0.460, -0.202, 0.030)],
    "thumb": [(-0.620, 0.150, 0.150), (-0.362, 0.258, 0.238), (-0.236, 0.178, 0.250), (-0.170, 0.124, 0.212)],
}
FOLLOW = (1.0, 0.6, 0.25, 0.0)
RADII = {
    "index": (0.052, 0.046, 0.044, 0.040, 0.039, 0.035),
    "middle": (0.054, 0.047, 0.045, 0.041, 0.040, 0.036),
    "ring": (0.050, 0.044, 0.042, 0.038, 0.037, 0.033),
    "little": (0.044, 0.039, 0.037, 0.034, 0.033, 0.030),
    "thumb": (0.068, 0.058, 0.060, 0.052, 0.050, 0.044),
}


@dataclass
class Pose:
    """Where the hand is: the point of the tool (pixels), how high it is lifted (pixels), which way the hand
    points (radians, the direction of its fingers on the screen), how far the back of the hand has given
    (pixels, x y) and which way the forearm goes (radians)."""

    tip: tuple
    lift: float = 0.0
    yaw: float = -2.2
    give: tuple = (0.0, 0.0)
    arm: float = -2.4


def hand_shapes(pose: Pose, unit: float, tool: str | None, colour, skin=SKIN_TONE, reach: float = 4.0,
                hand: bool = True) -> list:
    """The shapes of a right hand holding ``tool`` (or only the tool when not ``hand``), posed by ``pose``,
    ``unit`` pixels to a hand length. The forearm reaches ``reach`` hand lengths away (off the picture)."""
    fx, fy = math.cos(pose.yaw), math.sin(pose.yaw)
    lx, ly = fy, -fx
    tx, ty = pose.tip
    lift = pose.lift

    def world(p, follow: float = 0.0) -> tuple:
        f, l, z = p
        return (tx + unit * (f * fx + l * lx) + follow * pose.give[0],
                ty + unit * (f * fy + l * ly) + follow * pose.give[1], unit * z + lift)

    def vec(p) -> tuple:
        f, l, z = p
        return (f * fx + l * lx, f * fy + l * ly, z)

    out = []
    if tool is not None:
        axis, up = vec(PEN_AXIS), vec(PEN_UP)
        # the back of the pen rests in the web of the thumb, so it moves a little with the back of the hand
        rest = [a * 0.45 * unit for a in axis]
        rest = (rest[0] + 0.9 * pose.give[0], rest[1] + 0.9 * pose.give[1], rest[2])
        norm = math.sqrt(sum(a * a for a in rest))
        axis = tuple(a / norm for a in rest)
        out += tool_shapes(tool, colour, (tx, ty, lift), axis, up, unit)
    if not hand:
        return out
    base = _lin(skin)
    back = base * torch.tensor([1.0, 1.0, 0.97])
    knuckle = base * torch.tensor([1.03, 0.92, 0.89])
    tip_tone = base * torch.tensor([1.03, 0.90, 0.88])
    u = unit
    # the back of the hand (wide at the knuckles, narrow at the wrist), the ball of the thumb, the web between
    # thumb and index
    out.append(Blob(world((-0.610, -0.120, 0.175), 1.0), (0.190 * u, 0.205 * u, 0.060 * u), pose.yaw,
                    back, (0.30, -0.26), SKIN, 0, 0.06 * u))
    out.append(Blob(world((-0.830, -0.090, 0.130), 1.0), (0.175 * u, 0.158 * u, 0.058 * u), pose.yaw,
                    back, (0.24, -0.20), SKIN, 0, 0.08 * u))
    out.append(Blob(world((-0.540, 0.170, 0.150), 1.0), (0.175 * u, 0.090 * u, 0.072 * u), pose.yaw - 0.40,
                    base, (0.12, 0.0), SKIN, 0, 0.05 * u))
    out.append(Blob(world((-0.370, 0.125, 0.232), 0.9), (0.100 * u, 0.062 * u, 0.060 * u), pose.yaw - 0.90,
                    base, (0.0, 0.0), SKIN, 0, 0.045 * u))
    wrist = world((-0.960, -0.085, 0.055), 1.0)
    # the tendons from the wrist to the knuckles, just under the skin
    for name, z0 in (("index", 0.215), ("middle", 0.205), ("ring", 0.185), ("little", 0.16)):
        knuckle_at = JOINTS[name][0]
        start = world((-0.86, -0.07 - 0.03 * ("index", "middle", "ring", "little").index(name), z0 - 0.04), 1.0)
        end = world((knuckle_at[0] + 0.05, knuckle_at[1], knuckle_at[2] - 0.035), 1.0)
        out.append(Cone(start, end, 0.016 * u, 0.020 * u, back, SKIN, 0, 0.05 * u))
    # the forearm, out of the picture, and the sleeve over it
    ax, ay = math.cos(pose.arm), math.sin(pose.arm)
    rise = 0.20

    def arm(d: float, z0: float) -> tuple:
        return (wrist[0] + ax * d * u, wrist[1] + ay * d * u, wrist[2] + (z0 + rise * d) * u)

    out.append(Cone(arm(-0.05, 0.0), arm(0.6, 0.0), 0.135 * u, 0.170 * u, base, SKIN, 0, 0.09 * u))
    out.append(Cone(arm(0.6, 0.0), arm(1.25, 0.0), 0.170 * u, 0.200 * u, base, SKIN, 0, 0.0))
    cloth = _lin((0.24, 0.25, 0.27))
    out.append(Cone(arm(1.05, 0.02), arm(1.17, 0.02), 0.235 * u, 0.242 * u, cloth * 1.12, CLOTH, 2))
    d = 1.15                                     # the sleeve in short pieces, so that each has a small box
    while d < reach:
        nxt = min(reach, d + 0.8)
        out.append(Cone(arm(d, 0.02), arm(nxt, 0.02), (0.235 + 0.004 * (d - 1.15)) * u,
                        (0.235 + 0.004 * (nxt - 1.15)) * u, cloth, CLOTH, 2))
        d = nxt
    # the fingers: three segments each, the knuckles of the hand at their roots
    for name, joints in JOINTS.items():
        radii = RADII[name]
        pts = [world(p, FOLLOW[i]) for i, p in enumerate(joints)]
        if name != "thumb":
            out.append(Cone(pts[0], pts[0], radii[0] * 1.06 * u, radii[0] * 1.06 * u, knuckle, SKIN, 0, 0.045 * u))
        for i in range(3):
            if name == "thumb" and i == 0:                 # the thumb's first bone is inside the ball of the thumb
                out.append(Cone(pts[0], pts[1], radii[0] * u, radii[1] * u, base, SKIN, 0, 0.05 * u))
                continue
            colour = tip_tone if i == 2 else (knuckle if i == 1 else base)
            nail = None
            if i == 2 and name in ("index", "middle", "thumb"):
                up = (0.0, 0.0, 1.0) if name != "thumb" else (lx * 0.55, ly * 0.55, 0.83)
                nail = (0.62, 0.40, up)
            out.append(Cone(pts[i], pts[i + 1], radii[2 * i] * u, radii[2 * i + 1] * u, colour, SKIN, 0,
                            0.016 * u, nail, 0.22 if i > 0 else 0.0))
    return out


# ------------------------------------------------------------------------------------------------ layout --
BAR = (29, 29, 30)                  # the strip along the top
RULE = (58, 58, 61)                 # 1 px rules on it
EMPTY = (38, 38, 40)                # a cell with nothing exposed yet
TEXT = (228, 228, 224)
DIM = (138, 138, 133)
ACCENT = (226, 84, 58)              # the playhead red, for the frame being drawn only
DESK = (0.705, 0.695, 0.675)        # matte warm grey (sRGB)
TRAY = (0.645, 0.636, 0.618)
ALUMINIUM = (0.78, 0.79, 0.80)
INK_WEIGHT = 3.5                    # how much slower the time-lapse runs while lines are drawn than for colour


@dataclass
class Layout:
    """Where everything goes in a picture ``width`` by ``height``: ``grid`` is the 8 px step at 1280 wide."""

    width: int
    height: int
    grid: float
    bar: int
    margin: int
    board: tuple                     # x0, y0, width, height of the drawing surface
    rim: int
    tray: tuple                      # x0, y0, x1, y1
    unit: float                      # pixels to a hand length
    cell: tuple                      # thumbnail width, height


def layout(width: int, size: tuple) -> Layout:
    """The studio for an animation of ``size (h, w)`` in a picture ``width`` wide, 16:9."""
    height = round(width * 9 / 16)
    height += height % 2
    g = width / 160.0
    bar = max(8, round(11 * g))
    margin = max(2, round(3 * g))
    unit = 0.27 * (height - bar)
    tray_w = round((0.9 * unit + 4 * g) / g) * g
    rim = max(1, round(0.75 * g))
    aw = width - 2 * margin - tray_w - round(4 * g) - 2 * rim
    ah = height - bar - 2 * margin - 2 * rim
    h, w = size
    scale = min(aw / w, ah / h)
    bw, bh = max(8, round(w * scale)), max(8, round(h * scale))
    x0 = margin + rim
    y0 = bar + margin + rim + (ah - bh) // 2
    tray = (round(width - margin - tray_w), y0 - rim, width - margin, height - margin - round(2 * g))
    ch = max(4, round(5.5 * g))
    cw = max(3, min(round(10 * g), round(ch * w / h)))
    return Layout(width, height, g, bar, margin, (x0, y0, bw, bh), rim, tray, unit, (cw, ch))


def _font(size: float, mono: bool = False):
    """A monospaced face for numbers, a plain grotesque (Pillow's own) for labels."""
    from PIL import ImageFont

    size = max(6, round(size))
    if mono:
        for name in ("DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "Menlo.ttc", "consola.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                   # Pillow before 10.1: the small bitmap font
        return ImageFont.load_default()


def _u8(image: torch.Tensor) -> torch.Tensor:
    """sRGB ``(3, h, w)`` 0..1 -> ``(h, w, 3)`` bytes."""
    return (image.clamp(0.0, 1.0) * 255.0 + 0.5).byte().permute(1, 2, 0).contiguous()


_TO_LINEAR = srgb_to_linear(torch.arange(256, dtype=torch.float32) / 255.0)
_TO_BYTES = (linear_to_srgb(torch.arange(4096, dtype=torch.float32) / 4095.0) * 255.0 + 0.5).byte()


def _from_u8(pixels: torch.Tensor) -> torch.Tensor:
    """``(h, w, 3)`` sRGB bytes -> linear ``(3, h, w)``, by table."""
    return _TO_LINEAR[pixels.permute(2, 0, 1).long()]


def _to_u8(linear: torch.Tensor) -> torch.Tensor:
    """Linear ``(3, h, w)`` -> ``(h, w, 3)`` sRGB bytes, by table."""
    return _TO_BYTES[(linear.clamp(0.0, 1.0) * 4095.0 + 0.5).long()].permute(1, 2, 0).contiguous()


def _noise(h: int, w: int, amount: float, seed: float = 0.0) -> torch.Tensor:
    y = torch.arange(h, dtype=torch.float32)[:, None]
    x = torch.arange(w, dtype=torch.float32)[None, :]
    fine = torch.frac(torch.sin(y * 12.9898 + x * 78.233 + seed) * 43758.5453) - 0.5
    return 1.0 + amount * fine


# ------------------------------------------------------------------------------------------------ the tray --
@dataclass
class Item:
    """A thing in the tray: a tool lying down, or a pot of cel paint."""

    key: tuple
    tool: str
    colour: tuple
    at: tuple = (0.0, 0.0)           # where the hand takes it (pixels): the point of a tool, the middle of a pot


def held(line: Line) -> tuple:
    """What a hand holds to draw ``line``: the tool's name and colour."""
    tool = line.tool if line.tool in TOOLS else "pen"
    if line.kind in CELS:
        tool = "brush"
    return tool, tuple(round(float(c), 4) for c in line.colour)


def item_key(holding: tuple | None) -> tuple | None:
    """The tray item a held tool is: both ends of a twin marker are one marker, the cel brush is one brush."""
    if holding is None:
        return None
    tool, colour = holding
    if tool in ("marker", "fine marker"):
        return ("marker", colour)
    if tool == "brush":
        return ("brush", None)
    return (tool, colour)


def _pot(canvas: torch.Tensor, cx: float, cy: float, radius: float, colour) -> None:
    """A round pot of paint seen from above, its lid off: a rim, the paint a little below it (linear, in
    place)."""
    _, h, w = canvas.shape
    x0, x1 = max(0, int(cx - radius - 3)), min(w, int(cx + radius + 4))
    y0, y1 = max(0, int(cy - radius - 3)), min(h, int(cy + radius + 4))
    if x1 <= x0 or y1 <= y0:
        return
    xs = torch.arange(x0, x1, dtype=torch.float32)[None, :] + 0.5 - cx
    ys = torch.arange(y0, y1, dtype=torch.float32)[:, None] + 0.5 - cy
    d = (xs * xs + ys * ys).sqrt()
    region = canvas[:, y0:y1, x0:x1]
    # its shadow, to the lower right
    sd = ((xs - 0.12 * radius) ** 2 + (ys - 0.16 * radius) ** 2).sqrt()
    region.mul_(1.0 - 0.28 * (0.5 - (sd - radius) / (0.18 * radius + 1.0)).clamp(0.0, 1.0))
    cover = (radius - d + 0.5).clamp(0.0, 1.0)
    inner = 0.78 * radius
    ring = ((d - inner) / (radius - inner)).clamp(0.0, 1.0)
    # the rim: a rounded lip lit from the top left
    lip = (2.0 * ring - 1.0).clamp(-1.0, 1.0)
    nx, ny = xs / d.clamp(min=1e-3) * lip, ys / d.clamp(min=1e-3) * lip
    nz = (1.0 - nx * nx - ny * ny).clamp(min=0.0).sqrt()
    light = (nx * float(LIGHT[0]) + ny * float(LIGHT[1]) + nz * float(LIGHT[2])).clamp(min=0.0)
    rim_colour = _lin((0.90, 0.90, 0.89))[:, None, None] * (0.25 + 0.85 * light)
    paint = _lin(colour)[:, None, None]
    # the paint: flat, in the rim's shadow on the side the light comes from, a small glint
    toward = -(xs * float(LIGHT[0]) + ys * float(LIGHT[1])) / max(radius, 1.0)
    shade_in = 0.82 + 0.18 * (1.0 - (toward * 1.4).clamp(0.0, 1.0)) * (d < inner).float()
    glint = torch.exp(-(((xs + 0.30 * radius) ** 2 + (ys + 0.34 * radius) ** 2) / (0.012 * radius * radius + 1.0)))
    inside = paint * shade_in * 0.92 + 0.10 * glint
    a = ((inner - d) + 0.5).clamp(0.0, 1.0)
    look = inside * a + rim_colour * (1.0 - a)
    region.mul_(1.0 - cover).add_(look * cover)


class Tray:
    """The tray of tools beside the board, drawn once for each tool that can be missing from it."""

    def __init__(self, items: list, lay: Layout, pots: list) -> None:
        self.lay, self.items, self.pots = lay, items, pots
        x0, y0, x1, y1 = lay.tray
        u, g = lay.unit, lay.grid
        pad = max(2, round(2 * g))
        y = y0 + pad + 0.06 * u
        for item in items:
            r = max(max(p[2], p[3]) for p in tool_parts(item.tool, posted=False) if p[0] != "clip") * u
            item.at = (x0 + pad + 0.02 * u, y + r)
            y += 2 * r + max(3.0, 1.1 * g + 0.035 * u)
        radius = max(3.0, 0.085 * u)
        across = max(1, int((x1 - x0 - 2 * pad) // (2.6 * radius)))
        y += radius * 0.6
        for i, pot in enumerate(pots):
            col, row = i % across, i // across
            pot.at = (x0 + pad + radius + col * 2.6 * radius, y + radius + row * 2.5 * radius)
        last = max([it.at[1] + 0.06 * u for it in items] + [p.at[1] + radius for p in pots] + [y0 + 4 * g])
        self.box = (x0, y0, x1, min(y1, int(last + pad)))
        self.radius = radius
        self.cache: dict = {}

    def picture(self, missing) -> torch.Tensor:
        """The tray (linear RGB, its box) with every tool in it but ``missing``."""
        if missing in self.cache:
            return self.cache[missing]
        x0, y0, x1, y1 = self.box
        w, h = x1 - x0, y1 - y0
        tray = _lin(TRAY)[:, None, None] * _noise(h, w, 0.025, 3.0)
        tray = tray.expand(3, h, w).clone()
        if h > 2 and w > 2:                                # an inset: the shadow of the edge inside it
            tray[:, :2] *= 0.86
            tray[:, :, :2] *= 0.88
        u = self.lay.unit
        shapes = []
        for item in self.items:
            if item.key == missing:
                continue
            r = max(max(p[2], p[3]) for p in tool_parts(item.tool, posted=False) if p[0] != "clip") * u
            start = (item.at[0] - x0, item.at[1] - y0, r)
            shapes += tool_shapes(item.tool, item.colour, start, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), u, posted=False)
        if shapes:
            g = render(shapes, 0, 0, w, h)
            lit = shade(g, u)
            pad = max(4, round(0.05 * u))
            shadow = cast_shadow(g["cover"], g["height"], u, pad)[pad: pad + h, pad: pad + w]
            tray = tray * (1.0 - shadow) * (1.0 - g["cover"]) + lit * g["cover"]
        for pot in self.pots:
            _pot(tray, pot.at[0] - x0, pot.at[1] - y0, self.radius, pot.colour)
        self.cache[missing] = tray
        return tray


# ------------------------------------------------------------------------------------------------ motion --
@dataclass
class Step:
    """A stretch of the making-of: ``draw`` a line, ``move`` the hand (lifted) from ``a`` to ``b``, or ``dwell``
    at ``a``; from ``t0`` to ``t1`` seconds of the video."""

    kind: str
    t0: float
    t1: float
    a: tuple = (0.0, 0.0)
    b: tuple = (0.0, 0.0)
    lift: float = 0.0
    index: int = -1
    times: torch.Tensor | None = None
    swap: tuple | None = None        # the tool held from the middle of this step on


def _ease(u: float) -> float:
    """Minimum jerk: how far a hand has got along a reach at ``u`` of its time (0..1)."""
    return u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)


def _cel_area(line: Line) -> float:
    area = []
    for ring in line.ring_points():
        x, y = ring[:, 0], ring[:, 1]
        area.append(abs(float((x * torch.roll(y, -1) - torch.roll(x, -1) * y).sum())) / 2.0)
    return max(0.0, area[0] - sum(area[1:])) if area else 0.0


class Spring:
    """A damped spring: a value pulled toward a target, ``hz`` times a second, a little overshoot when
    ``damping`` is below 1."""

    def __init__(self, hz: float, damping: float, angle: bool = False) -> None:
        self.w, self.z, self.angle = 2.0 * math.pi * hz, damping, angle
        self.x: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def __call__(self, target, dt: float) -> torch.Tensor:
        target = torch.as_tensor(target, dtype=torch.float32)
        if self.x is None:
            self.x, self.v = target.clone(), torch.zeros_like(target)
            return self.x
        n = 4
        for _ in range(n):
            gap = target - self.x
            if self.angle:
                gap = torch.remainder(gap + math.pi, 2 * math.pi) - math.pi
            self.v = self.v + (self.w * self.w * gap - 2.0 * self.z * self.w * self.v) * (dt / n)
            self.x = self.x + self.v * (dt / n)
        return self.x


class Motion:
    """The hand's dynamics: the point of the tool is put exactly where it must be; the back of the hand gives a
    little and comes back, the hand turns about the point and the forearm follows the wrist, all on damped
    springs, so they lag and overshoot a little as a real arm does."""

    def __init__(self, fps: float, unit: float, shoulder: tuple) -> None:
        self.dt, self.unit, self.shoulder = 1.0 / fps, unit, shoulder
        self.yaw = Spring(1.5, 0.5, angle=True)
        self.give = Spring(2.6, 0.42)
        self.arm = Spring(1.1, 0.65, angle=True)
        self.last: tuple | None = None

    def __call__(self, tip: tuple, lift: float) -> Pose:
        u = self.unit
        vx, vy = ((tip[0] - self.last[0]) / self.dt, (tip[1] - self.last[1]) / self.dt) if self.last else (0.0, 0.0)
        self.last = tip
        towards = math.atan2(tip[1] - self.shoulder[1], tip[0] - self.shoulder[0])
        fx, fy = math.cos(towards), math.sin(towards)
        speed = math.hypot(vx, vy)
        side = (vx * fy - vy * fx) / (speed + 1.5 * u)       # moving across the hand turns the wrist with it
        yaw = float(self.yaw(towards + 0.19 + 0.16 * side, self.dt))
        lag = 0.012
        gx, gy = -vx * lag, -vy * lag
        size = math.hypot(gx, gy)
        if size > 0.05 * u:
            gx, gy = gx * 0.05 * u / size, gy * 0.05 * u / size
        give = self.give((gx, gy), self.dt)
        fx, fy = math.cos(yaw), math.sin(yaw)
        wrist = (tip[0] + u * (-0.96 * fx - 0.085 * fy), tip[1] + u * (-0.96 * fy + 0.085 * fx))
        to_shoulder = math.atan2(self.shoulder[1] - wrist[1], self.shoulder[0] - wrist[0])
        arm = float(self.arm(to_shoulder, self.dt))
        back = yaw + math.pi
        bend = (arm - back + math.pi) % (2 * math.pi) - math.pi
        arm = back + max(-0.55, min(0.55, bend))
        return Pose(tip, lift, yaw, (float(give[0]), float(give[1])), arm)


# ------------------------------------------------------------------------------------------------ the studio --
@dataclass
class Shot:
    """One picture of the making-of: ``picture (H, W, 3)`` bytes, its ``time`` in seconds, the frame being drawn
    (``frame``, -1 for the background, None while the animation plays), where the point of the tool is
    (``tip``, pixels, None with no hand on the picture), what the hand does (``doing``) and holds."""

    picture: torch.Tensor
    time: float
    frame: int | None
    tip: tuple | None
    doing: str
    holding: tuple | None = None


@dataclass
class Phase:
    """The lines drawn on one sheet (``frame`` -1: the background), in about ``budget`` seconds."""

    lines: list
    frame: int
    budget: float
    steps: list = field(default_factory=list)
    ends: list = field(default_factory=list)          # when each line is finished


class Studio:
    """Everything that stays the same through a making-of, and the pictures of it (see :func:`shots`)."""

    def __init__(self, animation, width: int = 1280, fps: float = 24, first: float = 14.0, later: float = 3.0,
                 onion: float = 0.16, hand: bool = True, skin=None) -> None:
        self.anim, self.fps, self.first, self.later, self.onion = animation, float(fps), first, later, onion
        self.hand, self.skin = hand, tuple(skin) if skin is not None else SKIN_TONE
        self.lay = lay = layout(width, animation.size)
        self.background = list(getattr(animation, "background", None) or [])
        holds = getattr(animation, "holds", None)
        on = 1 if holds is not None else max(1, int(animation.settings.get("on", 1)))
        self.holds = [max(1, int(x)) for x in holds] if holds is not None else [on] * len(animation)
        self.rate = float(animation.fps) * on               # exposure frames a second
        x0, y0, self.bw, self.bh = lay.board
        self.origin = (x0, y0)
        self.shoulder = (lay.width * 1.02, lay.height * 1.9)
        self.fonts: dict = {}
        self.cel_tables: dict = {}
        self.thumbs: dict = {}
        self.played: dict = {}
        self.live = None                                     # the thumbnail of the frame being drawn
        self.board_px: torch.Tensor | None = None            # the board as shown, bytes (bh, bw, 3)
        self.missing = "nothing yet"
        self.items, self.pots = self._things()
        self.tray = Tray(self.items, lay, self.pots)
        self.where = {thing.key: thing.at for thing in self.items + self.pots}
        self.base = self._chrome()
        self.motion = Motion(fps, lay.unit, self.shoulder)

    # ----------------------------------------------------------------------------------- the still parts --
    def font(self, size: float, mono: bool = False):
        key = (round(size), mono)
        if key not in self.fonts:
            self.fonts[key] = _font(size, mono)
        return self.fonts[key]

    def _things(self) -> tuple:
        """The tools in the tray in the order they are first used, and the pots of cel paint."""
        items, pots, seen = [], [], set()
        for line in self.background + [line for sheet in self.anim.sheets for line in sheet.lines]:
            holding = held(line)
            key = item_key(holding)
            if key not in seen:
                seen.add(key)
                tool = "marker" if key[0] == "marker" else holding[0]
                items.append(Item(key, tool, holding[1]))
            if holding[0] == "brush" and ("pot", holding[1]) not in seen:
                seen.add(("pot", holding[1]))
                pots.append(Item(("pot", holding[1]), "pot", holding[1]))
        return items, pots

    def _chrome(self) -> torch.Tensor:
        """The picture with nothing on the board: the desk, the board's edge, the tray (bytes)."""
        lay = self.lay
        W, H = lay.width, lay.height
        desk = (_lin(DESK)[:, None, None] * _noise(H, W, 0.03)).expand(3, H, W).clone()
        x0, y0 = self.origin
        bw, bh, r = self.bw, self.bh, lay.rim
        if self.anim.surface == "paper":                       # a sheet lying on the desk: a hairline edge
            desk[:, y0 + 1: y0 + bh + 2, x0 + 1: x0 + bw + 2] *= 0.86
        else:                                                  # the aluminium frame of a whiteboard
            desk[:, y0 - r + 1: y0 + bh + r + 2, x0 - r + 1: x0 + bw + r + 2] *= 0.84
            desk[:, y0 - r: y0 + bh + r, x0 - r: x0 + bw + r] = _lin(ALUMINIUM)[:, None, None]
            if r >= 3:
                desk[:, y0 - r, x0 - r: x0 + bw + r] = _lin((0.90, 0.905, 0.91))[:, None]
                desk[:, y0 - r: y0 + bh + r, x0 - r] = _lin((0.88, 0.885, 0.89))[:, None]
                desk[:, y0 + bh + r - 1, x0 - r: x0 + bw + r] = _lin((0.60, 0.61, 0.62))[:, None]
                desk[:, y0 - r: y0 + bh + r, x0 + bw + r - 1] = _lin((0.62, 0.63, 0.64))[:, None]
            desk[:, y0 - 1, x0 - 1: x0 + bw + 1] = _lin((0.58, 0.59, 0.60))[:, None]
            desk[:, y0 - 1: y0 + bh + 1, x0 - 1] = _lin((0.58, 0.59, 0.60))[:, None]
        tx0, ty0, tx1, ty1 = self.tray.box
        desk[:, ty0 + 1: ty1 + 2, tx0 + 1: tx1 + 2] *= 0.88    # the tray stands a little off the desk
        out = _to_u8(desk)
        out[: lay.bar] = torch.tensor(BAR, dtype=torch.uint8)
        return out

    # ----------------------------------------------------------------------------------- the board --
    def _board(self, frame: int) -> Board:
        return Board(self.bw, self.bh, self.anim.surface, seed=max(0, frame))

    def _box(self, line: Line) -> tuple | None:
        """The part of the board a line can change: its points, its nib and the waver of the hand."""
        pts = line.points * float(self.bw)
        pad = 0.55 * TOOLS.get(line.tool, TOOLS["pen"]).width * self.bw * 1.6 + 5.0
        x0, y0 = int(float(pts[:, 0].min()) - pad), int(float(pts[:, 1].min()) - pad)
        x1, y1 = int(float(pts[:, 0].max()) + pad) + 2, int(float(pts[:, 1].max()) + pad) + 2
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(self.bw, x1), min(self.bh, y1)
        return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1, y1)

    def _show(self, canvas: torch.Tensor, box=None) -> None:
        """Put a board canvas (linear), or the part of it in ``box``, on the picture."""
        if box is None:
            self.board_px = _to_u8(canvas)
            return
        x0, y0, x1, y1 = box
        self.board_px[y0:y1, x0:x1] = _to_u8(canvas[:, y0:y1, x0:x1])

    def _clean(self, lines: list, frame: int, start: torch.Tensor | None) -> torch.Tensor:
        """A sheet drawn again without the lightbox under it, the way the animation shows it."""
        board = self._board(frame)
        if start is not None:
            board.canvas = start.clone()
        for line in lines:
            board.draw(line)
        return board.canvas

    # ----------------------------------------------------------------------------------- the plan --
    def _between(self, before, want) -> str:
        """What the hand does to go from holding ``before`` to ``want``: nothing, turn a twin marker round,
        dip the brush in another pot, or go to the tray."""
        if before is None or want == before:
            return "hop"
        if item_key(want) == item_key(before):
            return "dip" if want[0] == "brush" else "turn"
        return "tray"

    def _plan(self, phase: Phase, t: float, here: tuple, holding) -> tuple:
        """The steps that draw a phase from time ``t``, the hand at ``here`` holding ``holding``. Returns the
        time, place and tool at its end."""
        lines, steps = phase.lines, phase.steps
        u, scale = self.lay.unit, float(self.bw)
        raw, shapes = [], []
        for line, (begin, times) in zip(lines, timing(lines), strict=True):
            if line.kind in CELS:                            # a cel is painted in rows, its time is its area
                raw.append(0.12 + _cel_area(line) / (3.0 * TOOLS["brush"].width) / TOOLS["brush"].speed)
                shapes.append(None)
            else:
                d = float(times[-1] - begin)
                # the time-lapse is cut slower for the lines than for the colouring: lines are what one
                # wants to see drawn
                raw.append(max(d, 1e-3) * (INK_WEIGHT if line.kind in ("line", "sketch") else 1.0))
                shapes.append((times - begin) / d if d > 0 else torch.linspace(0.0, 1.0, len(times)))
        starts = [self._at(line, 0.0) for line in lines]
        ends = [self._at(line, 1.0) for line in lines]
        kinds, before = [], holding
        for line in lines:
            kinds.append(self._between(before, held(line)))
            before = held(line)
        hops = [0.05 + math.dist(here if i == 0 else ends[i - 1], starts[i]) / scale / 1.8 if k == "hop" else 0.0
                for i, k in enumerate(kinds)]
        trips, turns = kinds.count("tray") + kinds.count("dip"), kinds.count("turn")
        trip = min(0.7, 0.3 * phase.budget / trips) if trips else 0.0
        turn = min(0.35, 0.08 * phase.budget / turns) if turns else 0.0
        rest = max(0.25 * phase.budget, phase.budget - trips * trip - turns * turn)
        factor = rest / max(1e-6, sum(raw) + sum(hops))
        for i, line in enumerate(lines):
            a, want = (here if i == 0 else ends[i - 1]), held(line)
            if kinds[i] == "hop":
                d = hops[i] * factor
                if math.dist(a, starts[i]) > 0.5:
                    lift = min(0.2 * u, 0.03 * u + 0.3 * math.dist(a, starts[i]))
                    steps.append(Step("move", t, t + d, a, starts[i], lift))
                t += d
            elif kinds[i] == "turn":
                steps.append(Step("move", t, t + turn, a, starts[i], 0.12 * u, swap=want))
                t += turn
            else:
                spot = self.where[("pot", want[1])] if kinds[i] == "dip" else self.where[item_key(want)]
                if kinds[i] == "tray":
                    spot = (spot[0] + 0.08 * u, spot[1])
                steps.append(Step("move", t, t + 0.4 * trip, a, spot, 0.24 * u))
                steps.append(Step("dwell", t + 0.4 * trip, t + 0.6 * trip, spot, spot, 0.04 * u, swap=want))
                steps.append(Step("move", t + 0.6 * trip, t + trip, spot, starts[i], 0.24 * u))
                t += trip
            d = raw[i] * factor
            steps.append(Step("draw", t, t + d, starts[i], ends[i], 0.0, i, shapes[i]))
            phase.ends.append(t + d)
            t += d
        return t, ends[-1], held(lines[-1])

    def plan(self) -> tuple:
        """Every phase with its steps, the tool held from each time on, and when it all ends."""
        u, lay = self.lay.unit, self.lay
        phases = [Phase(self.background, -1, 0.32 * self.first)] if self.background else []
        phases += [Phase(sheet.lines, i, self.first if i == 0 else self.later)
                   for i, sheet in enumerate(self.anim.sheets)]
        away = (lay.width + 0.3 * u, lay.height + 0.2 * u)  # off the picture, toward the shoulder
        first = next((p.lines[0] for p in phases if p.lines), None)
        holding = held(first) if first is not None else ("pen", TOOLS["pen"].ink)
        swaps = [(0.0, holding)]
        t, here = 0.0, away
        for phase in phases:
            if phase.lines:
                if here == away:                              # the hand comes in
                    start = self._at(phase.lines[0], 0.0)
                    phase.steps.append(Step("move", t, t + 0.8, away, start, 0.3 * u))
                    t, here = t + 0.8, start
                t, end, holding = self._plan(phase, t, here, holding)
                here = (end[0] + 0.12 * u, end[1] + 0.10 * u)
                phase.steps.append(Step("move", t, t + 0.45, end, here, 0.12 * u))
            else:
                phase.steps.append(Step("dwell", t, t + 0.45, here, here, 0.12 * u))
            t += 0.45
            swaps += [((s.t0 + s.t1) / 2.0, s.swap) for s in phase.steps if s.swap is not None]
        leave = Step("move", t, t + 0.7, here, away, 0.3 * u)
        return phases, swaps, leave

    def _at(self, line: Line, share: float) -> tuple:
        """Where the point of the tool is (picture pixels) ``share`` (0..1, of its points) through ``line``."""
        x0, y0 = self.origin
        if line.kind in CELS and line.rings is not None:
            p = self._cel_point(line, share)
            return (x0 + p[0], y0 + p[1])
        pts, scale = line.points, float(self.bw)
        j = min(max(share, 0.0), 1.0) * (len(pts) - 1)
        k = min(int(j), max(0, len(pts) - 2))
        f = j - k if len(pts) > 1 else 0.0
        p = pts[k] * (1 - f) + pts[min(k + 1, len(pts) - 1)] * f
        return (x0 + float(p[0]) * scale, y0 + float(p[1]) * scale)

    def _cel_point(self, line: Line, share: float) -> tuple:
        """Where the brush is while a cel shape is painted: back and forth along the edge of the paint, which
        moves across the shape the way Board.peek shows it."""
        if line.uid not in self.cel_tables:
            self.cel_tables[line.uid] = self._cel_table(line)
        lo, hi, passes = self.cel_tables[line.uid]
        n = len(lo) - 1
        j = min(max(share, 0.0), 1.0) * n
        k = min(int(j), n - 1)
        f = j - k
        a, b = lo[k] * (1 - f) + lo[k + 1] * f, hi[k] * (1 - f) + hi[k + 1] * f
        sweep = share * passes
        q = sweep - math.floor(sweep)
        q = _ease(q if int(sweep) % 2 == 0 else 1.0 - q)
        p = a + (b - a) * q
        return (float(p[0]), float(p[1]))

    def _cel_table(self, line: Line) -> tuple:
        """For 25 steps of the paint's edge across a cel shape, where the edge enters and leaves the shape."""
        scale = float(self.bw)
        pts = line.points * scale
        x0, y0 = max(0, math.floor(float(pts[:, 0].min()))), max(0, math.floor(float(pts[:, 1].min())))
        w = max(1, min(self.bw, math.ceil(float(pts[:, 0].max())) + 1) - x0)
        h = max(1, min(self.bh, math.ceil(float(pts[:, 1].max())) + 1) - y0)
        down = max(1.0, max(w, h) / 64.0)
        mh, mw = math.ceil(h / down) + 1, math.ceil(w / down) + 1
        mask = torch.zeros(mh, mw)
        got = fill_rings((pts - torch.tensor([float(x0), float(y0)])) / down, line.rings, mh, mw, ss=1)
        if got is not None:
            my0, mx0, cov = got
            mask[my0: my0 + cov.shape[0], mx0: mx0 + cov.shape[1]] = cov
        span = 0.77 * w + 0.64 * h
        lo, hi = [], []
        here = pts.mean(0)
        xs = torch.linspace(0.0, float(w), 48)
        for s in torch.linspace(0.0, 1.0, 25).tolist():
            ys = (s * span - 0.77 * xs) / 0.64                # the edge: 0.77 x + 0.64 y = s * span in its box
            ok = (ys >= 0) & (ys <= h)
            ok &= mask[(ys / down).round().long().clamp(0, mh - 1), (xs / down).round().long().clamp(0, mw - 1)] > 0.5
            if bool(ok.any()):
                idx = ok.nonzero().flatten()
                a = torch.tensor([float(xs[idx[0]]) + x0, float(ys[idx[0]]) + y0])
                b = torch.tensor([float(xs[idx[-1]]) + x0, float(ys[idx[-1]]) + y0])
                here = (a + b) / 2
            else:
                a = b = here
            lo.append(a)
            hi.append(b)
        passes = max(1.0, span / (2.5 * TOOLS["brush"].width * scale))
        return torch.stack(lo), torch.stack(hi), passes

    # ----------------------------------------------------------------------------------- the strip --
    def _cells(self) -> list:
        """The cells of the exposure sheet: (key, its number on the sheet, the frame, held from the cell
        before)."""
        cells = [("bg", "BG", -1, False)] if self.background else []
        number = 1
        for i, n in enumerate(self.holds):
            for j in range(n):
                cells.append((i, str(number), i, j > 0))
                number += 1
        return cells

    def _bar(self, current: int | None, done: int, total: int, playing: bool = False) -> torch.Tensor:
        """The strip along the top (bytes ``(bar, W, 3)``): the frame counter, the exposure sheet, the readout
        of frames a second, exposure and lines drawn."""
        import numpy as np
        from PIL import Image, ImageDraw

        lay = self.lay
        g, W, m = lay.grid, lay.width, lay.margin
        image = Image.new("RGB", (W, lay.bar), BAR)
        draw = ImageDraw.Draw(image)
        text = g >= 4.5                                       # too small to read: the cells only
        left_w, right_w = round(17 * g), round(22 * g)
        if text:
            label, big, small, tiny = (self.font(1.3 * g), self.font(3.4 * g, True), self.font(1.9 * g, True),
                                       self.font(1.25 * g, True))
            number = "BG" if current == -1 else f"{(current or 0) + 1:02d}"
            draw.text((m, round(2.0 * g)), "BACKGROUND" if current == -1 else "FRAME", font=label, fill=DIM)
            draw.text((m, round(4.0 * g)), number, font=big, fill=TEXT)
            width = draw.textlength(number, font=big)
            draw.text((m + width + round(0.8 * g), round(5.3 * g)), f"/ {len(self.anim):02d}", font=small, fill=DIM)
            for x in (left_w, W - right_w):
                draw.line([(x, round(2 * g)), (x, lay.bar - round(2 * g))], fill=RULE, width=1)
            hold = self.holds[current] if current is not None and current >= 0 else self.holds[0]
            x = W - right_w + round(2 * g)
            for name, value, step in (("FPS", f"{self.rate:g}", 5), ("ON", f"{hold}s", 5),
                                      ("LOOP" if playing else "LINE", f"{done}/{total}", 0)):
                draw.text((x, round(2.0 * g)), name, font=label, fill=DIM)
                draw.text((x, round(4.7 * g)), value, font=small, fill=TEXT)
                x += round(step * g)
        cw, ch = lay.cell
        gap = max(1, round(0.5 * g))
        slot = cw + gap
        x_start = left_w + round(2 * g) if text else m
        x_end = W - right_w - round(2 * g) if text else W - m
        cells = self._cells()
        fit = max(1, (x_end - x_start + gap) // slot)
        here = next((k for k, c in enumerate(cells) if c[2] == current and not c[3]), 0)
        first = 0 if len(cells) <= fit else max(0, min(here - fit // 3, len(cells) - fit))
        top = round(4.0 * g) if text else max(1, round(0.15 * lay.bar))
        for k in range(first, min(len(cells), first + fit)):
            key, name, frame, on = cells[k]
            x = x_start + (k - first) * slot
            mine = frame == current
            if text and (not on or slot >= 3.5 * g):
                draw.text((x, round(1.9 * g)), name, font=tiny, fill=ACCENT if mine and not on else DIM)
            if on:                                            # held: a line through the cell, as on an x-sheet
                mid = top + ch // 2
                draw.line([(x - gap, mid), (x + cw - 1, mid)], fill=DIM, width=1)
                if k + 1 >= len(cells) or not cells[k + 1][3]:
                    draw.line([(x + cw - 1, mid - max(1, round(0.6 * g))), (x + cw - 1, mid + max(1, round(0.6 * g)))],
                              fill=DIM, width=1)
                continue
            thumb = self.live if mine and self.live is not None else self.thumbs.get(key)
            if thumb is not None:
                image.paste(thumb, (x, top))
            else:
                draw.rectangle([x, top, x + cw - 1, top + ch - 1], fill=EMPTY, outline=RULE)
            if mine:
                draw.rectangle([x - 1, top - 1, x + cw, top + ch], outline=ACCENT, width=max(1, round(0.25 * g)))
        return torch.from_numpy(np.asarray(image).copy())

    def _thumb(self, canvas: torch.Tensor):
        from PIL import Image

        cw, ch = self.lay.cell
        small = F.interpolate(linear_to_srgb(canvas)[None], size=(ch, cw), mode="area")[0]
        return Image.fromarray(_u8(small).numpy())

    # ----------------------------------------------------------------------------------- one picture --
    def picture(self, live: torch.Tensor | None, box, tip, lift: float, holding, bar: torch.Tensor) -> torch.Tensor:
        """The whole picture (bytes): the desk, the board with the stroke being drawn (``live`` in ``box``), the
        tray without the tool in hand, the strip, the hand."""
        lay = self.lay
        missing = item_key(holding) if tip is not None else None
        if missing != self.missing:
            tx0, ty0, tx1, ty1 = self.tray.box
            self.base[ty0:ty1, tx0:tx1] = _to_u8(self.tray.picture(missing))
            self.missing = missing
        out = self.base.clone()
        out[: lay.bar] = bar
        x0, y0 = self.origin
        out[y0: y0 + self.bh, x0: x0 + self.bw] = self.board_px
        if live is not None and box is not None:
            bx0, by0, bx1, by1 = box
            out[y0 + by0: y0 + by1, x0 + bx0: x0 + bx1] = _to_u8(live[:, by0:by1, bx0:bx1])
        if tip is not None and holding is not None:
            self._hand(out, tip, lift, holding)
        return out

    def _hand(self, out: torch.Tensor, tip: tuple, lift: float, holding) -> None:
        """The hand (or the tool alone) and its shadow, rendered in its own box only, onto ``out``."""
        lay = self.lay
        u, W, H = lay.unit, lay.width, lay.height
        pose = self.motion(tip, lift)
        fx, fy = math.cos(pose.yaw), math.sin(pose.yaw)
        wrist = (tip[0] - 0.96 * u * fx, tip[1] - 0.96 * u * fy)
        ax, ay = math.cos(pose.arm), math.sin(pose.arm)
        reach = 9.0
        for d in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 9.0):    # the forearm goes on off the picture
            ex, ey = wrist[0] + ax * d * u, wrist[1] + ay * d * u
            if ex < -0.4 * u or ex > W + 0.4 * u or ey < -0.4 * u or ey > H + 0.4 * u:
                reach = d
                break
        shapes = hand_shapes(pose, u, holding[0], holding[1], self.skin, reach, self.hand)
        boxes = [_box(s) for s in shapes]
        bx0, by0 = max(0, int(min(b[0] for b in boxes))), max(lay.bar, int(min(b[1] for b in boxes)))
        bx1, by1 = min(W, int(max(b[2] for b in boxes)) + 1), min(H, int(max(b[3] for b in boxes)) + 1)
        if bx1 <= bx0 or by1 <= by0:
            return
        g = render(shapes, bx0, by0, bx1 - bx0, by1 - by0)
        lit = shade(g, u, lift)
        pad = round(0.9 * u)
        shadow = cast_shadow(g["cover"], g["height"], u, pad)
        cx0, cy0, cx1, cy1 = max(0, bx0 - pad), max(lay.bar, by0 - pad), min(W, bx1 + pad), min(H, by1 + pad)
        sx, sy = cx0 - (bx0 - pad), cy0 - (by0 - pad)
        region = _from_u8(out[cy0:cy1, cx0:cx1])
        region = region * (1.0 - shadow[sy: sy + cy1 - cy0, sx: sx + cx1 - cx0])
        c = g["cover"]
        iy, ix = by0 - cy0, bx0 - cx0
        inner = region[:, iy: iy + c.shape[0], ix: ix + c.shape[1]]
        region[:, iy: iy + c.shape[0], ix: ix + c.shape[1]] = inner * (1.0 - c) + lit * c
        out[cy0:cy1, cx0:cx1] = _to_u8(region)

    # ----------------------------------------------------------------------------------- all of it --
    def _where_now(self, step: Step, share: float) -> tuple:
        """Where the point of the tool is, and how high, during a move or a dwell."""
        if step.kind == "dwell":
            return step.a, step.lift * (1.0 - math.sin(math.pi * share))
        e = _ease(share)
        (ax, ay), (bx, by) = step.a, step.b
        bow = 0.10 * math.sin(math.pi * e)                    # a hand travels in a slight arc
        return (ax + (bx - ax) * e - (by - ay) * bow, ay + (by - ay) * e + (bx - ax) * bow), \
            step.lift * math.sin(math.pi * e)

    def _drawing(self, board: Board, line: Line, step: Step, share: float, dense: dict) -> tuple:
        """A line part drawn: where the point is, and the board with that much of it (``Board.peek``)."""
        if line.kind in CELS or step.times is None:
            upto = max(2, min(len(line), round(share * len(line))))
            return self._at(line, share), board.peek(line, upto) if len(line) >= 2 else None
        if len(line) < 2:
            return self._at(line, 0.0), None
        times = step.times
        i = int(torch.searchsorted(times, torch.tensor([share]))[0].clamp(1, len(line) - 1))
        f = (share - float(times[i - 1])) / max(1e-6, float(times[i] - times[i - 1]))
        along = (i - 1 + min(max(f, 0.0), 1.0)) / (len(line) - 1)
        if line.uid not in dense:                             # points every 2 px, so the ink keeps up with the pen
            dense[line.uid] = _densify(line, float(self.bw))
        fine, where = dense[line.uid]
        j = along * (len(line) - 1)
        k = min(int(j), len(line) - 2)
        at = float(where[k]) + (j - k) * float(where[k + 1] - where[k])
        return self._at(line, along), board.peek(fine, max(2, min(len(fine), int(at) + 1)))

    def shots(self, loops: int = 2):
        """Every picture of the making-of in order (:class:`Shot`)."""
        phases, swaps, leave = self.plan()
        swap_times = [s[0] for s in swaps]
        fps, k = self.fps, 0

        def holding_at(time: float):
            return swaps[max(0, bisect.bisect_right(swap_times, time) - 1)][1]

        bg_canvas, last_clean = None, None
        for phase in phases:
            frame, lines = phase.frame, phase.lines
            board = self._board(frame)
            if bg_canvas is not None:
                board.canvas = bg_canvas.clone()
            start = board.canvas.clone()
            if frame >= 0 and last_clean is not None and self.onion > 0:     # the lightbox
                board.canvas = board.canvas * (1.0 - self.onion) + last_clean * self.onion
            self._show(board.canvas)
            self.live, last_live = None, -1e9
            starts = [s.t0 for s in phase.steps]
            done, dense = 0, {}
            while (k + 1) / fps <= phase.steps[-1].t1 + 1e-9:
                k += 1
                time = k / fps
                while done < len(lines) and phase.ends[done] <= time:          # finished lines, drawn for good
                    box = self._box(lines[done])
                    board.draw(lines[done])
                    if box is not None:
                        self._show(board.canvas, box)
                    done += 1
                step = phase.steps[max(0, bisect.bisect_right(starts, time) - 1)]
                share = min(1.0, max(0.0, (time - step.t0) / max(1e-6, step.t1 - step.t0)))
                live, box = None, None
                if step.kind == "draw" and step.index == done and done < len(lines):
                    tip, live = self._drawing(board, lines[done], step, share, dense)
                    box, lift, doing = self._box(lines[done]), 0.0, "drawing"
                elif step.kind == "draw":
                    tip, lift, doing = step.b, 0.0, "drawing"
                else:
                    (tip, lift), doing = self._where_now(step, share), \
                        "changing tools" if step.kind == "dwell" or step.swap else "moving"
                if time - last_live >= 0.5:                   # the frame's cell fills in as it is drawn
                    self.live, last_live = self._thumb(live if live is not None else board.canvas), time
                holding = holding_at(time)
                bar = self._bar(frame, done, len(lines))
                yield Shot(self.picture(live, box, tip, lift, holding, bar), time, frame, tip, doing, holding)
            clean = self._clean(lines, frame, start)
            self.thumbs["bg" if frame < 0 else frame] = self._thumb(clean)
            if frame < 0:
                bg_canvas = clean
            else:
                last_clean = clean
                self.played[frame] = _to_u8(clean)
        self.live = None
        last = len(self.anim) - 1
        bar = self._bar(last, len(self.anim.sheets[last].lines), len(self.anim.sheets[last].lines))
        while (k + 1) / fps <= leave.t1 + 1e-9:               # the hand goes
            k += 1
            time = k / fps
            share = min(1.0, max(0.0, (time - leave.t0) / (leave.t1 - leave.t0)))
            tip, lift = self._where_now(leave, share)
            holding = holding_at(time)
            yield Shot(self.picture(None, None, tip, lift, holding, bar), time, last, tip, "leaving", holding)
        clock = 0.0
        for loop in range(loops):                             # and the animation plays
            for i in range(len(self.anim)):
                self.board_px = self.played[i]
                n = round((clock + self.holds[i] / self.rate) * fps) - round(clock * fps)
                clock += self.holds[i] / self.rate
                bar = self._bar(i, loop + 1, loops, playing=True)
                for _ in range(max(1, n)):
                    k += 1
                    yield Shot(self.picture(None, None, None, 0.0, None, bar), k / fps, None, None, "playing")


def _densify(line: Line, scale: float, most: float = 2.0) -> tuple:
    """``line`` with points every ``most`` pixels at most, and for each of its own points its number among
    them."""
    pts, press = line.points, line.pressure
    n = ((pts[1:] - pts[:-1]).norm(dim=-1) * scale / most).ceil().clamp(min=1).long()
    steps = torch.repeat_interleave(torch.arange(len(n)), n)
    frac = torch.cat([torch.arange(int(c)) / int(c) for c in n.tolist()])[:, None]
    dense = torch.cat([pts[:-1][steps] + (pts[1:] - pts[:-1])[steps] * frac, pts[-1:]])
    dp = torch.cat([press[:-1][steps] + (press[1:] - press[:-1])[steps] * frac[:, 0], press[-1:]])
    where = torch.cat([torch.zeros(1, dtype=torch.long), n.cumsum(0)])
    return Line(dense, dp, line.tool, line.colour, line.marker, line.kind, line.age, line.uid), where


def shots(animation, width: int = 1280, fps: int = 24, first: float = 14.0, later: float = 3.0, loops: int = 2,
          onion: float = 0.16, hand: bool = True, skin=None):
    """The pictures of the making-of one by one (:class:`Shot`), as :func:`making_of` writes them."""
    yield from Studio(animation, width, fps, first, later, onion, hand, skin).shots(loops)


def making_of(animation, path, width: int = 1280, fps: int = 24, first: float = 14.0, later: float = 3.0,
              loops: int = 2, onion: float = 0.16, hand: bool = True, skin=None) -> Path:
    """A video of ``animation`` being drawn by hand in the studio (see the module's notes): a background, if it
    has one, drawn first, then the first frame in about ``first`` seconds and every next one in about ``later``
    (lines traced from the frame before go faster), each over the frame before at ``onion`` strength; then the
    finished animation played ``loops`` times. ``hand`` False shows the tool alone; ``skin`` is the colour of
    the hand (sRGB). Written with :class:`helix_lm.paint.animate.VideoWriter`: ``.mp4`` (ffmpeg) or ``.gif``."""
    writer = VideoWriter(path, fps)
    for shot in shots(animation, width, fps, first, later, loops, onion, hand, skin):
        writer.add(shot.picture.permute(2, 0, 1).float() / 255.0)
    return writer.close()


__all__ = ["Blob", "Cone", "Item", "Layout", "Motion", "Phase", "Pose", "Shot", "Spring", "Step", "Studio", "Tray",
           "cast_shadow", "hand_shapes", "held", "item_key", "layout", "making_of", "render", "shade", "shots",
           "tool_parts", "tool_shapes"]
