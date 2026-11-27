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
    board ``(h, w)``, normal ``(3, h, w)``, colour (linear RGB, unlit) ``(3, h, w)`` and material numbers."""
    height = torch.full((h, w), -1e9)
    normal = torch.zeros(3, h, w)
    normal[2] = 1.0
    albedo = torch.zeros(3, h, w)
    params = torch.zeros(4, h, w)
    group = torch.full((h, w), -1, dtype=torch.long)
    cover = torch.zeros(h, w)
    for s in shapes:
        bx0, by0, bx1, by1 = _box(s, aa)
        ix0, iy0 = max(x0, int(math.floor(bx0))), max(y0, int(math.floor(by0)))
        ix1, iy1 = min(x0 + w, int(math.ceil(bx1)) + 1), min(y0 + h, int(math.ceil(by1)) + 1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        px = torch.arange(ix0, ix1, dtype=torch.float32)[None, :] + 0.5
        py = torch.arange(iy0, iy1, dtype=torch.float32)[:, None] + 0.5
        tau = None
        if isinstance(s, Cone):
            cov, top, nrm, tau = _cone(s, px, py, aa)
        else:
            cov, top, nrm = _blob(s, px, py, aa)
        if not bool((cov > 0).any()):
            continue
        sl = (slice(iy0 - y0, iy1 - y0), slice(ix0 - x0, ix1 - x0))
        colour = s.colour[:, None, None].expand(3, *cov.shape).clone()
        mat = MATERIALS[s.material][:, None, None].expand(4, *cov.shape).clone()
        if isinstance(s, Cone) and (s.nail is not None or s.creases > 0):
            _decals(s, tau, nrm, top, colour, mat)
        old_h, old_g = height[sl], group[sl]
        empty, has = old_h < -1e8, cov > 0
        same = (old_g == s.group) & (s.k > 0) & ~empty
        k = max(s.k, 1e-3)
        blend = (0.5 + 0.5 * (top - old_h) / k).clamp(0.0, 1.0)
        mix = torch.where(same, blend, (top > old_h).float() * cov)       # another group: on top or under
        mix = torch.where(empty, torch.ones_like(mix), mix) * has
        new_h = torch.where(same, old_h + (top - old_h) * blend + k * blend * (1.0 - blend),
                            torch.maximum(top, old_h))
        height[sl] = torch.where(has, torch.where(empty, top, new_h), old_h)
        n = normal[(slice(None), *sl)] * (1.0 - mix) + nrm * mix
        normal[(slice(None), *sl)] = n / n.norm(dim=0, keepdim=True).clamp(min=1e-6)
        albedo[(slice(None), *sl)] = albedo[(slice(None), *sl)] * (1.0 - mix) + colour * mix
        params[(slice(None), *sl)] = params[(slice(None), *sl)] * (1.0 - mix) + mat * mix
        group[sl] = torch.where(mix > 0.5, torch.full_like(old_g, s.group), old_g)
        cover[sl] = torch.maximum(cover[sl], cov)
    return {"cover": cover, "height": height.clamp(min=0.0), "normal": normal, "albedo": albedo, "params": params}


def _decals(s: Cone, tau: torch.Tensor, nrm: torch.Tensor, top: torch.Tensor, colour: torch.Tensor,
            mat: torch.Tensor) -> None:
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
    to shadow on skin, light from the room, a soft sheen, darker where the shapes crowd each other."""
    n, albedo, p = g["normal"], g["albedo"], g["params"]
    wrap, scatter, shine, sharp = p[0], p[1], p[2], p[3].clamp(min=1.0)
    height = g["height"] - lift
    near = gaussian_blur(height, max(1.0, 0.05 * unit))
    crowd = (1.0 - 2.6 * (near - height).clamp(min=0.0) / unit).clamp(0.4, 1.0)
    ndl = (n * LIGHT[:, None, None]).sum(0)
    diffuse = ((ndl + wrap) / (1.0 + wrap)).clamp(min=0.0) * (1.0 - 0.8 * _self_shadow(height, unit))
    band = _smooth(0.0, 0.32, diffuse) * _smooth(0.64, 0.32, diffuse) * scatter
    red = torch.tensor([0.22, 0.05, 0.015])[:, None, None] * band
    ndh = (n * HALF[:, None, None]).sum(0).clamp(min=0.0)
    gloss = shine * ndh ** sharp * crowd
    sun = torch.tensor(SUN)[:, None, None]
    sky = torch.tensor(SKY)[:, None, None] * (0.55 + 0.45 * n[2]) * crowd
    metal = (p[2] > 0.7).float()                                       # metal shows the room more than lamp
    out = albedo * (sun * diffuse * (0.9 + 0.1 * crowd) + sky) + albedo * red + sun * gloss
    out = out + metal * 0.25 * albedo * (0.5 + 0.5 * n[1].neg()) * crowd
    return out


def _self_shadow(height: torch.Tensor, unit: float) -> torch.Tensor:
    """Where a part of the shapes is in the shadow of another (the pen on a finger, a finger on the back of the
    hand): looked for by walking from each pixel toward the light over the heights, 0..1 ``(h, w)``."""
    h, w = height.shape
    away = LIGHT[:2] / LIGHT[:2].norm()
    rise = float(LIGHT[2] / LIGHT[:2].norm())
    soft = 0.012 * unit + 0.5
    worst = torch.zeros_like(height)
    step = 1.5
    while step < 0.4 * unit:
        dx, dy = round(float(away[0]) * step), round(float(away[1]) * step)
        if abs(dx) < w and abs(dy) < h:
            moved = torch.zeros_like(height)
            ys, yd = (slice(max(0, dy), h + min(0, dy)), slice(max(0, -dy), h + min(0, -dy)))
            xs, xd = (slice(max(0, dx), w + min(0, dx)), slice(max(0, -dx), w + min(0, -dx)))
            moved[yd, xd] = height[ys, xs]
            reach = math.hypot(dx, dy)
            worst = torch.maximum(worst, ((moved - reach * rise - height) / soft).clamp(0.0, 1.0))
        step *= 1.45
    return gaussian_blur(worst, 0.6 + 0.004 * unit)


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
PEN_BODY, INK, CAP, STEEL, FELT, BRISTLE, PAINT, WOOD = "body", "ink", "cap", "steel", "felt", "bristle", "paint", "wood"


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

    out.append(Cone(arm(-0.05, 0.0), arm(1.25, 0.0), 0.135 * u, 0.200 * u, base, SKIN, 0, 0.09 * u))
    out.append(Cone(arm(1.25, 0.0), arm(reach, 0.0), 0.200 * u, 0.215 * u, base, SKIN, 0, 0.03 * u))
    cloth = _lin((0.24, 0.25, 0.27))
    out.append(Cone(arm(1.05, 0.02), arm(1.17, 0.02), 0.235 * u, 0.242 * u, cloth * 1.12, CLOTH, 2))
    out.append(Cone(arm(1.15, 0.02), arm(reach, 0.02), 0.235 * u, 0.25 * u, cloth, CLOTH, 2))
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
