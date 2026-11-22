# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The canvas: strokes turned into paint, with every step differentiable.

A stroke is a curve with a width. For each pixel the renderer finds the nearest point of the curve, how far along
the stroke that is and on which side, and from those: whether the pixel is under the brush, how much paint was
left at that point, which bristle passed over it. Pressure and speed change the mark the way they do for a hand:

* pressure spreads the brush (wider), presses more paint on (denser, thicker), and leaves a pool of paint where
  the brush landed and where it was lifted;
* speed empties the brush sooner, breaks the mark into bristle streaks, and ends it in a flick; a slow stroke
  trembles a little (in the final picture only).

Two renderers share that code. :func:`render_windows` and :func:`composite_level` paint thousands of strokes at
once, in small windows that are then laid on the canvas; they are what training and planning use.
:func:`render_painting` paints a finished painting's strokes one after another at any size, with the relief of
thick paint lit from the upper left. A test checks that both give the same picture.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .brushes import brush_profile
from .strokes import (
    A_ALPHA,
    A_COLOR,
    A_COLOR2,
    A_PHASE,
    A_PRESSURE,
    A_PROFILE,
    A_R0,
    A_R1,
    A_SPEED,
    A_STREAK,
    Strokes,
    geometry,
)

AA = 0.7                                     # pixels over which an edge goes from paint to no paint
THREADS = 420.0                              # threads of the canvas across a picture's width
TAU = 2.0 * math.pi


def curve_points(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, segments: int) -> torch.Tensor:
    """``(..., 2)`` control points -> ``(..., segments + 1, 2)`` points along the curve."""
    t = torch.linspace(0.0, 1.0, segments + 1, device=p0.device, dtype=p0.dtype)[:, None]
    return (1 - t) ** 2 * p0[..., None, :] + 2 * (1 - t) * t * p1[..., None, :] + t ** 2 * p2[..., None, :]


def add_tremor(pts: torch.Tensor, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """A slow hand is not perfectly steady: move the points of ``pts (M, S + 1, 2)`` sideways by a smooth wobble
    of ``amplitude (M,)`` pixels that is zero at both ends."""
    tangent = torch.cat([pts[:, 1:2] - pts[:, 0:1], pts[:, 2:] - pts[:, :-2], pts[:, -1:] - pts[:, -2:-1]], 1)
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], -1)
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    u = torch.linspace(0.0, 1.0, pts.shape[1], device=pts.device, dtype=pts.dtype)[None]
    wobble = torch.sin(TAU * (1.7 * u + phase[:, None])) * torch.sin(math.pi * u)
    return pts + normal * (amplitude[:, None] * wobble)[..., None]


def _nearest(pts: torch.Tensor, gx: torch.Tensor, gy: torch.Tensor):
    """For points ``(gx, gy) (P,)`` and polylines ``pts (M, S + 1, 2)``: distance to the line, how far along it
    the nearest point is (0..1), how far to the side of it the point lies (signed, measured square to the
    stroke's direction there, so it does not turn round the stroke's ends), and how far past an end it lies.
    Each ``(M, P)``."""
    segments = pts.shape[1] - 1
    best = far = side = along = None
    for i in range(segments):
        a, ab = pts[:, i], pts[:, i + 1] - pts[:, i]
        apx, apy = gx[None] - a[:, 0:1], gy[None] - a[:, 1:2]
        len2 = ab.pow(2).sum(-1, keepdim=True) + 1e-8
        t = (apx * ab[:, 0:1] + apy * ab[:, 1:2]) / len2
        tc = t.clamp(0.0, 1.0)
        dx, dy = apx - tc * ab[:, 0:1], apy - tc * ab[:, 1:2]
        d2 = dx * dx + dy * dy
        past = torch.zeros_like(d2)
        if i == 0:
            past = past + torch.relu(-t) * len2.sqrt()
        if i == segments - 1:
            past = past + torch.relu(t - 1.0) * len2.sqrt()
        cross = (ab[:, 0:1] * apy - ab[:, 1:2] * apx) / len2.sqrt()
        if best is None:
            best, along, side, far = d2, tc / segments, cross, past
        else:
            closer = d2 < best
            best = torch.where(closer, d2, best)
            along = torch.where(closer, (i + tc) / segments, along)
            side = torch.where(closer, cross, side)
            far = torch.where(closer, past, far)
    return (best + 1e-8).sqrt(), along, side, far


def _hash(x: torch.Tensor) -> torch.Tensor:
    """A number in 0..1 for each number in ``x`` that looks random (the sine hash of shader code)."""
    return torch.frac(torch.sin(x * 12.9898 + 4.1414) * 43758.5453)


def _noise(x: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
    """Smooth random values in 0..1 along ``x``: a new one at every whole number, eased from one to the next.
    ``seed`` (broadcast against ``x``) makes each stroke's values its own."""
    with torch.no_grad():
        i = torch.floor(x)
        a, b = _hash(i + 31.7 * seed), _hash(i + 1.0 + 31.7 * seed)
    f = x - i
    f = f * f * (3.0 - 2.0 * f)
    return a + (b - a) * f


def stroke_fields(pts: torch.Tensor, r0: torch.Tensor, r1: torch.Tensor, profile: torch.Tensor,
                  pressure: torch.Tensor, speed: torch.Tensor, phase: torch.Tensor, gx: torch.Tensor,
                  gy: torch.Tensor, aa: float = AA) -> dict:
    """The paint a stroke leaves on the points ``(gx, gy) (P,)``, everything in the same pixel units.
    ``pts (M, S + 1, 2)``, ``r0, r1, pressure, speed (M,)``, ``profile (M, fields)``, ``phase (M, 5)``.
    Returns ``paint (M, P)`` (how much paint, before the stroke's opacity), ``second (M, P)`` (where the brush's
    second colour shows), ``height (M, P)`` (how thick the paint lies), ``dry (M, P)`` (how dry the brush is
    there: a dry brush only catches the tops of the canvas's weave) and ``tone (M, P)`` (around 1: the track of
    one hair is a little lighter or darker than its neighbour's).

    A brush is a row of hairs. Each hair leaves its own track along the stroke, loaded a little differently from
    its neighbours; the tracks run on straight past the stroke's ends and each hair lifts off at its own
    moment, so a fast or dry stroke ends ragged; and along a track the paint breaks up as it runs out."""
    d, u, lat, far = _nearest(pts, gx, gy)
    hard, taper, end_round, bristle, count, decay, fade, thick = (profile[:, i : i + 1] for i in range(8))
    p, s = pressure[:, None], speed[:, None]
    length = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1).sum(-1, keepdim=True)
    flick = 0.7 * s * ((u - 0.55) / 0.45).clamp(0.0, 1.0) ** 2            # a fast stroke thins out at its end
    wobble = 1.0 + 0.05 * torch.sin(TAU * (1.3 * u + phase[:, 3:4])) \
        + 0.03 * torch.sin(TAU * (3.1 * u + phase[:, 4:5]))
    w = (r0[:, None] + (r1 - r0)[:, None] * u) * (0.75 + 0.5 * p) * (1.0 - taper * (2.0 * u - 1.0) ** 4)
    w = w * (1.0 - flick) * wobble                                         # no edge of a stroke is quite even
    edge = aa + (1.0 - hard) * 0.9 * w
    cover = torch.sigmoid(4.0 * (w - d) / edge)
    square = torch.sigmoid(4.0 * (0.5 * aa - far) / aa)
    cover = cover * (end_round + (1.0 - end_round) * square)
    v = (lat / (w + 1e-6)).clamp(-1.0, 1.0)
    across = (0.5 * v + 0.5) * count                                       # which hair, counted across the brush
    hairs = 0.6 * _noise(across, phase[:, 0:1]) + 0.4 * _noise(3.3 * across, phase[:, 1:2])
    dryness = (bristle * (1.25 - 0.5 * p) * (0.7 + 0.6 * s)).clamp(max=1.0)
    half = 0.5 * (r0 + r1)[:, None]
    rag = dryness * (0.35 + 0.9 * s) * 1.5 * half                          # how unevenly the hairs lift off
    end = rag * _noise(across, phase[:, 2:3]) ** 2
    start = 0.3 * rag * _noise(across, phase[:, 2:3] + 0.5) ** 2
    to_end, to_start = (1.0 - u) * length - far + w, u * length - far + w
    cover = cover * torch.sigmoid(4.0 * (to_end - end) / (aa + 0.3 * end)) \
        * torch.sigmoid(4.0 * (to_start - start) / (aa + 0.3 * start))
    thin = 1.0 - torch.exp(-3.0 * decay * (0.4 + 1.6 * s) * u)             # the brush running out of paint
    with torch.no_grad():
        hair = torch.floor(across)
    breaks = _noise(u * length / (1.5 * half + 1.0), phase[:, 1:2] + 0.137 * hair)
    marks = hairs * (0.6 + 0.4 * breaks)                                   # 0..1, the gaps the hairs leave
    texture = 1.0 - dryness * (0.25 + 0.75 * thin) * marks
    fading = 1.0 - (fade * (0.5 + s)).clamp(max=1.0) * thin
    pool = 0.35 * p * (1.0 - s)                                            # paint left where the brush landed
    density = 1.0 + pool * (torch.exp(-(u / 0.1) ** 2) + 0.5 * torch.exp(-((1.0 - u) / 0.08) ** 2))
    paint = (0.65 + 0.35 * p) * density * cover * texture * fading
    second = torch.sigmoid(7.0 * (_noise(0.6 * across + 1.0, phase[:, 3:4]) - 0.5))   # the second colour, in tracks
    ridge = 0.75 + 0.5 * v * v                                             # pressed paint piles up at the edges
    height = thick * (0.3 + 1.1 * p) * (1.0 - 0.5 * s) * density * ridge * cover * (0.55 + 0.45 * (1.0 - marks))
    tone = 1.0 + (0.05 + 0.13 * dryness) * (1.0 - 2.0 * hairs)
    return {"paint": paint, "second": second, "height": height, "dry": dryness * (0.25 + 0.75 * thin), "tone": tone}


def canvas_weave(height: int, width: int, pitch: float, device=None) -> torch.Tensor:
    """A primed canvas as the height of its surface, 0..1 at each pixel: threads ``pitch`` pixels apart that go
    over and under each other, none quite straight and no two the same thickness."""
    ys = torch.arange(height, device=device, dtype=torch.float32)[:, None] + 0.5
    xs = torch.arange(width, device=device, dtype=torch.float32)[None, :] + 0.5
    xw = xs + 0.35 * pitch * torch.sin(TAU * ys / (37.0 * pitch) + 1.3)    # the threads wander a little
    yw = ys + 0.35 * pitch * torch.sin(TAU * xs / (41.0 * pitch) + 0.4)
    i, j = torch.floor(xw / pitch), torch.floor(yw / pitch)
    warp = torch.sin(math.pi * xw / pitch).abs() * (0.5 + 0.5 * torch.cos(math.pi * yw / pitch + math.pi * i))
    weft = torch.sin(math.pi * yw / pitch).abs() * (0.5 + 0.5 * torch.cos(math.pi * xw / pitch + math.pi * (j + 1)))
    return torch.maximum(warp * (0.75 + 0.5 * _hash(i)), weft * (0.75 + 0.5 * _hash(j + 7.0))).clamp(0.0, 1.0)


def window_grid(size: int, device=None, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    c = torch.arange(size, device=device, dtype=dtype) + 0.5
    gy, gx = torch.meshgrid(c, c, indexing="ij")
    return gx.reshape(-1), gy.reshape(-1)


def render_windows(strokes: Strokes, size: int = 32, segments: int = 6, aa: float = AA,
                   chunk: int = 4096, soft: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Strokes with any leading shape ``(..., N)`` -> ``alpha (..., N, size, size)`` and the colour laid at each
    pixel ``rgb (..., N, 3, size, size)`` (linear RGB). Each stroke is drawn alone in its own window. ``soft``
    softens every brush's edge (see :func:`helix_lm.paint.brushes.brush_profile`)."""
    lead = strokes.keep.shape
    flat = strokes.flat()
    gx, gy = window_grid(size, flat.geom.device, flat.geom.dtype)
    alphas, rgbs = [], []
    for a in range(0, flat.keep.shape[0], chunk):
        part = flat.index(slice(a, a + chunk))
        g = geometry(part.geom)
        pts = curve_points(g["p0"], g["p1"], g["p2"], segments) * size
        f = stroke_fields(pts, g["r0"] * size, g["r1"] * size, brush_profile(part.brush, soft), g["pressure"],
                          g["speed"], part.phase, gx, gy, aa)
        alphas.append(((g["opacity"] * part.keep)[:, None] * f["paint"]).clamp(max=1.0))
        rgbs.append((part.color[:, :, None] + (part.color2 - part.color)[:, :, None]
                     * (part.streak[:, None] * f["second"])[:, None]) * f["tone"][:, None])
    alpha = torch.cat(alphas).reshape(*lead, size, size)
    rgb = torch.cat(rgbs).reshape(*lead, 3, size, size)
    return alpha, rgb


def composite_level(canvas: torch.Tensor, alpha: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """Lay a level's windows on the canvas. ``canvas (B, 3, H, W)`` linear RGB with ``H = Gy * cell`` and
    ``W = Gx * cell``; ``alpha (B, Gy, Gx, N, R, R)`` and ``rgb (B, Gy, Gx, N, 3, R, R)`` with ``R = 2 * cell``:
    the window of a cell is centred on it and twice its size, so neighbours overlap by half. Strokes are laid in
    order of their number; for each number the windows are laid in four groups (even and odd rows and columns),
    because the windows of one group do not touch each other and can be laid together."""
    b, gy, gx, n, r = alpha.shape[:5]
    cell, pad = r // 2, r // 4
    h, w = canvas.shape[-2:]
    assert h == gy * cell and w == gx * cell, (canvas.shape, alpha.shape)
    out = F.pad(canvas, (pad, pad, pad, pad), mode="replicate")
    hp, wp = out.shape[-2:]
    for k in range(n):
        for pi in (0, 1):
            ny = len(range(pi, gy, 2))
            for pj in (0, 1):
                nx = len(range(pj, gx, 2))
                if ny == 0 or nx == 0:
                    continue
                a = alpha[:, pi::2, pj::2, k].permute(0, 1, 3, 2, 4).reshape(b, 1, ny * r, nx * r)
                c = rgb[:, pi::2, pj::2, k].permute(0, 3, 1, 4, 2, 5).reshape(b, 3, ny * r, nx * r)
                where = (pj * cell, wp - pj * cell - nx * r, pi * cell, hp - pi * cell - ny * r)
                a, c = F.pad(a, where), F.pad(c, where)
                out = out * (1.0 - a) + c * a
    return out[..., pad : pad + h, pad : pad + w]


def cut_windows(image: torch.Tensor, cell: int) -> torch.Tensor:
    """``(B, C, Gy * cell, Gx * cell)`` -> ``(B, Gy, Gx, C, 2 * cell, 2 * cell)``: the window of every cell, as
    :func:`composite_level` lays them."""
    b, c, h, w = image.shape
    pad = cell // 2
    padded = F.pad(image, (pad, pad, pad, pad), mode="replicate")
    cols = F.unfold(padded, kernel_size=2 * cell, stride=cell)                 # (B, C * R * R, Gy * Gx)
    return cols.reshape(b, c, 2 * cell, 2 * cell, h // cell, w // cell).permute(0, 4, 5, 1, 2, 3)


def light_relief(image: torch.Tensor, height: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
    """Thick paint catches the light: shade ``image (3, H, W)`` (linear) by the slope of ``height (H, W)`` as lit
    from the upper left."""
    if strength <= 0:
        return image
    k = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], device=image.device) / 16.0
    smooth = F.conv2d(F.pad(height[None, None], (1, 1, 1, 1), mode="replicate"), k[None, None])[0, 0]
    dx = smooth - torch.roll(smooth, 1, 1)
    dy = smooth - torch.roll(smooth, 1, 0)
    dx[:, 0], dy[0] = 0.0, 0.0
    scale = image.shape[-1] / 512.0                      # the same paint looks the same at any picture size
    shade = (1.0 - strength * scale * (dx + dy)).clamp(0.72, 1.35)
    return image * shade


@torch.no_grad()
def render_painting(rows: torch.Tensor, width: int, height: int, ground: torch.Tensor, segments: int = 14,
                    relief: float = 0.5, tremor: bool = True, canvas: torch.Tensor | None = None,
                    paint_height: torch.Tensor | None = None, finish: bool = True, aa: float = AA,
                    weave: float = 0.0, scale: float | None = None,
                    origin: tuple = (0.0, 0.0)) -> tuple[torch.Tensor, torch.Tensor]:
    """A painting's strokes ``rows (N, ABS)`` (units of the canvas width) painted in order on a canvas of
    ``width x height`` pixels toned with ``ground (3,)`` (linear RGB). Returns ``(image (3, H, W) linear RGB, the
    height of the paint (H, W))``. Pass the ``canvas`` and ``paint_height`` of an earlier call (with
    ``finish=False``) to go on painting where it stopped.

    ``weave`` (0..1) gives the canvas its cloth: a dry brush then only catches the tops of the threads, thin
    paint lets them show and thick paint buries them. ``scale`` is how many pixels the unit of the strokes is
    (the picture's width when not given) and ``origin`` where the picture's corner lies on the canvas, for a
    canvas larger than the picture, whose edge then shows."""
    device = rows.device
    scale = float(width if scale is None else scale)
    cloth = None
    pitch = scale / THREADS
    strength = weave * min(1.0, max(0.0, (pitch - 1.4) / 1.6))            # threads too fine to draw are left out
    if strength > 0:
        cloth = canvas_weave(height, width, pitch, device)
    if canvas is None:
        canvas = ground.to(device).float()[:, None, None].expand(3, height, width).clone()
        if cloth is not None:                              # the tone was brushed on: it is not perfectly even
            ys = torch.arange(height, device=device, dtype=torch.float32)[:, None] / scale
            xs = torch.arange(width, device=device, dtype=torch.float32)[None, :] / scale
            streaks = torch.sin(TAU * (9.0 * ys + 1.7 * xs)) + torch.sin(TAU * (23.0 * ys - 3.1 * xs + 0.3)) \
                + torch.sin(TAU * (4.3 * xs + 0.8 * ys + 0.6))
            canvas = canvas * (1.0 + 0.02 * strength * streaks)
        paint_height = torch.zeros(height, width, device=device)
    pts_all = curve_points(rows[:, 0:2], rows[:, 2:4], rows[:, 4:6], segments) * scale
    pts_all = pts_all + torch.tensor(origin, device=device, dtype=pts_all.dtype)
    if tremor and len(rows):
        steadiness = (1.0 - rows[:, A_SPEED]) ** 2
        pts_all = add_tremor(pts_all, 0.3 * steadiness * rows[:, A_R0] * scale, rows[:, A_PHASE][:, 4])
    reach = 1.3 * torch.maximum(rows[:, A_R0], rows[:, A_R1]) * scale * 2.2 + 2.0 + 2.0 * aa if len(rows) else None
    for i in range(rows.shape[0]):
        pts, m = pts_all[i], float(reach[i])
        x0, x1 = int(max(0.0, float(pts[:, 0].min()) - m)), int(min(float(width), float(pts[:, 0].max()) + m + 1))
        y0, y1 = int(max(0.0, float(pts[:, 1].min()) - m)), int(min(float(height), float(pts[:, 1].max()) + m + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        ys = torch.arange(y0, y1, device=device, dtype=torch.float32) + 0.5
        xs = torch.arange(x0, x1, device=device, dtype=torch.float32) + 0.5
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        row = rows[i : i + 1]
        f = stroke_fields(pts[None], row[:, A_R0] * scale, row[:, A_R1] * scale, row[:, A_PROFILE],
                          row[:, A_PRESSURE], row[:, A_SPEED], row[:, A_PHASE], gx.reshape(-1), gy.reshape(-1), aa)
        a = (row[0, A_ALPHA] * f["paint"][0]).clamp(max=1.0).reshape(y1 - y0, x1 - x0)
        if cloth is not None:                              # a dry brush skips the hollows between the threads
            dry = f["dry"][0].reshape(y1 - y0, x1 - x0)
            buried = (paint_height[y0:y1, x0:x1] / 0.35).clamp(0.0, 1.0)      # unless paint already fills them
            catch = torch.sigmoid(9.0 * (cloth[y0:y1, x0:x1] - 0.25 - 0.45 * dry))
            a = a * (1.0 - strength * dry * (1.0 - buried) * (1.0 - catch))
        second = (row[0, A_STREAK] * f["second"][0]).reshape(y1 - y0, x1 - x0)
        c1, c2 = row[0, A_COLOR], row[0, A_COLOR2]
        rgb = (c1[:, None, None] + (c2 - c1)[:, None, None] * second) * f["tone"][0].reshape(y1 - y0, x1 - x0)
        canvas[:, y0:y1, x0:x1] = canvas[:, y0:y1, x0:x1] * (1.0 - a) + rgb * a
        thick = f["height"][0].reshape(y1 - y0, x1 - x0)
        paint_height[y0:y1, x0:x1] = paint_height[y0:y1, x0:x1] * (1.0 - 0.85 * a) + a * thick
    if not finish:
        return canvas, paint_height
    surface = paint_height
    if cloth is not None:                                  # the cloth shows where the paint is thin
        surface = paint_height + 0.2 * strength * cloth * (1.0 - (paint_height / 0.45).clamp(0.0, 1.0))
    return light_relief(canvas, surface, relief).clamp(0.0, 1.0), paint_height


__all__ = ["AA", "THREADS", "add_tremor", "canvas_weave", "composite_level", "curve_points", "cut_windows",
           "light_relief", "render_painting", "render_windows", "stroke_fields", "window_grid"]
