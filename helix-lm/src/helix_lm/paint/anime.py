# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Anime drawn by hand, frame by frame, the way a studio makes it.

A video (from an open video model, or your own) is read once for its space: how things move, how near they
are, which pixels are the character and which the set, how the camera moves, and one clean picture of the set
(:mod:`helix_lm.paint.space`). Then:

1. **The colours** are chosen once for the whole clip (:func:`helix_lm.paint.cel.palette_for`), so the
   character's hair is the same red in every drawing.
2. **The background** is painted once, in detail, from the clean picture of the set, and held under every
   drawing (or moved with the camera), as anime background art is. It does not boil or morph.
3. **The timing** is worked out as on an exposure sheet (:func:`helix_lm.paint.motion.exposure`): fast action
   on ones, most movement on twos, slow movement on threes, holds when the character is still.
4. **Each drawing** inks and paints only the character, exactly to its frame (:func:`helix_lm.paint.cel.paint_frame`):
   line art, flat colour up to the lines, hard-edged shadow and highlight shapes.
5. **Faces and small parts** (eyes, mouth, paws, hands) are drawn again at three times the size and put in
   place (:mod:`helix_lm.paint.detail`). From one drawing to the next a part that only moved is carried over
   rigidly (the same eye, turned and shifted, no morph); it is drawn again only when it really changed (a
   blink, a mouth opening).
6. **Each line keeps its name** from drawing to drawing (``Line.uid``) where it is the same line moved, and
   the path of every line through time is then smoothed (:func:`helix_lm.paint.motion.steady`), which takes
   out wobble and boil and keeps arcs and ease.
7. **Every drawing is checked** against its frame (:func:`helix_lm.paint.check.score`); the numbers are kept
   with the animation.

What this does not do: it cannot make a video model's shapes better than they are. If the model melts a hand
for three frames, the drawing follows the frame; steadying takes out shakes, not a melt that lasts.
"""

from __future__ import annotations

import math
import time

import torch

from .cel import paint_frame, palette_for
from .check import fix, score
from .detail import REDRAW, draw_details, find_details, follow, merge
from .draw import CELS, Line
from .flow import optical_flow, sample
from .motion import exposure, keys, motion_amount, steady
from .space import light_direction, normals, read_space

STYLE_WAVER = 0.12             # a clean anime line wavers far less than a marker sketch
# added to what is asked of the video model, so it imagines something a cel painter draws well
ASK = "anime style, cel shading, clean line art, flat colours, crisp details, detailed background"


# ------------------------------------------------------------------------------------- the background --
BACKGROUND_MARKERS = 100        # the background's colours are numbered from here, apart from the character's


def background_art(plate, palette: torch.Tensor | None, frame_size: tuple, *, lines: str = "pen",
                   detail: float = 0.85, colours: int = 28) -> list:
    """The set painted once (``plate``, :class:`helix_lm.paint.space.Plate`), its lines in board widths of a
    frame of ``frame_size (H, W)`` placed as frame 0 sees it. Background art is painted with more colours and
    finer detail than a character's cel (``colours`` of its own, found in the set; ``palette`` None), and its
    colours are numbered from :data:`BACKGROUND_MARKERS`, so recolouring the character leaves the set alone."""
    _, w = frame_size
    pw = plate.image.shape[-1]
    own = palette_for([plate.image], colours) if palette is None else palette
    drawn = paint_frame(plate.image, own, line_tool=lines, detail=detail, background=True)
    ox, oy = plate.origin
    shift = torch.tensor([float(ox), float(oy)])
    out = []
    for line in drawn:
        moved = line.moved((line.points * pw + shift) / w, age=0)
        if moved.marker >= 0:
            moved.marker += BACKGROUND_MARKERS
        out.append(moved)
    return out


def on_camera(lines: list, camera: torch.Tensor, width: int) -> list:
    """``lines`` (board widths of frame 0) as frame t's ``camera (2, 3)`` sees them."""
    a, b = camera[:, :2].float(), camera[:, 2].float()
    return [line.moved(((line.points * width) @ a.T + b) / width, age=line.age) for line in lines]


def still_camera(camera: torch.Tensor, tolerance: float = 1e-4) -> bool:
    return bool((camera.float() - torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])).abs().max() < tolerance)


# ----------------------------------------------------------------------------- lines that stay the same --
def _centre_area(line: Line, scale: float) -> tuple:
    ring = line.ring_points()[0] * scale
    x, y = ring[:, 0], ring[:, 1]
    xs, ys = torch.roll(x, -1), torch.roll(y, -1)
    cross = x * ys - xs * y
    area = 0.5 * float(cross.sum())
    if abs(area) < 1e-6:
        return ring.mean(0), 0.0
    cx = float(((x + xs) * cross).sum()) / (6.0 * area)
    cy = float(((y + ys) * cross).sum()) / (6.0 * area)
    return torch.tensor([cx, cy]), abs(area)


def carry_names(before: list, after: list, flow: torch.Tensor, scale: float, *, reach: float = 2.5) -> int:
    """Give each line of drawing ``after`` the name (uid) and age of the line of ``before`` it is, moved by
    ``flow (2, H, W)`` (pixels, from the frame of ``before`` to that of ``after``): an ink line whose points lie
    within ``reach`` pixels (on average) of the moved line and of about its length, a cel shape of the same
    colour and kind whose middle moved to about the same place and whose size is about the same. One to one,
    the closest first. Returns how many were carried."""
    pairs = []
    old_ink = [(i, line) for i, line in enumerate(before) if line.kind in ("line", "sketch")]
    moved_ink = {}
    for i, line in old_ink:
        pts = line.points * scale
        moved_ink[i] = pts + sample(flow, pts)
    boxes = {i: (p.min(0).values - 2 * reach, p.max(0).values + 2 * reach) for i, p in moved_ink.items()}
    for j, line in enumerate(after):
        if line.kind not in ("line", "sketch"):
            continue
        pts = line.points * scale
        lo, hi = pts.min(0).values, pts.max(0).values
        length = line.length() * scale
        for i, old in old_ink:
            if old.kind != line.kind or old.marker != line.marker:
                continue
            blo, bhi = boxes[i]
            if bool((hi < blo).any() or (lo > bhi).any()):
                continue
            ratio = length / max(1e-6, old.length() * scale)
            if not 0.6 <= ratio <= 1.6:
                continue
            gap = float(torch.cdist(pts, moved_ink[i]).min(1).values.mean())
            if gap <= reach:
                pairs.append((gap, i, j))
    old_cels = [(i, line) for i, line in enumerate(before) if line.kind in CELS and line.rings]
    shapes = {}
    for i, line in old_cels:
        c, a = _centre_area(line, scale)
        shapes[i] = (c + sample(flow, c[None])[0], a)
    for j, line in enumerate(after):
        if line.kind not in CELS or not line.rings:
            continue
        c, a = _centre_area(line, scale)
        for i, old in old_cels:
            if old.kind != line.kind or old.marker != line.marker or old.colour != line.colour:
                continue
            oc, oa = shapes[i]
            if oa <= 0 or not 0.6 <= a / oa <= 1.6:
                continue
            gap = float((c - oc).norm())
            if gap <= max(3.0, 0.3 * math.sqrt(oa)):
                pairs.append((gap / max(1.0, math.sqrt(oa)), i, j))
    pairs.sort()
    used_old, used_new, carried = set(), set(), 0
    for _, i, j in pairs:
        if i in used_old or j in used_new:
            continue
        used_old.add(i)
        used_new.add(j)
        after[j].uid, after[j].age = before[i].uid, before[i].age + 1
        carried += 1
    return carried


# ---------------------------------------------------------------------------------- faces and details --
def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _inside(lines: list, box, scale: float) -> list:
    x0, y0, x1, y1 = box
    out = []
    for line in lines:
        c = line.points.mean(0) * scale
        if x0 <= float(c[0]) <= x1 and y0 <= float(c[1]) <= y1:
            out.append(line)
    return out


def detail_pass(frame: torch.Tensor, lines: list, palette: torch.Tensor, drawer, *, before=None, most: int = 6,
                scale_up: int = 3) -> tuple[list, list, dict]:
    """The drawing ``lines`` of ``frame`` with its faces and small parts drawn again with care. ``before`` is
    ``(frame, parts, detail lines)`` of the drawing before: a part that is the same part moved is carried over
    rigidly when it did not change. Returns the new lines, the parts with their detail lines (for the next
    drawing), and counts."""
    h, w = frame.shape[-2:]
    parts = find_details(frame, max_parts=most)
    if not parts:
        return lines, [], {"parts": 0, "carried": 0, "drawn": 0}
    carried, drawn, detail_lines, kept_parts = 0, 0, [], []
    flow = None
    for part in parts:
        own = None
        if before is not None:
            last_frame, last_parts = before
            match = max(last_parts, key=lambda p: _iou(p[0].box, part.box) if p[0].kind == part.kind else -1.0,
                        default=None)
            if match is not None and match[0].kind == part.kind and _iou(match[0].box, part.box) > 0.3:
                if flow is None:
                    flow = optical_flow(last_frame, frame)
                moved, change = follow(match[1], match[0].box, last_frame, frame, flow)
                if change < REDRAW and moved:
                    own, carried = moved, carried + 1
        if own is None:
            own = draw_details(frame, [part], palette, drawer=drawer, scale_up=scale_up)
            drawn += 1
        kept_parts.append((part, own))
        detail_lines += own
    return merge(lines, detail_lines, parts), kept_parts, {"parts": len(parts), "carried": carried, "drawn": drawn}


# ----------------------------------------------------------------------------------------- the clip --
def draw_anime(frames: torch.Tensor, fps: float, *, palette: torch.Tensor | None = None, colours: int = 16,
               lines: str = "pen", detail: float = 0.6, depth="model", base: int = 2, shading: str = "frame",
               details: bool = True, check: bool = True, steadiness: float = 1.0, start=None,
               background: list | None = None, log=None) -> dict:
    """
    Draw ``frames (T, 3, H, W)`` (sRGB, ``fps`` frames a second) as anime, as the module's notes tell. Returns a
    dict: ``sheets`` (one :class:`helix_lm.paint.animate.Sheet` a drawing), ``holds`` (frames each drawing is
    shown), ``drawn`` (which frames got a drawing), ``palette``, ``background`` (the background art's lines),
    ``cameras`` (one ``(2, 3)`` a drawing), ``checks`` (how each drawing scored against its frame), ``space``.

    ``depth`` "model" asks a depth model (Depth Anything V2 Small), None goes without. ``base`` is the timing
    for most movement (2: twos; 1: every frame; 3: threes). ``shading`` "frame" takes shadow shapes from the
    frame's own light and dark, "3d" from the surface's direction (depth) and one light for the whole clip,
    which cannot flicker. ``start`` is ``(sheet, frame)`` of a drawing to go on from (the next clip), and
    ``background`` the background art already painted for it (it is then not painted again). ``check``
    draws the places where a drawing is clearly off its frame again (:func:`helix_lm.paint.check.fix`), keeping
    only what matches better.
    """
    from .animate import Sheet

    began = time.time()
    say = log or (lambda _: None)
    frames = frames.float().clamp(0.0, 1.0)
    t_count, _, h, w = frames.shape
    space = read_space(frames, depth=depth)
    say(f"read the clip's space in {time.time() - began:.0f} s: the character in "
        f"{float(space.character.float().mean()):.0%} of the picture, the camera "
        f"{'still' if all(still_camera(c) for c in space.cameras) else 'moving'}")
    palette = palette_for(list(frames) + [space.plate.image], colours) if palette is None else palette
    if background is None:
        background = background_art(space.plate, None, (h, w), lines=lines)
        say(f"painted the background once: {len(background)} shapes and lines")

    amount = motion_amount(space.flows, space.character)
    picked, _ = keys(amount)
    sheet_of = exposure(amount, fps, base=base, keys=picked)
    drawn = [i for i, _ in sheet_of]
    holds = [int(held) for _, held in sheet_of]
    say(f"timing: {len(drawn)} drawings for {t_count} frames ({len(drawn) / max(1, t_count) * fps:.1f} a second), "
        f"keys at {picked}")

    light = None
    if shading == "3d" and space.depth is not None:
        n0 = normals(space.depth[0])
        light = light_direction(frames[0], n0, space.character[0])

    def drawer(crop, pal):
        return paint_frame(crop, pal, line_tool=lines, detail=min(1.0, detail + 0.25), background=True)

    sheets, checks = [], []
    last_lines, last_frame, last_parts = None, None, []
    if start:
        last_lines, last_frame = start[0].lines, start[1].float()
    for k, i in enumerate(drawn):
        frame = frames[i]
        extra = {}
        if light is not None:
            extra = {"normals": normals(space.depth[i]), "light": light}
        subject = space.character[i]
        if float(subject.float().mean()) < 0.002:       # no character found in this frame: the painter judges
            subject = None
        own = paint_frame(frame, palette, line_tool=lines, detail=detail, background=False, subject=subject, **extra)
        counts = {}
        if details:
            before = (last_frame, last_parts) if last_parts else None
            own, last_parts, counts = detail_pass(frame, own, palette, drawer, before=before)
        under = background_for(background, space.cameras[i], w)
        if check:
            own = fix(own, frame, drawer, palette, rounds=1, under=under)
        carried = 0
        if last_lines is not None:
            flow = optical_flow(last_frame, frame)
            carried = carry_names(last_lines, own, flow, float(w))
        mark = _score(own, frame, palette, under)
        checks.append({"frame": i, "error": round(float(mark["error"]), 4), "bare": round(float(mark["bare"]), 4),
                       "chamfer": round(float(mark["chamfer"]), 2)})
        sheets.append(Sheet(own, carried, len(own) - carried, last_lines is None))
        say(f"drawing {k + 1} of {len(drawn)} (frame {i + 1}, held {holds[k]}): {len(own)} lines, {carried} the same "
            f"lines moved, {counts.get('parts', 0)} faces and parts ({counts.get('carried', 0)} carried over); "
            f"colour off by {checks[-1]['error']:.3f}, bare {checks[-1]['bare']:.1%}")
        last_lines, last_frame = own, frame
    if steadiness > 0 and len(sheets) >= 4:
        sheets = steady(sheets, strength=steadiness, keys=[drawn.index(k) for k in picked if k in drawn])
        say("steadied every line's path through time")
    say(f"drawn in {time.time() - began:.0f} s")
    return {"sheets": sheets, "holds": holds, "drawn": drawn, "palette": palette, "background": background,
            "cameras": space.cameras[drawn], "checks": checks, "space": space}


def _score(lines: list, frame: torch.Tensor, palette: torch.Tensor, under: list) -> dict:
    """How the drawing over its background matches the frame (:func:`helix_lm.paint.check.score`)."""
    try:
        return score(lines, frame, palette, under=under)
    except TypeError:                                   # a check without layers: score the drawing alone
        return score(under + lines, frame, palette)


def then(camera: torch.Tensor, before: torch.Tensor) -> torch.Tensor:
    """The camera ``(2, 3)`` that does ``before`` and then ``camera``."""
    return torch.cat([camera[:, :2] @ before[:, :2], (camera[:, :2] @ before[:, 2:] + camera[:, 2:])], 1)


def background_for(background: list, camera: torch.Tensor, width: int) -> list:
    """The background art as the frame with ``camera`` sees it."""
    return background if still_camera(camera) else on_camera(background, camera, width)


__all__ = ["ASK", "BACKGROUND_MARKERS", "STYLE_WAVER", "background_art", "background_for", "carry_names",
           "detail_pass", "draw_anime", "on_camera", "still_camera", "then"]
