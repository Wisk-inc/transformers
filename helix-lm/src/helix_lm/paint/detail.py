# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Faces and small details, drawn with care and kept on model from frame to frame.

People watching a film look at faces first, and in a face at the eyes and the mouth; when someone speaks the
mouth gets more of the looks, when they feel something the eyes do (eye tracking studies of film viewing, as
reported in The Conversation, "What eye tracking tells us about the way we watch films", and Carter et al.,
"Animated character appearance affects viewing patterns", Carnegie Mellon). Hands are watched too: Preston
Blair (Cartoon Animation) notes that a hand alone can tell a story. So those few places are where a drawing
is judged, and where a blurred or jumbled line shows most. The rules used here:

1. **Find the face by its eyes.** An anime eye is built as an upper lash line, heavier in the middle and
   tapering at the ends, a large iris that touches the lids, a darker pupil inside it and a small white
   highlight on the iris, set the same way in both eyes by one light (Clip Studio, "Paint anime eyes in 6
   steps"; Skillshare and Udemy eye tutorials). So an eye is looked for as a small, compact, dark closed shape,
   often with a light hole in it. Blair builds a head on an eye line at right angles to its centre line, the
   two eyes anchored on that line and the nose and mouth below on the centre line; so two eye shapes of about
   the same size, about level, with a mouth or nose below them and a head shape of one colour round them, make
   a face (:func:`find_details`). Other dense small detail (paws, fingers, toes, ears, a ruff) is found where
   many short, strongly curved borders come together.
2. **Draw those parts bigger.** A small part is cut out with a margin, enlarged sharply (bicubic, then a little
   unsharp masking) and drawn at that size, so the pens follow its real shapes, and the lines are brought back
   into the frame (:func:`draw_details`, :func:`merge`). A marker nib does not shrink with the drawing, so where
   a pen width cannot be scaled down the colour of the part comes back as cel paint, filled exactly.
3. **Keep it on model.** Model sheets exist so that many animators' drawings of a character look as if one
   artist drew them all, and clean-up keeps to the proportions of the construction (Wikipedia, "Model sheet";
   the clean-up hints of Disney-trained animators). A face that only moves or turns is therefore not drawn
   again, and never morphed: its drawing is carried into the next frame by one rotation, one uniform scale and
   one shift (:func:`follow`), so its shapes stay exactly the same and both eyes stay on the same eye line, the
   same size. Only when the part itself changes (a blink, a mouth opening: anime mouths are a few simple
   shapes, from a closed line to an open shape, CharacterHub's anime mouth guide) is it drawn again.

:mod:`helix_lm.paint.check` measures how well a drawing matches its frame and redraws where it does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .color import image_to_oklab, linear_to_srgb, srgb_to_linear
from .draw import CELS, SURFACES, TOOLS, Line, colour_labels, colour_lines, draw_picture, line_strength, look, stamp
from .flow import optical_flow, sample
from .see import gaussian_blur, saliency

REDRAW = 0.02          # a part whose aligned picture changed in more than this share of its box is drawn again
KINDS = ("face", "eye", "mouth", "hand", "detail")


@dataclass
class Part:
    """A place in a frame to draw with care: its ``box`` (x0, y0, x1, y1) in pixels, its ``kind`` (one of
    :data:`KINDS`), how sure the finder is (``score``), the ``size`` (H, W) of the frame it was found in, and
    for an eye or a mouth the number of its face in the list (``face``, -1 for none)."""

    box: tuple
    kind: str
    score: float = 0.0
    size: tuple = (0, 0)
    face: int = -1

    @property
    def width(self) -> float:
        return float(self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])

    @property
    def centre(self) -> tuple:
        return 0.5 * (self.box[0] + self.box[2]), 0.5 * (self.box[1] + self.box[3])


# ------------------------------------------------------------------------------------------- shapes --
def _odd(x: float) -> int:
    k = max(1, int(round(x)))
    return k if k % 2 else k + 1


def components(mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """The connected parts of ``mask (H, W)`` (pixels touching at a side or a corner): a number for each part
    ``(H, W)`` long (-1 outside the mask), numbered in the order their first pixels come row by row, and how
    many there are. A union and find over the pixels: along every pair of neighbours the larger root is hooked
    under the smaller, then every pixel's pointer is followed to its root, until no pair disagrees: a few
    rounds, however long and winding the shapes (the same labels as SciPy's ``ndimage.label``)."""
    h, w = mask.shape
    mask = mask.bool()
    if not bool(mask.any()):
        return torch.full((h, w), -1, dtype=torch.long), 0
    index = torch.arange(h * w).view(h, w)
    pairs = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):  # each pair of neighbours once
        ys0, ys1 = slice(0, h - dy), slice(dy, h)
        xs0, xs1 = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
        both = mask[ys0, xs0] & mask[ys1, xs1]
        pairs.append(torch.stack([index[ys0, xs0][both], index[ys1, xs1][both]]))
    a, b = torch.cat(pairs, 1)
    parent = torch.arange(h * w)
    while True:
        pa, pb = parent[a], parent[b]
        differ = pa != pb
        if not bool(differ.any()):
            break
        parent.scatter_reduce_(0, torch.maximum(pa, pb)[differ], torch.minimum(pa, pb)[differ], "amin")
        while True:                                   # every pointer straight to its root
            nxt = parent[parent]
            if torch.equal(nxt, parent):
                break
            parent = nxt
    roots, inverse = torch.unique(parent.view(h, w)[mask], return_inverse=True)
    out = torch.full((h, w), -1, dtype=torch.long)
    out[mask] = inverse
    return out, len(roots)


def _stats(comp: torch.Tensor, n: int, value: torch.Tensor) -> dict:
    """For each part: its area, box (x0, y0, x1, y1, pixel edges), centre and the mean of ``value`` over it."""
    ys, xs = torch.nonzero(comp >= 0, as_tuple=True)
    c = comp[ys, xs]
    area = torch.bincount(c, minlength=n).float()
    big = float(max(comp.shape) + 1)

    def reduce(v, how, fill):
        return torch.full((n,), fill).scatter_reduce(0, c, v.float(), how, include_self=True)

    return {"area": area, "x0": reduce(xs, "amin", big), "y0": reduce(ys, "amin", big),
            "x1": reduce(xs, "amax", -1.0) + 1, "y1": reduce(ys, "amax", -1.0) + 1,
            "cx": reduce(xs + 0.5, "sum", 0.0) / area.clamp(min=1),
            "cy": reduce(ys + 0.5, "sum", 0.0) / area.clamp(min=1),
            "mean": reduce(value[ys, xs], "sum", 0.0) / area.clamp(min=1)}


def mask_rings(mask: torch.Tensor) -> list:
    """The outline of ``mask (H, W)`` as closed rings of points (pixels, x then y), so that filling them even-odd
    (:func:`helix_lm.paint.draw.fill_rings`) gives the mask back, holes and all. The rings run along the pixels'
    edges through the middle of each edge, so a slant is a straight line and not a staircase; that cuts each
    corner by an eighth of a pixel (a lone pixel comes back half covered)."""
    m = F.pad(mask.bool()[None, None].float(), (1, 1, 1, 1))[0, 0].bool()
    inside = m[1:-1, 1:-1]
    starts, ends = [], []
    for out, (sx, sy, ex, ey) in (
        (inside & ~m[:-2, 1:-1], (0, 0, 1, 0)),     # the top edge, left to right
        (inside & ~m[1:-1, 2:], (1, 0, 1, 1)),      # the right edge, downward
        (inside & ~m[2:, 1:-1], (1, 1, 0, 1)),      # the bottom edge, right to left
        (inside & ~m[1:-1, :-2], (0, 1, 0, 0)),     # the left edge, upward
    ):
        ys, xs = torch.nonzero(out, as_tuple=True)
        starts.append(torch.stack([xs + sx, ys + sy], -1))
        ends.append(torch.stack([xs + ex, ys + ey], -1))
    start, end = torch.cat(starts), torch.cat(ends)
    if len(start) == 0:
        return []
    width = int(mask.shape[1]) + 2
    key_start = (start[:, 1] * width + start[:, 0]).tolist()
    key_end = (end[:, 1] * width + end[:, 0]).tolist()
    leaving: dict = {}
    for i, k in enumerate(key_start):
        leaving.setdefault(k, []).append(i)
    used = [False] * len(key_start)
    rings = []
    for first in range(len(key_start)):
        if used[first]:
            continue
        loop, i = [], first
        while not used[i]:
            used[i] = True
            loop.append(i)
            options = leaving[key_end[i]]
            nxt = next((j for j in options if not used[j]), None)
            if nxt is None:
                break
            i = nxt
        ids = torch.tensor(loop)
        mid = (start[ids] + end[ids]).float() / 2.0
        if len(mid) >= 3:                            # points in a straight row are not needed
            a, b = mid - torch.roll(mid, 1, 0), torch.roll(mid, -1, 0) - mid
            bend = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).abs() > 1e-6
            mid = mid[bend] if int(bend.sum()) >= 3 else mid
        rings.append(mid)
    return rings


# ---------------------------------------------------------------------------------------- clipping --
def _inside_box(pts: torch.Tensor, box) -> torch.Tensor:
    return (pts[:, 0] >= box[0]) & (pts[:, 0] <= box[2]) & (pts[:, 1] >= box[1]) & (pts[:, 1] <= box[3])


def clip_polyline(pts: torch.Tensor, pressure: torch.Tensor, box, keep_inside: bool = True) -> list:
    """The pieces of a line ``pts (K, 2)`` (pixels) inside ``box`` (or outside it), cut exactly where the line
    crosses the box's edge: a list of ``(points, pressure)``."""
    if len(pts) < 2:
        inside = bool(_inside_box(pts, box).all()) if len(pts) else False
        return [(pts, pressure)] if inside == keep_inside and len(pts) else []
    lo, hi = pts.min(0).values, pts.max(0).values
    if float(lo[0]) >= box[0] and float(hi[0]) <= box[2] and float(lo[1]) >= box[1] and float(hi[1]) <= box[3]:
        return [(pts, pressure)] if keep_inside else []
    if float(hi[0]) < box[0] or float(lo[0]) > box[2] or float(hi[1]) < box[1] or float(lo[1]) > box[3]:
        return [] if keep_inside else [(pts, pressure)]
    a, b = pts[:-1], pts[1:]
    d = b - a
    cuts = []
    for axis, edge in ((0, box[0]), (0, box[2]), (1, box[1]), (1, box[3])):
        step = torch.where(d[:, axis].abs() > 1e-9, d[:, axis], torch.full_like(d[:, axis], 1e-9))
        cuts.append((edge - a[:, axis]) / step)
    t = torch.stack(cuts, 1)                                     # (K-1, 4)
    ok = (t > 1e-6) & (t < 1 - 1e-6)
    seg, which = torch.nonzero(ok, as_tuple=True)
    tt = t[seg, which]
    key = torch.cat([torch.arange(len(pts), dtype=torch.float32), seg.float() + tt])
    allp = torch.cat([pts, a[seg] + d[seg] * tt[:, None]])
    allq = torch.cat([pressure, pressure[:-1][seg] + (pressure[1:] - pressure[:-1])[seg] * tt])
    order = key.argsort()
    allp, allq = allp[order], allq[order]
    middle = 0.5 * (allp[1:] + allp[:-1])
    inside = _inside_box(middle, box) == keep_inside
    pieces, run = [], []
    for i, flag in enumerate(inside.tolist()):
        if flag:
            if not run:
                run = [i]
            run.append(i + 1)
        elif run:
            pieces.append(run)
            run = []
    if run:
        pieces.append(run)
    return [(allp[torch.tensor(r)], allq[torch.tensor(r)]) for r in pieces]


def clip_ring(ring: torch.Tensor, box) -> torch.Tensor:
    """A closed ring ``(K, 2)`` (pixels) cut to the inside of ``box`` (Sutherland and Hodgman, 1974). Each ring
    of an even-odd shape can be cut alone: inside the box every point is inside as many rings as before."""
    pts = ring
    for axis, edge, keep_below in ((0, box[0], False), (0, box[2], True), (1, box[1], False), (1, box[3], True)):
        if len(pts) == 0:
            break
        nxt = torch.roll(pts, -1, 0)
        a_in = (pts[:, axis] <= edge) if keep_below else (pts[:, axis] >= edge)
        b_in = (nxt[:, axis] <= edge) if keep_below else (nxt[:, axis] >= edge)
        d = nxt[:, axis] - pts[:, axis]
        t = (edge - pts[:, axis]) / torch.where(d.abs() > 1e-9, d, torch.full_like(d, 1e-9))
        cross = pts + (nxt - pts) * t[:, None]
        out = torch.zeros(len(pts), 2, 2)
        use = torch.zeros(len(pts), 2, dtype=torch.bool)
        out[:, 0] = torch.where((a_in != b_in)[:, None], cross, nxt)
        use[:, 0] = b_in | (a_in != b_in)
        out[:, 1] = nxt
        use[:, 1] = ~a_in & b_in
        pts = out[use]
    return pts


def _nib(line: Line, scale: float) -> float:
    """About how far (pixels) a line's ink reaches beside its middle on a board ``scale`` pixels wide."""
    tool = TOOLS[line.tool]
    return 0.5 * tool.width * scale * (1.0 + 0.5 * tool.swell) * float(getattr(line, "width", 1.0))


def clip_lines(lines: list, box, scale: float, keep_inside: bool = True) -> list:
    """``lines`` (board widths of a board ``scale`` pixels wide) cut to the inside of ``box`` (pixels) or to its
    outside. A cel shape is cut exactly either way (to the outside: cut to the four strips round the box, all
    kept as one shape, so the cuts between them cancel out when it is filled). A marker stroke cut to the
    inside stops a nib's half width before the box's edge, so its ink ends there; cut to the outside it stops
    at the edge, and what is drawn in the box goes over the little its nib reaches in."""
    out = []
    far = 1e6
    for line in lines:
        pts = line.points * scale
        if line.kind in CELS and line.rings is not None:
            lo, hi = pts.min(0).values, pts.max(0).values
            apart = (float(hi[0]) <= box[0] or float(lo[0]) >= box[2] or float(hi[1]) <= box[1]
                     or float(lo[1]) >= box[3])
            if apart:
                if not keep_inside:
                    out.append(line)
                continue
            windows = [box] if keep_inside else [(-far, -far, box[0], far), (box[2], -far, far, far),
                                                 (box[0], -far, box[2], box[1]), (box[0], box[3], box[2], far)]
            rings = [clip_ring(r * scale, window) for window in windows for r in line.ring_points()]
            rings = [r for r in rings if len(r) >= 3]
            if rings:
                out.append(Line(torch.cat(rings) / scale, torch.ones(sum(len(r) for r in rings)), line.tool,
                                line.colour, line.marker, line.kind, line.age, line.uid, [len(r) for r in rings]))
            continue
        cut = box
        if line.marker >= 0 and keep_inside:         # colour: the ink reaches a nib's half width past the line
            r = _nib(line, scale)
            cut = (box[0] + r, box[1] + r, box[2] - r, box[3] - r)
            if cut[2] <= cut[0] or cut[3] <= cut[1]:
                continue
        pieces = clip_polyline(pts, line.pressure, cut, keep_inside)
        if len(pieces) == 1 and len(pieces[0][0]) == len(pts) and torch.equal(pieces[0][0], pts):
            out.append(line)
            continue
        for p, q in pieces:
            if len(p) >= 2 and float((p[1:] - p[:-1]).norm(dim=-1).sum()) >= 1.0:
                piece = Line(p / scale, q, line.tool, line.colour, line.marker, line.kind, line.age, None)
                if hasattr(line, "width"):
                    piece.width = line.width
                out.append(piece)
    return out


# ------------------------------------------------------------------------------------------ finding --
def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _within(a, b) -> float:
    """How much of box ``a`` lies inside box ``b``."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def _clamp_box(box, h: int, w: int) -> tuple:
    return (max(0.0, float(box[0])), max(0.0, float(box[1])), min(float(w), float(box[2])),
            min(float(h), float(box[3])))


def _small_dark(lab: torch.Tensor, side: int) -> torch.Tensor:
    """How much darker each pixel is than the light round it, for shapes smaller than about a sixth of the
    picture (a morphological closing of the lightness, minus the lightness: a large dark area is not filled
    in by the closing, so it does not count)."""
    k = _odd(0.16 * side)
    light = lab[0][None, None]

    def grow(x):                                     # a square's maximum, row then column: much quicker
        x = F.max_pool2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="replicate"), (1, k), 1)
        return F.max_pool2d(F.pad(x, (0, 0, k // 2, k // 2), mode="replicate"), (k, 1), 1)

    closed = -grow(-grow(light))
    return (closed - light)[0, 0]


def _eye_shapes(lab: torch.Tensor, side: int) -> list:
    """Shapes that may be eyes: small, compact and dark against the light round them. Each is a dict with its
    box, centre, area and how much light hole (a highlight) it holds. They are looked for at a few levels of
    darkness, from mid to near black, because at a lighter level an eye can run into dark hair beside it."""
    light = lab[0]
    darker = _small_dark(lab, side) > 0.22
    least, most = max(6.0, (0.012 * side) ** 2), 0.22 * side
    out: list = []
    for level in (0.62, 0.5, 0.4, 0.3):
        comp, n = components(darker & (light < level))
        if n == 0:
            continue
        st = _stats(comp, n, light)
        bw_all, bh_all = st["x1"] - st["x0"], st["y1"] - st["y0"]
        fits = ((st["area"] >= least) & (torch.maximum(bw_all, bh_all) <= most) & (bw_all >= 0.35 * bh_all)
                & (bw_all <= 3.2 * bh_all) & (st["area"] >= 0.3 * bw_all * bh_all))
        for i in torch.nonzero(fits).flatten().tolist():
            area, bw, bh = float(st["area"][i]), float(bw_all[i]), float(bh_all[i])
            cx, cy = float(st["cx"][i]), float(st["cy"][i])
            if any(abs(cx - e["c"][0]) < 0.5 * e["w"] and abs(cy - e["c"][1]) < 0.5 * e["h"] for e in out):
                continue                             # found already, whole, at a lighter level
            x0, y0, x1, y1 = int(st["x0"][i]), int(st["y0"][i]), int(st["x1"][i]), int(st["y1"][i])
            own = comp[y0:y1, x0:x1] == i
            if float(torch.quantile(light[y0:y1, x0:x1][own], 0.2)) > 0.42:  # an eye has a dark lash line or pupil
                continue
            holes, count = components(F.pad((~own).float()[None, None], (1, 1, 1, 1), value=1.0)[0, 0].bool())
            lit = 0.0
            if count > 1:
                outside = int(holes[0, 0])
                inner = (holes[1:-1, 1:-1] >= 0) & (holes[1:-1, 1:-1] != outside)
                bright = light[y0:y1, x0:x1] > max(0.7, float(st["mean"][i]) + 0.3)
                lit = float((inner & bright).sum())
            out.append({"box": (float(x0), float(y0), float(x1), float(y1)), "c": (cx, cy), "area": area, "w": bw,
                        "h": bh, "lit": lit, "light": float(st["mean"][i])})
    out.sort(key=lambda e: -e["area"] * (1.5 if e["lit"] > 0 else 1.0))
    return out[:40]


def _faces_by_eyes(image: torch.Tensor, lab: torch.Tensor) -> list:
    """Faces found by a pair of eyes (see the module's notes): ``[(face box, score, eye a, eye b, mouth box)]``."""
    h, w = lab.shape[-2:]
    side = min(h, w)
    eyes = _eye_shapes(lab, side)
    pairs = []
    for i in range(len(eyes)):
        for j in range(i + 1, len(eyes)):
            a, b = eyes[i], eyes[j]
            if a["c"][0] > b["c"][0]:
                a, b = b, a
            dx, dy = b["c"][0] - a["c"][0], b["c"][1] - a["c"][1]
            d = math.hypot(dx, dy)
            size = math.sqrt(min(a["area"], b["area"]) / max(a["area"], b["area"]))
            tall = min(a["h"], b["h"]) / max(a["h"], b["h"])
            wide = 0.5 * (a["w"] + b["w"])
            tilt = abs(math.degrees(math.atan2(dy, dx)))
            if size < 0.6 or tall < 0.55 or tilt > 30.0 or not (1.3 * wide <= d <= 5.0 * wide):
                continue
            pairs.append((size + tall + (1.0 - tilt / 30.0), a, b, d, (dx / d, dy / d)))
    found = []
    flat = lab.reshape(3, -1)
    for base, a, b, d, (ex, ey) in sorted(pairs, key=lambda p: -p[0]):
        nx, ny = -ey, ex                                       # down the face, at right angles to the eye line
        mx, my = 0.5 * (a["c"][0] + b["c"][0]), 0.5 * (a["c"][1] + b["c"][1])
        seeds = []
        for u, v in ((0.0, 0.35), (-0.5, 0.5), (0.5, 0.5), (0.0, 0.15)):         # between and under the eyes
            x, y = mx + (ex * u + nx * v) * d, my + (ey * u + ny * v) * d
            seeds.append((min(w - 1, max(0, int(x))), min(h - 1, max(0, int(y)))))
        colours = torch.stack([flat[:, y * w + x] for x, y in seeds])
        skin = colours.median(0).values
        if float((colours - skin).norm(dim=1).max()) > 0.12 or float(skin[0]) < 0.35:
            continue
        # the head: the shape of the skin's colour round the eyes, no larger than a head on this eye line
        corners = torch.tensor([[ex * u * d + nx * v * d + mx, ey * u * d + ny * v * d + my]
                                for u in (-1.45, 1.45) for v in (-1.6, 1.9)])
        limit = _clamp_box((float(corners[:, 0].min()), float(corners[:, 1].min()), float(corners[:, 0].max()),
                            float(corners[:, 1].max())), h, w)
        lx0, ly0, lx1, ly1 = (int(v) for v in limit)
        if lx1 - lx0 < 4 or ly1 - ly0 < 4:
            continue
        near = (lab[:, ly0:ly1, lx0:lx1] - skin[:, None, None]).norm(dim=0) < 0.09
        comp, _ = components(near)
        sx, sy = seeds[0]
        own = comp[min(sy - ly0, comp.shape[0] - 1), min(sx - lx0, comp.shape[1] - 1)]
        if int(own) < 0:
            continue
        head = comp == own
        ys, xs = torch.nonzero(head, as_tuple=True)
        hbox = (float(xs.min()) + lx0, float(ys.min()) + ly0, float(xs.max()) + 1 + lx0, float(ys.max()) + 1 + ly0)
        surrounds = all(hbox[0] <= e["c"][0] <= hbox[2] and hbox[1] <= e["c"][1] <= hbox[3] for e in (a, b))
        if not surrounds or float(head.sum()) < d * d or hbox[2] - hbox[0] < 1.25 * d:   # too small for a head
            continue
        # it ends somewhere: a colour that runs on past three sides of where a head could be is the ground round
        # the eyes, not a head (a side at the picture's edge does not count; the neck may run on below)
        runs = sum(bool(edge.any()) and not at_border for edge, at_border in (
            (head[:, 0], lx0 == 0), (head[:, -1], lx1 == w), (head[0], ly0 == 0), (head[-1], ly1 == h)))
        if runs >= 3:
            continue
        # a mouth or a nose: something on the face below the eyes, on the centre line
        feature = ((lab[:, ly0:ly1, lx0:lx1] - skin[:, None, None]).norm(dim=0) > 0.15) & ~head
        fcomp, fn = components(feature)
        mouth, best = None, 0.0
        if fn:
            st = _stats(fcomp, fn, lab[0, ly0:ly1, lx0:lx1])
            for k in range(fn):
                fx, fy = float(st["cx"][k]) + lx0 - mx, float(st["cy"][k]) + ly0 - my
                down, along = fx * nx + fy * ny, fx * ex + fy * ey
                area = float(st["area"][k])
                if 0.3 * d <= down <= 1.5 * d and abs(along) <= 0.45 * d and area <= (0.7 * d) ** 2 and area > best:
                    best = area
                    mouth = (float(st["x0"][k]) + lx0, float(st["y0"][k]) + ly0, float(st["x1"][k]) + lx0,
                             float(st["y1"][k]) + ly0)
        score = base + 1.0 + (0.7 if mouth else 0.0) + 0.4 * ((a["lit"] > 0) + (b["lit"] > 0))
        if score < 3.4:
            continue
        # the face: the head's colour, the eyes, the mouth, and at least where a face lies on this eye line (so an
        # eye at the edge of the colour found, a cheek in shadow, is not cut in two)
        least = torch.tensor([[mx + (ex * u + nx * v) * d, my + (ey * u + ny * v) * d]
                              for u in (-1.2, 1.2) for v in (-0.8, 1.5)])
        expect = (float(least[:, 0].min()), float(least[:, 1].min()), float(least[:, 0].max()),
                  float(least[:, 1].max()))
        boxes = [hbox, expect, a["box"], b["box"]] + ([mouth] if mouth else [])
        face = (min(x[0] for x in boxes), min(x[1] for x in boxes), max(x[2] for x in boxes), max(x[3] for x in boxes))
        face = _clamp_box((max(face[0], limit[0]) - 0.08 * d, max(face[1], limit[1]) - 0.08 * d,
                           min(face[2], limit[2]) + 0.08 * d, min(face[3], limit[3]) + 0.08 * d), h, w)
        if any(_box_iou(face, f[0]) > 0.2 or a in (f[2], f[3]) or b in (f[2], f[3]) for f in found):
            continue
        found.append((face, score, a, b, mouth))
    return found


def _eye_part_box(e: dict, h: int, w: int) -> tuple:
    """An eye's box from its dark shape: the lower lid and the white of the eye lie under the dark lash line."""
    x0, y0, x1, y1 = e["box"]
    gx, gy = 0.2 * (x1 - x0), 0.2 * (y1 - y0)
    return _clamp_box((x0 - gx, y0 - gy, x1 + gx, y1 + 2.0 * gy), h, w)


def _landmark_box(pts: torch.Tensor, ids, grow: float, h: int, w: int) -> tuple:
    q = pts if ids is None else pts[ids]
    lo, hi = q.min(0).values, q.max(0).values
    g = grow * (hi - lo)
    return _clamp_box((float(lo[0] - g[0]), float(lo[1] - g[1]), float(hi[0] + g[0]), float(hi[1] + g[1])), h, w)


def _mediapipe_faces(image: torch.Tensor) -> list:
    """Human faces by MediaPipe's face landmarks, when the package is there (it is optional, and this path is
    not covered by the tests): ``[(face box, eye box, eye box, mouth box)]``, or ``[]``."""
    try:
        import mediapipe as mp
        import numpy as np

        h, w = image.shape[-2:]
        rgb = (image.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=4) as mesh:
            got = mesh.process(rgb)
        out = []
        for face in got.multi_face_landmarks or []:
            pts = torch.tensor([[p.x * w, p.y * h] for p in face.landmark])
            out.append(tuple(_landmark_box(pts, ids, grow, h, w) for ids, grow in (
                (None, 0.08), ([33, 133, 159, 145, 160, 144], 0.25), ([362, 263, 386, 374, 387, 373], 0.25),
                ([61, 291, 0, 17, 13, 14], 0.25))))
        return out
    except Exception:                                # not installed, or another version of its interface
        return []


def detail_map(image: torch.Tensor, lab: torch.Tensor | None = None) -> torch.Tensor:
    """Where a picture has dense small detail, ``(H, W)``: many short, strong borders that turn every way (a
    paw, fingers, an ear, a ruff; not a long straight edge, not faint texture), weighted by how much the place
    stands out (:func:`helix_lm.paint.see.saliency`)."""
    h, w = image.shape[-2:]
    side = min(h, w)
    strength, angle = line_strength(image)
    strong = (strength > 0.08).float()
    sigma = max(1.5, 0.012 * side)
    gx, gy = torch.cos(2 * angle) * strength, torch.sin(2 * angle) * strength
    one_way = (gaussian_blur(gx, sigma) ** 2 + gaussian_blur(gy, sigma) ** 2).sqrt() / (gaussian_blur(strength, sigma)
                                                                                         + 1e-6)
    dense = gaussian_blur(strong, 1.5 * sigma) * (1.0 - one_way).clamp(0.0, 1.0)
    return dense * (0.4 + 0.6 * saliency(image))


def _dense_details(image: torch.Tensor, lab: torch.Tensor, labels, taken: list, most: int) -> list:
    h, w = image.shape[-2:]
    side = min(h, w)
    if most <= 0:
        return []
    detail = detail_map(image, lab)
    if labels is not None:                           # many small shapes of the given labels count as detail too
        small = torch.zeros(h, w)
        for value in torch.unique(labels).tolist():
            comp, n = components(labels == value)
            if n:
                area = torch.bincount(comp[comp >= 0], minlength=n).float()
                tiny = (area >= 3) & (area <= (0.04 * side) ** 2)
                small += (tiny[comp.clamp(min=0)] & (comp >= 0)).float()
        detail = detail * (0.7 + 0.6 * gaussian_blur(small, max(1.5, 0.012 * side)).clamp(0, 1))
    for box in taken:
        x0, y0, x1, y1 = (int(v) for v in box)
        detail[y0:y1, x0:x1] = 0.0
    out = []
    k = _odd(0.14 * side)
    least, top = max(6.0, 0.05 * side), None
    while len(out) < most:
        mass = _box_mean(detail, k)                  # the densest windows of about a seventh of the picture
        peak = float(mass.max())
        top = peak if top is None else top
        if peak < max(0.02, 0.3 * top):
            break
        y, x = divmod(int(mass.argmax()), w)
        wx0, wy0, wx1, wy1 = max(0, x - k // 2), max(0, y - k // 2), min(w, x + k // 2 + 1), min(h, y + k // 2 + 1)
        window = detail[wy0:wy1, wx0:wx1]
        ys, xs = torch.nonzero(window > 0.3 * float(window.max()), as_tuple=True)
        x0, y0 = float(xs.min()) + wx0, float(ys.min()) + wy0
        x1, y1 = float(xs.max()) + 1 + wx0, float(ys.max()) + 1 + wy0
        g = 0.08 * k
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        half_w, half_h = max(0.5 * (x1 - x0) + g, 0.5 * least), max(0.5 * (y1 - y0) + g, 0.5 * least)
        box = _clamp_box((cx - half_w, cy - half_h, cx + half_w, cy + half_h), h, w)
        bx0, by0, bx1, by1 = _box_int(box, h, w)
        detail[by0:by1, bx0:bx1] = 0.0
        detail[wy0:wy1, wx0:wx1] *= 0.5
        out.append(Part(box, "detail", peak, (h, w)))
    return out


def _box_mean(x: torch.Tensor, k: int) -> torch.Tensor:
    """``x (H, W)`` averaged over a ``k`` by ``k`` square round each pixel (by running sums, so a large square
    costs no more than a small one)."""
    h, w = x.shape
    r = k // 2
    p = F.pad(x.double()[None, None], (r + 1, r, r + 1, r))[0, 0]
    c = p.cumsum(0).cumsum(1)
    total = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (total / (k * k)).float()[:h, :w]


def find_details(image: torch.Tensor, labels: torch.Tensor | None = None, max_parts: int = 12, *,
                 landmarks: bool = True) -> list:
    """The parts of ``image (3, H, W)`` (sRGB) to draw with care, most important first: faces (found by a pair
    of eyes, see the module's notes, or by MediaPipe's face landmarks when that package is installed and
    ``landmarks``), each followed by its eyes and its mouth, then other dense small detail (paws, fingers,
    toes, ears). ``labels (H, W)``, the shapes the picture is seen in (for example
    :attr:`helix_lm.paint.draw.Look.labels`), lets many small shapes count as detail too. Returns
    :class:`Part` s, at most ``max_parts``.

    What it cannot do: a face is found only when both eyes are open and seen (a face in profile, with its
    eyes closed or turned away is not; a head drawn that way still comes as dense detail, and a face found in
    one frame can be followed through the next, :func:`follow`); a hand, a paw or an ear comes as ``detail``,
    as it is not told apart from other dense detail (``hand`` is for a caller that knows)."""
    image = image.float().clamp(0.0, 1.0)
    h, w = image.shape[-2:]
    lab = image_to_oklab(image)
    parts: list = []
    faces = _mediapipe_faces(image) if landmarks else []
    for face, left, right, mouth in faces:
        number = len(parts)
        parts += [Part(face, "face", 1.0, (h, w)), Part(left, "eye", 1.0, (h, w), number),
                  Part(right, "eye", 1.0, (h, w), number), Part(mouth, "mouth", 1.0, (h, w), number)]
    if not faces:
        for face, score, a, b, mouth in _faces_by_eyes(image, lab):
            number = len(parts)
            parts += [Part(face, "face", score, (h, w)), Part(_eye_part_box(a, h, w), "eye", score, (h, w), number),
                      Part(_eye_part_box(b, h, w), "eye", score, (h, w), number)]
            if mouth:
                gx, gy = 0.3 * (mouth[2] - mouth[0]) + 1, 0.3 * (mouth[3] - mouth[1]) + 1
                parts.append(Part(_clamp_box((mouth[0] - gx, mouth[1] - gy, mouth[2] + gx, mouth[3] + gy), h, w),
                                  "mouth", score, (h, w), number))
    parts = parts[:max_parts]
    taken = [p.box for p in parts if p.kind == "face"]
    return parts + _dense_details(image, lab, labels, taken, max_parts - len(parts))


# ------------------------------------------------------------------------------------------ drawing --
def enlarge(crop: torch.Tensor, factor: int, sharpen: float = 0.5) -> torch.Tensor:
    """``crop (3, h, w)`` made ``factor`` times larger, sharply: bicubic, then a little unsharp masking so the
    borders stay crisp (the borders of a cel drawing are hard; a soft enlargement would make them look
    blurred to the line finder)."""
    if factor <= 1:
        return crop
    up = F.interpolate(crop[None].float(), scale_factor=factor, mode="bicubic", align_corners=False)[0]
    up = up.clamp(0.0, 1.0)
    if sharpen > 0:
        up = (up + sharpen * (up - gaussian_blur(up, 0.5 * factor))).clamp(0.0, 1.0)
    return up


def _shaded(colour, opacity: float = TOOLS["marker"].opacity) -> tuple:
    """The colour a marker gives where it goes over its own colour once more (it is see-through: it darkens)."""
    lin = srgb_to_linear(torch.tensor(colour, dtype=torch.float32))
    return tuple(linear_to_srgb(lin * (1.0 - opacity * (1.0 - lin))).tolist())


def on_board(palette: torch.Tensor, surface: str = "whiteboard",
             opacity: float = TOOLS["marker"].opacity) -> torch.Tensor:
    """The colours ``palette (K, 3)`` (sRGB) markers leave on a bare ``surface``: they are see-through, so a
    marker looks lighter than its ink. Cel paint in these colours sits beside marker colouring unnoticed."""
    board = srgb_to_linear(torch.tensor(SURFACES.get(surface, SURFACES["whiteboard"])))
    return linear_to_srgb(board * (1.0 - opacity * (1.0 - srgb_to_linear(palette.float()))))


def accents(palette: torch.Tensor, surface: str = "whiteboard") -> torch.Tensor:
    """The colours to paint a part's small shapes with beside marker colouring: what each marker leaves on the
    board (:func:`on_board`), so a colour looks the same in the part as everywhere else, except for the darks
    a see-through marker would leave clearly lighter (pupils, lash lines, a nose, paws): those keep their own
    colour, so they stay as dark and clear as they are drawn."""
    board = on_board(palette, surface)
    lift = image_to_oklab(board.T[:, :, None])[0, :, 0] - image_to_oklab(palette.float().T[:, :, None])[0, :, 0]
    return torch.where((lift > 0.1)[:, None], palette.float(), board)


def _cel(mask: torch.Tensor, colour, marker: int, kind: str = "cel") -> Line | None:
    rings = mask_rings(mask)
    if not rings:
        return None
    pts = torch.cat(rings)
    return Line(pts, torch.ones(len(pts)), "brush", colour, marker, kind, 0, None, [len(r) for r in rings])


def cel_shapes(labels: torch.Tensor, palette: torch.Tensor, darker: torch.Tensor | None = None, grow: int = 1) -> list:
    """Cel paint for a picture seen as palette colours: for each colour of ``labels (H, W)`` (-1 is left bare)
    the shapes it covers, filled exactly (points in pixels), the largest first; and over them, where
    ``darker``, the shadow shapes in that colour gone over once more. Each shape is grown by ``grow`` pixels
    under the next, so no light seam shows between two shapes that meet."""
    out = []
    counts = sorted(((int((labels == k).sum()), k) for k in range(len(palette))), reverse=True)
    for count, k in counts:
        if count == 0:
            continue
        mask = labels == k
        if grow > 0:
            mask = F.max_pool2d(mask[None, None].float(), 2 * grow + 1, 1, grow)[0, 0] > 0
        line = _cel(mask, tuple(palette[k].tolist()), k)
        if line is not None:
            out.append(line)
    if darker is not None:
        for count, k in counts:
            if count:
                line = _cel(darker & (labels == k), _shaded(tuple(palette[k].tolist())), k, "cel-shadow")
                if line is not None:
                    out.append(line)
    return out


def joining(labels: torch.Tensor, outside: torch.Tensor, share: float = 0.0) -> torch.Tensor:
    """The shapes of ``labels (H, W)`` that reach into ``outside (H, W)`` (with more than ``share`` of their
    pixels there): their labels, -1 elsewhere."""
    out = torch.full_like(labels, -1)
    for k in torch.unique(labels[outside]).tolist():
        if k < 0:
            continue
        comp, n = components(labels == k)
        there = torch.bincount(comp[outside & (labels == k)] + 1, minlength=n + 1).float()
        every = torch.bincount(comp[labels == k] + 1, minlength=n + 1).float().clamp(min=1)
        reach = (there > 0) & (there > share * every)
        out[reach[comp + 1] & (labels == k)] = k
    return out


def _outside(h: int, w: int, inner) -> torch.Tensor:
    outside = torch.ones(h, w, dtype=torch.bool)
    x0, y0, x1, y1 = (int(round(v)) for v in inner)
    outside[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = False
    return outside


def _seen(crop: torch.Tensor, palette: torch.Tensor, detail: float):
    seen = look(crop, palette, detail)
    ink = draw_picture(crop, palette, colour=None, detail=detail, seen=seen)
    labels, darker = colour_labels(seen.planes, palette)
    return ink, labels, darker


def draw_crop(crop: torch.Tensor, palette: torch.Tensor, detail: float = 0.85) -> list:
    """A part drawn as cel animation is: the ink lines as :func:`helix_lm.paint.draw.draw_picture` draws them
    with much detail, and the colour as cel paint of the palette's colours, shape by shape
    (:func:`cel_shapes`). Marker hatching would only come near those shapes, and when a picture is enlarged its
    dark outlines are wider than the pen, so they are painted as the shapes they are rather than left to two
    thin lines. The white the picture leaves as paper (the white of an eye, a highlight) is painted too, in
    the whiteboard's colour, so a character drawn this way and laid over its background does not show the
    background through its eyes. Points in the crop's board widths. (:func:`draw_details` uses this, and
    colours the shapes that go on past the part as the frame's markers do.)"""
    ink, labels, darker = _seen(crop, palette, detail)
    w = float(labels.shape[1])
    paper = F.max_pool2d((labels < 0)[None, None].float(), 3, 1, 1)[0, 0] > 0
    white = _cel(paper, SURFACES["whiteboard"], -1)
    cels = ([white] if white is not None else []) + cel_shapes(labels, palette, darker)
    return ink + [line.moved(line.points / w, age=0) for line in cels]


def small_shapes(labels: torch.Tensor, area: float) -> torch.Tensor:
    """The shapes of ``labels (H, W)`` no larger than ``area`` pixels: their labels, -1 elsewhere."""
    out = torch.full_like(labels, -1)
    for k in torch.unique(labels).tolist():
        if k < 0:
            continue
        comp, n = components(labels == k)
        size = torch.bincount(comp[comp >= 0], minlength=n)
        out[(comp >= 0) & (size[comp.clamp(min=0)] <= area)] = k
    return out


def _default_part(image: torch.Tensor, palette: torch.Tensor, box, window, big: torch.Tensor, factor: int):
    """A part drawn the default way, to sit in a frame coloured with markers: the ink and the part's own shapes
    as :func:`draw_crop` draws them (board widths of ``big``, the part's ``window`` enlarged ``factor``
    times), in the colours the markers leave on the board (:func:`on_board`), its small darks kept dark
    (:func:`accents`); and the shapes that go on well past the part (more than 30% of them outside it, like
    the ground behind) coloured with markers at the frame's own size, so they join the frame's colouring
    outside: the same nibs, the same slant, and gone over again where the frame, looked at at its own size,
    is darker than its marker (frame board widths). Bare paper wholly inside the part (a highlight, the white
    of an eye) is painted the whiteboard's colour, so no marker the frame laid there shows through it."""
    h, w = image.shape[-2:]
    x0, y0, x1, y1 = window
    ink, labels, darker = _seen(big, palette, 0.85)
    bh, bw = labels.shape
    inner = tuple((v - o) * factor for v, o in zip(box, (x0, y0, x0, y0), strict=True))
    join = joining(labels, _outside(bh, bw, inner), 0.3)
    inside = torch.where(join >= 0, torch.full_like(labels, -1), labels)
    tiny = small_shapes(inside, (0.2 * factor * min(box[2] - box[0], box[3] - box[1])) ** 2)
    large = torch.where(tiny >= 0, torch.full_like(labels, -1), inside)
    own = cel_shapes(large, on_board(palette), darker) + cel_shapes(tiny, accents(palette), darker)
    paper = (labels < 0) & ~joining(labels + 1, _outside(bh, bw, inner)).eq(0)
    blank = _cel(paper, SURFACES["whiteboard"], -1)  # a highlight stays white over the frame's markers under it
    own += [blank] if blank is not None else []
    drawn = ink + [line.moved(line.points / float(bw), age=0) for line in own]
    votes = F.one_hot(join + 1, len(palette) + 1).permute(2, 0, 1).float()[None]
    small = F.avg_pool2d(votes, factor)[0].argmax(0) - 1                  # the window's own pixels
    small = small[: y1 - y0, : x1 - x0]
    shade = look(image[:, y0:y1, x0:x1], palette).darker
    bx0, by0, bx1, by1 = _box_int(box, h, w)
    masks, shades = {}, {}
    for k in torch.unique(small).tolist():
        if k < 0:
            continue
        m = torch.zeros(h, w, dtype=torch.bool)
        m[y0:y1, x0:x1] = small == k
        keep = torch.zeros(h, w, dtype=torch.bool)
        keep[by0:by1, bx0:bx1] = True
        masks[k] = m & keep
        sh = torch.zeros(h, w, dtype=torch.bool)
        sh[y0:y1, x0:x1] = shade
        shades[k] = masks[k] & sh
    colour = colour_lines(torch.zeros(h, w, dtype=torch.long), torch.zeros(h, w, dtype=torch.bool), palette,
                          float(w), masks, shades) if masks else []
    return drawn, colour


def _as_cels(lines: list, height: int, width: int) -> list:
    """Marker strokes drawn on a board ``width`` by ``height`` pixels turned into cel paint: for each marker and
    layer, the area its strokes cover as a filled shape (points in pixels of that board)."""
    groups: dict = {}
    for line in lines:
        groups.setdefault((line.marker, line.kind, line.colour), []).append(line)
    out = []
    for (marker, kind, colour), group in groups.items():
        cel = "cel-shadow" if kind == "shade" else "cel"
        line = _cel(stamp(group, height, width, float(width)), _shaded(colour) if kind == "shade" else colour,
                    marker, cel)
        if line is not None:
            out.append(line)
    return out


def _can_shrink() -> bool:
    """Whether a line's nib can be made narrower than its tool's (a ``width`` on :class:`Line`)."""
    return "width" in getattr(Line, "__slots__", ()) or hasattr(Line, "width")


def draw_details(image: torch.Tensor, parts: list, palette: torch.Tensor, drawer=None, scale_up: int = 3) -> list:
    """The lines that draw each part of ``image (3, H, W)`` (sRGB) at ``scale_up`` times its size, in the
    frame's board widths, each cut to its part's box (so :func:`merge` can put them in place of the frame's
    own lines there). ``drawer(crop, palette) -> list[Line]`` draws a picture, its lines in that picture's
    board widths (by default :func:`draw_crop`, with the shapes that go on past a part coloured as the frame's
    markers colour them, see :func:`_default_part`; a cel painter fits the same way). A part lying inside
    another part of the list is drawn with that one, and where two parts overlap the one earlier in the list
    (the more important) draws the overlap.

    Ink lines come back with the frame's own pens, which are already the finest. Cel paint comes back as it
    was drawn, cut a pixel past its box so no seam shows. Marker strokes, whose nib would be several times too
    wide on the frame's board, come back as cel paint of the same colours, filled where they coloured the
    enlarged part (or, when :class:`helix_lm.paint.draw.Line` has a ``width``, with their nib made narrower by
    the enlargement)."""
    drawer = drawer or draw_crop
    image = image.float().clamp(0.0, 1.0)
    h, w = image.shape[-2:]
    scale = float(w)
    shrink = _can_shrink()
    out = []
    boxes = [p.box if isinstance(p, Part) else tuple(p) for p in parts]
    drawn_boxes: list = []
    for i, box in enumerate(boxes):
        if any(j != i and _within(box, other) >= 0.8 and (_within(other, box) < 0.8 or j < i)
               for j, other in enumerate(boxes)):
            continue
        earlier = [b for b in drawn_boxes if _box_iou(b, box) > 0]   # what an earlier part drew is not drawn twice
        drawn_boxes.append(box)
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bw < 2 or bh < 2:
            continue
        margin = max(4.0, 0.2 * max(bw, bh))
        x0, y0 = max(0, int(math.floor(box[0] - margin))), max(0, int(math.floor(box[1] - margin)))
        x1, y1 = min(w, int(math.ceil(box[2] + margin))), min(h, int(math.ceil(box[3] + margin)))
        big = enlarge(image[:, y0:y1, x0:x1], scale_up)
        bh_up, bw_up = big.shape[-2:]
        if drawer is draw_crop:                      # the default: shapes going on past the part get markers
            drawn, colour = _default_part(image, palette, box, (x0, y0, x1, y1), big, scale_up)
            for other in earlier:
                colour = clip_lines(colour, other, scale, keep_inside=False)
            out += colour
        else:
            drawn = drawer(big, palette)
        factor = (x1 - x0) / scale                   # the crop's board width in the frame's board widths
        offset = torch.tensor([x0, y0], dtype=torch.float32)
        markers_ = [line for line in drawn if line.marker >= 0 and line.kind not in CELS]
        rest = [line for line in drawn if not (line.marker >= 0 and line.kind not in CELS)]
        if markers_ and not shrink:
            rest += [line.moved(line.points / bw_up, age=0) for line in _as_cels(markers_, bh_up, bw_up)]
            markers_ = []
        mapped = []
        for line in rest + markers_:
            moved = line.moved((line.points * (x1 - x0) + offset) / scale, age=line.age)
            if shrink and line.marker >= 0 and line.kind not in CELS:
                moved.width = float(getattr(line, "width", 1.0)) * factor
            mapped.append(moved)
        cels = [line for line in mapped if line.kind in CELS]
        lines = [line for line in mapped if line.kind not in CELS]
        mine = clip_lines(lines, box, scale, keep_inside=True)
        mine += clip_lines(cels, (box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1), scale, keep_inside=True)
        for other in earlier:
            mine = clip_lines(mine, other, scale, keep_inside=False)
        out += mine
    return out


def _assemble(kept: list, new: list) -> list:
    """Lines put together in the order they go on: ink (kept, then new); the kept cel paint; the markers, the
    kept and the new strokes of one marker one after the other (while wet they blend, so where they meet at a
    part's edge they do not darken); then the new cel paint, which covers whatever of the old reached into its
    place (cel paint covers what is under it but stays under the ink)."""
    def group(lines, kinds):
        return [line for line in lines if line.kind in kinds]

    ink = ("sketch", "line")
    colour = sorted(group(kept, ("fill", "shade")) + group(new, ("fill", "shade")),
                    key=lambda line: (line.kind == "shade", line.marker))
    return group(kept, ink) + group(new, ink) + group(kept, CELS) + colour + group(new, CELS)


def _dense_points(line: Line, scale: float) -> torch.Tensor:
    pts = line.points * scale
    if len(pts) < 2 or line.rings is not None:
        return pts
    seg = pts[1:] - pts[:-1]
    n = seg.norm(dim=-1).ceil().clamp(min=1).long()
    steps = torch.repeat_interleave(torch.arange(len(seg)), n)
    frac = torch.cat([torch.arange(int(k)) / int(k) for k in n.tolist()])
    return torch.cat([pts[:-1][steps] + seg[steps] * frac[:, None], pts[-1:]])


def merge(frame_lines: list, detail_lines: list, parts: list) -> list:
    """The frame's lines with what lay inside the parts' boxes taken out (a line crossing a box is cut at its
    edge) and the detail lines (:func:`draw_details`) put in. Marker colour of a shape that goes on past a
    part (the detail lines colour it there with that marker, at the frame's size) is kept as the frame drew
    it, so it runs on through the part's edge unbroken; the detail's strokes of that marker only fill what
    the frame's leave bare (hatched anew there, with the frame's nibs and slant)."""
    if not parts:
        return list(frame_lines)
    size = next((p.size for p in parts if isinstance(p, Part) and p.size[1]), None)
    if size is None:
        raise ValueError("merge needs parts that know the size of their frame (Part.size)")
    h, w = size
    scale = float(w)
    boxes = [p.box if isinstance(p, Part) else tuple(p) for p in parts]
    colour = ("fill", "shade")
    goes_on = [{(line.marker, line.kind) for line in detail_lines if line.kind in colour
                and bool(_inside_box(line.points * scale, box).any())} for box in boxes]
    kept = list(frame_lines)
    for box, keys in zip(boxes, goes_on, strict=True):
        through = [line for line in kept if line.kind in colour and (line.marker, line.kind) in keys]
        rest = [line for line in kept if not (line.kind in colour and (line.marker, line.kind) in keys)]
        kept = through + clip_lines(rest, box, scale, keep_inside=False)
    new = [line for line in detail_lines if not (line.kind in colour and any((line.marker, line.kind) in keys
                                                                              for keys in goes_on))]
    for key in sorted(set().union(*goes_on)):        # where the part wants a marker the frame's strokes left bare
        want = stamp([x for x in detail_lines if (x.marker, x.kind) == key], h, w, scale)
        have = stamp([x for x in kept if (x.marker, x.kind) == key], h, w, scale)
        gap = want & ~have
        if not bool(gap.any()):
            continue
        sample = next(x for x in detail_lines if (x.marker, x.kind) == key)
        pigments = torch.zeros(key[0] + 1, 3)
        pigments[key[0]] = torch.tensor(sample.colour)
        masks = {key[0]: gap} if key[1] == "fill" else {}
        shades = {key[0]: gap} if key[1] == "shade" else {}
        new += colour_lines(torch.zeros(h, w, dtype=torch.long), torch.zeros(h, w, dtype=torch.bool), pigments,
                            scale, masks, shades)
    return _assemble(kept, new)


# ---------------------------------------------------------------------------------------- following --
def _similarity(a: float, b: float, tx: float, ty: float) -> torch.Tensor:
    return torch.tensor([[a, -b, tx], [b, a, ty]], dtype=torch.float32)


def fit_similarity(points: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None = None,
                   iterations: int = 10, spread: float = 0.5) -> torch.Tensor:
    """The rotation, uniform scale and shift that carry ``points (N, 2)`` onto ``targets (N, 2)`` best, by
    iteratively reweighted least squares (Huber's weights, ``spread`` pixels): points that do not move with the
    rest (another thing, a bad match) lose their say. Returns the ``(2, 3)`` matrix ``[[a, -b, tx], [b, a, ty]]``."""
    p, q = points.double(), targets.double()
    wt = torch.ones(len(p), dtype=torch.float64) if weight is None else weight.double().clamp(min=0)
    centre = (p * wt[:, None]).sum(0) / wt.sum().clamp(min=1e-12)
    x, y = (p - centre).unbind(-1)
    zero, one = torch.zeros_like(x), torch.ones_like(x)
    design = torch.cat([torch.stack([x, -y, one, zero], -1), torch.stack([y, x, zero, one], -1)])  # (2N, 4)
    rhs = torch.cat([q[:, 0], q[:, 1]])
    robust = torch.ones_like(wt)
    theta = torch.tensor([1.0, 0.0, float(centre[0]), float(centre[1])], dtype=torch.float64)
    for _ in range(iterations):
        ww = torch.cat([wt * robust, wt * robust])
        normal = design.T @ (design * ww[:, None]) + 1e-9 * torch.eye(4, dtype=torch.float64)
        theta = torch.linalg.solve(normal, design.T @ (rhs * ww))
        res = (design @ theta - rhs).view(2, -1).norm(dim=0)
        robust = torch.where(res <= spread, torch.ones_like(res), spread / res.clamp(min=1e-12))
    a, b, tx, ty = theta.tolist()
    # back from the centred points: q = M (p - c) + t  ->  q = M p + (t - M c)
    cx, cy = float(centre[0]), float(centre[1])
    return _similarity(a, b, tx - (a * cx - b * cy), ty - (b * cx + a * cy))


def _apply(matrix: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    return pts @ matrix[:, :2].T + matrix[:, 2]


def _gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p = F.pad(x[None], (1, 1, 1, 1), mode="replicate")[0]
    return (p[:, 1:-1, 2:] - p[:, 1:-1, :-2]) / 2.0, (p[:, 2:, 1:-1] - p[:, :-2, 1:-1]) / 2.0


def _align(before: torch.Tensor, after: torch.Tensor, box, matrix: torch.Tensor, iterations: int = 12) -> torch.Tensor:
    """``matrix`` refined so that ``after`` (OKLab ``(3, H, W)``), looked at through it, matches ``before`` inside
    ``box``, by Gauss and Newton's method on the pictures themselves (Lucas and Kanade's image alignment with a
    rotation, a scale and a shift), first on blurred pictures, then on sharper ones. Pixels that do not match
    after the move count less (Huber), so a background that stays still or a part that changes shape does not
    pull the fit."""
    x0, y0, x1, y1 = (int(v) for v in box)
    ys, xs = torch.meshgrid(torch.arange(y0, y1, dtype=torch.float32) + 0.5,
                            torch.arange(x0, x1, dtype=torch.float32) + 0.5, indexing="ij")
    p = torch.stack([xs.flatten(), ys.flatten()], -1)
    centre = p.mean(0)
    pc = (p - centre).double()
    channel = torch.tensor([1.0, 1.5, 1.5], dtype=torch.float64)
    m = matrix.double()
    a, b = float(m[0, 0]), float(m[1, 0])
    q = (m[:, :2] @ centre.double() + m[:, 2])               # where the box's centre goes
    for sigma in (1.2, 0.6):
        ref_img = gaussian_blur(before, sigma)
        img = gaussian_blur(after, sigma)
        gx, gy = _gradients(img)
        stack = torch.cat([img, gx, gy])
        ref = sample(ref_img, p).double()                    # (N, 3)
        for _ in range(iterations):
            rot = torch.tensor([[a, -b], [b, a]], dtype=torch.float64)
            target = (pc @ rot.T + q).float()
            got = sample(stack, target).double()
            value, ix, iy = got[:, :3], got[:, 3:6], got[:, 6:9]
            res = (value - ref) * channel                    # (N, 3)
            jac = torch.stack([ix * pc[:, :1] + iy * pc[:, 1:], -ix * pc[:, 1:] + iy * pc[:, :1], ix, iy], -1)
            jac = jac * channel[None, :, None]               # (N, 3, 4)
            size = res.norm(dim=1)
            robust = torch.where(size <= 0.04, torch.ones_like(size), 0.04 / size.clamp(min=1e-12))
            jw = jac * robust[:, None, None]
            normal = torch.einsum("nci,ncj->ij", jw, jac) + 1e-8 * torch.eye(4, dtype=torch.float64)
            step = torch.linalg.solve(normal, -torch.einsum("nci,nc->i", jw, res))
            a, b = a + float(step[0]), b + float(step[1])
            q = q + step[2:]
            if float(step[2:].norm()) < 1e-3 and float(step[:2].abs().max()) < 1e-5:
                break
    rot = torch.tensor([[a, -b], [b, a]], dtype=torch.float64)
    shift = q - rot @ centre.double()
    return _similarity(a, b, float(shift[0]), float(shift[1]))


def _box_int(box, h: int, w: int) -> tuple:
    x0, y0 = max(0, int(math.floor(box[0]))), max(0, int(math.floor(box[1])))
    return x0, y0, max(x0 + 1, min(w, int(math.ceil(box[2])))), max(y0 + 1, min(h, int(math.ceil(box[3]))))


def track(part_box, before_frame: torch.Tensor, frame: torch.Tensor, flow: torch.Tensor | None = None) -> tuple:
    """How the part in ``part_box`` (pixels) of ``before_frame`` moved as a whole into ``frame`` (both sRGB
    ``(3, H, W)``): the similarity transform ``(2, 3)`` (pixels to pixels) fitted to the optical flow inside the
    box (``flow (2, H, W)`` from ``before_frame`` to ``frame``, measured round the box when not given) by
    :func:`fit_similarity` and refined on the pictures (:func:`_align`), and how much the part itself changed
    after that move: the share of the box's pixels that still differ clearly (OKLab distance above 0.1, with a
    pixel's leeway), 0..1. Compare it with :data:`REDRAW`."""
    h, w = frame.shape[-2:]
    x0, y0, x1, y1 = _box_int(part_box, h, w)
    before_lab, frame_lab = image_to_oklab(before_frame.float()), image_to_oklab(frame.float())
    if flow is None:                                  # the flow round the box only: much quicker than everywhere
        m = max(12, int(0.6 * max(x1 - x0, y1 - y0)))
        wx0, wy0, wx1, wy1 = max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m)
        local = optical_flow(before_frame[:, wy0:wy1, wx0:wx1], frame[:, wy0:wy1, wx0:wx1])
        flow_box = local[:, y0 - wy0 : y1 - wy0, x0 - wx0 : x1 - wx0]
    else:
        flow_box = flow[:, y0:y1, x0:x1]
    ys, xs = torch.meshgrid(torch.arange(y0, y1, dtype=torch.float32) + 0.5,
                            torch.arange(x0, x1, dtype=torch.float32) + 0.5, indexing="ij")
    p = torch.stack([xs.flatten(), ys.flatten()], -1)
    gx, gy = _gradients(gaussian_blur(before_lab, 1.0))
    texture = gaussian_blur((gx ** 2 + gy ** 2).sum(0), 1.5)[y0:y1, x0:x1].flatten()
    matrix = fit_similarity(p, p + flow_box.reshape(2, -1).T, texture + 1e-3 * float(texture.max() + 1e-9))
    matrix = _align(before_lab, frame_lab, (x0, y0, x1, y1), matrix)
    seen = sample(frame_lab, _apply(matrix, p)).T.reshape(3, y1 - y0, x1 - x0)
    padded = F.pad(before_lab[None], (1, 1, 1, 1), mode="replicate")[0]
    best = torch.full((y1 - y0, x1 - x0), float("inf"))
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            shifted = padded[:, y0 + dy : y1 + dy, x0 + dx : x1 + dx]
            best = torch.minimum(best, (seen - shifted).norm(dim=0))
    return matrix, float((best > 0.1).float().mean())


def move_lines(lines: list, matrix: torch.Tensor, width: float) -> list:
    """``lines`` (board widths of a frame ``width`` pixels wide) carried by the similarity ``matrix (2, 3)``
    (pixels): every point moved alike, so each shape stays exactly the same; each line keeps its uid."""
    return [line.moved(_apply(matrix, line.points * width) / width) for line in lines]


def move_box(box, matrix: torch.Tensor) -> tuple:
    """The box (pixels) round ``box`` carried by ``matrix``."""
    corners = torch.tensor([[box[0], box[1]], [box[2], box[1]], [box[0], box[3]], [box[2], box[3]]],
                           dtype=torch.float32)
    moved = _apply(matrix, corners)
    return (float(moved[:, 0].min()), float(moved[:, 1].min()), float(moved[:, 0].max()), float(moved[:, 1].max()))


def follow(lines: list, part_box, before_frame: torch.Tensor, frame: torch.Tensor,
           flow: torch.Tensor | None = None) -> tuple[list, float]:
    """A part's drawing carried rigidly into the next frame: ``lines`` (the part's lines, board widths of the
    frame) moved by the rotation, uniform scale and shift that best explain how the part in ``part_box``
    (pixels) moved from ``before_frame`` to ``frame`` (:func:`track`), with no change of shape and their uids
    kept, and how much the part itself changed after that move (0..1; above :data:`REDRAW` it is worth drawing
    again: a blink, a mouth opening)."""
    matrix, change = track(part_box, before_frame, frame, flow)
    return move_lines(lines, matrix, float(frame.shape[-1])), change


__all__ = ["KINDS", "REDRAW", "Part", "accents", "cel_shapes", "clip_lines", "clip_polyline", "clip_ring",
           "components", "detail_map", "draw_crop", "draw_details", "enlarge", "find_details", "fit_similarity",
           "follow", "joining", "mask_rings", "merge", "move_box", "move_lines", "on_board", "small_shapes", "track"]
