# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Drawing with pens on a whiteboard, the way a person draws.

A picture is drawn, not filtered. The lines a person would draw are found first: the picture is seen in planes
(:func:`helix_lm.paint.mind.simplify`), the borders between the planes are thinned to one pixel and followed
like a pen follows them, through crossings where the line goes on straight, and broken where an arm would lift.
Then the colour goes on with markers, one marker at a time, back and forth inside each shape, light colours
first, and where a shape is darker than its marker a second layer goes across the first. Each line is a list of
points with the pressure of the pen at each, a tool and a colour, so the drawing can be replayed stroke by
stroke at the speed a hand would move.

Tools (:data:`TOOLS`): ``pencil`` (grey, broken by the surface's tooth), ``pen`` (a fineliner), ``brush pen``
(thin to thick with pressure), ``marker`` and ``fine marker`` (see-through colour that darkens where it
overlaps once dry, a little ink pooled where the nib rests). Strokes of one marker laid one after another
blend while wet; a layer gone over it later darkens it. Surfaces: ``whiteboard`` and ``paper``.

The hand is not perfectly steady: every time a line is drawn it wavers a little and may run a pixel past its
end. Lines kept from one frame to the next are drawn again with a new waver, which is the slight "boil" of
hand-drawn animation.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .color import extract_palette, image_to_oklab, linear_to_oklab, linear_to_srgb, srgb_to_linear
from .see import gaussian_blur

AA = 0.8                       # pixels over which an edge goes from ink to no ink


@dataclass(frozen=True)
class Tool:
    name: str
    width: float               # the nib, as a share of the board's width, at middle pressure
    opacity: float
    multiply: bool             # marker ink is see-through: it darkens what is under it
    taper: float               # how thin the ends of a line get (0: not at all, 1: to nothing)
    grain: float               # how much the surface's tooth breaks the line (pencil)
    swell: float               # how much pressure widens the line
    speed: float               # how fast a hand moves it, in board widths a second
    ink: tuple = (0.07, 0.07, 0.09)


TOOLS = {
    "pencil": Tool("pencil", 0.0024, 0.62, False, 0.5, 0.85, 0.5, 1.1, (0.30, 0.30, 0.33)),
    "pen": Tool("pen", 0.0021, 0.96, False, 0.35, 0.0, 0.35, 0.55),
    "brush pen": Tool("brush pen", 0.0032, 0.96, False, 0.9, 0.0, 1.1, 0.5),
    "marker": Tool("marker", 0.016, 0.86, True, 0.05, 0.06, 0.1, 0.9),
    "fine marker": Tool("fine marker", 0.0062, 0.86, True, 0.1, 0.05, 0.15, 0.8),
    # cel paint: opaque, filled exactly up to the ink lines and laid under them, as on the back of a cel
    "brush": Tool("brush", 0.02, 1.0, False, 0.0, 0.0, 0.0, 1.6),
}
TOOL_NAMES = tuple(TOOLS)
# sketch and line are ink; fill and shade are marker layers; cel, cel-shadow and cel-light are cel paint (a
# flat base colour, the hard-edged shadow shape over it, the highlight shape): new kinds go at the end, so that
# older .pinxa files still read
KINDS = ("sketch", "line", "fill", "shade", "cel", "cel-shadow", "cel-light")
CELS = ("cel", "cel-shadow", "cel-light")
_uids = itertools.count(1)
SURFACES = {"whiteboard": (0.955, 0.958, 0.962), "paper": (0.968, 0.955, 0.922)}


class Line:
    """One stroke of a tool: ``points (K, 2)`` in board widths (x, y), the pen's ``pressure (K,)`` 0..1, the
    ``tool`` (a name of :data:`TOOLS`), the ``colour`` (sRGB), for colour the number of the ``marker`` on the
    palette (-1 for ink), its ``kind`` (see :data:`KINDS`) and its ``age`` (for how many frames it has been
    traced over from the frame before; 0 for a line drawn new). ``uid`` names the line across frames: a line
    traced onto the next frame keeps it, so its path through the animation can be followed (and smoothed).
    A cel shape (``kind`` in :data:`CELS`) is a filled outline: ``points`` are its rings one after the other
    (the outer edge, then any holes) and ``rings`` their numbers of points; it is filled exactly, even-odd."""

    __slots__ = ("age", "colour", "kind", "marker", "points", "pressure", "rings", "tool", "uid")

    def __init__(self, points, pressure, tool: str, colour, marker: int = -1, kind: str = "line", age: int = 0,
                 uid: int | None = None, rings=None):
        self.points = torch.as_tensor(points, dtype=torch.float32)
        self.pressure = torch.as_tensor(pressure, dtype=torch.float32)
        self.tool, self.colour = tool, tuple(float(c) for c in colour)
        self.marker, self.kind, self.age = int(marker), kind, int(age)
        self.uid = next(_uids) if uid is None else int(uid)
        self.rings = None if rings is None else tuple(int(n) for n in rings)

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def length(self) -> float:
        if self.rings is not None:                   # a cel shape: the length of its edge
            return float(sum(ring.norm(dim=-1).sum() for ring in self.ring_points(close=True)))
        return float((self.points[1:] - self.points[:-1]).norm(dim=-1).sum()) if len(self) > 1 else 0.0

    def ring_points(self, close: bool = False) -> list:
        """A cel shape's rings as ``(K, 2)`` points each (with ``close``, the steps from point to point instead,
        the last one back to the first)."""
        out, at = [], 0
        for n in self.rings or (len(self),):
            ring = self.points[at : at + n]
            out.append(torch.roll(ring, -1, 0) - ring if close else ring)
            at += n
        return out

    def moved(self, points, pressure=None, age: int | None = None) -> Line:
        return Line(points, self.pressure if pressure is None else pressure, self.tool, self.colour, self.marker,
                    self.kind, self.age + 1 if age is None else age, self.uid, self.rings)

    def reversed(self) -> Line:
        if self.rings is not None:                   # a filled shape has no direction to draw it in
            return self
        return Line(self.points.flip(0), self.pressure.flip(0), self.tool, self.colour, self.marker, self.kind,
                    self.age, self.uid)

    def __repr__(self) -> str:
        return f"Line({self.kind}, {self.tool}, {len(self)} points, marker {self.marker}, age {self.age})"


def fill_rings(points: torch.Tensor, rings, height: int, width: int, ss: int = 4):
    """How much of each pixel the shape with these rings covers (even-odd, so a ring inside another is a
    hole), exactly up to its edge: ``points (N, 2)`` in pixels (x, y), ``rings`` the numbers of points of each
    ring. Returns ``(y0, x0, coverage (h, w))`` for the part of the board it touches, or None. Each pixel is
    looked at ``ss`` by ``ss`` times, so an edge is smooth."""
    pts = torch.as_tensor(points, dtype=torch.float32)
    if len(pts) < 3:
        return None
    x0 = max(0, int(math.floor(float(pts[:, 0].min()))))
    y0 = max(0, int(math.floor(float(pts[:, 1].min()))))
    x1 = min(width, int(math.ceil(float(pts[:, 0].max()))) + 1)
    y1 = min(height, int(math.ceil(float(pts[:, 1].max()))) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    a_list, b_list, at = [], [], 0
    for n in rings or (len(pts),):
        ring = pts[at : at + n]
        at += n
        if len(ring) >= 2:
            a_list.append(ring)
            b_list.append(torch.roll(ring, -1, 0))
    a, b = torch.cat(a_list), torch.cat(b_list)
    a = (a - torch.tensor([x0, y0], dtype=torch.float32)) * ss        # in samples of the box
    b = (b - torch.tensor([x0, y0], dtype=torch.float32)) * ss
    rows, cols = (y1 - y0) * ss, (x1 - x0) * ss
    lo, hi = torch.minimum(a[:, 1], b[:, 1]), torch.maximum(a[:, 1], b[:, 1])
    j0 = torch.ceil(lo - 0.5).clamp(0, rows).long()                    # the sample rows (at j + 0.5) it crosses
    j1 = torch.ceil(hi - 0.5).clamp(0, rows).long()
    count = (j1 - j0).clamp(min=0)
    total = int(count.sum())
    if total == 0:
        return None
    edge = torch.repeat_interleave(torch.arange(len(a)), count)
    j = j0[edge] + torch.arange(total) - torch.repeat_interleave(torch.cumsum(count, 0) - count, count)
    ya, yb, xa, xb = a[edge, 1], b[edge, 1], a[edge, 0], b[edge, 0]
    x = xa + (j.float() + 0.5 - ya) * (xb - xa) / (yb - ya)
    i = torch.ceil(x - 0.5).clamp(0, cols).long()                      # samples from here on are across it
    flips = torch.zeros(rows * (cols + 1), dtype=torch.int32)
    flips.index_put_((j * (cols + 1) + i,), torch.ones(total, dtype=torch.int32), accumulate=True)
    inside = (flips.view(rows, cols + 1).cumsum(1)[:, :cols] % 2).float()
    cov = F.avg_pool2d(inside[None, None], ss)[0, 0]
    return y0, x0, cov


# ------------------------------------------------------------------------------------------- rendering --
def _hash(x: torch.Tensor) -> torch.Tensor:
    return torch.frac(torch.sin(x * 12.9898 + 78.233) * 43758.5453)


def surface(height: int, width: int, kind: str = "whiteboard") -> tuple[torch.Tensor, torch.Tensor]:
    """A bare surface as linear RGB ``(3, H, W)`` and its tooth ``(H, W)`` in 0..1 (what a pencil catches)."""
    base = srgb_to_linear(torch.tensor(SURFACES.get(kind, SURFACES["whiteboard"])))
    ys = torch.linspace(0.0, 1.0, height)[:, None]
    xs = torch.linspace(0.0, 1.0, width)[None, :]
    yi, xi = torch.arange(height, dtype=torch.float32)[:, None], torch.arange(width, dtype=torch.float32)[None, :]
    noise = _hash(yi * 0.7316 + xi * 1.913)
    if kind == "paper":
        fibres = 0.5 + 0.5 * torch.sin(xi * 0.37 + 3.0 * _hash(yi * 0.13))
        tooth = (0.65 * noise + 0.35 * fibres).clamp(0.0, 1.0)
        light = 1.0 - 0.025 * (noise - 0.5)
    else:                                                  # glossy: a soft sheen across the board, no tooth
        tooth = 0.5 + 0.3 * (noise - 0.5)
        light = 1.0 + 0.018 * torch.exp(-((xs - 0.3 - 0.4 * ys) ** 2) / 0.02) - 0.012 * ys
    return base[:, None, None] * light, tooth


def _waver(points: torch.Tensor, amount: float, seed: float, overshoot: float) -> torch.Tensor:
    """``points (K, 2)`` in pixels, as an unsteady hand draws them: a slow sideways waver of about ``amount``
    pixels, and the ends run on by up to ``overshoot`` pixels."""
    if len(points) < 2 or (amount <= 0 and overshoot <= 0):
        return points
    step = (points[1:] - points[:-1]).norm(dim=-1)
    s = torch.cat([torch.zeros(1), step.cumsum(0)])
    total = float(s[-1]) + 1e-6
    tangent = torch.cat([points[1:2] - points[:1], points[2:] - points[:-2], points[-1:] - points[-2:-1]])
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    normal = torch.stack([-tangent[:, 1], tangent[:, 0]], -1)
    r = _hash(torch.tensor([seed, seed + 1.7, seed + 3.1, seed + 5.3]))
    wave = 0.6 * torch.sin(2 * math.pi * (s / 90.0 + r[0])) + 0.4 * torch.sin(2 * math.pi * (s / 31.0 + r[1]))
    out = points + normal * (amount * wave)[:, None]
    if overshoot > 0 and total > 12:
        out = torch.cat([out[:1] - tangent[:1] * overshoot * float(r[2]), out, out[-1:] + tangent[-1:] * overshoot
                         * float(r[3])])
    return out


def _cover(pts: torch.Tensor, radius: torch.Tensor, height: int, width: int, aa: float = AA, chunk: int = 12):
    """How much of each pixel a line of half widths ``radius (K,)`` along ``pts (K, 2)`` (pixels) covers:
    ``(y0, x0, coverage (h, w))`` over the line's box, or None when it is off the board."""
    reach = float(radius.max()) + 2.0 * aa + 1.0
    x0, x1 = max(0, int(float(pts[:, 0].min()) - reach)), min(width, int(float(pts[:, 0].max()) + reach) + 1)
    y0, y1 = max(0, int(float(pts[:, 1].min()) - reach)), min(height, int(float(pts[:, 1].max()) + reach) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    cov = torch.zeros(y1 - y0, x1 - x0)
    if len(pts) == 1:
        pts, radius = torch.cat([pts, pts + 0.01]), torch.cat([radius, radius])
    for i in range(0, len(pts) - 1, chunk):
        a, b = pts[i : i + chunk], pts[i + 1 : i + chunk + 1]
        a, ra, rb = a[: len(b)], radius[i : i + chunk][: len(b)], radius[i + 1 : i + chunk + 1]
        m = float(torch.maximum(ra, rb).max()) + 2.0 * aa + 1.0
        both = torch.cat([a, b])
        cx0, cx1 = max(x0, int(float(both[:, 0].min()) - m)), min(x1, int(float(both[:, 0].max()) + m) + 1)
        cy0, cy1 = max(y0, int(float(both[:, 1].min()) - m)), min(y1, int(float(both[:, 1].max()) + m) + 1)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        gy, gx = torch.meshgrid(torch.arange(cy0, cy1, dtype=torch.float32) + 0.5,
                                torch.arange(cx0, cx1, dtype=torch.float32) + 0.5, indexing="ij")
        p = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)[:, None]          # (P, 1, 2)
        ab = (b - a)[None]
        t = (((p - a[None]) * ab).sum(-1) / ab.pow(2).sum(-1).clamp(min=1e-6)).clamp(0.0, 1.0)
        d = (p - a[None] - t[..., None] * ab).norm(dim=-1)
        r = ra[None] + (rb - ra)[None] * t
        c = ((r - d) / aa + 0.5).clamp(0.0, 1.0).max(1).values.reshape(cy1 - cy0, cx1 - cx0)
        sub = cov[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
        torch.maximum(sub, c, out=sub)
    return y0, x0, cov


def _fine(pts: torch.Tensor, pressure: torch.Tensor, most: float = 3.0):
    """Points every ``most`` pixels at most along a line (pixels), the pressure carried between them, and for
    each original point its number among the new ones."""
    if len(pts) < 2:
        return pts, pressure, torch.arange(len(pts))
    seg = (pts[1:] - pts[:-1]).norm(dim=-1)
    n = (seg / most).ceil().clamp(min=1).long()
    if int(n.max()) == 1:
        return pts, pressure, torch.arange(len(pts))
    steps = torch.repeat_interleave(torch.arange(len(seg)), n)
    frac = torch.cat([torch.arange(int(k)) / int(k) for k in n.tolist()])[:, None]
    dense = torch.cat([pts[:-1][steps] + (pts[1:] - pts[:-1])[steps] * frac, pts[-1:]])
    press = torch.cat([pressure[:-1][steps] + (pressure[1:] - pressure[:-1])[steps] * frac[:, 0], pressure[-1:]])
    where = torch.cat([torch.zeros(1, dtype=torch.long), n.cumsum(0)])
    return dense, press, where


def radii(line: Line, scale: float, upto: int | None = None, points: torch.Tensor | None = None,
          pressure: torch.Tensor | None = None) -> torch.Tensor:
    """The half widths of a line in pixels at each point: the nib, widened by pressure, thinned at the ends
    where the pen lands and lifts. ``points`` (board widths) and ``pressure`` stand in for the line's own."""
    tool = TOOLS[line.tool]
    pts = line.points if points is None else points
    p = line.pressure if pressure is None else pressure
    p = p[:upto] if upto else p
    step = (pts[1:] - pts[:-1]).norm(dim=-1) * scale
    s = torch.cat([torch.zeros(1), step.cumsum(0)])
    total = float(s[-1])
    ramp = max(3.0, 0.15 * total)
    end = (torch.minimum(s, total - s) / ramp).clamp(0.0, 1.0)
    if upto:
        end = end[:upto]
    r = 0.5 * tool.width * scale * (1.0 - 0.5 * tool.swell + tool.swell * p)
    return (r * ((1.0 - tool.taper) + tool.taper * end.sqrt())).clamp(min=0.35)


class Board:
    """A whiteboard (or a sheet of paper) that lines are drawn on, in order. ``scale`` is how many pixels a
    board width is (the board's width in pixels when not given) and ``origin`` where the drawing's corner
    lies, so that a drawing can be put on a larger board with a margin."""

    def __init__(self, width: int, height: int, kind: str = "whiteboard", scale: float | None = None,
                 origin: tuple = (0.0, 0.0), waver: float = 0.35, seed: int = 0) -> None:
        self.width, self.height, self.kind = width, height, kind
        self.scale = float(width if scale is None else scale)
        self.origin = torch.tensor(origin, dtype=torch.float32)
        self.canvas, self.tooth = surface(height, width, kind)
        self.waver, self.seed, self.count = waver, seed, 0
        self.wet, self.wet_key = torch.zeros(height, width), None     # marker ink not yet dry
        # the ink lines alone (colour times cover, and cover), so that cel paint can go under them:
        # canvas = paint * (1 - ink_cover) + ink
        self.ink, self.ink_cover = torch.zeros(3, height, width), torch.zeros(height, width)

    def pixels(self, line: Line) -> torch.Tensor:
        return line.points * self.scale + self.origin

    def _paint(self, line: Line, upto: int | None, canvas: torch.Tensor | None) -> None:
        """Cel paint: the shape filled exactly, opaque, under the ink lines already drawn. Drawn part way
        (``upto`` of its points), the brush has gone over that share of it, in rows across the shape."""
        got = fill_rings(self.pixels(line), line.rings, self.height, self.width)
        if got is None:
            return
        y0, x0, cov = got
        h, w = cov.shape
        if upto is not None and upto < len(line):
            share = upto / max(1, len(line))
            yy = torch.arange(h, dtype=torch.float32)[:, None]
            xx = torch.arange(w, dtype=torch.float32)[None, :]
            along = (0.77 * xx + 0.64 * yy) / max(1.0, 0.77 * w + 0.64 * h)
            cov = cov * ((share - along) * max(h, w) / 3.0 + 0.5).clamp(0.0, 1.0)
        canvas = self.canvas if canvas is None else canvas
        region = canvas[:, y0 : y0 + h, x0 : x0 + w]
        ink = self.ink[:, y0 : y0 + h, x0 : x0 + w]
        under = 1.0 - self.ink_cover[y0 : y0 + h, x0 : x0 + w]
        paint = srgb_to_linear(torch.tensor(line.colour))[:, None, None]
        region.mul_(1.0 - cov).add_(cov * (ink + under * paint))

    def _ink(self, line: Line, upto: int | None = None, canvas: torch.Tensor | None = None) -> None:
        if line.kind in CELS:
            self._paint(line, upto, canvas)
            return
        tool = TOOLS[line.tool]
        key = (line.marker, line.kind, line.colour)
        commit = canvas is None                              # drawn for good, not only looked at
        if commit and tool.multiply and key != self.wet_key:   # another marker: what was there has dried
            self.wet.zero_()
            self.wet_key = key
        # points every few pixels, so that the thinning toward the ends follows the line, not only its points
        dense, press, where = _fine(self.pixels(line), line.pressure)
        upto = int(where[upto - 1]) + 1 if upto else None
        pts = dense[:upto] if upto else dense
        r = radii(line, self.scale, upto, (dense - self.origin) / self.scale, press)
        if self.waver > 0 and line.kind in ("line", "sketch"):
            amount = self.waver * (2.0 if line.kind == "sketch" else 1.0) * self.scale / 768.0
            wavered = _waver(pts, amount, self.seed * 97.0 + self.count * 1.618, 1.2 * amount if not upto else 0.0)
            if len(wavered) != len(pts):                     # it ran on past its ends: the ends are thin
                r = torch.cat([r[:1] * 0.6, r, r[-1:] * 0.6])
            pts = wavered
        got = _cover(pts, r, self.height, self.width)
        if got is None:
            return
        y0, x0, cov = got
        h, w = cov.shape
        canvas = self.canvas if canvas is None else canvas
        region = canvas[:, y0 : y0 + h, x0 : x0 + w]
        if tool.grain > 0:                                   # the surface's tooth catches the pencil, or not
            tooth = self.tooth[y0 : y0 + h, x0 : x0 + w]
            p = float(line.pressure.mean())
            cov = cov * (1.0 - tool.grain * (1.0 - tooth) * (1.15 - p)).clamp(0.0, 1.0)
        alpha = tool.opacity * cov
        if tool.multiply and len(pts) > 1:                   # ink pools where the nib rested at the start
            start = ((torch.arange(h)[:, None] + y0 + 0.5 - pts[0, 1]) ** 2
                     + (torch.arange(w)[None, :] + x0 + 0.5 - pts[0, 0]) ** 2)
            alpha = alpha * (1.0 + 0.12 * torch.exp(-start / (1.4 * float(r[0]) ** 2 + 1e-6)))
        alpha = alpha.clamp(0.0, 1.0)
        ink = srgb_to_linear(torch.tensor(line.colour))[:, None, None]
        if tool.multiply:
            # wet on wet the ink of one marker does not build up: only what this stroke adds beyond the ink
            # already laid there darkens the board
            wet = self.wet[y0 : y0 + h, x0 : x0 + w] if key == self.wet_key else torch.zeros_like(alpha)
            more = torch.maximum(alpha, wet)
            region.mul_((1.0 - more * (1.0 - ink)) / (1.0 - wet * (1.0 - ink)).clamp(min=1e-4))
            if commit:
                self.wet[y0 : y0 + h, x0 : x0 + w] = more
        else:
            region.mul_(1.0 - alpha).add_(ink * alpha)
            if commit and line.kind in ("line", "sketch"):
                self.ink[:, y0 : y0 + h, x0 : x0 + w].mul_(1.0 - alpha).add_(ink * alpha)
                cover = self.ink_cover[y0 : y0 + h, x0 : x0 + w]
                cover.mul_(1.0 - alpha).add_(alpha)

    def draw(self, line: Line) -> None:
        self._ink(line)
        self.count += 1

    def settle(self) -> None:
        """Make what is on the board one layer under what comes next: later cel paint then covers its ink too
        (a character's cel laid over the background art)."""
        self.ink.zero_()
        self.ink_cover.zero_()
        self.wet.zero_()
        self.wet_key = None

    def peek(self, line: Line, upto: int) -> torch.Tensor:
        """The board as it looks with the first ``upto`` points of ``line`` drawn, without drawing it."""
        canvas = self.canvas.clone()
        if upto >= 2:
            self._ink(line, upto, canvas)
        return canvas

    def picture(self, canvas: torch.Tensor | None = None) -> torch.Tensor:
        return linear_to_srgb((self.canvas if canvas is None else canvas).clamp(0.0, 1.0))


def render(lines: list, width: int, height: int, kind: str = "whiteboard", waver: float = 0.35, seed: int = 0,
           scale: float | None = None, origin: tuple = (0.0, 0.0), under: list | None = None) -> torch.Tensor:
    """A drawing as an sRGB picture ``(3, height, width)``. ``under`` (lines) is drawn first as a layer of its
    own, such as the background art under a character's cel."""
    board = Board(width, height, kind, scale, origin, waver, seed)
    if under:
        for line in under:
            board.draw(line)
        board.settle()
    for line in lines:
        board.draw(line)
    return board.picture()


# ------------------------------------------------------------------------------------------- seeing --
def _minpool(x: torch.Tensor, k: int) -> torch.Tensor:
    return -F.max_pool2d(-x[None, None], k, 1, k // 2)[0, 0]


def thin(mask: torch.Tensor) -> torch.Tensor:
    """Lines one pixel wide from a mask of thicker ones (Zhang and Suen, 1984)."""
    m = mask.to(torch.uint8).clone()
    while True:
        changed = False
        for step in (0, 1):
            p = F.pad(m, (1, 1, 1, 1))
            n = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]
            count = sum(x.int() for x in n)
            turns = sum(((n[i] == 0) & (n[(i + 1) % 8] == 1)).int() for i in range(8))
            if step == 0:
                c = (n[0] * n[2] * n[4] == 0) & (n[2] * n[4] * n[6] == 0)
            else:
                c = (n[0] * n[2] * n[6] == 0) & (n[0] * n[4] * n[6] == 0)
            drop = (m == 1) & (count >= 2) & (count <= 6) & (turns == 1) & c
            if bool(drop.any()):
                m[drop] = 0
                changed = True
        if not changed:
            return m.bool()


def line_strength(planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """How strongly the picture changes at each pixel, in light or in colour, and which way (radians) it
    changes fastest: both ``(H, W)``."""
    lab = image_to_oklab(planes)
    lab = torch.stack([gaussian_blur(c, 1.0) for c in lab])
    gx, gy = torch.zeros_like(lab), torch.zeros_like(lab)
    gx[:, :, 1:-1] = (lab[:, :, 2:] - lab[:, :, :-2]) / 2.0
    gy[:, 1:-1] = (lab[:, 2:] - lab[:, :-2]) / 2.0
    weight = torch.tensor([1.0, 1.6, 1.6])[:, None, None]
    jxx, jyy, jxy = (weight * gx * gx).sum(0), (weight * gy * gy).sum(0), (weight * gx * gy).sum(0)
    angle = 0.5 * torch.atan2(2.0 * jxy, jxx - jyy)
    big = 0.5 * (jxx + jyy + ((jxx - jyy) ** 2 + 4.0 * jxy ** 2).sqrt())
    return big.clamp(min=0.0).sqrt(), angle


def find_lines(planes: torch.Tensor, detail: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """The lines a person would draw: one pixel wide where the planes meet (the middle of each change kept,
    weak changes only where they join strong ones), and how strong each is. ``detail`` 0..1: more detail
    draws weaker borders too."""
    strength, angle = line_strength(planes)
    dy, dx = torch.sin(angle), torch.cos(angle)
    h, w = strength.shape
    gy, gx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")

    def at(oy, ox):
        grid = torch.stack([(gx + ox) / max(1, w - 1) * 2 - 1, (gy + oy) / max(1, h - 1) * 2 - 1], -1)[None]
        return F.grid_sample(strength[None, None], grid, mode="bilinear", padding_mode="border",
                             align_corners=True)[0, 0]

    crest = (strength >= at(dy, dx)) & (strength >= at(-dy, -dx))
    high = 0.032 * (1.35 - detail)
    low = 0.45 * high
    strong, weak = crest & (strength >= high), crest & (strength >= low)
    for _ in range(40):                                    # weak lines that hang on to strong ones
        grown = weak & (F.max_pool2d(strong[None, None].float(), 3, 1, 1)[0, 0] > 0)
        if bool((grown == strong).all()):
            break
        strong = grown
    skeleton = thin(F.max_pool2d(strong[None, None].float(), 3, 1, 1)[0, 0].bool() & (strength >= low))
    return skeleton, strength


ORTHOGONAL = ((0, 1), (1, 0), (0, -1), (-1, 0))
DIAGONAL = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def trace(skeleton: torch.Tensor, least: int = 6) -> list:
    """Follow one-pixel lines the way a pen does: from loose ends first, through crossings where the line goes
    on most nearly straight, then the branches from where they leave a line, then closed loops. Returns the
    paths as ``(K, 2)`` tensors of pixel centres (x, y), each at least ``least`` pixels long."""
    ys, xs = skeleton.nonzero(as_tuple=True)
    on = set(zip(ys.tolist(), xs.tolist(), strict=True))
    seen: set = set()

    def free(y, x):
        around = [(y + a, x + b) for a, b in ORTHOGONAL + DIAGONAL]
        return [p for p in around if p in on and p not in seen]

    def walk(path: list) -> list:
        while True:
            y, x = path[-1]
            options = free(y, x)
            if not options:
                return path
            if len(path) >= 3:                             # go on most nearly the way it was going
                back = path[max(0, len(path) - 4)]
                vy, vx = y - back[0], x - back[1]
                norm = math.hypot(vy, vx) or 1.0
                options.sort(key=lambda p: -((p[0] - y) * vy + (p[1] - x) * vx) / (norm * math.hypot(p[0] - y,
                                                                                                    p[1] - x)))
            nxt = options[0]
            seen.add(nxt)
            path.append(nxt)

    def ends_first(p):
        y, x = p
        return sum((y + a, x + b) in on for a, b in ORTHOGONAL + DIAGONAL) != 1

    paths = []
    for start in sorted(on, key=lambda p: (ends_first(p), p)):
        if start in seen:
            continue
        if ends_first(start):
            touching = [(start[0] + a, start[1] + b) for a, b in ORTHOGONAL + DIAGONAL
                        if (start[0] + a, start[1] + b) in seen]
            seen.add(start)
            path = walk(([touching[0]] if touching else []) + [start])
        else:
            seen.add(start)
            path = walk([start])
        if len(path) >= least:
            paths.append(torch.tensor([[x + 0.5, y + 0.5] for y, x in path], dtype=torch.float32))
    return paths


def smooth_path(path: torch.Tensor, spacing: float = 2.5, sigma: float = 1.2) -> torch.Tensor:
    """A traced pixel path made smooth (the stair steps of the pixels taken out) and resampled every
    ``spacing`` pixels."""
    if len(path) >= 5 and sigma > 0:
        k = torch.arange(-3, 4, dtype=torch.float32)
        k = torch.exp(-(k ** 2) / (2 * sigma ** 2))
        k = k / k.sum()
        padded = F.pad(path.T[None], (3, 3), mode="replicate")
        smooth = F.conv1d(padded, k.view(1, 1, -1).expand(2, 1, -1), groups=2)[0].T
        smooth[0], smooth[-1] = path[0], path[-1]
        path = smooth
    step = (path[1:] - path[:-1]).norm(dim=-1)
    s = torch.cat([torch.zeros(1), step.cumsum(0)])
    total = float(s[-1])
    count = max(2, int(total / spacing) + 1)
    want = torch.linspace(0.0, total, count)
    i = torch.searchsorted(s, want).clamp(1, len(s) - 1)
    t = ((want - s[i - 1]) / (s[i] - s[i - 1]).clamp(min=1e-6))[:, None]
    return path[i - 1] + (path[i] - path[i - 1]) * t


def _sample(field: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """``field (H, W)`` or ``(C, H, W)`` at points ``(K, 2)`` (pixels, x y), bilinear."""
    f = field[None] if field.dim() == 2 else field
    h, w = f.shape[-2:]
    grid = torch.stack([pts[:, 0] / w * 2 - 1, pts[:, 1] / h * 2 - 1], -1)[None, None]
    out = F.grid_sample(f[None].float(), grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0]
    return out[0] if field.dim() == 2 else out


def ink_lines(paths: list, strength: torch.Tensor, scale: float, tool: str = "pen", colour=None,
              longest: float = 0.38, kind: str = "line") -> list:
    """Traced paths -> :class:`Line` objects of ``tool``: smoothed, pressed harder along stronger borders, and
    a path longer than ``longest`` board widths broken into pieces an arm would draw in one go."""
    colour = TOOLS[tool].ink if colour is None else colour
    top = float(torch.quantile(strength.flatten()[:: max(1, strength.numel() // 50000)], 0.97)) + 1e-6
    out = []
    for path in paths:
        pts = smooth_path(path)
        pieces = max(1, math.ceil(float((pts[1:] - pts[:-1]).norm(dim=-1).sum()) / (longest * scale)))
        for part in torch.tensor_split(torch.arange(len(pts)), pieces):
            ids = torch.cat([part[:1] - 1, part]) if part[0] > 0 else part   # the pieces join
            piece = pts[ids]
            if len(piece) < 2:
                continue
            pressure = (0.45 + 0.55 * (_sample(strength, piece) / top)).clamp(0.3, 1.0)
            out.append(Line(piece / scale, pressure, tool, colour, -1, kind))
    return out


# --------------------------------------------------------------------------------------------- colour --
def markers(frames: list, count: int = 10, seed: int = 0) -> torch.Tensor:
    """The markers for a set of pictures (sRGB ``(3, H, W)`` each): ``count`` colours (sRGB, light to dark)
    that between them can colour every picture, white paper left out. The same markers colour every frame of
    an animation, which is a good part of why its character keeps its look."""
    tiles = [F.interpolate(f[None].float(), size=(96, 96), mode="bilinear", align_corners=False)[0] for f in frames]
    joined = torch.cat(tiles, -1)
    lab = image_to_oklab(joined)
    chroma = (lab[1] ** 2 + lab[2] ** 2).sqrt()
    keep = ~((lab[0] > 0.9) & (chroma < 0.035))
    if int(keep.sum()) < count:
        return torch.tensor([[0.5, 0.5, 0.5]])
    flat = joined.reshape(3, -1)[:, keep.flatten()]
    side = int(math.sqrt(flat.shape[1]))
    palette, _ = extract_palette(flat[:, : side * side].reshape(3, side, side), count, seed=seed)
    light = linear_to_oklab(srgb_to_linear(palette))[:, 0]
    return palette[light.argsort(descending=True)]


def absorb_slivers(labels: torch.Tensor, count: int, width: int = 5) -> torch.Tensor:
    """Labels with every part thinner than ``width`` pixels given to the shapes beside it. Where two colours
    meet, the pixels between them often look like a third, darker colour; coloured as shapes of their own they
    would draw a dark rim round everything."""
    onehot = F.one_hot(labels, count).permute(2, 0, 1).float()
    opened = F.max_pool2d(-F.max_pool2d(-onehot[None], width, 1, width // 2), width, 1, width // 2)[0] * onehot
    kept = opened.sum(0) > 0
    out = labels.clone()
    for size in (9, 17, 33):
        if bool(kept.all()):
            break
        votes = F.avg_pool2d(opened[None], size, 1, size // 2, count_include_pad=False)[0]
        near = votes.argmax(0)
        fill = ~kept & (votes.sum(0) > 0)
        out = torch.where(fill, near, out)
        opened = opened + F.one_hot(near, count).permute(2, 0, 1).float() * fill
        kept = kept | fill
    return out


def colour_labels(planes: torch.Tensor, palette: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Which marker colours each pixel (-1: left as white paper), smoothed so each shape takes one marker, and
    where a shape is clearly darker than its marker (to be gone over again): both ``(H, W)``."""
    lab = image_to_oklab(planes)
    pigments = linear_to_oklab(srgb_to_linear(palette))
    d = ((lab.reshape(3, -1).T[:, None] - pigments[None]) * torch.tensor([1.0, 1.4, 1.4])).pow(2).sum(-1)
    labels = d.argmin(1).reshape(lab.shape[1:])
    chroma = (lab[1] ** 2 + lab[2] ** 2).sqrt()
    paper = (lab[0] > 0.9) & (chroma < 0.04)
    labels = torch.where(paper, torch.full_like(labels, len(palette)), labels)
    for _ in range(2):                                     # each shape takes one marker
        votes = F.one_hot(labels, len(palette) + 1).permute(2, 0, 1).float()[None]
        labels = F.avg_pool2d(votes, 5, 1, 2, count_include_pad=False)[0].argmax(0)
    labels = absorb_slivers(labels, len(palette) + 1)
    labels = torch.where(labels == len(palette), torch.full_like(labels, -1), labels)
    marker_light = torch.cat([pigments[:, 0], torch.ones(1)])[labels.clamp(min=-1)]
    darker = (gaussian_blur(lab[0], 1.5) < marker_light - 0.09) & (labels >= 0)
    darker = F.avg_pool2d(darker[None, None].float(), 5, 1, 2)[0, 0] > 0.6
    return labels, darker


def hatch(mask: torch.Tensor, angle: float, spacing: float, nib: float, phase: float = 0.0, rows: int = 8) -> list:
    """Fill ``mask (H, W)`` the way a marker fills a shape: parallel strokes ``spacing`` pixels apart at
    ``angle`` (radians), gone back and forth without lifting where one row leads on to the next, ``rows`` rows
    at most in one go (a hand moves on along the shape). Returns paths in pixels ``(K, 2)``; rows shorter than
    the nib are left out (they need a finer marker)."""
    ys, xs = mask.nonzero(as_tuple=True)
    if len(ys) == 0:
        return []
    h, w = mask.shape
    along = torch.tensor([math.cos(angle), math.sin(angle)])
    across = torch.tensor([-math.sin(angle), math.cos(angle)])
    p = torch.stack([xs.float() + 0.5, ys.float() + 0.5], -1)
    centre = p.mean(0)
    u, v = (p - centre) @ along, (p - centre) @ across
    us = torch.arange(float(u.min()) - 1.0, float(u.max()) + 1.0, 1.0)
    vs = torch.arange(float(v.min()) + spacing * (0.5 + phase % 1.0) - spacing, float(v.max()) + 1.0, spacing)
    if len(vs) == 0 or len(us) < 2:
        return []
    pts = centre + us[None, :, None] * along + vs[:, None, None] * across            # (rows, steps, 2)
    xi, yi = pts[..., 0].floor().long(), pts[..., 1].floor().long()
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    inside &= mask[yi.clamp(0, h - 1), xi.clamp(0, w - 1)]
    edges = torch.diff(F.pad(inside.int(), (1, 1)), dim=1)
    runs_per_row = []
    # the nib's middle turns inside the shape, so its edge reaches just over the border: up to the line, a
    # touch over it, and no more (the turns run along the border, and a whole nib there would colour the
    # shape next door)
    shrink = 0.22 * nib
    for row in range(len(vs)):
        starts = (edges[row] == 1).nonzero().flatten()
        stops = (edges[row] == -1).nonzero().flatten() - 1
        runs = []
        for a, b in zip(starts.tolist(), stops.tolist(), strict=True):
            ua, ub = float(us[a]) + shrink, float(us[b]) - shrink
            if ub - ua >= 0.6 * nib:
                runs.append((centre + ua * along + float(vs[row]) * across, centre + ub * along
                             + float(vs[row]) * across))
        runs_per_row.append(runs)
    chains: list = []
    open_: list = []                                   # numbers of the chains the last row can still lead on
    for runs in runs_per_row:
        taken: list = []
        for a, b in runs:
            best, gap, near_a = -1, 2.4 * spacing, True
            for c in open_:
                if c in taken:
                    continue
                last = chains[c][-1]
                for end, is_a in ((a, True), (b, False)):
                    d = float((last - end).norm())
                    if d < gap:
                        best, gap, near_a = c, d, is_a
            if best >= 0 and len(chains[best]) >= 2 * rows:      # the hand lifts and starts the next stretch
                best = -1
            if best < 0:
                chains.append([a, b])
                taken.append(len(chains) - 1)
            else:
                chains[best].extend([a, b] if near_a else [b, a])
                taken.append(best)
        open_ = taken
    return [torch.stack(c) for c in chains]


def _marker_angle(marker: int) -> float:
    """The slant a hand colours with: about 40 degrees, a little different for each marker, the same in every
    frame."""
    return math.radians(40.0 + 9.0 * (((marker * 37) % 5) - 2))


def stamp(lines: list, height: int, width: int, scale: float, grow: float = 0.0) -> torch.Tensor:
    """Where ``lines`` lay ink, roughly and quickly: ``(H, W)`` bool (points along each segment every pixel,
    widened to each line's half width plus ``grow`` pixels)."""
    out = torch.zeros(height, width, dtype=torch.bool)
    by_width: dict = {}
    for line in lines:
        pts = line.points * scale
        if len(pts) > 1:
            seg = pts[1:] - pts[:-1]
            n = (seg.norm(dim=-1).ceil().clamp(min=1)).long()
            steps = torch.repeat_interleave(torch.arange(len(seg)), n)
            frac = torch.cat([torch.arange(int(k)) / int(k) for k in n.tolist()]) if len(n) else torch.zeros(0)
            pts = torch.cat([pts[:-1][steps] + seg[steps] * frac[:, None], pts[-1:]])
        r = int(round(0.5 * TOOLS[line.tool].width * scale + grow))
        by_width.setdefault(max(0, r), []).append(pts)
    for r, groups in by_width.items():
        pts = torch.cat(groups)
        xi, yi = pts[:, 0].long().clamp(0, width - 1), pts[:, 1].long().clamp(0, height - 1)
        dots = torch.zeros(height, width)
        dots[yi, xi] = 1.0
        if r > 0:
            dots = F.max_pool2d(dots[None, None], 2 * r + 1, 1, r)[0, 0]
        out |= dots > 0
    return out


def _dense(line: Line, scale: float) -> torch.Tensor:
    """The points of a line every pixel along it, in pixels."""
    pts = line.points * scale
    if len(pts) < 2:
        return pts
    seg = pts[1:] - pts[:-1]
    n = seg.norm(dim=-1).ceil().clamp(min=1).long()
    steps = torch.repeat_interleave(torch.arange(len(seg)), n)
    frac = torch.cat([torch.arange(int(k)) / int(k) for k in n.tolist()])
    return torch.cat([pts[:-1][steps] + seg[steps] * frac[:, None], pts[-1:]])


def dedupe(lines: list, height: int, width: int, scale: float, share: float = 0.85) -> list:
    """The lines without those that only go over what others already drew: older lines (traced from earlier
    frames) and longer ones are kept first, and a line of which ``share`` or more lies on the ink of the same
    pen, or of the same marker in the same layer, is left out. Without this, a drawing traced frame after frame
    would gather more and more strokes on top of each other."""
    keep = []
    rasters: dict = {}
    for i in sorted(range(len(lines)), key=lambda i: (-lines[i].age, -lines[i].length())):
        line = lines[i]
        group = ("ink", line.kind) if line.marker < 0 else (line.marker, line.kind)
        raster = rasters.setdefault(group, torch.zeros(height, width, dtype=torch.bool))
        pts = _dense(line, scale)
        xi, yi = pts[:, 0].long().clamp(0, width - 1), pts[:, 1].long().clamp(0, height - 1)
        if len(pts) and float(raster[yi, xi].float().mean()) >= share:
            continue
        keep.append(i)
        r = 1.5 if line.marker < 0 else 0.45 * TOOLS[line.tool].width * scale
        k = torch.arange(-int(r), int(r) + 1)
        oy, ox = torch.meshgrid(k, k, indexing="ij")
        disc = (oy ** 2 + ox ** 2) <= r * r + 0.5
        oy, ox = oy[disc], ox[disc]
        raster[(yi[:, None] + oy).clamp(0, height - 1), (xi[:, None] + ox).clamp(0, width - 1)] = True
    keep.sort()
    return [lines[i] for i in keep]


def colour_lines(labels: torch.Tensor, darker: torch.Tensor, palette: torch.Tensor, scale: float,
                 masks: dict | None = None, shade_masks: dict | None = None) -> list:
    """The marker strokes that colour a picture whose pixels are labelled by marker: for each marker its
    shapes back and forth with the broad nib, what is too narrow for that with the fine one, and the darker
    parts gone over again across the first layer. ``masks`` and ``shade_masks`` ({marker: (H, W) bool})
    limit the work to parts of the picture (when a frame is mostly traced from the one before)."""
    h, w = labels.shape
    broad, fine = TOOLS["marker"].width * scale, TOOLS["fine marker"].width * scale
    out = []
    for k in range(len(palette)):
        colour = tuple(palette[k].tolist())
        region = (labels == k) if masks is None else masks.get(k, torch.zeros(h, w, dtype=torch.bool))
        if int(region.sum()) < 0.5 * fine * fine:
            continue
        angle = _marker_angle(k)
        wide = [Line(p / scale, torch.full((len(p),), 0.75), "marker", colour, k, "fill")
                for p in hatch(region, angle, 0.62 * broad, broad)]
        out += wide
        left = region & ~stamp(wide, h, w, scale)
        if int(left.sum()) > fine * fine:
            out += [Line(p / scale, torch.full((len(p),), 0.75), "fine marker", colour, k, "fill")
                    for p in hatch(left, angle, 0.62 * fine, fine)]
        shade = (darker & (labels == k)) if shade_masks is None else shade_masks.get(k)
        if shade is not None and int(shade.sum()) > broad * broad:
            out += [Line(p / scale, torch.full((len(p),), 0.75), "marker", colour, k, "shade")
                    for p in hatch(shade, angle + math.radians(80.0), 0.75 * broad, broad)]
    return out


# ---------------------------------------------------------------------------------------------- order --
def order(lines: list, start=(0.0, 0.0)) -> list:
    """The order a person draws in: light pencil first, then the ink lines (the long ones near where the pen
    is first, then on to whichever line starts nearest, drawn from its nearer end), then colour one marker at a
    time from the lightest, then the darker layers."""
    out, here = [], torch.tensor(start, dtype=torch.float32)

    def nearest(group: list) -> None:
        nonlocal here
        rest = list(group)
        while rest:
            best, cost, flip = 0, float("inf"), False
            for i, line in enumerate(rest):
                for end, f in ((0, False), (-1, True)):
                    c = float((line.points[end] - here).norm()) - 0.25 * min(line.length(), 0.3)
                    if c < cost:
                        best, cost, flip = i, c, f
            line = rest.pop(best)
            line = line.reversed() if flip and line.kind in ("line", "sketch") else line
            out.append(line)
            here = line.points[-1]

    for kind in ("sketch", "line"):
        group = [line for line in lines if line.kind == kind]
        long = [line for line in group if line.length() > 0.08]
        nearest(long)
        nearest([line for line in group if line.length() <= 0.08])
    for kind in CELS:
        group = [line for line in lines if line.kind == kind]
        for colour in sorted({line.colour for line in group}, key=lambda c: -sum(c)):
            nearest([line for line in group if line.colour == colour])
    for kind in ("fill", "shade"):
        group = [line for line in lines if line.kind == kind]
        for k in sorted({line.marker for line in group}):
            for tool in ("marker", "fine marker"):
                nearest([line for line in group if line.marker == k and line.tool == tool])
    return out


def timing(lines: list, traced_speed: float = 2.5) -> list:
    """When each point of each line is drawn, in seconds from the start: ``[(start, times (K,)), ...]``. A
    hand slows on a curve and goes faster along a straight line, lifts the pen and moves to the next line,
    and stops for a moment to take another marker. A line traced from the frame before (age above 0) goes
    ``traced_speed`` times faster: the animator is following a line already there."""
    out, clock, here, tool = [], 0.0, None, None
    for line in lines:
        t = TOOLS[line.tool]
        key = (line.tool, line.marker)
        if tool is not None and key != tool:
            clock += 0.7                                     # cap on, the next marker taken, cap off
        tool = key
        if here is not None:
            clock += 0.05 + float((line.points[0] - here).norm()) / 1.8
        pts = line.points
        seg = (pts[1:] - pts[:-1]).norm(dim=-1)
        if len(pts) > 2:
            a, b = pts[1:-1] - pts[:-2], pts[2:] - pts[1:-1]
            turn = torch.atan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(-1)).abs()
            turn = torch.cat([turn[:1], turn])
        else:
            turn = torch.zeros(len(seg))
        speed = t.speed * (0.35 + 0.65 / (1.0 + 3.0 * turn)) * (traced_speed if line.age > 0 else 1.0)
        times = clock + torch.cat([torch.zeros(1), (seg / speed).cumsum(0)])
        out.append((clock, times))
        clock = float(times[-1])
        here = pts[-1]
    return out


# -------------------------------------------------------------------------------------------- drawing --
@dataclass
class Look:
    """What is seen in one picture before drawing it."""

    planes: torch.Tensor       # (3, H, W) sRGB: the picture seen in planes
    lines: torch.Tensor        # (H, W) bool: the lines to draw, one pixel wide
    strength: torch.Tensor     # (H, W): how strong the borders are
    labels: torch.Tensor       # (H, W) long: which marker colours each pixel (-1 paper)
    darker: torch.Tensor       # (H, W) bool: parts darker than their marker


def look(image: torch.Tensor, palette: torch.Tensor, detail: float = 0.5) -> Look:
    """See ``image (3, H, W)`` (sRGB) as something to draw: in planes, its lines, and its colours as the
    markers of ``palette``."""
    from .mind import simplify

    planes, _ = simplify(image.float().clamp(0.0, 1.0), torch.zeros(image.shape[-2:]), 0.45 + 0.3 * (1.0 - detail),
                         0.5)
    lines, strength = find_lines(planes, detail)
    labels, darker = colour_labels(planes, palette)
    # the pen draws the lines; the dark border the picture may have round its shapes is not coloured again
    band = F.max_pool2d(lines[None, None].float(), 9, 1, 4)[0, 0] > 0
    outside = F.one_hot(labels + 1, len(palette) + 1).permute(2, 0, 1).float() * (~band)
    votes = F.avg_pool2d(outside[None], 15, 1, 7, count_include_pad=False)[0]
    near = votes.argmax(0) - 1
    labels = torch.where(band & (votes.sum(0) > 0), near, labels)
    darker = darker & ~band
    return Look(planes, lines, strength, labels, darker)


def draw_picture(image: torch.Tensor, palette: torch.Tensor | None = None, *, lines: str = "pen",
                 colour: str | None = "marker", sketch: bool = False, detail: float = 0.5,
                 seen: Look | None = None) -> list:
    """Every line it takes to draw ``image (3, H, W)`` (sRGB), in the order a person draws them: (a light
    pencil sketch of the main lines when ``sketch``), the ink lines with the ``lines`` tool, then the colour
    with markers (``colour`` None draws lines only). Points are in board widths of the picture's width."""
    palette = markers([image]) if palette is None else palette
    seen = look(image, palette, detail) if seen is None else seen
    h, w = image.shape[-2:]
    scale = float(w)
    paths = trace(seen.lines, least=max(6, round(0.02 * scale)))
    out = ink_lines(paths, seen.strength, scale, lines)
    if sketch:
        main = [p for p in paths if len(p) > 0.06 * scale]
        out += [Line(line.points, line.pressure * 0.6, "pencil", TOOLS["pencil"].ink, -1, "sketch")
                for line in ink_lines(main, seen.strength, scale, "pencil", kind="sketch")]
    if colour:
        out += colour_lines(seen.labels, seen.darker, palette, scale)
    return order(out)


__all__ = ["CELS", "KINDS", "SURFACES", "TOOLS", "TOOL_NAMES", "Board", "Line", "Look", "Tool", "absorb_slivers",
           "fill_rings", "colour_labels", "colour_lines", "dedupe", "draw_picture", "find_lines", "hatch", "ink_lines",
           "line_strength", "look", "markers", "order", "radii", "render", "smooth_path", "stamp", "surface", "thin",
           "timing", "trace"]
