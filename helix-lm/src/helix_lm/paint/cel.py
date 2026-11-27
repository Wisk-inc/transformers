# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Ink and paint for one frame, the way an anime studio's finishing department (shiage) does it: the frame becomes
clean line art and flat, opaque cel paint filled exactly up to the lines, with a hard-edged shadow (kage) and a
highlight (hi) on each part.

The rules come from how people do it by hand:

- One colour design for the whole clip. A colour designer (shikisai sekkei) picks the colours once and every
  frame is painted with them, so nothing flickers (Sakugabooru glossary; Anime News Network's interview with the
  colour designer Arisa Komatsu, 2024). Here :func:`palette_for` picks them from all the frames together, and
  each colour can have its own shadow and highlight tones.
- Paint is flat and opaque, on the back of the cel, filled up to the line with nothing left white. The line art
  has to be closed for that (Clip Studio, "Simple anime-style coloring techniques"; "Traditional animation" on
  Wikipedia). Here every border is found once and shared by the shapes on both sides, so the paint meets with no
  gap, and the board puts the paint under the ink.
- Shading is a hard-edged shape, not a blend: one shadow a little darker than the base and one highlight a little
  lighter (Clip Studio tips "Shading your drawings like an anime movie"; MediBang "How to color like anime
  style"). A shadow is not only darker: its hue moves toward blue or purple and it is a little more saturated; a
  highlight is warmer (Clip Studio; Graphixly "Tips for cel shading").
- The border of a shadow is not drawn in black. On the in-betweens it is a coloured pencil line, traced on the
  cel in a darker tone of the colour (a colour trace line, iro-tore) or left out (Clip Studio's colour trace
  pens and actions).
- Line weight: the silhouette, where the character meets the background, is drawn heavier, the inner lines
  thinner, and a line tapers where the pen lands and lifts (Proko, "How to draw with line weight"; Clip Studio,
  "Hints on how to improve your line art"; Hertzmann, "How to draw pictures, part 4: line thickness").
  Backgrounds in anime are painted, not inked: borders inside the background get no black line.

How one frame is done (:func:`paint_frame`):

1. :func:`regions`: the frame smoothed without blurring its edges, each pixel given the nearest colour of the
   palette in OKLab, neighbours voting so the borders do not fray, slivers and specks given to the part beside
   them;
2. :func:`shade`: what is background (flooded in from the edges of the picture across soft, small changes of
   colour; it stops at the character's silhouette) and, for every part of the character, one shadow and one
   highlight shape: from the darker and lighter tones of its colour the frame shows, from the frame's own light
   and dark inside it, from normals and a light, or, where the frame shows it flat, from its silhouette blown
   up into a rounded relief;
3. the borders between parts are followed from junction to junction and smoothed once, keeping real corners
   (:func:`outlines` does the same for a single mask); each cel shape is put together from them;
4. :func:`line_art`: ink along the borders, joined through junctions into long strokes, heavy on the silhouette,
   lighter inside, colour trace lines on the shadows, and the small dark marks of the frame the paint lost (an
   eye, a mouth).

All of it is rules, nothing is learned, and the frames are judged one at a time: the same palette keeps the
colours of an animation steady, but the shapes of one frame are found again in the next, not carried over.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .color import image_to_oklab, oklab_to_linear, srgb_to_oklab
from .draw import TOOLS, Line, order, smooth_path, stamp, thin, trace
from .see import gaussian_blur

WEIGHT = torch.tensor([1.0, 1.5, 1.5])      # in OKLab, hue and saturation count a little more than light
SHADOW_HUE = math.radians(-75.0)            # OKLab hue a shadow moves toward: between blue and purple
LIGHT_HUE = math.radians(75.0)              # and a highlight: a warm yellow


def _odd(x: float) -> int:
    k = max(1, int(round(x)))
    return k if k % 2 else k + 1


def _distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """How different OKLab colours ``(..., 3)`` look."""
    return ((x - y) * WEIGHT).norm(dim=-1)


# ------------------------------------------------------------------------------------------- colour --
def _bilateral(lab: torch.Tensor, passes: int = 3, sigma_colour: float = 0.045) -> torch.Tensor:
    """``lab (3, H, W)`` smoothed among neighbours of nearly the same colour only, so noise and soft steps go
    and edges stay where they are (a bilateral filter, 3 by 3, run ``passes`` times)."""
    h, w = lab.shape[-2:]
    wt = (WEIGHT ** 2)[:, None, None]
    for _ in range(passes):
        pad = F.pad(lab[None], (1, 1, 1, 1), mode="replicate")[0]
        num, den = torch.zeros_like(lab), torch.zeros(h, w)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                near = pad[:, 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
                g = torch.exp(((near - lab) ** 2 * wt).sum(0) * (-0.5 / sigma_colour ** 2) - 0.3 * (dy * dy + dx * dx))
                num += g * near
                den += g
        lab = num / den
    return lab


def _edges(lab: torch.Tensor) -> torch.Tensor:
    """How much each pixel differs from its most different 4-neighbour, ``(H, W)``."""
    pad = F.pad(lab[None], (1, 1, 1, 1), mode="replicate")[0]
    h, w = lab.shape[-2:]
    out = torch.zeros(h, w)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        near = pad[:, 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        out = torch.maximum(out, ((near - lab) * WEIGHT[:, None, None]).norm(dim=0))
    return out


def _gamut(lab: torch.Tensor) -> torch.Tensor:
    """OKLab colours ``(..., 3)`` brought inside sRGB by lowering their chroma (light and hue kept)."""
    lab = lab.clone()
    lab[..., 0] = lab[..., 0].clamp(0.0, 1.0)
    lo, hi = torch.zeros(lab.shape[:-1]), torch.ones(lab.shape[:-1])
    rgb = oklab_to_linear(lab)
    fine = ((rgb >= -1e-4) & (rgb <= 1 + 1e-4)).all(-1)
    if bool(fine.all()):
        return lab
    for _ in range(12):
        mid = (lo + hi) / 2
        trial = torch.cat([lab[..., :1], lab[..., 1:] * mid[..., None]], -1)
        rgb = oklab_to_linear(trial)
        ok = ((rgb >= -1e-4) & (rgb <= 1 + 1e-4)).all(-1)
        lo, hi = torch.where(ok, mid, lo), torch.where(ok, hi, mid)
    scale = torch.where(fine, torch.ones_like(lo), lo)
    return torch.cat([lab[..., :1], lab[..., 1:] * scale[..., None]], -1)


def _to_srgb(lab: torch.Tensor) -> torch.Tensor:
    from .color import linear_to_srgb

    return linear_to_srgb(oklab_to_linear(_gamut(lab)).clamp(0.0, 1.0))


def _turn_hue(lab: torch.Tensor, toward: float, share: float, most: float) -> torch.Tensor:
    """OKLab colours with their hue turned ``share`` of the way toward the hue ``toward`` (radians), by at most
    ``most`` radians."""
    a, b = lab[..., 1], lab[..., 2]
    hue = torch.atan2(b, a)
    gap = torch.remainder(toward - hue + math.pi, 2 * math.pi) - math.pi
    hue = hue + (share * gap).clamp(-most, most)
    chroma = torch.hypot(a, b)
    return torch.stack([lab[..., 0], chroma * torch.cos(hue), chroma * torch.sin(hue)], -1)


def shadow_colour(colour: torch.Tensor) -> torch.Tensor:
    """The anime shadow (kage) of sRGB colours ``(..., 3)``: about a tenth darker in OKLab lightness, the hue
    turned a little toward blue and purple, a dull colour made more saturated (a strong one is saturated enough
    once it is darker). A white or a grey gets a cool lavender shadow."""
    lab = srgb_to_oklab(colour.float())
    chroma = torch.hypot(lab[..., 1], lab[..., 2])
    turned = _turn_hue(lab, SHADOW_HUE, 0.1, math.radians(8))
    grow = 1.0 + 0.3 * (1.0 - chroma / 0.12).clamp(min=0.0)
    most = torch.maximum(torch.full_like(chroma, 0.15), 0.85 * chroma)
    new = torch.minimum(chroma * grow, most)
    turned[..., 1:] = turned[..., 1:] * (new / chroma.clamp(min=1e-6))[..., None]
    grey = ((0.05 - chroma) / 0.05).clamp(0.0, 1.0)            # how neutral it is: a tint of its own
    tint = 0.03 * torch.stack([torch.full_like(chroma, math.cos(SHADOW_HUE)),
                               torch.full_like(chroma, math.sin(SHADOW_HUE))], -1)
    turned[..., 1:] = turned[..., 1:] * (1 - grey[..., None]) + tint * grey[..., None]
    turned[..., 0] = lab[..., 0] - torch.minimum(torch.full_like(chroma, 0.11), 0.3 * lab[..., 0])
    return _to_srgb(turned)


def light_colour(colour: torch.Tensor) -> torch.Tensor:
    """The anime highlight (hi) of sRGB colours ``(..., 3)``: lighter, a little less saturated, warmer."""
    lab = srgb_to_oklab(colour.float())
    turned = _turn_hue(lab, LIGHT_HUE, 0.12, math.radians(14))
    turned[..., 1:] = turned[..., 1:] * 0.9
    turned[..., 0] = lab[..., 0] + (1.0 - lab[..., 0]) * 0.45
    return _to_srgb(turned)


def _hue_gap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """How far apart the hues of OKLab colours ``(..., 3)`` are, radians 0..pi."""
    return (torch.remainder(torch.atan2(x[..., 2], x[..., 1]) - torch.atan2(y[..., 2], y[..., 1]) + math.pi,
                            2 * math.pi) - math.pi).abs()


def _family(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Where OKLab colour ``y`` is a darker or lighter tone of colour ``x`` (both ``(..., 3)``), as a shadow or
    a highlight of one material is: the same hue and about the same saturation (chroma for its lightness), or
    both near grey; and clearly but not wholly lighter or darker."""
    cx, cy = torch.hypot(x[..., 1], x[..., 2]), torch.hypot(y[..., 1], y[..., 2])
    sx, sy = cx / x[..., 0].clamp(min=0.05), cy / y[..., 0].clamp(min=0.05)
    ratio = torch.maximum(sx, sy) / torch.minimum(sx, sy).clamp(min=1e-3)
    hue = (cx > 0.04) & (cy > 0.04) & (_hue_gap(x, y) < math.radians(25)) & (ratio < 1.7)
    grey = (cx < 0.045) & (cy < 0.045)
    step = (x[..., 0] - y[..., 0]).abs()
    return (hue | grey) & (step > 0.025) & (step < 0.32)


def palette_for(frames, count: int = 16, *, side: int = 256, least: float = 0.0003) -> torch.Tensor:
    """The colours to paint a set of frames (sRGB ``(3, H, W)`` each) with: at most ``count`` distinct colours
    (sRGB ``(K, 3)``, light to dark), each covering at least ``least`` of the frames' area. More colours than
    that are found first (k-means in OKLab), then the two most alike are made one until ``count`` are left.
    Colours are picked from the flat parts of the frames (the pixels on an edge are a mix of the colours either
    side and pick none, unless one is the lightest or darkest of its neighbours: the middle of a small dot), with
    small but clearly different colours counted more than their area alone (an eye, a nose), and whites kept as
    colours of their own. The same palette paints every frame of an animation."""
    pixels = []
    for frame in frames:
        f = frame.float().clamp(0.0, 1.0)
        h, w = f.shape[-2:]
        k = min(1.0, side / max(h, w))
        if k < 1.0:
            f = F.interpolate(f[None], size=(max(8, round(h * k)), max(8, round(w * k))), mode="area")[0]
        lab = _bilateral(image_to_oklab(f), passes=2)
        light = lab[None, :1]
        peak = (F.max_pool2d(light, 3, 1, 1) <= light) | (-F.max_pool2d(-light, 3, 1, 1) >= light)
        flat = (_edges(lab) < 0.035) | peak[0, 0]
        pixels.append(lab.reshape(3, -1)[:, flat.flatten()].T)
    x = torch.cat(pixels)
    if len(x) == 0:
        return torch.tensor([[0.5, 0.5, 0.5]])
    step = torch.tensor([0.02, 0.014, 0.014])
    bins, owner, counts = torch.unique(torch.round(x / step).long(), dim=0, return_inverse=True, return_counts=True)
    means = torch.zeros(len(bins), 3).index_add_(0, owner, x) / counts[:, None]
    weight = counts.float() ** 0.6                # small colours count for more than their area alone
    centres = means[counts.argmax()][None]
    for _ in range(1, min(count + 8, len(means))):
        d = _distance(means[:, None], centres[None]).min(1).values
        score = weight * d ** 2
        if float(score.max()) <= 0:
            break
        centres = torch.cat([centres, means[score.argmax()][None]])
    for _ in range(25):
        assign = _distance(means[:, None], centres[None]).argmin(1)
        total = torch.zeros(len(centres)).index_add_(0, assign, weight)
        moved = torch.zeros_like(centres).index_add_(0, assign, means * weight[:, None])
        moved = moved / total[:, None].clamp(min=1e-9)
        centres = torch.where(total[:, None] > 0, moved, centres)
    assign = _distance(means[:, None], centres[None]).argmin(1)
    area = torch.zeros(len(centres)).index_add_(0, assign, counts.float())
    colour = torch.zeros_like(centres).index_add_(0, assign, means * counts[:, None].float()) / area[:, None].clamp(1)
    share = area / float(counts.sum())
    keep = share >= least
    if bool(keep.any()):                         # a small colour clearly unlike the others stays (a highlight)
        apart = _distance(colour[:, None], colour[keep][None]).min(1).values
        keep |= (share >= least / 8) & (apart > 0.09)
    colour, share = colour[keep], share[keep]
    # the two most alike colours become one, until there are ``count`` left and no two nearly the same: a
    # gradient loses a band before a small colour unlike the rest is lost
    while len(colour) > 1:
        d = _distance(colour[:, None], colour[None]) + torch.eye(len(colour)) * 9.0
        i, j = divmod(int(d.argmin()), len(colour))
        if float(d[i, j]) >= 0.035 and len(colour) <= count:
            break
        a, b = (i, j) if share[i] >= share[j] else (j, i)
        colour[a] = (colour[a] * share[a] + colour[b] * share[b]) / (share[a] + share[b])
        share[a] = share[a] + share[b]
        rest = torch.arange(len(colour)) != b
        colour, share = colour[rest], share[rest]
    colour = colour[colour[:, 0].argsort(descending=True)]
    return _to_srgb(colour).clamp(0.0, 1.0)


# -------------------------------------------------------------------------------------------- regions --
def parts(labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """The connected parts of a label map ``(H, W)`` (pixels side by side with the same label, 4-connected):
    ``(ids (H, W) long, count)``, numbered in the order their first pixel comes, row by row."""
    labels = torch.as_tensor(labels)
    h, w = labels.shape
    big = h * w
    if big == 0:
        return torch.zeros(h, w, dtype=torch.long), 0
    idx = torch.arange(big).view(h, w)
    # each run of one label along a row starts as one tree, its root the run's first pixel
    start = torch.ones(h, w, dtype=torch.bool)
    start[:, 1:] = labels[:, 1:] != labels[:, :-1]
    parent = torch.cummax(torch.where(start, idx, 0), dim=1).values.flatten()
    # runs that touch the row below with the same label are joined: each round every root is hung under the
    # smallest root it touches, then every pixel is pointed straight at its root
    down = (labels[1:] == labels[:-1]).flatten()
    u, v = idx[:-1].flatten()[down], idx[1:].flatten()[down]
    while len(u):
        ru, rv = parent[u], parent[v]
        differ = ru != rv
        if not bool(differ.any()):
            break
        ru, rv = ru[differ], rv[differ]
        u, v = u[differ], v[differ]
        parent.scatter_reduce_(0, torch.maximum(ru, rv), torch.minimum(ru, rv), "amin")
        while True:
            jumped = parent[parent]
            if torch.equal(jumped, parent):
                break
            parent = jumped
    _, inverse = torch.unique(parent, return_inverse=True)
    return inverse.view(h, w), int(inverse.max()) + 1


def _adjacency(ids: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """How many pixel sides each pair of touching parts shares: ``(first, second, count)``, both ways round."""
    a = torch.cat([ids[:, :-1].flatten(), ids[:, 1:].flatten(), ids[:-1].flatten(), ids[1:].flatten()])
    b = torch.cat([ids[:, 1:].flatten(), ids[:, :-1].flatten(), ids[1:].flatten(), ids[:-1].flatten()])
    keep = a != b
    key, count = torch.unique(a[keep] * n + b[keep], return_counts=True)
    return key // n, key % n, count


def _steps(ids: torch.Tensor, n: int, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                                   torch.Tensor]:
    """Like :func:`_adjacency`, with how sharply the frame ``lab (3, H, W)`` (OKLab, smoothed) changes across
    each border: the mean OKLab step from one pixel to the next over it. A painted gradient changes softly
    across the border between two of its bands; the silhouette of a thing is a crisp step."""
    wt = WEIGHT[:, None, None]
    sx = ((lab[:, :, 1:] - lab[:, :, :-1]) * wt).norm(dim=0)
    sy = ((lab[:, 1:] - lab[:, :-1]) * wt).norm(dim=0)
    a = torch.cat([ids[:, :-1].flatten(), ids[:, 1:].flatten(), ids[:-1].flatten(), ids[1:].flatten()])
    b = torch.cat([ids[:, 1:].flatten(), ids[:, :-1].flatten(), ids[1:].flatten(), ids[:-1].flatten()])
    step = torch.cat([sx.flatten(), sx.flatten(), sy.flatten(), sy.flatten()])
    keep = a != b
    key, inverse, count = torch.unique(a[keep] * n + b[keep], return_inverse=True, return_counts=True)
    total = torch.zeros(len(key)).index_add_(0, inverse, step[keep])
    return key // n, key % n, count, total / count.float()


def _merge_small(labels: torch.Tensor, least: float, pigments: torch.Tensor | None = None, rounds: int = 8) -> \
        torch.Tensor:
    """Labels with every part smaller than ``least`` pixels given to the neighbour it shares the most border
    with (the larger one when two share as much). With ``pigments (K, 3)`` (the OKLab colour of each label), a
    part of at least a quarter of that size that stands out strongly from every neighbour stays (a nose)."""
    for _ in range(rounds):
        ids, n = parts(labels)
        area = torch.bincount(ids.flatten(), minlength=n)
        small = area < least
        if not bool(small.any()) or n < 2:
            break
        a, b, count = _adjacency(ids, n)
        colour = torch.zeros(n, dtype=labels.dtype).scatter_(0, ids.flatten(), labels.flatten())
        if pigments is not None:
            d = _distance(pigments[colour[a]], pigments[colour[b]])
            apart = torch.full((n,), 9.0).scatter_reduce(0, a, d, "amin")
            small &= ~((apart > 0.3) & (area >= least / 4))
        mine = small[a]
        a, b, count = a[mine], b[mine], count[mine]
        if len(a) == 0:
            break
        most = torch.zeros(n, dtype=count.dtype).scatter_reduce(0, a, count, "amax", include_self=False)
        best = count == most[a]
        size = torch.zeros(n, dtype=area.dtype).scatter_reduce(0, a[best], area[b[best]], "amax", include_self=False)
        best = best & (area[b] == size[a])
        target = torch.full((n,), n, dtype=torch.long).scatter_reduce(0, a[best], b[best], "amin")
        own = torch.arange(n)
        target = torch.where(target < n, target, own)
        # two small parts that would go into each other: the larger stays
        back = (target[target] == own) & (target != own)
        stays = back & ((area > area[target]) | ((area == area[target]) & (own < target)))
        target = torch.where(stays, own, target)
        for _ in range(8):
            target = target[target]
        new = colour[target][ids]
        if torch.equal(new, labels):
            break
        labels = new
    return labels


def regions(image: torch.Tensor, palette: torch.Tensor, *, detail: float = 0.5, smooth: float = 0.03) -> torch.Tensor:
    """The clean flat-colour regions of a frame ``(3, H, W)`` (sRGB): which colour of ``palette (K, 3)`` paints
    each pixel, ``(H, W)`` long. The frame is smoothed among pixels of nearly the same colour (edges stay sharp),
    each pixel takes the nearest palette colour in OKLab, its neighbours vote (``smooth``, in OKLab distance) so
    that borders run clean instead of fraying where the frame lies between two colours, parts thinner than a few
    pixels (the mixed colour along an edge, a thin outline of the frame) go to the parts beside them, and parts
    smaller than a speck go to the neighbour they share the most border with (unless they stand out strongly:
    a nose, a pupil). Away from the frame's edges the colour is judged over a few pixels around, so that where a
    soft gradient is painted in bands their borders run smooth, not along the noise. ``detail`` 0..1 keeps
    smaller parts. With one palette for all frames the colours do not flicker."""
    image = image.float().clamp(0.0, 1.0)
    w = image.shape[-1]
    lab = _bilateral(image_to_oklab(image))
    edge = F.max_pool2d((_edges(lab) > 0.03).float()[None, None], 5, 1, 2)[0, 0] > 0
    lab = torch.where(edge[None], lab, gaussian_blur(lab, 2.0))
    pigments = srgb_to_oklab(palette.float())
    count = len(pigments)
    cost = torch.stack([((lab - p[:, None, None]) * WEIGHT[:, None, None]).norm(dim=0) for p in pigments])
    labels = cost.argmin(0)
    for _ in range(2):
        votes = F.one_hot(labels, count).permute(2, 0, 1).float()[None]
        agree = F.avg_pool2d(votes, 3, 1, 1, count_include_pad=False)[0]
        labels = (cost - smooth * agree).argmin(0)
    labels = _absorb(labels, cost, _odd(max(3.0, 0.006 * w * (1.5 - detail))))
    return _merge_small(labels, max(6.0, (0.0095 * w * (1.5 - detail)) ** 2), pigments)


def _pad(x: torch.Tensor, r: int, fill) -> torch.Tensor:
    out = torch.full((x.shape[0] + 2 * r, x.shape[1] + 2 * r), fill, dtype=x.dtype)
    out[r : r + x.shape[0], r : r + x.shape[1]] = x
    return out


def _absorb(labels: torch.Tensor, cost: torch.Tensor, width: int) -> torch.Tensor:
    """Labels with every part thinner than ``width`` pixels given to the parts beside it: each such pixel takes,
    of the labels kept next to it, the one whose colour is nearest its own (``cost (K, H, W)``), growing in from
    the kept pixels one ring at a time. Where two colours meet, the pixels between them often look like a third
    colour; painted as parts of their own they would draw a rim round everything."""
    h, w = labels.shape
    r = width // 2
    window = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    pad = _pad(labels, r, -1)
    inner = torch.ones(h, w, dtype=torch.bool)               # the whole square round it has its label
    for dy, dx in window:
        inner &= pad[r + dy : r + dy + h, r + dx : r + dx + w] == labels
    pad_inner = _pad(inner, r, False)
    kept = torch.zeros(h, w, dtype=torch.bool)               # opened: within such a square of its own label
    for dy, dx in window:
        kept |= pad_inner[r + dy : r + dy + h, r + dx : r + dx + w] & (pad[r + dy : r + dy + h, r + dx : r + dx + w]
                                                                       == labels)
    out = labels.clone()
    ring = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    for _ in range(4 * width):
        if bool(kept.all()):
            break
        pad_out, pad_kept = _pad(out, 1, 0), _pad(kept, 1, False)
        best, pick = torch.full((h, w), 1e9), out
        for dy, dx in ring:
            near = pad_out[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            c = torch.where(pad_kept[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w], cost.gather(0, near[None])[0], 1e9)
            better = c < best
            best, pick = torch.where(better, c, best), torch.where(better, near, pick)
        fill = ~kept & (best < 1e9)
        out = torch.where(fill, pick, out)
        kept = kept | fill
    return out


# ------------------------------------------------------------------------------------------- borders --
@dataclass
class Borders:
    """The borders between the parts of a label map, followed from junction to junction, each found once and
    shared by the parts on both sides. ``points[i]`` is a border ``(K, 2)`` in pixels (x, y); ``a[i]`` is the
    part on its left going along it (in the picture, with y down: the side where ``cross(step, toward) > 0``),
    ``b[i]`` the part on its right; ``start[i]`` and ``end[i]`` number the corners it runs between (-1 for a
    closed loop). Part ``outside`` stands for off the picture."""

    points: list
    a: np.ndarray
    b: np.ndarray
    start: np.ndarray
    end: np.ndarray
    outside: int

    def rings(self, inside: np.ndarray) -> list:
        """The closed outlines ``(K, 2)`` (pixels) of the set of parts where ``inside (outside + 1,)`` is true,
        each going round with the set on its left, holes included."""
        sel = np.flatnonzero(inside[self.a] != inside[self.b])
        out, leaving = [], {}
        for i in sel.tolist():
            pts, s, e = self.points[i], int(self.start[i]), int(self.end[i])
            if not inside[self.a[i]]:
                pts, s, e = pts[::-1], e, s
            if s < 0:
                out.append(pts)
                continue
            leaving.setdefault(s, []).append((e, pts))
        while leaving:
            first = next(iter(leaving))
            ring, here = [], first
            while here in leaving:
                e, pts = leaving[here].pop()
                if not leaving[here]:
                    del leaving[here]
                ring.append(pts[:-1])
                here = e
                if here == first:
                    break
            if ring:
                out.append(np.concatenate(ring))
        return [np.ascontiguousarray(r) for r in out if len(r) >= 3]


def _turn(p: np.ndarray, k: int, closed: bool) -> np.ndarray:
    """How sharply a path ``(K, 2)`` turns at each point, over ``k`` points either side (radians)."""
    n = len(p)
    i = np.arange(n)
    before = p[(i - k) % n] if closed else p[np.clip(i - k, 0, n - 1)]
    after = p[(i + k) % n] if closed else p[np.clip(i + k, 0, n - 1)]
    u, v = p - before, after - p
    nu, nv = np.hypot(u[:, 0], u[:, 1]), np.hypot(v[:, 0], v[:, 1])
    cos = (u * v).sum(1) / np.maximum(nu * nv, 1e-9)
    return np.where((nu > 1e-6) & (nv > 1e-6), np.arccos(np.clip(cos, -1.0, 1.0)), 0.0)


def _corners(p: np.ndarray, closed: bool, sharp: float = math.radians(50)) -> np.ndarray:
    """The points of a pixel path where it has a real corner: it turns sharply over four points and nearly as
    sharply over two (a curve turns half as much over half the length; the stair steps of the pixels turn
    over one)."""
    n = len(p)
    if n < 9:
        return np.zeros(0, dtype=np.int64)
    wide, near = _turn(p, 4, closed), _turn(p, 2, closed)
    cand = (wide > sharp) & (near > 0.75 * wide)
    if not closed:
        cand[:3] = cand[-3:] = False
    pad = np.concatenate([wide[-3:], wide, wide[:3]]) if closed else np.pad(wide, 3)
    top = np.max(np.stack([pad[j : j + n] for j in range(7)]), 0)
    cand &= wide >= top
    idx = np.flatnonzero(cand)
    if len(idx) > 1:                                    # a flat top: one corner, not two
        idx = idx[np.r_[True, np.diff(idx) > 3]]
    return idx


def _kernel(sigma: float, most: int) -> np.ndarray:
    r = max(0, min(int(3 * sigma + 0.5), most))
    k = np.exp(-(np.arange(-r, r + 1) ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _blur(p: np.ndarray, k: np.ndarray, closed: bool) -> np.ndarray:
    r = len(k) // 2
    if r < 1:
        return p
    if closed:
        ext = np.concatenate([p[-r:], p, p[:r]])
    else:                                              # mirrored through the ends, so they stay put
        ext = np.concatenate([2 * p[0] - p[1 : r + 1][::-1], p, 2 * p[-1] - p[-r - 1 : -1][::-1]])
    out = np.stack([np.convolve(ext[:, 0], k, "valid"), np.convolve(ext[:, 1], k, "valid")], 1)
    if not closed:
        out[0], out[-1] = p[0], p[-1]
    return out


def _gauss(p: np.ndarray, sigma: float, closed: bool = False) -> np.ndarray:
    """A path ``(K, 2)`` smoothed along its length (the ends of an open one kept where they are). Smoothed twice
    and the second smoothing taken back from the first (``2 G(p) - G(G(p))``): the stair steps of the pixels go
    as with one smoothing, but a curve keeps its size instead of shrinking."""
    n = len(p)
    if n < 3 or sigma <= 0:
        return p
    k = _kernel(sigma, (n - 1) // 2 if closed else n - 1)
    once = _blur(p, k, closed)
    return 2 * once - _blur(once, k, closed)


def _simplify(p: np.ndarray, tolerance: float) -> np.ndarray:
    """A path with the points left out that lie within ``tolerance`` of the line through the points kept
    (Douglas and Peucker, 1973). The ends are kept."""
    n = len(p)
    if n <= 2 or tolerance <= 0:
        return p
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        seg = p[j] - p[i]
        length = math.hypot(seg[0], seg[1])
        rel = p[i + 1 : j] - p[i]
        if length < 1e-9:
            d = np.hypot(rel[:, 0], rel[:, 1])
        else:
            d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / length
        m = int(d.argmax())
        if d[m] > tolerance:
            k = i + 1 + m
            keep[k] = True
            stack += [(i, k), (k, j)]
    return p[keep]


def _clean_path(p: np.ndarray, closed: bool, sigma: float, tolerance: float) -> np.ndarray:
    """A border made smooth (the stair steps of the pixels taken out) while its real corners stay sharp, then
    simplified. A closed border comes back without its first point repeated at the end."""
    if len(p) < 3:
        return p
    corners = _corners(p, closed)
    if closed and len(corners) == 0:
        smooth = _gauss(p, sigma, closed=True)
        far = int(np.hypot(*(smooth - smooth[0]).T).argmax())
        half = [_simplify(smooth[: far + 1], tolerance), _simplify(np.concatenate([smooth[far:], smooth[:1]]),
                                                                    tolerance)]
        return np.concatenate([half[0][:-1], half[1][:-1]])
    if closed:
        p = np.concatenate([np.roll(p, -int(corners[0]), 0), p[corners[0] : corners[0] + 1]])
        corners = np.concatenate([[0], (corners[1:] - corners[0]), [len(p) - 1]])
    else:
        corners = np.concatenate([[0], corners, [len(p) - 1]])
    pieces = []
    for i, j in zip(corners[:-1], corners[1:], strict=True):
        piece = _simplify(_gauss(p[i : j + 1], sigma), tolerance)
        pieces.append(piece if not pieces else piece[1:])
    out = np.concatenate(pieces)
    return out[:-1] if closed else out


def borders(ids, count: int | None = None, *, sigma=1.6, tolerance: float = 0.15) -> Borders:
    """The borders of a label map ``ids (H, W)`` (any integers ``0 .. count - 1``): every run of pixel sides
    between the same two parts, from one junction (where three parts meet) to the next, with points on the
    middle of each pixel side (what marching squares gives for a mask), smoothed keeping real corners
    (``sigma`` points along the border; a function of the two parts' arrays can give each border its own),
    then simplified to ``tolerance`` pixels. Borders with the outside are kept straight and pushed two pixels
    past the edge of the picture, so the paint covers it to the edge."""
    ids = np.asarray(torch.as_tensor(ids).cpu(), dtype=np.int64)
    h, w = ids.shape
    n = int(ids.max()) + 1 if count is None else int(count)
    pad = np.full((h + 2, w + 2), n, dtype=np.int64)
    pad[1:-1, 1:-1] = ids
    cols = w + 1
    left, right = pad[1:-1, :-1], pad[1:-1, 1:]           # either side of each upright pixel side (h, w + 1)
    above, below = pad[:-1, 1:-1], pad[1:, 1:-1]           # either side of each level pixel side (h + 1, w)
    vi, vj = np.nonzero(left != right)
    hi, hj = np.nonzero(above != below)
    # an upright side goes down from corner (i, j) to (i + 1, j) with the left pixel on its left; a level side
    # goes right from (i, j) to (i, j + 1) with the pixel below on its left (y is down)
    c_from = np.concatenate([vi * cols + vj, hi * cols + hj])
    c_to = np.concatenate([(vi + 1) * cols + vj, hi * cols + hj + 1])
    side_a = np.concatenate([left[vi, vj], below[hi, hj]])
    side_b = np.concatenate([right[vi, vj], above[hi, hj]])
    mid = np.concatenate([np.stack([vj, vi + 0.5], 1), np.stack([hj + 0.5, hi], 1)]).astype(np.float64)
    m = len(c_from)
    corner = np.concatenate([c_from, c_to])
    side = np.concatenate([np.arange(m), np.arange(m)])
    which = np.repeat(np.array([0, 1]), m)
    sort = np.argsort(corner, kind="stable")
    corner, side, which = corner[sort], side[sort], which[sort]
    first = np.flatnonzero(np.r_[True, corner[1:] != corner[:-1]]) if m else np.zeros(0, dtype=np.int64)
    degree = np.diff(np.r_[first, len(corner)])
    junction = np.zeros((m, 2), dtype=bool)
    junction[side, which] = np.repeat(degree, degree) >= 3
    nxt = np.full((m, 2), -1, dtype=np.int64)
    two = first[degree == 2]
    nxt[side[two], which[two]] = side[two + 1]
    nxt[side[two + 1], which[two + 1]] = side[two]
    cf, ct, nx, jn = c_from.tolist(), c_to.tolist(), nxt.tolist(), junction.tolist()
    seen = bytearray(m)
    runs = []

    def walk(k: int, forward: bool) -> None:
        start, head, head_forward, run = (cf[k] if forward else ct[k]), k, forward, []
        while True:
            seen[k] = 1
            run.append(k)
            e = 1 if forward else 0
            here = ct[k] if forward else cf[k]
            if jn[k][e]:
                runs.append((run, head, head_forward, start, here))
                return
            k2 = nx[k][e]
            if k2 == head or k2 < 0:
                runs.append((run, head, head_forward, -1, -1))
                return
            forward = cf[k2] == here
            k = k2

    for k, e in np.argwhere(junction).tolist():
        if not seen[k]:
            walk(k, e == 0)
    for k in range(m):
        if not seen[k]:
            walk(k, True)
    count_runs = len(runs)
    a = np.empty(count_runs, dtype=np.int64)
    b = np.empty(count_runs, dtype=np.int64)
    start = np.empty(count_runs, dtype=np.int64)
    end = np.empty(count_runs, dtype=np.int64)
    raw = []
    for i, (run, head, head_forward, s, e) in enumerate(runs):
        a[i], b[i] = (side_a[head], side_b[head]) if head_forward else (side_b[head], side_a[head])
        start[i], end[i] = s, e
        pts = mid[run]
        if s >= 0:
            pts = np.concatenate([[[s % cols, s // cols]], pts, [[e % cols, e // cols]]]).astype(np.float64)
        raw.append(pts)
    sig = sigma(a, b) if callable(sigma) else np.full(count_runs, float(sigma))
    points = []
    for i, pts in enumerate(raw):
        closed = start[i] < 0
        if a[i] != n and b[i] != n:
            pts = _clean_path(pts, closed, float(sig[i]), tolerance)
        else:
            pts = _simplify(pts, 0.01) if not closed else pts
        pts = pts.copy()
        for c, top in ((0, w), (1, h)):                   # on the edge of the picture: pushed past it
            pts[:, c] = np.where(pts[:, c] <= 1e-6, -2.0, np.where(pts[:, c] >= top - 1e-6, top + 2.0, pts[:, c]))
        points.append(pts)
    return Borders(points, a, b, start, end, n)


def _grow(ring: np.ndarray, by: float) -> np.ndarray:
    """A ring (going round with its shape on the left) moved ``by`` pixels outward, corners mitred."""
    if by <= 0 or len(ring) < 3:
        return ring
    before = ring - np.roll(ring, 1, 0)
    after = np.roll(ring, -1, 0) - ring
    for v in (before, after):
        v /= np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-9)[:, None]
    n1 = np.stack([before[:, 1], -before[:, 0]], 1)
    n2 = np.stack([after[:, 1], -after[:, 0]], 1)
    s = n1 + n2
    dot = np.clip((n1 * n2).sum(1), -0.9, 1.0)
    return ring + s * (by / (1.0 + dot))[:, None].clip(max=2.0 * by)


def outlines(mask, *, sigma: float = 1.6, tolerance: float = 0.15) -> list:
    """The outlines of a mask ``(H, W)`` as closed rings ``(K, 2)`` in pixels (x, y; pixel ``(i, j)`` covers
    ``j .. j + 1`` by ``i .. i + 1``): subpixel (through the middle of each pixel side on the border, as marching
    squares draws it), smoothed so no stair steps are left while real corners stay sharp, and simplified.
    Holes are rings of their own; filled even-odd (:func:`helix_lm.paint.draw.fill_rings`) they give the mask
    back. A mask that reaches the edge of the picture gets rings that go two pixels past it."""
    m = torch.as_tensor(mask).bool().long()
    if not bool(m.any()):
        return []
    bd = borders(m, 2, sigma=sigma, tolerance=tolerance)
    return [torch.from_numpy(r).float() for r in bd.rings(np.array([False, True, False]))]


def cel_shape(rings: list, colour, scale: float, marker: int = -1, kind: str = "cel") -> Line | None:
    """A cel :class:`Line` from rings in pixels (``scale`` pixels to a board width): opaque brush paint filled
    exactly, even-odd."""
    rings = [torch.as_tensor(r, dtype=torch.float32) for r in rings if len(r) >= 3]
    if not rings:
        return None
    pts = torch.cat(rings) / scale
    return Line(pts, torch.ones(len(pts)), "brush", tuple(float(c) for c in colour), marker, kind,
                rings=[len(r) for r in rings])


# -------------------------------------------------------------------------------------------- shading --
@dataclass
class Shading:
    """The paint of every part of a frame: which region each pixel is on, and which paint (tone) it takes. A
    region is numbered as one of the frame's :func:`parts` (its base part; a shadow tone part beside it belongs
    to it), and ``back``, indexed by those numbers, says which are background. Each region has one base tone
    (``cel``) and may have one shadow tone (``cel-shadow``) and one highlight tone (``cel-light``) on it;
    ``colour``, ``marker`` (the palette colour it belongs to), ``kind`` and ``region`` describe each tone."""

    regions: torch.Tensor      # (H, W) long
    tones: torch.Tensor        # (H, W) long
    colour: torch.Tensor       # (T, 3) sRGB
    marker: torch.Tensor       # (T,) long
    kind: tuple                # (T,) str
    region: torch.Tensor       # (T,) long
    back: torch.Tensor         # (R,) bool

    def _of(self, kind: str) -> torch.Tensor:
        flags = torch.tensor([k == kind for k in self.kind], dtype=torch.bool)
        return flags[self.tones]

    @property
    def shadow(self) -> torch.Tensor:
        """``(H, W)`` bool: the pixels in shadow."""
        return self._of("cel-shadow")

    @property
    def light(self) -> torch.Tensor:
        """``(H, W)`` bool: the pixels in a highlight."""
        return self._of("cel-light")

    def masks(self, region: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The shadow and the highlight of one region, ``(H, W)`` bool each."""
        on = self.regions == region
        return self.shadow & on, self.light & on


def _gentle(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Where two OKLab colours ``(..., 3)`` differ only a little: close, or the same hue and saturation in a
    lighter or darker tone (the steps of a painted gradient)."""
    cx, cy = torch.hypot(x[..., 1], x[..., 2]), torch.hypot(y[..., 1], y[..., 2])
    ratio = torch.maximum(cx, cy) / torch.minimum(cx, cy).clamp(min=1e-3)
    tone = (cx > 0.04) & (cy > 0.04) & (_hue_gap(x, y) < math.radians(20)) & (ratio < 1.5) & \
        ((x[..., 0] - y[..., 0]).abs() < 0.25)
    return (_distance(x, y) < 0.1) | tone


def _backdrop(ids: torch.Tensor, n: int, lab: torch.Tensor, seen: torch.Tensor | None = None, subject=None) -> \
        torch.Tensor:
    """Which parts are background ``(n,)``. A character is a group of parts in colours of its own, set off from
    what is round it; the background is what is left. It starts from the parts that reach the top or a side of
    the picture (a character often stands on the bottom edge, or is cut off by it) and from long thin parts that
    stretch over most of it (a pillar), and spreads from part to part across every border of some length where
    the colour changes only a little (:func:`_gentle`: a glow in the sky, snow in the air, the dark side of a
    pillar). It does not cross the silhouette of a character, where the colour jumps, nor (with the frame
    ``seen (3, H, W)``, OKLab smoothed) a border the frame shows as a crisp edge unless the colours either side
    are nearly the same: a shirt in nearly the colour of the sky behind it is still the character's. ``lab (n,
    3)`` is the OKLab colour of each part. Tiny parts almost wholly inside the background are background too (a
    snowflake). With a ``subject`` mask ``(H, W)``, the parts mostly outside it."""
    h, w = ids.shape
    flat = ids.flatten()
    area = torch.bincount(flat, minlength=n).float()
    if subject is not None:
        inside = torch.zeros(n).index_add_(0, flat, torch.as_tensor(subject).float().flatten())
        return inside / area.clamp(min=1) < 0.5
    span = []
    for c in (torch.arange(w).repeat(h), torch.arange(h).repeat_interleave(w)):
        lo = torch.full((n,), 1 << 30).scatter_reduce(0, flat, c, "amin")
        hi = torch.full((n,), -1).scatter_reduce(0, flat, c, "amax")
        span.append((hi - lo + 1).float())
    long = torch.maximum(span[0] / w, span[1] / h)
    thin = (long >= 0.45) & (torch.minimum(span[0], span[1]) <= 0.25 * torch.maximum(span[0], span[1]))
    edge = torch.cat([ids[0], ids[:, 0], ids[:, -1]])
    back = ((torch.bincount(edge, minlength=n) > 0) & (area >= 0.002 * h * w)) | thin
    if n < 2:
        return back
    if seen is None:
        a, b, count = _adjacency(ids, n)
        soft = torch.ones(len(a), dtype=torch.bool)
    else:
        a, b, count, step = _steps(ids, n, seen)
        gap = _distance(lab[a], lab[b])
        soft = (step < 0.45 * gap) | (gap < 0.05)
    border = torch.zeros(n).index_add_(0, a, count.float())
    gentle = _gentle(lab[a], lab[b]) & soft & (count.float() >= torch.minimum(0.1 * border[b],
                                                                              torch.full_like(border[b], 12.0)))
    ga, gb = a[gentle], b[gentle]
    for _ in range(64):
        grow = torch.zeros(n, dtype=torch.bool)
        grow[gb[back[ga]]] = True
        if not bool((grow & ~back).any()):
            break
        back = back | grow
    share = torch.zeros(n).index_add_(0, a, count.float() * back[b].float()) / border.clamp(min=1)
    back = back | ((share >= 0.97) & (area < 0.001 * h * w))
    # a character is a group of touching parts: a small group far smaller than the largest is a bit of the
    # background the flood went round (a line on a wall, a clump of snow)
    group = torch.arange(n)
    both = ~back[a] & ~back[b]
    ga, gb = a[both], b[both]
    for _ in range(n):
        low = group.clone().scatter_reduce_(0, ga, group[gb], "amin")
        low = low[low]
        if torch.equal(low, group):
            break
        group = low
    size = torch.zeros(n).index_add_(0, group, area * (~back).float())
    keep = size >= min(0.2 * float(size.max()), 0.01 * h * w)
    return back | ~keep[group]


def _relief(mask: torch.Tensor, to_light: torch.Tensor, radius: float) -> torch.Tensor:
    """How much light ``to_light (3,)`` gives a figure ``mask (H, W)`` seen as a rounded relief: the figure is
    blown up like a cushion (each point as high as a circle of ``radius`` pixels allows at its distance from the
    edge) and lit (Lambert, ``(H, W)``; 1 outside). This is how a flat drawing gets a shape to shade when nothing
    else says how it is turned (as in Johnston's "Lumo", 2002, and Sykora's "Ink-and-Ray", 2014)."""
    from .flow import nearest

    h, w = mask.shape
    half = min(h, w) >= 64
    small = (F.avg_pool2d(mask.float()[None, None], 2)[0, 0] > 0.5) if half else mask
    _, dist = nearest(~small)
    if half:
        dist = F.interpolate((dist * 2.0)[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    d = (dist / radius).clamp(0.0, 1.0)
    height = gaussian_blur(radius * (1.0 - (1.0 - d) ** 2).sqrt() * mask, 1.0)
    gx, gy = torch.zeros_like(height), torch.zeros_like(height)
    gx[:, 1:-1] = (height[:, 2:] - height[:, :-2]) / 2.0
    gy[1:-1] = (height[2:] - height[:-2]) / 2.0
    normal = torch.stack([-gx, -gy, torch.ones_like(gx)])
    normal = normal / normal.norm(dim=0, keepdim=True)
    lambert = (normal * to_light[:, None, None]).sum(0)
    return torch.where(mask, lambert, torch.ones_like(lambert))


def _hysteresis(strong: torch.Tensor, weak: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """The parts of ``weak`` (each kept within one region of ``ids``) that hold a pixel of ``strong``."""
    pieces, n = parts(torch.where(weak, ids, torch.full_like(ids, -1)))
    hit = torch.zeros(n, dtype=torch.bool)
    hit[pieces[strong & weak]] = True
    return hit[pieces] & weak


def _tidy(mask: torch.Tensor, ids: torch.Tensor, size: int, least: float) -> torch.Tensor:
    """``mask`` made into clean hard-edged shapes: opened and closed with a square of ``size`` pixels (no thin
    bits, no small holes), kept inside the regions it was in, and pieces smaller than ``least`` pixels left
    out."""
    m = mask[None, None].float()
    if size > 1:
        p = size // 2
        m = F.max_pool2d(-F.max_pool2d(-m, size, 1, p), size, 1, p)            # open
        m = -F.max_pool2d(-F.max_pool2d(m, size, 1, p), size, 1, p)            # close
    out = (m[0, 0] > 0.5) & (ids >= 0)
    # a pixel the closing added keeps only if its own region has some of the mask next to it
    near = F.max_pool2d(torch.where(mask, ids, torch.full_like(ids, -1))[None, None].float(), 3, 1, 1)[0, 0]
    out &= mask | (near.long() == ids)
    pieces, n = parts(torch.where(out, ids, torch.full_like(ids, -1)))
    area = torch.bincount(pieces.flatten(), minlength=n)
    return out & (area[pieces] >= least)


def _tone_parts(ids: torch.Tensor, n: int, paint: torch.Tensor, base: torch.Tensor, area: torch.Tensor,
                back: torch.Tensor):
    """Which parts of a character are a darker or lighter tone of the colour beside them (:func:`_family`), so
    together one region with its shadow and highlight rather than regions of their own: ``(region (n,), role
    (n,))``, the region each part is on and its role there (0 the base, 1 a shadow tone, 2 a highlight tone).
    Parts are grouped across borders between tones of one colour; the base of a group is the colour with the
    most area times chroma squared (the one that makes its look), every part of the group near its hue and
    lightness is a tone of it, darker or lighter, and any other part (an iris among the tones of the skin)
    stays a region of its own. ``paint (n,)`` is each part's palette number."""
    region = torch.arange(n)
    role = torch.zeros(n, dtype=torch.long)
    if n < 2:
        return region, role
    a, b, _ = _adjacency(ids, n)
    kin = _family(base[a], base[b]) & ~back[a] & ~back[b]
    ka, kb = a[kin], b[kin]
    group = torch.arange(n)
    for _ in range(n):
        low = group.clone().scatter_reduce_(0, ka, group[kb], "amin")
        low = low[low]
        if torch.equal(low, group):
            break
        group = low
    k = int(paint.max()) + 1
    # the base is the colour that most makes the look of the group: much of it, and vivid (red hair is
    # painted red with a darker shadow, even where the frame shows more of the dark)
    vivid = (base[:, 1] ** 2 + base[:, 2] ** 2) + 0.002
    used = torch.zeros(n * k).index_add_(0, group * k + paint, area * vivid).view(n, k)
    main = used.argmax(1)
    for g in torch.unique(group[torch.bincount(group, minlength=n)[group] > 1]).tolist():
        members = torch.nonzero(group == g).flatten()
        colour = int(main[g])
        own = members[paint[members] == colour]
        root = int(own[area[own].argmax()])
        main_lab = base[own[0]]
        grey = float(torch.hypot(main_lab[1], main_lab[2])) < 0.045
        for s in members.tolist():
            c = int(paint[s])
            near = grey or float(_hue_gap(main_lab, base[s])) < math.radians(35)
            if c == colour:
                region[s] = root
            elif near and 0.02 < abs(float(base[s, 0] - main_lab[0])) < 0.3:
                region[s] = root
                role[s] = 1 if float(base[s, 0]) < float(main_lab[0]) else 2
    return region, role


def shade(image: torch.Tensor, labels: torch.Tensor, palette: torch.Tensor, normals: torch.Tensor | None = None,
          light: torch.Tensor | None = None, *, detail: float = 0.5, subject=None, relief: bool = True) -> Shading:
    """The shadow (kage) and the highlight (hi) of every region of a frame. ``image (3, H, W)`` is the frame
    (sRGB), ``labels (H, W)`` its regions as palette numbers (:func:`regions`), ``palette (K, 3)`` the colours.

    Background regions (see :class:`Shading`) are painted flat. On a character, the parts painted in a darker
    (or lighter) tone of the colour of the part beside them are that part's shadow (or highlight), in one colour:
    the tone the frame shows most. Where a region has no such tones, the shadow is where the frame is clearly
    darker than the region's palette colour (two levels with hysteresis) and the highlight where it is clearly
    lighter. A part of the character the frame shows flat gets, with ``relief``, the shadow its figure would
    have blown up into a rounded relief (:func:`_relief`): a band along the side away from the light. With
    ``normals (3, H, W)`` (unit vectors, x right, y down, z toward the viewer) the shadow is instead where the
    surface turns from ``light (3,)`` (toward the light; from the upper left and in front by default): Lambert
    below a threshold; and the highlight where the light is reflected toward the viewer. Every shadow and
    highlight is made a clean hard-edged shape of a least size. A shadow found from the light of the frame takes
    the palette's tone of its colour nearest to what the frame shows there, or else the colour made darker, a
    little bluer and (if dull) more saturated (:func:`shadow_colour`); a highlight lighter and warmer
    (:func:`light_colour`)."""
    image = image.float().clamp(0.0, 1.0)
    labels = torch.as_tensor(labels).long()
    h, w = labels.shape
    ids, n = parts(labels)
    pigments = srgb_to_oklab(palette.float())
    flat_ids = ids.flatten()
    paint = torch.zeros(n, dtype=torch.long).scatter_(0, flat_ids, labels.flatten())
    base = pigments[paint]
    area = torch.bincount(flat_ids, minlength=n).float()
    lab = image_to_oklab(image)
    seen = _bilateral(lab)
    back = _backdrop(ids, n, base, seen, subject)
    region, role = _tone_parts(ids, n, paint, base, area, back)
    reg = region[ids]
    sub = role[ids]
    fg = ~back[reg]
    has_tone = torch.zeros(n, dtype=torch.bool)
    has_tone[region[role > 0]] = True
    size = _odd(max(3.0, 0.008 * w * (1.5 - detail)))
    least = max(8.0, (0.02 * w * (1.5 - detail)) ** 2)
    # pixels next to a border are a mix of the colours either side: no shadow or highlight starts there
    edge = torch.zeros(h, w, dtype=torch.bool)
    edge[:, 1:] |= reg[:, 1:] != reg[:, :-1]
    edge[1:] |= reg[1:] != reg[:-1]
    inner = F.max_pool2d(edge[None, None].float(), 5, 1, 2)[0, 0] == 0
    to_light = torch.tensor([-0.5, -0.6, 0.62]) if light is None else torch.as_tensor(light).float()
    to_light = to_light / to_light.norm().clamp(min=1e-6)
    if normals is not None:
        nrm = torch.as_tensor(normals).float()
        nrm = nrm / nrm.norm(dim=0, keepdim=True).clamp(min=1e-6)
        lambert = (nrm * to_light[:, None, None]).sum(0)
        reflect = 2 * lambert * nrm[2] - to_light[2]          # the reflection toward the viewer (0, 0, 1)
        shadow = _hysteresis((lambert < 0.18) & fg, (lambert < 0.3) & fg, reg)
        lit = _hysteresis((reflect > 0.97) & fg, (reflect > 0.93) & fg, reg)
    else:
        diff = seen[0] - base[region][ids][..., 0]
        allowed = fg & ~has_tone[reg]
        shadow = _hysteresis((diff < -0.07) & allowed & inner, (diff < -0.035) & allowed, reg)
        lit = _hysteresis((diff > 0.1) & allowed & inner, (diff > 0.05) & allowed, reg)
        # where the frame paints a region in tones of its colour, the shadow and the highlight are those tones,
        # each smoothed over a few pixels into one clean shape (most of the pixels round it decide)
        toned = fg & has_tone[reg]
        if bool(toned.any()):
            sigma = max(1.0, 0.005 * w * (1.5 - detail))
            weight = gaussian_blur(toned.float(), sigma).clamp(min=1e-6)
            shadow |= toned & (gaussian_blur((toned & (sub == 1)).float(), sigma) / weight > 0.5)
            lit |= toned & (gaussian_blur((toned & (sub == 2)).float(), sigma) / weight > 0.5)
    shadow = _tidy(shadow, reg, size, least)
    lit = _tidy(lit, reg, size, least / 2) & ~shadow
    # a region mostly "in shadow" is only darker than its palette colour: it has no shadow of its own; and a
    # highlight is a small shape, not most of a part
    region_area = torch.bincount(reg.flatten(), minlength=n).float().clamp(min=1)
    for m, most in ((shadow, 0.6), (lit, 0.35)):
        share = torch.bincount(reg[m], minlength=n).float() / region_area
        m &= share[reg] < most
    if relief and normals is None and bool(fg.any()):
        # a part of the character the frame shows flat: shaded as the rounded figure would be
        flat = torch.ones(n, dtype=torch.bool)
        flat[reg[shadow]] = False
        more = _tidy((_relief(fg, to_light, 0.03 * w) < 0.3) & fg & flat[reg] & ~lit, reg, size, least)
        shadow |= more
    # the colour of each region's shadow and highlight: the tone the frame shows most there
    colour, marker, kind, owner = [], [], [], []
    base_tone = torch.full((n,), -1, dtype=torch.long)
    for r in torch.nonzero(region == torch.arange(n)).flatten().tolist():
        base_tone[r] = len(colour)
        colour.append(palette[paint[r]].float())
        marker.append(int(paint[r]))
        kind.append("cel")
        owner.append(r)
    tones = base_tone[reg]
    for m, name, role_id, make in ((shadow, "cel-shadow", 1, shadow_colour), (lit, "cel-light", 2, light_colour)):
        if not bool(m.any()):
            continue
        k = len(palette)
        used = torch.bincount(reg[m] * k + labels[m], minlength=n * k).view(n, k).float()
        mean = torch.zeros(n, 3).index_add_(0, reg[m], seen[:, m].T) / used.sum(1).clamp(min=1)[:, None]
        tone_ids = torch.full((n,), -1, dtype=torch.long)
        for r in torch.nonzero(used.sum(1) > 0).flatten().tolist():
            mine = int(paint[r])
            counts = used[r].clone()
            counts[mine] = 0
            tone_of = [int(i) for i in torch.nonzero(counts > 0).flatten().tolist()
                       if bool(_family(pigments[mine], pigments[i]))
                       and (float(pigments[i, 0]) < float(pigments[mine, 0])) == (role_id == 1)]
            if tone_of:
                pick = max(tone_of, key=lambda i: float(counts[i]))
                made, mark = palette[pick].float(), pick
            else:
                # the palette's own tone of this colour, when it is nearly what the frame shows here
                gap = _distance(pigments, mean[r])
                pick = int(gap.argmin())
                made, mark = make(palette[mine].float()), mine
                if pick != mine and float(gap[pick]) < 0.05 and bool(_family(pigments[mine], pigments[pick])) and \
                        (float(pigments[pick, 0]) < float(pigments[mine, 0])) == (role_id == 1):
                    made, mark = palette[pick].float(), pick
            tone_ids[r] = len(colour)
            colour.append(made)
            marker.append(mark)
            kind.append(name)
            owner.append(r)
        tones = torch.where(m, tone_ids[reg], tones)
    return Shading(reg, tones, torch.stack(colour), torch.tensor(marker), tuple(kind), torch.tensor(owner), back)


def _flat_shading(labels: torch.Tensor, palette: torch.Tensor | None, image: torch.Tensor, subject=None) -> Shading:
    """A :class:`Shading` with no shadows: each region one tone, its palette colour (or the frame's mean)."""
    ids, n = parts(labels)
    flat = ids.flatten()
    paint = torch.zeros(n, dtype=torch.long).scatter_(0, flat, torch.as_tensor(labels).long().flatten())
    if palette is not None:
        colour = palette.float()[paint]
    else:
        lab = image_to_oklab(image.float().clamp(0, 1)).reshape(3, -1).T
        mean = torch.zeros(n, 3).index_add_(0, flat, lab) / torch.bincount(flat, minlength=n).float()[:, None]
        colour = _to_srgb(mean)
    back = _backdrop(ids, n, srgb_to_oklab(colour), _bilateral(image_to_oklab(image.float().clamp(0, 1))), subject)
    return Shading(ids, ids.clone(), colour, paint, ("cel",) * n, torch.arange(n), back)


# ------------------------------------------------------------------------------------------- line art --
SILHOUETTE, INNER, TRACE, DETAIL = 1.0, 0.4, 0.12, 0.6      # pen pressure of each kind of line


def _tangent(p: np.ndarray, at_end: bool, reach: float = 4.0) -> np.ndarray:
    """The way a path leaves its start (or its end), over about ``reach`` pixels."""
    q = p[::-1] if at_end else p
    step = np.hypot(*(q[1:] - q[:-1]).T)
    s = np.cumsum(step)
    i = int(np.searchsorted(s, reach))
    v = q[min(i + 1, len(q) - 1)] - q[0]
    return v / max(1e-9, math.hypot(v[0], v[1]))


def _strokes(bd: Borders, style: np.ndarray, key: list, pressure: np.ndarray, least: float) -> list:
    """Border chains joined into strokes a pen draws in one go: at each junction the two ends that go on most
    nearly straight (and are drawn with the same ink) are joined, the others end there. Returns ``(points (K, 2),
    pressure (K,), closed, key)`` in pixels."""
    drawn = np.flatnonzero(style > 0).tolist()
    ends: dict = {}
    for c in drawn:
        if bd.start[c] >= 0:
            ends.setdefault(int(bd.start[c]), []).append((c, 0))
            ends.setdefault(int(bd.end[c]), []).append((c, 1))
    link: dict = {}
    for here in ends.values():
        if len(here) < 2:
            continue
        cand = []
        for i in range(len(here)):
            for j in range(i + 1, len(here)):
                (c1, e1), (c2, e2) = here[i], here[j]
                if key[c1] != key[c2] or c1 == c2:
                    continue
                t1, t2 = _tangent(bd.points[c1], e1 == 1), _tangent(bd.points[c2], e2 == 1)
                turn = math.pi - math.acos(max(-1.0, min(1.0, float((t1 * t2).sum()))))
                most = math.radians(100 if style[c1] == style[c2] == 1 else 60)
                if turn <= most:
                    cand.append((turn, i, j))
        used: set = set()
        for _, i, j in sorted(cand):
            if i in used or j in used:
                continue
            used |= {i, j}
            link[here[i]] = here[j]
            link[here[j]] = here[i]
    out, done = [], set()

    def piece(c: int, enter: int):
        p = bd.points[c] if enter == 0 else bd.points[c][::-1]
        return p, np.full(len(p), pressure[c])

    for c in sorted(drawn, key=lambda c: -len(bd.points[c])):
        if c in done:
            continue
        if bd.start[c] < 0:
            done.add(c)
            p, q = piece(c, 0)
            out.append((p, q, True, key[c]))
            continue
        # back to where the stroke starts: a free end, or round a loop to this chain again
        cur, enter = c, 0
        while True:
            prev = link.get((cur, enter))
            if prev is None or prev[0] == c:
                break
            cur, enter = prev[0], 1 - prev[1]
        seq = []
        while cur not in done:
            done.add(cur)
            seq.append((cur, enter))
            nxt = link.get((cur, 1 - enter))
            if nxt is None:
                break
            cur, enter = nxt
        last, last_enter = seq[-1]
        closed = len(seq) > 1 and link.get((last, 1 - last_enter)) == seq[0]
        pts, prs = [], []
        for i, (cc, ee) in enumerate(seq):
            p, q = piece(cc, ee)
            pts.append(p if i == 0 else p[1:])
            prs.append(q if i == 0 else q[1:])
        p, q = np.concatenate(pts), np.concatenate(prs)
        if closed:
            p, q = p[:-1], q[:-1]
        out.append((p, q, closed, key[c]))
    kept = []
    for p, q, closed, k in out:
        length = float(np.hypot(*(p[1:] - p[:-1]).T).sum()) if len(p) > 1 else 0.0
        if length >= least or (closed and length >= 2):
            kept.append((p, q, closed, k))
    return kept


def _stroke_line(p: np.ndarray, q: np.ndarray, closed: bool, tool: str, colour, scale: float) -> Line:
    """A stroke as a pen :class:`Line` (points in board widths). The pressure eases from one kind of line to the
    next, and a closed stroke starts at its sharpest corner and runs on past it, so the taper where the pen
    lands is drawn over."""
    if closed and len(p) >= 4:
        turn = _turn(p, 2, True)
        s = int(turn.argmax())
        p, q = np.roll(p, -s, 0), np.roll(q, -s, 0)
        step = np.hypot(*(np.roll(p, -1, 0) - p).T)
        total = float(step.sum())
        run = np.cumsum(step)
        k = int(np.searchsorted(run, 0.16 * total)) + 2
        p = np.concatenate([p, p[: min(k, len(p))]])
        q = np.concatenate([q, q[: min(k, len(q))]])
    if len(q) >= 5:
        q = np.convolve(np.pad(q, 2, mode="edge"), np.ones(5) / 5, "valid")
    return Line(torch.from_numpy(p / scale).float(), torch.from_numpy(q).float().clamp(0.0, 1.0), tool, colour, -1,
                "line")


def _darker(colour: torch.Tensor, by: float = 0.09) -> torch.Tensor:
    """Darker tones of sRGB colours ``(..., 3)``, for colour trace lines."""
    lab = srgb_to_oklab(colour.float())
    lab[..., 0] = lab[..., 0] - by
    lab[..., 1:] = lab[..., 1:] * 1.08
    return _to_srgb(lab)


def _marks(image: torch.Tensor, shading: Shading, inked: torch.Tensor, w: int) -> list:
    """The small dark marks of the frame on a character (an eye, a nostril, a mouth, a line of fur) that the
    paint has lost: pixels darker than their paint and unlike any mix of it with the paint a few pixels away (so
    not the blend along a border), away from the ink already drawn (``inked (H, W)``). Returns pixel paths
    ``(K, 2)`` (x, y), thinned to one pixel and followed like a pen; a dot is a path of two points."""
    lab = image_to_oklab(image)
    own = srgb_to_oklab(shading.colour)[shading.tones].permute(2, 0, 1)
    wt = WEIGHT[:, None, None]
    best = ((lab - own) * wt).norm(dim=0)
    r = 2
    h, wd = own.shape[-2:]
    pad = F.pad(own[None], (r, r, r, r), mode="replicate")[0]
    for dy, dx in ((-r, 0), (r, 0), (0, -r), (0, r), (-r, -r), (r, r), (-r, r), (r, -r)):
        seg = pad[:, r + dy : r + dy + h, r + dx : r + dx + wd] - own
        t = (((lab - own) * seg * wt ** 2).sum(0) / ((seg * wt) ** 2).sum(0).clamp(min=1e-9)).clamp(0.0, 1.0)
        best = torch.minimum(best, ((lab - own - t * seg) * wt).norm(dim=0))
    marks = (best > 0.12) & (lab[0] < own[0] - 0.06) & ~shading.back[shading.regions] & ~inked
    if int(marks.sum()) < 3:
        return []
    pieces, n = parts(marks.long())
    area = torch.bincount(pieces.flatten(), minlength=n)
    marks &= area[pieces] >= 3
    paths = trace(thin(marks), least=3)
    out = [p for p in paths if float((p[1:] - p[:-1]).norm(dim=-1).sum()) >= 2.0]
    drawn = torch.zeros(h, wd, dtype=torch.bool)
    for p in out:
        drawn[p[:, 1].long().clamp(0, h - 1), p[:, 0].long().clamp(0, wd - 1)] = True
    hit = torch.zeros(n, dtype=torch.bool)
    hit[pieces[drawn & marks]] = True
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32) + 0.5, torch.arange(wd, dtype=torch.float32) + 0.5,
                            indexing="ij")
    for k in torch.nonzero((area >= 3) & ~hit).flatten().tolist():
        m = (pieces == k) & marks
        if not bool(m.any()):
            continue
        c = torch.stack([xs[m].mean(), ys[m].mean()])
        out.append(torch.stack([c - torch.tensor([0.4, 0.0]), c + torch.tensor([0.4, 0.0])]))
    return out


def line_art(labels: torch.Tensor, image: torch.Tensor, *, palette: torch.Tensor | None = None,
             shading: Shading | None = None, tool: str = "pen", detail: float = 0.5, background: bool = True,
             trace_lines: bool = True, subject=None, bd: Borders | None = None) -> list:
    """The ink lines of a frame ``image (3, H, W)`` with regions ``labels (H, W)`` (:func:`regions`): pen
    :class:`Line` objects (points in board widths of the frame's width) along the borders between regions,
    followed from junction to junction and joined through junctions into long strokes. The silhouette (a border
    between a character and the background) is pressed hardest, so drawn heaviest; lines inside a character are
    lighter; every line tapers where the pen lands and lifts (see :func:`helix_lm.paint.draw.radii`). Borders
    between nearly the same colours are not drawn; a border between tones of one colour, and the edge of a
    shadow, is a thin colour trace line in a darker tone of the colour (``trace_lines`` False leaves them out).
    Borders inside the background are painted, not inked: only strong ones get a colour line, and with
    ``background`` False none. The small dark marks of the frame on a character that the paint lost (an eye, a
    mouth) are drawn too. ``shading`` (:func:`shade`) and ``bd`` (its :func:`borders`) are what
    :func:`paint_frame` already has."""
    image = image.float().clamp(0.0, 1.0)
    labels = torch.as_tensor(labels).long()
    h, w = labels.shape
    scale = float(w)
    shading = _flat_shading(labels, palette, image, subject) if shading is None else shading
    if bd is None:
        bd = borders(shading.tones, len(shading.kind), sigma=_sigmas(shading))
    tone_lab = srgb_to_oklab(shading.colour)
    count = len(shading.kind)
    base_tone = torch.zeros(len(shading.back), dtype=torch.long)
    for t, k in enumerate(shading.kind):
        if k == "cel":
            base_tone[shading.region[t]] = t
    ink = TOOLS[tool].ink
    a, b = torch.from_numpy(bd.a), torch.from_numpy(bd.b)
    off = (a >= count) | (b >= count)
    ta, tb = a.clamp(max=count - 1), b.clamp(max=count - 1)
    ra, rb = shading.region[ta], shading.region[tb]
    ba, bb = shading.back[ra], shading.back[rb]
    la, lb = tone_lab[base_tone[ra]], tone_lab[base_tone[rb]]
    gap = _distance(la, lb)
    skip = 0.045 + 0.04 * (1.0 - detail)
    same = ra == rb                                        # the edge of a shadow or a highlight
    tone_gap = _distance(tone_lab[ta], tone_lab[tb])
    kin = _family(la, lb) & (gap < 0.15)
    style = torch.zeros(len(a), dtype=torch.long)
    silhouette = ~off & ~same & (ba ^ bb)
    inner = ~off & ~same & ~ba & ~bb & (gap >= skip) & ~kin
    shadow_edge = ~off & same & (tone_gap >= skip) & (torch.tensor(background) | ~ba)
    tone_edge = ~off & ~same & ~ba & ~bb & (gap >= skip) & kin
    back_edge = ~off & ~same & ba & bb & (gap > 0.14) & torch.tensor(background)
    style[silhouette], style[inner] = 1, 2
    if trace_lines:
        style[shadow_edge | tone_edge | back_edge] = 3
    pressure = torch.where(style == 1, SILHOUETTE, torch.where(style == 2, INNER, TRACE)).numpy()
    # the colour of a colour trace line: a darker tone of the darker side
    dark_tone = torch.where(tone_lab[ta, 0] < tone_lab[tb, 0], ta, tb)
    dark_base = torch.where(la[:, 0] < lb[:, 0], base_tone[ra], base_tone[rb])
    traced = torch.where(same, dark_tone, dark_base)
    tones_dark = [tuple(c) for c in _darker(shading.colour, 0.05).tolist()]
    back_dark = [tuple(c) for c in _darker(shading.colour, 0.06).tolist()]
    key: list = []
    for i, st in enumerate(style.tolist()):
        if st in (1, 2):
            key.append("ink")
        elif st == 3:
            key.append(back_dark[int(traced[i])] if bool(back_edge[i]) else tones_dark[int(traced[i])])
        else:
            key.append(None)
    out = []
    for p, q, closed, k in _strokes(bd, style.numpy(), key, pressure, max(2.5, 0.006 * w)):
        out.append(_stroke_line(p, q, closed, tool, ink if k == "ink" else k, scale))

    inked = stamp([line for line in out if line.colour == tuple(float(c) for c in ink)], h, w, scale, grow=1.5)
    for path in _marks(image, shading, inked, w):
        pts = smooth_path(path, spacing=1.5, sigma=1.0) if len(path) >= 5 else path
        if len(pts) >= 2:
            out.append(Line(pts / scale, torch.full((len(pts),), DETAIL), tool, ink, -1, "line"))
    return out


def _sigmas(shading: Shading):
    """How smooth each border is made: the edge of a shadow and the borders inside the background a little
    smoother (they are soft in the frame) than the borders a pen follows."""
    region = shading.region.numpy()
    back = shading.back.numpy()
    count = len(region)

    def sigma(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ra = region[np.minimum(a, count - 1)]
        rb = region[np.minimum(b, count - 1)]
        soft = (ra == rb) | (back[ra] & back[rb])
        return np.where(soft, 2.6, 1.6)

    return sigma


# ---------------------------------------------------------------------------------------------- frame --
def cels(shading: Shading, bd: Borders, scale: float, *, background: bool = True, grow: float = 0.5) -> list:
    """The cel paint of a frame: for each region its base colour over the whole region (under its shadows and
    highlights), then each shadow and highlight shape, all built from the shared borders ``bd`` (so neighbours
    meet exactly) and grown ``grow`` pixels so that no seam of the board shows between them. ``scale`` is the
    frame's width in pixels."""
    tone_region = shading.region.numpy()
    count = len(tone_region)
    out = []
    for r in range(len(shading.back)):
        if not background and bool(shading.back[r]):
            continue
        mine = np.flatnonzero(tone_region == r)
        if len(mine) == 0:
            continue
        for t in mine.tolist():
            kind = shading.kind[t]
            inside = np.zeros(count + 1, dtype=bool)
            if kind == "cel":
                inside[mine] = True
            else:
                inside[t] = True
            rings = [_grow(ring, grow) for ring in bd.rings(inside)]
            line = cel_shape(rings, shading.colour[t].tolist(), scale, int(shading.marker[t]), kind)
            if line is not None:
                out.append(line)
    return out


def paint_frame(image: torch.Tensor, palette: torch.Tensor | None = None, *, normals: torch.Tensor | None = None,
                light: torch.Tensor | None = None, line_tool: str = "pen", detail: float = 0.5,
                background: bool = True, trace_lines: bool = True, subject=None, relief: bool = True) -> list:
    """One frame ``image (3, H, W)`` (sRGB) inked and painted as an anime cel: cel paint (``cel`` base colours
    from ``palette``, ``cel-shadow`` and ``cel-light`` shapes; ``marker`` is the palette number) and line art
    with ``line_tool`` (``pen``, ``brush pen``), in the order a person draws them (:func:`helix_lm.paint.draw.order`:
    the lines, then the paint, base before shadow before highlight). Points are in board widths of the frame's
    width. Give the same ``palette`` (:func:`palette_for` of all frames) to every frame of an animation.
    ``normals`` and ``light`` shade from the shape of the surface (:func:`shade`); ``detail`` 0..1 keeps smaller
    parts; ``background`` False paints only the character; ``subject`` (an ``(H, W)`` mask) says where the
    character is when the frame does not make it plain; ``relief`` False leaves a part the frame shows flat
    without a shadow; ``trace_lines`` False draws no colour trace lines."""
    image = image.float().clamp(0.0, 1.0)
    palette = palette_for([image]) if palette is None else palette.float()
    w = image.shape[-1]
    labels = regions(image, palette, detail=detail)
    shading = shade(image, labels, palette, normals, light, detail=detail, subject=subject, relief=relief)
    bd = borders(shading.tones, len(shading.kind), sigma=_sigmas(shading))
    ink = line_art(labels, image, palette=palette, shading=shading, tool=line_tool, detail=detail,
                   background=background, trace_lines=trace_lines, bd=bd)
    return order(ink + cels(shading, bd, float(w), background=background))


__all__ = ["Borders", "Shading", "borders", "cel_shape", "cels", "light_colour", "line_art", "outlines",
           "paint_frame", "palette_for", "parts", "regions", "shade", "shadow_colour"]
