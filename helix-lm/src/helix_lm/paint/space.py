# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The space a clip happens in: how near each thing is, which way its surfaces face and where the light comes from,
what moves (the character) and what stays (the set), how the camera moves, and one clean picture of the whole set
with the character taken out.

An anime background is painted once, in detail, and the characters are animated over it; when the camera pans,
the painting slides under them. That is what this module is for. Redrawing the set in every frame makes it boil
and morph; drawing it once from :func:`background_plate` and holding it (or sliding it with :func:`camera`) keeps
it still and sharp, and only the character (:func:`separate`) is drawn frame by frame.

Points are in pixels, x then y, with the top left corner of a picture at (0, 0), so the centre of pixel ``(i, j)``
is at ``(j + 0.5, i + 0.5)`` (as in :func:`helix_lm.paint.flow.sample`). A camera is a ``(2, 3)`` matrix that
carries a point of frame 0 to where the same piece of background is in frame t.

The order of calls for a clip ``frames (T, 3, H, W)`` (or :func:`read_space`, which does all of it)::

    flows = clip_flows(frames)
    depth = estimate_depth(frames, flows=flows)        # None without transformers or the model
    character = separate(frames, flows, depth)
    cameras = camera(flows, ~character, frames)
    plate = background_plate(frames, character, cameras)
    background_t = place_background(plate, cameras[t], H, W)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .color import image_to_oklab, srgb_to_linear
from .flow import optical_flow, warp
from .see import gaussian_blur

# Depth Anything V2 Small is Apache-2.0 (its model card on the Hugging Face Hub, checked in November 2026); the
# Base and Large checkpoints are CC-BY-NC-4.0, not for commercial use, so they are not the default.
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


# ------------------------------------------------------------------------------------------------ helpers --
def _grid(h: int, w: int) -> torch.Tensor:
    """The centres of the pixels of an ``h`` by ``w`` picture, ``(2, h, w)``, x then y."""
    gy, gx = torch.meshgrid(torch.arange(h, dtype=torch.float32) + 0.5, torch.arange(w, dtype=torch.float32) + 0.5,
                            indexing="ij")
    return torch.stack([gx, gy])


def _apply(matrix: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """``matrix (2, 3)`` applied to ``points (2, ...)``."""
    m = matrix.to(points.dtype)
    flat = points.reshape(2, -1)
    return (m[:, :2] @ flat + m[:, 2:]).reshape(points.shape)


def _square(matrix: torch.Tensor) -> torch.Tensor:
    return torch.cat([matrix.double(), torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)])


def _compose(after: torch.Tensor, before: torch.Tensor) -> torch.Tensor:
    """The matrix that does ``before`` and then ``after``."""
    return (_square(after) @ _square(before))[:2].float()


def _inverse(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(_square(matrix))[:2].float()


def _identity() -> torch.Tensor:
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _look_up(image: torch.Tensor, points: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """``image (C, H, W)`` at ``points (2, h, w)`` (pixels): ``(C, h, w)``, the edge repeated outside."""
    hh, ww = image.shape[-2:]
    grid = torch.stack([points[0] / ww * 2 - 1, points[1] / hh * 2 - 1], -1)[None]
    return F.grid_sample(image[None].float(), grid, mode=mode, padding_mode="border", align_corners=False)[0]


def _invert_flow(flow: torch.Tensor, rounds: int = 3) -> torch.Tensor:
    """The flow back: for ``flow (2, H, W)`` from a to b, where each pixel of b came from in a."""
    back = -flow
    for _ in range(rounds):
        back = -warp(flow, back)
    return back


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """``mask (..., H, W)`` grown by a square ``2 * radius + 1`` wide (a row and then a column, the same thing
    and much faster for a large square), worked out only in the box round the mask."""
    if radius <= 0:
        return mask
    if mask.dim() > 2:
        return torch.stack([_dilate(m, radius) for m in mask.reshape(-1, *mask.shape[-2:])]).reshape(mask.shape)
    box = _bounds(mask, radius)
    if box is None:
        return mask.clone()
    k = 2 * radius + 1
    m = mask[box].float()[None, None]
    m = F.max_pool2d(m, (1, k), 1, (0, radius))
    m = F.max_pool2d(m, (k, 1), 1, (radius, 0))
    out = torch.zeros_like(mask, dtype=torch.bool)
    out[box] = m[0, 0] > 0.5
    return out


def _bounds(mask: torch.Tensor, margin: int) -> tuple[slice, slice] | None:
    """The rows and columns round ``mask (H, W)``, ``margin`` pixels wider (None when it is empty)."""
    rows, cols = mask.any(1).nonzero()[:, 0], mask.any(0).nonzero()[:, 0]
    if len(rows) == 0:
        return None
    h, w = mask.shape
    return (slice(max(0, int(rows[0]) - margin), min(h, int(rows[-1]) + margin + 1)),
            slice(max(0, int(cols[0]) - margin), min(w, int(cols[-1]) + margin + 1)))


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """``mask`` shrunk by a square ``2 * radius + 1`` wide (the edge of the picture counts as on the mask)."""
    if radius <= 0:
        return mask
    if mask.dim() > 2:
        return torch.stack([_erode(m, radius) for m in mask.reshape(-1, *mask.shape[-2:])]).reshape(mask.shape)
    box = _bounds(mask, 0)
    if box is None:
        return mask.clone()
    k = 2 * radius + 1
    m = F.pad(mask[box].float()[None, None], (radius,) * 4, value=0.0)
    h, w = mask.shape
    # beyond the edge of the picture counts as on the mask; inside the picture, beyond the box it is off
    m[..., : radius, :] = 1.0 if box[0].start == 0 else 0.0
    m[..., -radius:, :] = 1.0 if box[0].stop == h else 0.0
    m[..., :, : radius] = torch.maximum(m[..., :, : radius], torch.tensor(1.0 if box[1].start == 0 else 0.0))
    m[..., :, -radius:] = torch.maximum(m[..., :, -radius:], torch.tensor(1.0 if box[1].stop == w else 0.0))
    m = -F.max_pool2d(-m, (1, k), 1, 0)
    m = -F.max_pool2d(-m, (k, 1), 1, 0)
    out = torch.zeros_like(mask, dtype=torch.bool)
    out[box] = m[0, 0] > 0.5
    return out


def _box(x: torch.Tensor, radius: int) -> torch.Tensor:
    """The mean of ``x (..., H, W)`` over a square ``2 * radius + 1`` wide (only the part inside the picture)."""
    lead = x.shape[:-2]
    y = x.reshape(-1, 1, *x.shape[-2:])
    y = F.avg_pool2d(y, 2 * radius + 1, 1, radius, count_include_pad=False)
    return y.reshape(*lead, *x.shape[-2:])


def _components(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The 8-connected parts of ``mask (H, W)``: for every pixel a number for its part, below ``H * W`` (-1 off
    the mask), and the size of the part it is in (0 off the mask). Every pixel points to a smaller one of its
    part; along each pair of neighbours the larger root is hung under the smaller, and the pointers are followed
    to their roots (Shiloach and Vishkin, 1982), so even a long winding part takes few rounds. Only the box round
    the mask is worked on."""
    h, w = mask.shape
    box = _bounds(mask, 0)
    if box is None:
        return torch.full((h, w), -1, dtype=torch.long), torch.zeros(h, w, dtype=torch.long)
    if (box[0].stop - box[0].start) * (box[1].stop - box[1].start) < h * w:
        part_label, part_area = _components(mask[box])
        label, area = torch.full((h, w), -1, dtype=torch.long), torch.zeros(h, w, dtype=torch.long)
        label[box], area[box] = part_label, part_area
        return label, area
    n = h * w
    flat_mask = mask.flatten()
    on = flat_mask.nonzero()[:, 0]
    index = torch.arange(n).reshape(h, w)
    pairs = []
    for a, b in ((mask[:, :-1] & mask[:, 1:], (index[:, :-1], index[:, 1:])),
                 (mask[:-1] & mask[1:], (index[:-1], index[1:])),
                 (mask[:-1, :-1] & mask[1:, 1:], (index[:-1, :-1], index[1:, 1:])),
                 (mask[:-1, 1:] & mask[1:, :-1], (index[:-1, 1:], index[1:, :-1]))):
        pairs.append(torch.stack([b[0][a], b[1][a]]))
    u, v = torch.cat(pairs, 1)
    parent = torch.arange(n)
    while True:
        pu, pv = parent[u], parent[v]
        differ = pu != pv
        if not bool(differ.any()):
            break
        lo, hi = torch.minimum(pu, pv)[differ], torch.maximum(pu, pv)[differ]
        parent.scatter_reduce_(0, hi, lo, reduce="amin")
        while True:
            up = parent[parent]
            if torch.equal(up, parent):
                break
            parent = up
    label = torch.where(flat_mask, parent, torch.full_like(parent, -1))
    sizes = torch.bincount(label[on], minlength=n)
    area = torch.zeros(n, dtype=torch.long)
    area[on] = sizes[label[on]]
    return label.reshape(h, w), area.reshape(h, w)


def _fill_holes(mask: torch.Tensor, largest: float) -> torch.Tensor:
    """``mask`` with the holes in it of at most ``largest`` pixels filled (a hole is a part of what is off the
    mask that does not reach the edge of the picture). Only the box round the mask is looked at: all that lies
    outside it is one with the edge of the picture."""
    box = _bounds(mask, 1)
    if box is None:
        return mask
    part = mask[box]
    label, area = _components(~part)
    if not bool((label >= 0).any()):
        return mask
    edge = torch.cat([label[0], label[-1], label[:, 0], label[:, -1]])
    edge = edge[edge >= 0].unique()
    reaches = torch.zeros(part.numel() + 1, dtype=torch.bool)
    reaches[edge] = True
    hole = (label >= 0) & ~reaches[label.clamp(min=0)] & (area <= largest)
    out = mask.clone()
    out[box] = part | hole
    return out


def _drop_small(mask: torch.Tensor, least: float) -> torch.Tensor:
    _, area = _components(mask)
    return mask & (area >= least)


def _guided(guide: torch.Tensor, source: torch.Tensor, radius: int, eps: float) -> torch.Tensor:
    """The guided filter (He, Sun and Tang, 2010): ``source (H, W)`` made to follow the edges of ``guide (C, H,
    W)``. Inside a part of even colour it is ``source`` averaged; across an edge of the guide it changes as
    sharply as the guide does."""
    c, h, w = guide.shape
    mean_i = _box(guide, radius)
    mean_p = _box(source, radius)
    cov_ip = _box(guide * source, radius) - mean_i * mean_p
    pairs = _box(guide[:, None] * guide[None], radius) - mean_i[:, None] * mean_i[None]
    system = pairs.permute(2, 3, 0, 1).reshape(-1, c, c) + eps * torch.eye(c)
    a = torch.linalg.solve(system, cov_ip.permute(1, 2, 0).reshape(-1, c, 1))[..., 0]
    a = a.T.reshape(c, h, w)
    b = mean_p - (a * mean_i).sum(0)
    return (_box(a, radius) * guide).sum(0) + _box(b, radius)


def _fill(image: torch.Tensor, known: torch.Tensor, rounds: int = 24) -> torch.Tensor:
    """``image (C, H, W)`` where ``known (H, W)``, and smoothly filled from around where not: pulled down a
    pyramid and pushed back up (a smooth fill, like a wash of paint, not invented detail)."""
    if bool(known.all()) or not bool(known.any()):
        return image
    k = known.float()[None]
    levels = [(image * k, k)]
    while not bool((levels[-1][1] > 0).all()) and min(levels[-1][1].shape[-2:]) > 1:
        num, den = levels[-1]
        levels.append((F.avg_pool2d(num[None], 2, ceil_mode=True)[0], F.avg_pool2d(den[None], 2, ceil_mode=True)[0]))
    num, den = levels[-1]
    filled = num / den.clamp(min=1e-8)
    for num, den in reversed(levels[:-1]):
        up = F.interpolate(filled[None], size=num.shape[-2:], mode="bilinear", align_corners=False)[0]
        a = den.clamp(0.0, 1.0)
        filled = a * num / den.clamp(min=1e-8) + (1.0 - a) * up
    out = torch.where(known[None], image, filled)
    for _ in range(rounds):                                    # smooth the blocks of the pyramid away
        out = torch.where(known[None], image, _box(out, 1))
    return out


def _lightness_smooth(image: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    return gaussian_blur(image_to_oklab(image)[0], sigma)


def _texture(light: torch.Tensor, window: float = 2.5) -> torch.Tensor:
    """How well a pixel's motion can be seen: the smaller eigenvalue of the structure tensor (as in
    :func:`helix_lm.paint.flow.optical_flow`)."""
    gx, gy = torch.zeros_like(light), torch.zeros_like(light)
    gx[:, 1:-1] = (light[:, 2:] - light[:, :-2]) / 2.0
    gy[1:-1] = (light[2:] - light[:-2]) / 2.0
    sxx, syy, sxy = (gaussian_blur(v, window) for v in (gx * gx, gy * gy, gx * gy))
    return (0.5 * (sxx + syy) - (0.25 * (sxx - syy) ** 2 + sxy ** 2).sqrt()).clamp(min=0.0)


# --------------------------------------------------------------------------------------------- motion --
KINDS = ("translation", "similarity", "affine")


def _jacobian(points: torch.Tensor, kind: str) -> torch.Tensor:
    """How a point ``(N, 2)`` (normalised) moves with each number of a motion of ``kind``: ``(N, 2, P)``."""
    x, y = points[:, 0], points[:, 1]
    one, zero = torch.ones_like(x), torch.zeros_like(x)
    if kind == "translation":
        rows = [[one, zero], [zero, one]]
    elif kind == "similarity":
        rows = [[x, -y, one, zero], [y, x, zero, one]]
    elif kind == "affine":
        rows = [[x, y, one, zero, zero, zero], [zero, zero, zero, x, y, one]]
    else:
        raise ValueError(f"a camera motion is one of {', '.join(KINDS)}, not {kind!r}")
    return torch.stack([torch.stack(r, -1) for r in rows], 1)


def _matrix(theta: torch.Tensor, kind: str, centre: torch.Tensor, scale: float) -> torch.Tensor:
    """The motion ``theta`` (normalised coordinates) as a ``(2, 3)`` matrix on pixels: ``x -> x + move(x)``."""
    t = theta.double()
    if kind == "translation":
        lin, shift = torch.eye(2, dtype=torch.float64), t
    elif kind == "similarity":
        lin = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64) + torch.stack(
            [torch.stack([t[0], -t[1]]), torch.stack([t[1], t[0]])])
        shift = t[2:4]
    else:
        lin = torch.eye(2, dtype=torch.float64) + torch.stack([t[0:2], t[3:5]])
        shift = torch.stack([t[2], t[5]])
    c = centre.double()
    return torch.cat([lin, (c + scale * shift - lin @ c)[:, None]], 1).float()


def _fit_motion(flow: torch.Tensor, weight: torch.Tensor | None = None, kind: str = "similarity",
                rounds: int = 8, most: int = 30000) -> tuple[torch.Tensor, float]:
    """The one camera motion ``(2, 3)`` that best explains ``flow (2, H, W)`` where ``weight (H, W)`` allows, by
    least squares reweighted so that what moves on its own (the character) is outvoted (Tukey's biweight). Also
    returns how far, in pixels, the flow of a typical background pixel is from it."""
    h, w = flow.shape[-2:]
    step = max(1, int(math.sqrt(h * w / most)))
    points = _grid(h, w)[:, ::step, ::step].reshape(2, -1).T
    moves = flow[:, ::step, ::step].reshape(2, -1).T
    wt = torch.ones(len(points)) if weight is None else weight[::step, ::step].reshape(-1).float()
    keep = wt > 0
    if int(keep.sum()) < 8:
        return _identity(), 0.0
    points, moves, wt = points[keep].double(), moves[keep].double(), wt[keep].double()
    centre = torch.tensor([w / 2.0, h / 2.0], dtype=torch.float64)
    scale = max(h, w) / 2.0
    jac = _jacobian((points - centre) / scale, kind)
    target = moves / scale
    robust = torch.ones_like(wt)
    theta = torch.zeros(jac.shape[-1], dtype=torch.float64)
    spread = 0.0
    p = jac.shape[-1]
    rows = jac.reshape(-1, p)                         # x and y of every point, one row each
    goal = target.reshape(-1)
    for _ in range(rounds):
        ww = (wt * robust).repeat_interleave(2)
        lhs = rows.T @ (rows * ww[:, None]) + 1e-12 * torch.eye(p, dtype=torch.float64)
        theta = torch.linalg.solve(lhs, rows.T @ (goal * ww))
        off = (rows @ theta - goal).reshape(-1, 2).norm(dim=-1) * scale
        spread = float(1.4826 * off.median())
        cut = 4.685 * max(spread, 0.05)
        robust = ((1.0 - (off / cut) ** 2).clamp(min=0.0)) ** 2
    return _matrix(theta, kind, centre.float(), scale), spread


def _align(ref: torch.Tensor, ref_ok: torch.Tensor, img: torch.Tensor, img_ok: torch.Tensor, start: torch.Tensor,
           kind: str, rounds: int = 10, most: int = 60000) -> torch.Tensor:
    """Refine ``start (2, 3)``, the motion from picture ``ref (H, W)`` (lightness) to ``img``, so that ``img`` at
    the moved point looks like ``ref`` (Lucas and Kanade on the whole picture, robust to what moves): the
    background seen in both, matched directly, so small errors of the flow do not add up over a clip."""
    h, w = ref.shape
    step = max(1, int(math.sqrt(h * w / most)))
    q = _grid(h, w)[:, ::step, ::step].reshape(2, -1)
    r0 = ref[::step, ::step].reshape(-1)
    ok0 = ref_ok[::step, ::step].reshape(-1)
    gx, gy = torch.zeros_like(img), torch.zeros_like(img)
    gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
    gy[1:-1] = (img[2:] - img[:-2]) / 2.0
    stack = torch.stack([img, gx, gy, img_ok.float()])
    centre = torch.tensor([w / 2.0, h / 2.0], dtype=torch.float64)
    scale = max(h, w) / 2.0
    matrix = start.clone()
    for _ in range(rounds):
        p = _apply(matrix, q)
        got = _look_up(stack, p[:, None, :])[:, 0]
        inside = (p[0] >= 1.0) & (p[0] <= w - 1.0) & (p[1] >= 1.0) & (p[1] <= h - 1.0)
        use = ok0 & inside & (got[3] > 0.99)
        if int(use.sum()) < 50:
            break
        jac = _jacobian(((p[:, use].T.double() - centre) / scale), kind)
        grad = got[1:3, use].T.double()
        rows = scale * torch.einsum("ni,nip->np", grad, jac)
        res = (got[0, use] - r0[use]).double()
        sigma = max(float(1.4826 * res.abs().median()), 1e-4)
        robust = (1.345 * sigma / res.abs().clamp(min=1e-12)).clamp(max=1.0)      # Huber
        lhs = rows.T @ (rows * robust[:, None]) + 1e-9 * torch.eye(rows.shape[1], dtype=torch.float64)
        delta = -torch.linalg.solve(lhs, rows.T @ (res * robust))
        matrix = _compose(_matrix(delta, kind, centre.float(), scale), matrix)
        if float(delta.abs().max()) * scale < 1e-3:
            break
    return matrix


def clip_flows(frames: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """How the picture moves between neighbouring frames of ``frames (T, 3, H, W)``: ``(T - 1, 2, H, W)``, entry
    ``t`` from frame t to frame t + 1 (:func:`helix_lm.paint.flow.optical_flow`). ``scale`` below 1 measures it on
    smaller frames, faster and coarser."""
    t, _, h, w = frames.shape
    if t < 2:
        return torch.zeros(0, 2, h, w)
    small = frames.float()
    if scale < 1.0:
        small = F.interpolate(small, scale_factor=scale, mode="area")
    out = torch.stack([optical_flow(small[i], small[i + 1]) for i in range(t - 1)])
    if scale < 1.0:
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        out[:, 0] *= w / small.shape[-1]
        out[:, 1] *= h / small.shape[-2]
    return out


def camera(flows: torch.Tensor, background_masks: torch.Tensor | None = None, frames: torch.Tensor | None = None,
           kind: str = "similarity", still: float = 0.25) -> torch.Tensor:
    """
    How the camera moves: ``(T, 2, 3)``, the matrix for frame t carrying a point of frame 0 to where that piece
    of background is in frame t (frame 0's is the identity). ``flows (T - 1, 2, H, W)`` are from
    :func:`clip_flows`, ``background_masks (T, H, W)`` say where the background is (None: everywhere; what moves
    on its own is outvoted anyway). ``kind`` is ``translation``, ``similarity`` (a pan, a zoom, a turn) or
    ``affine``.

    Each step from one frame to the next is fitted to the flow of the background by reweighted least squares
    (Tukey), and the steps are chained. With ``frames`` given, each frame is then matched directly to frame 0
    (or, once the camera has moved too far for that, to the frame before), so that small errors of the flow do
    not add up. A camera that moves less than ``still`` pixels anywhere in the picture over the whole clip is
    taken to be still: every matrix is the identity, so the frames are used as they are, not resampled.
    """
    t_count = len(flows) + 1
    h, w = flows.shape[-2:]
    size = max(h, w)
    margin = max(2, round(0.03 * size))                  # the flow is smoothed about this far round the character
    bg = None if background_masks is None else background_masks.bool()
    textures = None
    if frames is not None:
        lights = [_lightness_smooth(f) for f in frames.float()]
        textures = [_texture(lt) for lt in lights]
    cams = [_identity()]
    for t in range(t_count - 1):
        weight = torch.ones(h, w)
        if bg is not None:
            weight = _erode(bg[t], margin).float()
            if float(weight.sum()) < 0.05 * h * w:
                weight = bg[t].float()
        if textures is not None:
            tex = textures[t]
            top = float(torch.quantile(tex.flatten()[:: max(1, tex.numel() // 20000)], 0.9))
            weight = weight * (0.1 + (tex / max(top, 1e-12)).clamp(max=1.0))
        step, _ = _fit_motion(flows[t], weight, kind)
        cams.append(_compose(step, cams[-1]))
    if frames is not None:
        oks = [torch.ones(h, w, dtype=torch.bool) if bg is None else _erode(bg[t], 2) for t in range(t_count)]
        corners = torch.tensor([[0.0, w, 0.0, w], [0.0, 0.0, h, h]])
        for t in range(1, t_count):
            moved = _apply(cams[t], corners)
            shift = (moved - corners).abs().max(1).values
            if float(shift[0]) < 0.6 * w and float(shift[1]) < 0.6 * h:
                cams[t] = _align(lights[0], oks[0], lights[t], oks[t], cams[t], kind)
            else:
                step = _compose(cams[t], _inverse(cams[t - 1]))
                cams[t] = _compose(_align(lights[t - 1], oks[t - 1], lights[t], oks[t], step, kind), cams[t - 1])
    out = torch.stack(cams)
    corners = torch.tensor([[0.0, w, 0.0, w], [0.0, 0.0, h, h]])
    most = max(float((_apply(c, corners) - corners).abs().max()) for c in out)
    if most < still:
        out = _identity().expand(t_count, 2, 3).clone()
    return out


# ---------------------------------------------------------------------------------------------- depth --
_DEPTH_MODELS: dict = {}


def depth_estimator(model: str = DEPTH_MODEL, device=None):
    """A function from a frame ``(3, H, W)`` (sRGB, 0..1) to its relative depth ``(h, w)`` (larger is nearer, any
    scale), using a depth model from the Hugging Face Hub through transformers, loaded once and kept. None, with
    a warning that says why, when transformers or the model is not there."""
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError:
        warnings.warn("estimate_depth needs transformers (pip install 'helix-lm[paint]'); going on without depth",
                      stacklevel=3)
        return None
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    key = (model, str(device))
    if key not in _DEPTH_MODELS:
        try:
            processor = AutoImageProcessor.from_pretrained(model)
            net = AutoModelForDepthEstimation.from_pretrained(model).to(device).eval()
        except Exception as error:                           # no network, no such model, no disk space
            warnings.warn(f"could not load the depth model {model!r} ({type(error).__name__}: {error}); going on "
                          "without depth", stacklevel=3)
            return None
        _DEPTH_MODELS[key] = (processor, net)
    processor, net = _DEPTH_MODELS[key]

    def estimate(frame: torch.Tensor) -> torch.Tensor:
        picture = (frame.detach().float().cpu().clamp(0, 1).permute(1, 2, 0) * 255).round().byte().numpy()
        inputs = processor(images=picture, return_tensors="pt").to(device)
        with torch.inference_mode():
            return net(**inputs).predicted_depth[0].float().cpu()

    return estimate


def _fit_scale(x: torch.Tensor, y: torch.Tensor, rounds: int = 5) -> tuple[float, float]:
    """``s`` and ``b`` with ``s * x + b`` close to ``y`` (robust least squares)."""
    x, y = x.double(), y.double()
    weight = torch.ones_like(x)
    s, b = 1.0, 0.0
    for _ in range(rounds):
        sw = weight.sum()
        mx, my = (weight * x).sum() / sw, (weight * y).sum() / sw
        vx = (weight * (x - mx) ** 2).sum()
        if float(vx) <= 1e-12:
            return 1.0, float(my - mx)
        s = float((weight * (x - mx) * (y - my)).sum() / vx)
        b = float(my - s * mx)
        off = (s * x + b - y).abs()
        cut = 4.685 * max(float(1.4826 * off.median()), 1e-6)
        weight = ((1.0 - (off / cut) ** 2).clamp(min=0.0)) ** 2
    return s, b


def estimate_depth(frames: torch.Tensor, model: str = DEPTH_MODEL, device=None, *, estimator=None,
                   flows: torch.Tensor | None = None, steady: float = 0.5) -> torch.Tensor | None:
    """
    How near each pixel of ``frames (T, 3, H, W)`` (sRGB, 0..1) is: ``(T, H, W)`` in 0..1, larger is nearer, the
    same scale for the whole clip (0 and 1 are about the farthest and the nearest one per cent of it). It is
    relative depth (the model sees which things are in front of which, not how many metres away they are), and
    it is fine for telling a character from the set behind it and for the slope of a surface.

    The model is Depth Anything V2 Small (``depth-anything/Depth-Anything-V2-Small-hf``, Apache-2.0; the Base and
    Large ones are CC-BY-NC-4.0) through transformers, loaded once and kept (:func:`depth_estimator`).
    ``estimator`` is any function from a frame ``(3, H, W)`` to relative depth ``(h, w)`` (larger is nearer) to
    use instead. Returns None, with a warning, when transformers or the model is not available.

    A depth model looks at every frame alone, so its depths flicker and its scale wanders from frame to frame.
    Each frame is therefore matched in scale and offset to the frame before, carried over by the flow (``flows``
    from :func:`clip_flows`, measured on half-sized frames when not given), and where the two agree they are
    mixed (``steady`` is the share of the frame before): a still thing keeps its depth, and a thing that moves
    keeps its own outline.
    """
    frames = frames.float()
    t_count, _, h, w = frames.shape
    if estimator is None:
        estimator = depth_estimator(model, device)
        if estimator is None:
            return None
    raw = []
    for frame in frames:
        d = torch.as_tensor(estimator(frame)).float()
        d = d.reshape(1, 1, *d.shape[-2:])
        raw.append(F.interpolate(d, size=(h, w), mode="bicubic", align_corners=False)[0, 0])
    raw = torch.stack(raw)
    lo, hi = torch.quantile(raw[0].flatten()[:: max(1, h * w // 50000)], torch.tensor([0.01, 0.99]))
    out = [(raw[0] - lo) / (hi - lo).clamp(min=1e-8)]
    if t_count > 1 and steady > 0:
        flows = clip_flows(frames, 0.5) if flows is None else flows
    for t in range(1, t_count):
        if steady <= 0:
            s, b = _fit_scale(raw[t].flatten()[::7], out[0].flatten()[::7])
            out.append(s * raw[t] + b)
            continue
        before = warp(out[-1][None], _invert_flow(flows[t - 1]))[0]
        s, b = _fit_scale(raw[t].flatten()[::7], before.flatten()[::7])
        now = s * raw[t] + b
        agree = (now - before).abs() < 0.04
        out.append(torch.where(agree, (1.0 - steady) * now + steady * before, now))
    depth = torch.stack(out)
    lo, hi = torch.quantile(depth.flatten()[:: max(1, depth.numel() // 200000)], torch.tensor([0.01, 0.99]))
    return ((depth - lo) / (hi - lo).clamp(min=1e-8)).clamp(0.0, 1.0)


# -------------------------------------------------------------------------------------------- surfaces --
def _diffuse(x: torch.Tensor, across_x: torch.Tensor, across_y: torch.Tensor, rounds: int) -> torch.Tensor:
    """``x (H, W)`` smoothed by ``rounds`` steps of averaging with its four neighbours, each pair joined by a
    weight (``across_x (H, W - 1)``, ``across_y (H - 1, W)``; 0 where an edge parts them)."""
    for _ in range(rounds):
        total = x.clone()
        count = torch.ones_like(x)
        total[:, :-1] += across_x * x[:, 1:]
        count[:, :-1] += across_x
        total[:, 1:] += across_x * x[:, :-1]
        count[:, 1:] += across_x
        total[:-1] += across_y * x[1:]
        count[:-1] += across_y
        total[1:] += across_y * x[:-1]
        count[1:] += across_y
        x = total / count
    return x


def normals(depth: torch.Tensor, scale: float | None = None, smooth: int = 6) -> torch.Tensor:
    """
    Which way the surface faces at each pixel of ``depth (H, W)`` (larger is nearer): unit vectors ``(3, H, W)``,
    x to the right, y down, z out of the picture toward the viewer, so a surface seen face on is ``(0, 0, 1)``
    and the right side of a ball is ``(1, 0, 0)``-ish. ``scale`` is how many pixels toward the viewer one unit of
    depth is (by default a quarter of the picture's longer side, as relative depth has no true scale).

    The slopes are smoothed (``smooth`` rounds) but not across an outline: where the depth jumps (a step much
    larger than the slopes around it) neighbours are not mixed, and each pixel takes its slope from the side it
    belongs to, so the edge of a character is not bent toward the background behind it.
    """
    h, w = depth.shape
    scale = 0.25 * max(h, w) if scale is None else scale
    z = depth.float() * scale
    dx, dy = z[:, 1:] - z[:, :-1], z[1:] - z[:-1]
    slope = torch.zeros_like(z)
    slope[:, :-1] += dx.abs()
    slope[:, 1:] += dx.abs()
    slope[:-1] += dy.abs()
    slope[1:] += dy.abs()
    around = gaussian_blur(slope / 4.0, 3.0)
    jump_x = dx.abs() > 4.0 * torch.minimum(around[:, 1:], around[:, :-1]) + 0.5
    jump_y = dy.abs() > 4.0 * torch.minimum(around[1:], around[:-1]) + 0.5
    join_x, join_y = (~jump_x).float(), (~jump_y).float()
    z = _diffuse(z, join_x, join_y, max(1, smooth // 2))
    dx, dy = z[:, 1:] - z[:, :-1], z[1:] - z[:-1]
    # each pixel's slope from the side it belongs to: the one-sided difference that does not cross a jump (the
    # smaller one when neither does or both do)
    fx = torch.cat([dx, dx[:, -1:]], 1)
    bx = torch.cat([dx[:, :1], dx], 1)
    fjx = torch.cat([jump_x, torch.ones(h, 1, dtype=torch.bool)], 1)
    bjx = torch.cat([torch.ones(h, 1, dtype=torch.bool), jump_x], 1)
    gx = torch.where(fjx & ~bjx, bx, torch.where(bjx & ~fjx, fx, torch.where(fx.abs() < bx.abs(), fx, bx)))
    gx = torch.where(~fjx & ~bjx, 0.5 * (fx + bx), gx)
    fy = torch.cat([dy, dy[-1:]], 0)
    by = torch.cat([dy[:1], dy], 0)
    fjy = torch.cat([jump_y, torch.ones(1, w, dtype=torch.bool)], 0)
    bjy = torch.cat([torch.ones(1, w, dtype=torch.bool), jump_y], 0)
    gy = torch.where(fjy & ~bjy, by, torch.where(bjy & ~fjy, fy, torch.where(fy.abs() < by.abs(), fy, by)))
    gy = torch.where(~fjy & ~bjy, 0.5 * (fy + by), gy)
    gx = _diffuse(gx, join_x, join_y, smooth)
    gy = _diffuse(gy, join_x, join_y, smooth)
    n = torch.stack([-gx, -gy, torch.ones_like(gx)])
    return n / n.norm(dim=0, keepdim=True)


def _sphere_directions(count: int) -> torch.Tensor:
    """``count`` directions spread evenly over the sphere (a Fibonacci lattice), ``(count, 3)``."""
    i = torch.arange(count, dtype=torch.float64) + 0.5
    z = 1.0 - 2.0 * i / count
    r = (1.0 - z * z).clamp(min=0.0).sqrt()
    phi = math.pi * (3.0 - math.sqrt(5.0)) * i
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], -1)


def light_direction(image: torch.Tensor, normals: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Where the light comes from: a unit vector ``(3,)`` from the surface toward the light, in the axes of
    :func:`normals` (x right, y down, z toward the viewer). ``image (3, H, W)`` is sRGB, ``normals (3, H, W)``
    the surface, ``mask (H, W)`` the part to judge by (one material is best: the light is told from how the
    brightness of one colour changes with the way its surface faces).

    The brightness (linear luminance) is fitted as ``a + b * max(0, n . l)`` (Lambert's law with an ambient
    part): every direction of a fine lattice over the sphere is tried, ``a`` and ``b`` by least squares for each,
    then the best is refined on a finer lattice around it; the pixels that fit worst (outlines, highlights, other
    colours) are weighed down and the fit is done again. With no brightness that follows the surface, the light
    is taken to be from the front, ``(0, 0, 1)``.
    """
    lin = srgb_to_linear(image.float().clamp(0, 1))
    y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    ok = torch.isfinite(normals).all(0)
    if mask is not None:
        ok = ok & mask.bool()
    n = normals[:, ok].T.double()
    y = y[ok].double()
    if len(y) < 10:
        return torch.tensor([0.0, 0.0, 1.0])
    step = max(1, len(y) // 30000)
    n, y = n[::step], y[::step]
    weight = torch.ones_like(y)

    def best(directions: torch.Tensor) -> tuple[torch.Tensor, float]:
        s = (n @ directions.T).clamp(min=0.0)                       # (N, K)
        sw = weight.sum()
        ms = (weight[:, None] * s).sum(0) / sw
        my = (weight * y).sum() / sw
        vs = (weight[:, None] * (s - ms) ** 2).sum(0)
        cs = (weight[:, None] * (s - ms) * (y - my)[:, None]).sum(0)
        gain = torch.where((vs > 1e-9) & (cs > 0), cs * cs / vs.clamp(min=1e-12), torch.zeros_like(vs))
        k = int(gain.argmax())
        return directions[k], float(gain[k])

    coarse = _sphere_directions(4000)
    light, gain = best(coarse)
    if gain <= 0:
        return torch.tensor([0.0, 0.0, 1.0])
    for _ in range(3):
        local = _sphere_directions(20000)
        local = local[local[:, 2] > math.cos(math.radians(6.0))]  # a cap of 6 degrees round the pole
        # turn the cap's pole onto the current light
        z = light / light.norm()
        helper = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64) if abs(float(z[0])) < 0.9 else torch.tensor(
            [0.0, 1.0, 0.0], dtype=torch.float64)
        x = torch.linalg.cross(helper, z)
        x = x / x.norm()
        yy = torch.linalg.cross(z, x)
        candidates = torch.cat([local @ torch.stack([x, yy, z]), light[None]])
        light, gain = best(candidates)
        s = (n @ light).clamp(min=0.0)
        sw = weight.sum()
        ms, my = (weight * s).sum() / sw, (weight * y).sum() / sw
        b = (weight * (s - ms) * (y - my)).sum() / (weight * (s - ms) ** 2).sum().clamp(min=1e-12)
        a = my - b * ms
        off = (a + b * s - y).abs()
        cut = 4.685 * max(float(1.4826 * off.median()), 1e-6)
        weight = ((1.0 - (off / cut) ** 2).clamp(min=0.0)) ** 2
    return (light / light.norm()).float()


# ----------------------------------------------------------------------------------------------- layers --
def _otsu(values: torch.Tensor, bins: int = 64) -> float:
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-9:
        return hi
    hist = torch.histc(values.float(), bins, lo, hi).double()
    p = hist / hist.sum()
    centres = torch.linspace(lo, hi, bins, dtype=torch.float64)
    w0 = p.cumsum(0)
    m0 = (p * centres).cumsum(0)
    mt = m0[-1]
    between = (mt * w0 - m0) ** 2 / (w0 * (1 - w0)).clamp(min=1e-12)
    return float(centres[int(between[:-1].argmax())])


def _pop_out(depth: torch.Tensor, width: int) -> torch.Tensor:
    """How much nearer each pixel of ``depth (H, W)`` is than the background behind it: ``depth`` less its
    opening by a square ``width`` pixels wide (what is narrower than that and nearer than round it is taken
    away), measured on a small copy. The edges of the picture are repeated outward first, so a layer that runs
    out of the picture (the ground in front, a ceiling) counts as wide and stays background."""
    h, w = depth.shape
    f = max(1, round(max(h, w) / 128))
    small = F.avg_pool2d(depth[None, None], f, ceil_mode=True) if f > 1 else depth[None, None]
    r = max(1, round(width / f / 2))
    sh, sw = small.shape[-2:]
    k = 2 * r + 1
    padded = F.pad(small, (2 * r, 2 * r, 2 * r, 2 * r), mode="replicate")
    low = -F.max_pool2d(-F.max_pool2d(-padded, (1, k), 1, (0, r)), (k, 1), 1, (r, 0))
    low = F.max_pool2d(F.max_pool2d(low, (1, k), 1, (0, r)), (k, 1), 1, (r, 0))[..., 2 * r : 2 * r + sh,
                                                                                 2 * r : 2 * r + sw]
    low = F.avg_pool2d(low, 3, 1, 1, count_include_pad=False)
    back = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    return (depth - back).clamp(min=0.0)


def _tracks(labels: list, parts: list, moved: list, flows: torch.Tensor, enough: float) -> list:
    """Which parts of every frame to keep (a list of ``(parts[t],)`` bools). ``labels[t] (H, W)`` numbers the
    parts of frame t (-1 off them), ``moved[t]`` says how much each moves against the camera (-1: never keep).
    A part and a part of the next frame that, carried over by the flow, cover each other by at least 30 per
    cent of the smaller are one track; a track is kept when, over its frames, the upper quartile of how much it
    moves is ``enough``, so a character that stops for a moment is kept and a still thing whose flow was poor
    in one frame is not."""
    t_count = len(labels)
    offsets = [0]
    for n in parts:
        offsets.append(offsets[-1] + n)
    root = list(range(offsets[-1]))

    def find(i: int) -> int:
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    for t in range(t_count - 1):
        if parts[t] == 0 or parts[t + 1] == 0:
            continue
        h, w = labels[t].shape
        points = _grid(h, w) + flows[t]
        carried = _look_up(labels[t + 1].float()[None], points, "nearest")[0].round().long()
        both = (labels[t] >= 0) & (carried >= 0)
        pair = labels[t][both] * parts[t + 1] + carried[both]
        overlap = torch.bincount(pair, minlength=parts[t] * parts[t + 1]).reshape(parts[t], parts[t + 1])
        own = torch.bincount(labels[t][labels[t] >= 0], minlength=parts[t])
        then = torch.bincount(labels[t + 1][labels[t + 1] >= 0], minlength=parts[t + 1])
        linked = overlap.double() >= 0.3 * torch.minimum(own[:, None], then[None]).double()
        for i, j in linked.nonzero().tolist():
            if float(moved[t][i]) >= 0 and float(moved[t + 1][j]) >= 0:
                a, b = find(offsets[t] + i), find(offsets[t + 1] + j)
                root[max(a, b)] = min(a, b)
    by_track: dict = {}
    for t in range(t_count):
        for i in range(parts[t]):
            frames_of = by_track.setdefault(find(offsets[t] + i), {})
            frames_of[t] = max(frames_of.get(t, -1.0), float(moved[t][i]))
    kept = {r for r, f in by_track.items() if float(torch.tensor(list(f.values())).quantile(0.75)) >= enough}
    return [torch.tensor([find(offsets[t] + i) in kept and float(moved[t][i]) >= 0 for i in range(parts[t])],
                         dtype=torch.bool) for t in range(t_count)]


def _colour_bins(lab: torch.Tensor, bins: int = 16) -> torch.Tensor:
    """OKLab ``(..., 3, H, W)`` -> the number of its colour's bin, ``(..., H, W)``."""
    li = (lab[..., 0, :, :] * bins).long().clamp(0, bins - 1)
    ai = ((lab[..., 1, :, :] + 0.25) / 0.5 * bins).long().clamp(0, bins - 1)
    bi = ((lab[..., 2, :, :] + 0.25) / 0.5 * bins).long().clamp(0, bins - 1)
    return (li * bins + ai) * bins + bi


def _colour_grow(lab: torch.Tensor, masks: torch.Tensor, reach: int) -> torch.Tensor:
    """``masks (T, H, W)`` grown by the small pieces next to them that have the character's colours and not
    those of the set round it (a faint tail tip the depth missed): the colours of the character and of a ring
    round it are counted over the whole clip, and a piece of the ring is added when its colours are far more
    the character's. A piece that runs to the outside of the ring is a part of the set of a like colour (snow
    under a white paw) and is left out."""
    t_count, h, w = masks.shape
    bins = _colour_bins(lab)
    inner = torch.stack([_erode(m, 1) for m in masks])
    near = torch.stack([_dilate(m, reach) for m in masks])
    ring = near & ~torch.stack([_dilate(m, 2) for m in masks])
    size = 16 ** 3
    own = torch.bincount(bins[inner], minlength=size).double() + 0.5
    round_about = torch.bincount(bins[ring], minlength=size).double() + 0.5
    own, round_about = own / own.sum(), round_about / round_about.sum()
    share = own / (own + round_about)
    out = masks.clone()
    for t in range(t_count):
        if not bool(masks[t].any()):
            continue
        piece = near[t] & ~masks[t] & (share[bins[t]] >= 0.8)
        if not bool(piece.any()):
            continue
        label, _ = _components(piece)
        outer = near[t] & ~_dilate(masks[t], reach - 1)
        table = torch.zeros(h * w + 1, dtype=torch.bool)
        table[label[piece & _dilate(masks[t], 1)]] = True
        table[label[piece & outer]] = False
        out[t] = masks[t] | (table[label.clamp(min=0)] & piece)
    return out


def separate(frames: torch.Tensor, flows: torch.Tensor | None = None, depth: torch.Tensor | None = None, *,
             cameras: torch.Tensor | None = None, kind: str = "similarity", log=None) -> torch.Tensor:
    """
    The moving character: ``(T, H, W)``, True on the character (the foreground), False on the set (the
    background), for ``frames (T, 3, H, W)`` (sRGB). ``flows`` are from :func:`clip_flows` (measured when not
    given), ``depth (T, H, W)`` from :func:`estimate_depth` (larger is nearer; without it the character is found
    by how it differs from the set alone, which fails for a character that walks on the spot).

    1. The camera's own motion is fitted to the flow (what moves on its own is outvoted), and what moves against
       it is the motion of the character (smoothed by the flow, so only roughly where).
    2. The candidates: with depth, the parts nearer than the background right behind them (the depth less its
       opening, :func:`_pop_out`); without, the parts that differ from the clip's median background (built with
       the camera's motion), with their holes filled.
    3. A candidate is the character when enough of it moves against the camera, or when most of it lies on the
       character of the frame before or after (carried over by the flow), so a character that stops for a
       moment is kept. Small specks are dropped and holes filled.
    4. The outline is laid onto the edges of the frame (a guided filter with the colour, and the depth, as the
       guide), and every frame votes with its two neighbours, carried over by the flow (half its own, a quarter
       each of theirs), so the outline is steady and a part missed in one frame is taken back.
    """
    frames = frames.float()
    t_count, _, h, w = frames.shape
    size = max(h, w)
    area = h * w
    if t_count < 2:
        return torch.zeros(t_count, h, w, dtype=torch.bool)
    flows = clip_flows(frames) if flows is None else flows.float()
    grid = _grid(h, w)
    residual = []                                     # the flow less the camera's own motion, (T - 1, 2, H, W)
    for t in range(t_count - 1):
        step, _ = _fit_motion(flows[t], None, kind)
        residual.append(flows[t] - (_apply(step, grid) - grid))
    residual = torch.stack(residual)
    speed = (residual[:, 0] ** 2 + residual[:, 1] ** 2).sqrt()
    noise = float(speed.flatten()[:: max(1, speed.numel() // 200000)].median())
    enough = max(0.25, 1.5 * noise)
    if cameras is None:
        cameras = camera(flows, None, frames, kind)
    lab = image_to_oklab(frames)
    candidates = []
    if depth is None:
        plate = background_plate(frames, torch.zeros(t_count, h, w, dtype=torch.bool), cameras, grow=0)
        change = torch.stack([(lab[t] - image_to_oklab(place_background(plate, cameras[t], h, w))).norm(dim=0)
                              for t in range(t_count)])
        floor = float(change.flatten()[:: max(1, change.numel() // 200000)].median())
        found = change > max(0.06, 8.0 * floor)
        for t in range(t_count):
            c = _erode(_dilate(found[t], 2), 2)
            c = _fill_holes(c, 0.25 * area)
            candidates.append(_drop_small(c, 0.002 * area))
    else:
        depth = depth.float()
        for t in range(t_count):
            near = _pop_out(depth[t], round(0.3 * size)) > 0.035
            c = _erode(_dilate(near, 1), 1)
            c = _fill_holes(c, 0.02 * area)
            candidates.append(_drop_small(c, 0.002 * area))
    # which candidates are the character: each is followed through the clip (a track), and a track is the
    # character when it moves against the camera in at least a quarter of its frames; a part that spans the
    # picture from side to side is a layer of the set
    labels, parts, moved = [], [], []
    for t in range(t_count):
        label, _ = _components(candidates[t])
        on = label >= 0
        ids, which = torch.unique(label[on], return_inverse=True)
        count = torch.bincount(which, minlength=len(ids)).double().clamp(min=1)
        motion = torch.zeros(len(ids), dtype=torch.float64)
        for i in ([t] if t < t_count - 1 else []) + ([t - 1] if t > 0 else []):
            vx = torch.bincount(which, residual[i, 0][on].double(), len(ids)) / count
            vy = torch.bincount(which, residual[i, 1][on].double(), len(ids)) / count
            mag = torch.bincount(which, speed[i][on].double(), len(ids)) / count
            motion = torch.maximum(motion, torch.maximum((vx * vx + vy * vy).sqrt(), mag - noise))
        full = torch.full((area,), -1, dtype=torch.long)
        full[on.flatten()] = which
        full = full.reshape(h, w)
        for side_a, side_b in ((full[:, 0], full[:, -1]), (full[0], full[-1])):
            across = set(side_a[side_a >= 0].tolist()) & set(side_b[side_b >= 0].tolist())
            for p in across:
                motion[p] = -1.0
        labels.append(full)
        parts.append(len(ids))
        moved.append(motion)
    keep = _tracks(labels, parts, moved, flows, enough)
    backs = [_invert_flow(flows[t]) for t in range(t_count - 1)]

    def chosen(t: int) -> torch.Tensor:
        table = torch.cat([keep[t], torch.zeros(1, dtype=torch.bool)])
        return table[torch.where(labels[t] >= 0, labels[t], parts[t])]

    masks = torch.stack([chosen(t) for t in range(t_count)])
    # what still differs from the set right next to the character is the character too (a tail tip, a far leg
    # that the depth model saw as farther): the set there is known from the frames where it was not covered
    if bool(masks.any()):
        plate = background_plate(frames, masks, cameras)
        reach = max(3, round(0.04 * size))
        for t in range(t_count):
            if not bool(masks[t].any()):
                continue
            placed = place_background(plate, cameras[t], h, w)
            trusted = _place_count(plate, cameras[t], h, w) >= 3
            differs = (lab[t] - image_to_oklab(placed)).norm(dim=0) > 0.05
            grown = masks[t] | (differs & trusted & _dilate(masks[t], reach))
            label, _ = _components(grown)
            touch = label[masks[t]].unique()
            table = torch.zeros(area + 1, dtype=torch.bool)
            table[touch] = True
            masks[t] = table[label.clamp(min=0)] & (label >= 0)
            masks[t] = _fill_holes(masks[t], 0.002 * area)
        masks = _colour_grow(lab, masks, max(3, round(0.06 * size)))
    # lay the outline on the frame's edges, then vote with the neighbours
    radius = max(2, round(0.008 * size))
    soft = torch.zeros(t_count, h, w)
    for t in range(t_count):
        box = _bounds(masks[t], 2 * radius + 2)          # farther out the filter gives 0 anyway
        if box is None:
            continue
        guide = [lab[t, 0], 2.0 * lab[t, 1], 2.0 * lab[t, 2]] + ([depth[t]] if depth is not None else [])
        guide = torch.stack(guide)[:, box[0], box[1]]
        soft[t][box] = _guided(guide, masks[t][box].float(), radius, 1e-3).clamp(0.0, 1.0)
    out = []
    for t in range(t_count):
        m = soft[t] >= 0.5
        if 0 < t < t_count - 1:                      # a part both neighbours have and this frame missed
            vote = 0.5 * soft[t] + 0.25 * warp(soft[t - 1][None], backs[t - 1])[0] + 0.25 * warp(
                soft[t + 1][None], flows[t])[0]
            m = m | (vote >= 0.5)
        m = _fill_holes(m, 0.002 * area)
        out.append(_drop_small(m, 0.002 * area))
    if log:
        log(f"the character covers {float(torch.stack(out).float().mean()):.1%} of the frames")
    return torch.stack(out)


# --------------------------------------------------------------------------------------------- the set --
@dataclass
class Plate:
    """One picture of the whole set: ``image (3, H', W')`` sRGB, in the coordinates of frame 0 moved by
    ``origin`` (the point of frame 0 at its top left corner, x then y), ``count (H', W')`` how many frames saw
    the set at each pixel, and ``seen`` where at least one did (elsewhere it is filled in from round about)."""

    image: torch.Tensor
    origin: tuple
    count: torch.Tensor

    @property
    def seen(self) -> torch.Tensor:
        return self.count > 0


def background_plate(frames: torch.Tensor, fg_masks: torch.Tensor, cameras: torch.Tensor, grow: int | None = None,
                     most: float = 3.0) -> Plate:
    """
    The set, painted once: one clean picture of all the background seen in ``frames (T, 3, H, W)``, in the
    coordinates of frame 0 and larger than a frame when the camera pans, with the character (``fg_masks (T, H,
    W)``, grown by ``grow`` pixels so that its outline does not stay behind) taken out. ``cameras (T, 2, 3)`` are
    from :func:`camera`. ``most`` limits the size to that many frames each way.

    Every pixel of the plate is the median of what the frames show there where the set is visible (by lightness:
    the colour of the frame whose lightness is the median, so the colour is one that was really seen). The frames
    are lined up by the camera before the median, and moved with bicubic sampling (with a still camera they are
    not moved at all), so the plate is as sharp as the frames, not an average of misaligned ones. What the
    character covered in every frame is filled smoothly from round about.
    """
    frames = frames.float()
    t_count, _, h, w = frames.shape
    grow = max(2, round(0.006 * max(h, w))) if grow is None else grow
    corners = torch.tensor([[0.0, w, 0.0, w], [0.0, 0.0, h, h]])
    seen_corners = torch.cat([_apply(_inverse(c), corners) for c in cameras], 1)
    lo = seen_corners.min(1).values
    hi = seen_corners.max(1).values
    lo = torch.floor(lo + 1e-3)
    hi = torch.ceil(hi - 1e-3)
    lo = torch.maximum(lo, torch.tensor([-(most - 1) / 2 * w, -(most - 1) / 2 * h]).floor())
    hi = torch.minimum(hi, torch.tensor([w + (most - 1) / 2 * w, h + (most - 1) / 2 * h]).ceil())
    origin = (int(lo[0]), int(lo[1]))
    pw, ph = int(hi[0]) - origin[0], int(hi[1]) - origin[1]
    q = _grid(ph, pw) + torch.tensor([float(origin[0]), float(origin[1])])[:, None, None]
    covered = _dilate(fg_masks.bool(), grow)
    colours, lights, oks = [], [], []
    for t in range(t_count):
        p = _apply(cameras[t], q)
        if torch.allclose(cameras[t], _identity()) and origin == (0, 0) and (ph, pw) == (h, w):
            colour, cover = frames[t], covered[t]
            inside = torch.ones(h, w, dtype=torch.bool)
        else:
            colour = _look_up(frames[t], p, "bicubic").clamp(0.0, 1.0)
            cover = _look_up(covered[t].float()[None], p)[0] > 0.01
            inside = (p[0] >= 0.5 - 1e-3) & (p[0] <= w - 0.5 + 1e-3) & (p[1] >= 0.5 - 1e-3) & (p[1] <= h - 0.5 + 1e-3)
        ok = inside & ~cover
        colours.append(colour)
        lights.append(torch.where(ok, image_to_oklab(colour)[0], torch.full((ph, pw), float("nan"))))
        oks.append(ok)
    count = torch.stack(oks).sum(0)
    seen = count > 0
    pick = torch.stack(lights).nanmedian(0).indices
    image = torch.stack(colours).gather(0, pick[None, None].expand(1, 3, ph, pw))[0]
    image = _fill(torch.where(seen[None], image, torch.zeros_like(image)), seen)
    return Plate(image.clamp(0.0, 1.0), origin, count)


def _place_count(plate: Plate, camera_t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """How many frames saw the set at each pixel of frame t, ``(h, w)``."""
    x0, y0 = plate.origin
    p = _apply(_inverse(camera_t), _grid(h, w)) - torch.tensor([float(x0), float(y0)])[:, None, None]
    return _look_up(plate.count.float()[None], p, "nearest")[0].round().long()


def place_background(plate: Plate, camera_t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """The set as frame t sees it: ``plate`` (:func:`background_plate`) through ``camera_t (2, 3)`` (that frame's
    camera from :func:`camera`), ``(3, h, w)``. A still camera gives the plate's own pixels, not resampled."""
    x0, y0 = plate.origin
    ph, pw = plate.image.shape[-2:]
    if torch.equal(camera_t.float(), _identity()) and x0 <= 0 and y0 <= 0 and w - x0 <= pw and h - y0 <= ph:
        return plate.image[:, -y0 : h - y0, -x0 : w - x0].clone()
    p = _apply(_inverse(camera_t), _grid(h, w)) - torch.tensor([float(x0), float(y0)])[:, None, None]
    return _look_up(plate.image, p, "bicubic").clamp(0.0, 1.0)


# ------------------------------------------------------------------------------------------- all at once --
@dataclass
class Space:
    """What :func:`read_space` finds in a clip: the flow between neighbouring frames, the depth (None without a
    depth model), the character's masks, the cameras and the plate of the set."""

    flows: torch.Tensor
    depth: torch.Tensor | None
    character: torch.Tensor
    cameras: torch.Tensor
    plate: Plate

    def background(self, t: int) -> torch.Tensor:
        """The plate as frame ``t`` sees it."""
        h, w = self.character.shape[-2:]
        return place_background(self.plate, self.cameras[t], h, w)


def read_space(frames: torch.Tensor, depth: torch.Tensor | None | str = "model", *, estimator=None,
               kind: str = "similarity", log=None) -> Space:
    """All of it for ``frames (T, 3, H, W)``: the flow, the depth (``depth="model"`` asks the depth model, None
    goes without, a tensor is used as it is), the character, the cameras and the plate."""
    frames = frames.float()
    flows = clip_flows(frames)
    if isinstance(depth, str):
        depth = estimate_depth(frames, estimator=estimator, flows=flows)
    character = separate(frames, flows, depth, kind=kind, log=log)
    cameras = camera(flows, ~character, frames, kind)
    plate = background_plate(frames, character, cameras)
    return Space(flows, depth, character, cameras, plate)


__all__ = ["DEPTH_MODEL", "KINDS", "Plate", "Space", "background_plate", "camera", "clip_flows", "depth_estimator",
           "estimate_depth", "light_direction", "normals", "place_background", "read_space", "separate"]
