# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The self-check: the drawing laid over its frame and looked at the way an animator checks a drawing against
the model, and redrawn where it is wrong.

:func:`score` draws the lines on a board the size of the frame and compares: how different the colours look
(OKLab distance, over the whole picture and for each of the palette's colours), how much of the board was left
bare where the frame has colour, and how well the ink follows the frame's borders (the chamfer distance: how
far, on average, each drawn line is from the nearest border of the frame, and each border from the nearest
drawn line). It also gives a mask of the bad places: where the colour is clearly wrong over more than a
line's width, or the board is bare where it should not be.

:func:`fix` redraws those places, each as its own part (:func:`helix_lm.paint.detail.draw_details`): what lies
inside it is taken out, new lines drawn for it, cut to it, and a change is kept only where it makes the
drawing better.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .color import delta_e, image_to_oklab, srgb_to_oklab
from .detail import Part, components, draw_details, merge
from .draw import SURFACES, Board, find_lines, markers, thin
from .flow import nearest
from .see import gaussian_blur

WRONG = 0.12           # an OKLab distance at which a colour is clearly wrong (a JND is about 0.02)
FAR = 10.0             # the chamfer distance counts a line at most this far (pixels) from where it should be


def _open(mask: torch.Tensor, size: int = 3) -> torch.Tensor:
    m = mask[None, None].float()
    m = -F.max_pool2d(-m, size, 1, size // 2)
    return F.max_pool2d(m, size, 1, size // 2)[0, 0] > 0.5


def _labels(image: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    lab = image_to_oklab(image).reshape(3, -1).T
    pigments = srgb_to_oklab(palette.float())
    return torch.cdist(lab, pigments).argmin(1).reshape(image.shape[-2:])


def score(lines: list, image: torch.Tensor, palette: torch.Tensor | None = None, *, surface: str = "whiteboard",
          waver: float = 0.0, seed: int = 0) -> dict:
    """How well ``lines`` (board widths) draw ``image (3, H, W)`` (sRGB), drawn on a ``surface`` of the image's
    size. Returns a dict:

    - ``error``: the mean OKLab distance between the drawing and the frame; ``error_map`` the same per pixel
      ``(H, W)``; ``regions`` the mean for the pixels of each palette colour ``{marker: error}``;
    - ``bare``: the share of the frame's coloured pixels left as bare board, ``bare_pixels`` how many;
    - ``ink_to_edge``, ``edge_to_ink``, ``chamfer``: how far (pixels) the ink lies from the frame's borders, the
      borders from the ink, and their mean (each counted up to :data:`FAR`);
    - ``bad``: ``(H, W)`` bool, the places that are clearly wrong;
    - ``loss``: one number to make smaller (the error, plus the bare share and the chamfer distance weighted
      in); ``picture``: the drawing.
    """
    image = image.float().clamp(0.0, 1.0)
    h, w = image.shape[-2:]
    palette = markers([image]) if palette is None else palette
    board = Board(w, h, surface, waver=waver, seed=seed)
    for line in lines:
        board.draw(line)
    picture = board.picture()
    error = delta_e(picture, image)
    lab_pic, lab_img = image_to_oklab(picture), image_to_oklab(image)
    bare_lab = srgb_to_oklab(torch.tensor(SURFACES.get(surface, SURFACES["whiteboard"])))[:, None, None]
    coloured = (lab_img - bare_lab).norm(dim=0) > 0.06
    bare = ((lab_pic - bare_lab).norm(dim=0) < 0.02) & coloured
    ink = board.ink_cover > 0.5
    edges, _ = find_lines(image, 0.6)
    if bool(ink.any()) and bool(edges.any()):
        drawn = thin(ink)
        to_edge, to_ink = nearest(edges)[1].clamp(max=FAR), nearest(drawn)[1].clamp(max=FAR)
        ink_to_edge, edge_to_ink = float(to_edge[drawn].mean()), float(to_ink[edges].mean())
    else:
        ink_to_edge = edge_to_ink = FAR if bool(ink.any()) != bool(edges.any()) else 0.0
    labels = _labels(image, palette)
    regions = {k: float(error[labels == k].mean()) for k in range(len(palette)) if bool((labels == k).any())}
    # wrong over more than a line's width: a band thinner than five pixels is a line drawn a little thinner
    # or a little off, which drawing the place again with the same pens would not change
    wrong = gaussian_blur(error, 1.0) > WRONG
    bad = _open(wrong, 5) | _open(bare, 3)
    if bool(bad.any()):                               # specks are not worth redrawing
        comp, n = components(bad)
        area = torch.bincount(comp[comp >= 0], minlength=n)
        bad = (comp >= 0) & (area[comp.clamp(min=0)] >= 6)
    chamfer = 0.5 * (ink_to_edge + edge_to_ink)
    share = float(bare.sum()) / max(1, int(coloured.sum()))
    return {"error": float(error.mean()), "error_map": error, "regions": regions, "bare": share,
            "bare_pixels": int(bare.sum()), "ink_to_edge": ink_to_edge, "edge_to_ink": edge_to_ink,
            "chamfer": chamfer, "bad": bad, "loss": float(error.mean()) + 0.2 * share + 0.004 * chamfer,
            "picture": picture}


def bad_boxes(bad: torch.Tensor, most: int = 8, grow: float = 0.3) -> list:
    """The bad places of a :func:`score` as boxes (pixels), each grown by ``grow`` of its size (at least three
    pixels), boxes that then overlap joined, the ``most`` largest."""
    h, w = bad.shape
    comp, n = components(bad)
    boxes = []
    for k in range(n):
        ys, xs = torch.nonzero(comp == k, as_tuple=True)
        x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1
        g = max(3.0, grow * max(x1 - x0, y1 - y0))
        boxes.append([max(0.0, x0 - g), max(0.0, y0 - g), min(float(w), x1 + g), min(float(h), y1 + g)])
    joined = True
    while joined:
        joined = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    joined = True
                    break
            if joined:
                break
    boxes.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    return [tuple(b) for b in boxes[:most]]


def fix(lines: list, image: torch.Tensor, drawer=None, palette: torch.Tensor | None = None, rounds: int = 2, *,
        scale_up: int = 2, most: int = 8) -> list:
    """``lines`` with their bad places (:func:`score`) drawn again, up to ``rounds`` times. Each place is a
    part (:func:`bad_boxes`): what lies inside it is taken out (a line crossing its edge is cut there) and
    ``drawer(crop, palette)`` draws it anew, enlarged ``scale_up`` times and cut to it
    (:func:`helix_lm.paint.detail.draw_details` and :func:`helix_lm.paint.detail.merge`). A place's new
    drawing is kept only when it matches the frame better there, and a round only when the whole drawing
    scores better (a lower ``loss``)."""
    image = image.float().clamp(0.0, 1.0)
    h, w = image.shape[-2:]
    palette = markers([image]) if palette is None else palette
    best = score(lines, image, palette)
    for _ in range(rounds):
        parts = [Part(box, "detail", 1.0, (h, w)) for box in bad_boxes(best["bad"], most)]
        if not parts:
            break
        drawn = [draw_details(image, [part], palette, drawer, scale_up) for part in parts]
        tried = score(merge(lines, [x for d in drawn for x in d], parts), image, palette)
        keep = []
        for part, new in zip(parts, drawn, strict=True):  # each place on its own: better there, or as it was
            x0, y0, x1, y1 = (int(v) for v in part.box)
            if float(tried["error_map"][y0:y1, x0:x1].mean()) < float(best["error_map"][y0:y1, x0:x1].mean()) - 1e-4:
                keep.append((part, new))
        if not keep:
            break
        candidate = merge(lines, [x for _, new in keep for x in new], [part for part, _ in keep])
        result = score(candidate, image, palette)
        if result["loss"] >= best["loss"]:
            break
        lines, best = candidate, result
    return lines


__all__ = ["FAR", "WRONG", "bad_boxes", "fix", "score"]
