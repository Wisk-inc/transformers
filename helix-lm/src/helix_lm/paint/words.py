# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Words, where an eye that reads them makes them useful.

Painting a thing straight from words by moving strokes until the eye is pleased does not work (see
:mod:`helix_lm.paint.add`). What does work is asking the eye questions it was trained to answer:

* **where is it?** :meth:`helix_lm.paint.eye.Eye.find` blurs the picture away part by part and watches how well
  the words still fit. :func:`box_of` turns the answer into a box, :func:`cut_out` into the outline of the
  thing. So the subject of a painting, or the part of a painting to change, can be named: ``"the face"``,
  ``"the sky"``;
* **which picture shows it?** :func:`best_picture` picks, from pictures of one's own, the one the words fit
  best. :mod:`helix_lm.paint.library` finds such pictures in an open collection;
* **what does it feel like?** :func:`feeling_from_words` places a phrase among the eight feelings.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .emotion import EMOTIONS, Feeling, feeling_of
from .io import load_image
from .see import gaussian_blur

# what each feeling is, said in a few words the eye can compare a phrase with
SAID = {"awe": "awe, majesty, grandeur, the sublime, a vast and mighty sight",
        "serenity": "serenity, calm, peace, stillness, a quiet restful scene",
        "joy": "joy, happiness, delight, celebration, a bright cheerful day",
        "tenderness": "tenderness, gentle love, affection, care, a soft warm closeness",
        "melancholy": "melancholy, sadness, longing, sorrow, loneliness, grey and wistful",
        "fear": "fear, dread, terror, anxiety, a dark threatening nightmare",
        "anger": "anger, rage, fury, violence, a fierce burning storm",
        "mystery": "mystery, the unknown, secrets, the uncanny, dusk and shadows"}


def box_of(where: torch.Tensor, cut: float = 0.45, pad: float = 0.04) -> tuple[float, float, float, float]:
    """The box round the strongest patch of a map ``(h, w)`` (as :meth:`Eye.find` gives): ``(left, top, right,
    bottom)`` as shares of the picture. The patch is everything above ``cut`` of the map's peak that hangs
    together with the peak; ``pad`` is added all round."""
    size = 64
    m = F.interpolate(where[None, None].float(), size=(size, size), mode="bicubic", align_corners=False)[0, 0]
    above = m >= cut * float(m.max())
    patch = torch.zeros_like(above)
    patch.view(-1)[int(m.argmax())] = True
    while True:                                          # grow from the peak through what is above the cut
        grown = F.max_pool2d(patch[None, None].float(), 3, 1, 1)[0, 0].bool() & above
        if bool((grown == patch).all()):
            break
        patch = grown
    ys, xs = patch.any(1).nonzero().flatten(), patch.any(0).nonzero().flatten()
    left, right = float(xs[0]) / size - pad, (float(xs[-1]) + 1) / size + pad
    top, bottom = float(ys[0]) / size - pad, (float(ys[-1]) + 1) / size + pad
    return max(0.0, left), max(0.0, top), min(1.0, right), min(1.0, bottom)


def cut_out(eye, image: torch.Tensor, words: str, grid: int = 10) -> torch.Tensor:
    """How much of each pixel of ``image (3, H, W)`` belongs to the thing the words name: ``(H, W)`` in 0..1,
    soft at its border. The eye says roughly where the thing is; its colours against the colours at the
    picture's edge then say where it ends."""
    from .color import image_to_oklab

    h, w = image.shape[-2:]
    rough = F.interpolate(eye.find(image, words, grid)[None, None], size=(h, w), mode="bicubic",
                          align_corners=False)[0, 0].clamp(0.0, 1.0)
    lab = image_to_oklab(image).reshape(3, -1).T
    sure, edge = rough.flatten() > 0.6, torch.ones(h, w, dtype=torch.bool)
    edge[h // 10 : h - h // 10, w // 10 : w - w // 10] = False
    outside = edge.flatten() & (rough.flatten() < 0.3)
    if int(sure.sum()) < 20 or int(outside.sum()) < 20:
        return rough

    def centres(points: torch.Tensor, k: int = 6) -> torch.Tensor:
        c = points[torch.linspace(0, points.shape[0] - 1, k).long()].clone()
        for _ in range(8):
            owner = torch.cdist(points, c).argmin(1)
            for j in range(k):
                if bool((owner == j).any()):
                    c[j] = points[owner == j].mean(0)
        return c

    step = max(1, int(sure.sum()) // 4000)
    thing, around = centres(lab[sure][::step]), centres(lab[outside][:: max(1, int(outside.sum()) // 4000)])
    near_thing = torch.cdist(lab, thing).min(1).values
    near_around = torch.cdist(lab, around).min(1).values
    colour = torch.sigmoid((near_around - near_thing) / 0.02).reshape(h, w)      # 1 where it looks like the thing
    alpha = (colour * (0.25 + 0.75 * rough) + 0.6 * (rough - 0.5).clamp(min=0.0)).clamp(0.0, 1.0)
    alpha = alpha * (rough > 0.12)                                               # nothing far from where it is
    return gaussian_blur(alpha, max(1.0, 0.012 * max(h, w))).clamp(0.0, 1.0)


def best_picture(eye, words: str, pictures: list) -> tuple[int, list]:
    """Which of ``pictures`` (paths or ``(3, H, W)`` tensors) the words fit best: its number, and every
    picture's score."""
    said = eye.text([words])[0]
    scores = []
    with torch.no_grad():
        for picture in pictures:
            image = picture if isinstance(picture, torch.Tensor) else load_image(picture, max_side=512)
            scores.append(float(eye.score(image[None], said)))
    return max(range(len(scores)), key=lambda i: scores[i]), scores


def feeling_from_words(words: str, eye, sharp: float = 30.0) -> tuple[Feeling, dict]:
    """A phrase (``"quiet sadness at dusk"``) placed among the eight feelings by an eye that reads text: the
    feeling to paint with (a blend of the nearest ones) and how much of each went into it."""
    with torch.no_grad():
        vectors = eye.text([words] + [SAID[name] for name in EMOTIONS]).float().cpu()
    near = torch.softmax(sharp * (vectors[1:] @ vectors[0]), 0)
    shares = {name: float(v) for name, v in zip(EMOTIONS, near, strict=True)}
    strong = {name: share for name, share in shares.items() if share >= 0.15} or {max(shares, key=shares.get): 1.0}
    return feeling_of(strong), shares


__all__ = ["SAID", "best_picture", "box_of", "cut_out", "feeling_from_words"]
