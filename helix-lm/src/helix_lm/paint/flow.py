# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
How a picture moves from one frame to the next, and where the nearest line is.

:func:`optical_flow` is Lucas and Kanade's method (1981) made dense and run coarse to fine: at each size of the
picture, each pixel's small neighbourhood is matched in the next frame by the brightness gradients, the second
frame is warped by what was found, and the match is refined. It is what lets an animator's drawing be carried
from one frame to the next, the way a sheet is laid on the lightbox over the last one.

:func:`nearest` gives, for every pixel, the nearest pixel of a mask (jump flooding, Rong and Tan 2006), so a
line carried over from the frame before can be checked against, and pulled onto, the lines of the new frame.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .color import image_to_oklab
from .see import gaussian_blur


def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """``image (C, H, W)`` sampled at each pixel plus ``flow (2, H, W)`` (pixels, x then y)."""
    c, h, w = image.shape
    gy, gx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    grid = torch.stack([(gx + flow[0] + 0.5) / w * 2 - 1, (gy + flow[1] + 0.5) / h * 2 - 1], -1)[None]
    return F.grid_sample(image[None].float(), grid, mode="bilinear", padding_mode="border", align_corners=False)[0]


def _gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gx, gy = torch.zeros_like(x), torch.zeros_like(x)
    gx[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2.0
    gy[1:-1] = (x[2:] - x[:-2]) / 2.0
    return gx, gy


def optical_flow(first: torch.Tensor, second: torch.Tensor, iterations: int = 5, window: float = 2.5,
                 smallest: int = 24) -> torch.Tensor:
    """Where each pixel of ``first`` went in ``second`` (both sRGB ``(3, H, W)``): ``(2, H, W)`` in pixels,
    x then y, so that ``second`` at ``x + flow(x)`` looks like ``first`` at ``x``."""
    a0, b0 = image_to_oklab(first)[0], image_to_oklab(second)[0]
    pyramid = [(a0, b0)]
    while min(pyramid[-1][0].shape) >= 2 * smallest:
        a, b = pyramid[-1]
        pyramid.append((F.avg_pool2d(a[None, None], 2)[0, 0], F.avg_pool2d(b[None, None], 2)[0, 0]))
    flow = torch.zeros(2, *pyramid[-1][0].shape)
    for level, (a, b) in enumerate(reversed(pyramid)):
        if level > 0:
            flow = 2.0 * F.interpolate(flow[None], size=a.shape, mode="bilinear", align_corners=False)[0]
        a = gaussian_blur(a, 0.8)
        b = gaussian_blur(b, 0.8)
        for _ in range(iterations):
            bw = warp(b[None], flow)[0]
            ix, iy = _gradients(0.5 * (a + bw))
            it = bw - a
            sxx, syy, sxy = (gaussian_blur(v, window) for v in (ix * ix, iy * iy, ix * iy))
            sxt, syt = gaussian_blur(ix * it, window), gaussian_blur(iy * it, window)
            reg = 1e-5
            det = (sxx + reg) * (syy + reg) - sxy * sxy
            du = -((syy + reg) * sxt - sxy * syt) / det
            dv = -((sxx + reg) * syt - sxy * sxt) / det
            step = torch.stack([du, dv]).clamp(-2.0, 2.0)
            # trust the match only where the picture has detail in two directions (a corner, a curve); in a
            # plain part, motion comes from the detail round it, and with none near, it is taken to be still
            trust = 0.5 * (sxx + syy) - (0.25 * (sxx - syy) ** 2 + sxy ** 2).sqrt()
            trust = trust.clamp(min=0.0)
            flow = flow + step * (trust > 1e-7)
            # at the coarsest size a plain part with no detail near is taken to be still; finer sizes only
            # refine (a pull toward stillness there would shrink every motion a little at every step)
            floor = 1e-9
            if level == 0:
                floor += 0.05 * float(torch.quantile(trust.flatten()[:: max(1, trust.numel() // 20000)], 0.9))
            spread = max(1.5, 0.02 * max(a.shape))
            weight = gaussian_blur(trust, spread) + floor
            flow = torch.stack([gaussian_blur(f * trust, spread) / weight for f in flow])
    return flow


def sample(field: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """``field (C, H, W)`` at ``points (K, 2)`` (pixels, x then y), bilinear: ``(K, C)``."""
    h, w = field.shape[-2:]
    grid = torch.stack([points[:, 0] / w * 2 - 1, points[:, 1] / h * 2 - 1], -1)[None, None]
    return F.grid_sample(field[None].float(), grid, mode="bilinear", padding_mode="border",
                         align_corners=False)[0, :, 0].T


def nearest(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For every pixel, the nearest pixel where ``mask (H, W)`` is set: its centre ``(2, H, W)`` (x then y) and
    the distance to it ``(H, W)`` (a large number when the mask is empty)."""
    h, w = mask.shape
    big = float(h + w) * 4.0
    gy, gx = torch.meshgrid(torch.arange(h, dtype=torch.float32) + 0.5, torch.arange(w, dtype=torch.float32) + 0.5,
                            indexing="ij")
    seed = torch.stack([torch.where(mask, gx, torch.full_like(gx, -big)), torch.where(mask, gy, torch.full_like(gy,
                                                                                                         -big))])
    step = 1
    while step < max(h, w):
        step *= 2
    while step >= 1:
        best = seed
        dist = ((seed[0] - gx) ** 2 + (seed[1] - gy) ** 2)
        for dy in (-step, 0, step):
            for dx in (-step, 0, step):
                if dy == 0 and dx == 0:
                    continue
                shifted = torch.full_like(seed, -big)
                ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
                xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
                if ys.stop <= ys.start or xs.stop <= xs.start:
                    continue
                shifted[:, yd, xd] = seed[:, ys, xs]
                d = (shifted[0] - gx) ** 2 + (shifted[1] - gy) ** 2
                better = d < dist
                best = torch.where(better[None], shifted, best)
                dist = torch.where(better, d, dist)
        seed = best
        step //= 2
    distance = ((seed[0] - gx) ** 2 + (seed[1] - gy) ** 2).sqrt()
    return seed, torch.where(seed[0] < -big / 2, torch.full_like(distance, big), distance)


__all__ = ["nearest", "optical_flow", "sample", "warp"]
