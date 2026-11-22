# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Looking at a picture the way a painter does before the first stroke: where the edges run (strokes follow them),
what draws the eye (that part gets the care), and the few big masses of light and dark (the block-in).

Nothing here is learned. With a pretrained vision model (the eye, :mod:`helix_lm.paint.eye`) its opinion of what
stands out is mixed into the saliency.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .color import image_to_oklab


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """``(..., H, W)`` blurred with a Gaussian of ``sigma`` pixels (edges repeated)."""
    if sigma <= 0.05:
        return x
    radius = max(1, int(3.0 * sigma + 0.5))
    k = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    lead = x.shape[:-2]
    y = x.reshape(-1, 1, *x.shape[-2:])
    y = F.conv2d(F.pad(y, (radius, radius, 0, 0), mode="replicate"), k[None, None, None, :])
    y = F.conv2d(F.pad(y, (0, 0, radius, radius), mode="replicate"), k[None, None, :, None])
    return y.reshape(*lead, *x.shape[-2:])


def lightness(image: torch.Tensor) -> torch.Tensor:
    """sRGB ``(3, H, W)`` -> OKLab lightness ``(H, W)``."""
    return image_to_oklab(image)[0]


def orientation_field(image: torch.Tensor, sigma: float = 1.2, window: float = 2.5) -> dict:
    """Which way a stroke should go at each pixel: along the edge, not across it. Returns ``cos2`` and ``sin2``
    (the direction with its angle doubled, so that opposite directions agree), ``coherence`` (0: no direction,
    1: one clear direction) and ``energy`` (how strong the edges are), each ``(H, W)``."""
    light = gaussian_blur(lightness(image), sigma)
    p = F.pad(light[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    gx = (p[1:-1, 2:] - p[1:-1, :-2]) / 2.0
    gy = (p[2:, 1:-1] - p[:-2, 1:-1]) / 2.0
    jxx, jyy, jxy = gaussian_blur(gx * gx, window), gaussian_blur(gy * gy, window), gaussian_blur(gx * gy, window)
    across = 0.5 * torch.atan2(2.0 * jxy, jxx - jyy)
    along = across + math.pi / 2.0
    spread = torch.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
    return {"cos2": torch.cos(2.0 * along), "sin2": torch.sin(2.0 * along),
            "coherence": spread / (jxx + jyy + 1e-8), "energy": torch.sqrt(jxx + jyy + 1e-12)}


def saliency(image: torch.Tensor, eye_map: torch.Tensor | None = None) -> torch.Tensor:
    """What draws the eye, ``(H, W)`` in 0..1: places that differ from their surroundings in lightness or
    colour at three sizes, strong colour, and a mild preference for the middle. ``eye_map`` (any size, any
    range) is a pretrained model's opinion of which part the picture is about
    (:meth:`helix_lm.paint.eye.Eye.saliency`)."""
    lab = image_to_oklab(image)
    size = max(image.shape[-2:]) / 256.0
    out = torch.zeros_like(lab[0])
    for sigma in (1.5, 3.0, 6.0):
        centre, around = gaussian_blur(lab, sigma * size), gaussian_blur(lab, 4.0 * sigma * size)
        diff = (centre - around).abs()
        out = out + diff[0] + 2.0 * diff[1] + 2.0 * diff[2]
    out = out + 0.35 * torch.sqrt(lab[1] ** 2 + lab[2] ** 2)
    out = gaussian_blur(out, 0.03 * max(image.shape[-2:]))
    h, w = out.shape
    ys = torch.linspace(-1, 1, h, device=out.device)[:, None]
    xs = torch.linspace(-1, 1, w, device=out.device)[None, :]
    out = out * (0.6 + 0.4 * torch.exp(-(ys ** 2 + xs ** 2) / (2 * 0.6 ** 2)))
    lo, hi = torch.quantile(out.flatten()[:: max(1, out.numel() // 20000)], torch.tensor([0.05, 0.98]))
    out = ((out - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)
    if eye_map is not None:
        # the eye says which part the picture is about, broadly; what stands out by colour and contrast says
        # where to look inside it. So the eye's part is in focus as a whole, and details count by how much
        # the eye cares for the place they are in
        e = F.interpolate(eye_map.float()[None, None], size=(h, w), mode="bicubic", align_corners=False)[0, 0]
        e = ((e - e.min()) / (e.max() - e.min() + 1e-8)).clamp(0.0, 1.0).to(out.device)
        out = 0.4 * e + 0.6 * out * (0.35 + 0.65 * e)
    return out / out.max().clamp(min=1e-8)


def focal_point(sal: torch.Tensor) -> tuple[float, float]:
    """The centre of what stands out most: ``(y, x)`` in 0..1, the weighted middle of the top eighth."""
    cut = torch.quantile(sal.flatten()[:: max(1, sal.numel() // 20000)], 0.875)
    weight = torch.relu(sal - cut) + 1e-8
    h, w = sal.shape
    ys = (torch.arange(h, device=sal.device) + 0.5) / h
    xs = (torch.arange(w, device=sal.device) + 0.5) / w
    total = weight.sum()
    return float((weight.sum(1) * ys).sum() / total), float((weight.sum(0) * xs).sum() / total)


def value_masses(light: torch.Tensor, masses: int = 3, softness: float = 0.03) -> tuple[torch.Tensor, torch.Tensor]:
    """The picture squinted at: ``masses`` levels of lightness (by k-means from evenly spaced starts) and the
    lightness of the mass each pixel belongs to ``(H, W)``, with soft borders between masses."""
    flat = light.flatten()[:: max(1, light.numel() // 40000)]
    centres = torch.quantile(flat, torch.linspace(0.1, 0.9, masses, device=light.device))
    for _ in range(12):
        owner = (flat[:, None] - centres[None]).abs().argmin(1)
        for k in range(masses):
            if bool((owner == k).any()):
                centres[k] = flat[owner == k].mean()
    centres = centres.sort().values
    weights = torch.softmax(-((light[..., None] - centres) ** 2) / (2 * softness ** 2), -1)
    return centres, (weights * centres).sum(-1)


__all__ = ["focal_point", "gaussian_blur", "lightness", "orientation_field", "saliency", "value_masses"]
