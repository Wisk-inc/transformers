# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Colour for a painter: sRGB and linear light, OKLab for judging how different two colours look, pigment mixing, and
the palette of a picture.

Paint is mixed in linear light with a Kubelka-Munk rule (each pigment's absorption over scattering adds up), so
blue and yellow make a green, not the grey that averaging their RGB values gives. Differences are measured in
OKLab (Ottosson, 2020), where equal distances look about equally different.
"""

from __future__ import annotations

import math

import torch


def srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.04045, c / 12.92, ((c.clamp(min=0.04045) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    c = c.clamp(0.0, 1.0)
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c.clamp(min=0.0031308) ** (1 / 2.4) - 0.055)


_M1 = ((0.4122214708, 0.5363325363, 0.0514459929),
       (0.2119034982, 0.6806995451, 0.1073969566),
       (0.0883024619, 0.2817188376, 0.6299787005))
_M2 = ((0.2104542553, 0.7936177850, -0.0040720468),
       (1.9779984951, -2.4285922050, 0.4505937099),
       (0.0259040371, 0.7827717662, -0.8086757660))
_M2I = ((1.0, 0.3963377774, 0.2158037573),
        (1.0, -0.1055613458, -0.0638541728),
        (1.0, -0.0894841775, -1.2914855480))
_M1I = ((4.0767416621, -3.3077115913, 0.2309699292),
        (-1.2684380046, 2.6097574011, -0.3413193965),
        (-0.0041960863, -0.7034186147, 1.7076147010))


def _mat(rows, like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(rows, dtype=like.dtype, device=like.device)


def linear_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    """``(..., 3)`` linear RGB -> OKLab (L in 0..1, a and b about -0.3..0.3)."""
    lms = rgb @ _mat(_M1, rgb).T
    lms = torch.sign(lms) * (lms.abs() + 1e-10).pow(1.0 / 3.0)
    return lms @ _mat(_M2, rgb).T


def oklab_to_linear(lab: torch.Tensor) -> torch.Tensor:
    lms = lab @ _mat(_M2I, lab).T
    return (lms ** 3) @ _mat(_M1I, lab).T


def srgb_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    return linear_to_oklab(srgb_to_linear(rgb))


def oklab_to_srgb(lab: torch.Tensor) -> torch.Tensor:
    return linear_to_srgb(oklab_to_linear(lab))


def image_to_oklab(image: torch.Tensor, linear: bool = False) -> torch.Tensor:
    """``(..., 3, H, W)`` sRGB (or linear) image -> ``(..., 3, H, W)`` OKLab."""
    x = image.movedim(-3, -1)
    return (linear_to_oklab(x) if linear else srgb_to_oklab(x)).movedim(-1, -3)


def oklab_to_image(lab: torch.Tensor) -> torch.Tensor:
    return oklab_to_srgb(lab.movedim(-3, -1)).movedim(-1, -3)


def mix_pigments(colors: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Kubelka-Munk mixing. ``colors (..., P, 3)`` linear RGB, ``weights (..., P)`` summing to 1 -> ``(..., 3)``.
    A pigment alone gives itself back; two pigments mix the way paint does (darker and more saturated than the
    average of their light)."""
    r = colors.clamp(0.004, 0.996)
    ks = (1.0 - r) ** 2 / (2.0 * r)
    mix = (weights[..., None] * ks).sum(-2)
    return 1.0 + mix - torch.sqrt(mix * mix + 2.0 * mix)


def snap_to_palette(colors: torch.Tensor, palette: torch.Tensor, strength: float = 1.0,
                    temperature: float = 0.0015) -> torch.Tensor:
    """Move ``colors (..., 3)`` (linear RGB) toward mixes of the palette's pigments ``(P, 3)``: each colour becomes
    a mix weighted toward the pigments nearest to it in OKLab. ``strength`` 0 keeps the colour, 1 uses the mix."""
    if strength <= 0:
        return colors
    d2 = (linear_to_oklab(colors)[..., None, :] - linear_to_oklab(palette)).pow(2).sum(-1)
    mixed = mix_pigments(palette, torch.softmax(-d2 / temperature, -1))
    return colors + strength * (mixed - colors)


def _root(m: torch.Tensor, power: float) -> torch.Tensor:
    """A symmetric positive matrix raised to ``power``."""
    values, vectors = torch.linalg.eigh(m.double())
    return ((vectors * values.clamp(min=1e-10) ** power) @ vectors.T).float()


def colour_statistics(lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The mean ``(3,)`` and the covariance ``(3, 3)`` of the colours of an OKLab picture ``(3, H, W)``."""
    flat = lab.reshape(3, -1)
    flat = flat[:, :: max(1, flat.shape[1] // 60000)]
    mean = flat.mean(1)
    centred = flat - mean[:, None]
    return mean, centred @ centred.T / max(1, centred.shape[1] - 1)


def match_colours(lab: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """Move the colours of an OKLab picture ``(3, H, W)`` so that as a whole they have the ``mean`` and the
    ``cov`` of another picture's: the straight-line map that carries one cloud of colours onto the other while
    moving each colour as little as possible (Pitie and Kokaram, 2007). Unlike matching each channel alone,
    this turns the picture's own colour directions into the other's: a blue flag in a photo painted after a
    brown and olive painting becomes one of that painting's darks, not a blue with less saturation."""
    own_mean, own_cov = colour_statistics(lab)
    eye = 1e-6 * torch.eye(3)
    a, b = own_cov + eye, cov.float() + eye
    half, back = _root(a, 0.5), _root(a, -0.5)
    move = back @ _root(half @ b @ half, 0.5) @ back
    flat = lab.reshape(3, -1)
    return (move @ (flat - own_mean[:, None]) + mean.float()[:, None]).reshape(lab.shape)


def extract_palette(image: torch.Tensor, colors: int = 12, iterations: int = 24, seed: int = 0,
                    samples: int = 20000) -> tuple[torch.Tensor, torch.Tensor]:
    """The palette of an sRGB image ``(3, H, W)``: ``(colors (k, 3) sRGB sorted dark to light, shares (k,))`` by
    k-means in OKLab, started with k-means++ from a seeded generator, so the same picture gives the same palette."""
    g = torch.Generator().manual_seed(seed)
    lab = image_to_oklab(image.float().cpu()).reshape(3, -1).T
    if lab.shape[0] > samples:
        lab = lab[torch.randperm(lab.shape[0], generator=g)[:samples]]
    k = min(colors, lab.shape[0])
    centers = lab[torch.randint(0, lab.shape[0], (1,), generator=g)]
    for _ in range(1, k):
        d2 = torch.cdist(lab, centers).pow(2).min(1).values
        pick = torch.multinomial(d2 / d2.sum().clamp(min=1e-12) + 1e-12, 1, generator=g)
        centers = torch.cat([centers, lab[pick]])
    for _ in range(iterations):
        owner = torch.cdist(lab, centers).argmin(1)
        sums = torch.zeros_like(centers).index_add_(0, owner, lab)
        counts = torch.zeros(k).index_add_(0, owner, torch.ones(lab.shape[0]))
        centers = torch.where(counts[:, None] > 0, sums / counts[:, None].clamp(min=1), centers)
    owner = torch.cdist(lab, centers).argmin(1)
    shares = torch.zeros(k).index_add_(0, owner, torch.ones(lab.shape[0])) / lab.shape[0]
    keep = shares > 0
    centers, shares = centers[keep], shares[keep]
    order = centers[:, 0].argsort()
    return oklab_to_srgb(centers[order]).clamp(0, 1), shares[order]


def delta_e(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """OKLab distance between two sRGB images ``(..., 3, H, W)`` -> ``(..., H, W)``."""
    return (image_to_oklab(a) - image_to_oklab(b)).pow(2).sum(-3).add(1e-12).sqrt()


TAU = 2.0 * math.pi

__all__ = ["TAU", "colour_statistics", "delta_e", "extract_palette", "image_to_oklab", "linear_to_oklab",
           "linear_to_srgb", "match_colours", "mix_pigments", "oklab_to_image", "oklab_to_linear", "oklab_to_srgb",
           "snap_to_palette", "srgb_to_linear", "srgb_to_oklab"]
