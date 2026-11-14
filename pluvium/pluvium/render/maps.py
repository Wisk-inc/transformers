# Copyright 2026 Nathan. Apache-2.0.
"""
Maps: one picture of where rain is likely.

Land is drawn from the terrain raster (darker with height), the sea plain, and every place is a dot coloured by the
chosen quantity -- the chance of rain in the next 24 hours, of thunder, or the flood level -- on a colour scale with a
legend. Pillow only, so it runs anywhere the forecast does.
"""

from __future__ import annotations

import numpy as np

SCALE = [(0.0, (240, 240, 235)), (0.2, (198, 219, 239)), (0.4, (107, 174, 214)), (0.6, (33, 113, 181)),
         (0.8, (8, 69, 148)), (1.0, (8, 29, 88))]


def _colour(v: float) -> tuple:
    v = float(np.clip(v, 0, 1))
    for (a, ca), (b, cb) in zip(SCALE[:-1], SCALE[1:]):
        if v <= b:
            w = (v - a) / (b - a)
            return tuple(int(ca[i] + (cb[i] - ca[i]) * w) for i in range(3))
    return SCALE[-1][1]


def draw_map(path, terrain, lat: np.ndarray, lon: np.ndarray, values: np.ndarray, box: tuple, title: str,
             width: int = 1200, dot: int = 3) -> str:
    """Write a PNG of ``values`` (0-1) at points over the terrain in ``box = (south, north, west, east)``."""
    from PIL import Image, ImageDraw

    s, n, w, e = box
    height = int(width * (n - s) / ((e - w) * np.cos(np.radians(0.5 * (s + n)))))
    yy = np.linspace(n, s, height)
    xx = np.linspace(w, e, width)
    la, lo = np.meshgrid(yy, xx, indexing="ij")
    land = terrain.sample(terrain.land.astype(np.float32), la.ravel(), lo.ravel()).reshape(la.shape) > 0.5
    elev = terrain.sample(terrain.elevation, la.ravel(), lo.ravel()).reshape(la.shape)
    shade = np.clip(200 - elev / 15.0, 110, 200).astype(np.uint8)
    rgb = np.where(land[..., None], np.stack([shade, shade + 15, shade - 10], -1).clip(0, 255),
                   np.array([225, 235, 245], np.uint8)).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(img)
    inside = (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)
    order = np.argsort(values[inside])
    px = ((lon[inside] - w) / (e - w) * (width - 1))[order]
    py = ((n - lat[inside]) / (n - s) * (height - 1))[order]
    for x, y, v in zip(px, py, values[inside][order]):
        draw.ellipse([x - dot, y - dot, x + dot, y + dot], fill=_colour(v))
    draw.rectangle([10, 10, 10 + 8 * len(title) + 20, 34], fill=(255, 255, 255))
    draw.text((20, 16), title, fill=(0, 0, 0))
    for k, v in enumerate(np.linspace(0, 1, 6)):
        draw.rectangle([20 + 50 * k, height - 40, 60 + 50 * k, height - 20], fill=_colour(v))
        draw.text((24 + 50 * k, height - 18), f"{v:.0%}", fill=(0, 0, 0))
    img.save(path)
    return str(path)


__all__ = ["draw_map"]
