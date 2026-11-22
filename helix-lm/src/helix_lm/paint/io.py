# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Pictures in and out: files, PIL images and arrays become ``(3, H, W)`` sRGB tensors in 0..1 and back."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


def load_image(source, max_side: int | None = None) -> torch.Tensor:
    """A path, a PIL image, a numpy array ``(H, W, 3)`` or a tensor ``(3, H, W)`` -> ``(3, H, W)`` sRGB in 0..1.
    A picture with transparency is laid on white. ``max_side`` shrinks a larger picture."""
    import numpy as np

    if isinstance(source, torch.Tensor):
        image = source.detach().float().cpu()
        image = image / 255.0 if image.max() > 1.5 else image
    else:
        from PIL import Image, ImageOps

        pil = source if isinstance(source, Image.Image) else Image.open(Path(source))
        pil = ImageOps.exif_transpose(pil) if not isinstance(source, np.ndarray) else pil
        if isinstance(source, np.ndarray):
            pil = Image.fromarray(source)
        if pil.mode in ("RGBA", "LA", "P"):
            rgba = pil.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            pil = Image.alpha_composite(white, rgba)
        image = torch.from_numpy(np.asarray(pil.convert("RGB")).copy()).permute(2, 0, 1).float() / 255.0
    if image.dim() == 2:
        image = image[None].expand(3, -1, -1)
    image = image[:3].clamp(0.0, 1.0)
    if max_side and max(image.shape[-2:]) > max_side:
        scale = max_side / max(image.shape[-2:])
        image = resize(image, (max(1, round(image.shape[-2] * scale)), max(1, round(image.shape[-1] * scale))))
    return image


def load_alpha(source) -> torch.Tensor | None:
    """The transparency of a picture file as ``(H, W)`` in 0..1, or None when it has none."""
    import numpy as np
    from PIL import Image

    if isinstance(source, (torch.Tensor, np.ndarray)):
        return None
    pil = source if isinstance(source, Image.Image) else Image.open(Path(source))
    if pil.mode not in ("RGBA", "LA") and "transparency" not in pil.info:
        return None
    return torch.from_numpy(np.asarray(pil.convert("RGBA")).copy())[..., 3].float() / 255.0


def resize(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """``(..., C, H, W)`` -> ``(..., C, size[0], size[1])``: averaging when shrinking, smooth when enlarging."""
    lead = image.shape[:-3]
    x = image.reshape(-1, *image.shape[-3:])
    shrink = size[0] < x.shape[-2] or size[1] < x.shape[-1]
    x = F.interpolate(x, size=size, mode="area") if shrink else F.interpolate(
        x, size=size, mode="bilinear", align_corners=False)
    return x.reshape(*lead, *x.shape[-3:])


def to_pil(image: torch.Tensor):
    import numpy as np
    from PIL import Image

    arr = (image.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def save_image(image: torch.Tensor, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil(image).save(path)
    return path


__all__ = ["load_alpha", "load_image", "resize", "save_image", "to_pil"]
