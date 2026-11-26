# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The imaginer: a pretrained image model paired with the painter.

Written rules can lay paint, but they do not know what a good painting looks like. A model trained on millions of
pictures does. So the two are paired: the image model looks at the photo and imagines it as a painting (which
shapes to merge, where an edge is lost, how a face is modelled in a few tones), and Pinxit then paints that,
stroke by stroke, with everything a stroke painting has: a canvas, the trace of the hand, a memory, changes
that can be taken back.

The model is Stable Diffusion 1.5 by default, run as image-to-image so the photo is where it starts from, with
a ControlNet that holds it to the photo's edges so that shapes and faces stay where they were. It is the part to
fine-tune: any LoRA made for it (a painter's manner, one's own paintings) is loaded with ``lora=``. Larger
models work the same way through ``diffusers`` (pass their name; ``control=None`` unless a ControlNet made for
them is given).

To be plain about what this is: the idea of the picture now comes from a model that works by removing noise.
What is drawn, every mark on the canvas, is still Pinxit's.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .emotion import strongest
from .io import to_pil
from .see import gaussian_blur

MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROL = "lllyasviel/control_v11p_sd15_canny"
PAINT = "oil painting on canvas, visible brushstrokes, painterly, loose expressive brushwork, by a master painter"
NOT_THIS = ("photo, photograph, photorealistic, realistic, camera, 3d render, smooth, blurry, text, watermark, "
            "signature, frame, border, deformed, ugly")
MOOD = {"awe": "majestic, grand, awe-inspiring light", "serenity": "calm, serene, still, soft light",
        "joy": "joyful, bright, lively colour", "tenderness": "tender, gentle, warm, intimate",
        "melancholy": "melancholy, muted, wistful, grey light", "fear": "ominous, dark, threatening, harsh shadows",
        "anger": "fierce, violent brushwork, burning reds", "mystery": "mysterious, dusk, deep shadows, veiled"}


def edges(image: torch.Tensor, low: float = 0.08, high: float = 0.2) -> torch.Tensor:
    """The picture's edges as thin white lines on black ``(3, H, W)``, the way a Canny detector draws them
    (what the edge ControlNet was trained on): smoothed, the steepest point across each edge kept, strong edges
    taken and weaker ones only where they touch a strong one."""
    grey = gaussian_blur(image.float().mean(0), 1.2)
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=image.device)
    padded = F.pad(grey[None, None], (1, 1, 1, 1), mode="replicate")
    gx, gy = F.conv2d(padded, kx[None, None])[0, 0], F.conv2d(padded, kx.T[None, None])[0, 0]
    steep = (gx ** 2 + gy ** 2).sqrt()
    steep = steep / steep.max().clamp(min=1e-8)
    turn = torch.round(torch.atan2(gy, gx) / (torch.pi / 4)) % 4          # which of four ways the edge is crossed
    crest = torch.zeros_like(steep, dtype=torch.bool)
    for k, (dy, dx) in enumerate(((0, 1), (1, 1), (1, 0), (1, -1))):
        ahead, behind = torch.roll(steep, (-dy, -dx), (0, 1)), torch.roll(steep, (dy, dx), (0, 1))
        crest |= (turn == k) & (steep >= ahead) & (steep >= behind)
    strong, weak = crest & (steep >= high), crest & (steep >= low)
    for _ in range(12):                                                   # weak edges that hang on to strong ones
        strong = weak & (F.max_pool2d(strong[None, None].float(), 3, 1, 1)[0, 0] > 0)
    return strong.float()[None].expand(3, -1, -1)


def describe(card=None, feeling_name: str | None = None, subject: str | None = None) -> str:
    """The words the image model is given: what is shown (when said), that it is a painting, in whose manner or
    with what handling (from the style card), and the feeling."""
    parts = [subject] if subject else []
    parts.append(PAINT)
    if card is not None and getattr(card, "measured", False):
        name = getattr(card, "name", "") or ""
        if "," in name:                                              # a famous painting: "Title, painter"
            title, painter = (s.strip() for s in name.rsplit(",", 1))
            parts.append(f"in the style of {painter}, {title}")
        if card.hardness < 0.4:
            parts.append("soft blended edges, sfumato, thin glazes")
        if card.marks() > 0.4:
            parts.append("thick impasto, short broken brushstrokes laid side by side")
        if card.outline > 0.3:
            parts.append("bold dark outlines round the shapes")
        if card.flat > 0.15 and card.hardness > 0.7:
            parts.append("flat areas of colour")
    if feeling_name in MOOD:
        parts.append(MOOD[feeling_name])
    return ", ".join(parts)


class Imaginer:
    """An image-to-image pipeline of ``diffusers`` (and a ControlNet for it, or None) as something the painter
    can ask: ``imaginer(photo, words) -> the photo imagined as a painting``."""

    def __init__(self, pipe, name: str = "", controlled: bool = False, steps: int = 20, guidance: float = 8.0,
                 strength: float | None = None) -> None:
        self.pipe, self.name, self.controlled, self.steps, self.guidance = pipe, name, controlled, steps, guidance
        # held to the photo's edges the model may repaint nearly everything; without them it must stay closer
        self.strength = strength if strength is not None else (0.8 if controlled else 0.58)

    @torch.no_grad()
    def __call__(self, image: torch.Tensor, words: str, strength: float | None = None, seed: int = 0,
                 negative: str = NOT_THIS) -> torch.Tensor:
        """``image (3, H, W)`` sRGB in 0..1 -> the same size, imagined as ``words`` say. ``strength`` is how far
        the model may move from the photo (0: not at all, 1: only the words and the edges are left)."""
        h, w = image.shape[-2:]
        hh, ww = max(64, h // 8 * 8), max(64, w // 8 * 8)
        start = F.interpolate(image[None].float(), size=(hh, ww), mode="bilinear", align_corners=False,
                              antialias=True)[0].clamp(0.0, 1.0)
        more = {"control_image": to_pil(edges(start))} if self.controlled else {}
        out = self.pipe(prompt=words, negative_prompt=negative, image=to_pil(start),
                        strength=self.strength if strength is None else strength, num_inference_steps=self.steps,
                        guidance_scale=self.guidance, generator=torch.Generator().manual_seed(seed),
                        output_type="pt", **more).images[0]
        out = out.float().cpu().clamp(0.0, 1.0)
        if out.shape[-2:] != (h, w):
            out = F.interpolate(out[None], size=(h, w), mode="bilinear", align_corners=False)[0]
        return out


def load_imaginer(name: str = MODEL, control: str | None = CONTROL, lora: str | None = None, device=None,
                  steps: int = 20, **options) -> Imaginer:
    """
    The image model ``name`` from the Hub (Stable Diffusion 1.5 by default: about 2 GB with its half-precision
    files, CreativeML OpenRAIL-M), held to the photo's edges by the ControlNet ``control`` (0.7 GB; None for
    none). ``lora`` is a LoRA to load into it: that is the fine-tune, a painter's manner or one's own
    paintings. It goes to the GPU when there is one. Other keywords go to ``diffusers``' ``from_pretrained``.
    Needs ``pip install "helix-lm[paint]"``.
    """
    from diffusers import AutoPipelineForImage2Image, ControlNetModel, DPMSolverMultistepScheduler

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if name == MODEL:
        options.setdefault("variant", "fp16")                        # the same weights in half the download
    if control:
        half = {"variant": "fp16"} if control == CONTROL else {}
        options["controlnet"] = ControlNetModel.from_pretrained(control, torch_dtype=dtype, **half)
    pipe = AutoPipelineForImage2Image.from_pretrained(name, torch_dtype=dtype, **options)
    if hasattr(pipe, "unet"):                                        # the Stable Diffusion family: fewer steps
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    if lora:
        pipe.load_lora_weights(lora)
    pipe.set_progress_bar_config(disable=True)
    return Imaginer(pipe.to(device), name, bool(control), steps)


def feeling_word(feeling_or_name) -> str | None:
    """The name of a feeling for :func:`describe`, from a name or a ``{name: share}`` reading."""
    if isinstance(feeling_or_name, str):
        return feeling_or_name
    if isinstance(feeling_or_name, dict) and feeling_or_name:
        return strongest(feeling_or_name)
    return None


__all__ = ["CONTROL", "MODEL", "Imaginer", "describe", "edges", "feeling_word", "load_imaginer"]
