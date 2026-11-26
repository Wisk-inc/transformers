# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The animator's assistant: an open video model that imagines the motion.

Asked for an animation in words, an open text-to-video model imagines how it moves, and Pinxit then draws every
frame of it by hand, one frame after the other (:mod:`helix_lm.paint.animate`). The video model decides what
happens; every line and every stroke of colour on the board is Pinxit's.

Models (:func:`load_animator`):

* ``"animatelcm"`` (the default where there is no large GPU): AnimateLCM's motion module on DreamShaper 8 (a
  Stable Diffusion 1.5 model that draws cartoons well, CreativeML OpenRAIL-M), eight steps a clip, about 3 GB
  in all. It runs on a CPU, slowly (about 20 minutes for 16 frames of 512 pixels on four cores). AnimateLCM's
  model card names no licence: ask its authors before using it commercially;
* ``"animatediff"``: AnimateDiff's motion module (v1.5-3) on the same base, more steps, often finer motion;
* ``"wan"``: Wan 2.2 TI2V 5B (Apache-2.0, about 34 GB, a large GPU), from words or from a picture of the
  character, so that the character looks the same in every clip; ``"wan-1.3b"``: Wan 2.1 T2V 1.3B (Apache-2.0,
  words only);
* the name of any other ``diffusers`` text-to-video pipeline on the Hub.

Each is the part to fine-tune: ``lora=`` loads a LoRA made for it (a style, a character).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .io import load_image, to_pil

BASE = "Lykon/dreamshaper-8"

DRAWABLE = "2d cartoon animation, clean bold outlines, flat colours, simple shapes, plain background"
NOT_THIS = ("photo, photorealistic, 3d render, blurry, noisy, grainy, text, watermark, logo, frame, deformed, "
            "extra limbs, flicker")
KNOWN = {
    "animatelcm": {"adapter": "wangfuyun/AnimateLCM", "lora": ("wangfuyun/AnimateLCM",
                                                               "AnimateLCM_sd15_t2v_lora.safetensors"),
                   "steps": 8, "guidance": 2.0, "fps": 8, "frames": 16, "size": (512, 512)},
    "animatediff": {"adapter": "guoyww/animatediff-motion-adapter-v1-5-3", "steps": 25, "guidance": 7.5, "fps": 8,
                    "frames": 16, "size": (512, 512)},
    "wan": {"repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "steps": 40, "guidance": 5.0, "fps": 24, "frames": 121,
            "size": (704, 1280)},
    "wan-1.3b": {"repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "steps": 40, "guidance": 5.0, "fps": 16,
                 "frames": 81, "size": (480, 832)},
}


def _frames(out) -> torch.Tensor:
    """What a ``diffusers`` video pipeline gives back -> ``(T, 3, H, W)`` sRGB in 0..1 on the CPU."""
    frames = out.frames if hasattr(out, "frames") else out
    if hasattr(frames, "__array__") and not isinstance(frames, (list, torch.Tensor)):   # Wan: a numpy array
        import numpy as np

        frames = torch.from_numpy(np.asarray(frames))
    if isinstance(frames, torch.Tensor):
        frames = frames[0] if frames.dim() == 5 else frames
        if frames.shape[1] != 3 and frames.shape[-1] == 3:
            frames = frames.permute(0, 3, 1, 2)
        if frames.shape[1] != 3 and frames.shape[0] == 3:         # (3, T, H, W)
            frames = frames.permute(1, 0, 2, 3)
        frames = frames.float().cpu()
        return ((frames + 1) / 2 if float(frames.min()) < -0.01 else frames).clamp(0.0, 1.0)
    if frames and isinstance(frames[0], list):
        frames = frames[0]
    import numpy as np

    pictures = [torch.from_numpy(np.array(f.convert("RGB") if hasattr(f, "convert") else f)).float() for f in frames]
    stack = torch.stack(pictures)
    stack = stack / 255.0 if float(stack.max()) > 1.5 else stack
    return stack.permute(0, 3, 1, 2).clamp(0.0, 1.0)


class Animator:
    """A video pipeline of ``diffusers`` as something the animator can ask: ``animator(words) -> frames``."""

    def __init__(self, pipe, kind: str, name: str = "", steps: int = 20, guidance: float = 6.0, fps: int = 8,
                 frames: int = 16, size: tuple = (512, 512), still=None) -> None:
        self.pipe, self.kind, self.name = pipe, kind, name
        self.steps, self.guidance, self.fps, self.frames, self.size = steps, guidance, fps, frames, size
        self.still = still               # for a character given as a picture: the pipeline that starts from one

    @torch.no_grad()
    def __call__(self, words: str, frames: int | None = None, size: tuple | None = None, seed: int = 0,
                 character=None, negative: str = NOT_THIS, strength: float = 0.75) -> torch.Tensor:
        """The animation ``words`` describe as ``(T, 3, H, W)`` sRGB frames. ``character`` (a picture) is how the
        character looks: the clip starts from it, so it keeps that look."""
        frames, (h, w) = frames or self.frames, size or self.size
        g = torch.Generator().manual_seed(seed)
        common = {"prompt": words, "negative_prompt": negative, "num_inference_steps": self.steps,
                  "guidance_scale": self.guidance, "generator": g}
        if character is not None:
            look = load_image(character, max_side=max(h, w)) if not isinstance(character, torch.Tensor) else character
            look = F.interpolate(look[None].float(), size=(h, w), mode="bilinear", align_corners=False)[0]
            if self.kind == "wan":
                out = self.still(image=to_pil(look), num_frames=frames, height=h, width=w, **common)
                return _frames(out)
            if self.still is not None:              # AnimateDiff: from the picture, held still, set moving
                zoom = [F.interpolate(look[None], scale_factor=1.0 + 0.04 * i / max(1, frames - 1), mode="bilinear",
                                      align_corners=False)[0] for i in range(frames)]
                video = [to_pil(z[:, (z.shape[1] - h) // 2 : (z.shape[1] - h) // 2 + h,
                                  (z.shape[2] - w) // 2 : (z.shape[2] - w) // 2 + w]) for z in zoom]
                return _frames(self.still(video=video, strength=strength, **common))
        if self.kind == "wan":
            return _frames(self.pipe(num_frames=frames, height=h, width=w, **common))
        return _frames(self.pipe(num_frames=frames, height=h, width=w, **common))


def load_animator(name: str = "auto", device=None, lora: str | None = None, base: str = BASE,
                  lora_scale: float = 0.8, **options) -> Animator:
    """
    The video model ``name`` (see the module's notes; ``"auto"`` picks Wan 2.2 on a GPU with 40 GB or more,
    AnimateLCM otherwise), ready to animate. ``base`` is the Stable Diffusion 1.5 model the AnimateDiff motion
    modules run on (any fine-tune of it works, a cartoon one gives cleaner lines). ``lora`` is a LoRA to load:
    the fine-tune. Other keywords go to ``diffusers``' ``from_pretrained``.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if name == "auto":
        big = device.type == "cuda" and torch.cuda.get_device_properties(device).total_memory > 40e9
        name = "wan" if big else "animatelcm"
    known = KNOWN.get(name)
    if known is None:                                   # any text-to-video pipeline on the Hub
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(name, torch_dtype=dtype, **options)
        if lora:
            pipe.load_lora_weights(lora)
        pipe.set_progress_bar_config(disable=True)
        return Animator(pipe.to(device), "other", name)
    if name.startswith("wan"):
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline

        repo = known["repo"]
        vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(repo, vae=vae, torch_dtype=torch.bfloat16 if device.type == "cuda"
                                           else torch.float32, **options)
        if lora:
            pipe.load_lora_weights(lora)
        # from_pipe, not the bare components: it also carries Wan 2.2's expand_timesteps setting over
        still = WanImageToVideoPipeline.from_pipe(pipe) if name == "wan" else None
        pipe.set_progress_bar_config(disable=True)
        if still is not None:
            still.set_progress_bar_config(disable=True)
        return Animator(pipe.to(device), "wan", repo, known["steps"], known["guidance"], known["fps"],
                        known["frames"], known["size"], still)
    from diffusers import AnimateDiffPipeline, AnimateDiffVideoToVideoPipeline, MotionAdapter

    half = {"variant": "fp16"}
    adapter = MotionAdapter.from_pretrained(known["adapter"], torch_dtype=dtype, **half)
    if base in (BASE, "stable-diffusion-v1-5/stable-diffusion-v1-5"):
        options.setdefault("variant", "fp16")             # the same weights in half the download
    pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype, **options)
    if name == "animatelcm":
        from diffusers import LCMScheduler

        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
        repo, weight = known["lora"]
        pipe.load_lora_weights(repo, weight_name=weight, adapter_name="lcm")
        names, scales = ["lcm"], [1.0]
        if lora:
            pipe.load_lora_weights(lora, adapter_name="yours")
            names, scales = names + ["yours"], scales + [lora_scale]
        pipe.set_adapters(names, scales)
    else:
        from diffusers import DDIMScheduler

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear", clip_sample=False,
                                                   timestep_spacing="linspace", steps_offset=1)
        if lora:
            pipe.load_lora_weights(lora)
    if hasattr(pipe.vae, "enable_slicing"):              # one frame at a time through the decoder: less memory
        pipe.vae.enable_slicing()
    if device.type == "cpu" and hasattr(pipe, "enable_free_noise_split_inference"):
        # eight frames at a time through each block: the same result in a fraction of the memory (16 frames of
        # 512 pixels with guidance did not fit in 15 GB otherwise)
        pipe.enable_free_noise_split_inference(spatial_split_size=256, temporal_split_size=8)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    still = AnimateDiffVideoToVideoPipeline(**pipe.components)
    still.set_progress_bar_config(disable=True)
    return Animator(pipe, "animatediff", name, known["steps"], known["guidance"], known["fps"], known["frames"],
                    known["size"], still)


__all__ = ["DRAWABLE", "KNOWN", "Animator", "load_animator"]
