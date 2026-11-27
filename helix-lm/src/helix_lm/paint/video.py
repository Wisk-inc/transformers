# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The animator's assistant: an open video model that imagines the motion.

Asked for an animation in words, an open text-to-video model imagines how it moves, and Pinxit then draws every
frame of it by hand, one frame after the other (:mod:`helix_lm.paint.animate`). The video model decides what
happens; every line and every stroke of colour on the board is Pinxit's. How good the shapes and the motion are
depends mostly on this model, so it can be switched at any time: ``load_animator(name)``. :func:`list_models`
gives the table (repo, licence, download, memory, what each is good for).

Models (:func:`load_animator`):

* ``"animatelcm"`` (the default where there is no large GPU): AnimateLCM's motion module on DreamShaper 8 (a
  Stable Diffusion 1.5 model that draws cartoons well, CreativeML OpenRAIL-M), eight steps a clip, about 3 GB
  in all. It runs on a CPU, slowly (about 20 minutes for 16 frames of 512 pixels on four cores). AnimateLCM's
  model card names no licence: ask its authors before using it commercially. Shapes morph;
* ``"animatediff"``: AnimateDiff's motion module (v1.5-3) on the same base, more steps, often finer motion;
* ``"wan"``: Wan 2.2 TI2V 5B (Apache-2.0, about 34 GB), from words or from a picture of the character, so that
  the character looks the same in every clip; ``"wan-1.3b"``: Wan 2.1 T2V 1.3B (Apache-2.0, words; a picture
  is held still and set moving, as with AnimateDiff);
* ``"wan-14b"``: Wan 2.2 T2V A14B (Apache-2.0): two 14B experts, one for the noisy start of each clip and one
  for the detail at the end. The best shapes and motion from words of the Wan family. Given a picture of the
  character it loads Wan 2.2 I2V A14B, which starts from the picture; ``"wan-i2v-14b"`` is that model alone;
* ``"wan-animate"``: Wan 2.2 Animate 14B (Apache-2.0): a picture of the character performs the motion of a video
  of someone moving (``motion=``, see below). The surest way to a character that keeps its look;
* ``"wan-vace"``, ``"wan-vace-1.3b"``: Wan 2.1 VACE 14B and 1.3B (Apache-2.0): words with the character given
  as a reference picture, and a pose (or depth, or line) video as the motion;
* ``"ltx"``: LTX-Video 0.9.7 13B dev (fast; LTXV Open Weights licence: free for a company under 10 million
  dollars a year of revenue, paid above); ``"ltx2"``: LTX-2 19B (the same terms; it makes sound too, dropped);
* ``"hunyuan"``: HunyuanVideo 1.5 480p (8.3B; the Tencent Hunyuan Community licence does not apply in the
  European Union, the United Kingdom or South Korea: it cannot be used there);
* ``"cogvideox"``: CogVideoX 5B (720x480 only, 8 frames a second; free for research, commercial use needs a free
  registration with its makers);
* ``"skyreels"``: SkyReels V2 14B 540p (Skywork Community licence), trained on film: lifelike motion;
* the repo id of any other ``diffusers`` video pipeline on the Hub (``DiffusionPipeline`` picks the class).

Everything but AnimateLCM and AnimateDiff needs a GPU. On one, the weights are bf16 and the memory is planned
from the card's size: everything on the GPU when it fits, else one model at a time on it
(``enable_model_cpu_offload``), else one layer at a time (slow), and the VAE decodes in tiles so a long clip
fits (``offload=`` chooses by hand). By these estimates every model here fits whole on the 96 GB RTX PRO 6000;
the 14B Wan T2V and I2V together do not, so switching between words and a picture loads the other one again
from the disk cache (a minute or two), sharing the text encoder and the VAE. Loading also needs about as much
CPU memory as the weights.

The motion of someone else (:func:`motion_videos`). Wan-Animate does not take the driving video itself but two
videos made from it: the pose (a stick figure drawn on black, at the size of the clip) and the face (the face
cut out and scaled to 512 by 512), 30 frames a second. ``motion=`` takes the driving video, ``(T, 3, H, W)`` sRGB
in 0..1 (or ``(video, fps)``, or a file), finds the person in every frame with RT-DETR and ViTPose
(``transformers``, Apache-2.0, about 0.5 GB, downloaded on first use), and draws the pose and cuts the face the
way Wan-Animate's own preprocessing does. ViTPose here knows 17 points of the body: the hands are not drawn and
the face box is worked out from the eyes, ears and nose, so fingers and the face follow less closely than with
Wan's own preprocessing (``wan/modules/animate/preprocess`` in the Wan2.2 repository: YOLOv10 and a whole-body
ViTPose-H, about 2.6 GB, onnxruntime). Its output can be given as it is: ``motion={"pose": ..., "face": ...}``;
keypoints of the 133-point COCO-WholeBody kind from any detector are drawn in full:
``motion={"video": ..., "keypoints": ...}``. The pose is not fitted to the character's proportions (Wan's
"retarget" step): a character with a big head and short legs moves with the driving person's proportions.
The driving video should show a person (or something shaped like one): an animal has no pose to find. VACE
takes the same stick figure as its control video, much like its own pose annotator draws it but not exactly
(``motion={"control": ...}`` gives it a control video of one's own: depth, lines, a pose).

Each is the part to fine-tune: ``lora=`` loads a LoRA made for it (a style, a character).

None of the large models was run here (there was no GPU and about 1.5 GB of disk). The arguments follow the
code of ``diffusers`` 0.43 and the model cards, the tests use stand-in pipelines, and the memory figures are
estimates from the sizes of the weights.
"""

from __future__ import annotations

import colorsys
import gc
import inspect
import math
import os

import torch
import torch.nn.functional as F

from .io import load_image, resize, to_pil

BASE = "Lykon/dreamshaper-8"

DRAWABLE = "2d cartoon animation, clean bold outlines, flat colours, simple shapes, plain background"
NOT_THIS = ("photo, photorealistic, 3d render, blurry, noisy, grainy, text, watermark, logo, frame, deformed, "
            "extra limbs, flicker")

WAN_SHARE = ("text_encoder", "tokenizer", "vae")                 # the same files in the T2V and the I2V repo
HUNYUAN_SHARE = ("text_encoder", "tokenizer", "text_encoder_2", "tokenizer_2")       # its two VAEs differ
APACHE = "Apache-2.0"
LTX_LICENCE = "LTXV Open Weights 0.X: free under 10 million dollars a year of revenue, a paid licence above"

# download and memory in GB: "weights" all parts in bf16, "largest" the biggest part, "work" the working memory
# at the model's own size; "still_gb" what the picture model adds when it is a model of its own
KNOWN = {
    "animatelcm": {"kind": "animatediff", "adapter": "wangfuyun/AnimateLCM",
                   "lora": ("wangfuyun/AnimateLCM", "AnimateLCM_sd15_t2v_lora.safetensors"),
                   "steps": 8, "guidance": 2.0, "fps": 8, "frames": 16, "size": (512, 512),
                   "licence": "none named (AnimateLCM); the base DreamShaper 8 is CreativeML OpenRAIL-M",
                   "download": 3, "weights": 3, "largest": 2, "work": 5, "gpu": False,
                   "about": "a quick try: runs on a CPU (about 20 minutes a clip); shapes morph"},
    "animatediff": {"kind": "animatediff", "adapter": "guoyww/animatediff-motion-adapter-v1-5-3", "steps": 25,
                    "guidance": 7.5, "fps": 8, "frames": 16, "size": (512, 512),
                    "licence": "none named (AnimateDiff's adapter); the base is CreativeML OpenRAIL-M",
                    "download": 3, "weights": 3, "largest": 2, "work": 5, "gpu": False,
                    "about": "finer motion than AnimateLCM, slower (25 steps)"},
    "wan": {"kind": "wan", "repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "pipeline": "WanPipeline",
            "still": "WanImageToVideoPipeline", "steps": 40, "guidance": 5.0, "fps": 24, "frames": 121,
            "size": (704, 1280), "multiple": 32, "step": 4, "licence": APACHE, "download": 34, "weights": 24,
            "largest": 11, "work": 10, "gpu": True,
            "about": "words or a picture in one 5B model, 720p at 24 frames a second; for 40 to 80 GB"},
    "wan-1.3b": {"kind": "wan", "repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "pipeline": "WanPipeline",
                 "still": "WanVideoToVideoPipeline", "held": True, "steps": 40, "guidance": 5.0, "fps": 16,
                 "frames": 81, "size": (480, 832), "multiple": 16, "step": 4, "licence": APACHE, "download": 29,
                 "weights": 15, "largest": 11, "work": 5, "gpu": True,
                 "about": "small Wan from words, 480p; a picture is held still and set moving"},
    "wan-14b": {"kind": "wan", "repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers", "pipeline": "WanPipeline",
                "still": "WanImageToVideoPipeline", "still_repo": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                "share": WAN_SHARE, "steps": 40, "guidance": 4.0, "extra": {"guidance_scale_2": 3.0},
                "still_extra": {"guidance_scale": 3.5, "guidance_scale_2": 3.5}, "fps": 16, "frames": 81,
                "size": (480, 832), "multiple": 16, "step": 4, "licence": APACHE, "download": 126,
                "still_download": 114, "weights": 69, "largest": 29, "work": 12, "still_gb": 57, "gpu": True,
                "about": "the best Wan shapes and motion from words (two 14B experts); a picture loads the I2V"},
    "wan-i2v-14b": {"kind": "wan-i2v", "repo": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                    "pipeline": "WanImageToVideoPipeline", "still": "same", "steps": 40, "guidance": 3.5,
                    "extra": {"guidance_scale_2": 3.5}, "fps": 16, "frames": 81, "size": (480, 832), "multiple": 16,
                    "step": 4, "licence": APACHE, "download": 126, "weights": 69, "largest": 29, "work": 12,
                    "gpu": True, "about": "starts from a picture of the character and keeps its look (character=)"},
    "wan-animate": {"kind": "wan-animate", "repo": "Wan-AI/Wan2.2-Animate-14B-Diffusers",
                    "pipeline": "WanAnimatePipeline", "steps": 20, "guidance": 1.0,
                    "extra": {"segment_frame_length": 77, "prev_segment_conditioning_frames": 1}, "fps": 30,
                    "frames": 77, "size": (720, 1280), "multiple": 16, "licence": APACHE, "download": 82,
                    "weights": 48, "largest": 35, "work": 14, "gpu": True,
                    "about": "a character picture performs the motion of a video (motion=): keeps the look best"},
    "wan-vace": {"kind": "vace", "repo": "Wan-AI/Wan2.1-VACE-14B-diffusers", "pipeline": "WanVACEPipeline",
                 "steps": 30, "guidance": 5.0, "fps": 16, "frames": 81, "size": (480, 832), "multiple": 16,
                 "step": 4, "licence": APACHE, "download": 75, "weights": 44, "largest": 32, "work": 12, "gpu": True,
                 "about": "words with the character as a reference picture; a pose video as the motion"},
    "wan-vace-1.3b": {"kind": "vace", "repo": "Wan-AI/Wan2.1-VACE-1.3B-diffusers", "pipeline": "WanVACEPipeline",
                      "steps": 30, "guidance": 5.0, "fps": 16, "frames": 81, "size": (480, 832), "multiple": 16,
                      "step": 4, "licence": APACHE, "download": 19, "weights": 16, "largest": 11, "work": 5,
                      "gpu": True, "about": "the same as wan-vace in a small model, 480p"},
    "ltx": {"kind": "ltx", "repo": "Lightricks/LTX-Video-0.9.7-dev", "pipeline": "LTXConditionPipeline",
            "still": "same", "steps": 30, "guidance": 3.0,
            "extra": {"frame_rate": 24, "decode_timestep": 0.05, "decode_noise_scale": 0.025}, "fps": 24,
            "frames": 121, "size": (480, 704), "multiple": 32, "step": 8, "licence": LTX_LICENCE, "download": 48,
            "weights": 38, "largest": 26, "work": 8, "gpu": True, "about": "fast 13B model; words or a first frame"},
    "ltx2": {"kind": "ltx", "repo": "Lightricks/LTX-2", "pipeline": "LTX2Pipeline",
             "still": "LTX2ImageToVideoPipeline", "steps": 40, "guidance": 4.0, "extra": {"frame_rate": 24.0},
             "fps": 24, "frames": 121, "size": (512, 768), "multiple": 32, "step": 8,
             "licence": "LTX-2 Community: free under 10 million dollars a year of revenue, a paid licence above",
             "download": 145, "weights": 68, "largest": 38, "work": 12, "gpu": True,
             "about": "19B, sharp and quick; it makes sound too (dropped here); the biggest download"},
    "hunyuan": {"kind": "hunyuan", "repo": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
                "pipeline": "HunyuanVideo15Pipeline", "still": "HunyuanVideo15ImageToVideoPipeline",
                "still_repo": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v", "share": HUNYUAN_SHARE,
                "steps": 50, "guidance": 6.0, "fps": 24, "frames": 121, "size": (480, 848), "multiple": 16,
                "step": 4, "licence": "Tencent Hunyuan Community: not valid in the EU, the UK or South Korea",
                "download": 53, "still_download": 39, "weights": 34, "largest": 17, "work": 10, "still_gb": 20,
                "gpu": True, "about": "8.3B, steady motion, good at words written in the picture; 480p"},
    "cogvideox": {"kind": "cogvideox", "repo": "zai-org/CogVideoX-5b", "pipeline": "CogVideoXPipeline",
                  "still": "CogVideoXImageToVideoPipeline", "still_repo": "zai-org/CogVideoX-5b-I2V",
                  "share": WAN_SHARE, "steps": 50, "guidance": 6.0, "extra": {"use_dynamic_cfg": True}, "fps": 8,
                  "frames": 49, "most": 49, "size": (480, 720), "fixed": True, "multiple": 16, "step": 4,
                  "licence": "CogVideoX: free for research; commercial use needs a free registration",
                  "download": 22, "still_download": 12, "weights": 22, "largest": 11, "work": 8, "still_gb": 11,
                  "gpu": True, "about": "5B, 6 seconds of 720x480 only at 8 frames a second"},
    "skyreels": {"kind": "skyreels", "repo": "Skywork/SkyReels-V2-T2V-14B-540P-Diffusers",
                 "pipeline": "SkyReelsV2Pipeline", "still": "SkyReelsV2ImageToVideoPipeline",
                 "still_repo": "Skywork/SkyReels-V2-I2V-14B-540P-Diffusers", "share": WAN_SHARE, "shift": 8.0,
                 "still_shift": 5.0, "steps": 50, "guidance": 6.0, "still_extra": {"guidance_scale": 5.0},
                 "fps": 24, "frames": 97, "size": (544, 960), "multiple": 16, "step": 4,
                 "licence": "Skywork Community License (commercial use on its terms)", "download": 80,
                 "still_download": 68, "weights": 41, "largest": 29, "work": 12, "still_gb": 34, "gpu": True,
                 "about": "14B trained on film: lifelike motion and camera; 540p"},
}
_BY_REPO = {known.get("repo"): name for name, known in KNOWN.items() if known.get("repo")}

# finding a person and their pose in a driving video (transformers, Apache-2.0, about 0.2 and 0.3 GB)
DETECTOR = "PekingU/rtdetr_r50vd_coco_o365"
POSE = "usyd-community/vitpose-base-simple"

# the stick figure of Wan-Animate's preprocessing: 20 body points in the OpenPose order (nose, neck, right
# shoulder, elbow, wrist, left shoulder, elbow, wrist, right hip, knee, ankle, left hip, knee, ankle, right eye,
# left eye, right ear, left ear, left toe, right toe), each the mean of two COCO-WholeBody points
_PAIR = ([0, 6, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 17, 20],
         [0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 18, 21])
_LIMBS = [(2, 3), (2, 6), (3, 4), (4, 5), (6, 7), (7, 8), (2, 9), (9, 10), (10, 11), (2, 12), (12, 13), (13, 14),
          (2, 1), (1, 15), (15, 17), (1, 16), (16, 18), (14, 19), (11, 20)]
_COLOURS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
            (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
            (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85), (200, 200, 0), (100, 100, 0)]
_HAND = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


# --------------------------------------------------------------------------------------------- frames --
def _one(picture) -> torch.Tensor:
    import numpy as np

    if isinstance(picture, torch.Tensor):
        return picture
    if hasattr(picture, "convert"):                               # a PIL image
        return torch.from_numpy(np.array(picture.convert("RGB")))
    return torch.from_numpy(np.array(picture))


def _frames(out) -> torch.Tensor:
    """What a ``diffusers`` video pipeline gives back -> ``(T, 3, H, W)`` sRGB in 0..1 on the CPU: an output with
    ``.frames`` or the frames themselves, as a list of pictures (PIL, arrays, tensors), a list of such lists, a
    numpy array or a tensor (``(B, T, H, W, 3)``, ``(B, C, T, H, W)``, ``(T, C, H, W)``...) in 0..1, -1..1 or
    0..255."""
    import numpy as np

    frames = out.frames if hasattr(out, "frames") else out
    if isinstance(frames, tuple):                                 # return_dict=False: (frames, ...)
        frames = frames[0]
    if isinstance(frames, list) and frames and isinstance(frames[0], (list, tuple)):
        frames = frames[0]                                        # a batch of clips: the first
    if isinstance(frames, list) and frames and np.ndim(frames[0]) == 4:
        frames = frames[0]                                        # a list of clips as arrays
    if isinstance(frames, list):
        frames = torch.stack([_one(f) for f in frames])
    elif not isinstance(frames, torch.Tensor):                    # Wan: a numpy array
        frames = torch.from_numpy(np.ascontiguousarray(np.asarray(frames)))
    whole = frames.dtype == torch.uint8
    frames = frames.detach().float().cpu()
    frames = frames[0] if frames.dim() == 5 else frames
    frames = frames[None] if frames.dim() == 3 else frames
    if frames.shape[1] != 3 and frames.shape[-1] == 3:
        frames = frames.permute(0, 3, 1, 2)
    elif frames.shape[1] != 3 and frames.shape[0] == 3:           # (3, T, H, W)
        frames = frames.permute(1, 0, 2, 3)
    if whole or float(frames.max()) > 1.5:
        frames = frames / 255.0
    elif float(frames.min()) < -0.01:
        frames = (frames + 1) / 2
    return frames.clamp(0.0, 1.0).contiguous()


def _ask(pipe, **kwargs):
    """Call a pipeline with the arguments it takes (one that takes ``**kwargs`` gets them all)."""
    try:
        params = inspect.signature(pipe).parameters
    except (TypeError, ValueError):
        return pipe(**kwargs)
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in params}
    return pipe(**kwargs)


def _takes(pipe, name: str) -> bool:
    try:
        params = inspect.signature(pipe).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _fit_picture(look: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """A picture ``(3, H, W)`` at ``h`` by ``w``: scaled, and when its shape differs, laid whole in the middle on
    the colour of its own border (a character on a plain ground stays whole and in proportion)."""
    big, wide = look.shape[-2:]
    if abs(big / wide - h / w) <= 0.02 * h / w:
        return resize(look, (h, w))
    s = min(h / big, w / wide)
    nh, nw = max(1, round(big * s)), max(1, round(wide * s))
    border = torch.cat([look[:, 0], look[:, -1], look[:, :, 0], look[:, :, -1]], dim=1).mean(1)
    canvas = border[:, None, None].expand(3, h, w).clone()
    top, left = (h - nh) // 2, (w - nw) // 2
    canvas[:, top : top + nh, left : left + nw] = resize(look, (nh, nw))
    return canvas


def _by_area(hw, area: float, multiple: int) -> tuple:
    """A size of about ``area`` pixels with the shape of ``hw``, sides a multiple of ``multiple`` (as Wan-Animate
    sizes its reference picture)."""
    h, w = hw
    nh = math.sqrt(area * h / w)
    nw = area / nh
    return (max(multiple, int(nh // multiple) * multiple), max(multiple, int(nw // multiple) * multiple))


def _reflect(have: int, n: int) -> list:
    """Indices that make ``have`` frames last ``n``: back and forth, as Wan-Animate pads its videos."""
    if have <= 1:
        return [0] * n
    out, i, step = [], 0, 1
    while len(out) < n:
        out.append(i)
        if not 0 <= i + step < have:
            step = -step
        i += step
    return out


def _retime(count: int, rate: float | None, fps: float) -> list:
    """Indices of the frames of a video of ``count`` frames at ``rate`` a second that make it ``fps`` a second."""
    if not rate or abs(rate - fps) < 0.01:
        return list(range(count))
    target = max(1, int(count / rate * fps))
    return [min(count - 1, round(i / fps * rate)) for i in range(target)]


def _pad_resize(video: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """``(..., 3, H, W)`` fitted inside ``h`` by ``w`` in proportion, on black (Wan-Animate's padding_resize)."""
    big, wide = video.shape[-2:]
    if (big, wide) == (h, w):
        return video
    lead = video.shape[:-3]
    flat = video.reshape(-1, *video.shape[-3:])
    if big / wide > h / w:
        nh, nw = h, max(1, int(h / big * wide))
    else:
        nh, nw = max(1, int(w / wide * big)), w
    fitted = F.interpolate(flat, size=(nh, nw), mode="bilinear", align_corners=False)
    out = torch.zeros(flat.shape[0], 3, h, w)
    top, left = (h - nh) // 2, (w - nw) // 2
    out[:, :, top : top + nh, left : left + nw] = fitted
    return out.reshape(*lead, 3, h, w)


# ------------------------------------------------------------------------------------ someone's motion --
@torch.no_grad()
def find_pose(frames: torch.Tensor, detector: str = DETECTOR, estimator: str = POSE, device=None,
              batch: int = 8) -> torch.Tensor:
    """The pose of the person in every frame of ``frames (T, 3, H, W)``: ``(T, 17, 3)``, the 17 COCO points of
    the body (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) as x, y in pixels and a score.
    RT-DETR finds the people (the biggest one is kept) and ViTPose finds their points; both come from
    ``transformers`` and are downloaded on first use."""
    from transformers import AutoImageProcessor, AutoModelForObjectDetection, VitPoseForPoseEstimation

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    find = AutoImageProcessor.from_pretrained(detector)
    finder = AutoModelForObjectDetection.from_pretrained(detector).to(device).eval()
    see = AutoImageProcessor.from_pretrained(estimator)
    seer = VitPoseForPoseEstimation.from_pretrained(estimator).to(device).eval()
    experts = getattr(getattr(seer.config, "backbone_config", None), "num_experts", 1)
    t, _, h, w = frames.shape
    out, last = torch.zeros(t, 17, 3), [0.0, 0.0, float(w), float(h)]
    for start in range(0, t, batch):
        pictures = [to_pil(f) for f in frames[start : start + batch]]
        found = find.post_process_object_detection(finder(**find(images=pictures, return_tensors="pt").to(device)),
                                                   target_sizes=[(h, w)] * len(pictures), threshold=0.4)
        boxes = []
        for result in found:
            people = result["boxes"][result["labels"] == 0].float().cpu()     # label 0 of COCO: a person
            if len(people):
                x0, y0, x1, y1 = people[((people[:, 2] - people[:, 0]) * (people[:, 3] - people[:, 1])).argmax()]
                last = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
            boxes.append([last])                                   # no one found: where they were last
        inputs = see(pictures, boxes=boxes, return_tensors="pt").to(device)
        extra = {"dataset_index": torch.zeros(len(pictures), dtype=torch.long, device=device)} if experts > 1 else {}
        for i, people in enumerate(see.post_process_pose_estimation(seer(**inputs, **extra), boxes=boxes)):
            out[start + i, :, :2] = people[0]["keypoints"].float().cpu()
            out[start + i, :, 2] = people[0]["scores"].float().cpu()
    return out


def _parts(kp: torch.Tensor):
    """One frame's points (17 COCO or 133 COCO-WholeBody, ``(K, 3)``) -> the 20 body points, the left and the
    right hand (21 points each, or None)."""
    if kp.shape[0] >= 133:
        return (kp[_PAIR[0]] + kp[_PAIR[1]]) / 2, kp[91:112], kp[112:133]
    body = torch.zeros(20, 3)
    body[:18] = (kp[_PAIR[0][:18]] + kp[_PAIR[1][:18]]) / 2
    return body, None, None


def _stamp(canvas: torch.Tensor, box, inside, colour) -> None:
    h, w = canvas.shape[-2:]
    x0, y0 = max(0, int(math.floor(box[0]))), max(0, int(math.floor(box[1])))
    x1, y1 = min(w - 1, int(math.ceil(box[2]))), min(h - 1, int(math.ceil(box[3])))
    if x1 < x0 or y1 < y0:
        return
    ys = torch.arange(y0, y1 + 1, dtype=torch.float32)[:, None]
    xs = torch.arange(x0, x1 + 1, dtype=torch.float32)[None, :]
    region = canvas[:, y0 : y1 + 1, x0 : x1 + 1]
    region[:, inside(xs, ys)] = torch.tensor(colour, dtype=torch.float32)[:, None]


def _ellipse(canvas, p, q, half: float, colour) -> None:
    mx, my = int((float(p[0]) + float(q[0])) / 2), int((float(p[1]) + float(q[1])) / 2)
    dx, dy = float(q[0] - p[0]), float(q[1] - p[1])
    a = max(int(math.hypot(dx, dy) / 2), 0.5)
    c, s = math.cos(math.atan2(dy, dx)), math.sin(math.atan2(dy, dx))
    reach = a + half + 1

    def inside(xs, ys):
        u, v = (xs - mx) * c + (ys - my) * s, (ys - my) * c - (xs - mx) * s
        return (u / a) ** 2 + (v / half) ** 2 <= 1.0

    _stamp(canvas, (mx - reach, my - reach, mx + reach, my + reach), inside, colour)


def _dot(canvas, p, radius: float, colour) -> None:
    cx, cy = int(float(p[0])), int(float(p[1]))
    _stamp(canvas, (cx - radius - 1, cy - radius - 1, cx + radius + 1, cy + radius + 1),
           lambda xs, ys: (xs - cx) ** 2 + (ys - cy) ** 2 <= (radius + 0.35) ** 2, colour)


def _line(canvas, p, q, width: float, colour) -> None:
    ax, ay, bx, by = int(float(p[0])), int(float(p[1])), int(float(q[0])), int(float(q[1]))
    half = max(width / 2, 0.5)
    length2 = max(1e-6, (bx - ax) ** 2 + (by - ay) ** 2)

    def inside(xs, ys):
        t = (((xs - ax) * (bx - ax) + (ys - ay) * (by - ay)) / length2).clamp(0, 1)
        return (xs - ax - t * (bx - ax)) ** 2 + (ys - ay - t * (by - ay)) ** 2 <= half * half

    _stamp(canvas, (min(ax, bx) - half - 1, min(ay, by) - half - 1, max(ax, bx) + half + 1,
                    max(ay, by) + half + 1), inside, colour)


def draw_pose(keypoints: torch.Tensor, frame_size: tuple, size: tuple | None = None,
              threshold: float = 0.5) -> torch.Tensor:
    """The pose video Wan-Animate is driven by, ``(T, 3, h, w)`` sRGB: every frame's points (``(T, K, 3)`` x, y in
    pixels of a frame of ``frame_size = (H, W)`` and a score; 17 COCO points or 133 COCO-WholeBody ones) drawn as
    its preprocessing draws them: the limbs as ellipses in the OpenPose colours at 60 %, the joints as dots, the
    hands (whole-body points only) as thin lines in a rainbow, all on black, then fitted into ``size`` in
    proportion."""
    h, w = frame_size
    stick = max(int(min(h, w) / 200) - 1, 1)
    finger = max(stick // 2, 1)
    out, before = [], None
    for kp in torch.as_tensor(keypoints, dtype=torch.float32):
        body, left, right = _parts(kp)
        if before is not None and float(body[:, :2].min(dim=1).values.max()) < 0:
            body = before                                          # lost altogether: as it was
        before = body
        canvas = torch.zeros(3, h, w)
        for (i, j), colour in zip(_LIMBS, _COLOURS, strict=False):
            p, q = body[i - 1], body[j - 1]
            if p[2] >= threshold and q[2] >= threshold:
                _ellipse(canvas, p, q, stick, [int(c * 0.6) for c in colour])
        for p, colour in zip(body, _COLOURS, strict=True):
            if p[2] >= threshold:
                _dot(canvas, p, stick, colour)
        for hand in (left, right):
            if hand is None:
                continue
            for k, (i, j) in enumerate(_HAND):
                p, q = hand[i], hand[j]
                if p[2] >= threshold and q[2] >= threshold and min(float(p[0]), float(p[1]), float(q[0]),
                                                                     float(q[1])) > 0.01:
                    rgb = [round(c * 255) for c in colorsys.hsv_to_rgb(k / len(_HAND), 1.0, 1.0)]
                    _line(canvas, p, q, finger, rgb)
            for p in hand:
                if p[2] >= threshold and float(p[0]) > 0.01 and float(p[1]) > 0.01:
                    _dot(canvas, p, finger, (0, 0, 255))
        out.append(canvas / 255.0)
    video = torch.stack(out) if out else torch.zeros(0, 3, h, w)
    return _pad_resize(video, *size) if size is not None else video


def _face_box(points: torch.Tensor, h: int, w: int, scale: float = 1.3):
    """Wan-Animate's face box around ``points (N, 2)`` (pixels): the area grown by ``scale``, more above than
    below, as ``(x0, x1, y0, y1)``; None when there is no area."""
    lo, hi = points.min(0).values, points.max(0).values
    iw, ih = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    if iw < 1 or ih < 1:
        return None
    area = iw * ih * scale
    dw, dh = (math.sqrt(area * iw / ih) - iw) / 2, (math.sqrt(area * ih / iw) - ih) / 4
    box = (int(max(float(lo[0]) - dw, 0)), int(min(float(hi[0]) + dw, w)), int(max(float(lo[1]) - 3 * dh, 0)),
           int(min(float(hi[1]) + dh, h)))
    return box if box[1] - box[0] >= 2 and box[3] - box[2] >= 2 else None


def _head(kp: torch.Tensor, least: float = 8.0, seen: float = 0.3):
    """For 17 COCO points: two corners of where the 68 points of the face would lie, worked out from the eyes,
    the ears and the nose, at least ``least`` pixels wide (or None when none of them is seen)."""
    nose, eyes, ears, shoulders = kp[0], kp[1:3], kp[3:5], kp[5:7]
    eye_ok, ear_ok = eyes[:, 2] >= seen, ears[:, 2] >= seen
    if not bool(eye_ok.any() or ear_ok.any() or nose[2] >= seen):
        return None
    head = kp[:5][kp[:5, 2] >= seen][:, :2]
    width = [least, 1.5 * float((head.max(0).values - head.min(0).values).max())]
    if bool(ear_ok.all()):
        width.append(0.9 * float((ears[0, :2] - ears[1, :2]).norm()))
    if bool(eye_ok.all()):
        width.append(float((eyes[0, :2] - eyes[1, :2]).norm()) / 0.42)
    if bool((shoulders[:, 2] >= seen).all()):                     # a head is about half the shoulders across
        width.append(0.45 * float((shoulders[0, :2] - shoulders[1, :2]).norm()))
    fw = max(width)
    x = float(eyes[eye_ok][:, 0].mean()) if bool(eye_ok.any()) else float(head[:, 0].mean())
    y = float(eyes[eye_ok][:, 1].mean()) if bool(eye_ok.any()) else float(head[:, 1].mean())
    return torch.tensor([[x - fw / 2, y - 0.25 * fw], [x + fw / 2, y + 0.85 * fw]])


def face_crops(frames: torch.Tensor, keypoints: torch.Tensor, size: int = 512) -> torch.Tensor:
    """The face video Wan-Animate reads the expression from: the face of every frame cut out (the box of its 68
    whole-body points grown by 1.3, or one worked out from the eyes, ears and nose for 17 COCO points) and scaled
    to ``size`` by ``size``. A frame where no face is seen takes the box of the nearest one that has one."""
    t, _, h, w = frames.shape
    kps = torch.as_tensor(keypoints, dtype=torch.float32)
    boxes = []
    for kp in kps:
        corners = kp[23:91, :2] if kp.shape[0] >= 133 else _head(kp, least=min(h, w) / 16)
        boxes.append(None if corners is None else _face_box(corners, h, w))
    known = [i for i, b in enumerate(boxes) if b is not None]
    side = min(h, w) // 3
    default = ((w - side) // 2, (w + side) // 2, h // 10, h // 10 + side)
    out = []
    for i, frame in enumerate(frames):
        box = boxes[i] if boxes[i] is not None else boxes[min(known, key=lambda j: abs(j - i))] if known else default
        x0, x1, y0, y1 = box
        out.append(F.interpolate(frame[None, :, y0:y1, x0:x1], size=(size, size), mode="bilinear",
                                 align_corners=False, antialias=True)[0].clamp(0, 1))
    return torch.stack(out) if out else torch.zeros(0, 3, size, size)


def _clip(source):
    """A driving video -> ``(T, 3, H, W)`` and its frames a second (None when not known)."""
    if isinstance(source, tuple):
        return _frames(source[0]), float(source[1])
    if isinstance(source, str) or hasattr(source, "__fspath__"):
        from .animate import read_video

        return read_video(source, max_side=1280)
    return _frames(source), None


def motion_videos(motion, size: tuple, fps: float = 30.0, frames: int | None = None, face: bool = True) -> dict:
    """
    What Wan-Animate needs from a driving video, as lists of PIL pictures: ``"pose"`` (the stick figure, ``size =
    (h, w)``) and ``"face"`` (512 by 512), ``fps`` frames a second, ``frames`` long (the whole video when None; a
    short one goes back and forth). ``motion`` is the driving video (``(T, 3, H, W)`` sRGB in 0..1, assumed to
    be at ``fps`` already, or ``(video, its fps)``, or a file), or a dict of what is ready: ``{"pose", "face"}``
    from Wan's own preprocessing, ``{"video", "keypoints"}`` with points found elsewhere (17 or 133 a frame),
    and ``"fps"``. ``"background"`` and ``"mask"`` (the white part is redrawn) are passed on for replacing a
    person in a video. Without ready points the person is found with :func:`find_pose`.
    """
    h, w = size
    given = dict(motion) if isinstance(motion, dict) else {"video": motion}
    rate = given.get("fps")
    videos = {}
    if "pose" in given:
        videos["pose"] = _pad_resize(_frames(given["pose"]), h, w)
        if face:
            if "face" not in given:
                raise ValueError("a ready pose video needs its face video too: motion={'pose': ..., 'face': ...}")
            videos["face"] = _frames(given["face"])
    else:
        video, known = _clip(given["video"])
        rate = rate or known
        points = given.get("keypoints")
        points = find_pose(video) if points is None else torch.as_tensor(points, dtype=torch.float32)
        videos["pose"] = draw_pose(points, tuple(video.shape[-2:]), (h, w))
        if face:
            videos["face"] = face_crops(video, points)
    for extra in ("background", "mask"):
        if given.get(extra) is not None:
            videos[extra] = _pad_resize(_frames(given[extra]), h, w)
    keep = _retime(len(videos["pose"]), rate, fps)
    n = frames or len(keep)
    keep = [keep[i] for i in (_reflect(len(keep), n) if n > len(keep) else range(n))]
    return {name: [to_pil(v[min(i, len(v) - 1)]) for i in keep] for name, v in videos.items()}


# ------------------------------------------------------------------------------------------ the animator --
def _shared(pipe, names) -> dict:
    parts = getattr(pipe, "components", None) or {}
    return {k: v for k, v in parts.items() if k in names and v is not None}


def _let_go() -> None:
    """Give back the memory of a pipeline nothing refers to any more."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Animator:
    """A video pipeline of ``diffusers`` as something the animator can ask: ``animator(words) -> frames``.

    ``pipe`` makes a clip from words, ``still`` from a picture (a pipeline that starts from it, or for AnimateDiff
    and ``held`` ones a video-to-video pipeline the picture is held still for). ``multiple`` is what the sides
    must divide by, ``step`` what the frame count less one must divide by (Wan: 4, LTX: 8), ``most`` the longest
    clip, ``fixed`` a model of one size only. ``load_still`` and ``load_words`` make a pipeline when it is first
    needed (``keep`` False: the other one is let go first, for models that do not both fit)."""

    def __init__(self, pipe, kind: str, name: str = "", steps: int = 20, guidance: float = 6.0, fps: int = 8,
                 frames: int = 16, size: tuple = (512, 512), still=None, *, multiple: int = 8, step: int = 1,
                 most: int | None = None, fixed: bool = False, extra: dict | None = None,
                 still_extra: dict | None = None, held: bool | None = None, load_still=None, load_words=None,
                 keep: bool = True, place=None, share=()) -> None:
        self.pipe, self.kind, self.name = pipe, kind, name
        self.steps, self.guidance, self.fps, self.frames, self.size = steps, guidance, fps, frames, size
        self.still = still               # for a character given as a picture: the pipeline that starts from one
        self.multiple, self.step, self.most, self.fixed = multiple, step, most, fixed
        self.extra, self.still_extra = dict(extra or {}), dict(still_extra or {})
        self.held = kind == "animatediff" if held is None else held
        self.load_still, self.load_words, self.keep, self.share = load_still, load_words, keep, share
        self.place = place or (lambda p: p)
        self._active = pipe

    # ------------------------------------------------------------------------------------- sizes --
    def fit(self, size: tuple) -> tuple:
        """The size ``(h, w)`` the model makes nearest to ``size``."""
        if self.fixed:
            return tuple(self.size)
        m = self.multiple
        return tuple(max(m, int(s / m + 0.5) * m) for s in size)            # the nearest, halves up

    def count(self, frames: int) -> int:
        """The number of frames the model makes for ``frames`` asked: at least as many, ``step * k + 1``."""
        n = max(1, int(frames))
        n = min(n, self.most) if self.most else n
        return 1 + self.step * max(1, math.ceil((n - 1) / self.step)) if self.step > 1 else n

    # ------------------------------------------------------------------------------------ pipelines --
    def _switch(self, build, other: str):
        donor = getattr(self, other)
        parts = _shared(donor, self.share)
        if not self.keep and donor is not None:                 # the two do not fit: let the other one go first
            if hasattr(donor, "remove_all_hooks"):
                donor.remove_all_hooks()
            setattr(self, other, None)
            self._active = donor = None
            _let_go()
        return build(donor, parts)

    def _picture(self):
        if self.still is None and self.load_still is not None:
            self.still = self._switch(self.load_still, "pipe")
        return self.still

    def _words(self):
        if self.pipe is None and self.load_words is not None:
            self.pipe = self._switch(self.load_words, "still")
        return self.pipe

    def _use(self, pipe):
        """``pipe``, placed on the device if it was not the one used last (two pipelines that share their models
        take turns with the offload hooks)."""
        if pipe is not None and pipe is not self._active:
            pipe = self.place(pipe)
            self._active = pipe
        return pipe

    # -------------------------------------------------------------------------------------- asking --
    @torch.no_grad()
    def __call__(self, words: str, frames: int | None = None, size: tuple | None = None, seed: int = 0,
                 character=None, negative: str = NOT_THIS, strength: float = 0.75, motion=None) -> torch.Tensor:
        """The animation ``words`` describe as ``(T, 3, H, W)`` sRGB frames. ``character`` (a picture) is how the
        character looks: the clip starts from it, so it keeps that look (VACE takes it as a reference picture
        instead). ``motion`` (Wan-Animate and VACE) is a video of someone moving: the character performs that
        motion (see :func:`motion_videos`). ``frames`` is rounded up to a count the model makes and the clip is
        cut back to it; ``size`` is rounded to the sides it makes."""
        h, w = self.fit(size or self.size)
        common = {"prompt": words, "negative_prompt": negative, "num_inference_steps": self.steps,
                  "guidance_scale": self.guidance, "generator": torch.Generator().manual_seed(seed), **self.extra}
        if self.kind == "wan-animate":
            return self._perform(common, character, motion, frames, h * w)
        if motion is not None and self.kind != "vace":
            raise ValueError(f"motion= needs wan-animate or wan-vace; {self.name or self.kind} only takes words "
                             "and a picture")
        if self.kind == "vace":
            return self._vace(common, character, motion, frames, h, w)
        want = min(frames or self.frames, self.most or 10**9)
        n = self.count(want)
        if character is not None:
            look = _fit_picture(load_image(character, max_side=max(h, w)), h, w)
            still = self._picture()
            if self.held and still is not None:                    # from the picture, held still, set moving
                zoom = [F.interpolate(look[None], scale_factor=1.0 + 0.04 * i / max(1, n - 1), mode="bilinear",
                                      align_corners=False)[0] for i in range(n)]
                video = [to_pil(z[:, (z.shape[1] - h) // 2 : (z.shape[1] - h) // 2 + h,
                                  (z.shape[2] - w) // 2 : (z.shape[2] - w) // 2 + w]) for z in zoom]
                out = _ask(self._use(still), video=video, strength=strength, height=h, width=w, **common)
                return _frames(out)[:want]
            if still is not None:
                call = {**common, **self.still_extra}
                out = _ask(self._use(still), image=to_pil(look), num_frames=n, height=h, width=w, **call)
                return _frames(out)[:want]
            if self.kind == "other" and _takes(self.pipe, "image"):
                out = _ask(self._use(self.pipe), image=to_pil(look), num_frames=n, height=h, width=w, **common)
                return _frames(out)[:want]
        if self.kind == "wan-i2v":
            raise ValueError(f"{self.name} starts from a picture: give character= (or use wan-14b for words)")
        out = _ask(self._use(self._words()), num_frames=n, height=h, width=w, **common)
        return _frames(out)[:want]

    def _vace(self, common: dict, character, motion, frames, h: int, w: int) -> torch.Tensor:
        call = {}
        if character is not None:                                  # VACE lays it on white at the clip's size
            call["reference_images"] = [to_pil(load_image(character, max_side=max(h, w)))]
        want = frames or self.frames
        if motion is not None:
            given = dict(motion) if isinstance(motion, dict) else {"video": motion}
            if "control" in given:
                given["pose"] = given.pop("control")
            control = motion_videos(given, (h, w), fps=self.fps, face=False)["pose"]
            want = frames or len(control)
            n = self.count(want)
            call["video"] = [control[i] for i in (_reflect(len(control), n) if n > len(control) else range(n))]
        n = self.count(want)
        out = _ask(self._use(self._words()), num_frames=n, height=h, width=w, **common, **call)
        return _frames(out)[:want]

    def _perform(self, common: dict, character, motion, frames, area: int) -> torch.Tensor:
        if character is None or motion is None:
            raise ValueError("wan-animate makes a character perform a motion: give character= (a picture of it) "
                             "and motion= (a video of someone moving, see motion_videos)")
        look = load_image(character)
        h, w = _by_area(look.shape[-2:], area, self.multiple)      # the clip takes the picture's shape
        got = motion_videos(motion, (h, w), fps=self.fps, frames=frames)
        call = {"image": to_pil(resize(look, (h, w))), "pose_video": got["pose"], "face_video": got["face"],
                "height": h, "width": w, "mode": "animate"}
        if "background" in got and "mask" in got:
            call.update(background_video=got["background"], mask_video=got["mask"], mode="replace")
        out = _ask(self._use(self._words()), **common, **call)
        return _frames(out)[: len(got["pose"])]


# ------------------------------------------------------------------------------------------- loading --
def _vram_gb(device) -> float:
    return torch.cuda.get_device_properties(device).total_memory / 1e9


def _auto(device) -> str:
    """The model for this machine: Wan 2.2 A14B on a GPU of 80 GB or more, TI2V 5B from 40 GB, else AnimateLCM."""
    if device.type != "cuda":
        return "animatelcm"
    gb = _vram_gb(device)
    return "wan-14b" if gb >= 80 else "wan" if gb >= 40 else "animatelcm"


def _plan(known: dict, device, offload) -> str:
    """How the model goes on the device: ``"none"`` (all of it on the GPU), ``"model"`` (one part at a time) or
    ``"sequential"`` (one layer at a time), by the card's size unless ``offload`` says."""
    if offload is True:
        return "model"
    if offload is False:
        return "none"
    if offload not in (None, "auto"):
        if offload not in ("none", "model", "sequential"):
            raise ValueError(f"offload is None, 'none', 'model' or 'sequential', not {offload!r}")
        return offload
    if device.type != "cuda":
        return "none"
    gb = _vram_gb(device)
    if gb >= known["weights"] + known["work"] + 2:
        return "none"
    return "model" if gb >= known["largest"] + known["work"] + 2 else "sequential"


def _placer(device, how: str):
    def place(pipe):
        if device.type == "cuda" and how == "model":
            pipe.enable_model_cpu_offload(device=device)
        elif device.type == "cuda" and how == "sequential":
            pipe.enable_sequential_cpu_offload(device=device)
        else:
            pipe = pipe.to(device)
        vae = getattr(pipe, "vae", None)
        if device.type == "cuda" and vae is not None:                # a long clip decodes in tiles
            for switch in ("enable_tiling", "enable_slicing"):
                if hasattr(vae, switch):
                    getattr(vae, switch)()
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        return pipe

    return place


def _builder(cls_name: str, repo: str, dtype, wan_vae: bool, shift, lora, options: dict, same: bool = False):
    """Makes the pipeline ``cls_name``: from the donor pipeline's own models when ``same`` (one set of weights,
    another pipeline), else from ``repo`` with the donor's shared parts (text encoder, VAE) handed over."""

    def build(donor, parts):
        import diffusers

        cls = getattr(diffusers, cls_name)
        if same and donor is not None:     # from_pipe, not the bare parts: it carries Wan 2.2's expand_timesteps
            return cls.from_pipe(donor)
        parts = dict(parts)
        if wan_vae and "vae" not in parts:                        # Wan's VAE is kept in float32
            parts["vae"] = diffusers.AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32)
        pipe = cls.from_pretrained(repo, torch_dtype=dtype, **parts, **options)
        if shift is not None and getattr(pipe, "scheduler", None) is not None:
            pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config, flow_shift=shift)
        if lora:
            pipe.load_lora_weights(lora)
        return pipe

    return build


def _big(name: str, known: dict, device, lora, offload, options: dict) -> Animator:
    cuda = device.type == "cuda"
    dtype = torch.bfloat16 if cuda else torch.float32
    how = _plan(known, device, offload)
    place = _placer(device, how)
    wan_vae = known["pipeline"].startswith(("Wan", "SkyReels"))
    words = _builder(known["pipeline"], known["repo"], dtype, wan_vae, known.get("shift"), lora, options,
                     same=bool(known.get("still")) and not known.get("still_repo") and known["still"] != "same")
    pipe = place(words(None, {}))
    still, load_still, keep = None, None, True
    if known.get("still") == "same":                               # one pipeline takes words and a picture
        still = pipe
    elif known.get("still"):
        # the picture model: the same weights in another pipeline, or a model of its own (its LoRA is not loaded:
        # a LoRA belongs to the model it was made for)
        separate = known.get("still_repo")
        load_still = _builder(known["still"], separate or known["repo"], dtype, wan_vae, known.get("still_shift"),
                              None, options, same=not separate)
        if separate and cuda:
            keep = how == "none" and _vram_gb(device) >= known["weights"] + known["still_gb"] + known["work"] + 2
    return Animator(pipe, known["kind"], known["repo"], known["steps"], known["guidance"], known["fps"],
                    known["frames"], known["size"], still, multiple=known.get("multiple", 16),
                    step=known.get("step", 1), most=known.get("most"), fixed=known.get("fixed", False),
                    extra=known.get("extra"), still_extra=known.get("still_extra"), held=known.get("held", False),
                    load_still=load_still, load_words=words, keep=keep, place=place, share=known.get("share", ()))


def _animatediff(name: str, known: dict, device, lora, base: str, lora_scale: float, options: dict) -> Animator:
    from diffusers import AnimateDiffPipeline, AnimateDiffVideoToVideoPipeline, MotionAdapter

    dtype = torch.float16 if device.type == "cuda" else torch.float32
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


def load_animator(name: str = "auto", device=None, lora: str | None = None, base: str = BASE,
                  lora_scale: float = 0.8, offload=None, **options) -> Animator:
    """
    The video model ``name`` (see the module's notes and :func:`list_models`), ready to animate. ``"auto"``
    picks by the GPU: 80 GB or more Wan 2.2 A14B (T2V for words, I2V when a character picture is given), 40 to
    80 GB Wan 2.2 TI2V 5B, else AnimateLCM. A repo id of a model in the list loads that model; any other repo id
    loads with ``DiffusionPipeline`` (20 steps, 16 frames, sides of 16: change ``.steps``, ``.frames``, ``.fps``
    on the animator to suit). ``offload`` is how the model goes on the GPU (None: by the card's size; ``"none"``,
    ``"model"``, ``"sequential"``). ``base`` is the Stable Diffusion 1.5 model the AnimateDiff motion modules run
    on (any fine-tune of it works, a cartoon one gives cleaner lines). ``lora`` is a LoRA to load: the
    fine-tune. Other keywords go to ``diffusers``' ``from_pretrained``.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    name = _auto(device) if name == "auto" else _BY_REPO.get(name, name)
    known = KNOWN.get(name)
    if known is None and "/" not in name and not os.path.isdir(name):
        raise ValueError(f"no video model {name!r}: one of auto, {', '.join(KNOWN)}, or a repo id on the Hub")
    if known is None:                                   # any video pipeline on the Hub
        from diffusers import DiffusionPipeline

        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        pipe = DiffusionPipeline.from_pretrained(name, torch_dtype=dtype, **options)
        if lora:
            pipe.load_lora_weights(lora)
        how = _plan({"weights": 0, "largest": 0, "work": 0}, device, offload) if offload is not None else "none"
        return Animator(_placer(device, how)(pipe), "other", name, multiple=16)
    if known["kind"] == "animatediff":
        return _animatediff(name, known, device, lora, base, lora_scale, options)
    return _big(name, known, device, lora, offload, options)


def list_models() -> list:
    """Every known model as a row: ``name``, ``repo``, ``licence``, ``download_gb`` (about), ``picture_gb`` (about,
    the further download of its picture model, loaded when a character picture is first given), ``vram_gb``
    (about, all on the GPU), ``offload_gb`` (about, one part at a time on it; slower), ``gpu`` (False: it also
    runs on a CPU), ``fps``, ``frames``, ``size`` and ``good_for``. Only AnimateLCM and AnimateDiff run without a
    GPU; the others would take hours or days a clip on a CPU, when they fit in its memory at all."""
    rows = []
    for name, known in KNOWN.items():
        rows.append({"name": name, "repo": known.get("repo") or known["adapter"], "licence": known["licence"],
                     "download_gb": known["download"], "picture_gb": known.get("still_download", 0),
                     "vram_gb": known["weights"] + known["work"], "offload_gb": known["largest"] + known["work"],
                     "gpu": known["gpu"], "fps": known["fps"], "frames": known["frames"], "size": known["size"],
                     "good_for": known["about"]})
    return rows


__all__ = ["DRAWABLE", "KNOWN", "Animator", "draw_pose", "face_crops", "find_pose", "list_models", "load_animator",
           "motion_videos"]
