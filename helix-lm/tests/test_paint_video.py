# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The video models without a GPU, the network or ``diffusers``: stand-in pipelines take its place, and every model
is checked for what it is asked with, how it is put on the GPU, how it switches to its picture model, and how
the clip comes back. The large models themselves were not run here."""

import sys
import types

import numpy as np
import pytest
import torch
from PIL import Image

from helix_lm.paint import video
from helix_lm.paint.video import (
    KNOWN,
    Animator,
    _auto,
    _frames,
    _retime,
    draw_pose,
    face_crops,
    list_models,
    load_animator,
    motion_videos,
)


# ------------------------------------------------------------------------------------- stand-ins --
class Out:
    def __init__(self, frames):
        self.frames = frames


class Scheduler:
    def __init__(self, config=None):
        self.config = dict(config or {"flow_shift": 1.0})

    @classmethod
    def from_config(cls, config, **changes):
        return cls({**config, **changes})


class Vae:
    def __init__(self, dtype=None):
        self.dtype, self.tiling, self.slicing = dtype, False, False

    def enable_tiling(self):
        self.tiling = True

    def enable_slicing(self):
        self.slicing = True


class Pipe:
    """A stand-in for a diffusers pipeline: it notes how it was made, put on the device and asked."""

    made: list = []

    def __init__(self, repo="", parts=None):
        self.repo, self.parts, self.asked = repo, dict(parts or {}), []
        self.placed = self.lora = self.dtype = None
        self.scheduler = Scheduler()
        self.vae = self.parts.get("vae") or Vae()
        self.text_encoder = self.parts.get("text_encoder") or object()
        self.tokenizer = self.parts.get("tokenizer") or object()
        self.unhooked = 0

    @property
    def components(self):
        return {"vae": self.vae, "text_encoder": self.text_encoder, "tokenizer": self.tokenizer,
                "scheduler": self.scheduler}

    @classmethod
    def from_pretrained(cls, repo, **kwargs):
        dtype = kwargs.pop("torch_dtype", None)
        pipe = cls(repo, kwargs)
        pipe.dtype = dtype
        Pipe.made.append((cls.__name__, repo, kwargs))
        return pipe

    @classmethod
    def from_pipe(cls, other, **kwargs):
        pipe = cls(other.repo, other.components)
        Pipe.made.append((cls.__name__, "from_pipe", {}))
        return pipe

    def to(self, device):
        self.placed = ("all", str(device))
        return self

    def enable_model_cpu_offload(self, gpu_id=None, device=None):
        self.placed = ("model", str(device))

    def enable_sequential_cpu_offload(self, gpu_id=None, device=None):
        self.placed = ("sequential", str(device))

    def remove_all_hooks(self):
        self.unhooked += 1

    def set_progress_bar_config(self, **kwargs):
        pass

    def load_lora_weights(self, lora, **kwargs):
        self.lora = lora

    def __call__(self, **kwargs):
        self.asked.append(kwargs)
        n = kwargs.get("num_frames") or len(kwargs.get("pose_video") or kwargs.get("video") or [None])
        h, w = kwargs.get("height") or 48, kwargs.get("width") or 64
        return Out(np.random.rand(1, n, h, w, 3).astype(np.float32))           # Wan gives a numpy array


class HunyuanVideo15Pipeline(Pipe):                    # no guidance_scale: its guider holds it
    def __call__(self, prompt=None, negative_prompt=None, height=None, width=None, num_frames=121,
                 num_inference_steps=50, generator=None):
        return super().__call__(prompt=prompt, negative_prompt=negative_prompt, height=height, width=width,
                                num_frames=num_frames, num_inference_steps=num_inference_steps, generator=generator)


class HunyuanVideo15ImageToVideoPipeline(Pipe):        # no height or width: they come from the picture
    def __call__(self, image, prompt=None, negative_prompt=None, num_frames=121, num_inference_steps=50,
                 generator=None):
        return super().__call__(image=image, prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames,
                                num_inference_steps=num_inference_steps, generator=generator)


class AutoencoderKLWan:
    @classmethod
    def from_pretrained(cls, repo, subfolder=None, torch_dtype=None):
        Pipe.made.append(("AutoencoderKLWan", repo, {"torch_dtype": torch_dtype}))
        return Vae(torch_dtype)


NAMES = ["WanPipeline", "WanImageToVideoPipeline", "WanVideoToVideoPipeline", "WanVACEPipeline", "WanAnimatePipeline",
         "LTXConditionPipeline", "LTX2Pipeline", "LTX2ImageToVideoPipeline", "CogVideoXPipeline",
         "CogVideoXImageToVideoPipeline", "SkyReelsV2Pipeline", "SkyReelsV2ImageToVideoPipeline", "DiffusionPipeline"]


@pytest.fixture
def diffusers(monkeypatch):
    module = types.ModuleType("diffusers")
    for name in NAMES:
        setattr(module, name, type(name, (Pipe,), {}))
    module.HunyuanVideo15Pipeline = HunyuanVideo15Pipeline
    module.HunyuanVideo15ImageToVideoPipeline = HunyuanVideo15ImageToVideoPipeline
    module.AutoencoderKLWan = AutoencoderKLWan
    monkeypatch.setitem(sys.modules, "diffusers", module)
    Pipe.made = []
    return module


def gpu(monkeypatch, gb: float) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device=None: types.SimpleNamespace(total_memory=gb * 1e9))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def made(cls_name):
    return [m for m in Pipe.made if m[0] == cls_name]


def person(points: int = 133) -> torch.Tensor:
    """A person facing us in a 200 by 200 frame: ``(points, 3)`` x, y, score (COCO or COCO-WholeBody)."""
    kp = torch.zeros(133, 3)
    body = [(100, 40), (106, 34), (94, 34), (112, 38), (88, 38), (120, 70), (80, 70), (130, 100), (70, 100),
            (135, 125), (65, 125), (112, 130), (88, 130), (114, 160), (86, 160), (115, 188), (85, 188),
            (120, 195), (118, 196), (112, 192), (80, 195), (82, 196), (88, 192)]
    kp[: len(body), :2] = torch.tensor(body, dtype=torch.float32)
    turn = torch.linspace(0, 2 * np.pi, 69)[:-1]
    kp[23:91, 0], kp[23:91, 1] = 100 + 14 * torch.cos(turn), 42 + 16 * torch.sin(turn)
    for start, (wx, wy), side in ((91, (135, 125), 1), (112, (65, 125), -1)):
        kp[start, :2] = torch.tensor([wx, wy], dtype=torch.float32)
        for finger in range(5):
            for joint in range(4):
                kp[start + 1 + 4 * finger + joint, :2] = torch.tensor(
                    [wx + side * (4 + 4 * joint), wy + 4 + 3 * finger + 2 * joint], dtype=torch.float32)
    kp[:, 2] = 1.0
    return kp[:points].clone()


# ------------------------------------------------------------------------------------- sizes, frames --
def test_frame_counts_and_sizes_are_rounded_to_what_each_model_makes():
    wan = Animator(None, "wan", multiple=16, step=4)
    assert [wan.count(n) for n in (1, 16, 17, 81, 82)] == [5, 17, 17, 81, 85]          # Wan: 4k + 1
    assert Animator(None, "ltx", step=8).count(16) == 17 and Animator(None, "ltx", step=8).count(121) == 121
    assert Animator(None, "cogvideox", step=4, most=49).count(81) == 49
    assert wan.fit((100, 200)) == (96, 208) and Animator(None, "ltx", multiple=32).fit((100, 200)) == (96, 192)
    assert Animator(None, "cogvideox", size=(480, 720), fixed=True).fit((512, 512)) == (480, 720)
    assert Animator(None, "animatediff").fit((64, 64)) == (64, 64)
    assert _retime(8, 15.0, 30.0) == [0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 7]       # 15 a second to 30
    assert _retime(5, None, 30.0) == list(range(5))


def test_every_output_form_becomes_frames_in_0_to_1():
    pil = [Image.fromarray((np.random.rand(32, 48, 3) * 255).astype(np.uint8)) for _ in range(3)]
    assert _frames(Out([pil])).shape == (3, 3, 32, 48) and _frames(Out(pil)).shape == (3, 3, 32, 48)
    assert _frames(np.random.rand(1, 5, 32, 48, 3)).shape == (5, 3, 32, 48)
    whole = _frames((np.random.rand(4, 16, 24, 3) * 255).astype(np.uint8))
    assert whole.shape == (4, 3, 16, 24) and float(whole.max()) <= 1.0 and float(whole.max()) > 0.5
    signed = torch.rand(1, 3, 7, 16, 24) * 2 - 1                              # (B, C, T, H, W) in -1..1
    got = _frames(signed)
    assert got.shape == (7, 3, 16, 24) and torch.allclose(got, ((signed[0].permute(1, 0, 2, 3) + 1) / 2))
    plain = torch.rand(6, 3, 16, 24)                                          # (T, C, H, W) in 0..1: as it is
    assert torch.equal(_frames(plain), plain)
    assert _frames(torch.rand(1, 6, 16, 24, 3)).shape == (6, 3, 16, 24)
    assert _frames(((torch.rand(2, 3, 8, 8),),)).shape == (2, 3, 8, 8)       # return_dict=False: a tuple
    assert _frames([torch.rand(3, 8, 8) for _ in range(4)]).shape == (4, 3, 8, 8)
    assert _frames([np.random.rand(3, 8, 8, 3)]).shape == (3, 3, 8, 8)       # a list of clips as arrays
    big = _frames(torch.rand(1, 2, 8, 8, 3) * 255)
    assert float(big.max()) <= 1.0 and float(big.max()) > 0.5


# ------------------------------------------------------------------------------------------ loading --
def test_auto_picks_the_model_by_the_size_of_the_gpu(monkeypatch, diffusers):
    assert _auto(torch.device("cpu")) == "animatelcm"
    for gb, name in ((96, "wan-14b"), (80, "wan-14b"), (48, "wan"), (40, "wan"), (24, "animatelcm")):
        gpu(monkeypatch, gb)
        assert _auto(torch.device("cuda")) == name
    gpu(monkeypatch, 96)
    animator = load_animator("auto")
    assert animator.name == "Wan-AI/Wan2.2-T2V-A14B-Diffusers" and animator.kind == "wan"
    assert animator.pipe.dtype == torch.bfloat16 and animator.pipe.placed == ("all", "cuda")   # 96 GB: all of it
    assert animator.pipe.vae.dtype == torch.float32 and animator.pipe.vae.tiling      # Wan's VAE, in tiles
    assert (animator.fps, animator.frames, animator.size) == (16, 81, (480, 832))
    gpu(monkeypatch, 48)
    assert load_animator("auto").name == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def test_memory_is_planned_by_the_size_of_the_gpu(monkeypatch, diffusers):
    for gb, how in ((96, "all"), (48, "model"), (24, "sequential")):
        gpu(monkeypatch, gb)
        assert load_animator("wan-14b").pipe.placed == (how, "cuda")
    gpu(monkeypatch, 96)
    assert load_animator("wan-14b", offload="model").pipe.placed == ("model", "cuda")
    assert load_animator("wan-animate").pipe.placed == ("all", "cuda")
    assert load_animator("ltx2").pipe.placed == ("all", "cuda")
    with pytest.raises(ValueError):
        load_animator("wan", offload="half")


def test_the_14b_wan_lets_its_words_model_go_for_its_picture_model_and_back(monkeypatch, diffusers):
    gpu(monkeypatch, 96)
    animator = load_animator("wan-14b", lora="me/a-style")
    words = animator.pipe
    assert words.lora == "me/a-style" and not animator.keep            # both together need about 140 GB
    clip = animator("a fox runs", frames=16, size=(100, 200), seed=3)
    asked = words.asked[-1]
    assert clip.shape == (16, 3, 96, 208) and asked["num_frames"] == 17
    assert (asked["height"], asked["width"]) == (96, 208)
    assert asked["guidance_scale"] == 4.0 and asked["guidance_scale_2"] == 3.0 and asked["num_inference_steps"] == 40
    clip = animator("a fox runs", frames=16, size=(100, 200), character=torch.rand(3, 50, 100))
    (_, repo, parts), = made("WanImageToVideoPipeline")
    picture = animator.still
    assert repo == "Wan-AI/Wan2.2-I2V-A14B-Diffusers" and animator.pipe is None and words.unhooked == 1
    assert parts["text_encoder"] is words.text_encoder and parts["vae"] is words.vae      # shared, not loaded again
    assert len(made("AutoencoderKLWan")) == 1 and picture.placed == ("all", "cuda") and picture.lora is None
    asked = picture.asked[-1]
    assert isinstance(asked["image"], Image.Image) and asked["image"].size == (208, 96)
    assert asked["guidance_scale"] == 3.5 and asked["guidance_scale_2"] == 3.5 and clip.shape[0] == 16
    animator("a fox sleeps")                                             # words again: the T2V comes back
    assert animator.still is None and len(made("WanPipeline")) == 2 and animator.pipe.lora == "me/a-style"
    assert animator.pipe.text_encoder is words.text_encoder


def test_each_model_is_asked_with_the_arguments_it_takes(monkeypatch, diffusers):
    gpu(monkeypatch, 96)
    look = torch.rand(3, 60, 80)

    wan = load_animator("wan")                                           # TI2V 5B: sides of 32, 24 a second
    wan("a fox", frames=10, size=(100, 200))
    assert wan.pipe.asked[-1]["num_frames"] == 13 and wan.pipe.asked[-1]["height"] == 96
    wan("a fox", frames=10, size=(100, 200), character=look)
    assert made("WanImageToVideoPipeline")[0][1] == "from_pipe" and wan.keep and wan.pipe is not None
    assert wan.still.asked[-1]["image"].size == (192, 96)

    small = load_animator("wan-1.3b")                                    # a picture held still, set moving
    small("a fox", frames=16, size=(64, 64), character=look, strength=0.6)
    held = small.still.asked[-1]
    assert made("WanVideoToVideoPipeline")[0][1] == "from_pipe" and len(held["video"]) == 17
    assert held["strength"] == 0.6 and held["video"][0].size == (64, 64)

    i2v = load_animator("wan-i2v-14b")
    with pytest.raises(ValueError, match="starts from a picture"):
        i2v("a fox")
    i2v("a fox", character=look)
    assert "image" in i2v.pipe.asked[-1] and i2v.pipe.asked[-1]["guidance_scale_2"] == 3.5

    vaes = len(made("AutoencoderKLWan"))
    ltx = load_animator("ltx")                                           # one pipeline for words and a picture
    ltx("a fox", frames=16, size=(100, 200))
    ltx("a fox", frames=16, size=(100, 200), character=look)
    first, second = ltx.pipe.asked[-2:]
    assert first["num_frames"] == 17 and (first["height"], first["width"]) == (96, 192)
    assert first["frame_rate"] == 24 and first["decode_timestep"] == 0.05 and "image" in second
    assert ltx.still is ltx.pipe and ltx.pipe.dtype == torch.bfloat16 and len(made("AutoencoderKLWan")) == vaes

    ltx2 = load_animator("ltx2")
    ltx2("a fox", character=look)
    assert made("LTX2ImageToVideoPipeline")[0][1] == "from_pipe" and ltx2.still.asked[-1]["num_frames"] == 121

    hunyuan = load_animator("hunyuan")
    hunyuan("a fox", frames=30)
    assert "guidance_scale" not in hunyuan.pipe.asked[-1] and hunyuan.pipe.asked[-1]["num_frames"] == 33
    hunyuan("a fox", frames=30, character=look)
    (_, repo, parts), = made("HunyuanVideo15ImageToVideoPipeline")
    assert repo.endswith("480p_i2v") and "text_encoder" in parts and "vae" not in parts        # its VAE differs
    assert "height" not in hunyuan.still.asked[-1] and isinstance(hunyuan.still.asked[-1]["image"], Image.Image)

    cog = load_animator("cogvideox")                                     # 720x480 only, 49 frames at most
    clip = cog("a fox", frames=81, size=(512, 512))
    asked = cog.pipe.asked[-1]
    assert (asked["height"], asked["width"], asked["num_frames"]) == (480, 720, 49) and asked["use_dynamic_cfg"]
    assert clip.shape == (49, 3, 480, 720) and cog.keep                  # both fit on 96 GB
    cog("a fox", character=look)
    assert cog.pipe is not None and cog.still.asked[-1]["image"].size == (720, 480)

    sky = load_animator("skyreels")
    assert sky.pipe.scheduler.config["flow_shift"] == 8.0                # its repo says 1.0, its card 8.0
    sky("a fox", character=look)
    assert sky.still.scheduler.config["flow_shift"] == 5.0 and sky.still.asked[-1]["guidance_scale"] == 5.0
    assert sky.pipe is None                                              # two 14B models do not both fit

    with pytest.raises(ValueError, match="motion"):
        ltx("a fox", motion=torch.rand(5, 3, 32, 32))


def test_vace_takes_the_character_as_a_reference_and_a_pose_video_as_the_motion(monkeypatch, diffusers):
    gpu(monkeypatch, 96)
    vace = load_animator("wan-vace-1.3b")
    vace("a fox dances", frames=16, size=(64, 96), character=torch.rand(3, 40, 30))
    asked = vace.pipe.asked[-1]
    assert len(asked["reference_images"]) == 1 and asked["reference_images"][0].size == (30, 40)
    assert "video" not in asked and asked["num_frames"] == 17
    clip = vace("a fox dances", size=(64, 96), motion={"control": torch.rand(10, 3, 40, 40)})
    asked = vace.pipe.asked[-1]
    assert asked["num_frames"] == 13 and len(asked["video"]) == 13 and asked["video"][0].size == (96, 64)
    assert clip.shape[0] == 10                                           # as long as the motion


def test_wan_animate_gets_the_pose_and_face_videos_of_the_motion(monkeypatch, diffusers):
    gpu(monkeypatch, 96)
    found = []

    def fake_find(frames):
        found.append(len(frames))
        return person(17)[None].expand(len(frames), -1, -1).clone()

    monkeypatch.setattr(video, "find_pose", fake_find)
    animate = load_animator("wan-animate")
    character = torch.rand(3, 300, 200)                                  # standing: taller than wide
    with pytest.raises(ValueError, match="perform"):
        animate("a fox", character=character)
    driving = torch.rand(8, 3, 200, 200)
    clip = animate("a fox dances", size=(64, 96), character=character, motion=driving)
    asked = animate.pipe.asked[-1]
    assert (asked["height"], asked["width"]) == (96, 64) and asked["image"].size == (64, 96)   # the picture's shape
    assert len(asked["pose_video"]) == 8 and asked["pose_video"][0].size == (64, 96)
    assert len(asked["face_video"]) == 8 and asked["face_video"][0].size == (512, 512)
    assert asked["mode"] == "animate" and asked["guidance_scale"] == 1.0 and asked["segment_frame_length"] == 77
    assert clip.shape[0] == 8 and found == [8] and animate.fps == 30
    animate("a fox dances", size=(64, 96), character=character, motion=(driving, 15.0))     # 15 a second to 30
    assert len(animate.pipe.asked[-1]["pose_video"]) == 16
    animate("a fox dances", frames=12, size=(64, 96), character=character, motion=driving)   # back and forth
    assert len(animate.pipe.asked[-1]["pose_video"]) == 12
    ready = {"pose": torch.rand(6, 3, 96, 64), "face": torch.rand(6, 3, 512, 512),
             "background": torch.rand(6, 3, 96, 64), "mask": torch.rand(6, 3, 96, 64)}
    animate("a fox", size=(64, 96), character=character, motion=ready)
    asked = animate.pipe.asked[-1]
    assert asked["mode"] == "replace" and len(asked["background_video"]) == 6 and len(asked["mask_video"]) == 6
    assert found == [8, 8, 8]                                            # ready videos: no pose was looked for


# -------------------------------------------------------------------------------- the pose, the face --
def test_the_pose_is_drawn_as_wan_animate_draws_it():
    whole = draw_pose(person(133)[None], (200, 200))
    assert whole.shape == (1, 3, 200, 200)
    assert torch.allclose(whole[0, :, 70, 90], torch.tensor([153, 0, 0]) / 255.0)  # neck to right shoulder, 60 %
    assert torch.allclose(whole[0, :, 40, 100], torch.tensor([1.0, 0.0, 0.0]))      # the nose, a red dot
    tip = person(133)[95]                                                # a finger of the left hand: blue dot
    assert torch.allclose(whole[0, :, int(tip[1]), int(tip[0])], torch.tensor([0.0, 0.0, 1.0]))
    assert float(whole[0, :, :20, :20].abs().sum()) == 0.0                         # black around
    body = draw_pose(person(17)[None], (200, 200))
    assert float(body[0, :, int(tip[1]), int(tip[0])].sum()) == 0.0                # 17 points: no hands
    assert torch.allclose(body[0, :, 70, 90], whole[0, :, 70, 90])
    wide = draw_pose(person(133)[None], (200, 200), (100, 300))         # fitted in proportion, on black
    assert wide.shape == (1, 3, 100, 300) and float(wide[0, :, :, :100].sum()) == 0.0 and float(wide.sum()) > 0
    lost = person(133)
    lost[:, 2] = 0.0
    assert float(draw_pose(lost[None], (200, 200)).sum()) == 0.0                  # nothing seen, nothing drawn


def test_the_face_is_cut_out_and_scaled_to_512():
    frames = torch.zeros(3, 3, 200, 200)
    yy, xx = torch.meshgrid(torch.arange(200.0), torch.arange(200.0), indexing="ij")
    frames[:, 0, ((xx - 100) ** 2 + (yy - 42) ** 2) < 14**2] = 1.0       # a red face
    for points in (133, 17):
        kps = person(points)[None].repeat(3, 1, 1)
        kps[1, :, 2] = 0.0                                                # nothing seen in the middle frame
        faces = face_crops(frames, kps)
        assert faces.shape == (3, 3, 512, 512)
        assert float(faces[0, 0, 256, 256]) > 0.9 and float(faces[0, 0].mean()) > 0.3
        assert float(faces[1, 0, 256, 256]) > 0.9                         # it took a neighbour's box
    ready = motion_videos({"pose": torch.rand(4, 3, 50, 50), "face": torch.rand(4, 3, 512, 512), "fps": 15.0},
                          (64, 96), fps=30.0)
    assert len(ready["pose"]) == 8 and ready["pose"][0].size == (96, 64) and len(ready["face"]) == 8


# -------------------------------------------------------------------------------- any model, the list --
def test_unknown_repo_ids_go_through_diffusion_pipeline(monkeypatch, diffusers):
    animator = load_animator("someone/a-video-model", lora="me/style")
    (cls, repo, _), = made("DiffusionPipeline")
    assert repo == "someone/a-video-model" and animator.kind == "other" and animator.pipe.lora == "me/style"
    assert animator.pipe.dtype == torch.float32 and animator.pipe.placed == ("all", "cpu")
    animator("a fox", size=(100, 100))
    assert animator.pipe.asked[-1]["num_frames"] == 16 and animator.pipe.asked[-1]["height"] == 96
    animator("a fox", character=torch.rand(3, 32, 32))
    assert isinstance(animator.pipe.asked[-1]["image"], Image.Image)
    gpu(monkeypatch, 96)
    assert load_animator("someone/another").pipe.dtype == torch.bfloat16
    named = load_animator("Skywork/SkyReels-V2-T2V-14B-540P-Diffusers")              # a known repo id: by name
    assert named.kind == "skyreels" and named.pipe.scheduler.config["flow_shift"] == 8.0


def test_the_list_of_models_says_what_each_is_and_needs():
    rows = {row["name"]: row for row in list_models()}
    assert set(rows) == set(KNOWN)
    for want in ("wan-14b", "wan-i2v-14b", "wan-animate", "wan-vace", "wan-vace-1.3b", "ltx", "hunyuan",
                 "cogvideox", "skyreels"):
        assert want in rows
    assert {name for name, row in rows.items() if not row["gpu"]} == {"animatelcm", "animatediff"}
    assert all(row["licence"] and row["repo"] and row["good_for"] for row in rows.values())
    assert all(row["vram_gb"] <= 94 for row in rows.values())          # every one fits whole on 96 GB
    assert all(row["offload_gb"] <= row["vram_gb"] for row in rows.values())
    assert "UK" in rows["hunyuan"]["licence"] and rows["wan-14b"]["picture_gb"] > 100
    assert rows["wan-animate"]["fps"] == 30 and rows["wan"]["repo"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
