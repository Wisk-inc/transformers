# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
GPU-only failures, caught on a CPU.

Tensors that would be on the GPU (a model's weights, its inputs) are tagged and the tag follows every op; a tensor
made with ``device=`` or moved with ``.to`` goes with either side. An op that mixes a tagged tensor with an
untagged one of more than one element would raise on a GPU ("Expected all tensors to be on the same device"), and
is reported with the line that made it. In 2.4.0 the prepare stage's breath check, the glitch detector, three
checks of the fail-safe loop, a low-pass and the converter all did this: none had ever run on a GPU.
"""

import traceback

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_flatten, tree_map

SR = 24000
ALLOWED = {torch.Tensor.__getitem__, torch.Tensor.__setitem__, torch.Tensor.copy_, torch.Tensor.to,
           torch.Tensor.index_put_, torch.Tensor.index_select, torch.index_select, torch.Tensor.type_as}


# calls that are about the tensor itself, not its values: they must reach the tensor, not a tagged copy of it
ITSELF = {torch.Tensor.requires_grad_, torch.Tensor.retain_grad, torch.Tensor.grad.__get__,
          torch.Tensor.grad.__set__, torch.Tensor.requires_grad.__set__, torch.Tensor.is_leaf.__get__,
          torch.Tensor.retains_grad.__get__}


class OnGpu(torch.Tensor):
    pass


class Placed(torch.Tensor):
    pass


def _gpu(x):
    return x.as_subclass(OnGpu) if isinstance(x, torch.Tensor) and not isinstance(x, OnGpu) else x


def _placed(x):
    return x.as_subclass(Placed) if isinstance(x, torch.Tensor) and not isinstance(x, (OnGpu, Placed)) else x


def _plain(x):
    return x.as_subclass(torch.Tensor) if isinstance(x, (OnGpu, Placed)) else x


class DeviceCheck(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.found = {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in ITSELF:
            with torch._C.DisableTorchFunctionSubclass():
                out = func(*args, **kwargs)
                if func == torch.Tensor.grad.__get__ and out is not None:   # a gradient lives where its tensor does
                    out = out.as_subclass(type(args[0]))
                return out
        tensors = [t for t in tree_flatten((args, kwargs))[0] if isinstance(t, torch.Tensor)]
        on_gpu = any(isinstance(t, OnGpu) for t in tensors)
        cpu_only = [t for t in tensors if not isinstance(t, (OnGpu, Placed)) and t.numel() > 1]
        if on_gpu and cpu_only and func not in ALLOWED:
            frame = next((f for f in reversed(traceback.extract_stack()[:-1]) if "helix_lm" in f.filename), None)
            where = f"{frame.filename.split('helix_lm/')[-1]}:{frame.lineno}" if frame else "?"
            self.found.setdefault(where, getattr(func, "__name__", str(func)))
        with torch._C.DisableTorchFunctionSubclass():
            out = func(*tree_map(_plain, args), **tree_map(_plain, kwargs))
        placed = "device" in kwargs or (func is torch.Tensor.to and len(args) + len(kwargs) > 1)
        if func is torch.Tensor.cpu:                             # brought back to the CPU: no longer tagged
            return out
        if on_gpu:
            return tree_map(_gpu, out)
        if placed or (tensors and not cpu_only and any(isinstance(t, Placed) for t in tensors)):
            return tree_map(_placed, out)
        return out


def _on_gpu(module):
    for mod in module.modules():
        for name, p in list(mod._parameters.items()):
            if p is not None:
                mod._parameters[name] = torch.nn.Parameter(p.data.as_subclass(OnGpu), requires_grad=p.requires_grad)
        for name, b in list(mod._buffers.items()):
            if b is not None:
                mod._buffers[name] = b.as_subclass(OnGpu)
    return module


@pytest.fixture(scope="module")
def tiny():
    from helix_lm.audio import AudioCodec, CodecConfig, SpeakerEncoder
    from helix_lm.audio.refiner import HarmonicNoiseRefiner, RefinerConfig
    from helix_lm.bench.data.synthetic import render
    from helix_lm.music.tristream import TriStreamConfig, TriStreamForSinging

    torch.manual_seed(0)
    cc = CodecConfig(dim=32, channels=8, num_codebooks=2, codebook_size=64)
    config = TriStreamConfig.named("ts-tiny", frame_dim=cc.dim, sample_rate=SR, semantic_codebook=cc.codebook_size,
                                   frame_rate=cc.frame_rate)
    return {
        "codec": _on_gpu(AudioCodec(cc)), "model": _on_gpu(TriStreamForSinging(config)),
        "speaker": _on_gpu(SpeakerEncoder(frame_dim=cc.dim, embedding_dim=config.speaker_dim)),
        "refiner": _on_gpu(HarmonicNoiseRefiner(RefinerConfig(sample_rate=SR, frame_dim=cc.dim,
                                                              frame_rate=cc.frame_rate))),
        "voice": render([(0.1, 1.4, 57), (1.5, 2.9, 64)], SR, "a", "mid", 3.0)[0],
    }


def test_the_prepare_stage_measures_on_the_gpu(tiny):
    from helix_lm.audio.singing import estimate_f0
    from helix_lm.music.regularize import breath_events, f0_range, is_dry_enough

    wave = tiny["voice"].clone().as_subclass(OnGpu)
    with DeviceCheck() as check:
        is_dry_enough(wave, SR)
        f0_range(wave, SR)
        breath_events(wave, SR)
        estimate_f0(wave[None], SR, frame_rate=25)
    assert not check.found, check.found


def test_the_eval_stage_runs_the_failsafe_loop_on_the_gpu(tiny):
    from helix_lm.audio.singing import estimate_f0
    from helix_lm.music.failsafe import Gates, synthesize_with_gates
    from helix_lm.music.tristream import noise_features, source_features

    codec, model, spk = tiny["codec"], tiny["model"], tiny["speaker"]
    wave = tiny["voice"].clone().as_subclass(OnGpu)
    with DeviceCheck() as check, torch.no_grad():
        frames = codec.listen(wave[None])
        f0 = estimate_f0(wave[None].as_subclass(torch.Tensor), SR, 25)[:, : frames.shape[1]].as_subclass(OnGpu)
        batch = {"source": source_features(f0, frames.pow(2).mean(-1)), "noise": noise_features(frames),
                 "frames": frames, "phonemes": torch.full((1, 1), -1, dtype=torch.long).as_subclass(OnGpu),
                 "speaker": spk(frames)}
        synthesize_with_gates(model.eval(), codec, batch, SR, f0[0], refiner=tiny["refiner"], source_wave=wave,
                              gates=Gates(steps=2, min_similarity=None, frame_rate=25.0))
    assert not check.found, check.found


def test_the_converter_takes_audio_from_the_cpu_to_a_gpu_model(tiny):
    from helix_lm.audio.singing import estimate_f0
    from helix_lm.music.conversion import Converter

    conv = Converter(tiny["model"].eval(), tiny["codec"], tiny["speaker"])
    voice = tiny["voice"]
    with DeviceCheck() as check, torch.no_grad():
        speaker = conv.embed(voice, SR)
        out, _ = conv.convert(voice, estimate_f0(voice[None], SR, 25)[0], speaker, SR, steps=2, reference=voice)
    assert not check.found, check.found
    assert out.shape == voice.shape


def test_sampling_draws_its_noise_where_the_generator_lives(tiny, monkeypatch):
    model = tiny["model"]
    calls = []
    real = torch.randn

    def spy(*args, **kwargs):
        calls.append((kwargs.get("device"), kwargs.get("generator")))
        return real(*args, **kwargs)

    from helix_lm.music.tristream import noise_features, source_features

    codec = tiny["codec"]
    with torch.no_grad():
        frames = codec.listen(tiny["voice"][None]).as_subclass(torch.Tensor)
        f0 = torch.full((1, frames.shape[1]), 220.0)
        batch = {"source": source_features(f0, frames.pow(2).mean(-1)), "noise": noise_features(frames),
                 "frames": frames, "phonemes": torch.full((1, 1), -1, dtype=torch.long), "speaker": torch.zeros(1, 16)}
        g = torch.Generator(device="cpu").manual_seed(3)
        monkeypatch.setattr(torch, "randn", spy)
        model.eval().sample(batch, steps=1, generator=g)
    drawn = [(d, gen) for d, gen in calls if gen is g]
    assert drawn and all(torch.device(d) == g.device for d, _ in drawn)
