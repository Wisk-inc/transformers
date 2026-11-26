# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The eye: a pretrained vision model that Pinxit is fine-tuned on top of.

The painter needs to know what is in the photo: what the subject is, where it ends, what matters. A model that
already sees is used for that and stays as it was trained; Pinxit adds small low-rank adapters (LoRA) to its
attention layers and its own stroke model on top, and only those are trained. So Pinxit is a fine-tune in the
usual sense: a base model, plus a small file of what was learned.

:func:`load_eye` wraps a model from the Hugging Face Hub. SigLIP 2 (``google/siglip2-base-patch16-256``,
Apache-2.0) is the default because it also reads text, which lets words be painted into a picture
(:mod:`helix_lm.paint.add`); DINOv2 (``facebook/dinov2-small``, Apache-2.0) is a smaller eye without text.
:class:`TinyEye` is an untrained stand-in for tests.
"""

from __future__ import annotations

import contextlib
import hashlib

import torch
import torch.nn.functional as F
from torch import nn

from .see import gaussian_blur

DEFAULT = "google/siglip2-base-patch16-256"
NORMS = {"siglip": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         "clip": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
         "dinov2": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))}


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank change: ``y = W x + (alpha / rank) B A x``."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.down = nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
        self.up = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank
        self.active = True                                   # False: the layer as it was before the adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.active:
            return out
        return out + F.linear(F.linear(x, self.down.to(x.dtype)), self.up.to(x.dtype)) * self.scale


def add_lora(module: nn.Module, rank: int = 8, alpha: float = 16.0,
             names=("q_proj", "k_proj", "v_proj", "out_proj", "query", "key", "value")) -> int:
    """Freeze ``module`` and give every attention projection in it (linear layers with these names) a LoRA.
    Returns how many layers got one."""
    for p in module.parameters():
        p.requires_grad_(False)
    count = 0
    for parent in list(module.modules()):
        # torch's own attention layer reads ``out_proj.weight`` itself and never calls the layer, so an adapter
        # there would do nothing at best (SigLIP's pooling head is built on one): it is left as it is
        if isinstance(parent, nn.MultiheadAttention):
            continue
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and name in names:
                setattr(parent, name, LoRALinear(child, rank, alpha))
                count += 1
    return count


def lora_state(module: nn.Module) -> dict:
    """Only what was learned: the low-rank matrices, by name."""
    return {k: v.detach().cpu() for k, v in module.state_dict().items() if k.endswith((".down", ".up"))}


def load_lora(module: nn.Module, state: dict) -> None:
    missing = module.load_state_dict(state, strict=False)
    wrong = list(missing.unexpected_keys)
    if wrong:
        raise KeyError(f"these adapter weights do not belong to this eye: {wrong[:3]}")


def lora_rank(state: dict) -> int:
    """The rank of saved adapters (0 for none)."""
    return next((int(v.shape[0]) for k, v in state.items() if k.endswith(".down")), 0)


class Eye(nn.Module):
    """What every eye gives: ``features(image)`` (a grid of feature vectors and one for the whole picture),
    ``saliency(image)``, and for an eye that reads text ``text(prompts)`` and ``score(images, text)``."""

    name = "eye"
    dim = 0
    has_text = False

    def features_many(self, images: list) -> dict:
        """Several pictures ``(3, H, W)`` (any sizes) at once: ``dense (K, dim, h, w)`` and ``pooled (K, dim)``."""
        raise NotImplementedError

    def features(self, image: torch.Tensor) -> dict:
        """One picture: ``dense (dim, h, w)``, a grid of feature vectors, and ``pooled (dim,)`` for the whole."""
        many = self.features_many([image])
        return {"dense": many["dense"][0], "pooled": many["pooled"][0]}

    @contextlib.contextmanager
    def as_pretrained(self):
        """Inside, the adapters are switched off and the eye is the model as its makers trained it. Words are
        judged this way: the adapters learn to describe a photo to the painter, and nothing in that training
        keeps pictures and words in step."""
        layers = [m for m in self.modules() if isinstance(m, LoRALinear)]
        before = [m.active for m in layers]
        for m in layers:
            m.active = False
        try:
            yield self
        finally:
            for m, was in zip(layers, before, strict=True):
                m.active = was

    @staticmethod
    def _parts(image: torch.Tensor, grid: int, window: int) -> tuple[list, list]:
        """The picture whole, then again with one part after another blurred away: the views, and for each
        blurred view the row and column of its part on a ``grid x grid`` board (parts are ``window`` squares
        wide and overlap)."""
        side = 32 * grid
        small = F.interpolate(image[None].float(), size=(side, side), mode="bilinear", align_corners=False,
                              antialias=True)[0]
        flat = 0.5 * gaussian_blur(small, side / 6.0) + 0.5 * small.mean((-1, -2), keepdim=True)
        cell = side // grid
        views, places = [small], []
        for y in range(grid - window + 1):
            for x in range(grid - window + 1):
                part = (slice(None), slice(y * cell, (y + window) * cell), slice(x * cell, (x + window) * cell))
                view = small.clone()
                view[part] = flat[part]
                views.append(view)
                places.append((y, x))
        return views, places

    @staticmethod
    def _board(places: list, values: torch.Tensor, grid: int, window: int) -> torch.Tensor:
        """What blurring each part changed, laid back on the board: every square gets the mean over the parts
        that covered it."""
        total, count = torch.zeros(grid, grid), torch.zeros(grid, grid)
        for (y, x), value in zip(places, values, strict=True):
            total[y : y + window, x : x + window] += value
            count[y : y + window, x : x + window] += 1.0
        return total / count.clamp(min=1.0)

    @torch.no_grad()
    def saliency(self, image: torch.Tensor, grid: int = 8, window: int = 3) -> torch.Tensor:
        """What the picture is about, as the eye sees it: ``(grid, grid)`` in 0..1, on the CPU, where the plan
        is made. Part by part the picture is blurred away and the eye is asked for its summary again; the more
        the summary changes, the more that part matters. (The patch features themselves do not say this: in
        SigLIP 2 a few stray patches stand out from the rest, not the subject.) The eye is asked as it came,
        without the adapters."""
        views, places = self._parts(image, grid, window)
        with self.as_pretrained():
            pooled = torch.cat([self.features_many(views[i : i + 16])["pooled"] for i in range(0, len(views), 16)])
        pooled = F.normalize(pooled.float(), dim=-1).cpu()
        out = self._board(places, 1.0 - (pooled[1:] * pooled[:1]).sum(-1), grid, window)
        return (out - out.min()) / (out.max() - out.min() + 1e-8)

    @torch.no_grad()
    def find(self, image: torch.Tensor, words: str, grid: int = 10, window: int = 3) -> torch.Tensor:
        """Where in the picture the thing the ``words`` name is: ``(grid, grid)`` in 0..1, on the CPU. Part by
        part the picture is blurred away and the eye is asked how well the words still fit it; where blurring
        hurts the fit most, the thing is. Needs an eye that reads text."""
        if not self.has_text:
            raise ValueError(f"{self.name} does not read text; use a SigLIP or CLIP eye to find things by words")
        said = self.text([words])[0]
        views, places = self._parts(image, grid, window)
        fit = torch.cat([self.score(torch.stack(views[i : i + 16]), said) for i in range(0, len(views), 16)])
        fit = fit.float().cpu()
        out = self._board(places, (fit[:1] - fit[1:]).clamp(min=0.0), grid, window)
        return out / out.max().clamp(min=1e-8)

    def text(self, prompts: list) -> torch.Tensor:
        raise NotImplementedError("this eye does not read text")

    def score(self, images: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("this eye does not read text")


class TinyEye(Eye):
    """An untrained eye for tests: a small fixed network, and words turned into vectors by hashing."""

    name = "tiny"
    has_text = True

    def __init__(self, dim: int = 24, seed: int = 0) -> None:
        super().__init__()
        self.dim = dim
        g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(nn.Conv2d(3, dim, 8, stride=8), nn.GELU(), nn.Conv2d(dim, dim, 1))
        for p in self.net.parameters():
            p.data = torch.randn(p.shape, generator=g) * 0.3

    def features_many(self, images: list) -> dict:
        device = next(self.parameters()).device
        x = torch.cat([F.interpolate(image[None].to(device), size=(64, 64), mode="bilinear", align_corners=False)
                       for image in images])
        dense = self.net(x)
        return {"dense": dense, "pooled": dense.mean((-1, -2))}

    def text(self, prompts: list) -> torch.Tensor:
        out = []
        for prompt in prompts:
            vec = torch.zeros(self.dim)
            for word in str(prompt).lower().split():
                h = int(hashlib.sha1(word.encode()).hexdigest()[:8], 16)
                vec[h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
            out.append(vec)
        return F.normalize(torch.stack(out), dim=-1).to(next(self.parameters()).device)

    def score(self, images: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(images.to(next(self.parameters()).device), size=(64, 64), mode="bilinear",
                          align_corners=False)
        return (F.normalize(self.net(x).mean((-1, -2)), dim=-1) * text).sum(-1)


class HubEye(Eye):
    """A vision model from the Hub (SigLIP, SigLIP 2 at a fixed size, CLIP or DINOv2), as loaded by
    ``transformers.AutoModel``; pass a ``model`` already built to wrap it without a download."""

    def __init__(self, name: str = DEFAULT, model=None, tokenizer=None, device=None) -> None:
        super().__init__()
        if model is None:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(name)
        self.name, self.model = name, model.eval()
        kind = model.config.model_type
        self.kind = "siglip" if "siglip" in kind else ("clip" if "clip" in kind else "dinov2")
        vision = getattr(model.config, "vision_config", model.config)
        self.size, self.patch = int(vision.image_size), int(vision.patch_size)
        self.dim = int(vision.hidden_size)
        self.skip = 0 if self.kind == "siglip" else 1 + int(getattr(model.config, "num_register_tokens", 0) or 0)
        self.has_text = self.kind in ("siglip", "clip")
        self.tokenizer = tokenizer
        mean, std = NORMS[self.kind]
        self.register_buffer("mean", torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer("std", torch.tensor(std)[None, :, None, None], persistent=False)
        for p in self.model.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.to(device)

    def _pixels(self, images: torch.Tensor) -> torch.Tensor:
        x = images.to(self.mean.device).float()
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False, antialias=True)
        return ((x - self.mean) / self.std).to(next(self.model.parameters()).dtype)

    @property
    def vision(self) -> nn.Module:
        """The part that looks at pictures: the only part given adapters."""
        return getattr(self.model, "vision_model", self.model)

    def features_many(self, images: list) -> dict:
        pixels = torch.cat([self._pixels(image[None]) for image in images])
        if self.kind == "dinov2":
            tokens = self.model(pixel_values=pixels).last_hidden_state
            pooled = tokens[:, 0]
        else:
            out = self.model.vision_model(pixel_values=pixels)
            tokens = out.last_hidden_state
            pooled = out.pooler_output if out.pooler_output is not None else tokens.mean(1)
        side = self.size // self.patch
        dense = tokens[:, self.skip : self.skip + side * side].transpose(1, 2).reshape(len(images), -1, side, side)
        return {"dense": dense.float(), "pooled": pooled.float()}

    @torch.no_grad()
    def text(self, prompts: list) -> torch.Tensor:
        if not self.has_text:
            raise NotImplementedError(f"{self.name} does not read text; use a SigLIP or CLIP eye")
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        tokens = self.tokenizer(list(prompts), padding="max_length", truncation=True, max_length=64,
                                return_tensors="pt").to(self.mean.device)
        feats = self.model.get_text_features(**{k: v for k, v in tokens.items() if k in ("input_ids",
                                                                                         "attention_mask")})
        feats = getattr(feats, "pooler_output", feats)
        return F.normalize(feats.float(), dim=-1)

    def score(self, images: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        with self.as_pretrained():
            feats = self.model.get_image_features(pixel_values=self._pixels(images))
        feats = getattr(feats, "pooler_output", feats)
        return (F.normalize(feats.float(), dim=-1) * text).sum(-1)


def eye_width(name: str) -> int:
    """How wide the features of the eye ``name`` are (0 for no eye), without loading its weights."""
    if not name or name == "none":
        return 0
    if name == "tiny":
        return TinyEye().dim
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(name)
    return int(getattr(config, "vision_config", config).hidden_size)


def load_eye(name: str = DEFAULT, device=None, lora: int = 0) -> Eye:
    """The eye ``name`` (``"tiny"`` for the untrained test eye), with LoRA of rank ``lora`` added to the part
    that looks at pictures when not 0."""
    eye = TinyEye() if name == "tiny" else HubEye(name)
    if lora and name != "tiny":
        add_lora(eye.vision, lora)
    return eye.to(device) if device is not None else eye


def eye_of(state: dict, device=None) -> Eye | None:
    """The eye a checkpoint was trained with, its adapters loaded; None when it was trained without one."""
    name = state.get("eye")
    if not name:
        return None
    saved = state.get("eye_lora") or {}
    eye = load_eye(name, device, lora=lora_rank(saved))
    if saved:
        load_lora(eye, saved)
    return eye


__all__ = ["DEFAULT", "Eye", "HubEye", "LoRALinear", "TinyEye", "add_lora", "eye_of", "eye_width", "load_eye",
           "load_lora", "lora_rank", "lora_state"]
