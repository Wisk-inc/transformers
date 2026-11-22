# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Pinxit, the trained hand, built on the Helix trunk.

For every cell of a pass the model reads one sequence and writes its strokes at the end of it:

    brief | style | feeling | memory | eye | palette (one token a pigment) | view (8 x 8 tokens) | stroke slots

* the **brief** is what the mind decided about the cell: which pass this is, how much the place matters, how far
  it is to be finished, how many strokes it may get, which way loose strokes flow, how fast and how hard the
  hand should tend to move;
* **style** and **feeling** are the style card and the feeling as vectors;
* **memory** is what the model wrote down the last time it worked here (in the pass before, or before a change
  to a finished painting);
* the **eye** token is what a pretrained vision model sees at this place (:mod:`helix_lm.paint.eye`);
* the **view** is the cell's window of the plan and of the canvas as it is now.

The trunk is :class:`helix_lm.anansi.AnansiModel`, the looped Helix body TriStream also uses: it reads left to
right, so each stroke slot sees the whole brief and view and every slot before it, and decides one stroke: ten
numbers of gesture, a brush, two mixes of the palette's pigments, how much of the second shows, and whether to
make the stroke at all (stopping early is a decision the model makes, not a threshold applied afterwards). After
the last slot it writes the memory for the next visit. More trunk loops at painting time (``loops=``) buy more
thought per stroke.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..anansi import AnansiConfig, AnansiModel
from .brushes import BRUSHES
from .color import linear_to_oklab, linear_to_srgb, mix_pigments
from .emotion import Feeling
from .engine import BRIEF, View
from .eye import eye_of
from .strokes import GEOM, PHASES, Strokes
from .style import STYLE_DIM

FEEL = 17 + len(BRUSHES)


def feeling_vector(feeling: Feeling) -> torch.Tensor:
    """A :class:`Feeling` as ``FEEL`` numbers."""
    f = feeling
    return torch.tensor([f.saturation, 20 * f.warmth, 5 * f.key, f.contrast, f.glow, f.temperature, f.speed,
                         f.pressure, f.length, f.width, f.bend, f.jitter, f.restraint, 5 * f.magnify, 5 * f.stretch,
                         f.tilt / 4.0, f.masses, *f.brushes], dtype=torch.float32)


@dataclass
class PinxitConfig:
    name: str = "px-small"
    window: int = 32
    strokes: int = 14
    palette: int = 12
    hidden: int = 256
    heads: int = 8
    mlp: int = 1024
    prelude: int = 1
    core: int = 4
    coda: int = 1
    loops: float = 2.0
    streams: int = 2
    style_tokens: int = 2
    eye_dim: int = 0
    memory_dim: int = 64

    @classmethod
    def named(cls, size: str, **overrides) -> PinxitConfig:
        rows = {
            "px-tiny": {"hidden": 48, "heads": 4, "mlp": 96, "prelude": 1, "core": 1, "coda": 1, "loops": 1.0,
                        "streams": 1, "memory_dim": 16},
            "px-small": {"hidden": 256, "heads": 8, "mlp": 1024, "prelude": 1, "core": 4, "coda": 1, "loops": 2.0,
                         "streams": 2},
            "px-base": {"hidden": 448, "heads": 8, "mlp": 1792, "prelude": 2, "core": 6, "coda": 2, "loops": 2.0,
                        "streams": 2, "memory_dim": 96},
            "px-large": {"hidden": 768, "heads": 12, "mlp": 3072, "prelude": 2, "core": 8, "coda": 2, "loops": 3.0,
                         "streams": 4, "memory_dim": 128},
        }
        if size not in rows:
            raise KeyError(f"no size {size!r}; there are {', '.join(rows)}")
        return cls(name=size, **{**rows[size], **overrides})

    def trunk(self) -> AnansiConfig:
        return AnansiConfig(
            vocab_size=8, hidden_size=self.hidden, intermediate_size=self.mlp, num_attention_heads=self.heads,
            num_key_value_heads=max(1, self.heads // 2), head_dim=self.hidden // self.heads,
            num_recurrent_heads=max(1, self.heads // 2), prelude_layers=self.prelude, core_layers=self.core,
            coda_layers=self.coda, mean_loops=self.loops, min_loops=1, max_loops=max(2, math.ceil(self.loops) + 2),
            num_streams=self.streams, ngram_memory_size=0, mtp_depth=0, loss_chunk_tokens=0, block_size=16,
            local_blocks=8, index_branching=4, index_topk=4, landmark_dim=32, attention_backend="sdpa",
        )  # fmt: skip


class Pinxit(nn.Module):
    def __init__(self, config: PinxitConfig) -> None:
        super().__init__()
        self.config = c = config
        h = c.hidden
        self.brief = nn.Sequential(nn.Linear(BRIEF, h), nn.GELU(), nn.Linear(h, h))
        self.style = nn.Linear(STYLE_DIM, c.style_tokens * h)
        self.feeling = nn.Sequential(nn.Linear(FEEL, h), nn.GELU(), nn.Linear(h, h))
        self.memory_in = nn.Linear(c.memory_dim, h)
        self.no_memory = nn.Parameter(torch.zeros(h))
        self.eye_in = nn.Linear(c.eye_dim, h) if c.eye_dim else None
        self.no_eye = nn.Parameter(torch.zeros(h))
        self.pigment = nn.Sequential(nn.Linear(3, h), nn.GELU(), nn.Linear(h, h))
        self.view = nn.Sequential(nn.Conv2d(9, h // 4, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(h // 4, h // 2, 3, stride=2, padding=1), nn.GELU(),
                                  nn.Conv2d(h // 2, h, 3, stride=2, padding=1))
        self.view_place = nn.Parameter(torch.zeros((c.window // 4) ** 2, h))
        self.slots = nn.Parameter(torch.zeros(c.strokes, h))
        self.kind = nn.Parameter(torch.zeros(8, h))          # which part of the sequence a token belongs to
        self.trunk = AnansiModel(c.trunk())
        self.geom = nn.Linear(h, GEOM)
        self.brush = nn.Linear(h, len(BRUSHES))
        self.mix = nn.Linear(h, 2 * h)
        self.tint = nn.Linear(h, 6)
        self.streak = nn.Linear(h, 1)
        self.keep = nn.Linear(h, 1)
        self.memory_out = nn.Linear(h, c.memory_dim)
        for p in (self.view_place, self.slots, self.kind, self.no_memory, self.no_eye):
            nn.init.normal_(p, std=0.02)
        nn.init.constant_(self.keep.bias, 1.0)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, target: torch.Tensor, canvas: torch.Tensor, brief: torch.Tensor, palette: torch.Tensor,
                style: torch.Tensor, feeling: torch.Tensor, memory: torch.Tensor | None = None,
                eye: torch.Tensor | None = None, loops: int | None = None,
                has_memory: torch.Tensor | None = None) -> dict:
        """``target, canvas (A, 3, R, R)`` linear RGB; ``brief (A, BRIEF)``; ``palette (P, 3)`` or ``(A, P, 3)``
        linear RGB; ``style (STYLE_DIM,)`` or ``(A, STYLE_DIM)``; ``feeling (FEEL,)`` or ``(A, FEEL)``; ``memory
        (A, memory_dim)`` and ``eye (A, eye_dim)`` or None; ``has_memory (A,)`` bool says which cells of a mixed
        batch have a memory at all. Returns the strokes (``keep`` as a probability), the raw choices
        (``keep_logit``, ``brush_logit``, ``mix``) and ``memory (A, memory_dim)``."""
        c, a = self.config, target.shape[0]
        h = c.hidden
        dtype = self.geom.weight.dtype
        palette = palette.expand(a, -1, -1) if palette.dim() == 3 else palette[None].expand(a, -1, -1)
        t, v = linear_to_srgb(target.float()) * 2 - 1, linear_to_srgb(canvas.float()) * 2 - 1
        view = self.view(torch.cat([t, v, t - v], 1).to(dtype)).flatten(2).transpose(1, 2) + self.view_place
        remembered = self.no_memory.expand(a, h) if memory is None else self.memory_in(memory.to(dtype))
        if memory is not None and has_memory is not None:
            remembered = torch.where(has_memory[:, None], remembered, self.no_memory.to(remembered.dtype))
        seen = self.no_eye.expand(a, h) if (eye is None or self.eye_in is None) else self.eye_in(eye.to(dtype))
        parts = [self.brief(brief.to(dtype))[:, None],
                 self.style(style.to(dtype).expand(a, -1)).reshape(a, c.style_tokens, h),
                 self.feeling(feeling.to(dtype).expand(a, -1))[:, None], remembered[:, None], seen[:, None],
                 self.pigment(linear_to_oklab(palette.float()).to(dtype)), view, self.slots.expand(a, -1, -1)]
        tokens = torch.cat([p + self.kind[i] for i, p in enumerate(parts)], 1)
        hidden, _, _ = self.trunk(inputs_embeds=tokens, num_loops=loops)
        hidden = hidden.float()
        p_at = 1 + c.style_tokens + 3
        pigments = hidden[:, p_at : p_at + palette.shape[1]]
        out = hidden[:, -c.strokes :]
        ask = self.mix(out.to(dtype)).float().reshape(a, c.strokes, 2, h)
        mix = torch.einsum("ansh,aph->ansp", ask, pigments) / math.sqrt(h)          # (A, N, 2, P)
        weights = torch.softmax(mix, -1)
        tint = 0.04 * torch.tanh(self.tint(out.to(dtype)).float()).reshape(a, c.strokes, 2, 3)
        colour = (mix_pigments(palette[:, None, None].float(), weights) + tint).clamp(0.0, 1.0)
        keep_logit = self.keep(out.to(dtype)).float()[..., 0]
        brush_logit = self.brush(out.to(dtype)).float()
        strokes = Strokes(torch.sigmoid(self.geom(out.to(dtype)).float()), torch.softmax(brush_logit, -1),
                          colour[:, :, 0], colour[:, :, 1], torch.sigmoid(self.streak(out.to(dtype)).float())[..., 0],
                          torch.sigmoid(keep_logit), torch.zeros(a, c.strokes, PHASES, device=target.device))
        return {"strokes": strokes, "keep_logit": keep_logit, "brush_logit": brush_logit, "mix": mix,
                "memory": self.memory_out(out[:, -1].to(dtype)).float()}

    # ---------------------------------------------------------------------------------------- files --
    def save(self, path, **extra) -> None:
        torch.save({"config": asdict(self.config), "model": self.state_dict(), **extra}, path)

    @classmethod
    def from_state(cls, state: dict, device=None) -> Pinxit:
        """The model of a checkpoint, with the averaged weights when it has them (those are the ones to paint
        with)."""
        model = cls(PinxitConfig(**state["config"]))
        model.load_state_dict(state.get("ema") or state["model"])
        return model.to(device or "cpu").eval()

    @classmethod
    def load(cls, path, device=None) -> Pinxit:
        return cls.from_state(torch.load(path, map_location="cpu", weights_only=True), device)


class ModelHand:
    """The trained model as the engine's hand. It proposes all of a pass's strokes in one go; ``polish`` is the
    share of the untrained hand's refinement steps still run on them (0: the model's strokes as they are).
    ``eye`` gives it what a pretrained vision model sees at each cell."""

    name = "pinxit"
    refines = False

    def __init__(self, model: Pinxit, eye=None, polish: float = 0.25, loops: int | None = None) -> None:
        self.model, self.eye, self.steps_share, self.loops = model.eval(), eye, polish, loops
        self.remembered = None
        self._seen = None

    def look(self, image: torch.Tensor) -> None:
        """Let the eye see the picture once, before the passes."""
        self._seen = self.eye.features(image)["dense"] if self.eye is not None else None

    @torch.no_grad()
    def propose(self, view: View) -> Strokes:
        model, c = self.model, self.model.config
        device = next(model.parameters()).device
        seen = None
        if self._seen is not None and model.eye_in is not None:
            grid = torch.stack([(view.cx.float() + 0.5) / view.level.gx * 2 - 1,
                                (view.cy.float() + 0.5) / view.level.gy * 2 - 1], -1)[None, :, None].to(device)
            seen = F.grid_sample(self._seen[None].to(device).float(), grid, mode="bilinear", padding_mode="border",
                                 align_corners=False)[0, :, :, 0].T
        palette = view.plan.palette[: c.palette]
        if palette.shape[0] < c.palette:
            palette = torch.cat([palette, palette[-1:].expand(c.palette - palette.shape[0], 3)])
        memory = None if view.memory is None else view.memory.to(device)
        out = model(view.target.to(device), view.canvas.to(device), view.brief.to(device), palette.to(device),
                    view.plan.card.vector().to(device), feeling_vector(view.plan.feeling).to(device), memory, seen,
                    self.loops)
        self.remembered = out["memory"].cpu()
        s, n = out["strokes"], min(view.level.strokes, c.strokes)
        where = view.target.device
        allowed = torch.arange(n, device=where)[None] < view.counts[:, None]
        brush = F.one_hot(s.brush[:, :n].argmax(-1), len(BRUSHES)).float()
        phase = torch.rand(s.keep.shape[0], n, PHASES, generator=view.generator)
        return Strokes(s.geom[:, :n].to(where), brush.to(where), s.color[:, :n].to(where), s.color2[:, :n].to(where),
                       s.streak[:, :n].to(where), (allowed & (s.keep[:, :n].to(where) > 0.3)).float(), phase.to(where))


def load_hand(model, size: str = "px-base", polish: float = 0.25, device=None, loops: int | None = None):
    """The trained painter, ready for :func:`helix_lm.paint.paint`: ``(hand, eye)``. ``model`` is a checkpoint
    file, or a Hub repository that holds one for ``size``. The eye is the pretrained vision model the checkpoint
    was trained with, its adapters loaded (None for a model trained without an eye)::

        hand, eye = load_hand("CorxLabs/Pinxit-1")
        painting = paint("photo.jpg", style="a_painting.jpg", emotion="majesty", hand=hand, eye=eye)
    """
    from pathlib import Path

    path = Path(str(model))
    if not path.exists():
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(str(model), f"pinxit/{size}/latest.pt"))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(path, map_location="cpu", weights_only=True)
    net = Pinxit.from_state(state, device)
    eye = eye_of(state, device) if net.eye_in is not None else None
    return ModelHand(net, eye, polish=polish, loops=loops), eye


__all__ = ["FEEL", "ModelHand", "Pinxit", "PinxitConfig", "feeling_vector", "load_hand"]
