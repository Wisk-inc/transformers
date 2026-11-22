# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Style: what can be read off a painting and carried to another picture.

:func:`read_style` measures a painting and returns a :class:`StyleCard`: its palette, how its lights and darks
are spread, how strong and how warm its colours are, the tone of its ground, and its brushwork (how long, wide,
bent and scattered the strokes are, how hard their edges, how much texture, whether they all run one way), with
the feeling :func:`helix_lm.paint.emotion.read_emotion` finds in it. These are plain image statistics, not a
trained model's features; the trained model takes the card as a vector (:meth:`StyleCard.vector`) beside what
its eye sees in the painting.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .brushes import BRUSHES
from .color import extract_palette, image_to_oklab, linear_to_oklab, oklab_to_linear, srgb_to_linear
from .emotion import EMOTIONS, FLOWS, read_emotion
from .io import load_image, resize
from .see import gaussian_blur, orientation_field

QUANTILES = 17


@dataclass
class StyleCard:
    name: str = ""
    palette: list = field(default_factory=list)        # sRGB colours, dark to light
    shares: list = field(default_factory=list)         # how much of the picture each takes
    l_quantiles: list = field(default_factory=list)    # lightness at 17 even steps of the distribution
    ab_mean: list = field(default_factory=lambda: [0.0, 0.0])
    ab_std: list = field(default_factory=lambda: [0.05, 0.05])
    ground: list = field(default_factory=lambda: [0.45, 0.40, 0.33])     # sRGB tone of the bare canvas
    length: float = 1.0
    width: float = 1.0
    bend: float = 0.3
    jitter: float = 0.15
    opacity: float = 0.92
    hardness: float = 0.7
    texture: float = 0.3
    coherence: float = 0.4
    flow: str = "follow"
    flow_angle: float = 0.0
    brushes: list = field(default_factory=lambda: [1.0, 1.0, 1.2, 0.5, 0.3, 0.5])
    emotion: dict = field(default_factory=dict)
    measured: bool = False                              # False: defaults, not read from a painting

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> StyleCard:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_json(), indent=1))
        return path

    @classmethod
    def load(cls, path) -> StyleCard:
        return cls.from_json(json.loads(Path(path).read_text()))

    def palette_linear(self, device=None) -> torch.Tensor:
        return srgb_to_linear(torch.tensor(self.palette, dtype=torch.float32, device=device))

    def vector(self) -> torch.Tensor:
        """The card as ``STYLE_DIM`` numbers for the model: brushwork, colour statistics, the feeling, and the
        palette's four heaviest colours in OKLab."""
        lab = linear_to_oklab(self.palette_linear()) if self.palette else torch.zeros(0, 3)
        order = torch.tensor(self.shares).argsort(descending=True)[:4] if self.shares else torch.zeros(0).long()
        top = torch.zeros(4, 3)
        top[: len(order)] = lab[order]
        q = torch.tensor(self.l_quantiles if self.l_quantiles else [0.5] * QUANTILES)
        flow = torch.zeros(len(FLOWS))
        flow[FLOWS.index(self.flow) if self.flow in FLOWS else 0] = 1.0
        feel = torch.tensor([self.emotion.get(e, 0.0) for e in EMOTIONS])
        nums = torch.tensor([self.length, self.width, self.bend, self.jitter, self.opacity, self.hardness,
                             self.texture, self.coherence, math.cos(2 * self.flow_angle),
                             math.sin(2 * self.flow_angle), *self.ab_mean, *self.ab_std, float(self.measured)])
        brushes = torch.tensor(self.brushes) / max(sum(self.brushes), 1e-6)
        return torch.cat([nums, q[:: 4], flow, feel, brushes, top.flatten()]).float()


STYLE_DIM = 15 + 5 + len(FLOWS) + len(EMOTIONS) + len(BRUSHES) + 12


def _stats(image: torch.Tensor) -> dict:
    lab = image_to_oklab(image)
    light = lab[0]
    bands, prev = [], light
    for sigma in (1.0, 2.0, 4.0, 8.0):
        low = gaussian_blur(light, sigma)
        bands.append(float((prev - low).abs().mean()))
        prev = low
    total = sum(bands) + 1e-9
    field_ = orientation_field(image, 1.5, 3.0)
    weight = field_["energy"] * field_["coherence"]
    angle = torch.remainder(0.5 * torch.atan2(field_["sin2"], field_["cos2"]), math.pi)
    hist = torch.zeros(12).index_add_(0, (angle / math.pi * 12).long().clamp(0, 11).flatten(), weight.flatten())
    hist = hist / hist.sum().clamp(min=1e-9)
    return {"lab": lab, "bands": bands, "fine": bands[0] / total, "centroid": sum(i * b for i, b in enumerate(bands))
            / total, "coherence": float((field_["coherence"] * field_["energy"]).sum() / field_["energy"].sum()),
            "hist": hist, "sharp": bands[0] / (bands[1] + 1e-9)}


def read_style(source, name: str = "", colors: int = 12, seed: int = 0) -> StyleCard:
    """Measure a painting (a path, a PIL image or a ``(3, H, W)`` sRGB tensor)."""
    image = load_image(source, max_side=384)
    s = _stats(image)
    lab = s["lab"]
    palette, shares = extract_palette(image, colors, seed=seed)
    q = torch.quantile(lab[0].flatten()[:: max(1, lab[0].numel() // 40000)], torch.linspace(0, 1, QUANTILES))
    texture = max(0.0, min(1.0, (s["fine"] - 0.12) / 0.30))
    coherence = max(0.0, min(1.0, s["coherence"]))
    hardness = max(0.2, min(1.0, s["sharp"]))
    peak, where = float(s["hist"].max()), int(s["hist"].argmax())
    angle = (where + 0.5) / 12.0 * math.pi
    if peak > 0.2:
        flow = "horizontal" if min(angle, math.pi - angle) < 0.3 else ("vertical" if abs(angle - math.pi / 2) < 0.3
                                                                       else "directional")
    else:
        flow = "swirl" if coherence > 0.45 else "follow"
    weights = [hardness * (0.5 + texture), (1 - coherence) * 0.8, 0.7, texture * coherence * 1.2,
               max(0.0, hardness - 0.6) * texture * 2.0, (1 - hardness) * (1 - texture) * 1.5]
    # the ground: the tone a painter would have laid first, a muted colour a little under the middle lightness
    target = float(q[5])
    lab_palette = linear_to_oklab(srgb_to_linear(palette))
    pick = ((lab_palette[:, 0] - target).abs() - 0.5 * shares).argmin()
    ground = lab_palette[pick] * torch.tensor([1.0, 0.6, 0.6])
    ground_srgb = (oklab_to_linear(ground).clamp(0, 1) ** (1 / 2.2)).tolist()
    card = StyleCard(
        name=name, palette=palette.tolist(), shares=shares.tolist(), l_quantiles=q.tolist(),
        ab_mean=[float(lab[1].mean()), float(lab[2].mean())], ab_std=[float(lab[1].std()), float(lab[2].std())],
        ground=ground_srgb, length=0.7 + 0.9 * coherence, width=0.7 + 0.3 * s["centroid"],
        bend=max(0.05, min(0.8, 0.25 + 0.5 * coherence * (1.0 - min(1.0, 3.0 * peak)))),
        jitter=0.5 * (1.0 - coherence), opacity=0.75 if (hardness < 0.4 and texture < 0.3) else 0.92,
        hardness=hardness, texture=texture, coherence=coherence, flow=flow, flow_angle=angle, brushes=weights,
        measured=True)
    card.emotion = read_emotion(image, texture)
    return card


def own_card(image: torch.Tensor, colors: int = 12, seed: int = 0) -> StyleCard:
    """The card of a photo painted in its own colours with ordinary brushwork (when no style is given)."""
    palette, shares = extract_palette(resize(image, (min(256, image.shape[-2]), min(256, image.shape[-1]))),
                                      colors, seed=seed)
    lab = image_to_oklab(image)
    q = torch.quantile(lab[0].flatten()[:: max(1, lab[0].numel() // 40000)], torch.linspace(0, 1, QUANTILES))
    return StyleCard(name="own colours", palette=palette.tolist(), shares=shares.tolist(), l_quantiles=q.tolist(),
                     ab_mean=[float(lab[1].mean()), float(lab[2].mean())],
                     ab_std=[float(lab[1].std()), float(lab[2].std())], measured=False)


__all__ = ["QUANTILES", "STYLE_DIM", "StyleCard", "own_card", "read_style"]
