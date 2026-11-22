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
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .brushes import BRUSHES
from .color import (
    colour_statistics,
    extract_palette,
    image_to_oklab,
    linear_to_oklab,
    oklab_to_linear,
    srgb_to_linear,
)
from .emotion import EMOTIONS, FLOWS, read_emotion
from .io import load_image, resize
from .see import gaussian_blur, orientation_field, slope

QUANTILES = 17


@dataclass
class StyleCard:
    name: str = ""
    palette: list = field(default_factory=list)        # sRGB colours, dark to light
    shares: list = field(default_factory=list)         # how much of the picture each takes
    l_quantiles: list = field(default_factory=list)    # lightness at 17 even steps of the distribution
    ab_mean: list = field(default_factory=lambda: [0.0, 0.0])
    ab_std: list = field(default_factory=lambda: [0.05, 0.05])
    lab_mean: list = field(default_factory=list)       # the mean colour in OKLab, and
    lab_cov: list = field(default_factory=list)        # how its colours spread (3 x 3): which hues go with which tones
    ground: list = field(default_factory=lambda: [0.45, 0.40, 0.33])     # sRGB tone of the bare canvas
    length: float = 1.0
    width: float = 1.0
    bend: float = 0.3
    jitter: float = 0.15
    opacity: float = 0.92
    hardness: float = 1.0                               # 0: no edge is hard (sfumato); 1: every edge is crisp
    texture: float = 0.3                                # how much the paint itself shows
    outline: float = 0.0                                # how much the shapes are drawn round with dark lines
    flat: float = 0.0                                   # how much of the picture is even colour (a print, a poster)
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

    def softness(self) -> float:
        """How much softer than usual every brush's edge is made (1: as the brushes are)."""
        return 1.0 + 2.5 * (1.0 - self.hardness) ** 2 if self.measured else 1.0

    def marks(self) -> float:
        """How much the picture is built from separate marks of paint all over (Van Gogh, Monet): its paint
        shows, and it is not made of even areas (Hokusai's print has detail at the size of a mark too, in its
        lines, but between them the colour lies flat)."""
        return self.texture * (1.0 - max(0.0, min(1.0, (self.flat - 0.08) / 0.10))) if self.measured else 0.0

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
                             self.texture, self.outline, self.coherence, math.cos(2 * self.flow_angle),
                             math.sin(2 * self.flow_angle), *self.ab_mean, *self.ab_std, float(self.measured)])
        brushes = torch.tensor(self.brushes) / max(sum(self.brushes), 1e-6)
        return torch.cat([nums, q[:: 4], flow, feel, brushes, top.flatten()]).float()


STYLE_DIM = 16 + 5 + len(FLOWS) + len(EMOTIONS) + len(BRUSHES) + 12


def _stats(image: torch.Tensor) -> dict:
    """What can be measured of a painting's handling at 384 pixels: its bands of detail, how hard its edges
    are, whether its shapes are outlined, how much its paint shows and which way its marks run."""
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
    # edges: in the middle of the picture's strong edges, how much steeper is the edge seen sharply than seen
    # blurred? A clean cut is about two and a half times steeper, an edge smoked away (sfumato) hardly at all
    fine, coarse = slope(gaussian_blur(light, 1.0)), slope(gaussian_blur(light, 3.0))
    middle = coarse >= 0.98 * F.max_pool2d(coarse[None, None], 5, 1, 2)[0, 0]
    strong = coarse >= torch.quantile(coarse.flatten()[:: max(1, coarse.numel() // 40000)], 0.9)
    pick = middle & strong
    crisp = float((fine[pick] / (coarse[pick] + 1e-6)).median()) if bool(pick.any()) else 2.45
    # outlines: thin marks darker than what lies three pixels away on both sides of them, in any of four
    # directions. (The edge of a dark shape is darker than one side only, and does not count.)
    near = gaussian_blur(light, 0.8)
    lines = torch.zeros_like(near)
    for dy, dx in ((0, 3), (3, 0), (2, 2), (2, -2)):
        ahead = torch.roll(near, (dy, dx), (0, 1)) - near
        behind = torch.roll(near, (-dy, -dx), (0, 1)) - near
        lines = torch.maximum(lines, torch.minimum(ahead, behind).clamp(min=0.0))
    lines = lines[4:-4, 4:-4]
    outline = float(torch.quantile(lines.flatten()[:: max(1, lines.numel() // 40000)], 0.97)) / (
        float(light.std()) + 1e-6)
    # the paint itself: detail at the size of a brush mark, against the picture's whole range of light and dark
    paint = (bands[1] + bands[2]) / (float(light.std()) + 1e-6)
    # even areas: the share of the picture where light and dark hardly change over a few pixels
    spread = (gaussian_blur(light * light, 3.0) - gaussian_blur(light, 3.0) ** 2).clamp(min=0.0).sqrt()
    return {"lab": lab, "bands": bands, "centroid": sum(i * b for i, b in enumerate(bands)) / total,
            "coherence": float((field_["coherence"] * field_["energy"]).sum() / field_["energy"].sum()),
            "hist": hist, "crisp": crisp, "outline": outline, "paint": paint,
            "flat": float((spread < 0.012).float().mean())}


def read_style(source, name: str = "", colors: int = 12, seed: int = 0) -> StyleCard:
    """Measure a painting (a path, a PIL image or a ``(3, H, W)`` sRGB tensor): its palette and how much of the
    picture each pigment takes, its lights and darks, how hard its edges are, whether its shapes are outlined,
    how much the paint shows, how long and wide its marks are and which way they run, the brushes that would
    make such marks, the tone of its ground, and the feeling of its colours."""
    image = load_image(source, max_side=384)
    s = _stats(image)
    lab = s["lab"]
    palette, shares = extract_palette(image, colors, seed=seed)
    q = torch.quantile(lab[0].flatten()[:: max(1, lab[0].numel() // 40000)], torch.linspace(0, 1, QUANTILES))

    def unit(value: float, low: float, high: float) -> float:
        return max(0.0, min(1.0, (value - low) / (high - low)))

    # the ranges come from fourteen paintings and from plain shapes blurred by known amounts: a clean cut
    # measures 2.47 and the same cut blurred by three pixels 1.43; the Mona Lisa is the softest painting (1.84)
    # and the smoothest (0.10), Monet's Water Lilies the crispest (2.53), Van Gogh's Bedroom the most outlined
    # (1.01, Hokusai's Great Wave 0.67, the Mona Lisa 0.18) and the most visibly painted (0.31). Dark dashes
    # between lighter ones also count as lines, so Monet's Water Lilies reads as somewhat outlined too
    hardness = max(0.1, unit(s["crisp"], 1.75, 2.45))
    texture = unit(s["paint"], 0.08, 0.30)
    outline = unit(s["outline"], 0.5, 1.0)
    coherence = max(0.0, min(1.0, s["coherence"]))
    peak, where = float(s["hist"].max()), int(s["hist"].argmax())
    angle = (where + 0.5) / 12.0 * math.pi
    if peak > 0.2:
        flow = "horizontal" if min(angle, math.pi - angle) < 0.3 else ("vertical" if abs(angle - math.pi / 2) < 0.3
                                                                       else "directional")
    else:
        flow = "swirl" if (coherence > 0.45 and texture > 0.4) else "follow"
    # flat, round, filbert, dry, knife, wash
    weights = [hardness * (0.4 + texture), 0.3, 0.7, texture * coherence * 1.2,
               max(0.0, hardness - 0.6) * texture * 2.0, (1.0 - hardness) * (1.0 - texture) * 2.5]
    # the ground: the tone a painter would have laid first, a muted colour a little under the middle lightness
    target = float(q[5])
    lab_palette = linear_to_oklab(srgb_to_linear(palette))
    pick = ((lab_palette[:, 0] - target).abs() - 0.5 * shares).argmin()
    ground = lab_palette[pick] * torch.tensor([1.0, 0.6, 0.6])
    ground_srgb = (oklab_to_linear(ground).clamp(0, 1) ** (1 / 2.2)).tolist()
    mean, cov = colour_statistics(lab)
    card = StyleCard(
        name=name, palette=palette.tolist(), shares=shares.tolist(), l_quantiles=q.tolist(),
        ab_mean=[float(lab[1].mean()), float(lab[2].mean())], ab_std=[float(lab[1].std()), float(lab[2].std())],
        lab_mean=mean.tolist(), lab_cov=cov.tolist(),
        ground=ground_srgb, length=0.7 + 0.9 * coherence, width=0.7 + 0.3 * s["centroid"],
        bend=max(0.05, min(0.8, 0.25 + 0.5 * coherence * (1.0 - min(1.0, 3.0 * peak)))),
        jitter=0.5 * (1.0 - coherence),
        # soft and smooth means thin layers one over another (glazes); visible paint is laid opaque
        opacity=0.55 + 0.4 * max(hardness, texture),
        hardness=hardness, texture=texture, outline=outline, flat=s["flat"], coherence=coherence, flow=flow,
        flow_angle=angle, brushes=weights, measured=True)
    card.emotion = read_emotion(image, texture)
    return card


STYLES = Path(__file__).with_name("styles")


def _words(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def style_names() -> dict:
    """The paintings whose measured style comes with the package: ``{name: "Title, painter, year"}``. Only the
    measurements are kept (palette, tones, edges, brushwork), not the pictures."""
    out = {}
    for path in sorted(STYLES.glob("*.json")):
        data = json.loads(path.read_text())
        out[path.stem.replace("_", " ")] = f"{data['title']}, {data['artist']}, {data['year']}"
    return out


def named_style(name: str) -> StyleCard | None:
    """The style of a famous painting by name: ``"mona lisa"``, ``"the starry night"``, or a painter's name
    (``"vermeer"``). None when no painting of that name comes with the package (see :func:`style_names`)."""
    asked = _words(name)
    if not asked:
        return None
    loose = None
    for path in sorted(STYLES.glob("*.json")):
        data = json.loads(path.read_text())
        known = [_words(k) for k in (path.stem, data["title"], data["artist"], *data.get("aliases", []))]
        if asked in known:
            return StyleCard.from_json(data)
        if loose is None and any(set(asked.split()) <= set(k.split()) for k in known):
            loose = data
    return StyleCard.from_json(loose) if loose is not None else None


def own_card(image: torch.Tensor, colors: int = 12, seed: int = 0) -> StyleCard:
    """The card of a photo painted in its own colours with ordinary brushwork (when no style is given)."""
    palette, shares = extract_palette(resize(image, (min(256, image.shape[-2]), min(256, image.shape[-1]))),
                                      colors, seed=seed)
    lab = image_to_oklab(image)
    q = torch.quantile(lab[0].flatten()[:: max(1, lab[0].numel() // 40000)], torch.linspace(0, 1, QUANTILES))
    return StyleCard(name="own colours", palette=palette.tolist(), shares=shares.tolist(), l_quantiles=q.tolist(),
                     ab_mean=[float(lab[1].mean()), float(lab[2].mean())],
                     ab_std=[float(lab[1].std()), float(lab[2].std())], measured=False)


__all__ = ["QUANTILES", "STYLE_DIM", "StyleCard", "named_style", "own_card", "read_style", "style_names"]
