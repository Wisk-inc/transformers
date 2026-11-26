# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The painter's mind: everything decided before the first stroke.

A camera records what is there. A painter decides what the picture is about and changes the rest to serve it.
:func:`make_plan` does that in the order a painter would, and writes down each move in ``plan.report``:

1. **Compose.** Find what draws the eye. Try other framings (the subject on a third, a tighter crop) and keep
   the one the rules of composition like best; the photo's own framing wins when nothing is clearly better.
2. **Distort on purpose.** Enlarge the subject a little, make the picture taller or wider about it, lean it a
   few degrees: whatever the feeling asks for.
3. **Push the colour.** Carry the style's lights, darks and colours over; then warm the lights and cool the
   shadows, gather the values into a few masses, raise the contrast and the colour where the subject is and
   quiet them elsewhere, tint the shadows with the complement of the light, let the lights glow, and darken
   the corners so the eye stays in.
4. **Decide what to finish.** ``plan.finish`` says how far each part is taken. The subject is finished; the
   rest stops early, and the edges of the canvas may stay bare.
5. **Give loose strokes a direction.** Where the picture has no direction of its own, strokes flow toward the
   subject, away from the light, or as the style's strokes do.

Every move has a strength, and every strength can be set to 0. These are a painter's working rules written as
arithmetic, not a trained model: they are the experimental part of Pinxit, and the trained model learns its
strokes inside the plan they make.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .color import (
    extract_palette,
    image_to_oklab,
    linear_to_oklab,
    match_colours,
    oklab_to_image,
    srgb_to_linear,
)
from .emotion import Feeling
from .io import resize
from .see import focal_point, gaussian_blur, saliency, slope, value_masses
from .style import StyleCard

THIRDS = ((1 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1 / 3), (2 / 3, 2 / 3))


@dataclass
class Settings:
    work: int = 512                 # the longer side of the canvas the strokes are planned on, in pixels
    detail: int | None = None       # how many passes from broad to fine (None: as many as ``work`` needs)
    compose: float = 0.6            # 0 keeps the photo's framing
    exaggerate: float = 0.6         # 0 keeps the photo's shapes and colours
    abstraction: float = 0.6        # how far the photo is seen in planes and steps of tone; 0 keeps every detail
    restraint: float | None = None  # how much stays unfinished (None: what the feeling asks for)
    style_strength: float = 0.8     # how far lights, darks and colours move to the style's
    palette_strength: float = 0.7   # how strictly strokes use the palette's pigments
    budget: int = 3400              # the most strokes a painting may have
    colors: int = 12                # pigments on the palette
    steps: float = 1.0              # more (or fewer) refinement steps per pass; 0 lays the strokes as proposed
    weave: float = 0.6              # how much the canvas's cloth shows; 0 paints on a smooth board
    border: float = 0.025           # the bare edge of the canvas around a saved picture, as a share of its width
    seed: int = 0


@dataclass
class Plan:
    target: torch.Tensor            # (3, H, W) sRGB: what the painter aims at
    source: torch.Tensor            # (3, H, W) sRGB: the photo, framed the same way, before the colour was pushed
    focus: torch.Tensor             # (H, W) 0..1: where the picture is about
    finish: torch.Tensor            # (H, W) 0..1: how far each part is taken
    flow: torch.Tensor              # (3, H, W): cos and sin of twice the direction for loose strokes, and its weight
    focal: tuple                    # (y, x) in 0..1
    ground: torch.Tensor            # (3,) linear RGB: the tone of the bare canvas
    palette: torch.Tensor           # (P, 3) linear RGB
    feeling: Feeling
    card: StyleCard
    settings: Settings
    report: list = field(default_factory=list)

    @property
    def restraint(self) -> float:
        return self.feeling.restraint if self.settings.restraint is None else self.settings.restraint


# ------------------------------------------------------------------------------------------ compose --
def composition_score(sal: torch.Tensor, box: tuple) -> dict:
    """How the rules of composition judge a framing ``box = (y0, x0, height, width)`` (shares of the picture)
    of a saliency map: how near the subject is to a third, how much of what matters is kept, how the weight is
    balanced, how much is cut away, and whether the subject is crowded against an edge."""
    h, w = sal.shape
    y0, x0, bh, bw = box
    a, b = int(round(y0 * h)), int(round(x0 * w))
    crop = sal[a : max(a + 2, int(round((y0 + bh) * h))), b : max(b + 2, int(round((x0 + bw) * w)))]
    fy, fx = focal_point(crop)
    near = min(math.hypot(fy - py, fx - px) for py, px in THIRDS)
    kept = float(crop.sum() / sal.sum().clamp(min=1e-8))
    xs = (torch.arange(crop.shape[1], device=sal.device) + 0.5) / crop.shape[1]
    balance = 1.0 - 2.0 * abs(float((crop.sum(0) * xs).sum() / crop.sum().clamp(min=1e-8)) - 0.5)
    margin = min(fy, fx, 1.0 - fy, 1.0 - fx)
    total = -3.0 * near + 1.5 * kept + 0.4 * balance - 0.6 * (1.0 - bh * bw) - 5.0 * max(0.0, 0.12 - margin)
    return {"score": total, "thirds": near, "kept": kept, "balance": balance, "focal": (fy, fx)}


def compose(sal: torch.Tensor, strength: float) -> tuple[tuple, str]:
    """The framing to paint: ``(box, what was done)``. Framings a little tighter than the photo, with the subject
    on each of the four thirds, are scored against the photo's own; the best one is used, moved toward by
    ``strength``."""
    full = (0.0, 0.0, 1.0, 1.0)
    if strength <= 0:
        return full, "kept the photo's framing"
    small = F.interpolate(sal[None, None], size=(64, max(8, round(64 * sal.shape[1] / sal.shape[0]))),
                          mode="area")[0, 0]
    fy, fx = focal_point(small)
    best, best_score = full, composition_score(small, full)["score"] + 0.05
    for zoom in (0.92, 0.84, 0.76):
        for py, px in THIRDS:
            y0 = min(max(fy - py * zoom, 0.0), 1.0 - zoom)
            x0 = min(max(fx - px * zoom, 0.0), 1.0 - zoom)
            score = composition_score(small, (y0, x0, zoom, zoom))["score"]
            if score > best_score:
                best, best_score = (y0, x0, zoom, zoom), score
    if best == full:
        return full, "kept the photo's framing: no other framing was clearly better"
    box = tuple(f + strength * (b - f) for f, b in zip(full, best, strict=True))
    cy, cx = (fy - box[0]) / box[2], (fx - box[1]) / box[3]
    side = ("upper " if cy < 0.45 else "lower " if cy > 0.55 else "") + ("left" if cx < 0.5 else "right")
    return box, f"reframed: kept {box[2] * box[3]:.0%} of the photo, the subject toward the {side} third"


def warp_grid(h: int, w: int, box: tuple, focal: tuple, magnify: float = 0.0, stretch: float = 0.0,
              tilt: float = 0.0, device=None) -> torch.Tensor:
    """Where each pixel of the picture to paint comes from in the photo, as a grid for ``grid_sample``
    ``(1, h, w, 2)``. ``box`` is the framing, ``focal (y, x)`` the subject inside it; ``magnify`` enlarges what
    is near the subject, ``stretch`` makes the picture taller (+) or wider (-) about it, ``tilt`` leans it by
    that many degrees."""
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h * 2 - 1
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w * 2 - 1
    qy, qx = torch.meshgrid(ys, xs, indexing="ij")
    cx, cy = 2 * focal[1] - 1, 2 * focal[0] - 1
    px, py = (qx - cx) * w / 2, (qy - cy) * h / 2
    r = torch.sqrt(px * px + py * py) / (0.5 * min(h, w))
    scale = 1.0 - magnify * torch.exp(-((r / 0.55) ** 2))
    t = math.radians(tilt)
    px, py = math.cos(t) * px + math.sin(t) * py, -math.sin(t) * px + math.cos(t) * py
    if stretch > 0:
        py = py / (1.0 + stretch)
    elif stretch < 0:
        px = px / (1.0 - stretch)
    ux, uy = cx + px * scale / (w / 2), cy + py * scale / (h / 2)
    y0, x0, bh, bw = box
    return torch.stack([(x0 + bw / 2) * 2 - 1 + ux * bw, (y0 + bh / 2) * 2 - 1 + uy * bh], -1)[None]


# --------------------------------------------------------------------------------------------- look --
def match_quantiles(light: torch.Tensor, quantiles: list) -> torch.Tensor:
    """Remap lightness ``(H, W)`` so that it is spread the way ``quantiles`` (the style's) are."""
    target = torch.tensor(quantiles, device=light.device, dtype=light.dtype)
    own = torch.quantile(light.flatten()[:: max(1, light.numel() // 40000)],
                         torch.linspace(0, 1, len(quantiles), device=light.device))
    idx = torch.bucketize(light, own[1:-1].contiguous())
    lo, hi = own[idx], own[idx + 1]
    t = ((light - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)
    return target[idx] + t * (target[idx + 1] - target[idx])


def push_colour(image: torch.Tensor, focus: torch.Tensor, focal: tuple, feeling: Feeling, card: StyleCard,
                exaggerate: float, style_strength: float) -> tuple[torch.Tensor, list]:
    """The photo's colours changed on purpose (see the module's notes): sRGB ``(3, H, W)`` in and out, with a
    line for the report about each change."""
    notes = []
    lab = image_to_oklab(image)
    light, a, b = lab[0].clone(), lab[1].clone(), lab[2].clone()
    size = max(image.shape[-2:])
    if card.measured and style_strength > 0 and card.lab_cov:
        # the whole cloud of the photo's colours carried onto the painting's, then drawn toward its pigments
        moved = match_colours(lab, torch.tensor(card.lab_mean), torch.tensor(card.lab_cov))
        pigments = linear_to_oklab(card.palette_linear())
        near = torch.softmax(-(moved.reshape(3, -1).T[:, None] - pigments[None]).pow(2).sum(-1) / 0.002, -1)
        drawn = (near @ pigments).T.reshape(moved.shape)
        moved = moved + 0.45 * (drawn - moved)
        light = light + style_strength * (moved[0] - light)
        light = light + style_strength * (match_quantiles(light, card.l_quantiles) - light)
        a, b = a + style_strength * (moved[1] - a), b + style_strength * (moved[2] - b)
        notes.append(f"the photo's colours carried {style_strength:.0%} of the way onto the style's and toward "
                     "its pigments")
    elif card.measured and style_strength > 0:                   # a card measured before colour spreads were kept
        light = light + style_strength * (match_quantiles(light, card.l_quantiles) - light)
        gain_a = card.ab_std[0] / (float(a.std()) + 1e-4)
        gain_b = card.ab_std[1] / (float(b.std()) + 1e-4)
        ma, mb = float(a.mean()), float(b.mean())
        a = a + style_strength * ((a - ma) * min(gain_a, 2.5) + card.ab_mean[0] - a)
        b = b + style_strength * ((b - mb) * min(gain_b, 2.5) + card.ab_mean[1] - b)
        notes.append(f"lights, darks and colours moved {style_strength:.0%} of the way to the style's")
    pivot = float(light.median())
    x = exaggerate
    contrast = 1.0 + x * (feeling.contrast - 1.0)
    light = pivot + (light - pivot) * contrast + x * feeling.key
    sat = 1.0 + x * (feeling.saturation - 1.0)
    a, b = a * sat + 0.4 * x * feeling.warmth, b * sat + x * feeling.warmth
    if x > 0 and feeling.temperature != 0:                       # warm lights, cool shadows (or the reverse)
        tone = ((light - pivot) / 0.5).clamp(-1.0, 1.0)
        b = b + 0.045 * x * feeling.temperature * tone
        a = a + 0.015 * x * feeling.temperature * tone
        notes.append(("lights warmed and shadows cooled" if feeling.temperature > 0 else
                      "lights cooled and shadows warmed") + f" ({abs(0.045 * x * feeling.temperature):.3f} OKLab)")
    if x > 0 and feeling.masses > 0:                             # squint: values gathered into three masses
        soft = gaussian_blur(light, 0.02 * size)
        _, massed = value_masses(soft, 3)
        light = light + x * feeling.masses * (massed - soft)
        notes.append(f"values gathered toward three masses ({x * feeling.masses:.0%})")
    if x > 0:                                                    # the subject gets the contrast and the colour
        wide = gaussian_blur(light, 0.06 * size)
        light = light + 0.6 * x * focus * (light - wide)
        chroma = 1.0 + 0.35 * x * focus - 0.25 * x * (1.0 - focus)
        a, b = a * chroma, b * chroma
        lights = light > torch.quantile(light.flatten()[:: max(1, light.numel() // 40000)], 0.7)
        if bool(lights.any()):                                   # shadows take the complement of the light
            la, lb = float(a[lights].mean()), float(b[lights].mean())
            norm = math.hypot(la, lb)
            if norm > 0.008:
                shade = ((pivot - light) / 0.35).clamp(0.0, 1.0)
                a = a - 0.03 * x * shade * la / norm
                b = b - 0.03 * x * shade * lb / norm
        notes.append("contrast and colour raised at the subject, quieted elsewhere; shadows tinted against the light")
    if x > 0 and feeling.glow > 0:
        high = ((light - 0.62) / 0.3).clamp(0.0, 1.0)
        bloom = gaussian_blur(high, 0.04 * size)
        light = light + x * feeling.glow * (0.05 * high + 0.05 * bloom)
        b = b + x * feeling.glow * 0.02 * (high + bloom) / 2
        a = a + x * feeling.glow * 0.006 * (high + bloom) / 2
        notes.append("the lights made to glow")
    if x > 0:                                                    # hold the eye in: the corners go quieter
        h, w = light.shape
        ys = (torch.arange(h, device=light.device) + 0.5) / h - focal[0]
        xs = (torch.arange(w, device=light.device) + 0.5) / w - focal[1]
        far = (ys[:, None] ** 2 + xs[None, :] ** 2).sqrt()
        light = light - 0.06 * x * (far / 0.9).clamp(0.0, 1.0) ** 2
    out = oklab_to_image(torch.stack([light.clamp(0.02, 0.985), a, b]))
    return out.clamp(0.0, 1.0), notes


def simplify(image: torch.Tensor, focus: torch.Tensor, amount: float,
             stepping: float = 1.0) -> tuple[torch.Tensor, str]:
    """
    See the picture as a painter does, in planes. A photo holds every pore and every smooth gradient, and strokes
    that chase those look like a filter over the photo. So before anything is painted, small differences inside
    a shape are levelled out while the borders between shapes are kept (Perona and Malik's diffusion, in OKLab),
    and lightness is gathered into a handful of tones, the way a passage is mixed from a few puddles of paint
    and not from a gradient. The subject keeps more of its modelling than the rest. ``stepping`` below 1 keeps
    more of the gradations (for a style painted without steps, like Leonardo's). sRGB ``(3, H, W)`` in and out,
    with a line for the report.
    """
    if amount <= 0:
        return image, ""
    lab = image_to_oklab(image)
    keep = 1.0 - 0.65 * focus
    k2 = (0.02 + 0.04 * amount) ** 2                             # differences smaller than this are levelled
    for _ in range(round(20 + 100 * amount)):
        total = torch.zeros_like(lab)
        for shift, dim in ((1, 1), (-1, 1), (1, 2), (-1, 2)):
            d = torch.roll(lab, shift, dim) - lab
            edge = 0 if shift == 1 else -1                       # nothing flows in across the picture's border
            if dim == 1:
                d[:, edge] = 0.0
            else:
                d[:, :, edge] = 0.0
            total = total + torch.exp(-d.pow(2).sum(0, keepdim=True) / k2) * d
        lab = lab + 0.2 * keep * total
    tones = round(14 - 8 * amount)
    q = lab[0] * tones
    stepped = (torch.floor(q) + torch.sigmoid(16.0 * (q - torch.floor(q) - 0.5))) / tones
    lab[0] = lab[0] + amount * stepping * keep * (stepped - lab[0])
    note = "seen in planes: detail levelled" + (f", light and dark in {tones} tones" if stepping > 0.5 else
                                                ", the gradations kept")
    return oklab_to_image(lab).clamp(0.0, 1.0), note


def draw_outlines(target: torch.Tensor, planes: torch.Tensor, dark: torch.Tensor,
                  strength: float) -> torch.Tensor:
    """Draw round the shapes, for a style that does (Hokusai's prints, Van Gogh's Bedroom): the borders between
    the planes of ``planes`` become thin lines in the palette's darkest pigment ``dark (3,)`` (sRGB) on
    ``target``. Both sRGB ``(3, H, W)``."""
    steep = slope(gaussian_blur(image_to_oklab(planes)[0], 1.2))
    sample = steep.flatten()[:: max(1, steep.numel() // 40000)]
    cut = max(float(torch.quantile(sample, 0.86)), 0.25 * float(torch.quantile(sample, 0.995)), 1e-4)
    crest = (steep >= 0.92 * F.max_pool2d(steep[None, None], 3, 1, 1)[0, 0]).float()      # the middle of a border
    line = crest * ((steep - cut) / cut).clamp(0.0, 1.0)         # nothing where the picture is even
    line = (2.2 * gaussian_blur(line, 0.9)).clamp(0.0, 1.0)
    lab = image_to_oklab(target)
    ink = image_to_oklab(dark.reshape(3, 1, 1).float())
    return oklab_to_image(lab + 0.85 * strength * line * (ink - lab)).clamp(0.0, 1.0)


# -------------------------------------------------------------------------------- finish and flow --
def finish_map(focus: torch.Tensor, restraint: float) -> torch.Tensor:
    """How far each part of the picture is taken, 0..1: the subject all the way, the rest less the more
    restraint there is, and a strip along the canvas's edges least of all."""
    finish = focus.clamp(0.0, 1.0) ** (0.3 + 0.9 * restraint)
    h, w = focus.shape
    ys = torch.minimum(torch.arange(h, device=focus.device) + 0.5, h - torch.arange(h, device=focus.device) - 0.5)
    xs = torch.minimum(torch.arange(w, device=focus.device) + 0.5, w - torch.arange(w, device=focus.device) - 0.5)
    edge = (torch.minimum(ys[:, None], xs[None, :]) / (0.07 * min(h, w))).clamp(0.0, 1.0)
    edge = edge * edge * (3.0 - 2.0 * edge)
    return finish * (1.0 - 0.5 * restraint * (1.0 - edge))


def flow_field(kind: str, shape: tuple, focal: tuple, light_at: tuple, angle: float = 0.0, seed: int = 0,
               device=None) -> torch.Tensor:
    """The direction given to strokes the picture gives none: ``(3, H, W)`` with cos and sin of twice the angle
    and a weight. ``toward`` spirals in to the subject, ``radial`` streams away from the light, ``swirl`` turns
    slowly from place to place, ``horizontal``, ``vertical`` and ``directional`` run one way."""
    h, w = shape
    ys = ((torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h)[:, None].expand(h, w)
    xs = ((torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w)[None, :].expand(h, w)
    weight = 1.0
    if kind == "toward":
        theta = torch.atan2((focal[0] - ys) * h, (focal[1] - xs) * w) + 0.6
    elif kind == "radial":
        theta = torch.atan2((ys - light_at[0]) * h, (xs - light_at[1]) * w)
    elif kind == "horizontal":
        theta = torch.zeros(h, w, device=device)
    elif kind == "vertical":
        theta = torch.full((h, w), math.pi / 2, device=device)
    elif kind == "directional":
        theta = torch.full((h, w), float(angle), device=device)
    elif kind == "swirl":
        g = torch.Generator().manual_seed(seed)
        coarse = torch.rand(1, 1, 5, 5, generator=g) * 2 * math.pi
        both = torch.cat([torch.cos(coarse), torch.sin(coarse)], 1)
        smooth = F.interpolate(both, size=(h, w), mode="bicubic", align_corners=True)[0].to(device)
        theta = torch.atan2(smooth[1], smooth[0]) / 2
    else:
        theta, weight = torch.zeros(h, w, device=device), 0.0
    return torch.stack([torch.cos(2 * theta), torch.sin(2 * theta), torch.full((h, w), weight, device=device)])


# --------------------------------------------------------------------------------------------- plan --
def working_size(height: int, width: int, work: int, cell: int = 16) -> tuple[int, int, int, int]:
    """``(grid rows, grid columns, canvas height, canvas width)`` of the finest pass for a photo of that shape:
    the canvas is a whole number of cells, with the photo's proportions to within a few percent."""
    aspect = height / width
    best = None
    for gy in range(1, 6):
        for gx in range(1, 6):
            err = abs(math.log((gy / gx) / aspect))
            if best is None or (err < best[0] - 0.02) or (abs(err - best[0]) <= 0.02 and gy * gx < best[1] * best[2]):
                best = (err, gy, gx)
    _, gy0, gx0 = best
    doublings = max(0, math.ceil(math.log2(max(work, 2 * cell) / (max(gy0, gx0) * cell)) - 0.2))
    return gy0, gx0, gy0 * cell * 2 ** doublings, gx0 * cell * 2 ** doublings


def make_plan(photo: torch.Tensor, card: StyleCard, feeling: Feeling, settings: Settings,
              eye_map: torch.Tensor | None = None, imagine=None, words: str = "") -> Plan:
    """Everything decided before painting ``photo`` (sRGB ``(3, H, W)``) in the style of ``card`` with
    ``feeling``. ``imagine`` is an image model paired with the painter
    (:class:`helix_lm.paint.imagine.Imaginer`): once the photo is framed it imagines it as a painting from
    ``words``, and that picture is what the painter then looks at instead of the photo."""
    photo = photo.float()
    _, _, h, w = working_size(photo.shape[-2], photo.shape[-1], settings.work)
    report = []
    seen = resize(photo, (min(photo.shape[-2], 2 * h), min(photo.shape[-1], 2 * w)))
    sal = saliency(resize(seen, (h, w)), eye_map)
    box, note = compose(sal, settings.compose)
    report.append(note)
    inner = (sal.shape[0] * box[0], sal.shape[1] * box[1])
    crop = sal[int(inner[0]) : int(inner[0] + box[2] * sal.shape[0]) + 1,
               int(inner[1]) : int(inner[1] + box[3] * sal.shape[1]) + 1]
    focal = focal_point(crop)
    x = settings.exaggerate
    lean = 0.0
    if x > 0 and feeling.tilt:
        ys = (torch.arange(crop.shape[0]) + 0.5) / crop.shape[0] - focal[0]
        xs = (torch.arange(crop.shape[1]) + 0.5) / crop.shape[1] - focal[1]
        lean = math.copysign(x * feeling.tilt, float((crop * ys[:, None] * xs[None, :]).sum()) or 1.0)
    grid = warp_grid(h, w, box, focal, x * feeling.magnify, x * feeling.stretch, lean)
    source = F.grid_sample(seen[None], grid, mode="bilinear", padding_mode="reflection", align_corners=False)[0]
    moves = [m for m, on in ((f"the subject enlarged {x * feeling.magnify:.0%}", x * feeling.magnify > 0.005),
                             (f"the picture made {'taller' if feeling.stretch > 0 else 'wider'} by "
                              f"{abs(x * feeling.stretch):.0%}", abs(x * feeling.stretch) > 0.005),
                             (f"leaned {abs(lean):.1f} degrees", abs(lean) > 0.05)) if on]
    if moves:
        report.append("distorted on purpose: " + ", ".join(moves))
    source = source.clamp(0.0, 1.0)
    if imagine is not None:
        source = imagine(source, words, seed=settings.seed).clamp(0.0, 1.0)
        report.append(f"imagined as a painting by {getattr(imagine, 'name', '') or 'the image model'}: "
                      f"{words[:110]}")
    sal = saliency(source, None if eye_map is None else F.grid_sample(
        eye_map.float()[None, None], grid, mode="bilinear", padding_mode="reflection", align_corners=False)[0, 0])
    ys = ((torch.arange(h) + 0.5) / h - focal[0])[:, None]
    xs = ((torch.arange(w) + 0.5) / w - focal[1])[None, :]
    near = torch.exp(-(ys ** 2 + xs ** 2) / (2 * 0.28 ** 2))
    focus = 0.6 * sal + 0.4 * near
    focus = focus / focus.max().clamp(min=1e-8)
    if imagine is not None:                                      # the image model has seen it in planes already
        planes, note = source, ""
    else:
        planes, note = simplify(source, focus, settings.abstraction, card.hardness if card.measured else 1.0)
    if note:
        report.append(note)
    target, notes = push_colour(planes, focus, focal, feeling, card, x, settings.style_strength)
    report += notes
    if card.measured and card.outline * settings.style_strength > 0.2:
        target = draw_outlines(target, planes, torch.tensor(card.palette[0]), card.outline * settings.style_strength)
        report.append(f"the shapes drawn round in the darkest pigment, as the style does ({card.outline:.0%})")
    if card.measured and card.hardness < 0.5:
        report.append(f"edges kept soft and paint laid in thin layers, as the style does (hardness "
                      f"{card.hardness:.2f})")
    restraint = feeling.restraint if settings.restraint is None else settings.restraint
    finish = finish_map(focus, restraint)
    report.append(f"restraint {restraint:.2f}: {float((finish >= 0.45 * restraint + 0.05).float().mean()):.0%} of "
                  "the picture gets the finest strokes, the rest is left broader")
    light = gaussian_blur(image_to_oklab(target)[0], 0.05 * max(h, w))
    top = (light == light.max()).nonzero()[0]
    light_at = ((float(top[0]) + 0.5) / h, (float(top[1]) + 0.5) / w)
    kind = card.flow if (card.measured and card.flow != "follow") else feeling.flow
    flow = flow_field(kind, (h, w), focal, light_at, card.flow_angle, settings.seed)
    if kind != "follow":
        report.append(f"loose strokes flow {kind}")
    if card.measured and card.palette:
        palette = card.palette_linear()
        ground = srgb_to_linear(torch.tensor(card.ground))
    else:
        colours, _ = extract_palette(target, settings.colors, seed=settings.seed)
        palette = srgb_to_linear(colours)
        lab = linear_to_oklab(srgb_to_linear(target.reshape(3, -1).T)).mean(0)
        # the tone of the bare canvas: a little under the picture's middle lightness, its colour muted and warmed
        tone = torch.stack([(0.9 * lab[0]).clamp(0.32, 0.6), lab[1] * 0.5 + 0.4 * feeling.warmth + 0.004,
                            lab[2] * 0.5 + feeling.warmth + 0.010])
        from .color import oklab_to_linear

        ground = oklab_to_linear(tone).clamp(0.0, 1.0)
    return Plan(target, source, focus, finish, flow, focal, ground, palette, feeling, card, settings, report)


__all__ = ["Plan", "Settings", "compose", "composition_score", "finish_map", "flow_field", "make_plan",
           "match_quantiles", "push_colour", "simplify", "warp_grid", "working_size"]
