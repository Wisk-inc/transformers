# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
A painting, and its memory.

A :class:`Painting` is not a picture but what was done: every stroke in order (where, how long, which brush,
which colours, how hard and how fast), the plan it was painted to (the framing, the pushed colours, what was to
be finished), what the trained model remembers of each place, and a history of every change. So a painting can
be

* rendered at any size (:meth:`Painting.render`), shown pass by pass (:meth:`stages`) or as it was painted
  (:meth:`timelapse`), and written out as vector paths (:meth:`to_svg`) or as a timed list of gestures
  (:meth:`performance`);
* saved whole (``painting.save("x.pinxit")``) and opened again later (:meth:`Painting.load`);
* changed: :meth:`add` paints something new into it, :meth:`touch` takes a part further, :meth:`rework` paints
  it again with another feeling, and :meth:`undo` takes the last change back. Changes are painted on top, the
  way a painter works over a canvas; nothing already there is thrown away, and each change keeps what it altered
  in the plan and in the model's memory, which is what makes undo exact.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .brushes import BRUSHES
from .color import image_to_oklab, linear_to_srgb, oklab_to_image
from .emotion import Feeling, feeling_of
from .engine import CELL, StudioHand, paint_level, plan_levels
from .io import load_alpha, load_image, resize, save_image, to_pil
from .mind import Plan, Settings, make_plan, push_colour, simplify
from .render import light_relief, render_painting
from .strokes import A_ALPHA, A_BRUSH, A_COLOR, A_LEVEL, A_PRESSURE, A_PROFILE, A_R0, A_R1, A_SPEED, ABS
from .style import StyleCard, named_style, own_card, read_style, style_names

VERSION = 1


class Painting:
    def __init__(self, rows: torch.Tensor, plan: Plan, history: list | None = None,
                 memory: dict | None = None, snapshots: list | None = None) -> None:
        self.rows, self.plan = rows, plan
        self.history = history or []
        self.memory = memory or {}
        self.snapshots = snapshots or []          # for each change after the first painting: what it altered

    # ------------------------------------------------------------------------------------- pictures --
    @property
    def aspect(self) -> float:
        return self.plan.target.shape[-2] / self.plan.target.shape[-1]

    def __len__(self) -> int:
        return int(self.rows.shape[0])

    def render(self, width: int = 1024, relief: float | None = None, upto: int | None = None, border: float = 0.0,
               weave: float | None = None) -> torch.Tensor:
        """The painting as an sRGB picture ``(3, H, W)``, ``width`` pixels wide. ``upto`` stops after that many
        strokes. ``border`` adds the bare edge of the canvas all round (a share of the picture's width): the
        strokes end where the painter stopped, not at a ruled line. ``relief`` is how strongly thick paint
        catches the light (by default more for a style with visible paint) and ``weave`` how much the cloth
        shows (by default the plan's setting)."""
        rows = self.rows if upto is None else self.rows[:upto]
        card, settings = self.plan.card, self.plan.settings
        if relief is None:
            relief = 0.35 + 0.6 * card.texture if card.measured else 0.6
        scale = width / (1.0 + 2.0 * border)
        height = max(1, round(scale * (self.aspect + 2.0 * border)))
        image, _ = render_painting(rows, width, height, self.plan.ground, relief=relief,
                                   weave=settings.weave if weave is None else weave, scale=scale,
                                   origin=(border * scale, border * scale))
        return linear_to_srgb(image)

    def stages(self, width: int = 512) -> list[tuple[str, torch.Tensor]]:
        """The bare canvas and the painting after each pass."""
        out = [("the toned canvas", self.render(width, upto=0))]
        level = self.rows[:, A_LEVEL]
        for k in sorted({int(v) for v in level.tolist()}):
            n = int((level <= k).sum()) if k < int(level.max()) else len(self)
            names = {0: "the block-in"}
            out.append((names.get(k, f"pass {k + 1}"), self.render(width, upto=n)))
        return out

    def sheet(self, path, width: int = 384) -> Path:
        """One picture that shows the work: the photo, the plan (after composing and pushing the colour), each
        pass, and the painting."""
        tiles = [resize(self.plan.source, (round(width * self.aspect), width)),
                 resize(self.plan.target, (round(width * self.aspect), width))]
        tiles += [image for _, image in self.stages(width)[1:]]
        return save_image(torch.cat(tiles, -1), path)

    def timelapse(self, path, width: int = 480, frames: int = 40, seconds: float = 6.0) -> Path:
        """An animation of the painting being made (GIF): early strokes one at a time, later ones in groups."""
        n = len(self)
        marks = sorted({min(n, round(n * (i / frames) ** 2.2)) for i in range(1, frames + 1)} | {n})
        height = max(1, round(width * self.aspect))
        canvas = paint_height = None
        done, pictures = 0, [to_pil(self.render(width, upto=0))]
        for mark in marks:
            canvas, paint_height = render_painting(self.rows[done:mark], width, height, self.plan.ground,
                                                   canvas=canvas, paint_height=paint_height, finish=False)
            pictures.append(to_pil(linear_to_srgb(light_relief(canvas, paint_height, 0.5).clamp(0, 1))))
            done = mark
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        each = max(40, int(1000 * seconds / len(pictures)))
        pictures[0].save(path, save_all=True, append_images=pictures[1:] + [pictures[-1]] * 8, duration=each, loop=0)
        return path

    def performance(self) -> list[dict]:
        """The painting as gestures in time: for each stroke when it starts and how long it takes (a slow stroke
        takes longer and is thought about longer first), with its brush, pressure and speed."""
        out, clock = [], 0.0
        length = (self.rows[:, 4:6] - self.rows[:, 0:2]).norm(dim=-1)
        for i in range(len(self)):
            speed, pressure = float(self.rows[i, A_SPEED]), float(self.rows[i, A_PRESSURE])
            clock += 0.05 + 0.5 * (1.0 - speed) ** 2
            duration = float(length[i]) / (0.25 + 2.75 * speed) + 0.03
            out.append({"start": round(clock, 3), "seconds": round(duration, 3),
                        "brush": BRUSHES[int(self.rows[i, A_BRUSH])], "pressure": round(pressure, 3),
                        "speed": round(speed, 3), "pass": int(self.rows[i, A_LEVEL]) + 1})
            clock += duration
        return out

    def to_svg(self, path, width: int = 1024) -> Path:
        """The strokes as vector paths (curves with a width and a colour; bristle marks are not drawn)."""
        height = round(width * self.aspect)
        g = (linear_to_srgb(self.plan.ground) * 255).round().int().tolist()
        lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                 f'viewBox="0 0 {width} {height}">',
                 f'<rect width="100%" height="100%" fill="rgb({g[0]},{g[1]},{g[2]})"/>']
        srgb = (linear_to_srgb(self.rows[:, A_COLOR]) * 255).round().int().tolist()
        pts = (self.rows[:, 0:6] * width).tolist()
        for i, p in enumerate(pts):
            row = self.rows[i]
            thick = float(row[A_R0] + row[A_R1]) * width * (0.75 + 0.5 * float(row[A_PRESSURE]))
            cap = "round" if float(row[A_PROFILE][2]) > 0.5 else "butt"
            lines.append(f'<path d="M{p[0]:.1f} {p[1]:.1f} Q{p[2]:.1f} {p[3]:.1f} {p[4]:.1f} {p[5]:.1f}" fill="none" '
                         f'stroke="rgb({srgb[i][0]},{srgb[i][1]},{srgb[i][2]})" stroke-width="{thick:.1f}" '
                         f'stroke-linecap="{cap}" stroke-opacity="{float(row[A_ALPHA]):.2f}"/>')
        lines.append("</svg>")
        path = Path(path)
        path.write_text("\n".join(lines))
        return path

    def summary(self) -> dict:
        """What the painting is made of, in numbers."""
        rows = self.rows
        level = rows[:, A_LEVEL]
        return {"strokes": len(self),
                "by_pass": {int(k) + 1: int((level == k).sum()) for k in sorted({int(v) for v in level.tolist()})},
                "by_brush": {b: int((rows[:, A_BRUSH] == i).sum()) for i, b in enumerate(BRUSHES)},
                "pressure": {"mean": float(rows[:, A_PRESSURE].mean()), "spread": float(rows[:, A_PRESSURE].std())},
                "speed": {"mean": float(rows[:, A_SPEED].mean()), "spread": float(rows[:, A_SPEED].std())},
                "finest_share": float((self.plan.finish >= 0.45 * self.plan.restraint + 0.05).float().mean()),
                "changes": [h["op"] for h in self.history]}

    # ------------------------------------------------------------------------------------- the file --
    def save(self, path, width: int = 1600) -> Path:
        """``.pinxit``: the whole painting with its memory, to open and change later. Any picture format
        (``.png``, ``.jpg``): the rendered picture. ``.svg``: vector paths. ``.gif``: the timelapse."""
        path = Path(path)
        kind = path.suffix.lower()
        if kind == ".svg":
            return self.to_svg(path, width)
        if kind == ".gif":
            return self.timelapse(path)
        if kind != ".pinxit":                         # the canvas as it would be photographed, edge and all
            return save_image(self.render(width, border=self.plan.settings.border), path)
        plan = self.plan
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"version": VERSION, "rows": self.rows, "target": plan.target, "source": plan.source,
                    "focus": plan.focus, "finish": plan.finish, "flow": plan.flow, "focal": list(plan.focal),
                    "ground": plan.ground, "palette": plan.palette, "feeling": json.dumps(plan.feeling.to_json()),
                    "card": json.dumps(plan.card.to_json()), "settings": json.dumps(plan.settings.__dict__),
                    "report": list(plan.report), "history": json.dumps(self.history),
                    "memory": {int(k): v for k, v in self.memory.items()}, "snapshots": self.snapshots}, path)
        return path

    @classmethod
    def load(cls, path) -> Painting:
        data = torch.load(Path(path), map_location="cpu", weights_only=True)
        feeling = json.loads(data["feeling"])
        feeling["brushes"] = tuple(feeling["brushes"])
        plan = Plan(data["target"], data["source"], data["focus"], data["finish"], data["flow"], tuple(data["focal"]),
                    data["ground"], data["palette"], Feeling(**feeling), StyleCard.from_json(json.loads(data["card"])),
                    Settings(**json.loads(data["settings"])), list(data["report"]))
        return cls(data["rows"], plan, json.loads(data["history"]), dict(data["memory"]),
                   list(data.get("snapshots", [])))

    # ---------------------------------------------------------------------------------- changes --
    def _snapshot(self, mask: torch.Tensor | None = None) -> dict:
        """What a change may alter, as it is now: the plan where ``mask`` is set (everywhere without one), the
        feeling, and what the model remembers."""
        plan = self.plan
        h, w = plan.target.shape[-2:]
        y0, y1, x0, x1 = 0, h, 0, w
        if mask is not None and bool((mask > 0).any()):
            ys, xs = (mask > 0).any(1).nonzero().flatten(), (mask > 0).any(0).nonzero().flatten()
            y0, y1, x0, x1 = int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1
        return {"box": [y0, y1, x0, x1], "target": plan.target[:, y0:y1, x0:x1].clone(),
                "source": plan.source[:, y0:y1, x0:x1].clone(), "focus": plan.focus[y0:y1, x0:x1].clone(),
                "finish": plan.finish[y0:y1, x0:x1].clone(), "feeling": json.dumps(plan.feeling.to_json()),
                "report": len(plan.report), "memory": {int(k): v.clone() for k, v in self.memory.items()}}

    def _restore(self, snapshot: dict) -> None:
        plan = self.plan
        y0, y1, x0, x1 = snapshot["box"]
        plan.target[:, y0:y1, x0:x1] = snapshot["target"]
        plan.source[:, y0:y1, x0:x1] = snapshot["source"]
        plan.focus[y0:y1, x0:x1] = snapshot["focus"]
        plan.finish[y0:y1, x0:x1] = snapshot["finish"]
        feeling = json.loads(snapshot["feeling"])
        feeling["brushes"] = tuple(feeling["brushes"])
        plan.feeling = Feeling(**feeling)
        del plan.report[snapshot["report"]:]
        self.memory = {int(k): v.clone() for k, v in snapshot["memory"].items()}

    @contextlib.contextmanager
    def _change(self, mask: torch.Tensor | None = None):
        """Around one change: remember what it may alter, so :meth:`undo` can put it back exactly; and if the
        change fails half-way, put it back at once."""
        snapshot, strokes, entries = self._snapshot(mask), len(self), len(self.history)
        # the maps are changed in place by some changes and replaced by others: work on copies either way
        self.plan.target, self.plan.source = self.plan.target.clone(), self.plan.source.clone()
        self.plan.focus, self.plan.finish = self.plan.focus.clone(), self.plan.finish.clone()
        try:
            yield
        except BaseException:
            self._restore(snapshot)
            self.rows = self.rows[:strokes]
            del self.history[entries:]
            raise
        self.snapshots.append(snapshot)

    def _levels(self):
        return plan_levels(*self.plan.target.shape[-2:], self.plan.settings.detail, self.plan.settings.steps)

    def _box_mask(self, box, feather: float = 0.15) -> torch.Tensor:
        """``box = (left, top, right, bottom)`` in 0..1 -> a soft mask ``(H, W)`` that is 1 inside."""
        h, w = self.plan.target.shape[-2:]
        x0, y0, x1, y1 = box
        ys = (torch.arange(h) + 0.5) / h
        xs = (torch.arange(w) + 0.5) / w
        soft = max(1e-3, feather * min(x1 - x0, y1 - y0))
        my = ((ys - y0) / soft).clamp(0, 1) * ((y1 - ys) / soft).clamp(0, 1)
        mx = ((xs - x0) / soft).clamp(0, 1) * ((x1 - xs) / soft).clamp(0, 1)
        return my[:, None] * mx[None, :]

    def _paint_region(self, mask: torch.Tensor, what: dict, hand=None, budget: int = 700, log=None) -> Painting:
        """Paint over the part of the canvas where ``mask`` is set, starting with strokes no larger than it."""
        hand = hand or StudioHand()
        levels = self._levels()
        h, w = self.plan.target.shape[-2:]
        extent = max(float(mask.sum(0).gt(0.5).sum()), float(mask.sum(1).gt(0.5).sum()), CELL)
        first = next((lv.index for lv in levels if h / lv.gy <= 0.6 * extent), levels[-1].index)
        generator = torch.Generator().manual_seed(self.plan.settings.seed + 1000 + len(self.history))
        before, rows = len(self), self.rows
        canvas, thick = render_painting(rows, w, h, self.plan.ground, relief=0.0, tremor=False, finish=False)
        for level in levels:
            if level.index < first:
                continue
            new, notes = paint_level(self.plan, level, len(levels), rows, hand, generator, region=mask,
                                     memory=self.memory.get(level.index), budget=max(1, budget - (len(rows) - before)),
                                     log=log, canvas=canvas)
            canvas, thick = render_painting(new, w, h, self.plan.ground, relief=0.0, tremor=False, canvas=canvas,
                                            paint_height=thick, finish=False)
            rows = torch.cat([rows, new])
            if "memory" in notes:                    # what the model now remembers of the places it worked
                store = self.memory.setdefault(level.index,
                                               torch.zeros(level.gy * level.gx, notes["memory"].shape[-1]))
                store[notes["cells_index"]] = notes["memory"].float().cpu()
        self.rows = rows
        self.history.append({**what, "from": before, "strokes": len(self) - before})
        return self

    def add(self, image=None, text: str | None = None, box=(0.3, 0.3, 0.7, 0.7), hand=None, eye=None, log=None,
            **options):
        """Paint something new into the painting, inside ``box = (left, top, right, bottom)`` (shares of the
        canvas). ``image`` (a picture, with or without transparency) is fitted into the box, brought into the
        painting's light and colour, and painted with the painting's own palette and brushes. ``text`` asks the
        eye for it (needs an eye that reads text, see :mod:`helix_lm.paint.add`)."""
        if (image is None) == (text is None):
            raise ValueError("give either image= or text=")
        if text is not None:
            from .add import add_text

            return add_text(self, text, box, eye=eye, log=log, **options)
        h, w = self.plan.target.shape[-2:]
        x0, y0, x1, y1 = box
        bh, bw = max(2, round((y1 - y0) * h)), max(2, round((x1 - x0) * w))
        thing, alpha = load_image(image), load_alpha(image)
        scale = min(bh / thing.shape[-2], bw / thing.shape[-1])
        th, tw = max(1, round(thing.shape[-2] * scale)), max(1, round(thing.shape[-1] * scale))
        thing = resize(thing, (th, tw))
        top, left = round(y0 * h) + (bh - th) // 2, round(x0 * w) + (bw - tw) // 2
        top, left = max(0, min(h - th, top)), max(0, min(w - tw, left))
        if alpha is not None:
            cover = resize(alpha[None], (th, tw))[0]
        else:                                        # no transparency: a soft-edged patch
            ys, xs = torch.linspace(-1, 1, th)[:, None], torch.linspace(-1, 1, tw)[None, :]
            cover = ((1.0 - torch.maximum(ys.abs(), xs.abs())) / 0.2).clamp(0, 1)
        place = (slice(top, top + th), slice(left, left + tw))
        mask = torch.zeros(h, w)
        mask[place] = cover
        grown = F.max_pool2d(mask[None, None], 2 * (CELL // 2) + 1, 1, CELL // 2)[0, 0]
        with self._change(grown):
            # into the painting's light: the thing's lightness and colour moved halfway to those of where it goes
            lab, here = image_to_oklab(thing), image_to_oklab(self.plan.target[:, place[0], place[1]])
            for c in range(3):
                lab[c] = lab[c] + 0.5 * (float(here[c].mean()) - float(lab[c].mean()))
            thing = oklab_to_image(lab).clamp(0, 1)
            for picture in (self.plan.target, self.plan.source):
                picture[:, place[0], place[1]] = picture[:, place[0], place[1]] * (1 - cover) + thing * cover
            self.plan.finish = torch.maximum(self.plan.finish, 0.95 * mask)
            self.plan.focus = torch.maximum(self.plan.focus, 0.8 * mask)
            return self._paint_region(grown, {"op": "add", "box": list(box)}, hand, log=log)

    def touch(self, box, finish: float = 1.0, hand=None, log=None) -> Painting:
        """Take a part of the painting further: it is finished to ``finish`` and worked again."""
        mask = self._box_mask(box)
        with self._change(mask):
            self.plan.finish = torch.maximum(self.plan.finish, finish * mask)
            return self._paint_region(mask, {"op": "touch", "box": list(box)}, hand, log=log)

    def rework(self, emotion=None, box=None, exaggerate: float | None = None, hand=None, log=None) -> Painting:
        """Paint the picture (or the part in ``box``) again, with another feeling, over what is there."""
        feeling = feeling_of(emotion) if emotion is not None else self.plan.feeling
        x = self.plan.settings.exaggerate if exaggerate is None else exaggerate
        mask = self._box_mask(box) if box is not None else torch.ones_like(self.plan.focus)
        with self._change(mask):
            card = self.plan.card
            planes, _ = simplify(self.plan.source, self.plan.focus, self.plan.settings.abstraction,
                                 card.hardness if card.measured else 1.0)
            target, notes = push_colour(planes, self.plan.focus, self.plan.focal, feeling, self.plan.card, x,
                                        self.plan.settings.style_strength)
            self.plan.target = self.plan.target * (1 - mask) + target * mask
            self.plan.feeling = feeling
            self.plan.report += [f"reworked: {n}" for n in notes]
            return self._paint_region(mask, {"op": "rework", "emotion": str(emotion),
                                             "box": list(box) if box else None}, hand, budget=1500, log=log)

    def undo(self) -> Painting:
        """Take back the last change: its strokes, and what it altered in the plan and in the model's memory.
        The painting is then exactly as it was before the change."""
        if len(self.history) <= 1:
            return self
        last = self.history.pop()
        self.rows = self.rows[: last["from"]]
        if len(self.snapshots) == len(self.history):          # a painting saved before changes kept snapshots
            self._restore(self.snapshots.pop())
        return self


def paint(photo, style=None, emotion=None, *, hand=None, eye=None, log=None, **settings) -> Painting:
    """
    Paint a photo. ``photo`` and ``style`` are paths, PIL images or ``(3, H, W)`` sRGB tensors; ``style`` may also
    be a :class:`~helix_lm.paint.style.StyleCard` or the name of a famous painting whose measured style comes
    with the package (``"mona lisa"``, ``"the starry night"``; :func:`~helix_lm.paint.style.style_names` lists
    them). ``emotion`` is a name (``"awe"``, ``"majesty"``,
    ``"melancholy"`` ...); without one, a painting given as the style passes on the feeling read from it, and a
    photo alone is painted without one. ``hand`` is what makes the strokes (the trained model's
    :class:`~helix_lm.paint.model.ModelHand`, or the untrained :class:`StudioHand`), ``eye`` a pretrained vision
    model that says what matters in the photo. Other keywords are :class:`~helix_lm.paint.mind.Settings`.
    """
    image = load_image(photo, max_side=2048)
    s = Settings(**settings)
    if isinstance(style, StyleCard):
        card = style
    elif isinstance(style, str) and not Path(style).exists():    # not a file: a famous painting, by name
        card = named_style(style)
        if card is None:
            raise FileNotFoundError(f"{style!r} is not a picture on disk and not a painting whose style comes "
                                    f"with the package; those are: {', '.join(style_names())}")
    else:
        card = read_style(style, seed=s.seed) if style is not None else own_card(image, s.colors, s.seed)
    if emotion is not None:
        feeling = feeling_of(emotion)
    else:
        feeling = feeling_of(card.emotion, 0.7) if (card.measured and card.emotion) else feeling_of(None)
    eye_map = eye.saliency(image) if eye is not None else None
    plan = make_plan(image, card, feeling, s, eye_map)
    if log:
        for line in plan.report:
            log("plan: " + line)
    levels = plan_levels(*plan.target.shape[-2:], s.detail, s.steps)
    hand = hand or StudioHand()
    if hasattr(hand, "look"):
        hand.look(plan.source)
    generator = torch.Generator().manual_seed(s.seed)
    rows, memory = torch.zeros(0, ABS), {}
    height, width = plan.target.shape[-2:]
    canvas = thick = None                                         # the painting so far, kept as a picture
    weights = [(lv.gy * lv.gx) ** 0.7 for lv in levels]
    for i, level in enumerate(levels):
        # the first two passes cover the canvas and take what they need; the rest share what is left of the budget
        cap = None if level.index <= 1 else max(1, int((s.budget - rows.shape[0]) * weights[i] / sum(weights[i:])))
        carried = None
        if i > 0 and levels[i - 1].index in memory:              # a cell starts from what was remembered of its parent
            prev = memory[levels[i - 1].index].reshape(levels[i - 1].gy, levels[i - 1].gx, -1)
            carried = prev.repeat_interleave(2, 0).repeat_interleave(2, 1).reshape(level.gy * level.gx, -1)
        new, notes = paint_level(plan, level, len(levels), rows, hand, generator, memory=carried, budget=cap, log=log,
                                 canvas=canvas)
        canvas, thick = render_painting(new, width, height, plan.ground, relief=0.0, tremor=False, canvas=canvas,
                                        paint_height=thick, finish=False)
        rows = torch.cat([rows, new])
        if "memory" in notes:
            store = torch.zeros(level.gy * level.gx, notes["memory"].shape[-1])
            store[notes["cells_index"]] = notes["memory"].float().cpu()
            memory[level.index] = store
    marks = card.marks() * s.style_strength
    if marks > 0.4:                                               # a style whose paint shows everywhere
        new, _ = paint_level(plan, levels[-1], len(levels), rows, StudioHand(), generator, log=log, canvas=canvas,
                             marks=marks)
        rows = torch.cat([rows, new])
        plan.report.append(f"a last pass of short marks all over, as the style's paint shows ({marks:.0%})")
    history = [{"op": "paint", "from": 0, "strokes": int(rows.shape[0]), "hand": getattr(hand, "name", "hand"),
                "emotion": str(emotion), "style": card.name}]
    return Painting(rows, plan, history, memory)


__all__ = ["Painting", "paint"]
