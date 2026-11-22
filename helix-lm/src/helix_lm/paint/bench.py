# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Judging a painting in numbers. None of these says whether a painting is good; they say whether it did what was
asked, and they let two hands (the trained model and the untrained rules) be compared on the same pictures.

* ``closeness``: how far the painting is from its plan, in OKLab, weighted by how much each part was to be
  finished (0 is identical; about 0.02 is a difference just seen). ``at_subject`` is the same where the picture
  was to be finished.
* ``left_broad``: the share of the canvas no stroke of the last two passes touches. Restraint shows here.
* ``strokes``, and how many of them the last pass made.
* ``pressure_spread`` and ``speed_spread``: how much the gestures differ from each other. A painting whose
  strokes are all pressed and moved alike has none of the hand in it.
* ``palette_use``: how close the stroke colours stay to mixes of the palette's pigments.
* the composition's rules for the framing that was chosen (the subject near a third, how much was kept, balance).
"""

from __future__ import annotations

import torch

from .color import delta_e, linear_to_oklab, linear_to_srgb, snap_to_palette
from .mind import composition_score
from .render import render_painting
from .strokes import A_COLOR, A_LEVEL, A_PRESSURE, A_SPEED


def judge(painting) -> dict:
    plan = painting.plan
    h, w = plan.target.shape[-2:]
    image, _ = render_painting(painting.rows, w, h, plan.ground, relief=0.0, tremor=False)
    miss = delta_e(linear_to_srgb(image), plan.target)
    weight = 0.25 + 0.75 * plan.finish
    subject = plan.finish > 0.6
    level = painting.rows[:, A_LEVEL]
    top = int(level.max()) if len(painting) else 0
    fine = painting.rows[level >= top - 1]
    bare = torch.zeros(3, h, w)
    if len(fine):                                             # where the fine strokes put any paint at all
        rows = fine.clone()
        rows[:, A_COLOR] = 1.0
        rows[:, 20:23] = 1.0
        bare, _ = render_painting(rows, w, h, torch.zeros(3), relief=0.0, tremor=False)
    colours = painting.rows[:, A_COLOR]
    snapped = snap_to_palette(colours, plan.palette, 1.0)
    rules = composition_score(plan.focus, (0.0, 0.0, 1.0, 1.0))
    return {"closeness": float((weight * miss).sum() / weight.sum()),
            "at_subject": float(miss[subject].mean()) if bool(subject.any()) else float("nan"),
            "left_broad": float((bare.max(0).values < 0.05).float().mean()),
            "strokes": len(painting), "last_pass": int((level == top).sum()),
            "pressure_spread": float(painting.rows[:, A_PRESSURE].std()) if len(painting) > 1 else 0.0,
            "speed_spread": float(painting.rows[:, A_SPEED].std()) if len(painting) > 1 else 0.0,
            "palette_use": float(1.0 - (linear_to_oklab(colours) - linear_to_oklab(snapped)).norm(dim=-1).mean() / 0.1
                                 ) if len(painting) else 0.0,
            "subject_off_third": rules["thirds"], "balance": rules["balance"]}


def compare(rows: list[dict]) -> str:
    """A table of :func:`judge` results: ``rows`` are dicts with ``name``, ``hand`` and the scores."""
    keys = ("closeness", "at_subject", "left_broad", "strokes", "pressure_spread", "speed_spread", "seconds")
    lines = ["| picture | hand | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for r in rows:
        cells = [f"{r[k]:.3f}" if isinstance(r.get(k), float) else str(r.get(k, "")) for k in keys]
        lines.append(f"| {r.get('name', '')} | {r.get('hand', '')} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


__all__ = ["compare", "judge"]
