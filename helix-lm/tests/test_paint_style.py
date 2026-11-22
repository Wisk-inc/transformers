# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Copying a painting's style: its handling is measured (soft or hard edges, outlines, how much the paint
shows), the photo's colours are carried onto the painting's, the painter then works that way, and famous
paintings can be asked for by name."""

import json

import pytest
import torch

from helix_lm.paint import named_style, paint, read_style, style_names
from helix_lm.paint.color import colour_statistics, image_to_oklab, match_colours
from helix_lm.paint.data import PlanStream
from helix_lm.paint.emotion import feeling_of
from helix_lm.paint.io import save_image
from helix_lm.paint.mind import Settings, draw_outlines, make_plan
from helix_lm.paint.see import gaussian_blur
from helix_lm.paint.strokes import A_PROFILE
from helix_lm.paint.style import STYLES, StyleCard


def _shapes(seed: int = 0, size: int = 160) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    img = torch.rand(3, 1, 1, generator=g).expand(3, size, size).clone()
    for _ in range(9):
        y, x = (torch.randint(0, size - 50, (2,), generator=g)).tolist()
        h, w = (torch.randint(24, 50, (2,), generator=g)).tolist()
        img[:, y:y + h, x:x + w] = torch.rand(3, 1, 1, generator=g)
    return img


def test_how_a_painting_is_handled_is_read_from_it():
    flat = _shapes()
    smoked = torch.stack([gaussian_blur(c, 5.0) for c in flat])                 # every edge smoked away
    soft, crisp = read_style(smoked), read_style(flat)
    assert soft.hardness < 0.35 < 0.7 < crisp.hardness and soft.softness() > 1.8 and crisp.softness() < 1.25
    assert soft.opacity < 0.8 < crisp.opacity                                   # thin layers, or opaque paint
    assert max(range(6), key=lambda i: soft.brushes[i]) == 5                    # the wash is its brush
    assert crisp.brushes[5] < 0.2 * sum(crisp.brushes)
    lined = flat.clone()
    edges = (flat - torch.roll(flat, 1, 1)).abs().sum(0) + (flat - torch.roll(flat, 1, 2)).abs().sum(0) > 0.05
    lined[:, edges | torch.roll(edges, 1, 0) | torch.roll(edges, 1, 1)] = 0.03  # every shape drawn round in black
    assert read_style(lined).outline > 0.5 > 0.15 >= crisp.outline and soft.outline == 0.0
    g = torch.Generator().manual_seed(1)
    dabbed = (flat + 0.5 * gaussian_blur(torch.randn(160, 160, generator=g), 1.6)[None]).clamp(0, 1)
    assert read_style(dabbed).texture > 0.6 > 0.3 > soft.texture                # paint that shows, or does not
    assert StyleCard().softness() == 1.0 and len(soft.lab_cov) == 3 and soft.measured


def test_the_photos_colours_are_carried_onto_the_paintings():
    g = torch.Generator().manual_seed(0)
    photo = torch.rand(3, 64, 64, generator=g) * 0.3 + 0.2
    photo[2, 20:44, 20:44] = 0.95                                               # a strong blue patch
    browns = torch.rand(1, 64, 64, generator=g) * torch.tensor([0.75, 0.5, 0.25])[:, None, None] + 0.1
    mean, cov = colour_statistics(image_to_oklab(browns))
    lab = image_to_oklab(photo)
    moved = match_colours(lab, mean, cov)
    got_mean, got_cov = colour_statistics(moved)
    assert torch.allclose(got_mean, mean, atol=1e-3) and torch.allclose(got_cov, cov, atol=2e-4)
    blue_before, blue_after = float(lab[2, 20:44, 20:44].mean()), float(moved[2, 20:44, 20:44].mean())
    assert blue_before < -0.1 and blue_after > -0.03              # the blue is gone: the painting has none
    card = read_style(browns)
    plan = make_plan(photo, card, feeling_of(None), Settings(work=96, compose=0.0, exaggerate=0.0, abstraction=0.0))
    assert float(image_to_oklab(plan.target)[2, 28:36, 28:36].mean()) > -0.03
    assert any("carried" in line for line in plan.report)
    old = StyleCard.from_json({k: v for k, v in card.to_json().items() if not k.startswith("lab_")})
    plan = make_plan(photo, old, feeling_of(None), Settings(work=96))            # a card from before still works
    assert any("moved" in line for line in plan.report)


def test_famous_paintings_can_be_asked_for_by_name(tmp_path):
    names = style_names()
    assert len(names) == 14 and names["mona lisa"].startswith("Mona Lisa, Leonardo da Vinci")
    for path in STYLES.glob("*.json"):                            # measurements only, no pictures
        data = json.loads(path.read_text())
        assert path.stat().st_size < 3000 and len(data["palette"]) == 12 and data["source"] and data["measured"]
    mona, bedroom, wave = named_style("Mona Lisa"), named_style("the bedroom"), named_style("great wave")
    assert mona.hardness < 0.3 and mona.softness() > 2.5 and mona.texture < 0.3 and mona.outline == 0.0
    assert bedroom.outline == 1.0 and bedroom.texture > 0.8 and wave.outline > 0.25 and wave.hardness > 0.8
    vermeer = named_style("vermeer")                              # smooth paint, but his edges are not smoked
    assert vermeer.texture < 0.3 and vermeer.hardness > mona.hardness + 0.3
    # Van Gogh builds his picture from marks of paint; Hokusai's print has fine lines but its colour lies flat
    assert named_style("starry night").marks() > 0.5 and wave.marks() == 0.0 and mona.marks() < 0.2
    assert StyleCard().marks() == 0.0
    assert named_style("vermeer").name.startswith("Girl with a Pearl Earring")
    assert named_style("van gogh").name.startswith("The Starry Night")
    assert named_style("a thing nobody painted") is None
    img = _shapes(3, 96)
    soft = paint(img, style="mona lisa", work=96, steps=0.25)
    assert any("edges kept soft" in line for line in soft.plan.report) and soft.plan.card.name.startswith("Mona")
    with pytest.raises(FileNotFoundError, match="mona lisa"):
        paint(img, style="a thing nobody painted", work=96)
    save_image(img, tmp_path / "mine.png")                        # and a picture of one's own is read as before
    own = paint(img, style=str(tmp_path / "mine.png"), work=96, steps=0.25)
    assert own.plan.card.measured and own.plan.card.name == ""


def test_the_painter_works_the_way_the_style_does():
    img = _shapes(5, 96)
    soft = paint(img, style="mona lisa", work=96, steps=0.25, seed=1)
    dashed = paint(img, style="the starry night", work=96, steps=0.25, seed=1)
    # Leonardo's edges are soft, Van Gogh's are not
    assert float(soft.rows[:, A_PROFILE][:, 0].mean()) < float(dashed.rows[:, A_PROFILE][:, 0].mean()) - 0.15
    # Van Gogh's paint shows all over: a last pass of short marks, which Leonardo's does not get
    assert any("short marks all over" in line for line in dashed.plan.report)
    assert not any("short marks" in line for line in soft.plan.report)
    plain = paint(img, style="the starry night", work=96, steps=0.25, seed=1, style_strength=0.0)
    assert len(dashed) > len(plain) + 30 and not any("short marks" in line for line in plain.plan.report)
    assert float(dashed.render(96, relief=0.0).std()) > 0         # and it renders
    # Hokusai draws round his shapes: the border between two planes goes dark
    two = torch.full((3, 64, 64), 0.75)
    two[:, :, 32:] = 0.45
    drawn = draw_outlines(two, two, torch.tensor([0.05, 0.05, 0.2]), 1.0)
    assert float(drawn[:, 10:54, 30:34].mean()) < 0.3 and float((drawn - two)[:, :, :20].abs().max()) < 0.02
    wave = make_plan(img, named_style("great wave"), feeling_of(None), Settings(work=96))
    assert any("drawn round" in line for line in wave.report)


def test_one_training_plan_in_four_is_a_painting_copied_as_it_is(tmp_path):
    photos, paintings = [], []
    for i in range(2):
        save_image(_shapes(10 + i, 96), tmp_path / f"photo{i}.png")
        save_image(_shapes(20 + i, 96), tmp_path / f"painting{i}.png")
        card = read_style(tmp_path / f"painting{i}.png")
        (tmp_path / f"painting{i}.card.json").write_text(json.dumps(card.to_json()))
        photos.append(tmp_path / f"photo{i}.png")
        paintings.append(tmp_path / f"painting{i}.png")
    stream = iter(PlanStream(photos, paintings, work=96, seed=0, copies=1.0))
    plan = next(stream)
    assert plan.settings.style_strength == 0.0 and plan.settings.abstraction == 0.0
    assert float((plan.target - plan.source).abs().max()) < 1e-3  # the painting itself is what is aimed at
    mixed = iter(PlanStream(photos, paintings, work=96, seed=0))
    kinds = [next(mixed).settings.style_strength == 0.0 for _ in range(24)]
    assert 2 <= sum(kinds) <= 12                                  # about one in four
