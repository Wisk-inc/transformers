# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Pinxit's canvas, hand and mind: strokes that carry pressure and speed, two renderers that agree, restraint,
composition, pushed colour, style and feeling read from a picture."""

import math

import torch

from helix_lm.paint.color import extract_palette, linear_to_oklab, mix_pigments, oklab_to_srgb, srgb_to_oklab
from helix_lm.paint.emotion import EMOTIONS, feeling_of, read_emotion, strongest
from helix_lm.paint.mind import Settings, compose, composition_score, finish_map, make_plan, working_size
from helix_lm.paint.painting import paint
from helix_lm.paint.render import composite_level, render_painting, render_windows
from helix_lm.paint.strokes import ABS, Strokes, geometry, inverse_geometry, random_strokes, to_absolute
from helix_lm.paint.style import STYLE_DIM, own_card, read_style


def _scene(h=96, w=128):
    """A dull field with one bright warm square left of centre and a dark band at the bottom."""
    g = torch.Generator().manual_seed(0)
    img = torch.rand(3, h, w, generator=g) * 0.08 + torch.tensor([0.30, 0.38, 0.50])[:, None, None]
    img[:, h // 4 : h // 2, w // 6 : w // 3] = torch.tensor([0.95, 0.70, 0.15])[:, None, None]
    img[:, 3 * h // 4 :] *= 0.35
    return img.clamp(0, 1)


def test_colour_round_trip_and_paint_mixes_like_paint():
    c = torch.rand(50, 3)
    assert torch.allclose(oklab_to_srgb(srgb_to_oklab(c)), c, atol=1e-4)
    blue, yellow = torch.tensor([0.05, 0.15, 0.7]), torch.tensor([0.9, 0.8, 0.05])
    both = torch.stack([blue, yellow])
    mixed = mix_pigments(both, torch.tensor([0.5, 0.5]))
    assert torch.allclose(mix_pigments(both, torch.tensor([1.0, 0.0])), blue.clamp(0.004, 0.996), atol=1e-5)
    assert float(mixed[1]) > float(mixed[0]) and float(mixed[1]) > float(mixed[2])       # green, not grey
    assert float(mixed.mean()) < float(both.mean())                                      # darker than the average
    img = torch.zeros(3, 40, 40)
    img[0, :, :20], img[2, :, 20:] = 1.0, 1.0
    colours, shares = extract_palette(img, 2)
    assert abs(float(shares[0]) - 0.5) < 0.05 and float((colours.max(1).values > 0.95).float().mean()) == 1.0


def test_the_two_renderers_paint_the_same_picture():
    g = torch.Generator().manual_seed(1)
    gy, gx, n, r = 3, 5, 4, 32
    s = random_strokes((gy * gx, n), g)
    alpha, rgb = render_windows(s, r, segments=8)
    ground = torch.tensor([0.2, 0.15, 0.1])
    canvas = ground[None, :, None, None].expand(1, 3, gy * 16, gx * 16)
    fast = composite_level(canvas, alpha.reshape(1, gy, gx, n, r, r), rgb.reshape(1, gy, gx, n, 3, r, r))[0]
    cy, cx = torch.meshgrid(torch.arange(gy), torch.arange(gx), indexing="ij")
    rows = to_absolute(s, cy.reshape(-1), cx.reshape(-1), 16, gx * 16, level=0)
    slow, _ = render_painting(rows, gx * 16, gy * 16, ground, segments=8, relief=0.0, tremor=False)
    assert rows.shape == (gy * gx * n, ABS) and float((fast - slow).abs().max()) < 1e-4
    big, height = render_painting(rows, 400, 240, ground)                    # any size, with the paint's relief
    assert big.shape == (3, 240, 400) and float(height.max()) > 0


def _one(pressure, speed, brush=0):
    geom = inverse_geometry(torch.tensor([[0.5]]), torch.tensor([[0.5]]), torch.tensor([[0.18]]),
                            torch.tensor([[0.05]]), torch.tensor([[0.0]]), 0.5, 1.0, 1.0, pressure, speed)
    b = torch.zeros(1, 1, 6)
    b[..., brush] = 1.0
    s = Strokes(geom, b, torch.ones(1, 1, 3), torch.ones(1, 1, 3), torch.zeros(1, 1), torch.ones(1, 1),
                torch.zeros(1, 1, 5))
    return render_windows(s, 64, segments=8)[0][0, 0]


def test_pressure_and_speed_leave_their_marks():
    heavy, light = _one(0.95, 0.3), _one(0.05, 0.3)
    assert float(heavy.sum()) > 1.3 * float(light.sum())                     # pressed: wider and denser
    wide = lambda a: float((a[:, 30] > 0.3).sum())                           # noqa: E731
    assert wide(heavy) > wide(light)
    fast, slow = _one(0.5, 0.95, brush=3), _one(0.5, 0.05, brush=3)
    start = lambda a: float(a[:, 14:22].sum())                               # noqa: E731
    end = lambda a: float(a[:, 42:50].sum())                                 # noqa: E731
    assert end(fast) / start(fast) < 0.6 * end(slow) / start(slow)           # a fast stroke runs dry and flicks out
    heavy_slow = _one(0.9, 0.05)[32]                                          # along the middle of the stroke
    assert float(heavy_slow[21:25].max()) > float(heavy_slow[30:34].max())    # paint pooled where the brush landed


def test_a_stroke_can_be_found_by_gradient_descent():
    target = _one(0.5, 0.5)
    geom = inverse_geometry(torch.tensor([[0.42]]), torch.tensor([[0.56]]), torch.tensor([[0.15]]),
                            torch.tensor([[0.05]]), torch.tensor([[0.4]]), 0.5, 1.0, 1.0, 0.5, 0.5)
    logit = torch.logit(geom.clamp(1e-3, 1 - 1e-3)).requires_grad_()
    b = torch.zeros(1, 1, 6)
    b[..., 0] = 1.0
    opt = torch.optim.Adam([logit], lr=0.05)
    first = None
    for _ in range(80):
        s = Strokes(torch.sigmoid(logit), b, torch.ones(1, 1, 3), torch.ones(1, 1, 3), torch.zeros(1, 1),
                    torch.ones(1, 1), torch.zeros(1, 1, 5))
        loss = (render_windows(s, 64, segments=8)[0][0, 0] - target).abs().mean()
        first = float(loss) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.35 * first
    assert float(geometry(torch.sigmoid(logit))["p0"].min()) >= 0.0          # and never leaves its window


def test_the_mind_composes_distorts_and_holds_back():
    img = _scene()
    plan = make_plan(img, own_card(img), feeling_of("awe"), Settings(work=128, abstraction=0.0))
    h, w = plan.target.shape[-2:]
    assert (h, w) == working_size(96, 128, 128)[2:] and abs(plan.focal[1] - 0.25) < 0.2
    assert any("warmed" in line for line in plan.report) and any("restraint" in line for line in plan.report)
    # warm lights, cool shadows: the b (yellow-blue) gap between lights and darks grows
    def gap(picture):
        lab = srgb_to_oklab(picture.permute(1, 2, 0))
        light = lab[..., 0] > lab[..., 0].median()
        return float(lab[..., 2][light].mean() - lab[..., 2][~light].mean())
    assert gap(plan.target) > gap(plan.source)
    still = make_plan(img, own_card(img), feeling_of(None),
                      Settings(work=128, compose=0.0, exaggerate=0.0, abstraction=0.0))
    assert float((still.target - still.source).abs().max()) < 1e-3          # every move can be switched off
    loose, tight = finish_map(plan.focus, 0.9), finish_map(plan.focus, 0.1)
    assert float(loose.mean()) < float(tight.mean()) and float(loose[0, 0]) < float(loose[h // 2, w // 2])
    centred = torch.full((64, 64), 1e-6)
    centred[30:34, 30:34] = 1.0
    moved, note = compose(centred, 1.0)
    assert moved != (0.0, 0.0, 1.0, 1.0) and "third" in note                 # a dot dead centre is moved to a third
    assert composition_score(centred, moved)["thirds"] < composition_score(centred, (0, 0, 1, 1))["thirds"]
    busy = torch.rand(64, 64, generator=torch.Generator().manual_seed(0))    # nothing stands out: nothing is cut
    assert compose(busy, 1.0)[0] == (0.0, 0.0, 1.0, 1.0)


def test_a_painting_is_strokes_and_restraint_shows():
    img = _scene(96, 96)                                                      # square: four passes at this size
    done = paint(img, emotion="majesty", work=128, steps=0.3, seed=3)
    assert 100 < len(done) <= 2600 and done.rows.shape[1] == ABS
    summary = done.summary()
    assert summary["pressure"]["spread"] > 0.05 and summary["speed"]["spread"] > 0.05   # no two gestures alike
    from helix_lm.paint.bench import judge

    held = judge(paint(img, emotion="majesty", work=128, steps=0.3, seed=3, restraint=0.95))
    free = judge(paint(img, emotion="majesty", work=128, steps=0.3, seed=3, restraint=0.0))
    assert held["last_pass"] < free["last_pass"] and held["left_broad"] > free["left_broad"]
    assert free["closeness"] < 0.08
    again = paint(img, emotion="majesty", work=128, steps=0.3, seed=3)
    assert torch.equal(again.rows, done.rows)                                 # the same seed: the same painting


def test_style_and_feeling_are_read_from_a_picture():
    g = torch.Generator().manual_seed(4)
    s = random_strokes((12, 8), g)
    s.geom[..., 4] = 0.25 + 0.02 * torch.rand(12, 8, generator=g)             # every stroke runs down the canvas
    s.color = torch.tensor([0.02, 0.05, 0.30]).expand(12, 8, 3).clone()
    s.color2 = s.color
    alpha, rgb = render_windows(s, 32, 8)
    canvas = torch.tensor([0.9, 0.5, 0.05])[None, :, None, None].expand(1, 3, 48, 64)
    picture = composite_level(canvas, alpha.reshape(1, 3, 4, 8, 32, 32), rgb.reshape(1, 3, 4, 8, 3, 32, 32))[0]
    from helix_lm.paint.color import linear_to_srgb

    card = read_style(linear_to_srgb(picture), name="made here")
    assert card.vector().shape == (STYLE_DIM,) and card.measured and card.flow == "vertical"
    lab = linear_to_oklab(card.palette_linear())
    assert float(lab[:, 2].max()) > 0.08 and float(lab[:, 2].min()) < -0.05   # the orange and the blue are both there
    assert set(card.emotion) == set(EMOTIONS) and math.isclose(sum(card.emotion.values()), 1.0, abs_tol=1e-4)
    bright = torch.ones(3, 16, 16) * torch.tensor([1.0, 0.85, 0.3])[:, None, None]
    dark = torch.ones(3, 16, 16) * torch.tensor([0.12, 0.16, 0.25])[:, None, None]
    assert strongest(read_emotion(dark)) in ("melancholy", "mystery", "fear")
    assert strongest(read_emotion(bright)) in ("joy", "tenderness", "serenity")
    assert feeling_of("majestic").flow == "toward" and feeling_of(None).tilt == 0.0
