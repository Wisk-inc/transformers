# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""What keeps a painting from looking like a filter over a photo: the marks of a brush's hairs run along the
stroke and end ragged, the canvas has a weave that a dry brush skips, a saved picture shows the bare edge of
the canvas, and the photo is seen in planes and tones before a stroke is made."""

import pytest
import torch

from helix_lm.paint import paint
from helix_lm.paint.brushes import BRUSHES, brush_profile, one_hot
from helix_lm.paint.color import image_to_oklab
from helix_lm.paint.mind import simplify
from helix_lm.paint.render import canvas_weave, render_painting, stroke_fields, window_grid
from helix_lm.paint.see import gaussian_blur
from helix_lm.paint.strokes import (
    A_ALPHA,
    A_BRUSH,
    A_COLOR,
    A_COLOR2,
    A_PHASE,
    A_PRESSURE,
    A_PROFILE,
    A_R0,
    A_R1,
    A_SPEED,
    ABS,
)


def _stroke(brush: str, pressure: float, speed: float, size: int = 96, radius: float = 9.0, seed: int = 0):
    """One straight stroke from left to right across the middle of a window: its fields, each ``(size, size)``."""
    gx, gy = window_grid(size)
    pts = torch.stack([torch.linspace(0.2 * size, 0.75 * size, 9), torch.full((9,), size / 2.0)], -1)[None]
    profile = brush_profile(one_hot(torch.tensor([BRUSHES.index(brush)])))
    phase = torch.rand(1, 5, generator=torch.Generator().manual_seed(seed))
    f = stroke_fields(pts, torch.tensor([radius]), torch.tensor([radius]), profile, torch.tensor([pressure]),
                      torch.tensor([speed]), phase, gx, gy)
    return {k: v.reshape(size, size) for k, v in f.items()}


def test_the_hairs_of_a_brush_leave_tracks_along_the_stroke_and_not_round_its_end():
    f = _stroke("dry", pressure=0.4, speed=0.5)
    tone, rows = f["tone"], slice(40, 57)                         # across the brush, inside its width
    middle, late = tone[rows, 45], tone[rows, 62]
    assert float(middle.std()) > 0.02                             # hair to hair the tone differs
    # the same hair is the same tone further along (the stroke's width is not quite even, so the tracks shift a
    # little: 0.8, not 1)
    assert float(torch.corrcoef(torch.stack([middle, late]))[0, 1]) > 0.8
    along = tone[48, 30:66]
    assert float(along.std()) < 0.2 * float(middle.std())         # along one hair it does not change
    # and the middle hair's track runs straight on past the stroke's end (which is at 72): were the marks rings
    # round the tip, the tone would change there as it does from hair to hair
    assert float(tone[48, 73:81].std()) < 0.2 * float(middle.std())


def test_a_fast_dry_stroke_ends_ragged_and_a_knife_ends_clean():
    def ends(f):                                                  # where the paint stops, hair by hair
        paint = f["paint"][41:56]
        there = paint > 0.25 * paint.max()
        return torch.tensor([float(row.nonzero().max()) for row in there if bool(row.any())])

    ragged, clean = ends(_stroke("dry", 0.3, 0.9)), ends(_stroke("knife", 0.8, 0.2))
    assert float(ragged.std()) > 3.0 * float(clean.std()) + 1.0 and float(clean.std()) < 1.5
    heavy, light = _stroke("flat", 0.9, 0.3), _stroke("flat", 0.2, 0.3)
    assert float(heavy["height"].max()) > 1.5 * float(light["height"].max())           # pressure leaves thick paint
    assert float(_stroke("dry", 0.3, 0.9)["dry"].max()) > 3 * float(_stroke("wash", 0.3, 0.2)["dry"].max())


def _row(x0, x1, y, radius, brush, color, pressure=0.6, speed=0.4, alpha=0.95):
    row = torch.zeros(ABS)
    row[0], row[1], row[2], row[3], row[4], row[5] = x0, y, (x0 + x1) / 2, y, x1, y
    row[A_R0] = row[A_R1] = radius
    row[A_ALPHA], row[A_PRESSURE], row[A_SPEED] = alpha, pressure, speed
    row[A_PROFILE] = brush_profile(one_hot(torch.tensor(BRUSHES.index(brush))))
    row[A_COLOR] = row[A_COLOR2] = torch.tensor(color)
    row[A_PHASE] = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    row[A_BRUSH] = BRUSHES.index(brush)
    return row


def test_the_canvas_has_a_weave_that_shows_through_thin_paint_and_that_a_dry_brush_skips():
    cloth = canvas_weave(120, 160, 4.0)
    assert float(cloth.min()) >= 0.0 and float(cloth.max()) <= 1.0 and float(cloth.std()) > 0.15
    ground, none = torch.tensor([0.4, 0.35, 0.3]), torch.zeros(0, ABS)
    bare, _ = render_painting(none, 1200, 600, ground, weave=0.6)
    smooth, _ = render_painting(none, 1200, 600, ground, weave=0.0)
    assert float(bare[0].std()) > 0.005 and float(smooth[0].std()) < 1e-6  # cloth, or a smooth board
    small, _ = render_painting(none, 300, 150, ground, weave=0.6)          # threads too fine to draw: left out
    assert float(small[0].std()) < 1e-6
    knife = _row(0.1, 0.9, 0.12, 0.05, "knife", (0.9, 0.9, 0.9), pressure=0.9, speed=0.1)[None]
    thick, height = render_painting(knife, 1200, 600, ground, weave=0.6)
    board, _ = render_painting(knife, 1200, 600, ground, weave=0.0)
    inside, beside = (slice(None), slice(110, 180), slice(300, 900)), (slice(None), slice(300, 370), slice(300, 900))
    assert float(height[inside[1:]].mean()) > 0.3
    # what the cloth changes in the picture: plenty on the bare canvas, little under thick paint
    assert float((thick - board)[inside].abs().mean()) < 0.4 * float((thick - board)[beside].abs().mean())
    dry = _row(0.1, 0.9, 0.35, 0.05, "dry", (0.95, 0.95, 0.95), pressure=0.2, speed=0.95)[None]
    skipped, _ = render_painting(dry, 1200, 600, ground, weave=0.9, relief=0.0)
    even, _ = render_painting(dry, 1200, 600, ground, weave=0.0, relief=0.0)
    cloth = canvas_weave(600, 1200, 1200 / 420.0)
    late = (slice(390, 450), slice(700, 1000))                    # the far end of the stroke, where it runs dry
    tops, hollows = cloth[late] > 0.7, cloth[late] < 0.2
    assert float(skipped[0][late][tops].mean()) > float(skipped[0][late][hollows].mean()) + 0.02
    assert abs(float(even[0][late][tops].mean()) - float(even[0][late][hollows].mean())) < 0.01


@pytest.fixture(scope="module")
def painted():
    g = torch.Generator().manual_seed(0)
    img = torch.rand(3, 96, 128, generator=g) * 0.1 + torch.tensor([0.30, 0.42, 0.55])[:, None, None]
    img[:, 30:66, 44:90] = torch.tensor([0.85, 0.55, 0.20])[:, None, None]
    return paint(img.clamp(0, 1), emotion="joy", work=128, steps=0.25, seed=1)


def test_a_saved_picture_shows_the_bare_edge_of_the_canvas(painted, tmp_path):
    from helix_lm.paint.io import load_image

    plain = painted.render(320)
    assert plain.shape == (3, round(320 * painted.aspect), 320)   # the plan's frame, exactly
    edged = painted.render(320, border=0.05)
    scale = 320 / 1.1
    assert edged.shape == (3, round(scale * (painted.aspect + 0.1)), 320)
    ground = torch.tensor(painted.plan.ground).pow(1 / 2.2)
    corner, middle = edged[:, :4, :4].mean((-1, -2)), edged[:, 100:140, 120:200]
    assert float((corner - ground).abs().max()) < 0.08            # the corner is bare, toned canvas
    assert float((middle - ground[:, None, None]).abs().mean()) > 0.05
    painted.save(tmp_path / "p.png", width=300)                   # a saved picture has the edge by default
    saved = load_image(tmp_path / "p.png")
    assert saved.shape[-1] == 300 and saved.shape[-2] > round(300 * painted.aspect)
    assert painted.plan.settings.border == 0.025 and painted.plan.settings.weave == 0.6


def test_the_photo_is_seen_in_planes_and_tones_and_the_subject_keeps_more():
    g = torch.Generator().manual_seed(0)
    img = torch.full((3, 96, 128), 0.35)
    img[:, :, 64:] = 0.7                                          # two shapes with a clear border
    img = img + torch.linspace(-0.08, 0.08, 96)[None, :, None]    # a smooth gradient down each
    img = (img + 0.05 * torch.randn(3, 96, 128, generator=g)).clamp(0, 1)          # and a photo's fine detail
    nowhere = torch.zeros(96, 128)
    out, note = simplify(img, nowhere, 0.7)
    assert "planes" in note and "tones" in note

    def detail(x):
        light = image_to_oklab(x)[0]
        return float((light - gaussian_blur(light, 1.5)).abs()[8:-8, 8:60].mean())

    assert detail(out) < 0.3 * detail(img)                        # the fine detail is levelled
    step = float(out[:, 20:76, 70:120].mean() - out[:, 20:76, 8:58].mean())
    assert step > 0.9 * 0.35 and float((out[:, 40, 66] - out[:, 40, 61]).abs().mean()) > 0.2   # the border is kept
    def unevenness(x):                                            # 1 for an even gradient, more for steps
        column = image_to_oklab(x)[0][8:-8, 30]
        jumps = (column[1:] - column[:-1]).abs()
        return float(jumps.max() / (jumps.mean() + 1e-9))

    even = img - 0.05 * torch.randn(3, 96, 128, generator=torch.Generator().manual_seed(0))
    assert unevenness(out) > 1.6 * unevenness(even.clamp(0, 1))   # the gradient has become steps of tone
    everywhere = torch.ones(96, 128)
    kept, _ = simplify(img, everywhere, 0.7)
    assert detail(kept) > 1.5 * detail(out)                       # at the subject more is kept
    same, nothing = simplify(img, nowhere, 0.0)
    assert same is img and nothing == ""
    soft, said = simplify(img, nowhere, 0.7, stepping=0.1)        # a style without steps keeps its gradations
    assert "gradations" in said and unevenness(soft) < 0.8 * unevenness(out)
