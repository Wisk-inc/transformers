# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Anime drawn the way a studio draws it: the background painted once and held, the character inked and
painted as a cel on each drawing, timed from how it moves, its lines named from drawing to drawing and
steadied through time."""

import torch

from helix_lm.paint.animate import Animation, animate
from helix_lm.paint.anime import BACKGROUND_MARKERS, carry_names, on_camera, still_camera, then
from helix_lm.paint.draw import CELS, Line, render
from helix_lm.paint.video import Animator


def scene(t: int, size: int = 96) -> torch.Tensor:
    """A sky, grass, and an orange ball with a dark outline and an eye, rolling right."""
    img = torch.ones(3, size, size) * torch.tensor([0.45, 0.7, 0.95])[:, None, None]
    img[:, int(0.72 * size):] = torch.tensor([0.35, 0.7, 0.3])[:, None, None]
    yy, xx = torch.meshgrid(torch.arange(size).float(), torch.arange(size).float(), indexing="ij")
    cx, cy, r = 0.3 * size + 3 * t, 0.5 * size, 0.16 * size
    d = ((xx - cx) ** 2 + (yy - cy) ** 2).sqrt()
    img[:, d < r] = 0.08
    img[:, d < r - 2.5] = torch.tensor([0.95, 0.5, 0.1])[:, None]
    img[:, ((xx - cx - 0.35 * r) ** 2 + (yy - cy + 0.2 * r) ** 2).sqrt() < 0.22 * r] = 0.05
    return img


def test_a_clip_is_drawn_as_anime_with_the_background_painted_once(tmp_path):
    clip = torch.stack([scene(t) for t in range(6)])
    said = []
    anim = animate(video=(clip, 12.0), work=96, depth=None, details=False, log=said.append)
    assert anim.settings["style"] == "anime" and anim.background and anim.holds
    assert sum(anim.holds) == 6                                    # every frame of the clip is shown
    assert all(line.marker < 0 or line.marker >= BACKGROUND_MARKERS for line in anim.background)
    assert any(line.kind in CELS for line in anim.background)
    kinds = {line.kind for sheet in anim.sheets for line in sheet.lines}
    assert "cel" in kinds and "line" in kinds                      # the character inked and painted
    assert sum(sheet.kept for sheet in anim.sheets[1:]) > 0        # lines named from drawing to drawing
    assert any("painted the background once" in s for s in said)
    picture = anim.frame(0, 96)
    board = torch.tensor([0.955, 0.958, 0.962])[:, None, None]
    assert float(((picture - board).abs().max(0).values < 0.01).float().mean()) < 0.01   # nothing left bare
    orange = (picture[0] > 0.7) & (picture[1] > 0.3) & (picture[1] < 0.7) & (picture[2] < 0.3)
    assert int(orange.sum()) > 200                                 # the ball keeps its colour, not washed out
    assert len(anim.settings["checks"]) == len(anim) and anim.settings["checks"][0]["error"] < 0.1
    back = Animation.load(anim.save(tmp_path / "ball.pinxa"))
    assert back.settings["style"] == "anime" and back.holds == anim.holds and back.cameras is not None
    assert torch.allclose(back.frame(2, 96), anim.frame(2, 96), atol=1e-5)


def test_a_line_keeps_its_name_when_it_only_moved():
    a = Line(torch.tensor([[0.1, 0.5], [0.3, 0.5], [0.5, 0.52]]), torch.ones(3), "pen", (0.1, 0.1, 0.1))
    b = Line(a.points + torch.tensor([3.0, 0.0]) / 100, torch.ones(3), "pen", (0.1, 0.1, 0.1))
    far = Line(a.points + torch.tensor([0.0, 0.3]), torch.ones(3), "pen", (0.1, 0.1, 0.1))
    flow = torch.zeros(2, 100, 100)
    flow[0] = 3.0                                                  # everything moved 3 pixels right
    assert carry_names([a], [b, far], flow, 100.0) == 1
    assert b.uid == a.uid and b.age == 1 and far.uid != a.uid


def test_the_background_moves_with_the_camera():
    square = torch.tensor([[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.4]])
    sky = Line(square, torch.ones(4), "brush", (0.2, 0.4, 0.9), marker=BACKGROUND_MARKERS, kind="cel", rings=(4,))
    pan = torch.tensor([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0]])
    assert still_camera(torch.eye(2, 3)) and not still_camera(pan)
    moved = on_camera([sky], pan, 100)[0]
    assert torch.allclose(moved.points, square + torch.tensor([0.1, 0.0]))
    twice = then(pan, pan)
    assert torch.allclose(twice, torch.tensor([[1.0, 0.0, 20.0], [0.0, 1.0, 0.0]]))
    picture = render([moved], 100, 100)
    assert float(picture[0, 30, 35]) < 0.3 and float(picture[0, 30, 25]) > 0.9      # sky moved, board left


def test_an_anime_clip_goes_on_with_the_same_background_and_colours():
    class Rolling:
        def __call__(self, **kwargs):
            rolled = torch.stack([scene(5 + t) for t in range(len(kwargs["video"]))])
            return type("Out", (), {"frames": rolled[None]})()

    clip = torch.stack([scene(t) for t in range(6)])
    anim = animate(video=(clip, 12.0), work=96, depth=None, details=False)
    background, palette, before = anim.background, anim.palette.clone(), len(anim)
    animator = Animator(None, "animatediff", "fake", 4, 2.0, 12, 5, (96, 96), Rolling())
    anim.extend("it rolls on", animator=animator)
    assert len(anim) > before and anim.background is background and torch.equal(anim.palette, palette)
    assert len(anim.holds) == len(anim) and len(anim.cameras) == len(anim)
    assert anim.history[-1]["op"] == "extend" and anim.history[-1]["style"] == "anime"


def test_an_animation_plays_at_any_frame_rate_with_its_timing_kept(tmp_path):
    from helix_lm.paint.animate import Sheet

    dot = Line(torch.tensor([[0.2, 0.2], [0.3, 0.2]]), torch.ones(2), "pen", (0.1, 0.1, 0.1))
    sheets = [Sheet([dot.moved(dot.points + 0.1 * i)]) for i in range(3)]
    anim = Animation(sheets, torch.tensor([[0.5, 0.5, 0.5]]), None, 12.0, holds=[2, 1, 3])
    assert len(list(anim.play(32))) == 6                           # half a second at 12 a second
    at24 = list(anim.play(32, fps=24))
    assert len(at24) == 12 and torch.equal(at24[0], at24[3]) and not torch.equal(at24[3], at24[4])
    from PIL import Image

    gif = Image.open(anim.save(tmp_path / "at24.gif", width=32, fps=24))
    shown = []
    for k in range(gif.n_frames):                                  # repeated frames are joined into longer ones
        gif.seek(k)
        shown.append(gif.info.get("duration", 0))
    assert abs(sum(shown) - 500) <= 30                             # still half a second


def test_a_shine_on_skin_is_a_lighter_skin_not_a_spot_of_sky():
    from helix_lm.paint.anime import tidy_spots

    square = torch.tensor([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
    skin = Line(square, torch.ones(4), "brush", (0.93, 0.72, 0.62), marker=2, kind="cel", rings=(4,))
    dot = Line(square * 0.02 + 0.5, torch.ones(4), "brush", (0.75, 0.92, 0.98), marker=0, kind="cel", rings=(4,))
    red = Line(square * 0.02 + 0.3, torch.ones(4), "brush", (0.5, 0.1, 0.1), marker=1, kind="cel", rings=(4,))
    assert tidy_spots([skin, dot, red], 200, 200) == 1
    r, g, b = dot.colour
    assert r > g > b and sum(dot.colour) > sum(skin.colour)        # a lighter skin tone, warm, not blue
    assert red.colour == (0.5, 0.1, 0.1)                           # a darker mark (a mole, a lash) is left alone
