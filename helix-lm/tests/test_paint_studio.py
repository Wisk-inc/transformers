# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The making-of studio: a hand draws every frame on the board, the exposure sheet fills in, the finished
animation plays. Small pictures, so the tests are quick."""

import math

import pytest
import torch
from test_paint_animate import scene

from helix_lm.paint.animate import Animation, Sheet, animate, read_video
from helix_lm.paint.draw import CELS, TOOLS, Line
from helix_lm.paint.studio import EMPTY, Pose, Spring, Studio, hand_shapes, making_of, render, shots


@pytest.fixture(scope="module")
def tiny():
    clip = torch.stack([scene(t, 96) for t in range(3)])
    return animate(video=(clip, 6.0), work=96, colours=5, style="marker")


def _skin(picture: torch.Tensor) -> torch.Tensor:
    """Where a picture (bytes, (H, W, 3)) is the colour of the hand's skin: warm, light, red over green over
    blue."""
    p = picture.float()
    r, g, b = p[..., 0], p[..., 1], p[..., 2]
    green, blue = g / r.clamp(min=1.0), b / r.clamp(min=1.0)
    return (r > 120) & (green > 0.55) & (green < 0.9) & (blue > 0.4) & (blue < 0.82) & (r - b > 35)


def test_every_frame_is_drawn_then_the_animation_plays(tiny, tmp_path):
    fps, loops = 8, 1
    studio = Studio(tiny, 320, fps, first=1.5, later=0.75)
    phases, _, leave = studio.plan()
    assert [p.frame for p in phases] == [0, 1, 2]
    assert abs(phases[0].steps[-1].t1 - phases[0].steps[0].t0 - (0.8 + 1.5 + 0.45)) < 0.05   # comes in, draws
    assert abs(phases[1].steps[-1].t1 - phases[1].steps[0].t0 - (0.75 + 0.45)) < 0.05        # about `later`
    pictures = list(studio.shots(loops))
    drawing = math.floor(leave.t1 * fps + 1e-6)
    playing = round(loops * len(tiny) / tiny.fps * fps)          # 6 drawings a second on an 8 fps video
    assert len(pictures) == drawing + playing and [s.frame for s in pictures[drawing:]] == [None] * playing
    assert pictures[0].picture.shape == (180, 320, 3)
    path = making_of(tiny, tmp_path / "making.gif", width=320, fps=fps, first=1.5, later=0.75, loops=loops)
    shown, _ = read_video(path)
    distinct = 1 + sum(not torch.equal(a.picture, b.picture) for a, b in zip(pictures, pictures[1:], strict=False))
    assert shown.shape[-1] == 320 and shown.shape[-2] == 180 and shown.shape[0] == distinct


def test_the_hand_holds_the_tool_where_the_line_is_drawn(tiny):
    def drawing(hand: bool) -> list:
        return [s for s in shots(tiny, width=480, fps=8, first=2.0, later=0.5, loops=0, hand=hand)
                if s.doing == "drawing" and s.frame == 0]

    with_hand, tool_only = drawing(True), drawing(False)
    assert len(with_hand) == len(tool_only) >= 8
    reach = 45                                                    # some 0.6 hand lengths at this size
    for a, b in zip(with_hand, tool_only, strict=True):
        assert a.tip == b.tip                                     # the point is where the line is, either way
        x, y = (round(c) for c in a.tip)
        box = (slice(max(0, y - reach), y + reach), slice(max(0, x - reach), x + reach))
        added = _skin(a.picture)[box] & ~_skin(b.picture)[box]
        assert int(added.sum()) > 100                             # fingers round the point in every picture


def test_the_exposure_sheet_shows_the_frames_drawn_and_holds(tiny):
    tiny.holds = [2, 1, 3]
    try:
        studio = Studio(tiny, 640, 8, first=1.0, later=0.5)
        cells = studio._cells()
        assert len(cells) == 6 and [c[3] for c in cells] == [False, True, False, False, True, True]
        pictures = list(studio.shots(loops=1))
    finally:
        del tiny.holds
    cw, ch = studio.lay.cell
    during = next(s for s in pictures if s.frame == 2)            # drawing the third frame
    places = {k: (x, y) for k, x, y in studio.cell_places(2)}
    first = during.picture[places[0][1] + 2: places[0][1] + ch - 2, places[0][0] + 2: places[0][0] + cw - 2]
    assert float(first.float().mean()) > 150                     # frame 1 is a light thumbnail of the drawing
    early = next(s for s in pictures if s.frame == 0 and s.time > 0.3)
    places = {k: (x, y) for k, x, y in studio.cell_places(0)}
    x, y = places[3]                                              # frame 3 is not drawn yet: an empty cell
    assert early.picture[y + ch // 2, x + cw // 2].tolist() == list(EMPTY)
    x, y = places[1]                                              # frame 1 is held a second frame: a line
    middle = early.picture[y + ch // 2, x + 1: x + cw - 1].float().mean()
    assert float(middle) > 100 and early.picture[y + 3, x + cw // 2].tolist() != list(EMPTY)
    played = [s for s in pictures if s.frame is None]
    assert len(played) == round((2 + 1 + 3) / 6.0 * 8)            # the holds are kept when it plays


def test_cel_shapes_and_a_background_are_painted_with_the_brush():
    def ring(x0, y0, x1, y1):
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    pts = torch.tensor(ring(0.2, 0.2, 0.8, 0.8) + ring(0.4, 0.4, 0.6, 0.6), dtype=torch.float32)
    red, dark, light = (0.85, 0.2, 0.15), (0.55, 0.1, 0.1), (0.98, 0.6, 0.5)
    lines = [Line([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8], [0.2, 0.2]], [0.8] * 5, "pen", TOOLS["pen"].ink),
             Line(pts, torch.ones(8), "brush", red, 0, "cel", rings=(4, 4)),
             Line(torch.tensor(ring(0.6, 0.2, 0.8, 0.8)), torch.ones(4), "brush", dark, 1, "cel-shadow", rings=(4,)),
             Line(torch.tensor(ring(0.2, 0.2, 0.35, 0.35)), torch.ones(4), "brush", light, 2, "cel-light",
                  rings=(4,)),
             Line([[0.1, 0.9], [0.9, 0.9]], [0.75, 0.75], "marker", (0.2, 0.4, 0.9), 3, "fill"),
             Line([[0.1, 0.92], [0.9, 0.92]], [0.75, 0.75], "marker", (0.2, 0.4, 0.9), 3, "shade")]
    sky = Line(torch.tensor(ring(0.0, 0.0, 1.0, 0.15)), torch.ones(4), "brush", (0.6, 0.8, 0.95), 4, "cel",
               rings=(4,))
    palette = torch.tensor([red, dark, light, (0.2, 0.4, 0.9), (0.6, 0.8, 0.95)])
    anim = Animation([Sheet(lines), Sheet([line.moved(line.points + 0.01) for line in lines])], palette, None, 6.0,
                     settings={"size": [96, 96]})
    anim.background = [sky]
    studio = Studio(anim, 320, 8, first=2.0, later=1.0)
    assert studio.lay.reference is None                           # no video drawn from: no print
    assert [item.tool for item in studio.items] == ["brush", "pen", "marker"] and len(studio.pots) == 4
    pictures = list(studio.shots(loops=1))
    assert pictures[0].frame == -1 and {s.frame for s in pictures} >= {-1, 0, 1, None}
    dips = [s for s in pictures if s.doing == "changing tools" and s.holding and s.holding[0] == "brush"]
    assert dips                                                   # the brush goes to the pots for each colour
    x0, y0 = studio.origin
    last = pictures[-1].picture
    inside = last[y0 + round(0.3 * studio.bh), x0 + round(0.32 * studio.bw)].int()
    assert inside[0] > inside[1] + 60                             # the red cel, under the ink
    top = last[y0 + 3, x0 + studio.bw // 2].int()
    assert top[2] > top[0] + 20                                   # the background's sky, in every frame


def test_an_anime_animation_is_drawn_background_first_then_each_cel():
    clip = torch.stack([scene(3 * t, 96) for t in range(5)])
    anim = animate(video=(clip, 12.0), work=96, style="anime", depth=None, details=False)
    assert anim.background and any(line.kind in CELS for sheet in anim.sheets for line in sheet.lines)
    studio = Studio(anim, 320, 8, first=2.0, later=0.6)
    phases, _, _ = studio.plan()
    assert phases[0].frame == -1 and abs(phases[0].budget - 0.32 * 2.0) < 1e-6   # the background, quicker
    cells = studio._cells()
    assert cells[0][1] == "BG" and len(cells) == 1 + sum(anim.holds)
    pictures = list(studio.shots(loops=1))
    assert pictures[0].frame == -1 and [s.frame for s in pictures if s.frame is not None][-1] == len(anim) - 1
    assert any(s.holding and s.holding[0] == "brush" and s.doing == "drawing" for s in pictures if s.frame >= 0)
    for i in range(len(anim)):                                    # it plays the frames as the animation draws them
        want = (anim.frame(i, studio.bw) * 255 + 0.5).byte().permute(1, 2, 0)
        assert int((studio.played[i].int() - want.int()).abs().max()) <= 1
    playing = [s for s in pictures if s.frame is None]
    assert len(playing) == round(sum(anim.holds) / anim.fps * 8)


def test_the_hand_is_shaded_and_lags_on_springs():
    spring = Spring(2.0, 0.4)
    spring(0.0, 1 / 24)
    path = [float(spring(1.0, 1 / 24)) for _ in range(48)]
    assert max(path) > 1.02 and abs(path[-1] - 1.0) < 0.05        # overshoots, then settles
    unit = 120.0
    shapes = hand_shapes(Pose((100.0, 100.0), 0.0, -2.6, (0.0, 0.0), 0.6), unit, "marker", (0.9, 0.3, 0.2))
    g = render(shapes, 0, 0, 320, 240)
    assert float(g["cover"][99:102, 99:102].max()) > 0.5          # the point of the marker is at the tip
    assert float(g["height"].max()) > 0.3 * unit                 # the hand stands up off the board
    lifted = hand_shapes(Pose((100.0, 100.0), 0.3 * unit, -2.6, (0.0, 0.0), 0.6), unit, "marker", (0.9, 0.3, 0.2))
    assert float(render(lifted, 0, 0, 320, 240)["height"].max()) > float(g["height"].max()) + 0.25 * unit
