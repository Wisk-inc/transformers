# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Drawing with pens and markers the way a person does, and animation drawn frame by frame over the frame
before (the lightbox), so that a character keeps its look. The video model is a stand-in here; the real one
was run by hand (docs/PINXIT.md)."""

import torch

from helix_lm.paint.animate import Animation, Sheet, animate, draw_frames, read_video, trace_over, write_video
from helix_lm.paint.draw import (
    TOOLS,
    Board,
    Line,
    draw_picture,
    find_lines,
    hatch,
    look,
    markers,
    order,
    render,
    thin,
    timing,
    trace,
)
from helix_lm.paint.flow import nearest, optical_flow, warp
from helix_lm.paint.see import gaussian_blur
from helix_lm.paint.video import Animator, _frames


def scene(t: int, size: int = 160) -> torch.Tensor:
    """A cartoon: sky, grass, a sun, and a red ball with a black outline and a white eye rolling right."""
    img = torch.ones(3, size, size) * torch.tensor([0.75, 0.88, 0.98])[:, None, None]
    img[:, int(0.7 * size):] = torch.tensor([0.45, 0.75, 0.35])[:, None, None]
    yy, xx = torch.meshgrid(torch.arange(size).float(), torch.arange(size).float(), indexing="ij")
    img[:, ((xx - 0.78 * size) ** 2 + (yy - 0.2 * size) ** 2) < (0.09 * size) ** 2] = torch.tensor([1.0, 0.85,
                                                                                                    0.2])[:, None]
    cx, cy, r = 0.25 * size + 5 * t, 0.55 * size, 0.13 * size
    img[:, ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2] = 0.05
    img[:, ((xx - cx) ** 2 + (yy - cy) ** 2) < (r - 3) ** 2] = torch.tensor([0.9, 0.2, 0.15])[:, None]
    img[:, ((xx - cx - 0.3 * r) ** 2 + (yy - cy + 0.25 * r) ** 2) < (0.22 * r) ** 2] = 1.0
    return img


def test_a_pen_line_is_pressed_tapered_and_drawn_by_an_unsteady_hand():
    # points are in board widths: on a board 800 by 400 the middle is at y 0.25
    line = Line([[0.1, 0.25], [0.5, 0.25], [0.9, 0.25]], [0.3, 1.0, 0.3], "brush pen", (0.0, 0.0, 0.0))
    steady = render([line], 800, 400, waver=0.0)
    dark = steady[0] < 0.5
    middle, end = int(dark[:, 400].sum()), int(dark[:, 200].sum())
    assert middle > end >= 1                                  # heavier in the middle, thinner where it lifts
    wavy = render([line], 800, 400, waver=3.0, seed=1)
    assert not torch.equal(steady, wavy) and not torch.equal(wavy, render([line], 800, 400, waver=3.0, seed=2))
    assert torch.equal(wavy, render([line], 800, 400, waver=3.0, seed=1))     # the same seed, the same hand
    pencil = Line([[0.1, 0.25], [0.9, 0.25]], [0.5, 0.5], "pencil", TOOLS["pencil"].ink)
    paper = render([pencil], 800, 400, "paper", waver=0.0)
    board = render([pencil], 800, 400, "whiteboard", waver=0.0)
    row = paper[0, 199:201, 150:650]
    assert float(row.std()) > 0.015 and float(row.std()) > 1.5 * float(board[0, 199:201, 150:650].std())  # tooth


def test_marker_ink_blends_wet_and_darkens_a_dry_layer():
    colour = (0.9, 0.5, 0.2)
    a = Line([[0.1, 0.30], [0.9, 0.30]], [0.75, 0.75], "marker", colour, 0, "fill")
    b = Line([[0.1, 0.31], [0.9, 0.31]], [0.75, 0.75], "marker", colour, 0, "fill")
    later = Line([[0.1, 0.30], [0.9, 0.30]], [0.75, 0.75], "marker", colour, 0, "shade")
    one, wet, dry = Board(200, 120), Board(200, 120), Board(200, 120)
    one.draw(a)
    wet.draw(a), wet.draw(b)
    dry.draw(a), dry.draw(later)
    both = slice(59, 61)                                         # where the two strokes overlap
    assert float((wet.canvas[0, both, 100] - one.canvas[0, both, 100]).abs().max()) < 0.01
    assert float(dry.canvas[0, both, 100].max()) < float(one.canvas[0, both, 100].min()) - 0.03
    pen = Board(800, 480, waver=0.0)                             # a pen line is two pixels wide on this board
    pen.draw(Line([[0.1, 0.3], [0.9, 0.3]], [0.8, 0.8], "pen", (0.05, 0.05, 0.05)))
    pen.draw(a)                                                  # marker over ink: the ink stays black
    assert float(pen.canvas[:, 239:241, 400].max()) < 0.05


def test_lines_are_found_thinned_and_followed_like_a_pen():
    blob = torch.zeros(40, 60, dtype=torch.bool)
    blob[18:23, 5:55] = True
    thinned = thin(blob)
    assert int(thinned.sum(0).max()) == 1 and int(thinned.sum()) > 40     # one pixel wide, still there
    cross = torch.zeros(41, 41, dtype=torch.bool)
    cross[20, 2:39] = True
    cross[2:39, 20] = True
    paths = trace(cross, least=3)
    longest = max(paths, key=len)
    span = longest.max(0).values - longest.min(0).values
    assert len(paths) <= 3 and max(float(span[0]), float(span[1])) > 34 and min(float(span[0]), float(span[1])) < 2
    img = scene(0)
    lines, strength = find_lines(img)
    assert bool(lines[int(0.55 * 160), int(0.25 * 160 - 0.13 * 160) - 2: int(0.25 * 160 - 0.13 * 160) + 3].any())
    assert not bool(lines[5:30, 5:60].any())                     # the even sky has no lines


def test_a_shape_is_coloured_back_and_forth_up_to_its_border():
    width = 400
    nib = TOOLS["marker"].width * width
    mask = torch.zeros(320, width, dtype=torch.bool)
    mask[80:240, 80:320] = True
    paths = hatch(mask, 0.7, 0.62 * nib, nib)
    assert 1 <= len(paths) <= 40 and max(len(p) for p in paths) >= 12        # many rows in each stroke
    pts = torch.cat(paths)
    assert float(pts[:, 0].min()) > 78 and float(pts[:, 0].max()) < 322     # the nib turns inside the shape
    lines = [Line(p / width, torch.full((len(p),), 0.75), "marker", (0.2, 0.4, 0.9), 0, "fill") for p in paths]
    picture = render(lines, width, 320, waver=0.0)
    inside, outside = picture[:, 100:220, 100:300], picture[:, 5:60, 5:395]
    assert float(inside[2].mean() - inside[0].mean()) > 0.3 and float((inside[2] - inside[0]).min()) > 0.2
    assert float(outside.std()) < 0.02                           # nothing outside


def test_a_picture_is_drawn_lines_first_then_light_markers_before_dark():
    img = scene(0)
    palette = markers([img], 6)
    lines = draw_picture(img, palette)
    kinds = [line.kind for line in lines]
    assert kinds.index("fill") > max(i for i, k in enumerate(kinds) if k == "line")
    order_of_markers = [line.marker for line in lines if line.kind == "fill"]
    firsts = list(dict.fromkeys(order_of_markers))
    assert firsts == sorted(firsts)                              # one marker at a time, lightest first
    picture = render(lines, 320, 320)
    ball = picture[:, int(0.55 * 320) - 10 : int(0.55 * 320) + 10, int(0.25 * 320) - 10 : int(0.25 * 320) + 10]
    assert float(ball[0].mean() - ball[1].mean()) > 0.3          # the ball is red
    sky = picture[:, 20:60, 20:100]
    assert float(sky[2].mean() - sky[0].mean()) > 0.08           # the sky is blue
    times = timing(lines)
    assert all(b[0] >= a[1][-1] for a, b in zip(times, times[1:], strict=False))   # one line after another
    curve = Line([[0.1, 0.1], [0.2, 0.2], [0.1, 0.3], [0.2, 0.4]], [0.5] * 4, "pen", (0, 0, 0))
    straight = Line([[0.1, 0.1], [0.1, 0.2], [0.1, 0.3], [0.1, 0.4]], [0.5] * 4, "pen", (0, 0, 0))
    per_length = [(timing([x])[0][1][-1] - timing([x])[0][0]) / x.length() for x in (curve, straight)]
    assert per_length[0] > per_length[1]                         # slower round a curve
    first = order([straight.reversed(), curve])[0]
    assert float(first.points[0].norm()) <= 0.2 or first is not None


def test_how_a_frame_moves_is_measured():
    g = torch.Generator().manual_seed(0)
    base = torch.stack([gaussian_blur(torch.rand(128, 128, generator=g), 3.0) for _ in range(3)])
    base = (base - base.min()) / (base.max() - base.min())
    shift = torch.zeros(2, 128, 128)
    shift[0], shift[1] = -4.0, 2.5
    flow = optical_flow(base, warp(base, shift))[:, 20:-20, 20:-20]
    assert abs(float(flow[0].mean()) - 4.0) < 0.2 and abs(float(flow[1].mean()) + 2.5) < 0.2
    assert float(flow[0].std()) < 0.5
    mask = torch.zeros(30, 40, dtype=torch.bool)
    mask[5, 5] = True
    where, dist = nearest(mask)
    assert where[:, 20, 30].tolist() == [5.5, 5.5] and abs(float(dist[20, 30]) - (25 ** 2 + 15 ** 2) ** 0.5) < 1e-4


def test_each_frame_is_traced_over_the_last_so_the_character_keeps_its_look():
    clip = torch.stack([scene(t) for t in range(5)])
    said = []
    anim = animate(video=(clip, 8.0), work=160, colours=6, log=said.append)
    assert len(anim) == 5 and anim.sheets[0].fresh and not any(s.fresh for s in anim.sheets[1:])
    for sheet in anim.sheets[1:]:
        assert sheet.kept > 2 * sheet.new                        # most of it traced, a little drawn new
    assert "traced from the frame before" in said[1]
    red = [line for line in anim.sheets[-1].lines if line.kind == "fill" and line.colour[0] > 0.6 > line.colour[1]]
    assert red and all(line.colour == red[0].colour for line in red)          # the same marker in every frame
    ball = [line for line in anim.sheets[0].lines if line.kind == "fill" and line.marker == red[0].marker]
    ball_then = torch.cat([line.points for line in ball]).mean(0)
    ball_now = torch.cat([line.points for line in red]).mean(0)
    assert 0.1 < float(ball_now[0] - ball_then[0]) < 0.16 and abs(float(ball_now[1] - ball_then[1])) < 0.02
    assert any(line.age >= 3 for line in anim.sheets[-1].lines)               # the sun traced all along
    cut = torch.stack([scene(0), torch.flip(scene(0), [1, 2])])
    jumped, _ = draw_frames(cut, anim.palette)
    assert jumped[1].fresh                                       # a cut is drawn from nothing


def test_an_animation_is_kept_played_changed_and_shown_being_drawn(tmp_path):
    clip = torch.stack([scene(t, 96) for t in range(3)])
    anim = animate(video=(clip, 6.0), work=96, colours=5)
    anim.save(tmp_path / "a.pinxa")
    back = Animation.load(tmp_path / "a.pinxa")
    assert len(back) == 3 and torch.equal(back.frame(2, 96), anim.frame(2, 96)) and back.fps == 6.0
    assert torch.equal(back.frames, anim.frames)
    write_video(back.play(96), tmp_path / "a.gif", 6.0)
    frames, fps = read_video(tmp_path / "a.gif")
    assert frames.shape[0] == 3 and abs(fps - 6.0) < 0.5
    marker = back.marker_at((0.25, 0.55))
    back.recolor((0.1, 0.3, 0.9), marker=marker)
    blue = back.frame(1, 96)
    x, y = round(0.25 * 96 + 5), round(0.55 * 96)
    assert float(blue[2, y - 3 : y + 3, x - 3 : x + 3].mean()) > float(blue[0, y - 3 : y + 3, x - 3 : x + 3].mean())
    back.retool("brush pen")
    assert all(line.tool == "brush pen" for s in back.sheets for line in s.lines if line.kind == "line")
    path = back.making_of(tmp_path / "making.gif", width=160, fps=6, first=1.0, later=0.5, loops=1)
    shown, _ = read_video(path)
    assert shown.shape[0] > 3 + 6 and shown.shape[-1] == 160    # every frame being drawn, then the animation
    back.save(tmp_path / "frames.png", width=64)
    assert (tmp_path / "frames.png").stat().st_size > 1000


def test_the_video_model_is_asked_for_a_drawable_clip():
    class FakePipe:
        def __init__(self):
            self.asked = []

        def __call__(self, **kwargs):
            self.asked.append(kwargs)
            return type("Out", (), {"frames": torch.rand(1, kwargs["num_frames"], 3, kwargs["height"],
                                                         kwargs["width"])})()

    pipe, still = FakePipe(), FakePipe()
    animator = Animator(pipe, "animatediff", "fake", 6, 2.0, 8, 4, (64, 64), still)
    clip = animator("a fox", seed=1)
    assert clip.shape == (4, 3, 64, 64) and pipe.asked[-1]["num_inference_steps"] == 6
    still.__call__ = None
    anim = animate("a fox running", animator=animator, work=64, colours=4)
    assert "2d cartoon animation, clean bold outlines" in pipe.asked[-1]["prompt"] and len(anim) == 4
    pictures = [torch.rand(32, 48, 3).numpy() for _ in range(2)]
    assert _frames(type("Out", (), {"frames": [pictures]})()).shape == (2, 3, 32, 48)
    assert _frames(torch.rand(1, 3, 5, 32, 48) * 2 - 1).shape == (5, 3, 32, 48)
    assert float(_frames(torch.rand(1, 5, 3, 8, 8) * 2 - 1).min()) >= 0.0
    character = torch.rand(3, 64, 64)

    class StillPipe(FakePipe):
        def __call__(self, **kwargs):
            self.asked.append(kwargs)
            return type("Out", (), {"frames": torch.rand(1, len(kwargs["video"]), 3, 64, 64)})()

    animator.still = StillPipe()
    animator("a fox", character=character)
    assert len(animator.still.asked[-1]["video"]) == 4 and animator.still.asked[-1]["strength"] == 0.75


def test_a_fast_video_is_drawn_on_twos_and_goes_on_from_its_last_frame():
    clip = torch.stack([scene(t, 64) for t in range(6)])
    said = []
    anim = animate(video=(clip, 24.0), work=64, colours=4, log=said.append)
    assert len(anim) == 3 and anim.fps == 12.0 and anim.settings["on"] == 2 and "drawn on 2s" in said[0]
    assert len(animate(video=(clip[:3], 24.0), work=64, colours=4, on=1)) == 3

    class Moving:
        def __call__(self, **kwargs):
            start = 5                                     # it goes on from the last frame drawn (4)
            return type("Out", (), {"frames": torch.stack([scene(start + t, 64) for t in range(6)])[None]})()

    animator = Animator(None, "animatediff", "fake", 4, 2.0, 24, 6, (64, 64), Moving())
    anim.extend("it rolls on", animator=animator)
    assert len(anim) == 3 + 2 and anim.history[-1] == {"op": "extend", "words": "it rolls on", "frames": 2}
    wan = type("Out", (), {"frames": torch.rand(1, 5, 32, 48, 3).numpy()})()      # Wan gives a numpy array
    assert _frames(wan).shape == (5, 3, 32, 48)


def test_trace_over_keeps_still_lines_exactly_where_they_were():
    img = scene(0, 120)
    palette = markers([img], 5)
    seen = look(img, palette)
    first = draw_picture(img, palette, seen=seen)
    again, kept, new = trace_over(first, torch.zeros(2, 120, 120), seen, palette)
    assert kept >= 0.9 * len(first) and new <= 0.1 * len(first) + 3


def test_the_animation_cell_is_plain_python_for_this_version_and_needs_no_token():
    from pathlib import Path

    import helix_lm

    cell = Path(__file__).resolve().parents[1] / "notebooks" / "pinxit_animate_cell.py"
    text = cell.read_text()
    compile(text, str(cell), "exec")
    assert f'HELIX = "{helix_lm.__version__}"' in text and "helix-lm[paint]==" in text
    assert "HF_TOKEN" not in text and "hf_" + "A" not in text
    assert "animate(WORDS" in text and "making_of(" in text and ".extend(NEXT" in text


def test_the_background_is_drawn_once_and_each_drawing_held_for_its_frames(tmp_path):
    square = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    sky = Line(square, torch.ones(4), "brush", (0.2, 0.3, 0.9), marker=0, kind="cel", rings=(4,))
    cloud = Line(square * 0.3 + 0.1, torch.ones(4), "brush", (0.1, 0.15, 0.6), marker=0, kind="cel-shadow", rings=(4,))
    ink = Line(torch.tensor([[0.2, 0.8], [0.8, 0.8]]), torch.ones(2) * 0.6, "pen", TOOLS["pen"].ink)
    sheets = [Sheet([ink.moved(ink.points + 0.02 * t)]) for t in range(3)]
    anim = Animation(sheets, torch.tensor([[0.2, 0.3, 0.9]]), None, 24.0, background=[sky, cloud], holds=[2, 1, 3])
    shown = list(anim.play(64))
    assert len(shown) == 6 and torch.equal(shown[0], shown[1]) and torch.equal(shown[3], shown[5])
    assert float(shown[0][2, 10, 50]) > 0.8 > float(shown[0][0, 10, 50])         # the sky under every frame
    back = Animation.load(anim.save(tmp_path / "held.pinxa"))
    assert back.holds == [2, 1, 3] and len(back.background) == 2 and back.background[1].rings == (4,)
    assert back.sheets[2].lines[0].uid == anim.sheets[2].lines[0].uid
    back.recolor((0.9, 0.2, 0.2), marker=0)
    assert torch.allclose(torch.tensor(back.background[0].colour), torch.tensor([0.9, 0.2, 0.2]), atol=1e-4)
    shade = back.background[1].colour
    assert shade[0] > shade[2] and sum(shade) < sum(back.background[0].colour)   # the shadow went red, still darker
