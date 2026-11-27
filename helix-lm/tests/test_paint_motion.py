# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The principles of animation on drawings over time: timing on ones, twos and threes with holds, keys and
breakdowns, steadying traced lines (the shake goes, the arc stays), in-betweens spaced by ease and moved on arcs,
and playing the drawings at any frame rate."""

import math

import torch

from helix_lm.paint.animate import Sheet
from helix_lm.paint.draw import Line
from helix_lm.paint.motion import (
    exaggerate,
    exposure,
    inbetween,
    inbetweens,
    keys,
    motion_amount,
    playback,
    spacing,
    steady,
    track_motion,
    tracks,
)

WIDE = 512.0                                                     # pixels a board width
K = 40
ANGLES = torch.arange(K) / K * 2 * math.pi
CIRCLE = torch.stack([ANGLES.cos(), ANGLES.sin()], -1)


def thrown(t: int, frames: int = 24) -> torch.Tensor:
    """Where a thrown ball is at frame ``t``: across the board on a parabola, 0.5 board widths high."""
    u = t / (frames - 1)
    return torch.tensor([0.2 + 0.6 * u, 0.75 - 2.0 * u * (1 - u)])


def pen(points, uid: int) -> Line:
    return Line(points, torch.full((len(points),), 0.7), "pen", (0.05, 0.05, 0.05), -1, "line", 1, uid)


def cel(rings: list, uid: int) -> Line:
    pts = torch.cat(rings)
    return Line(pts, torch.ones(len(pts)), "brush", (0.9, 0.3, 0.2), 0, "cel", 1, uid, [len(r) for r in rings])


def area(points: torch.Tensor) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float((x * torch.roll(y, -1) - torch.roll(x, -1) * y).sum()))


def shaken_throw(seed: int, frames: int = 24):
    """A disc outline (a pen line) and a disc with a hole (a cel shape) thrown on a parabola, traced with a shaky
    hand: the whole drawing off by about 0.5 pixels in each frame and every point by 0.3 more. The pen line is
    drawn from the other end every third frame, and a short line is drawn new in each frame."""
    g = torch.Generator().manual_seed(seed)
    true, sheets = [], []
    hole = 0.5 * CIRCLE[::2].flip(0)
    for t in range(frames):
        outline = thrown(t, frames) + 0.05 * CIRCLE
        true.append(outline)
        shake = torch.randn(2, generator=g) * 0.5 / WIDE + torch.randn(K, 2, generator=g) * 0.3 / WIDE
        drawn = outline + shake
        if t % 3 == 1:
            drawn = drawn.flip(0)
        shape = [thrown(t, frames) + 0.04 * CIRCLE + shake, thrown(t, frames) + 0.04 * hole + shake[::2]]
        new = pen(torch.tensor([[0.1, 0.1 + 0.01 * t], [0.2, 0.1]]), 1000 + t)
        sheets.append(Sheet([pen(drawn, 1), cel(shape, 2), new], 2, 1, t == 0))
    return sheets, true


def outline_error(sheets, true) -> float:
    """How far the pen line is from the true outline, in pixels, on average."""
    errs = []
    for t, sheet in enumerate(sheets):
        pts = sheet.lines[0].points
        pts = pts.flip(0) if t % 3 == 1 else pts
        errs.append(float((pts - true[t]).norm(dim=-1).mean()) * WIDE)
    return sum(errs) / len(errs)


def test_steady_takes_the_shake_out_of_a_thrown_ball_and_keeps_its_arc():
    real = torch.stack([thrown(t) for t in range(24)]) * WIDE
    for seed in range(3):
        sheets, true = shaken_throw(seed)
        given = [s.lines[0].points.clone() for s in sheets]
        out = steady(sheets)
        assert len(out) == 24 and all(len(o.lines) == 3 for o in out)
        before, after = outline_error(sheets, true), outline_error(out, true)
        assert before > 0.55 and after < 0.3                                     # the shake is gone
        path = torch.stack([o.lines[0].points.mean(0) for o in out]) * WIDE
        assert abs(float(path[:, 1].min() - real[:, 1].min())) < 0.5             # the arc is not flattened
        assert float((path - real).norm(dim=-1).mean()) < 0.3                    # nor moved
        rings = [o.lines[1] for o in out]
        assert all(r.rings == (K, K // 2) and len(r) == K + K // 2 for r in rings)
        centre = torch.stack([r.points[:K].mean(0) for r in rings]) * WIDE
        assert float((centre - real).norm(dim=-1).mean()) < 0.3                  # the cel shape is steadied too
        assert all(o.lines[2] is s.lines[2] for o, s in zip(out, sheets, strict=True))       # new lines untouched
        assert all(torch.equal(g, s.lines[0].points) for g, s in zip(given, sheets, strict=True))  # input unchanged
    assert out[4].lines[0].uid == 1 and torch.equal(out[4].lines[0].pressure, sheets[4].lines[0].pressure)
    pinned = steady(sheets, keys=[0, 11, 23])
    for t in (0, 11, 23):
        assert torch.equal(pinned[t].lines[0].points, sheets[t].lines[0].points)  # keys exactly as drawn
    assert outline_error(pinned, true) < outline_error(sheets, true)
    weak = steady(sheets, 0.3)
    assert outline_error(out, true) < outline_error(weak, true) < outline_error(sheets, true)


def test_steady_keeps_a_quick_swing_and_a_sharp_bounce_at_a_key():
    swing = [Sheet([pen(torch.tensor([0.5 + 0.04 * math.sin(2 * math.pi * t / 10), 0.5]) + 0.03 * CIRCLE, 3)])
             for t in range(30)]
    kept = steady(swing)
    x = torch.tensor([float(s.lines[0].points[:, 0].mean()) for s in kept])
    drawn = torch.tensor([float(s.lines[0].points[:, 0].mean()) for s in swing])
    assert float(x[5:25].max() - 0.5) > 0.95 * float(drawn[5:25].max() - 0.5)  # a swing 10 frames long stays
    bounce = [Sheet([pen(torch.tensor([0.3 + 0.01 * t, 0.2 + 0.6 * (1 - abs(t - 8) / 8)]) + 0.03 * CIRCLE, 4)])
              for t in range(17)]
    hit = steady(bounce, keys=keys(tracks(bounce))[0])
    assert torch.equal(hit[8].lines[0].points, bounce[8].lines[0].points)       # it reaches the floor


def test_lines_are_followed_by_uid_and_cel_rings_only_while_they_match():
    a = torch.tensor([[0.1, 0.1], [0.2, 0.1], [0.3, 0.12], [0.4, 0.15]])
    ring = 0.1 * CIRCLE[::4] + 0.5
    hole = 0.05 * CIRCLE[::8] + 0.5
    sheets = [Sheet([pen(a, 7), cel([ring, hole], 8)]),
              Sheet([pen((a + 0.01).flip(0), 7), cel([torch.roll(ring, 3, 0) + 0.01, hole + 0.01], 8)]),
              Sheet([pen(a + 0.02, 7), cel([ring + 0.02], 8)]),            # the hole has gone
              Sheet([pen(a + 0.03, 7)])]
    found = tracks(sheets)
    assert set(found) == {7, 8}
    line = found[7]
    assert line.frames == [0, 1, 2, 3] and line.points.shape == (4, 4, 2) and line.rings is None
    steps = line.points[1:] - line.points[:-1]
    assert torch.allclose(steps, torch.full_like(steps, 0.01), atol=1e-6)    # turned round where drawn backwards
    shape = found[8]
    assert shape.frames == [0, 1] and shape.rings == (10, 5) and shape.points.shape == (2, 15, 2)
    assert torch.allclose(shape.points[1] - shape.points[0], torch.full((15, 2), 0.01), atol=1e-6)  # ring turned
    fixed = tracks(sheets, count=9)[7]
    assert fixed.points.shape == (4, 9, 2)


def test_timing_is_ones_for_fast_twos_as_the_norm_threes_for_slow_and_holds_when_still():
    speeds = [10.0] * 24 + [3.0] * 24 + [0.6] * 24 + [0.05] * 24               # pixels a frame at 24 a second
    sheet = exposure(speeds, 24.0)
    assert sum(h for _, h in sheet) == 96 and sheet[0][0] == 0
    assert all(a + h == b for (a, h), (b, _) in zip(sheet, sheet[1:], strict=False))      # no gaps, no overlaps
    held = dict(sheet)
    assert all(held[i] == 1 for i in range(0, 24))                              # fast: on ones
    assert [held[i] for i in range(24, 48, 2)] == [2] * 12                      # the norm: on twos
    assert [held[i] for i in range(48, 72, 3)] == [3] * 8                       # slow: on threes
    assert held[72] == 24                                                       # still: one drawing held
    limited = exposure(speeds, 24.0, base=3)
    assert {h for i, h in limited if 24 <= i < 48} == {3}
    moving_hold = exposure(speeds, 24.0, longest_hold=8)
    assert [h for i, h in moving_hold if i >= 72] == [8, 8, 8]
    assert not any(i == 31 for i, _ in sheet)
    assert any(i == 31 for i, _ in exposure(speeds, 24.0, keys=[31]))           # a key always gets a drawing
    burst = [3.0] * 10 + [10.0] * 3 + [3.0] * 10 + [0.6] * 2 + [3.0] * 10
    timed = dict(exposure(burst, 24.0))
    assert all(timed.get(i) == 1 for i in (10, 11, 12))                         # a quick action keeps its ones
    assert not any(h == 3 for h in timed.values())                              # a short slow dip stays on twos
    half = exposure([2 * s for s in speeds], 12.0)                              # the same motion at 12 a second
    assert sum(h for _, h in half) == 96 and dict(half)[72] == 24
    assert all(h == 1 for i, h in half if i < 48)                               # ones and twos: every frame at 12
    slow = [h for i, h in half if 48 <= i < 72]
    assert sorted(set(slow)) == [1, 2] and abs(sum(slow) / len(slow) - 1.5) < 0.2  # threes: 1.5 frames at 12


def test_the_drawings_play_at_any_frame_rate_with_their_timing_kept():
    twos = exposure([3.0] * 24, 24.0)
    assert len(twos) == 12
    assert playback(twos, 24.0, 24.0) == [i // 2 for i in range(24)]
    assert playback(twos, 24.0, 12.0) == list(range(12))
    at30 = playback(twos, 24.0, 30.0)
    runs = [at30.count(d) for d in range(12)]
    assert len(at30) == 30 and at30 == sorted(at30) and set(runs) == {2, 3}      # 2.5 frames each on average
    assert playback(twos, 24.0, 60.0) == [i // 5 for i in range(60)]
    held = exposure([3.0] * 12 + [0.0] * 12, 24.0)
    at60 = playback(held, 24.0, 60.0)
    assert at60.count(len(held) - 1) == 30                                       # a half second hold stays one


def test_in_betweens_are_spaced_closer_to_the_keys_with_slow_in_and_out():
    chart = spacing(5, "in-out")
    gaps = [b - a for a, b in zip([0.0] + chart, chart + [1.0], strict=True)]
    assert gaps[0] < gaps[2] and gaps[-1] < gaps[2] and abs(chart[2] - 0.5) < 1e-6
    even = spacing(5, "linear")
    assert all(abs(b - a - 1 / 6) < 1e-6 for a, b in zip([0.0] + even, even + [1.0], strict=True))  # thirds, halves
    into = spacing(4, "in")
    assert into[-1] - into[-2] < into[1] - into[0]                               # bunched toward the second key
    out_of = spacing(4, "out")
    assert out_of[1] - out_of[0] < out_of[-1] - out_of[-2]                       # bunched toward the first
    a = pen(torch.tensor([[0.1, 0.5], [0.2, 0.5]]), 3)
    b = pen(torch.tensor([[0.5, 0.5], [0.6, 0.5]]), 3)
    mids = inbetweens(Sheet([a]), Sheet([b]), 5, "in-out")
    xs = [float(s.lines[0].points[0, 0]) for s in mids]
    assert all(abs(x - (0.1 + 0.4 * c)) < 1e-6 for x, c in zip(xs, chart, strict=True))
    steps = [b - a for a, b in zip([0.1] + xs, xs + [0.5], strict=True)]
    assert steps[0] < steps[2] > steps[-1]                                       # small steps near the keys


def test_in_betweens_move_on_arcs_round_a_pivot_or_through_a_breakdown():
    pivot = torch.tensor([0.5, 0.5])
    arm = torch.stack([torch.linspace(0.6, 0.8, 6), torch.full((6,), 0.5)], -1)   # pointing right of the pivot
    up = torch.stack([torch.full((6,), 0.5), 0.5 - (arm[:, 0] - 0.5)], -1)       # turned a quarter round
    a, b = pen(arm, 4), pen(up, 4)
    straight = inbetween(a, b, 0.5, "linear")
    turned = inbetween(a, b, 0.5, "linear", "pivot")
    given = inbetween(a, b, 0.5, "linear", (0.5, 0.5))
    reach = (arm - pivot).norm(dim=-1)
    assert float(((straight.points - pivot).norm(dim=-1) - reach).min()) < -0.03      # the straight cuts the corner
    assert torch.allclose((turned.points - pivot).norm(dim=-1), reach, atol=1e-5)     # the arc keeps the reach
    assert torch.allclose(turned.points, given.points, atol=1e-5)
    angle = torch.atan2(-(turned.points[:, 1] - 0.5), turned.points[:, 0] - 0.5)
    assert torch.allclose(angle, torch.full((6,), math.pi / 4), atol=1e-5)            # halfway round
    middle = pen(0.5 + 0.2 * torch.tensor([[1.0, -1.0]]) / math.sqrt(2) * torch.linspace(0.5, 1.5, 6)[:, None], 4)
    through = inbetween(a, b, 0.5, "linear", middle)
    assert torch.allclose(through.points, middle.points, atol=1e-6)                   # passes the breakdown
    near = inbetween(a, b, 0.25, "linear", middle)
    assert float((near.points - pivot).norm(dim=-1).sub(reach).abs().max()) < 0.01    # close to the circle
    assert inbetween(a, b, 0.0, "in-out", "pivot").uid == 4


def test_in_betweens_match_lines_by_uid_and_lines_in_one_key_come_and_go():
    seg = torch.stack([torch.linspace(0.1, 0.3, 5), torch.full((5,), 0.2)], -1)
    sheet_a = Sheet([pen(seg, 1), pen(seg + torch.tensor([0.0, 0.2]), 2), pen(seg + torch.tensor([0.0, 0.3]), 3)])
    shift = torch.tensor([0.2, 0.0])
    dense = torch.stack([torch.linspace(0.1, 0.3, 9), torch.full((9,), 0.2)], -1) + shift   # more points now
    sheet_b = Sheet([pen(dense, 1), pen((seg + torch.tensor([0.0, 0.2]) + shift).flip(0), 2),
                     pen(seg + torch.tensor([0.0, 0.45]) + shift, 4)])
    mids = inbetweens(sheet_a, sheet_b, 3, "linear")
    assert [sorted(line.uid for line in s.lines) for s in mids] == [[1, 2, 3], [1, 2, 4], [1, 2, 4]]
    assert [s.kept for s in mids] == [2, 2, 2] and not any(s.fresh for s in mids)
    first = {line.uid: line for line in mids[0].lines}
    for uid in (1, 2):
        pts = first[uid].points
        assert abs(float(pts[:, 0].min()) - 0.15) < 1e-5 and abs(float(pts[:, 0].max()) - 0.35) < 1e-5  # not crossed
    assert len(first[1]) == 9                                                    # resampled to match
    carried = first[3].points - sheet_a.lines[2].points
    assert torch.allclose(carried, torch.tensor([0.05, 0.0]).expand_as(carried), atol=1e-3)  # moves with the rest
    late = {line.uid: line for line in mids[2].lines}
    assert torch.allclose(late[4].points - sheet_b.lines[2].points, torch.tensor([-0.05, 0.0]).expand(5, 2),
                          atol=1e-3)


def test_keys_are_the_extremes_and_breakdowns_lie_between():
    t = torch.arange(33, dtype=torch.float32)
    speed = (20.0 * 2 * math.pi / 16 * torch.cos(2 * math.pi * t / 16)).abs()  # a swing to and fro, 16 frames
    found, between = keys(speed)
    assert found[0] == 0 and found[-1] == 32 and all(k in found for k in (4, 12, 20, 28))
    assert all(a < b < c for a, b, c in zip(found, between, found[1:], strict=False))
    assert 7 <= between[found.index(4)] <= 9                                     # the passing position
    sheets = []
    for f in range(17):                                                          # a ball dropped onto a floor
        y = 0.2 + 0.6 * (1 - abs(f - 8) / 8)
        sheets.append(Sheet([pen(torch.tensor([0.4 + 0.005 * f, y]) + 0.04 * CIRCLE, 9)]))
    bounce, _ = keys(tracks(sheets))
    assert 8 in bounce and bounce[0] == 0 and bounce[-1] == 16                   # it turns at the floor
    moved = track_motion(tracks(sheets), scale=WIDE)
    assert len(moved) == 17 and float(moved[3]) > 30
    still = keys(torch.tensor([3.0] * 8 + [0.0] * 8 + [3.0] * 8))[0]
    assert 8 in still and 15 in still                                            # a hold starts and ends on a key


def test_motion_is_measured_where_the_character_moves():
    flows = torch.zeros(3, 2, 64, 64)
    flows[0, 0, 10:30, 10:30], flows[0, 1, 10:30, 10:30] = 3.0, 4.0               # it moves 5 pixels
    flows[1, 0, 10:30, 10:30] = 1.0
    flows[2] += 0.02 * torch.randn(2, 64, 64, generator=torch.Generator().manual_seed(1))   # only noise
    steps = motion_amount(flows, between=False)
    assert abs(float(steps[0]) - 5.0) < 1e-4 and abs(float(steps[1]) - 1.0) < 1e-4 and float(steps[2]) < 0.1
    per_frame = motion_amount(flows)
    assert per_frame.shape == (4,) and abs(float(per_frame[1]) - 3.0) < 1e-4
    mask = torch.zeros(64, 64, dtype=torch.bool)
    mask[40:60, 40:60] = True
    assert float(motion_amount(flows, mask, between=False)[0]) < 0.1             # nothing moves in the mask
    assert len(exposure(motion_amount(flows), 24.0)) >= 1


def test_exaggerate_pushes_the_arc_and_stretches_along_fast_motion():
    sheets = [Sheet([pen(thrown(t) + 0.05 * CIRCLE, 1), pen(torch.tensor([[0.9, 0.9], [0.95, 0.9]]), 50 + t)])
              for t in range(24)]
    same = exaggerate(sheets)
    assert all(torch.equal(a.lines[0].points, b.lines[0].points) for a, b in zip(same, sheets, strict=True))
    higher = exaggerate(sheets, 0.5)
    top = min(float(s.lines[0].points[:, 1].mean()) for s in higher)
    assert top < min(float(s.lines[0].points[:, 1].mean()) for s in sheets) - 0.005   # the arc goes higher
    fast = [Sheet([pen(torch.tensor([0.1 + 0.03 * t, 0.5]) + 0.05 * CIRCLE, 1)]) for t in range(10)]
    stretched = exaggerate(fast, stretch=0.5)[5].lines[0].points
    span = stretched.max(0).values - stretched.min(0).values
    assert float(span[0]) > 0.105 and float(span[1]) < 0.095                     # longer along the motion
    assert abs(area(stretched) - area(fast[5].lines[0].points)) < 0.02 * area(fast[5].lines[0].points)
