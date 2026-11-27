# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Faces and small details drawn with care and kept on model from frame to frame, and the self-check that
finds what a drawing got wrong and draws it again. The face is a small synthetic anime face (a head, two eyes
with a lash line, an iris, a pupil and a highlight, a nose and a mouth) that moves and turns 10 degrees."""

import math

import torch
import torch.nn.functional as F

from helix_lm.paint.check import bad_boxes, fix, score
from helix_lm.paint.detail import (
    REDRAW,
    Part,
    clip_lines,
    clip_polyline,
    components,
    draw_crop,
    draw_details,
    find_details,
    fit_similarity,
    follow,
    mask_rings,
    merge,
    move_box,
    track,
)
from helix_lm.paint.draw import CELS, Line, draw_picture, fill_rings, markers

SIZE = 160
SKIN, OUTLINE, HAIR = (0.99, 0.86, 0.76), (0.12, 0.08, 0.08), (0.55, 0.30, 0.20)
IRIS, PUPIL, LASH, WHITE = (0.30, 0.18, 0.42), (0.07, 0.05, 0.10), (0.06, 0.05, 0.07), (1.0, 1.0, 1.0)
MOUTH, BACK = (0.62, 0.20, 0.22), (0.70, 0.84, 0.95)
EYES = ((-15.0, -2.0), (15.0, -2.0))             # the eyes' centres on the face, before it turns


def where(t: int) -> tuple:
    """The face's centre (pixels) and its turn (radians) in frame ``t``: 2.5 degrees a frame."""
    return 0.45 * SIZE + 2.0 * t, 0.5 * SIZE + 1.0 * t, math.radians(2.5 * t)


def _coords(t: int, ss: int = 4):
    g = (torch.arange(SIZE * ss, dtype=torch.float32) + 0.5) / ss
    yy, xx = torch.meshgrid(g, g, indexing="ij")
    cx, cy, a = where(t)
    dx, dy = xx - cx, yy - cy
    return math.cos(a) * dx + math.sin(a) * dy, -math.sin(a) * dx + math.cos(a) * dy


def _ellipse(u, v, cx, cy, a, b):
    return ((u - cx) / a) ** 2 + ((v - cy) / b) ** 2 < 1.0


def face(t: int = 0, blink: bool = False) -> torch.Tensor:
    """Frame ``t`` of the face, drawn four times larger and shrunk, so its edges are smooth."""
    u, v = _coords(t)
    img = torch.tensor(BACK)[:, None, None].expand(3, *u.shape).clone()

    def put(mask, colour):
        img[:, mask] = torch.tensor(colour)[:, None]

    put(_ellipse(u, v, 0, 0, 40, 46), OUTLINE)
    put(_ellipse(u, v, 0, 0, 38, 44), SKIN)
    put(_ellipse(u, v, 0, -30, 41, 22) & _ellipse(u, v, 0, 0, 40, 46) & (v < -24 + 0.12 * u.abs()), HAIR)
    for ex, ey in EYES:
        if blink:                                    # closed: only the lash line, curved down
            put(((v - ey - 2.0 - 0.02 * (u - ex) ** 2).abs() < 1.2) & ((u - ex).abs() < 9.5), LASH)
            continue
        e = _ellipse(u, v, ex, ey + 1.0, 9.5, 8.5)
        put(e, WHITE)
        put(e & _ellipse(u, v, ex, ey + 1.0, 6.0, 8.0), IRIS)
        put(e & _ellipse(u, v, ex, ey + 2.0, 3.0, 4.0), PUPIL)
        put(_ellipse(u, v, ex - 2.5, ey - 2.0, 2.2, 2.2), WHITE)
        put(_ellipse(u, v, ex, ey, 10.5, 10.5) & ~_ellipse(u, v, ex, ey + 1.5, 9.5, 8.5) & (v < ey - 1.0), LASH)
    put(((v - 22.0 - 0.03 * u ** 2).abs() < 1.1) & (u.abs() < 7.0), MOUTH)
    put(_ellipse(u, v, 0.0, 10.0, 1.2, 1.0), OUTLINE)
    return F.avg_pool2d(img[None], 4)[0]


def eye_centre(t: int, k: int) -> tuple:
    cx, cy, a = where(t)
    ex, ey = EYES[k]
    ey += 1.0
    return cx + math.cos(a) * ex - math.sin(a) * ey, cy + math.sin(a) * ex + math.cos(a) * ey


def eye_truth(t: int, k: int) -> torch.Tensor:
    """How much of each pixel eye ``k`` covers in frame ``t``."""
    u, v = _coords(t)
    ex, ey = EYES[k]
    return F.avg_pool2d(_ellipse(u, v, ex, ey + 1.0, 9.5, 8.5).float()[None, None], 4)[0, 0]


def eye_shape(t: int, k: int, n: int = 64) -> Line:
    """Eye ``k`` of frame ``t`` as a cel shape (its outline in board widths)."""
    cx, cy, a = where(t)
    ex, ey = EYES[k]
    s = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    u, v = ex + 9.5 * torch.cos(s), ey + 1.0 + 8.5 * torch.sin(s)
    pts = torch.stack([cx + math.cos(a) * u - math.sin(a) * v, cy + math.sin(a) * u + math.cos(a) * v], -1)
    return Line(pts / SIZE, torch.ones(n), "brush", IRIS, 0, "cel", rings=[n])


def coverage(line: Line) -> torch.Tensor:
    y0, x0, cov = fill_rings(line.points * SIZE, line.rings, SIZE, SIZE)
    full = torch.zeros(SIZE, SIZE)
    full[y0 : y0 + cov.shape[0], x0 : x0 + cov.shape[1]] = cov
    return full


def box_error(result: dict, box) -> float:
    x0, y0, x1, y1 = (int(v) for v in box)
    return float(result["error_map"][y0:y1, x0:x1].mean())


def test_shapes_are_found_traced_and_cut_exactly():
    mask = torch.zeros(40, 50, dtype=torch.bool)
    mask[5:15, 5:20] = True
    mask[8:12, 9:14] = False                         # a hole
    mask[25:35, 30:45] = True
    mask[30, 10:28] = True                           # a thin line, on its own
    labels, count = components(mask)
    assert count == 3 and int(labels[mask].min()) == 0 and bool((labels[~mask] == -1).all())
    rings = mask_rings(mask)
    y0, x0, cov = fill_rings(torch.cat(rings), [len(r) for r in rings], 40, 50)
    back = torch.zeros(40, 50)
    back[y0 : y0 + cov.shape[0], x0 : x0 + cov.shape[1]] = cov
    assert float((back - mask.float()).abs().sum()) < 0.13 * len(rings) * 8 + 12     # only the corners cut
    assert float(back[10, 11]) == 0.0 and float(back[6, 6]) == 1.0                   # the hole stays a hole
    pieces = clip_polyline(torch.tensor([[0.0, 5.0], [20.0, 5.0], [20.0, 15.0]]), torch.ones(3), (5, 0, 10, 10))
    assert len(pieces) == 1 and torch.allclose(pieces[0][0], torch.tensor([[5.0, 5.0], [10.0, 5.0]]))
    square = Line(torch.tensor([[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]) / 40, torch.ones(4),
                  "brush", (1, 0, 0), 0, "cel", rings=[4])
    box = (15.0, 15.0, 25.0, 40.0)
    inside, outside = clip_lines([square], box, 40.0, True), clip_lines([square], box, 40.0, False)

    def cover(lines):
        y0, x0, cov = fill_rings(lines[0].points * 40, lines[0].rings, 40, 40)
        full = torch.zeros(40, 40)
        full[y0 : y0 + cov.shape[0], x0 : x0 + cov.shape[1]] = cov
        return full

    assert torch.allclose(cover(inside) + cover(outside), cover([square]), atol=1e-5)   # cut, nothing lost
    assert float(cover(outside)[20, 20]) == 0.0 and float(cover(outside)[12, 12]) == 1.0


def test_a_face_is_found_by_its_two_eyes():
    for t in (0, 4):                                  # straight, and turned 10 degrees
        parts = find_details(face(t))
        faces = [p for p in parts if p.kind == "face"]
        eyes = [p for p in parts if p.kind == "eye"]
        assert len(faces) == 1 and len(eyes) == 2 and all(p.face == 0 for p in eyes)
        cx, cy, _ = where(t)
        x0, y0, x1, y1 = faces[0].box
        assert x0 < cx - 30 and x1 > cx + 30 and y0 < cy - 20 and y1 > cy + 35      # round the whole face
        for k in (0, 1):
            ex, ey = eye_centre(t, k)
            assert sum(p.box[0] < ex < p.box[2] and p.box[1] < ey < p.box[3] for p in eyes) == 1
        assert any(p.kind == "mouth" for p in parts)
    plain = torch.tensor(BACK)[:, None, None].expand(3, SIZE, SIZE).clone()
    plain[:, 60:70, 50:60] = 0.05                     # two dark dots side by side, with no head round them
    plain[:, 60:70, 90:100] = 0.05
    assert not any(p.kind == "face" for p in find_details(plain))
    assert not any(p.kind == "face" for p in find_details(face(4, blink=True)))   # no open eyes, no face


def test_a_face_is_followed_rigidly_and_its_eyes_stay_on_model():
    frames = [face(t) for t in range(5)]
    lines = [eye_shape(0, 0), eye_shape(0, 1)]
    uids = [line.uid for line in lines]
    box = next(p.box for p in find_details(frames[0]) if p.kind == "face")
    for t in range(1, 5):
        matrix, change = track(box, frames[t - 1], frames[t])
        lines, again = follow(lines, box, frames[t - 1], frames[t])
        assert change < REDRAW and abs(again - change) < 1e-6        # it only moved: no need to draw it again
        box = move_box(box, matrix)
    for k in (0, 1):                                  # the same shapes, where the eyes now are
        drawn, truth = coverage(lines[k]), eye_truth(4, k)
        assert float(torch.minimum(drawn, truth).sum() / torch.maximum(drawn, truth).sum()) > 0.95
    a, b = (line.points.mean(0) * SIZE for line in lines)
    assert abs(math.degrees(math.atan2(float(b[1] - a[1]), float(b[0] - a[0]))) - 10.0) < 0.5   # on the eye line
    assert abs(float((b - a).norm()) - 30.0) < 0.3    # as far apart as on the model
    area = [float(coverage(line).sum()) for line in lines]
    assert abs(area[0] - area[1]) < 0.01 * area[0]   # the same size
    assert [line.uid for line in lines] == uids and all(line.age == 4 for line in lines)
    _, blinked = track(box, frames[3], face(4, blink=True))
    assert blinked > REDRAW                           # a blink is drawn again


def test_similarity_is_fitted_robustly():
    g = torch.Generator().manual_seed(0)
    pts = torch.rand(200, 2, generator=g) * 50
    angle, scale = math.radians(7.0), 1.05
    rot = scale * torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    moved = pts @ rot.T + torch.tensor([3.0, -2.0])
    moved[:30] += torch.rand(30, 2, generator=g) * 20                 # things that moved some other way
    m = fit_similarity(pts, moved)
    assert torch.allclose(m[:, :2], rot, atol=2e-3) and torch.allclose(m[:, 2], torch.tensor([3.0, -2.0]), atol=0.1)


def test_details_drawn_bigger_are_more_accurate():
    image = face(4)
    palette = markers([image], 8)
    whole = draw_picture(image, palette)
    parts = find_details(image)
    detail = draw_details(image, parts, palette)
    face_box = next(p.box for p in parts if p.kind == "face")
    for line in detail:                               # every detail line lies in a part
        pts = line.points * SIZE
        assert bool(((pts[:, 0] >= face_box[0] - 1.5) & (pts[:, 0] <= face_box[2] + 1.5)).all())
    assert any(line.kind in CELS for line in detail) and any(line.kind == "line" for line in detail)
    merged = merge(whole, detail, parts)
    before, after = score(whole, image, palette), score(merged, image, palette)
    for part in parts:
        if part.kind == "eye":
            assert box_error(after, part.box) < 0.6 * box_error(before, part.box)
    assert after["error"] < before["error"]
    far = (0, 0, SIZE, 20)                            # far from the face the drawing is as it was
    assert abs(box_error(after, far) - box_error(before, far)) < 0.01


def test_the_check_finds_a_missing_eye_and_draws_it_again():
    image = face(0)
    palette = markers([image], 8)
    lines = draw_picture(image, palette)
    eye = max((p for p in find_details(image) if p.kind == "eye"), key=lambda p: p.box[0])
    x0, y0, x1, y1 = eye.box
    broken = [line for line in lines if not bool(((line.points[:, 0] * SIZE > x0) & (line.points[:, 0] * SIZE < x1)
                                                  & (line.points[:, 1] * SIZE > y0)
                                                  & (line.points[:, 1] * SIZE < y1)).any())]
    assert len(broken) < len(lines)
    whole, missing = score(lines, image, palette), score(broken, image, palette)
    assert missing["loss"] > whole["loss"] and box_error(missing, eye.box) > box_error(whole, eye.box)
    bad = missing["bad"][int(y0) : int(y1), int(x0) : int(x1)]
    assert float(bad.float().mean()) > 0.3            # it sees where the eye is missing
    ex, ey = eye_centre(0, 1)
    assert any(b[0] < ex < b[2] and b[1] < ey < b[3] for b in bad_boxes(missing["bad"]))
    fixed = fix(broken, image, None, palette)
    again = score(fixed, image, palette)
    assert again["loss"] < missing["loss"] and box_error(again, eye.box) < 0.6 * box_error(missing, eye.box)
    assert set(score(fixed, image, palette)) >= {"error", "regions", "bare", "chamfer", "bad", "loss"}


def test_a_character_is_checked_and_mended_over_its_background_without_touching_it():
    image = face(0)
    palette = markers([image], 8)
    ground = torch.tensor(BACK)[:, None, None].expand(3, SIZE, SIZE).clone()
    under = draw_picture(ground, palette)                        # the background art, drawn once

    def cel(crop, colours):                                      # the character is cel paint: opaque
        return draw_crop(crop, colours, 0.5)

    cx, cy, _ = where(0)

    ex, ey = eye_centre(0, 1)
    yy, xx = torch.meshgrid(torch.arange(SIZE) + 0.5, torch.arange(SIZE) + 0.5, indexing="ij")

    def mostly(line, centre, radii):                             # a cel shape by its area, a line by its points
        if line.rings is not None:
            place = ((xx - centre[0]) / radii[0]) ** 2 + ((yy - centre[1]) / radii[1]) ** 2 < 1.0
            return float((coverage(line) * place).sum() / coverage(line).sum()) > 0.5
        inside = ((line.points * SIZE - torch.tensor(centre)) / torch.tensor(radii)).pow(2).sum(-1) < 1.0
        return float(inside.float().mean()) > 0.5

    character = [line for line in cel(image, palette) if mostly(line, (cx, cy), (42.0, 48.0))]   # the head
    broken = [line for line in character if not mostly(line, (ex, ey), (12.0, 11.0))]           # an eye missing
    assert len(broken) < len(character)
    alone, over = score(broken, image, palette), score(broken, image, palette, under=under)
    assert alone["bare"] > 0.5 and over["bare"] < 0.05           # the background is there under it
    assert bool(over["layer"][int(cy), int(cx)]) and not bool(over["layer"][5, 5])
    eye = (ex - 10, ey - 9, ex + 10, ey + 9)
    assert float(over["bad"][int(ey) - 6 : int(ey) + 6, int(ex) - 6 : int(ex) + 6].float().mean()) > 0.3
    kept = list(under)
    fixed = fix(broken, image, cel, palette, under=under)
    again = score(fixed, image, palette, under=under)
    assert again["loss"] < over["loss"] and box_error(again, eye) < 0.7 * box_error(over, eye)
    assert under == kept and not any(line in fixed for line in under)   # the background is never changed


def test_merge_needs_to_know_the_frame_and_keeps_lines_outside_the_parts():
    line = Line([[0.1, 0.1], [0.9, 0.1]], [0.5, 0.5], "pen", (0, 0, 0))
    part = Part((40.0, 0.0, 80.0, 40.0), "detail", 1.0, (SIZE, SIZE))
    kept = merge([line], [], [part])
    xs = torch.cat([x.points[:, 0] for x in kept]) * SIZE
    assert len(kept) == 2 and not bool(((xs > 40.5) & (xs < 79.5)).any())     # cut out of the part, both sides kept
    assert merge([line], [], []) == [line]
