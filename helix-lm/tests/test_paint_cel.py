# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Anime ink and paint for one frame: clean flat regions on a shared palette, outlines that fill back to their
masks, shadows and highlights as hard-edged shapes, line art heavier on the silhouette, and the whole frame
rendered as a cel with no gaps and no blur."""

import math

import torch

from helix_lm.paint.cel import (
    borders,
    cels,
    light_colour,
    line_art,
    outlines,
    paint_frame,
    palette_for,
    parts,
    regions,
    shade,
    shadow_colour,
)
from helix_lm.paint.color import delta_e, srgb_to_oklab
from helix_lm.paint.draw import CELS, Board, fill_rings, render
from helix_lm.paint.see import gaussian_blur

SKY_TOP, SKY_LOW = (0.35, 0.6, 0.95), (0.75, 0.88, 1.0)
GROUND, SHIRT, SHIRT_DARK = (0.4, 0.72, 0.3), (0.85, 0.15, 0.2), (0.6, 0.08, 0.2)
SKIN, HAIR, HAIR_DARK, EYE = (1.0, 0.86, 0.74), (0.55, 0.3, 0.15), (0.35, 0.17, 0.1), (0.12, 0.1, 0.25)


def character(size: int = 160, shift: float = 0.0, noise: float = 0.01) -> tuple[torch.Tensor, dict]:
    """An anime-like frame: a sky gradient, the ground, and a character (hair with a shadow side, a head, two
    eyes with white highlights, a red shirt with a shadow side), a little soft and noisy like a video frame.
    Returns the frame and the masks of its parts."""
    s = size
    yy, xx = torch.meshgrid(torch.arange(s).float() + 0.5, torch.arange(s).float() + 0.5, indexing="ij")
    t = (yy / s)[None]
    img = (torch.tensor(SKY_TOP)[:, None, None] * (1 - t) + torch.tensor(SKY_LOW)[:, None, None] * t).expand(3, s, s)
    img = img.clone()
    cx = 0.5 * s + shift * s
    masks = {"ground": yy > 0.78 * s}
    masks["body"] = ((xx - cx).abs() < 0.17 * s) & (yy > 0.55 * s) & (yy < 0.85 * s)
    masks["head"] = ((xx - cx) ** 2 + (yy - 0.4 * s) ** 2) < (0.2 * s) ** 2
    masks["hair"] = (masks["head"] & (yy < 0.33 * s)) | ((((xx - cx) ** 2 + (yy - 0.3 * s) ** 2) < (0.21 * s) ** 2)
                                                         & (yy < 0.3 * s))
    eyes = torch.zeros(s, s, dtype=torch.bool)
    lights = torch.zeros(s, s, dtype=torch.bool)
    for ex in (-0.075, 0.075):
        eyes |= ((xx - cx - ex * s) / 0.035) ** 2 + ((yy - 0.43 * s) / 0.055) ** 2 < s ** 2
        lights |= ((xx - cx - ex * s + 0.012 * s) ** 2 + (yy - 0.415 * s) ** 2) < (0.016 * s) ** 2
    masks["eyes"], masks["lights"] = eyes, lights
    masks["shirt shadow"] = masks["body"] & (xx > cx + 0.06 * s)
    masks["hair shadow"] = masks["hair"] & (xx > cx + 0.05 * s)
    for name, colour in (("ground", GROUND), ("body", SHIRT), ("shirt shadow", SHIRT_DARK), ("head", SKIN),
                         ("hair", HAIR), ("hair shadow", HAIR_DARK), ("eyes", EYE), ("lights", (1.0, 1.0, 1.0))):
        img[:, masks[name]] = torch.tensor(colour)[:, None]
    masks["figure"] = masks["body"] | masks["head"] | masks["hair"]
    g = torch.Generator().manual_seed(0)
    img = gaussian_blur(img, 0.6) + noise * torch.randn(3, s, s, generator=g)
    return img.clamp(0, 1), masks


def coverage(rings, height: int, width: int) -> torch.Tensor:
    cov = torch.zeros(height, width)
    got = fill_rings(torch.cat(rings), [len(r) for r in rings], height, width)
    if got is not None:
        y0, x0, c = got
        cov[y0 : y0 + c.shape[0], x0 : x0 + c.shape[1]] = c
    return cov


def iou(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float(torch.minimum(a, b).sum() / torch.maximum(a, b).sum())


def test_outlines_fill_back_to_their_masks_holes_and_corners_included():
    yy, xx = torch.meshgrid(torch.arange(120.0), torch.arange(120.0), indexing="ij")
    disc = ((xx - 60) ** 2 + (yy - 55) ** 2) < 30 ** 2
    ring = disc & (((xx - 60) ** 2 + (yy - 55) ** 2) > 12 ** 2)
    square = (xx > 20) & (xx < 90) & (yy > 30) & (yy < 70)
    ell = square & ~((xx > 55) & (yy < 50))
    edge = (xx < 40) | (yy > 100)                         # it runs off the picture
    for mask in (disc, ring, square, ell, edge):
        rings = outlines(mask)
        assert iou(coverage(rings, 120, 120), mask) > 0.97
    assert len(outlines(ring)) == 2                       # the outer edge and the hole
    sq = outlines(square)[0]
    assert len(sq) <= 12                                  # straight sides simplified to their corners
    for corner in ((21.0, 31.0), (90.0, 31.0), (21.0, 70.0), (90.0, 70.0)):
        assert float((sq - torch.tensor(corner)).norm(dim=-1).min()) < 0.8      # and the corners kept sharp
    pts = outlines(disc)[0]
    radius = (pts - torch.tensor([60.5, 55.5])).norm(dim=-1)        # pixel (i, j) has its middle at j + 0.5
    assert float(radius.std()) < 0.35                     # round, no stair steps of the pixels
    assert outlines(torch.zeros(10, 10, dtype=torch.bool)) == []


def test_parts_are_connected_and_numbered_in_reading_order():
    labels = torch.tensor([[0, 0, 1, 1],
                           [2, 0, 1, 2],
                           [2, 2, 1, 2],
                           [1, 2, 2, 2]])
    ids, n = parts(labels)
    assert n == 4 and ids[0, 0] == 0 and ids[0, 2] == 1 and ids[1, 0] == 2 and ids[3, 0] == 3
    assert ids[1, 0] == ids[3, 2] and ids[1, 3] == ids[3, 2]        # the 2s go round to one part
    assert ids[3, 0] != ids[0, 2]                                   # a 1 touching only by a corner: its own part
    spiral = torch.zeros(64, 64, dtype=torch.long)
    spiral[::4] = 1
    spiral[:, 0] = 0
    for r in range(0, 64, 8):
        spiral[r : r + 5, 63] = 1
    for r in range(4, 64, 8):
        spiral[r : r + 5, 0] = 1
    ids, n = parts(spiral)
    assert int(ids[0, 5]) == int(ids[60, 5])                        # one long snake


def test_the_palette_has_the_colours_of_the_frames_whites_and_small_ones_too():
    img, _ = character()
    palette = palette_for([img])
    assert 6 <= len(palette) <= 16
    light = srgb_to_oklab(palette)[:, 0]
    assert bool((light[1:] <= light[:-1] + 1e-6).all())             # light to dark
    for colour in (SKIN, SHIRT, SHIRT_DARK, HAIR, HAIR_DARK, EYE, GROUND, (1.0, 1.0, 1.0)):
        gap = delta_e(palette.T[:, :, None], torch.tensor(colour)[:, None, None].expand(3, len(palette), 1))
        assert float(gap.min()) < 0.05, colour                      # the eye highlight is a colour of its own
    again = palette_for([img])
    assert torch.equal(palette, again)                              # the same frames, the same palette
    shifted, _ = character(shift=0.05)
    both = palette_for([img, shifted])
    assert len(both) <= 16
    for colour in (SKIN, SHIRT, HAIR, (1.0, 1.0, 1.0)):
        gap = delta_e(both.T[:, :, None], torch.tensor(colour)[:, None, None].expand(3, len(both), 1))
        assert float(gap.min()) < 0.05


def test_regions_are_flat_colour_without_specks_or_frayed_borders():
    img, masks = character()
    g = torch.Generator().manual_seed(1)
    specks = torch.rand(160, 160, generator=g) < 0.004              # snow in the air
    img[:, specks] = 1.0
    palette = palette_for([img])
    labels = regions(img, palette)
    ids, n = parts(labels)
    area = torch.bincount(ids.flatten())
    assert int(area.min()) >= 6 and n < 40                          # no specks, a few clean parts
    painted = palette[labels].permute(2, 0, 1)
    figure = masks["figure"] & ~masks["eyes"]
    inner = -torch.nn.functional.max_pool2d(-figure.float()[None, None], 5, 1, 2)[0, 0] > 0
    assert float(delta_e(painted, character()[0])[inner].mean()) < 0.04
    grow = lambda m: torch.nn.functional.max_pool2d(m.float()[None, None], 9, 1, 4)[0, 0] > 0    # noqa: E731
    for name, colour in (("head", SKIN), ("body", SHIRT)):
        core = masks[name] & inner & ~grow(masks["eyes"]) & ~grow(masks["hair"])
        if name == "body":                                      # the lit side, below the chin
            core &= (torch.arange(160)[None, :] < 0.52 * 160) & ~grow(masks["head"])
        got = painted[:, core].T
        assert float((got - got[0]).abs().max()) == 0.0             # one flat colour
        assert float(delta_e(got[:1].T[:, :, None], torch.tensor(colour)[:, None, None])) < 0.05
    # the eye highlight is kept as its own white part
    assert bool((painted[:, masks["lights"]].min(0).values > 0.9).any())


def test_neighbouring_cel_shapes_share_their_borders_and_leave_no_gap():
    img, _ = character()
    palette = palette_for([img])
    labels = regions(img, palette)
    shading = shade(img, labels, palette)
    bd = borders(shading.tones, len(shading.kind))
    paint = cels(shading, bd, 160.0)
    assert all(line.kind in CELS and line.rings for line in paint)
    for width in (160, 400):
        h = width
        picture = render(paint, width, h, waver=0.0)
        bare = Board(width, h).picture()
        assert float(((picture - bare).abs().max(0).values < 0.004).float().mean()) == 0.0
    # every border runs between two parts and is used by both of their shapes
    assert all(a != b for a, b in zip(bd.a.tolist(), bd.b.tolist(), strict=True))


def test_shadows_and_highlights_are_hard_shapes_in_anime_colours():
    s = 120
    yy, xx = torch.meshgrid(torch.arange(s).float() + 0.5, torch.arange(s).float() + 0.5, indexing="ij")
    img = torch.tensor([0.75, 0.88, 1.0])[:, None, None].expand(3, s, s).clone()
    ball = ((xx - 60) ** 2 + (yy - 60) ** 2) < 40 ** 2
    side = ball & (xx > 72)
    img[:, ball] = torch.tensor([0.95, 0.55, 0.2])[:, None]
    img[:, side] = torch.tensor([0.95, 0.55, 0.2])[:, None] * 0.84           # the frame shades its right side
    img = gaussian_blur(img, 0.8)
    palette = torch.tensor([[0.75, 0.88, 1.0], [0.95, 0.55, 0.2]])           # no shadow tone in the palette
    labels = regions(img, palette)
    shading = shade(img, labels, palette, relief=False)
    assert iou(shading.shadow, side) > 0.8
    tone = shading.colour[shading.tones[60, 90]]
    base = srgb_to_oklab(palette[1])
    dark = srgb_to_oklab(tone)
    assert float(dark[0]) < float(base[0]) - 0.07                           # darker
    hue = math.atan2(float(dark[2]), float(dark[1])) - math.atan2(float(base[2]), float(base[1]))
    assert hue < 0                                                          # turned toward blue and purple
    assert not bool(shading.shadow[:, :20].any())                           # the sky is painted flat
    # from normals: a sphere lit from the left has its shadow on the right
    flat = img.clone()
    flat[:, ball] = torch.tensor([0.95, 0.55, 0.2])[:, None]
    nx, ny = (xx - 60) / 40, (yy - 60) / 40
    nz = (1 - nx ** 2 - ny ** 2).clamp(min=0).sqrt()
    lit = shade(flat, regions(flat, palette), palette, torch.stack([nx, ny, nz]), torch.tensor([-1.0, 0.0, 0.4]))
    assert float(lit.shadow[:, 75:].float().sum()) > 10 * float(lit.shadow[:, :60].float().sum())
    assert bool(lit.light[:, :60].any()) and not bool(lit.light[:, 80:].any())
    # the colours an anime painter picks
    white = srgb_to_oklab(shadow_colour(torch.tensor([0.95, 0.95, 0.96])))
    assert float(white[2]) < -0.015                                         # a white's shadow is lavender
    warm = srgb_to_oklab(light_colour(torch.tensor([0.4, 0.6, 0.9])))
    assert float(warm[0]) > float(srgb_to_oklab(torch.tensor([0.4, 0.6, 0.9]))[0]) + 0.1


def test_line_art_is_heavier_on_the_silhouette_and_skips_nearly_equal_colours():
    img, masks = character()
    palette = palette_for([img])
    labels = regions(img, palette)
    lines = line_art(labels, img, palette=palette)
    assert lines and all(line.kind == "line" for line in lines)
    figure = masks["figure"].float()[None, None]
    near_figure = torch.nn.functional.max_pool2d(figure, 7, 1, 3)[0, 0] > 0
    sky = ~near_figure & (torch.arange(160)[:, None] < 0.75 * 160)
    for line in lines:
        p = line.points * 160
        xi, yi = p[:, 0].long().clamp(0, 159), p[:, 1].long().clamp(0, 159)
        assert float(sky[yi, xi].float().mean()) < 0.1              # no line across the bands of the sky
        if line.colour == (0.07, 0.07, 0.09):
            assert float(near_figure[yi, xi].float().mean()) > 0.9  # black ink only on the character
    rim = torch.nn.functional.max_pool2d(figure, 3, 1, 1)[0, 0] - (-torch.nn.functional.max_pool2d(-figure, 3, 1, 1)
                                                                    [0, 0])
    heavy = [line for line in lines if float(line.pressure.max()) > 0.9]
    light = [line for line in lines if float(line.pressure.max()) < 0.7]
    assert heavy and light
    for line in heavy:                                              # the heavy ones run round the silhouette
        p = line.points * 160
        xi, yi = p[:, 0].long().clamp(0, 159), p[:, 1].long().clamp(0, 159)
        assert float(rim[yi, xi].mean()) > 0.6
    silhouette = sum(line.length() for line in heavy) * 160
    assert silhouette > 0.8 * float(rim.sum()) / 2 and len(heavy) <= 6     # long strokes, joined at junctions


def test_a_frame_is_painted_as_a_clean_cel():
    img, masks = character()
    palette = palette_for([img])
    lines = paint_frame(img, palette)
    kinds = [line.kind for line in lines]
    assert "line" in kinds and "cel" in kinds and "cel-shadow" in kinds
    assert max(i for i, k in enumerate(kinds) if k == "line") < kinds.index("cel")       # lines, then paint
    paint = [line for line in lines if line.kind in CELS]
    assert all(0 <= line.marker < len(palette) for line in paint)
    for width in (160, 320):
        picture = render(lines, width, width)
        src = torch.nn.functional.interpolate(img[None], size=(width, width), mode="bilinear",
                                              align_corners=False)[0]
        assert float(delta_e(picture, src).mean()) < 0.03
        bare = Board(width, width).picture()
        assert float(((picture - bare).abs().max(0).values < 0.004).float().mean()) == 0.0   # no white gaps
    picture = render(lines, 320, 320, waver=0.0)
    # crisp: inside the shirt the paint is one colour, not a blur
    core = picture[:, int(0.62 * 320) : int(0.8 * 320), int(0.36 * 320) : int(0.5 * 320)]
    assert float(core.std(dim=(1, 2)).max()) < 0.01
    character_only = paint_frame(img, palette, background=False)
    sky = render(character_only, 160, 160)[:, 5:40, 5:40]
    assert float((sky - Board(160, 160).picture()[:, 5:40, 5:40]).abs().max()) < 0.004   # the sky left bare
    assert sum(1 for line in character_only if line.kind in CELS) < sum(1 for line in paint)


def test_the_same_palette_paints_the_next_frame_the_same_way():
    first, _ = character()
    second, _ = character(shift=0.02)                                       # 3 pixels to the right
    palette = palette_for([first, second])
    a, b = paint_frame(first, palette), paint_frame(second, palette)
    assert {line.colour for line in a if line.kind == "cel"} == {line.colour for line in b if line.kind == "cel"}
    gaps = delta_e(palette.T[:, :, None], torch.tensor(SKIN)[:, None, None].expand(3, len(palette), 1))
    skin = int(gaps.flatten().argmin())

    def face(lines: list) -> torch.Tensor:                                 # the middle of the skin's outline
        return torch.cat([line.points for line in lines if line.kind == "cel" and line.marker == skin]).mean(0)

    move = (face(b) - face(a)) * 160
    assert 2.0 < float(move[0]) < 4.5 and abs(float(move[1])) < 1.0        # the same shape, moved with the frame


def face(size: int = 160) -> tuple[torch.Tensor, dict]:
    """A face as a video model paints one: skin with a few soft darker blotches and a shadow under the fringe,
    red hair with a dark side, two eyes (a lash line in the colour of the dark of the hair, touching it, and a
    brown iris), a mouth, and the soft shadow the head casts on a cyan wall."""
    s = size
    yy, xx = torch.meshgrid(torch.arange(s).float() + 0.5, torch.arange(s).float() + 0.5, indexing="ij")
    wall, cast = torch.tensor([0.1, 0.69, 0.82]), torch.tensor([0.05, 0.45, 0.55])
    img = wall[:, None, None].expand(3, s, s).clone()
    blob = ((xx - 0.8 * s) / 0.12) ** 2 + ((yy - 0.45 * s) / 0.3) ** 2 < s ** 2
    soft = gaussian_blur(blob.float(), 4.0)
    img = img * (1 - soft) + cast[:, None, None] * soft                   # the shadow on the wall, soft-edged
    skin, blotch, fringe = (0.95, 0.7, 0.6), (0.9, 0.6, 0.5), (0.85, 0.53, 0.42)
    head = ((xx - 0.5 * s) ** 2 + (yy - 0.55 * s) ** 2) < (0.3 * s) ** 2
    hair = head & (yy < 0.38 * s) | ((((xx - 0.5 * s) ** 2 + (yy - 0.42 * s) ** 2) < (0.33 * s) ** 2) & (yy < 0.4 * s))
    masks = {"head": head & ~hair, "hair": hair}
    img[:, head] = torch.tensor(skin)[:, None]
    g = torch.Generator().manual_seed(3)
    for _ in range(5):                                                     # blotches a few pixels across
        cx, cy = 0.3 * s + 0.4 * s * torch.rand(1, generator=g), 0.6 * s + 0.2 * s * torch.rand(1, generator=g)
        spot = head & (((xx - cx) ** 2 + (yy - cy) ** 2) < (0.025 * s) ** 2)
        img[:, spot] = torch.tensor(blotch)[:, None]
    masks["fringe"] = head & ~hair & (yy < 0.46 * s)
    img[:, masks["fringe"]] = torch.tensor(fringe)[:, None]
    img[:, hair] = torch.tensor([0.75, 0.05, 0.03])[:, None]
    dark = torch.tensor([0.25, 0.04, 0.05])
    img[:, hair & (xx > 0.62 * s)] = dark[:, None]
    lashes = torch.zeros(s, s, dtype=torch.bool)
    irises = torch.zeros(s, s, dtype=torch.bool)
    for ex in (0.38, 0.62):
        lashes |= ((xx - ex * s).abs() < 0.07 * s) & ((yy - 0.52 * s).abs() < 1.6)
        irises |= ((xx - ex * s) ** 2 + (yy - 0.56 * s) ** 2) < (0.03 * s) ** 2
    lashes |= ((xx - 0.69 * s).abs() < 1.6) & (yy > 0.4 * s) & (yy < 0.53 * s)    # one runs up into the hair
    masks["lashes"], masks["irises"] = lashes & ~hair, irises
    img[:, lashes] = dark[:, None]
    img[:, irises] = torch.tensor([0.4, 0.27, 0.2])[:, None]
    masks["mouth"] = ((xx - 0.5 * s).abs() < 0.08 * s) & ((yy - 0.72 * s).abs() < 1.6)
    img[:, masks["mouth"]] = torch.tensor([0.8, 0.15, 0.15])[:, None]
    masks["cast"] = blob & (soft > 0.9) & ~head & ~hair
    img = gaussian_blur(img, 0.6) + 0.008 * torch.randn(3, s, s, generator=g)
    return img.clamp(0, 1), masks


def test_skin_is_one_flat_colour_with_a_few_simple_shadow_shapes():
    img, masks = face()
    palette = palette_for([img])
    labels = regions(img, palette)
    shading = shade(img, labels, palette, relief=False)
    middle = shading.regions[int(0.65 * 160), 80]
    skin = shading.regions == middle
    assert float((skin & masks["head"]).sum()) > 0.8 * float(masks["head"].sum() - masks["irises"].sum())
    shadow, _ = shading.masks(int(middle))
    pieces, count = parts(torch.where(shadow, 1, 0).long())
    assert len(torch.unique(pieces[shadow])) <= 3                          # a few shapes, not a patchwork
    assert iou(shadow, masks["fringe"]) > 0.6                              # the shadow under the fringe
    lower = masks["head"] & (torch.arange(160)[:, None] > 0.58 * 160)
    assert float(shadow[lower].float().mean()) < 0.05                     # the blotches painted as skin
    lit = shading.colour[shading.tones[lower & skin & ~shadow]]
    assert float((lit - lit[0]).abs().max()) == 0.0                         # one flat base colour
    assert float((lower & skin).sum()) > 0.9 * float((lower & ~masks["mouth"] & ~masks["irises"]).sum())


def test_the_features_of_a_face_stay_and_a_shadow_on_the_wall_is_background():
    img, masks = face()
    palette = palette_for([img])
    labels = regions(img, palette)
    shading = shade(img, labels, palette)
    skin = int(shading.regions[int(0.65 * 160), 80])
    hair = int(shading.regions[int(0.3 * 160), 80])
    for name in ("lashes", "irises"):
        on = shading.regions[masks[name]]
        mine = float(((on != skin) & (on != hair)).float().mean())
        assert mine > 0.7, name                                             # not swallowed by the skin or hair
    lash = shading.colour[shading.tones[masks["lashes"]]]
    assert float(srgb_to_oklab(lash)[:, 0].mean()) < 0.4                   # and painted dark
    cast = shading.regions[masks["cast"]]
    assert float(shading.back[cast].float().mean()) > 0.9                  # the cast shadow is the wall's
    lines = paint_frame(img, palette)
    assert all(line.tool == "brush pen" for line in lines if line.kind == "line")
    inside = -torch.nn.functional.max_pool2d(-masks["cast"].float()[None, None], 5, 1, 2)[0, 0] > 0
    for line in lines:
        if line.kind == "line":
            p = line.points * 160
            xi, yi = p[:, 0].long().clamp(0, 159), p[:, 1].long().clamp(0, 159)
            assert float(inside[yi, xi].float().mean()) < 0.05             # and gets no line
