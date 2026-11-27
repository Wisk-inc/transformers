# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The space of a clip: depth steadied over time, surfaces and light, the moving character told from the set,
the camera, and one clean, sharp picture of the set. Everything is synthetic and small; no network."""

import math
import sys

import torch

from helix_lm.paint.flow import warp
from helix_lm.paint.see import gaussian_blur
from helix_lm.paint.space import (
    background_plate,
    camera,
    clip_flows,
    estimate_depth,
    light_direction,
    normals,
    place_background,
    read_space,
    separate,
)


def texture(h: int, w: int, seed: int = 0) -> torch.Tensor:
    """A set with detail everywhere: blotches of colour at two sizes and thin stripes, sRGB ``(3, h, w)``."""
    g = torch.Generator().manual_seed(seed)
    big = torch.stack([gaussian_blur(torch.rand(h, w, generator=g), 6.0) for _ in range(3)])
    small = torch.stack([gaussian_blur(torch.rand(h, w, generator=g), 1.5) for _ in range(3)])
    x = big / big.amax((1, 2), keepdim=True) + 0.5 * small / small.amax((1, 2), keepdim=True)
    x = x + 0.15 * torch.sin(torch.arange(w).float() / 3.0)[None, None, :]
    x = (x - x.amin((1, 2), keepdim=True)) / (x.amax((1, 2), keepdim=True) - x.amin((1, 2), keepdim=True))
    return 0.15 + 0.6 * x * torch.tensor([0.5, 0.8, 1.0])[:, None, None]       # bluish, never orange


def disc_clip(frames: int = 10, size: int = 128, step: float = 7.0, pan: int = 0, radius: float = 14.0):
    """A ball (orange, black outline) rolling right over a textured set; the camera pans ``pan`` pixels a frame
    to the right (the set slides left). Returns the frames, the true masks and the set."""
    world = texture(size, size + pan * frames)
    yy, xx = torch.meshgrid(torch.arange(size).float() + 0.5, torch.arange(size).float() + 0.5, indexing="ij")
    clip, masks = [], []
    for t in range(frames):
        img = world[:, :, pan * t : pan * t + size].clone()
        r = ((xx - (24.0 + step * t)) ** 2 + (yy - 0.5 * size) ** 2).sqrt()
        inside = r < radius
        img[:, inside] = 0.05
        img[:, r < radius - 2.0] = torch.tensor([0.95, 0.45, 0.1])[:, None]
        clip.append(img)
        masks.append(inside)
    return torch.stack(clip), torch.stack(masks), world


def iou(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a & b).sum()) / max(1.0, float((a | b).sum()))


def test_the_moving_character_is_told_from_a_still_set_and_the_set_is_painted_once_without_it():
    frames, truth, world = disc_clip()
    flows = clip_flows(frames)
    found = separate(frames, flows)
    assert min(iou(found[t], truth[t]) for t in range(len(frames))) > 0.9
    cams = camera(flows, ~found, frames)
    assert torch.equal(cams, torch.eye(2, 3).expand(len(frames), 2, 3))        # still: the identity, exactly
    plate = background_plate(frames, found, cams)
    assert plate.image.shape == (3, 128, 128) and plate.origin == (0, 0) and bool(plate.seen.all())
    # where the set was seen the plate is the set itself, pixel for pixel: a median of the same pixels, not an
    # average of moved ones, so nothing is blurred; and no trace of the orange ball is left
    assert float((plate.image - world).abs().max()) < 1e-6
    assert float((plate.image[0] - plate.image[2]).max()) < 0.2
    assert torch.equal(place_background(plate, cams[3], 128, 128), plate.image)


def test_depth_finds_the_character_that_walks_on_the_spot():
    # the ball only bobs a little, so most of it is always in the same place: the median of the frames is the
    # ball itself, and only depth tells it from the set (it is nearer than the wall behind it)
    frames, truth, world = disc_clip(frames=8, step=0.0, radius=18.0)
    yy, xx = torch.meshgrid(torch.arange(128).float() + 0.5, torch.arange(128).float() + 0.5, indexing="ij")
    clip, masks, depth = [], [], []
    for t in range(8):
        cx, cy = 64.0 + 2.0 * math.sin(t), 60.0 + 3.0 * math.cos(1.3 * t)
        r = ((xx - cx) ** 2 + (yy - cy) ** 2).sqrt()
        img = world.clone()
        img[:, r < 18.0] = 0.05
        img[:, r < 16.0] = torch.tensor([0.95, 0.45, 0.1])[:, None]
        clip.append(img)
        masks.append(r < 18.0)
        depth.append(torch.where(r < 18.0, 0.6 + 0.2 * (1 - (r / 18.0) ** 2).clamp(min=0).sqrt(), 0.1 + 0.2 * yy / 128))
    frames, truth, depth = torch.stack(clip), torch.stack(masks), torch.stack(depth)
    space = read_space(frames, depth)
    assert min(iou(space.character[t], truth[t]) for t in range(8)) > 0.9
    plate = space.plate
    assert float((plate.image - world)[:, plate.seen].abs().max()) < 1e-6      # the set, where it was seen
    hole = ~plate.seen                                     # under the ball in every frame: filled from round it
    assert 0 < int(hole.sum()) < 0.1 * 128 * 128
    assert float((plate.image[0] - plate.image[2])[hole].max()) < 0.1          # set-coloured, not ball-coloured


def test_a_panning_camera_is_recovered_and_the_plate_covers_all_of_the_set_it_saw():
    frames, truth, world = disc_clip(frames=8, size=112, step=6.0, pan=2)
    flows = clip_flows(frames)
    found = separate(frames, flows)
    assert min(iou(found[t], truth[t]) for t in range(8)) > 0.85
    cams = camera(flows, ~found, frames, "similarity")
    for t in range(8):
        # frame 0's point x is in frame t at x - 2t (the set slides left under a camera panning right)
        assert abs(float(cams[t, 0, 2]) + 2.0 * t) < 0.2 and abs(float(cams[t, 1, 2])) < 0.2
        assert float((cams[t, :, :2] - torch.eye(2)).abs().max()) < 0.003
    plate = background_plate(frames, found, cams)
    assert plate.origin == (0, 0) and plate.image.shape == (3, 112, 112 + 2 * 7)
    seen = plate.seen
    assert float((plate.image - world[:, :, : 112 + 14])[:, seen].abs().mean()) < 0.01
    for t in (0, 4, 7):
        shown = place_background(plate, cams[t], 112, 112)
        away = ~torch.nn.functional.max_pool2d(truth[t].float()[None], 9, 1, 4)[0].bool()
        assert float((shown - frames[t])[:, away].abs().mean()) < 0.02


def test_depth_is_steadied_over_time_and_its_absence_is_not_a_crash(monkeypatch):
    frames, _, _ = disc_clip(frames=6, size=96, step=4.0)
    ramp = torch.linspace(0.2, 1.0, 96)[:, None].expand(96, 96)                 # the floor comes nearer below
    g = torch.Generator().manual_seed(1)
    calls = []

    def flickering(frame):                        # a model that sees every frame alone: its scale wanders
        calls.append(frame.shape)
        a, b = 0.5 + 1.5 * float(torch.rand(1, generator=g)), float(torch.rand(1, generator=g))
        return a * ramp[::2, ::2] + b + 0.02 * torch.rand(48, 48, generator=g)

    depth = estimate_depth(frames, estimator=flickering)
    assert depth.shape == (6, 96, 96) and len(calls) == 6
    assert float(depth.min()) >= 0.0 and float(depth.max()) <= 1.0
    assert float(depth[:, 80:].mean()) > float(depth[:, :16].mean()) + 0.5     # larger is nearer, kept
    middle = depth[:, 8:-8, 8:-8]
    assert float((middle - middle.mean(0)).abs().max()) < 0.06                  # no flicker from frame to frame
    monkeypatch.setitem(sys.modules, "transformers", None)
    with __import__("pytest").warns(UserWarning, match="transformers"):
        assert estimate_depth(frames) is None


def sphere(size: int = 96, radius: float = 30.0):
    yy, xx = torch.meshgrid(torch.arange(size).float() + 0.5, torch.arange(size).float() + 0.5, indexing="ij")
    dx, dy = xx - size / 2, yy - size / 2
    r2 = dx ** 2 + dy ** 2
    inside = r2 < radius ** 2
    height = (radius ** 2 - r2).clamp(min=0).sqrt()
    depth = torch.where(inside, 20.0 + height, torch.zeros_like(height))   # a ball in front of a far wall
    truth = torch.stack([dx, dy, height]) / radius
    truth = torch.where(inside[None], truth, torch.tensor([0.0, 0.0, 1.0])[:, None, None])
    return depth, truth, inside, r2.sqrt()


def test_the_normals_of_a_ball_face_out_and_do_not_bleed_into_the_wall_behind_it():
    depth, truth, inside, r = sphere()
    n = normals(depth, scale=1.0)
    assert torch.allclose(n.norm(dim=0), torch.ones(96, 96), atol=1e-4)
    core = r < 0.85 * 30
    angle = torch.rad2deg(torch.acos((n * truth).sum(0).clamp(-1, 1)))
    assert float(angle[core].mean()) < 4.0 and float(angle[core].max()) < 12.0
    c = 48
    assert float(n[0, c, c + 21]) > 0.5 and float(n[0, c, c - 21]) < -0.5        # right side right, left left
    assert float(n[1, c - 21, c]) < -0.5 and float(n[1, c + 21, c]) > 0.5        # top up (y is down), bottom down
    assert float(n[2, c, c]) > 0.99                                              # the front faces the viewer
    rim = (r > 26) & (r < 29)                                                    # just inside the outline
    assert float(((n[:2] * truth[:2]).sum(0) / truth[:2].norm(dim=0).clamp(min=1e-6))[rim].mean()) > 0.75
    wall = (r > 31) & (r < 34)                                                   # just outside it
    assert float(n[2][wall].min()) > 0.99


def test_the_light_is_found_from_how_a_shaded_ball_is_lit():
    depth, _, inside, r = sphere()
    n = normals(depth, scale=1.0)
    for light in ([-0.5, -0.6, 0.7], [0.8, 0.1, 0.6]):
        light = torch.tensor(light) / torch.tensor(light).norm()
        shade = 0.08 + 0.85 * (n * light[:, None, None]).sum(0).clamp(min=0)
        linear = shade[None] * torch.tensor([0.9, 0.5, 0.2])[:, None, None]       # an orange ball
        image = torch.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear.clamp(min=1e-8) ** (1 / 2.4) - 0.055)
        image[:, (r > 28.5) & (r < 30.5)] = 0.03                                  # its outline, inked
        found = light_direction(image, n, inside)
        assert abs(float(found.norm()) - 1.0) < 1e-5
        assert math.degrees(math.acos(float((found * light).sum().clamp(-1, 1)))) < 10.0


def test_flow_inverted_and_masks_steady_on_a_clip_of_ones():
    frames, truth, _ = disc_clip(frames=3, size=96, step=5.0)
    flows = clip_flows(frames, scale=0.5)
    assert flows.shape == (2, 2, 96, 96)
    moved = warp(truth[0].float()[None], -flows[0])[0] > 0.5   # roughly where the ball went
    assert float((moved & truth[1]).sum()) > 0.3 * float(truth[1].sum())
    assert separate(frames[:1]).shape == (1, 96, 96)
