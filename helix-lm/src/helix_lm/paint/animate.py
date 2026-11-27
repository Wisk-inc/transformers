# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Animation drawn by hand, one frame after the other.

An animator does not draw every frame from nothing. The last drawing goes on the lightbox and a new sheet over
it: what has not changed is traced, what has moved is drawn where it now is, and the same pens and markers are
used throughout. That is what keeps a character looking like itself from frame to frame. Pinxit works the same
way (:func:`draw_frames`):

1. the frames to draw come from an open video model asked in words (:mod:`helix_lm.paint.video`), or from a
   video of one's own;
2. the markers are chosen once, from all the frames together, and every frame is coloured with them;
3. the first frame is drawn in full: the lines with a pen, then the colour marker by marker
   (:mod:`helix_lm.paint.draw`);
4. for every next frame, how the picture moved is measured (:func:`helix_lm.paint.flow.optical_flow`) and the
   last drawing is laid over the new frame, moved with it. A line that still lies on a line of the new frame
   is traced (pulled onto it, drawn again by hand, so it is never quite the same line: the "boil" of drawn
   animation); a line that no longer fits is left off; a line of the new frame that nothing covers is drawn new.
   Colour is done the same way, shape by shape. A frame that changed almost everywhere (a cut) is drawn from
   nothing.

An :class:`Animation` keeps all of it: every frame's lines, the markers, the frames it was drawn from. It can be
saved and opened again, played (``.mp4``, ``.gif``), shown being drawn on a whiteboard with the pen in hand
(:meth:`Animation.making_of`), recoloured in every frame at once, drawn with another pen, or carried on with a
next clip that starts where this one ends (:meth:`Animation.extend`).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .color import linear_to_oklab, linear_to_srgb, oklab_to_linear, srgb_to_linear
from .draw import (
    KINDS,
    TOOL_NAMES,
    TOOLS,
    Board,
    Line,
    _cover,
    colour_lines,
    dedupe,
    draw_picture,
    ink_lines,
    look,
    markers,
    order,
    stamp,
    timing,
    trace,
)
from .flow import nearest, optical_flow, sample
from .io import load_image, to_pil
from .see import gaussian_blur

VERSION = 1


@dataclass
class Sheet:
    """One frame's drawing: its lines in the order they were drawn, how many were traced from the frame before
    and how many drawn new, and whether it was drawn from nothing."""

    lines: list
    kept: int = 0
    new: int = 0
    fresh: bool = True


# ----------------------------------------------------------------------------------------- lightbox --
def _odd(x: float) -> int:
    k = max(1, int(round(x)))
    return k if k % 2 else k + 1


def _open(mask: torch.Tensor, size: int) -> torch.Tensor:
    """``mask`` without the parts thinner than ``size`` pixels."""
    if size <= 1:
        return mask
    m = mask[None, None].float()
    m = -F.max_pool2d(-m, size, 1, size // 2)
    return (F.max_pool2d(m, size, 1, size // 2)[0, 0] > 0.5) & mask


def trace_over(before: list, flow: torch.Tensor, seen, palette: torch.Tensor, lines: str = "pen",
               colour: str | None = "marker") -> tuple[list, int, int]:
    """The lightbox: the drawing ``before`` laid over the new frame (``seen``, see :func:`helix_lm.paint.draw.look`)
    and moved with it by ``flow (2, H, W)``. Returns the new frame's lines, how many were traced from ``before``
    and how many drawn new."""
    h, w = seen.lines.shape
    scale = float(w)
    near_xy, dist = nearest(seen.lines)
    kept_ink, kept_colour = [], []
    for line in before:
        pts = line.points * scale
        moved = pts + sample(flow, pts)
        if len(pts) > 1:                                   # torn or squeezed by the move: drawn again instead
            before_len = (pts[1:] - pts[:-1]).norm(dim=-1)
            after_len = (moved[1:] - moved[:-1]).norm(dim=-1)
            change = (after_len - before_len).abs() / before_len.clamp(min=1.0)
            if float(change.max()) > 0.35 and float((after_len - before_len).abs().max()) > 2.0:
                continue
        if line.kind in ("line", "sketch"):
            d = sample(dist[None], moved)[:, 0]
            if float(d.mean()) <= 1.8 and float((d < 2.5).float().mean()) >= 0.75:
                target = sample(near_xy, moved)
                close = (d < 2.5)[:, None]
                pulled = torch.where(close, moved + 0.6 * (target - moved), moved)
                if len(pulled) >= 3:                       # the pull makes small kinks: smooth them out
                    pulled = torch.cat([pulled[:1], (pulled[:-2] + 2 * pulled[1:-1] + pulled[2:]) / 4, pulled[-1:]])
                kept_ink.append(line.moved(pulled / scale))
        else:
            xi, yi = moved[:, 0].long().clamp(0, w - 1), moved[:, 1].long().clamp(0, h - 1)
            here = seen.labels[yi, xi] == line.marker
            if line.kind == "shade":
                here = here & seen.darker[yi, xi]
            inside = (moved[:, 0] >= 0) & (moved[:, 0] < w) & (moved[:, 1] >= 0) & (moved[:, 1] < h)
            if float((here & inside).float().mean()) >= 0.75:
                kept_colour.append(line.moved(moved / scale))
    covered = stamp(kept_ink, h, w, scale, grow=2.0)
    left = seen.lines & ~covered
    tool = lines or "pen"
    new_ink = ink_lines(trace(left, least=max(8, round(0.025 * scale))), seen.strength, scale, tool)
    new_colour: list = []
    if colour:
        broad = TOOLS["marker"].width * scale
        masks, shade_masks, drop = {}, {}, set()
        for k in range(len(palette)):
            for kind, store in (("fill", masks), ("shade", shade_masks)):
                region = seen.labels == k
                if kind == "shade":
                    region = region & seen.darker
                mine = [i for i, line in enumerate(kept_colour) if line.marker == k and line.kind == kind]
                if not bool(region.any()):
                    continue
                cov = stamp([kept_colour[i] for i in mine], h, w, scale, grow=1.5)
                uncovered = region & ~cov
                patch = (gaussian_blur(uncovered.float(), 1.5 * broad) > 0.45) & region
                for i in mine:                             # a stroke where most is missing: that part is redone
                    line = kept_colour[i]
                    p = (line.points * scale).long()
                    p[:, 0].clamp_(0, w - 1)
                    p[:, 1].clamp_(0, h - 1)
                    if float(patch[p[:, 1], p[:, 0]].float().mean()) > 0.3:
                        drop.add(i)
                store[k] = region & (patch | _open(uncovered, _odd(0.5 * broad)))
        kept_colour = [line for i, line in enumerate(kept_colour) if i not in drop]
        new_colour = colour_lines(seen.labels, seen.darker, palette, scale, masks=masks, shade_masks=shade_masks)
    drawn = order(dedupe(kept_ink + new_ink + kept_colour + new_colour, h, w, scale))
    kept = sum(1 for line in drawn if line.age > 0)
    return drawn, kept, len(drawn) - kept


def draw_frames(frames: torch.Tensor, palette: torch.Tensor | None = None, *, lines: str = "pen",
                colour: str | None = "marker", sketch: bool = False, detail: float = 0.5, start: list | None = None,
                start_frame: torch.Tensor | None = None, log=None) -> tuple[list, torch.Tensor]:
    """Draw ``frames (T, 3, H, W)`` (sRGB) one after the other, each over the last on the lightbox (see the
    module's notes). ``start`` and ``start_frame`` are a drawing and the frame it was drawn from to go on
    from. Returns the :class:`Sheet` of every frame and the markers used."""
    palette = markers(list(frames)) if palette is None else palette
    sheets, before, before_frame = [], start, start_frame
    for t, frame in enumerate(frames):
        seen = look(frame, palette, detail)
        sheet = None
        if before is not None:
            flow = optical_flow(before_frame, frame)
            drawn, kept, new = trace_over(before, flow, seen, palette, lines, colour)
            ink_before = sum(line.length() for line in before if line.kind == "line")
            ink_kept = sum(line.length() for line in drawn if line.kind == "line" and line.age > 0)
            if ink_before <= 0 or ink_kept / ink_before >= 0.3:
                sheet = Sheet(drawn, kept, new, False)
        if sheet is None:                                  # the first frame, or a cut: a new drawing
            drawn = draw_picture(frame, palette, lines=lines, colour=colour, sketch=sketch, detail=detail, seen=seen)
            sheet = Sheet(drawn, 0, len(drawn), True)
        sheets.append(sheet)
        if log:
            what = "drawn in full" if sheet.fresh else \
                f"{sheet.kept} lines traced from the frame before, {sheet.new} new"
            log(f"frame {t + 1} of {len(frames)}: {what}")
        before, before_frame = sheet.lines, frame
    return sheets, palette


# ------------------------------------------------------------------------------------------- video files --
def _ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:
        raise RuntimeError("writing or reading .mp4 needs ffmpeg: install it, or pip install imageio-ffmpeg "
                           "(a .gif needs neither)") from error


class VideoWriter:
    """Frames ``(3, H, W)`` (sRGB, 0..1) into a video file: ``.gif`` with Pillow, anything else with ffmpeg."""

    def __init__(self, path, fps: float) -> None:
        self.path, self.fps = Path(path), fps
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.gif = self.path.suffix.lower() == ".gif"
        self.pictures, self.proc, self.size = [], None, None

    def add(self, frame: torch.Tensor) -> None:
        frame = frame.clamp(0.0, 1.0)
        if self.gif:
            self.pictures.append(to_pil(frame))
            return
        h, w = frame.shape[-2:]
        if self.proc is None:
            self.size = (h - h % 2, w - w % 2)
            self.proc = subprocess.Popen(
                [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                 f"{self.size[1]}x{self.size[0]}", "-r", f"{self.fps}", "-i", "-", "-c:v", "libx264", "-pix_fmt",
                 "yuv420p", "-crf", "18", "-movflags", "+faststart", str(self.path)], stdin=subprocess.PIPE)
        frame = frame[:, : self.size[0], : self.size[1]]
        self.proc.stdin.write((frame * 255).round().byte().permute(1, 2, 0).contiguous().numpy().tobytes())

    def close(self) -> Path:
        if self.gif and self.pictures:
            self.pictures[0].save(self.path, save_all=True, append_images=self.pictures[1:],
                                  duration=round(1000 / self.fps), loop=0, optimize=False)
        elif self.proc is not None:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError(f"ffmpeg could not write {self.path}")
        return self.path


def write_video(frames, path, fps: float) -> Path:
    writer = VideoWriter(path, fps)
    for frame in frames:
        writer.add(frame)
    return writer.close()


def read_video(path, most: int | None = None, max_side: int = 640) -> tuple[torch.Tensor, float]:
    """A video (``.gif``, ``.mp4`` and whatever ffmpeg reads) or a folder of pictures as ``(T, 3, H, W)`` sRGB
    frames, and its frames a second."""
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))[:most]
        return torch.stack([load_image(p, max_side=max_side) for p in files]), 8.0
    if path.suffix.lower() == ".gif":
        from PIL import Image, ImageSequence

        with Image.open(path) as gif:
            fps = 1000.0 / max(20, gif.info.get("duration", 125))
            pictures = [load_image(f.convert("RGB"), max_side=max_side) for f in ImageSequence.Iterator(gif)][:most]
        return torch.stack(pictures), fps
    with tempfile.TemporaryDirectory() as tmp:
        probe = subprocess.run([_ffmpeg(), "-i", str(path)], capture_output=True, text=True).stderr
        fps = 8.0
        for part in probe.split(","):
            if part.strip().endswith(" fps"):
                fps = float(part.strip().split()[0])
                break
        subprocess.run([_ffmpeg(), "-loglevel", "error", "-i", str(path), *(["-frames:v", str(most)] if most else []),
                        f"{tmp}/%05d.png"], check=True)
        files = sorted(Path(tmp).glob("*.png"))
        return torch.stack([load_image(p, max_side=max_side) for p in files]), fps


# --------------------------------------------------------------------------------------- the whiteboard --
WALL, LEDGE = (0.83, 0.82, 0.79), (0.74, 0.75, 0.77)


def _capsule(canvas: torch.Tensor, a, b, radius: float, colour, alpha: float = 1.0, aa: float = 1.2) -> None:
    """A rounded bar from ``a`` to ``b`` (pixels) painted on ``canvas`` (linear RGB, in place)."""
    pts = torch.tensor([a, b], dtype=torch.float32)
    got = _cover(pts, torch.tensor([radius, radius]), canvas.shape[1], canvas.shape[2], aa)
    if got is None:
        return
    y0, x0, cov = got
    h, w = cov.shape
    ink = srgb_to_linear(torch.tensor(colour, dtype=torch.float32))[:, None, None]
    region = canvas[:, y0 : y0 + h, x0 : x0 + w]
    region.mul_(1.0 - alpha * cov).add_(ink * alpha * cov)


def _in_hand(canvas: torch.Tensor, tip, tool: str, colour, size: float) -> None:
    """The tool being drawn with, its point at ``tip`` (pixels), held the way a right hand holds it."""
    x, y = float(tip[0]), float(tip[1])
    lean = torch.tensor([0.42, 0.91])
    length = 0.24 * size
    end = (x + lean[0] * length, y + lean[1] * length)
    shade = (x + lean[0] * length + 0.02 * size, y + lean[1] * length + 0.025 * size)
    thick = {"pen": 0.011, "brush pen": 0.012, "pencil": 0.012, "marker": 0.02, "fine marker": 0.016}[tool] * size
    _capsule(canvas, (x + 0.01 * size, y + 0.012 * size), shade, thick * 1.1, (0.0, 0.0, 0.0), 0.18, 4.0)
    near = (x + lean[0] * 0.05 * size, y + lean[1] * 0.05 * size)
    if tool == "pencil":
        _capsule(canvas, near, end, thick, (0.95, 0.76, 0.18))
        _capsule(canvas, (x, y), near, 0.45 * thick, (0.86, 0.68, 0.46))
        _capsule(canvas, (x, y), (x + lean[0] * 0.012 * size, y + lean[1] * 0.012 * size), 0.25 * thick,
                 (0.22, 0.22, 0.24))
        rubber = (end[0] - lean[0] * 0.02 * size, end[1] - lean[1] * 0.02 * size)
        _capsule(canvas, rubber, end, thick, (0.9, 0.5, 0.55))
    elif tool in ("marker", "fine marker"):
        _capsule(canvas, near, end, thick, (0.94, 0.94, 0.93))
        band = (near[0] + lean[0] * 0.03 * size, near[1] + lean[1] * 0.03 * size)
        _capsule(canvas, near, band, thick * 1.02, colour)
        _capsule(canvas, (x, y), near, 0.4 * thick, colour)
    else:
        _capsule(canvas, near, end, thick, (0.13, 0.13, 0.15))
        grip = (near[0] + lean[0] * 0.05 * size, near[1] + lean[1] * 0.05 * size)
        _capsule(canvas, near, grip, thick * 1.05, (0.3, 0.3, 0.33))
        _capsule(canvas, (x, y), near, 0.3 * thick, (0.6, 0.6, 0.62))


def _ledge(canvas: torch.Tensor, top: int, palette: torch.Tensor, holding, size: float) -> None:
    """The ledge under the board with the pens and markers on it; the one in hand is missing."""
    h, w = canvas.shape[1:]
    canvas[:, top : top + max(2, int(0.012 * size))] = srgb_to_linear(torch.tensor((0.9, 0.9, 0.91)))[:, None, None]
    y = top + 0.04 * size
    x = 0.04 * size
    gap = 0.03 * size
    things = [("pencil", TOOLS["pencil"].ink), ("pen", TOOLS["pen"].ink)] + [("marker", tuple(c.tolist()))
                                                                            for c in palette]
    for tool, colour in things:
        length = 0.11 * size if tool == "marker" else 0.13 * size
        if x + length > w - 0.02 * size:
            break
        if holding is None or (tool, tuple(round(c, 3) for c in colour)) != holding:
            if tool == "marker":
                _capsule(canvas, (x, y), (x + length, y), 0.013 * size, (0.94, 0.94, 0.93))
                _capsule(canvas, (x + length - 0.025 * size, y), (x + length, y), 0.0135 * size, colour)
            elif tool == "pencil":
                _capsule(canvas, (x, y), (x + length, y), 0.007 * size, (0.95, 0.76, 0.18))
            else:
                _capsule(canvas, (x, y), (x + length, y), 0.006 * size, (0.13, 0.13, 0.15))
        x += length + gap


# -------------------------------------------------------------------------------------------- animation --
class Animation:
    """A drawn animation and its memory: every frame's lines, the markers, and the frames it was drawn from.
    ``background`` is the background art, drawn once and kept under every frame (None: each frame is drawn
    whole). ``holds`` says for how many frames of the video each drawing is shown, as on an exposure sheet
    (None: one each); ``fps`` is frames of the video a second."""

    def __init__(self, sheets: list, palette: torch.Tensor, frames: torch.Tensor | None, fps: float = 8.0,
                 words: str = "", surface: str = "whiteboard", history: list | None = None,
                 settings: dict | None = None, background: list | None = None, holds: list | None = None) -> None:
        self.sheets, self.palette, self.fps, self.words, self.surface = sheets, palette.float(), fps, words, surface
        self.source = None if frames is None else (frames.clamp(0, 1) * 255).round().byte()
        self.history = history or []
        self.settings = settings or {}
        self.background = background
        self.holds = holds
        self.cameras: torch.Tensor | None = None       # how the camera saw the background in each drawing
        self._backdrops: dict = {}

    def __len__(self) -> int:
        return len(self.sheets)

    def hold(self, index: int) -> int:
        """For how many frames of the video drawing ``index`` is shown."""
        return max(1, int(self.holds[index])) if self.holds else 1

    def board(self, index: int, width: int | None = None, waver: float | None = None) -> Board:
        """A board ready for drawing ``index``: bare, or with the background art already on it. The frame's own
        paint then covers the background but stays under the frame's own ink."""
        from .anime import background_for, still_camera

        h, w = self.size
        width = width or 2 * w
        waver = self.settings.get("waver", 0.35) if waver is None else waver
        board = Board(width, round(width * h / w), self.surface, waver=waver, seed=index)
        if self.background:
            camera = None if self.cameras is None or index >= len(self.cameras) else self.cameras[index]
            moving = camera is not None and not still_camera(camera)
            key = (width, self.surface, tuple(round(float(v), 3) for v in camera.flatten()) if moving else None)
            if key not in self._backdrops:
                under = Board(width, round(width * h / w), self.surface, waver=waver, seed=10_007)
                for line in background_for(self.background, camera, w) if moving else self.background:
                    under.draw(line)
                if len(self._backdrops) > 24:                    # a panning camera: keep the last few
                    self._backdrops.clear()
                self._backdrops[key] = under.canvas
            board.canvas = self._backdrops[key].clone()
        return board

    @property
    def size(self) -> tuple:
        if self.source is not None:
            return tuple(self.source.shape[-2:])
        return tuple(self.settings.get("size", (512, 512)))

    @property
    def frames(self) -> torch.Tensor | None:
        """The frames it was drawn from, sRGB ``(T, 3, H, W)``."""
        return None if self.source is None else self.source.float() / 255.0

    def frame(self, index: int, width: int | None = None, waver: float | None = None) -> torch.Tensor:
        """Frame ``index`` drawn, sRGB ``(3, H, W)``, ``width`` pixels wide (twice the frames' size when not
        given)."""
        board = self.board(index, width, waver)
        for line in self.sheets[index].lines:
            board.draw(line)
        return board.picture()

    def play(self, width: int | None = None):
        """Every frame of the video in turn (a drawing held for two frames comes twice)."""
        for i in range(len(self)):
            picture = self.frame(i, width)
            for _ in range(self.hold(i)):
                yield picture

    def save(self, path, width: int | None = None) -> Path:
        """``.pinxa``: the whole animation with its memory, to open and change later. ``.mp4``, ``.webm``,
        ``.mov``, ``.gif``: the animation. ``.png``, ``.jpg``: every frame side by side."""
        path = Path(path)
        kind = path.suffix.lower()
        if kind == ".pinxa":
            return self._save_state(path)
        if kind in (".png", ".jpg", ".jpeg"):
            tiles = [self.frame(i, width or 320) for i in range(len(self))]
            columns = min(len(tiles), 8)
            blank = torch.ones_like(tiles[0])
            rows = [torch.cat(tiles[r : r + columns] + [blank] * (columns - len(tiles[r : r + columns])), -1)
                    for r in range(0, len(tiles), columns)]
            to_pil(torch.cat(rows, -2)).save(path)
            return path
        return write_video(self.play(width), path, self.fps)

    def making_of(self, path, width: int = 720, fps: int = 24, first: float = 14.0, later: float = 3.0,
                  loops: int = 2, onion: float = 0.16) -> Path:
        """A video of the animation being drawn on a whiteboard: the pen or marker in hand moving as a hand
        would, the markers on the ledge, every frame drawn over a faint copy of the last (the lightbox), and the
        finished animation played ``loops`` times at the end. The first frame takes about ``first`` seconds,
        every next one about ``later`` (traced lines go faster than new ones)."""
        from PIL import ImageDraw, ImageFont

        h, w = self.size
        margin = round(0.04 * width)
        bw = width - 2 * margin
        bh = round(bw * h / w)
        ledge = round(0.1 * width)
        height = margin + bh + margin + ledge
        height += height % 2
        wall = srgb_to_linear(torch.tensor(WALL))[:, None, None].expand(3, height, width).clone()
        frame_colour = srgb_to_linear(torch.tensor((0.62, 0.63, 0.65)))[:, None, None]
        rim = max(2, round(0.008 * width))
        wall[:, margin - rim : margin + bh + rim, margin - rim : margin + bw + rim] = frame_colour
        font = ImageFont.load_default()
        writer = VideoWriter(path, fps)
        last = None

        ledges: dict = {}                                    # the ledge drawn once for each tool in hand

        def show(canvas: torch.Tensor, tip=None, holding=None, number: int = 0) -> None:
            scene = wall.clone()
            scene[:, margin : margin + bh, margin : margin + bw] = canvas
            if holding not in ledges:
                strip = wall.clone()
                _ledge(strip, margin + bh + rim + 2, self.palette, holding, width)
                ledges[holding] = strip[:, margin + bh + rim :]
            scene[:, margin + bh + rim :] = ledges[holding]
            if tip is not None:
                _in_hand(scene, (tip[0] + margin, tip[1] + margin), holding[0] if holding else "pen",
                         holding[1] if holding else (0.1, 0.1, 0.1), width)
            picture = to_pil(linear_to_srgb(scene.clamp(0, 1)))
            ImageDraw.Draw(picture).text((margin, max(2, margin // 4)), f"frame {number} of {len(self)}",
                                         fill=(70, 70, 75), font=font)
            writer.add(torch.from_numpy(np.asarray(picture).copy()).permute(2, 0, 1).float() / 255.0)

        for i, sheet in enumerate(self.sheets):
            board = Board(bw, bh, self.surface, seed=i)
            if last is not None and onion > 0:               # the last frame on the lightbox, under the new sheet
                board.canvas = board.canvas * (1.0 - onion) + last * onion
            times = timing(sheet.lines)
            total = float(times[-1][1][-1]) if times else 0.0
            speed = max(1e-6, total / (first if i == 0 else later))
            step = speed / fps
            clock, done = 0.0, 0
            while done < len(sheet.lines):
                clock += step
                while done < len(sheet.lines) and float(times[done][1][-1]) <= clock:
                    board.draw(sheet.lines[done])
                    done += 1
                if done >= len(sheet.lines):
                    break
                line, (begin, stamps) = sheet.lines[done], times[done]
                holding = (line.tool, tuple(round(c, 3) for c in line.colour))
                if clock < begin:                              # between lines: the pen on its way to the next one
                    show(board.canvas, board.pixels(line)[0], holding, i + 1)
                    continue
                upto = int((stamps <= clock).sum())
                show(board.peek(line, upto), board.pixels(line)[max(0, upto - 1)], holding, i + 1)
            last = board.canvas.clone()
            for _ in range(round(0.4 * fps)):
                show(board.canvas, None, None, i + 1)
        for _ in range(loops):
            for i in range(len(self)):
                board = Board(bw, bh, self.surface, seed=i)
                for line in self.sheets[i].lines:
                    board.draw(line)
                for _ in range(max(1, round(fps / self.fps))):
                    show(board.canvas, None, None, i + 1)
        return writer.close()

    # ------------------------------------------------------------------------------------------ changes --
    def marker_at(self, where: tuple, frame: int = 0) -> int:
        """The marker that colours the point ``where = (x, y)`` (shares of the picture) of a frame."""
        h, w = self.size
        best, gap = -1, float("inf")
        x, y = where[0], where[1] * h / w
        for line in self.sheets[frame].lines:
            if line.marker < 0:
                continue
            d = float((line.points - torch.tensor([x, y])).norm(dim=-1).min())
            if d < gap:
                best, gap = line.marker, d
        return best

    def recolor(self, to, marker: int | None = None, at: tuple | None = None, near=None) -> Animation:
        """Give one marker another colour (sRGB) in every frame at once: the marker by its number, by a point it
        colours (``at = (x, y)`` in the first frame) or by the colour nearest to ``near``."""
        if marker is None and at is not None:
            marker = self.marker_at(at)
        if marker is None and near is not None:
            lab = linear_to_oklab(srgb_to_linear(self.palette))
            want = linear_to_oklab(srgb_to_linear(torch.tensor(near, dtype=torch.float32)))
            marker = int((lab - want).norm(dim=-1).argmin())
        if marker is None or marker < 0:
            raise ValueError("say which marker: marker=, at=(x, y) or near=(r, g, b)")
        colour = tuple(float(c) for c in to)
        # every line of that marker moves by the same step in OKLab: the base colour becomes `to` exactly, and
        # its shadow, highlight and colour trace lines keep how they stood to it
        if marker < len(self.palette):
            base = self.palette[marker].clone()
            self.palette[marker] = torch.tensor(colour)
        else:                                   # one of the background's own colours: its base shape tells
            ours = [line for line in self._all_lines() if line.marker == marker]
            if not ours:
                raise ValueError(f"no colour number {marker} in this animation")
            base = torch.tensor(next((line.colour for line in ours if line.kind in ("cel", "fill")), ours[0].colour))
        step = linear_to_oklab(srgb_to_linear(torch.tensor(colour))) - linear_to_oklab(srgb_to_linear(base))
        for line in self._all_lines():
            if line.marker == marker:
                was = linear_to_oklab(srgb_to_linear(torch.tensor(line.colour)))
                line.colour = tuple(float(c) for c in linear_to_srgb(oklab_to_linear(was + step)).clamp(0.0, 1.0))
        self._backdrops = {}
        self.history.append({"op": "recolor", "marker": marker, "to": list(colour)})
        return self

    def _all_lines(self):
        yield from self.background or ()
        for sheet in self.sheets:
            yield from sheet.lines

    def retool(self, lines: str = "brush pen") -> Animation:
        """Draw every line of every frame with another pen (``pen``, ``brush pen``, ``pencil``)."""
        if lines not in TOOLS or TOOLS[lines].multiply:
            raise ValueError(f"lines are drawn with a pen or a pencil: one of pen, brush pen, pencil, not {lines!r}")
        for line in self._all_lines():
            if line.kind in ("line", "sketch") and line.tool != "pencil" and line.marker < 0:
                line.tool = lines
                line.colour = TOOLS[lines].ink
        self._backdrops = {}
        self.history.append({"op": "retool", "lines": lines})
        return self

    def extend(self, words: str, animator=None, frames: int | None = None, seed: int = 0, motion=None,
               log=None) -> Animation:
        """Go on with a next clip from where this one ends: the video model starts from the last frame (so the
        character and what is around it look the same), and the new frames are drawn over the last drawing
        with the same markers. ``motion`` is a video of someone moving, for a model that makes the character
        perform it (Wan-Animate)."""
        from .video import DRAWABLE, load_animator

        animator = animator or load_animator()
        h, w = self.size
        anime = self.settings.get("style") == "anime"
        if anime:
            from .anime import ASK
        clip = animator(f"{words}, {ASK if anime else DRAWABLE}", frames=frames, size=(h - h % 8, w - w % 8),
                        seed=seed, character=self.frames[-1], **({"motion": motion} if motion is not None else {}))
        clip = F.interpolate(clip, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)
        if anime:
            return self._go_on_anime(clip[1:] if len(clip) > 1 else clip, float(animator.fps), words, log)
        on = int(self.settings.get("on", 1))
        clip = clip[on::on] if len(clip) > on else clip       # its first frame is the last one already drawn
        sheets, _ = draw_frames(clip, self.palette, lines=self.settings.get("lines", "pen"),
                                colour=self.settings.get("colour", "marker"), start=self.sheets[-1].lines,
                                start_frame=self.frames[-1], log=log)
        self.sheets += sheets
        self.source = torch.cat([self.source, (clip.clamp(0, 1) * 255).round().byte()])
        self.history.append({"op": "extend", "words": words, "frames": len(sheets)})
        return self

    def _go_on_anime(self, clip: torch.Tensor, fps: float, words: str, log) -> Animation:
        from .anime import draw_anime, then

        got = draw_anime(clip, fps, palette=self.palette, lines=self.settings.get("lines", "pen"),
                         detail=self.settings.get("detail", 0.6), base=int(self.settings.get("on", 2)),
                         shading=self.settings.get("shading", "frame"), background=self.background,
                         start=(self.sheets[-1], self.frames[-1]), log=log)
        before = len(self.sheets)
        self.holds = (self.holds or [1] * before) + got["holds"]
        last = self.cameras[-1] if self.cameras is not None and len(self.cameras) else torch.eye(2, 3)
        cameras = torch.stack([then(c, last) for c in got["cameras"]])
        self.cameras = cameras if self.cameras is None else torch.cat([self.cameras, cameras])
        self.sheets += got["sheets"]
        self.source = torch.cat([self.source, (clip[got["drawn"]] * 255).round().byte()])
        self.settings["checks"] = self.settings.get("checks", []) + got["checks"]
        self._backdrops = {}
        self.history.append({"op": "extend", "words": words, "frames": len(got["sheets"]), "style": "anime"})
        return self

    def summary(self) -> dict:
        return {"frames": len(self), "fps": self.fps, "markers": len(self.palette),
                "lines": [len(s.lines) for s in self.sheets], "traced": [s.kept for s in self.sheets],
                "new": [s.new for s in self.sheets],
                "drawn_in_full": [i for i, s in enumerate(self.sheets) if s.fresh],
                "words": self.words}

    # -------------------------------------------------------------------------------------------- files --
    @staticmethod
    def _pack(lines: list) -> dict:
        return {"points": torch.cat([line.points for line in lines]) if lines else torch.zeros(0, 2),
                "pressure": torch.cat([line.pressure for line in lines]) if lines else torch.zeros(0),
                "sizes": torch.tensor([len(line) for line in lines], dtype=torch.long),
                "tools": torch.tensor([TOOL_NAMES.index(line.tool) for line in lines], dtype=torch.long),
                "kinds": torch.tensor([KINDS.index(line.kind) for line in lines], dtype=torch.long),
                "markers": torch.tensor([line.marker for line in lines], dtype=torch.long),
                "ages": torch.tensor([line.age for line in lines], dtype=torch.long),
                "colours": torch.tensor([line.colour for line in lines], dtype=torch.float32).reshape(-1, 3),
                "uids": torch.tensor([line.uid for line in lines], dtype=torch.long),
                "ring_counts": torch.tensor([len(line.rings or ()) for line in lines], dtype=torch.long),
                "rings": torch.tensor([n for line in lines for n in (line.rings or ())], dtype=torch.long)}

    @staticmethod
    def _unpack(f: dict) -> list:
        lines, at, ring_at = [], 0, 0
        counts = f["ring_counts"].tolist() if "ring_counts" in f else [0] * len(f["sizes"])
        for i, n in enumerate(f["sizes"].tolist()):
            rings = f["rings"][ring_at : ring_at + counts[i]].tolist() if counts[i] else None
            ring_at += counts[i]
            lines.append(Line(f["points"][at : at + n], f["pressure"][at : at + n], TOOL_NAMES[int(f["tools"][i])],
                              f["colours"][i].tolist(), int(f["markers"][i]), KINDS[int(f["kinds"][i])],
                              int(f["ages"][i]), int(f["uids"][i]) if "uids" in f else None, rings))
            at += n
        return lines

    def _save_state(self, path: Path) -> Path:
        frames = [self._pack(sheet.lines) | {"kept": sheet.kept, "new": sheet.new, "fresh": sheet.fresh}
                  for sheet in self.sheets]
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {"background": self._pack(self.background)} if self.background else {}
        if self.holds:
            extra["holds"] = torch.tensor(self.holds, dtype=torch.long)
        if self.cameras is not None:
            extra["cameras"] = self.cameras.float()
        torch.save(extra | {"version": VERSION, "sheets": frames, "palette": self.palette, "source": self.source,
                    "fps": self.fps, "words": self.words, "surface": self.surface, "history": json.dumps(self.history),
                    "settings": json.dumps({k: v for k, v in self.settings.items() if k != "size"} | {"size": list(
                        self.size)})}, path)
        return path

    @classmethod
    def load(cls, path) -> Animation:
        data = torch.load(Path(path), map_location="cpu", weights_only=True)
        sheets = [Sheet(cls._unpack(f), int(f["kept"]), int(f["new"]), bool(f["fresh"])) for f in data["sheets"]]
        background = cls._unpack(data["background"]) if "background" in data else None
        holds = data["holds"].tolist() if "holds" in data else None
        anim = cls(sheets, data["palette"], None, float(data["fps"]), data["words"], data["surface"],
                   json.loads(data["history"]), json.loads(data["settings"]), background, holds)
        anim.source = data["source"]
        anim.cameras = data.get("cameras")
        return anim


def _fit(clip: torch.Tensor, work: int) -> torch.Tensor:
    if max(clip.shape[-2:]) == work:
        return clip.clamp(0, 1)
    h, w = clip.shape[-2:]
    f = work / max(h, w)
    return F.interpolate(clip, size=(round(h * f), round(w * f)), mode="bilinear", align_corners=False,
                         antialias=True).clamp(0, 1)


def _anime(clip: torch.Tensor, fps: float, words: str, *, work: int, like, colours, lines: str, detail: float,
           depth, shading: str, details: bool, on, surface: str, seed: int, log) -> Animation:
    from .anime import STYLE_WAVER, draw_anime

    clip = _fit(clip, work)
    got = draw_anime(clip, fps, palette=like.palette.clone() if like is not None else None, colours=colours or 16,
                     lines=lines, detail=detail, depth=depth, base=int(on or 2), shading=shading, details=details,
                     log=log)
    settings = {"style": "anime", "lines": lines, "detail": detail, "seed": seed, "on": int(on or 2),
                "waver": STYLE_WAVER, "shading": shading, "checks": got["checks"]}
    anim = Animation(got["sheets"], got["palette"], clip[got["drawn"]], fps, words, surface,
                     [{"op": "animate", "words": words, "frames": len(got["sheets"]), "style": "anime"}], settings,
                     background=got["background"], holds=got["holds"])
    anim.cameras = got["cameras"]
    return anim


def animate(words: str | None = None, *, video=None, animator=None, frames: int | None = None, size=None,
            fps: float | None = None, seed: int = 0, character=None, like: Animation | None = None,
            lines: str = "pen", colour: str | None = "marker", sketch: bool = False, detail: float = 0.5,
            colours: int | None = None, surface: str = "whiteboard", work: int = 512, drawable: bool = True,
            on: int | None = None, motion=None, style: str = "anime", depth="model", shading: str = "frame",
            details: bool = True, log=None) -> Animation:
    """
    An animation drawn by hand, frame by frame. Ask for it in ``words`` (an open video model imagines the
    motion: ``animator``, see :func:`helix_lm.paint.video.load_animator`), or give a ``video`` of your own (a
    file or a folder of pictures). ``character`` is a picture of how the character looks: the clip starts from
    it. ``like`` is an earlier animation whose markers to use, so a character keeps its colours from one clip to
    the next. ``lines`` is the pen (``pen``, ``brush pen``, ``pencil``), ``colour`` None draws lines only,
    ``sketch`` puts a light pencil sketch under the ink. ``work`` is the longer side the frames are drawn at.
    ``on`` is how many video frames each drawing is held for, as animators say "on ones" or "on twos": by
    default twos for a video of 16 frames a second or more (12 drawings a second from 24), else ones.
    ``motion`` is a video of someone moving (a file, a tensor or ``(video, fps)``): with Wan-Animate and a
    ``character`` picture, the character performs that motion.

    ``style`` "anime" (the default) draws as a studio does (:mod:`helix_lm.paint.anime`): the background
    painted once and held, the character inked and painted as a cel on each drawing, timed on ones, twos and
    threes from how fast it moves (``on`` then sets the usual timing, twos by default), faces and small parts
    drawn with care (``details``), every line steadied through time; ``depth`` "model" uses a depth model for
    the space (None goes without), ``shading`` "frame" or "3d". ``style`` "marker" draws every frame whole with
    see-through markers, as the first version did.
    """
    if (words is None) == (video is None):
        raise ValueError("ask for an animation in words, or give a video to draw")
    if video is not None:
        clip, rate = video if isinstance(video, tuple) else read_video(video, frames)
        fps = fps or rate
    else:
        from .video import DRAWABLE, load_animator

        animator = animator or load_animator()
        want = (animator.size if size is None else (size if isinstance(size, tuple) else (size, size)))
        if drawable:
            from .anime import ASK

            words_for_model = f"{words}, {ASK if style == 'anime' else DRAWABLE}"
        else:
            words_for_model = words
        clip = animator(words_for_model, frames=frames,
                        size=(want[0] - want[0] % 8, want[1] - want[1] % 8), seed=seed, character=character,
                        **({"motion": motion} if motion is not None else {}))
        fps = fps or animator.fps
        if log:
            log(f"the video model imagined {len(clip)} frames of {clip.shape[-1]}x{clip.shape[-2]}")
    clip, fps = clip.float(), float(fps or 8.0)
    if style == "anime":
        return _anime(clip, fps, words or str(video), work=work, like=like, colours=colours, lines=lines,
                      detail=detail, depth=depth, shading=shading, details=details, on=on, surface=surface,
                      seed=seed, log=log)
    colours = colours or 10
    on = max(1, int(on or (2 if fps >= 16 else 1)))
    if on > 1 and len(clip) > on:
        clip, fps = clip[::on], fps / on
        if log:
            log(f"drawn on {on}s: {len(clip)} drawings, {fps:g} a second")
    if max(clip.shape[-2:]) != work:
        h, w = clip.shape[-2:]
        f = work / max(h, w)
        clip = F.interpolate(clip, size=(round(h * f), round(w * f)), mode="bilinear", align_corners=False,
                             antialias=True).clamp(0, 1)
    palette = like.palette.clone() if like is not None else markers(list(clip), colours, seed)
    sheets, palette = draw_frames(clip, palette, lines=lines, colour=colour, sketch=sketch, detail=detail, log=log)
    settings = {"lines": lines, "colour": colour, "sketch": sketch, "detail": detail, "seed": seed, "on": on}
    return Animation(sheets, palette, clip, fps, words or str(video), surface,
                     [{"op": "animate", "words": words, "frames": len(sheets)}], settings)


__all__ = ["Animation", "Sheet", "VideoWriter", "animate", "draw_frames", "read_video", "trace_over", "write_video"]
