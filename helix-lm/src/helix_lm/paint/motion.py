# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The principles of animation, as rules that work on drawings over time.

Pinxit draws every frame of a clip over the last on the lightbox (:mod:`helix_lm.paint.animate`), and a line traced
onto the next frame keeps its ``uid``, so its path through the animation can be followed. A video model's frames
melt a little from one to the next, and a line pulled onto them shakes. An animator's drawings do not: they hold
still where nothing moves, move on arcs, ease into and out of their poses, and are timed on ones, twos and threes.
This module does that to the drawings:

- :func:`motion_amount` measures how much the character moves in each frame (from optical flow), and
  :func:`track_motion` the same from the lines themselves;
- :func:`exposure` writes the timesheet: which frames get a drawing and for how many frames each is held, and
  :func:`playback` shows those drawings at any frame rate;
- :func:`keys` finds the key drawings (the extremes) and the breakdowns between them;
- :func:`tracks` follows each line through the frames, and :func:`steady` smooths its path through time;
- :func:`inbetween` and :func:`inbetweens` draw the in-betweens between two key drawings, spaced by a chart
  (:func:`spacing`) and moved on arcs;
- :func:`exaggerate` pushes the poses further, and squashes and stretches along the motion.

The rules, and where they come from:

1. **Timing.** The number of drawings for an action sets its speed, weight and mood (Thomas and Johnston, The
   Illusion of Life, 1981). Film and the anime timesheet run at 24 frames a second; a drawing shown for one frame
   is "on ones", for two "on twos" (12 drawings a second), for three "on threes" (8). Williams (The Animator's
   Survival Kit, 2001, "the great ones and twos battle") does most things on twos and fast action on ones, where
   twos would jump and strobe. Anime is limited animation: twos and threes are the norm, and one cut mixes them,
   slow movement on threes, faster on twos, violent action on ones (Animetudes, "Animation and subjectivity:
   towards a theory of framerate modulation", 2020; Full Frontal, "Animation fundamentals: timing and
   spacing"). Here: ones when the character moves faster than ``fast``, the ``base`` (twos) as the norm, one step
   slower (threes) below ``slow``. A short stretch of slower timing between faster ones takes the faster timing,
   so a quick action is never cut short; timing changes only at a run of frames, as an animator times an action.
2. **Holds.** Where nothing moves, one drawing is held, as anime does all the time. A hold must last long enough
   to be read (``least_hold``); a drawing held too long goes dead, so Williams' "moving hold" draws it again now
   and then with the slight drift of the video (``longest_hold``).
3. **The timesheet** (exposure sheet, dope sheet, X-sheet; anime's taimu shiito): one row for each frame and a
   drawing number where the drawing changes (Wikipedia, "Exposure sheet"). :func:`exposure` gives it as
   ``(frame, frames held)``. The timing is kept in seconds, so the drawings can be shown at any rate the user picks
   (:func:`playback`): on twos at 24 frames a second, a drawing is shown for 2.5 frames on average at 30 and for
   5 at 60, and nothing new is invented between the drawings.
4. **Pose to pose.** Keys first, then breakdowns, then in-betweens (Thomas and Johnston; Williams' keys,
   extremes, breakdowns and in-betweens; anime's genga, the key animation, and douga, the in-betweening done from
   the keys). The keys are the extremes, where the motion slows to a stop or turns sharply; a breakdown is the
   drawing between two keys that says how the move gets from one to the other, the passing position on its arc
   (Williams). :func:`keys` finds both.
5. **Slow in and slow out.** More drawings near the start and end of an action and fewer in the middle (Thomas
   and Johnston). The animator's timing chart (spacing chart) between two keys shows where each in-between goes:
   halves, thirds, and in-betweens bunched toward a key to slow into it or out of it (Williams; Brian Lemay's
   notes on timing charts). :func:`spacing` gives the chart. ``"in"`` is a slow in to the second key and
   ``"out"`` a slow out of the first, as animators say (not as CSS says); ``"in-out"`` is both.
6. **Arcs.** Natural motion moves on arcs, and limbs turn round joints (Thomas and Johnston), so an in-between is
   not the straight average of two keys. It goes on a curve through the breakdown, or turns round a pivot: given,
   or found from the two keys as the point that the turn best laying one on the other leaves still (Kabsch,
   1976). What the turn does not explain is carried straight.
7. **Straight ahead, kept steady.** Each frame traced over the last keeps the drawing alive but makes it boil and
   wobble. :func:`steady` smooths each line's path through time with a discrete smoothing spline (Whittaker,
   1923; Eilers, "A perfect smoother", 2003) that penalises third differences: a path with constant acceleration,
   such as a thrown ball's parabola or a steady ease, comes through unchanged, so arcs and ease are kept and the
   shake goes. The window is chosen for each line by generalised cross-validation (Craven and Wahba, 1979, with
   the weight of 1.4 on the fit's complexity of Kim and Gu, 2004): long where the path is simple, short where it
   changes quickly, never longer than the line lives. Key drawings are pinned and the path is smoothed apart on
   each side of them, so a sharp turn at a key (a bounce, a hit) stays sharp.
8. **Squash and stretch, and exaggeration.** A shape stretches along a fast motion and squashes when it stops or
   turns sharply, its volume (on paper, its area) kept, and poses go a little further than life (Thomas and
   Johnston, as Wikipedia's "Twelve basic principles of animation" sums them up). :func:`exaggerate` does both;
   both are off unless asked for.

Not done here: anticipation, follow through and overlapping action, secondary action, staging, solid drawing and
appeal belong to the motion and the drawing themselves (the video model and :mod:`helix_lm.paint.draw`). These
rules keep them (keys pinned, nothing smoothed beyond its window) but do not add them. Smears (Williams' stretched
in-betweens on fast action, one or two frames long) are not drawn.

Points are in board widths, as in :class:`helix_lm.paint.draw.Line`; motion is in pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import torch
import torch.nn.functional as F

from .draw import Line

RATE = 24.0                    # frames a second of film, and of the anime timesheet
EASES = ("linear", "in", "out", "in-out")
MOST = 256                     # the most points a strand of a line is resampled to


# -------------------------------------------------------------------------------------------- helpers --
def _runs(frames: list) -> list:
    """The positions in ``frames`` (increasing frame numbers) split into runs of consecutive frames."""
    out: list = []
    for i, f in enumerate(frames):
        if out and f == frames[out[-1][-1]] + 1:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _true_runs(mask: torch.Tensor) -> list:
    """``(start, end)`` of every run of True in ``mask (T,)``, ``end`` past the last."""
    out, start = [], None
    for t, on in enumerate(mask.tolist()):
        if on and start is None:
            start = t
        elif not on and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _robust(values: torch.Tensor, noise: float, share: float = 0.02) -> float:
    """How far the moving part moves, from the distances ``values`` that pixels or points moved: the median of
    those that move at least half as far as the fastest ``share`` of them (and more than ``noise``), so a still
    background does not hide a moving character and one wild pixel does not count."""
    v = values.flatten().float()
    if v.numel() == 0:
        return 0.0
    if v.numel() > 1_000_000:
        v = v[:: v.numel() // 1_000_000 + 1]
    top = float(torch.quantile(v, 1.0 - share))
    moving = v[v >= max(noise, 0.5 * top)]
    return float((moving if moving.numel() else v).median())


def _per_frame(steps: torch.Tensor) -> torch.Tensor:
    """The moves between neighbouring frames ``(T - 1,)`` as the speed at each frame ``(T,)``: the mean of the
    move into it and the move out of it."""
    if len(steps) == 0:
        return steps
    pad = torch.cat([steps[:1], steps, steps[-1:]])
    return 0.5 * (pad[:-1] + pad[1:])


def _median3(x: torch.Tensor) -> torch.Tensor:
    if len(x) < 3:
        return x
    pad = torch.cat([x[:1], x, x[-1:]])
    return torch.stack([pad[:-2], pad[1:-1], pad[2:]]).median(0).values


def _opening(n: torch.Tensor, width: int) -> torch.Tensor:
    """``n (T,)`` with every run of larger values shorter than ``width`` lowered to the values round it (the
    smallest of a window, then the largest: a morphological opening); runs of smaller values keep their length."""
    if width <= 1 or len(n) < 2:
        return n
    x = n.float()[None, None]
    a, b = width // 2, (width - 1) // 2
    low = -F.max_pool1d(-F.pad(x, (a, b), mode="replicate"), width, 1)
    return F.max_pool1d(F.pad(low, (b, a), mode="replicate"), width, 1)[0, 0].round().long()


def _interp(x: torch.Tensor, y: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """``y (m, C)`` known at increasing ``x (m,)``, linearly at ``q (n,)``."""
    if len(x) == 1:
        return y[:1].expand(len(q), -1).clone()
    i = torch.searchsorted(x.contiguous(), q.contiguous(), right=True).clamp(1, len(x) - 1)
    x0, x1 = x[i - 1], x[i]
    w = ((q - x0) / (x1 - x0).clamp(min=1e-12)).clamp(0.0, 1.0)[:, None]
    return y[i - 1] + (y[i] - y[i - 1]) * w


def _grid(n: int, closed: bool) -> torch.Tensor:
    return torch.arange(n, dtype=torch.float32) / (n if closed else max(1, n - 1))


def _params(pts: torch.Tensor, closed: bool, by_index: bool) -> torch.Tensor:
    """Where each point of a strand lies along it, 0 to 1 (a closed ring: 0 up to, not including, 1): by its
    number, or by the length along it."""
    k = len(pts)
    if not by_index and k >= 2:
        steps = (torch.roll(pts, -1, 0) - pts).norm(dim=-1) if closed else (pts[1:] - pts[:-1]).norm(dim=-1)
        s = torch.cat([steps.new_zeros(1), steps.cumsum(0)])
        if float(s[-1]) > 1e-9:
            return s[:k] / s[-1]
    return _grid(k, closed)


def _resample(values: torch.Tensor, u: torch.Tensor, n: int, closed: bool) -> torch.Tensor:
    """``values (k, C)`` at ``u (k,)`` along a strand, at ``n`` points evenly along it."""
    if closed:
        u, values = torch.cat([u, u.new_ones(1)]), torch.cat([values, values[:1]])
    return _interp(u, values, _grid(n, closed))


def _unsample(on_grid: torch.Tensor, u: torch.Tensor, closed: bool) -> torch.Tensor:
    """Values at ``n`` points evenly along a strand back at the strand's own points ``u (k,)``."""
    g = _grid(len(on_grid), closed)
    if closed:
        g, on_grid = torch.cat([g, g.new_ones(1)]), torch.cat([on_grid, on_grid[:1]])
    return _interp(g, on_grid, u)


def _align(ref: torch.Tensor, cur: torch.Tensor, closed: bool) -> tuple[bool, int]:
    """How to lay strand ``cur (n, 2)`` on ``ref (n, 2)``: whether to run it backwards (a line may be drawn from
    either end) and, for a ring, by how many points to turn it (a ring may start anywhere)."""
    if not closed:
        return float((cur.flip(0) - ref).pow(2).sum()) < float((cur - ref).pow(2).sum()), 0
    n = len(cur)
    idx = (torch.arange(n)[None, :] + torch.arange(n)[:, None]) % n        # idx[s, j] = (j + s) % n
    best = (float("inf"), False, 0)
    for flip in (False, True):
        c = cur.flip(0) if flip else cur
        cost = (c[idx] - ref[None]).pow(2).sum((-1, -2))
        s = int(cost.argmin())
        if float(cost[s]) < best[0] - 1e-12:
            best = (float(cost[s]), flip, s)
    return best[1], best[2]


def _lay(values: torch.Tensor, flip: bool, shift: int, closed: bool) -> torch.Tensor:
    v = values.flip(0) if flip else values
    return torch.roll(v, -shift, 0) if closed and shift else v


def _unlay(values: torch.Tensor, flip: bool, shift: int, closed: bool) -> torch.Tensor:
    v = torch.roll(values, shift, 0) if closed and shift else values
    return v.flip(0) if flip else v


def _strands(line: Line, pressure: bool = False) -> list:
    """A line as its strands: ``(values (k, 2), closed)`` for each ring of a cel shape, or one open strand; with
    ``pressure``, the values are x, y and the pen's pressure ``(k, 3)``."""
    vals = line.points.float()
    if pressure:
        p = line.pressure.float().flatten()
        if p.numel() != len(vals):
            p = torch.full((len(vals),), float(p.mean()) if p.numel() else 0.5)
        vals = torch.cat([vals, p[:, None]], 1)
    if line.rings is None:
        return [(vals, False)]
    out, at = [], 0
    for n in line.rings:
        out.append((vals[at : at + n], True))
        at += n
    return out


def _shape(line: Line) -> tuple:
    return ("open", 1) if line.rings is None else ("rings", len(line.rings))


def _lam(width: float, order: int = 3) -> float:
    """The smoothing weight that makes the smoother of ``order`` pass a wave of ``width`` frames at half its
    power: shakes faster than that go, slower motion stays."""
    if width <= 2.0:
        return 0.0
    return (2.0 * math.sin(math.pi / width)) ** (-2 * order)


def _whittaker(y: torch.Tensor, lam: float, pins=(), weights: torch.Tensor | None = None,
               order: int = 3) -> torch.Tensor:
    """``y (L, ...)`` smoothed along its first axis (Whittaker 1923, Eilers 2003): the ``x`` nearest ``y`` whose
    ``order``-th differences are small, ``sum w (x - y)^2 + lam * sum (D^order x)^2`` least, with the frames
    ``pins`` held exactly. With ``order`` 3 a path of constant acceleration passes unchanged."""
    n = y.shape[0]
    if n <= order or lam <= 0:
        return y.clone()
    flat = y.reshape(n, -1).double()
    d = torch.diff(torch.eye(n, dtype=torch.float64), n=order, dim=0)
    penalty = lam * d.T @ d
    w = torch.ones(n, dtype=torch.float64) if weights is None else weights.double()
    fixed = sorted({int(p) for p in pins if 0 <= p < n})
    free = [i for i in range(n) if i not in set(fixed)]
    if not free:
        return y.clone()
    a = penalty[free][:, free] + torch.diag(w[free] + 1e-9)
    rhs = w[free, None] * flat[free]
    if fixed:
        rhs = rhs - penalty[free][:, fixed] @ flat[fixed]
    out = flat.clone()
    out[free] = torch.linalg.solve(a, rhs)
    return out.reshape(y.shape).to(y.dtype)


def _carry(points: torch.Tensor, src: torch.Tensor, disp: torch.Tensor, k: int = 6) -> torch.Tensor:
    """How far ``points (P, 2)`` move, from how far the nearest of ``src (M, 2)`` moved (``disp (M, 2)``),
    weighted by nearness: so a line that has no partner is moved with the drawing round it."""
    if len(src) == 0 or len(points) == 0:
        return torch.zeros_like(points)
    if len(src) > 4000:
        step = len(src) // 4000 + 1
        src, disp = src[::step], disp[::step]
    k = min(k, len(src))
    out = []
    for chunk in torch.split(points, 2048):
        dist, idx = torch.cdist(chunk, src).topk(k, largest=False)
        w = 1.0 / (dist + 1e-4) ** 2
        w = w / w.sum(1, keepdim=True)
        out.append((w[..., None] * disp[idx]).sum(1))
    return torch.cat(out)


def _by_uid(lines: list) -> dict:
    out: dict = {}
    for line in lines:
        out.setdefault(line.uid, line)
    return out


# ---------------------------------------------------------------------------------------------- motion --
def motion_amount(flows, masks=None, *, between: bool = True, noise: float = 0.1,
                  share: float = 0.02) -> torch.Tensor:
    """How much the character moves in each frame, in pixels a frame. ``flows (N, 2, H, W)`` (or a list) are
    optical flows (:func:`helix_lm.paint.flow.optical_flow`, :func:`helix_lm.paint.space.clip_flows`), entry
    ``t`` from frame t to frame t + 1. ``masks`` are where the character is: one ``(H, W)`` for all, or one for
    each flow (or each frame; the first of each pair is used). For each flow, the median of how far the moving
    pixels went: those that move at least half as far as the fastest ``share`` of them, so a still background does
    not hide a moving character, and a hand that moves while the body is still still counts (it needs drawings).

    With ``between`` (the default), the ``N`` moves between frames are turned into the speed at each of the
    ``N + 1`` frames (the mean of the move into it and out of it); without, one number for each flow."""
    steps = []
    for i, flow in enumerate(flows):
        mag = flow.float().norm(dim=0)
        mask = None
        if masks is not None:
            mask = masks if torch.is_tensor(masks) and masks.dim() == 2 else masks[i]
            mask = mask.bool()
        values = mag[mask] if mask is not None and int(mask.sum()) >= 4 else mag
        steps.append(_robust(values, noise, share))
    out = torch.tensor(steps, dtype=torch.float32)
    return _per_frame(out) if between else out


def _track_steps(trs: dict, frames: int, scale: float, noise: float):
    """From tracks: how far points moved into each frame (a list of tensors for each frame), and how sharply the
    lines turn at each frame (degrees, weighted by speed)."""
    moves: list = [[] for _ in range(frames)]
    turn_sum, turn_weight = torch.zeros(frames, dtype=torch.float64), torch.zeros(frames, dtype=torch.float64)
    for tr in trs.values():
        for run in _runs(tr.frames):
            if len(run) < 2:
                continue
            q = tr.points[run] * scale
            first = tr.frames[run[0]]
            v = q[1:] - q[:-1]                                   # (L - 1, N, 2): into frames first + 1 ...
            size = v.norm(dim=-1)
            for i in range(len(v)):
                if first + 1 + i < frames:
                    moves[first + 1 + i].append(size[i])
            if len(v) >= 2:
                a, b = v[:-1], v[1:]
                na, nb = size[:-1], size[1:]
                cos = ((a * b).sum(-1) / (na * nb).clamp(min=1e-9)).clamp(-1.0, 1.0)
                angle = torch.rad2deg(torch.acos(cos))
                w = torch.minimum(na, nb) * ((na > noise) & (nb > noise))
                for i in range(len(a)):
                    t = first + 1 + i
                    if t < frames:
                        turn_sum[t] += float((w[i] * angle[i]).sum())
                        turn_weight[t] += float(w[i].sum())
    turning = (turn_sum / turn_weight.clamp(min=1e-9)).float() * (turn_weight > 0)
    return moves, turning


def track_motion(trs: dict, frames: int | None = None, *, scale: float = 512.0, noise: float = 0.5) -> torch.Tensor:
    """How much the drawing moves in each frame, in pixels of a picture ``scale`` pixels wide, from the lines'
    tracks (:func:`tracks`): like :func:`motion_amount`, for when there is no flow at hand. ``frames`` is how many
    frames there are (by default one past the last frame of any track)."""
    if frames is None:
        frames = 1 + max((max(tr.frames) for tr in trs.values()), default=-1)
    if frames <= 0:
        return torch.zeros(0)
    moves, _ = _track_steps(trs, frames, scale, noise)
    steps = torch.tensor([_robust(torch.cat(m), noise) if m else 0.0 for m in moves[1:]], dtype=torch.float32)
    return _per_frame(steps) if frames > 1 else torch.zeros(1)


# ------------------------------------------------------------------------------------------- timesheet --
def exposure(motion, fps: float, base: int = 2, *, rate: float = RATE, fast: float = 6.0, slow: float = 1.0,
             still: float = 0.25, size: float = 512.0, least_run: int = 4, least_hold: int = 6,
             longest_hold: int | None = None, keys=None) -> list:
    """
    The timesheet: which frames of a clip get a drawing and for how many frames each is held, as a list of
    ``(frame_index, frames_held)`` (frames of the clip, at ``fps``) that covers the clip from start to end.

    ``motion (T,)`` is how much the character moves at each frame, in pixels a frame (:func:`motion_amount`),
    measured on frames whose longer side is ``size`` pixels. Timing is worked out on the anime timesheet,
    ``rate`` frames a second (24), whatever the clip's own ``fps``: with the speed in pixels a timesheet frame
    (for a picture 512 pixels across),

    - above ``fast``: on ones (fast action: twos would jump);
    - from ``slow`` to ``fast``: on ``base`` (2, twos, as anime and most hand-drawn animation; 1 for full
      animation, 3 for the limited look of TV anime);
    - below ``slow``: one step slower, ``base + 1`` (threes for slow movement);
    - at most ``still``, for at least ``least_hold`` timesheet frames: a hold, one drawing until it moves again,
      drawn again every ``longest_hold`` timesheet frames if given (a moving hold); a stillness too short to be
      read as a hold is timed as slow movement.

    A run of slower timing shorter than ``least_run`` timesheet frames takes the faster timing round it. Frames in
    ``keys`` always start a drawing. A drawing on twos is held 2 frames of a 24 frame a second clip, 2.5 on average
    of a 30 (2, 3, 2, 3), 1 of a 12; a clip slower than the timing asks for gets a drawing on every frame (to
    go finer, in-between it first, :func:`inbetweens`).
    """
    m = torch.as_tensor(motion, dtype=torch.float32).flatten()
    frames = len(m)
    if frames == 0:
        return []
    per = float(fps) / float(rate)                       # frames of the clip in one timesheet frame
    speed = _median3(m * per * 512.0 / float(size))
    base = max(1, int(base))
    n = torch.full((frames,), base, dtype=torch.long)
    n[speed < slow] = base + 1
    n[speed > fast] = 1
    n = _opening(n, max(1, round(least_run * per)))
    hold = speed <= still
    for start, end in _true_runs(hold):
        if end - start < least_hold * per:
            hold[start:end] = False
    kind = torch.where(hold, torch.zeros_like(n), n).tolist()
    bounds = {int(k) for k in (keys or ()) if 0 <= int(k) < frames}
    out, start = [], 0
    while start < frames:
        end = start + 1
        while end < frames and kind[end] == kind[start] and end not in bounds:
            end += 1
        if kind[start] == 0:
            step = float(end - start) if not longest_hold else max(1.0, longest_hold * per)
        else:
            step = max(1.0, kind[start] * per)
        # as many drawings as fit, spread evenly over the run (at a rate that is not a whole number of frames,
        # such as twos at 30 a second, they alternate 2, 3, 2, 3)
        count = max(1, int(math.floor((end - start) / step + 0.5)))
        marks = [start + int(math.floor(i * (end - start) / count + 0.5)) for i in range(count)]
        out += [(i, after - i) for i, after in zip(marks, marks[1:] + [end], strict=True)]
        start = end
    return out


def playback(exposures: list, fps: float, out_fps: float) -> list:
    """The drawings of a timesheet (:func:`exposure`, at the clip's ``fps``) shown at ``out_fps``: for each
    frame of the output, the number of the drawing on screen (a position in ``exposures``). The timing in seconds is
    kept and nothing is drawn between the drawings, so the animation keeps its look at any rate:
    ``[drawings[d] for d in playback(exposures, fps, out_fps)]`` is the animation at ``out_fps``."""
    starts, at = [], 0
    for _, held in exposures:
        starts.append(at)
        at += int(held)
    if at == 0:
        return []
    count = max(1, round(at * float(out_fps) / float(fps)))
    out, d = [], 0
    for k in range(count):
        t = k * float(fps) / float(out_fps)
        while d + 1 < len(starts) and starts[d + 1] <= t + 1e-6:
            d += 1
        out.append(d)
    return out


# ------------------------------------------------------------------------------------------------ keys --
def keys(motion, *, drop: float = 0.6, turn: float = 60.0, least: int = 3, reach: int = 8, still: float | None = None,
         noise: float = 0.5, scale: float = 512.0) -> tuple[list, list]:
    """
    The key drawings and the breakdowns between them: ``(keys, breakdowns)``, frame numbers in order.

    ``motion`` is either the speed at each frame (pixels a frame, :func:`motion_amount`) or the lines' tracks
    (:func:`tracks`, in a picture ``scale`` pixels wide), which also tell which way things move. Keys are the
    extremes: the first and last frames; where the motion slows to a stop (the speed falls to ``drop`` of the
    fastest within ``reach`` frames on both sides); the first and last frame of a stillness (at most ``still``
    pixels a frame, ``noise`` by default); and, with tracks, where the lines turn by more than ``turn`` degrees
    from one frame to the next (a bounce, a hit, the end of a swing). Keys are at least ``least`` frames apart;
    the stronger extreme is kept.

    Between two keys at least 3 frames apart the breakdown is the frame that shows the arc best: with tracks, the
    one that lies furthest from the straight line between the keys, else (and when the move is straight) the
    passing position, halfway along the distance travelled (Williams).
    """
    trs = motion if isinstance(motion, dict) else None
    if trs is not None:
        frames = 1 + max((max(tr.frames) for tr in trs.values()), default=-1)
        speed = track_motion(trs, frames, scale=scale, noise=noise)
        _, turning = _track_steps(trs, frames, scale, noise)
    else:
        speed = torch.as_tensor(motion, dtype=torch.float32).flatten()
        frames, turning = len(speed), None
    if frames == 0:
        return [], []
    if frames <= 2:
        return list(range(frames)), []
    still = noise if still is None else still
    m = speed.clone()
    m[1:-1] = (speed[:-2] + speed[1:-1] + speed[2:]) / 3.0
    score = {0: -2.0, frames - 1: -2.0}                     # lower is stronger

    def put(t: int, s: float) -> None:
        score[t] = min(score.get(t, s), s)

    for start, end in _true_runs(speed <= still):
        put(start, -1.0)
        if end - 1 - start >= least:
            put(end - 1, -1.0)
    for t in range(1, frames - 1):
        if speed[t] <= still or m[t] > m[t - 1] or m[t] > m[t + 1]:
            continue
        left = float(m[max(0, t - reach) : t].max())
        right = float(m[t + 1 : t + 1 + reach].max())
        peak = min(left, right)
        if peak > 0 and float(m[t]) <= drop * peak:
            put(t, float(m[t]) / peak)
    if turning is not None:
        for t in range(1, frames - 1):
            a = float(turning[t])
            if a >= turn and a >= float(turning[t - 1]) and a >= float(turning[t + 1]):
                put(t, -a / 180.0)
    chosen: list = []
    for t in sorted(score, key=lambda t: (score[t], t)):      # the two ends first, then the strongest
        if t in (0, frames - 1) or all(abs(t - c) >= least for c in chosen):
            chosen.append(t)
    chosen = sorted(set(chosen))
    breakdowns = []
    for k0, k1 in zip(chosen, chosen[1:], strict=False):
        if k1 - k0 < 3:
            continue
        t = _furthest(trs, k0, k1, scale, noise) if trs is not None else None
        if t is None:
            travel = speed[k0 + 1 : k1 + 1].double().cumsum(0)
            half = float(travel[-1]) * (0.5 - 1e-9)
            t = k0 + 1 + int((travel < half).sum()) if float(travel[-1]) > 0 else (k0 + k1) // 2
        breakdowns.append(min(max(t, k0 + 1), k1 - 1))
    return chosen, breakdowns


_find_keys = keys                                         # for steady(), whose argument is called keys


def _furthest(trs: dict, k0: int, k1: int, scale: float, noise: float) -> int | None:
    """The frame between keys ``k0`` and ``k1`` whose lines lie furthest from the straight line each point takes
    from one key to the other, or None when all of them move straight."""
    off = torch.zeros(k1 - k0 + 1, dtype=torch.float64)
    seen = torch.zeros(k1 - k0 + 1, dtype=torch.float64)
    for tr in trs.values():
        where = {f: i for i, f in enumerate(tr.frames)}
        if any(f not in where for f in range(k0, k1 + 1)):
            continue
        q = tr.points[[where[f] for f in range(k0, k1 + 1)]] * scale        # (L, N, 2)
        a, b = q[0], q[-1]
        chord = b - a
        length = chord.norm(dim=-1).clamp(min=1e-9)
        rel = q - a
        across = (rel[..., 0] * chord[:, 1] - rel[..., 1] * chord[:, 0]).abs() / length
        along_ok = length > noise
        if not bool(along_ok.any()):
            continue
        off += across[:, along_ok].double().mean(-1)
        seen += 1
    if float(seen.max()) == 0:
        return None
    off = off / seen.clamp(min=1)
    t = int(off[1:-1].argmax()) + 1
    return k0 + t if float(off[t]) > noise else None


# ---------------------------------------------------------------------------------------------- tracks --
@dataclass
class Track:
    """One line followed through the frames: the ``frames`` (sheet numbers, in order) it is in, and its points in
    each, resampled to the same number of points and laid the same way round, ``points (F, N, 2)`` in board widths.
    ``sizes`` is the number of points of each strand (one for an open line, one for each ring of a cel shape) and
    ``rings`` the same for a cel shape (None for an open line)."""

    uid: int
    kind: str
    frames: list
    points: torch.Tensor
    sizes: tuple
    rings: tuple | None
    where: list = field(default_factory=list, repr=False)     # the line's position in each frame's sheet
    maps: list = field(default_factory=list, repr=False)      # each frame, each strand: (u, closed, flip, shift)


def tracks(sheets: list, count: int | None = None, least: int = 2) -> dict:
    """
    Every line followed through the animation by its ``uid``: ``{uid: Track}`` for each line in at least
    ``least`` frames. Its points in each frame are resampled to a fixed number (``count``, or the most it has in
    any frame, at most 256): by their number when the line has as many points in every frame (a line traced
    with the optical flow keeps its points, so each point follows the same bit of the drawing), else evenly along
    its length. A line drawn from the other end in some frame is turned round, and a cel shape's rings are each
    resampled and turned to start where they did in the frame before. A frame where a cel shape has another number
    of rings than in its first frame (or a line turns into a shape) is left out of its track.
    """
    found: dict = {}
    for f, sheet in enumerate(sheets):
        here: set = set()
        for pos, line in enumerate(sheet.lines):
            if line.uid in here or len(line) < 2:
                continue
            here.add(line.uid)
            found.setdefault(line.uid, []).append((f, pos, line))
    out = {}
    for uid, items in found.items():
        shape = _shape(items[0][2])
        items = [it for it in items if _shape(it[2]) == shape]
        if len(items) < max(1, least):
            continue
        strands = [_strands(it[2]) for it in items]
        closed = shape[0] == "rings"
        sizes, by_index = [], []
        for j in range(len(strands[0])):
            ks = [len(s[j][0]) for s in strands]
            n = int(count) if count else max(ks)
            sizes.append(min(MOST, max(3 if closed else 2, n)))
            by_index.append(min(ks) == max(ks))
        frames_pts, maps, before = [], [], None
        for s in strands:
            parts, m = [], []
            for j, (p, _) in enumerate(s):
                u = _params(p, closed, by_index[j])
                q = _resample(p, u, sizes[j], closed)
                flip, shift = _align(before[j], q, closed) if before is not None else (False, 0)
                parts.append(_lay(q, flip, shift, closed))
                m.append((u, closed, flip, shift))
            before = parts
            frames_pts.append(torch.cat(parts))
            maps.append(m)
        out[uid] = Track(uid, items[0][2].kind, [it[0] for it in items], torch.stack(frames_pts), tuple(sizes),
                         tuple(sizes) if closed else None, [it[1] for it in items], maps)
    return out


def _moved(line: Line, delta: torch.Tensor, maps: list, sizes: tuple) -> Line:
    """``line`` with its track's points moved by ``delta (N, 2)``, the move carried back to the line's own points
    (so the line keeps its own points and detail, and only the slow change is added)."""
    parts, at = [], 0
    for (p, _), (u, closed, flip, shift), n in zip(_strands(line), maps, sizes, strict=True):
        d = _unlay(delta[at : at + n], flip, shift, closed)
        at += n
        parts.append(p + _unsample(d, u, closed))
    return line.moved(torch.cat(parts), age=line.age)


# ---------------------------------------------------------------------------------------------- steady --
WIDTHS = (3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 32.0, 48.0, 64.0)


def _best_width(y: torch.Tensor, pins: list, longest: float, gamma: float = 1.4) -> float:
    """The window for smoothing ``y (L, ...)`` chosen by generalised cross-validation (Craven and Wahba, 1979)
    with the fit's complexity counted ``gamma`` times, which keeps it from smoothing too little (Kim and Gu,
    2004): long where the path is simple, short where it changes quickly."""
    n = y.shape[0]
    flat = y.reshape(n, -1).double()
    fixed = sorted({p for p in pins if 0 <= p < n})
    free = [i for i in range(n) if i not in set(fixed)]
    d = torch.diff(torch.eye(n, dtype=torch.float64), n=3, dim=0)
    dd = d.T @ d
    best, best_score = WIDTHS[0], float("inf")
    for width in WIDTHS:
        if width > min(longest, n):
            break
        lam = _lam(width)
        a = torch.eye(len(free), dtype=torch.float64) + lam * dd[free][:, free]
        hat = torch.linalg.inv(a)
        rhs = flat[free] - (lam * dd[free][:, fixed] @ flat[fixed] if fixed else 0.0)
        rss = float((hat @ rhs - flat[free]).pow(2).sum()) / flat.shape[1]
        score = len(free) * rss / max(len(free) - gamma * float(hat.trace()), 0.5) ** 2
        if score < best_score:
            best, best_score = width, score
    return best


def _smooth(ys: torch.Tensor, strength: float, window: float | None, longest: float, pins: list,
            robust: bool) -> torch.Tensor:
    """A track's run ``ys (L, N, 2)`` smoothed through time, pinned frames held, smoothed apart on each side of
    them, each stretch with a window of its own no longer than itself."""
    n = len(ys)
    out = ys.clone()
    cuts = sorted({p for p in pins if 0 <= p < n})
    bounds = [0] + [p for p in cuts if 0 < p < n - 1] + [n - 1]
    for s, e in zip(bounds, bounds[1:], strict=False):
        seg = ys[s : e + 1]
        if len(seg) < 4:
            continue
        local = [p - s for p in cuts if s <= p <= e]
        width = float(window) if window else _best_width(seg, local, longest)
        lam = _lam(min(width * strength, len(seg)))
        if lam <= 0:
            continue
        x = _whittaker(seg, lam, local)
        if robust:                                           # a frame far off the path (a melt) counts less
            for _ in range(2):
                r = (seg - x).norm(dim=-1).mean(-1)
                typical = r.median().clamp(min=1e-9)
                w = (1.0 - (r / (3.0 * typical)).clamp(max=1.0) ** 2) ** 2 + 1e-3
                x = _whittaker(seg, lam, local, w)
        out[s : e + 1] = x
    if cuts:
        out[cuts] = ys[cuts]
    return out


def steady(sheets: list, strength: float = 1.0, keys=None, *, window: float | None = None, longest: float = 24.0,
           robust: bool = False, count: int | None = None) -> list:
    """
    The sheets with the boil, jitter and wobble taken out of every traced line's path through time, arcs and ease
    kept: a new list of sheets (the lines given are not changed).

    Each line followed through the frames by its uid (:func:`tracks`) has the path of each of its points smoothed
    through time by a smoothing spline that penalises third differences (see the module's notes), so motion with
    constant acceleration (a parabola, a steady ease) comes through as it was. Shakes faster than the window (in
    frames) go. The window is chosen for each line and each stretch of its life by cross-validation (long, up to
    ``longest`` frames, where its path is simple; short where it changes quickly, so a quick swing is kept), or is
    ``window`` when given; ``strength`` scales it. It is never longer than the stretch of frames the line lives
    through, and a line in fewer than 4 frames running is left as it is. A line new in a frame and not traced on
    is untouched.

    ``keys`` (frame numbers) are pinned exactly, and the path is smoothed apart on each side of each key, so a sharp
    turn at a key stays sharp; ``"auto"`` uses :func:`keys` on the tracks (those frames keep their own shake).
    With ``robust``, a frame whose line lies far off the smooth path (a melt of the video for a frame) counts less;
    leave it off when sharp turns are not keys, as it would round them further. The change is worked out on the
    resampled points and carried back to each line's own points, so a line keeps its points, its pressure and its
    detail.
    """
    trs = tracks(sheets, count)
    out = [list(s.lines) for s in sheets]
    if strength > 0 and trs:
        if isinstance(keys, str):
            if keys != "auto":
                raise ValueError(f"keys are frame numbers or 'auto', not {keys!r}")
            pins_all = set(_find_keys(trs)[0])
        else:
            pins_all = {int(k) for k in (keys or ())}
        for tr in trs.values():
            for run in _runs(tr.frames):
                if len(run) < 4:
                    continue
                ys = tr.points[run]
                pins = [i for i, idx in enumerate(run) if tr.frames[idx] in pins_all]
                delta = _smooth(ys, strength, window, longest, pins, robust) - ys
                for i, idx in enumerate(run):
                    if i in pins:
                        continue
                    f, pos = tr.frames[idx], tr.where[idx]
                    out[f][pos] = _moved(sheets[f].lines[pos], delta[i], tr.maps[idx], tr.sizes)
    return [replace(s, lines=lines) for s, lines in zip(sheets, out, strict=True)]


# ------------------------------------------------------------------------------------------ in-betweens --
def _ease(ease):
    """The spacing curve: a function from time between two keys (0 to 1) to how far along the move (0 to 1)."""
    if callable(ease):
        return ease
    power = 2.0
    if isinstance(ease, tuple):
        ease, power = ease
    name = str(ease).lower().replace("_", "-").replace(" ", "-")
    p = float(power)
    if name == "linear":
        return lambda t: t
    if name == "in":                                      # slow in to the second key
        return lambda t: 1.0 - (1.0 - t) ** p
    if name == "out":                                     # slow out of the first key
        return lambda t: t ** p
    if name == "in-out":
        return lambda t: t ** p / (t ** p + (1.0 - t) ** p) if 0.0 < t < 1.0 else float(t >= 1.0)
    raise ValueError(f"ease is one of {EASES} (or (name, power), or a function), not {ease!r}")


def spacing(count: int, ease="in-out") -> list:
    """The spacing chart: where each of ``count`` in-betweens, evenly timed between two keys, lies along the move
    from the first key (0) to the second (1). ``"linear"`` spaces them evenly (halves, thirds); ``"in"`` bunches
    them toward the second key (a slow in), ``"out"`` toward the first (a slow out), ``"in-out"`` toward both (Thomas
    and Johnston's slow in and slow out). ``("in-out", 3.0)`` eases harder; a function of time may be given."""
    f = _ease(ease)
    return [float(f(i / (count + 1))) for i in range(1, count + 1)]


def _onto(ref: torch.Tensor, sizes: tuple, closed: bool, line: Line) -> torch.Tensor | None:
    """``line``'s strands (x, y, pressure) resampled to ``sizes`` and laid the way ``ref`` runs, or None when the
    two cannot be paired (another number of rings, a line against a shape)."""
    strands = _strands(line, pressure=True)
    if len(strands) != len(sizes) or (line.rings is not None) != closed:
        return None
    out, at = [], 0
    for (v, _), n in zip(strands, sizes, strict=True):
        r = ref[at : at + n]
        at += n
        if len(v) < 2:
            return None
        if len(v) != n:
            v = _resample(v, _params(v[:, :2], closed, False), n, closed)
        flip, shift = _align(r[:, :2], v[:, :2], closed)
        out.append(_lay(v, flip, shift, closed))
    return torch.cat(out)


def _pair(a: Line, b: Line):
    """Two drawings of one line laid on one another: ``(A, B, sizes, closed)`` with ``A`` and ``B`` ``(N, 3)``
    (x, y, pressure) point for point, or None. Lines with as many points keep them (a traced line's points
    follow the drawing); else both are resampled evenly along their length to the larger number."""
    sa, sb = _strands(a, pressure=True), _strands(b, pressure=True)
    closed = a.rings is not None
    if len(sa) != len(sb) or (b.rings is not None) != closed:
        return None
    parts, sizes = [], []
    for (va, _), (vb, _) in zip(sa, sb, strict=True):
        if len(va) < 2 or len(vb) < 2:
            return None
        n = min(MOST, max(len(va), len(vb))) if len(va) != len(vb) else len(va)
        if len(va) != n:
            va = _resample(va, _params(va[:, :2], closed, False), n, closed)
        parts.append(va)
        sizes.append(n)
    big_a = torch.cat(parts)
    big_b = _onto(big_a, tuple(sizes), closed, b)
    return None if big_b is None else (big_a, big_b, tuple(sizes), closed)


def _rot(v: torch.Tensor, theta: float) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def _pivot(a: torch.Tensor, b: torch.Tensor, centre=None):
    """The turn from points ``a (N, 2)`` to ``b``: ``(pivot (2,), angle)``, the pivot given or the point the best
    rigid turn leaves still (Kabsch), or None when there is no turn to speak of."""
    if centre is not None:
        c = torch.as_tensor(centre, dtype=a.dtype).flatten()[:2]
        pa, pb = a - c, b - c
    else:
        pa, pb = a - a.mean(0), b - b.mean(0)
    theta = float(torch.atan2((pa[:, 0] * pb[:, 1] - pa[:, 1] * pb[:, 0]).sum(), (pa * pb).sum()))
    if abs(theta) < 1e-3:
        return None
    if centre is None:
        r = torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=a.dtype)
        c = torch.linalg.solve(torch.eye(2, dtype=a.dtype) - r, b.mean(0) - r @ a.mean(0))
    return c, theta


def _blend(big_a: torch.Tensor, big_b: torch.Tensor, s: float, breakdown=None, through: float = 0.5,
           pivot=None) -> torch.Tensor:
    """The points (x, y, pressure) ``s`` of the way from ``big_a`` to ``big_b``: on the curve through the
    ``breakdown`` (at ``through`` of the way), round the ``pivot``, or straight."""
    xa, xb = big_a[:, :2], big_b[:, :2]
    pressure = big_a[:, 2] + s * (big_b[:, 2] - big_a[:, 2])
    if breakdown is not None:
        u = min(max(float(through), 0.05), 0.95)
        l0 = (s - u) * (s - 1.0) / u
        l1 = s * (s - 1.0) / (u * (u - 1.0))
        l2 = s * (s - u) / (1.0 - u)
        xy = l0 * xa + l1 * breakdown[:, :2] + l2 * xb
    elif pivot is not None:
        c, theta = pivot
        turned = c + _rot(xa - c, theta)
        xy = c + _rot(xa - c, s * theta) + s * (xb - turned)
    else:
        xy = xa + s * (xb - xa)
    return torch.cat([xy, pressure[:, None]], 1)


def _arc(arc, big_a, big_b, sizes, closed):
    """The breakdown points or the pivot an ``arc`` asks for."""
    if arc is None:
        return None, None
    if isinstance(arc, Line):
        return _onto(big_a, sizes, closed, arc), None
    if isinstance(arc, str):
        if arc != "pivot":
            raise ValueError(f"arc is None, 'pivot', a pivot point (x, y), or a breakdown Line, not {arc!r}")
        return None, _pivot(big_a[:, :2], big_b[:, :2])
    return None, _pivot(big_a[:, :2], big_b[:, :2], arc)


def _line(a: Line, b: Line, s: float, mid: torch.Tensor, sizes: tuple, closed: bool) -> Line:
    return Line(mid[:, :2].contiguous(), mid[:, 2].clamp(0.0, 1.0), a.tool, a.colour if s < 0.5 else b.colour,
                a.marker, a.kind, a.age + 1, a.uid, sizes if closed else None)


def inbetween(a: Line, b: Line, t: float, ease="in-out", arc=None, *, through: float = 0.5) -> Line:
    """
    The in-between of one line between two key drawings of it, ``a`` and ``b`` (the same uid), at time ``t``
    (0 at ``a``, 1 at ``b``). The ``ease`` (see :func:`spacing`) turns time into how far along the move the line
    is, so with ``"in-out"`` the in-betweens bunch toward both keys. ``arc`` says how it moves:

    - None: straight from ``a`` to ``b``;
    - a :class:`~helix_lm.paint.draw.Line`: the breakdown, the same line between the keys; the in-between moves
      on the curve through it, passing it ``through`` of the way along (the passing position, 0.5);
    - ``"pivot"``: it turns round the point that the best rigid turn from ``a`` to ``b`` leaves still (a limb
      round its joint), and what the turn does not explain is carried straight;
    - a point ``(x, y)`` in board widths: it turns round that pivot.

    The two drawings are laid on one another point for point (a line drawn from the other end is turned round, a
    ring is turned to start where the other does); a cel shape with another number of rings in ``b`` cannot be
    in-betweened and pops from ``a`` to ``b`` halfway.
    """
    s = float(_ease(ease)(min(max(float(t), 0.0), 1.0)))
    pair = _pair(a, b)
    if pair is None:
        line = a if s < 0.5 else b
        return line.moved(line.points.clone(), age=line.age)
    big_a, big_b, sizes, closed = pair
    breakdown, pivot = _arc(arc, big_a, big_b, sizes, closed)
    return _line(a, b, s, _blend(big_a, big_b, s, breakdown, through, pivot), sizes, closed)


def inbetweens(sheet_a, sheet_b, count: int, ease="in-out", arc=None, *, through: float = 0.5,
               switch: float = 0.5) -> list:
    """
    ``count`` in-between drawings (sheets) between two key drawings, evenly timed and spaced by the ``ease``
    chart (:func:`spacing`). Lines in both keys (the same uid) are in-betweened (:func:`inbetween`); ``arc`` is
    None, ``"pivot"`` (each line turns round its own pivot found from the keys), a pivot point, or a breakdown
    sheet whose lines (by uid) the in-betweens pass through.

    A line in only one key is drawn as in-betweeners do, whole or not at all: a line only in ``sheet_a`` stays
    until the move is ``switch`` of the way along, a line only in ``sheet_b`` comes in from there, and either is
    moved with the in-betweened lines round it so it stays on the drawing. Up to ``switch`` the lines are in the
    order of ``sheet_a``, after it in that of ``sheet_b``.
    """
    lines_a, lines_b = _by_uid(sheet_a.lines), _by_uid(sheet_b.lines)
    through_lines = _by_uid(arc.lines) if hasattr(arc, "lines") else None
    pairs = {}
    for uid, la in lines_a.items():
        lb = lines_b.get(uid)
        pair = None if lb is None else _pair(la, lb)
        if pair is None:
            continue
        big_a, big_b, sizes, closed = pair
        if through_lines is not None:
            bd = through_lines.get(uid)
            breakdown, pivot = (None if bd is None else _onto(big_a, sizes, closed, bd)), None
        else:
            breakdown, pivot = _arc(arc, big_a, big_b, sizes, closed)
        pairs[uid] = (la, lb, big_a, big_b, sizes, closed, breakdown, pivot)
    out = []
    for s in spacing(count, ease):
        early = s < switch
        side = sheet_a if early else sheet_b
        mids = {uid: _blend(p[2], p[3], s, p[6], through, p[7]) for uid, p in pairs.items()}
        if pairs:
            src = torch.cat([(p[2] if early else p[3])[:, :2] for p in pairs.values()])
            disp = torch.cat([mids[uid][:, :2] for uid in pairs]) - src
        else:
            src = disp = torch.zeros(0, 2)
        lines, done = [], set()
        for line in side.lines:
            if line.uid in pairs and line.uid not in done:
                done.add(line.uid)
                la, lb, _, _, sizes, closed, _, _ = pairs[line.uid]
                lines.append(_line(la, lb, s, mids[line.uid], sizes, closed))
            else:
                lines.append(line.moved(line.points + _carry(line.points.float(), src, disp), age=line.age))
        out.append(replace(sheet_a, lines=lines, kept=len(done), new=len(lines) - len(done), fresh=False))
    return out


# ----------------------------------------------------------------------------------------- exaggeration --
def _average(ys: torch.Tensor, sigma: float) -> torch.Tensor:
    """``ys (L, ...)`` averaged through time by a Gaussian ``sigma`` frames wide, the path mirrored through its
    first and last points at the ends (so they stay where they are and a steady glide is not changed)."""
    n = ys.shape[0]
    r = min(int(math.ceil(3.0 * sigma)), n - 1)
    if r < 1 or sigma <= 0:
        return ys.clone()
    flat = ys.reshape(n, -1)
    before = 2.0 * flat[:1] - flat[1 : r + 1].flip(0)
    after = 2.0 * flat[-1:] - flat[-r - 1 : -1].flip(0)
    ext = torch.cat([before, flat, after])
    k = torch.exp(-(torch.arange(-r, r + 1, dtype=ys.dtype) ** 2) / (2.0 * sigma ** 2))
    k = k / k.sum()
    out = F.conv1d(ext.T[:, None], k.view(1, 1, -1))[:, 0].T
    return out.reshape(ys.shape)


def _stretched(p: torch.Tensor, along_from: torch.Tensor, across_from: torch.Tensor, d: torch.Tensor,
               f: float) -> torch.Tensor:
    """Points scaled by ``f`` along direction ``d`` (from the line through ``along_from``) and by ``1 / f``
    across it (about ``across_from``): the area is kept."""
    normal = torch.stack([-d[1], d[0]])
    a = (p - along_from) @ d
    b = (p - across_from) @ normal
    return p + d * (a * (f - 1.0))[:, None] + normal * (b * (1.0 / f - 1.0))[:, None]


def _squash_stretch(p, v, v_in, v_out, amount: float, noise: float) -> torch.Tensor:
    """How the tracked points ``p (P, 2)`` of one frame move to squash and stretch: stretched along the motion
    (velocities ``v``) by up to ``1 + amount``, and squashed against the side they hit where the motion turns back
    on itself (``v_in``, ``v_out``: the moves into and out of the frame). The area is kept, and points that do not
    move are not changed (one moving subject is assumed)."""
    start, p = p, p.clone()
    speed = v.norm(dim=-1)
    w = torch.where(speed > noise, speed, torch.zeros_like(speed))
    if float(w.sum()) > 0:
        c = (w[:, None] * p).sum(0) / w.sum()
        vg = (w[:, None] * v).sum(0) / w.sum()
        sg = float(vg.norm())
        if sg > noise:
            d = vg / sg
            size = float(((w * ((p - c) @ d) ** 2).sum() / w.sum()).sqrt())
            f = 1.0 + amount * sg / (4.0 * max(size, noise) + sg)
            share = (speed / (0.5 * sg)).clamp(0.0, 1.0)[:, None]
            p = p + (_stretched(p, c, c, d, f) - p) * share
    s_in, s_out = v_in.norm(dim=-1), v_out.norm(dim=-1)
    w = torch.where((s_in > noise) & (s_out > noise), s_in + s_out, torch.zeros_like(s_in))
    if float(w.sum()) > 0:
        gi, go = (w[:, None] * v_in).sum(0) / w.sum(), (w[:, None] * v_out).sum(0) / w.sum()
        if float((gi * go).sum()) < 0:                       # it turned back on itself: a hit, a bounce
            jolt = go - gi
            j = float(jolt.norm())
            d = jolt / j
            c = (w[:, None] * p).sum(0) / w.sum()
            a = (p - c) @ d
            moving = w > 0
            size = float(((w * a ** 2).sum() / w.sum()).sqrt())
            g = 1.0 / (1.0 + amount * j / (4.0 * max(size, noise) + j))
            contact = c + d * float(a[moving].min())          # the side it hits stays where it is
            share = ((s_in + s_out) / (0.5 * (float(gi.norm()) + float(go.norm())) + 1e-9)).clamp(0.0, 1.0)[:, None]
            p = p + (_stretched(p, contact, c, d, g) - p) * share
    return p - start


def exaggerate(sheets: list, amount: float = 0.0, *, stretch: float = 0.0, window: float = 12.0,
               noise: float = 0.5, scale: float = 512.0) -> list:
    """
    The animation pushed further than life: a new list of sheets.

    ``amount`` pushes every traced line's path away from its average along the motion: each point's path is
    compared with its average over about ``window`` frames (a Gaussian a quarter of that wide; the first and last
    frames the line is in stay where they are) and moved ``amount`` times further from it, so the extremes of a
    swing go further, an arc bends more and a steady glide is not changed. ``stretch`` squashes and stretches
    (Thomas and Johnston): the moving part of each frame is stretched along its motion by up to ``1 + stretch``
    as it goes faster, and squashed against the side it hits where the motion turns back on itself, its area
    kept. One moving subject is assumed: with two moving in different ways,
    they are stretched as one. ``noise`` is the speed (pixels a frame of a picture ``scale`` pixels wide) below
    which a point is taken to be still.

    Lines not traced from frame to frame are moved with the traced lines round them, so the drawing stays together.
    Both are 0 by default, which changes nothing.
    """
    if not amount and not stretch:
        return [replace(s, lines=list(s.lines)) for s in sheets]
    trs = tracks(sheets)
    noise_bw = noise / scale
    per_frame: list = [[] for _ in sheets]                   # [track, index, points, v, v_in, v_out, delta]
    for tr in trs.values():
        for run in _runs(tr.frames):
            if len(run) < 3:
                continue
            ys = tr.points[run]
            delta = torch.zeros_like(ys)
            if amount:
                delta = delta + amount * (ys - _average(ys, window / 4.0))
            step = ys[1:] - ys[:-1]
            v_in = torch.cat([torch.zeros_like(ys[:1]), step])
            v_out = torch.cat([step, torch.zeros_like(ys[:1])])
            v = torch.cat([step[:1], 0.5 * (step[1:] + step[:-1]), step[-1:]])
            for i, idx in enumerate(run):
                per_frame[tr.frames[idx]].append([tr, idx, ys[i], v[i], v_in[i], v_out[i], delta[i]])
    out = []
    for f, sheet in enumerate(sheets):
        items = per_frame[f]
        if not items:
            out.append(replace(sheet, lines=list(sheet.lines)))
            continue
        if stretch:
            cat = [torch.cat([it[k] for it in items]) for k in (2, 3, 4, 5)]
            move = _squash_stretch(*cat, stretch, noise_bw)
            at = 0
            for it in items:
                n = len(it[2])
                it[6] = it[6] + move[at : at + n]
                at += n
        lines = list(sheet.lines)
        done = set()
        for tr, idx, _, _, _, _, delta in items:
            pos = tr.where[idx]
            lines[pos] = _moved(sheet.lines[pos], delta, tr.maps[idx], tr.sizes)
            done.add(pos)
        src = torch.cat([it[2] for it in items])
        disp = torch.cat([it[6] for it in items])
        for pos, line in enumerate(sheet.lines):
            if pos not in done:
                lines[pos] = line.moved(line.points + _carry(line.points.float(), src, disp), age=line.age)
        out.append(replace(sheet, lines=lines))
    return out


__all__ = ["EASES", "RATE", "Track", "exaggerate", "exposure", "inbetween", "inbetweens", "keys", "motion_amount",
           "playback", "spacing", "steady", "track_motion", "tracks"]
