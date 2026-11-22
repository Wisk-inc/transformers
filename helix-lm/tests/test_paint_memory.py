# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""A painting keeps what was done, so it can be saved, opened and changed: things added, a part taken further,
the feeling changed, the last change taken back."""

import pytest
import torch

from helix_lm.paint import Painting, paint
from helix_lm.paint.eye import TinyEye


@pytest.fixture(scope="module")
def painted():
    g = torch.Generator().manual_seed(0)
    img = torch.rand(3, 96, 96, generator=g) * 0.1 + torch.tensor([0.35, 0.45, 0.30])[:, None, None]
    img[:, 20:50, 30:70] = torch.tensor([0.85, 0.25, 0.20])[:, None, None]
    return paint(img.clamp(0, 1), emotion="joy", work=128, steps=0.25, seed=1)


def test_a_painting_is_saved_whole_and_opens_the_same(painted, tmp_path):
    painted.save(tmp_path / "p.pinxit")
    back = Painting.load(tmp_path / "p.pinxit")
    assert torch.equal(back.rows, painted.rows) and back.plan.feeling == painted.plan.feeling
    assert back.history == painted.history and back.plan.card.palette == painted.plan.card.palette
    assert torch.allclose(back.render(96), painted.render(96))
    painted.save(tmp_path / "p.png", width=200)
    painted.save(tmp_path / "p.svg", width=200)
    painted.save(tmp_path / "p.gif")
    assert (tmp_path / "p.svg").read_text().count("<path") == len(painted)
    assert all((tmp_path / n).stat().st_size > 500 for n in ("p.png", "p.gif"))
    stages = painted.stages(64)
    assert stages[0][0] == "the toned canvas" and len(stages) >= 3
    trace = painted.performance()
    assert len(trace) == len(painted) and all(b["start"] > a["start"] for a, b in zip(trace, trace[1:], strict=False))
    slow = min(trace, key=lambda t: t["speed"])
    fast = max(trace, key=lambda t: t["speed"])
    assert slow["speed"] < fast["speed"]


def test_changes_are_painted_on_top_and_can_be_taken_back(painted, tmp_path):
    painted.save(tmp_path / "p.pinxit")
    p = Painting.load(tmp_path / "p.pinxit")
    before, picture = len(p), p.render(128)
    thing = torch.zeros(3, 20, 20)
    thing[2] = 1.0                                                             # a blue square
    p.add(image=thing, box=(0.05, 0.6, 0.35, 0.95))
    after = p.render(128)
    assert len(p) > before and p.history[-1]["op"] == "add"
    box = (slice(None), slice(84, 116), slice(12, 40))
    far = (slice(None), slice(0, 40), slice(80, 128))
    assert float((after[box] - picture[box]).abs().mean()) > 4 * float((after[far] - picture[far]).abs().mean())
    assert float(after[2][box[1:]].mean() - after[0][box[1:]].mean()) > float(
        picture[2][box[1:]].mean() - picture[0][box[1:]].mean())             # bluer where the square went
    count = len(p)
    p.touch((0.6, 0.6, 0.95, 0.95))
    assert len(p) >= count and p.history[-1]["op"] == "touch"
    p.add(text="a small red bird", box=(0.6, 0.05, 0.95, 0.4), eye=TinyEye(), steps=6)
    assert p.history[-1]["text"] == "a small red bird" and "eye_score" in p.history[-1]
    with pytest.raises(ValueError, match="reads text"):
        p.add(text="a bird", box=(0.6, 0.05, 0.95, 0.4))
    words = len(p)
    p.undo()
    assert len(p) < words and [h["op"] for h in p.history] == ["paint", "add", "touch"]
    p.rework("melancholy")
    assert p.plan.feeling.saturation < 0.9 and p.history[-1]["op"] == "rework"
    p.save(tmp_path / "p.pinxit")
    assert [h["op"] for h in Painting.load(tmp_path / "p.pinxit").history] == ["paint", "add", "touch", "rework"]


def test_undo_puts_the_painting_back_exactly_also_after_it_was_saved(painted, tmp_path):
    painted.save(tmp_path / "p.pinxit")
    p = Painting.load(tmp_path / "p.pinxit")
    plan = p.plan
    was = {"rows": p.rows.clone(), "target": plan.target.clone(), "source": plan.source.clone(),
           "focus": plan.focus.clone(), "finish": plan.finish.clone(), "feeling": plan.feeling,
           "report": list(plan.report), "picture": p.render(96)}
    thing = torch.zeros(3, 20, 20)
    thing[1] = 1.0
    p.add(image=thing, box=(0.05, 0.6, 0.35, 0.95))
    p.touch((0.5, 0.5, 0.9, 0.9))
    p.rework("melancholy")
    p.add(text="a small red bird", box=(0.6, 0.05, 0.95, 0.4), eye=TinyEye(), steps=4)
    assert not torch.equal(plan.target, was["target"]) and plan.feeling != was["feeling"] and len(p.snapshots) == 4
    p.save(tmp_path / "changed.pinxit")                           # the changes can still be taken back later
    p = Painting.load(tmp_path / "changed.pinxit")
    for _ in range(4):
        p.undo()
    plan = p.plan
    assert torch.equal(p.rows, was["rows"]) and [h["op"] for h in p.history] == ["paint"] and not p.snapshots
    for name in ("target", "source", "focus", "finish"):
        assert torch.equal(getattr(plan, name), was[name]), name
    assert plan.feeling == was["feeling"] and plan.report == was["report"]
    assert torch.equal(p.render(96), was["picture"])
    p.undo()                                                      # nothing left to take back: nothing happens
    assert torch.equal(p.rows, was["rows"])


def test_a_change_that_fails_leaves_the_painting_as_it_was(painted):
    class Broken:
        def propose(self, view):
            raise RuntimeError("the hand slipped")

    plan = painted.plan
    rows, finish, history = painted.rows.clone(), plan.finish.clone(), list(painted.history)
    with pytest.raises(RuntimeError, match="the hand slipped"):
        painted.touch((0.1, 0.1, 0.6, 0.6), hand=Broken())
    assert torch.equal(painted.rows, rows) and torch.equal(painted.plan.finish, finish)
    assert painted.history == history and not painted.snapshots


def test_words_can_be_painted_into_a_part_that_is_already_finished(painted):
    """The plan does not hold what the words ask for, so a finished canvas looks done there; the box is worked
    all the same."""
    from helix_lm.paint.color import srgb_to_linear
    from helix_lm.paint.engine import open_level
    from helix_lm.paint.io import resize

    plan, level = painted.plan, painted._levels()[-1]
    done = srgb_to_linear(resize(plan.target, level.size))        # a canvas that is the plan already
    mask = painted._box_mask((0.2, 0.2, 0.7, 0.7), feather=0.1)
    g = torch.Generator().manual_seed(0)
    assert open_level(plan, level, 4, done, g, region=mask) is None
    opened = open_level(plan, level, 4, done, g, region=mask, everything=True)
    assert opened is not None and int(opened["view"].counts.min()) == level.strokes
    inside = (opened["view"].cy.float() + 0.5) / level.gy
    assert float(inside.min()) > 0.15 and float(inside.max()) < 0.75           # and only the box
