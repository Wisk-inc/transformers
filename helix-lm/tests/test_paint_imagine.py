# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The painter paired with an image model that imagines the painting first, and words used where an eye reads
them: finding a thing, picking a picture, reading a feeling from a phrase. The image model is a stand-in here;
the real one was run by hand (docs/PINXIT.md)."""

import torch

from helix_lm.paint import named_style, paint
from helix_lm.paint.emotion import Feeling, feeling_of
from helix_lm.paint.eye import TinyEye
from helix_lm.paint.imagine import Imaginer, describe, edges
from helix_lm.paint.mind import Settings, make_plan
from helix_lm.paint.style import own_card
from helix_lm.paint.words import best_picture, box_of, cut_out, feeling_from_words


def _picture():
    img = torch.full((3, 96, 128), 0.8)
    img[:, 30:70, 40:90] = torch.tensor([0.8, 0.3, 0.1])[:, None, None]
    return img


class FakePipe:
    """Answers like a ``diffusers`` image-to-image pipeline: the picture made bluer, and what it was asked."""

    def __init__(self):
        self.asked = []

    def __call__(self, **kwargs):
        self.asked.append(kwargs)
        picture = torch.tensor(list(kwargs["image"].getdata()), dtype=torch.float32).T.reshape(
            3, kwargs["image"].size[1], kwargs["image"].size[0]) / 255.0
        picture[2] = (picture[2] + 0.3).clamp(0, 1)
        return type("Out", (), {"images": picture[None]})()


def test_edges_are_thin_lines_where_the_picture_changes():
    lines = edges(_picture())
    assert lines.shape == (3, 96, 128) and set(lines.unique().tolist()) <= {0.0, 1.0}
    assert float(lines[0, 30:70, 38:43].sum()) > 20 and float(lines[0, 40:60, 55:75].sum()) == 0.0
    assert float(lines[0, 2:20, 2:30].sum()) == 0.0                # nothing where the picture is even


def test_the_image_model_is_told_what_to_imagine():
    assert describe(None, None, "a cat").startswith("a cat, oil painting")
    words = describe(named_style("mona lisa"), "mystery", "a woman")
    assert "in the style of Leonardo da Vinci, Mona Lisa" in words and "sfumato" in words and "dusk" in words
    assert "impasto" in describe(named_style("the starry night")) and "outlines" in describe(named_style("the bedroom"))


def test_the_imagined_picture_is_what_gets_painted():
    pipe = FakePipe()
    imaginer = Imaginer(pipe, name="fake", controlled=True)
    img = _picture()
    out = imaginer(img, "a red box, oil painting", seed=3)
    assert out.shape == img.shape and float(out[2].mean()) > float(img[2].mean()) + 0.2
    asked = pipe.asked[-1]
    assert asked["prompt"] == "a red box, oil painting" and asked["strength"] == 0.8 and "control_image" in asked
    assert asked["image"].size == (128, 96) and "photorealistic" in asked["negative_prompt"]
    assert Imaginer(pipe, controlled=False).strength < 0.7          # without the edges it must stay nearer
    plan = make_plan(img, own_card(img), feeling_of(None), Settings(work=96), imagine=imaginer, words="a red box")
    assert any("imagined as a painting by fake: a red box" in line for line in plan.report)
    assert float(plan.source[2].mean()) > float(img[2].mean()) + 0.15
    assert not any("planes" in line for line in plan.report)        # the image model saw it in planes already
    painting = paint(img, emotion="joy", imagine=imaginer, subject="a red box", eye=TinyEye(), work=96, steps=0.25)
    assert pipe.asked[-1]["prompt"].startswith("a red box, oil painting") and "joyful" in pipe.asked[-1]["prompt"]
    assert len(painting) > 50


def test_words_find_things_pick_pictures_and_name_feelings():
    where = torch.zeros(10, 10)
    where[2:5, 6:9] = 1.0
    where[8, 0] = 0.6                                               # a second, weaker patch apart from it
    left, top, right, bottom = box_of(where, pad=0.0)
    assert 0.55 < left < 0.65 and 0.15 < top < 0.25 and 0.85 < right < 0.95 and 0.45 < bottom < 0.55
    eye = TinyEye()
    img = _picture()
    found = eye.find(img, "a red box")
    assert found.shape == (10, 10) and float(found.max()) == 1.0 and float(found.min()) >= 0.0
    mask = cut_out(eye, img, "a red box")
    assert mask.shape == (96, 128) and float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0
    number, scores = best_picture(eye, "a red box", [img, torch.rand(3, 64, 64)])
    assert number in (0, 1) and len(scores) == 2
    feeling, shares = feeling_from_words("quiet sadness at dusk", eye)
    assert isinstance(feeling, Feeling) and abs(sum(shares.values()) - 1.0) < 1e-5 and len(shares) == 8
