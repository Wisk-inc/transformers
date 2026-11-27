# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Pinxit: the painter built on Helix. It looks at a photo and paints it stroke by stroke, with a palette and
brushes, and keeps what it did so the painting can be changed later. See ``docs/PINXIT.md``.

    from helix_lm.paint import paint
    painting = paint("photo.jpg", style="a_painting.jpg", emotion="majesty")
    painting.save("painting.png")          # the picture
    painting.save("painting.pinxit")       # the painting with its memory, to change later
"""

from .emotion import EMOTIONS, Feeling, feeling_of, read_emotion
from .mind import Settings
from .painting import Painting, paint
from .style import StyleCard, named_style, read_style, style_names

_LATER = {"load_hand": "model", "ModelHand": "model", "Pinxit": "model", "PinxitConfig": "model", "load_eye": "eye",
          "load_imaginer": "imagine", "animate": "animate", "Animation": "animate", "load_animator": "video",
          "list_models": "video", "draw_anime": "anime", "read_space": "space", "paint_frame": "cel",
          "making_of": "studio"}


def __getattr__(name: str):
    """The trained model and the pretrained eye are loaded only when asked for: painting with the rules alone
    needs neither."""
    if name in _LATER:
        from importlib import import_module

        return getattr(import_module(f"{__name__}.{_LATER[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["EMOTIONS", "Animation", "Feeling", "ModelHand", "Painting", "Pinxit", "PinxitConfig", "Settings",
           "StyleCard", "animate", "draw_anime", "feeling_of", "list_models", "load_animator", "load_eye", "load_hand",
           "load_imaginer", "making_of", "named_style", "paint", "paint_frame", "read_emotion", "read_space",
           "read_style", "style_names"]
