# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Feeling: what a painting is meant to make the viewer feel, and how that changes the way it is painted.

Each emotion is a :class:`Feeling`: what it does to colour (saturation, warmth, how light the picture is, contrast,
glowing lights, warm lights against cool shadows), to the hand (how fast and how hard the brush moves, how long,
wide, bent and scattered the strokes are, which brushes), and to the painter's choices (how much is left
unfinished, how much the subject is enlarged, stretched or tilted, how strongly values are gathered into masses,
which way loose strokes flow).

:func:`read_emotion` guesses the feeling of a picture from its colours and its brushwork. It is a prior, not a
trained classifier: pleasure and arousal come from Valdez and Mehrabian's measurements of colour and emotion
(1994: pleasure = 0.69 brightness + 0.22 saturation, arousal = -0.31 brightness + 0.60 saturation), arousal is
raised by busy brushwork, and the nearest of eight hand-set prototypes wins.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch

from .color import image_to_oklab

EMOTIONS = ("awe", "serenity", "joy", "tenderness", "melancholy", "fear", "anger", "mystery")
ALIASES = {"majesty": "awe", "majestic": "awe", "grand": "awe", "calm": "serenity", "peace": "serenity",
           "happy": "joy", "happiness": "joy", "love": "tenderness", "tender": "tenderness", "sad": "melancholy",
           "sadness": "melancholy", "dread": "fear", "tension": "fear", "rage": "anger", "fury": "anger",
           "mysterious": "mystery", "dream": "mystery"}
FLOWS = ("follow", "toward", "radial", "horizontal", "vertical", "swirl", "directional")


@dataclass
class Feeling:
    saturation: float = 1.0      # colour strength
    warmth: float = 0.0          # shift toward yellow-red (+) or blue (-), in OKLab units
    key: float = 0.0             # the whole picture lighter (+) or darker (-)
    contrast: float = 1.0        # lights and darks pushed apart
    glow: float = 0.0            # the lights lifted, warmed and spread
    temperature: float = 0.25    # lights warmer and shadows cooler than they were (negative: the reverse)
    speed: float = 0.5           # how fast the brush tends to move
    pressure: float = 0.5        # how hard it tends to be pressed
    length: float = 1.0
    width: float = 1.0
    bend: float = 0.3
    jitter: float = 0.15         # how far strokes stray from the direction they were given
    restraint: float = 0.5       # how much of the picture is left at its first rough state
    magnify: float = 0.04        # the subject enlarged
    stretch: float = 0.0         # the picture made taller (+) or wider (-) about the subject
    tilt: float = 0.0            # degrees the picture leans, to break its balance
    masses: float = 0.25         # values gathered into a few masses
    brushes: tuple = (1.0, 1.0, 1.2, 0.5, 0.3, 0.5)     # flat, round, filbert, dry, knife, wash
    flow: str = "follow"         # where strokes go when the picture gives them no direction

    def blend(self, other: Feeling, share: float) -> Feeling:
        """This feeling with ``share`` (0..1) of another mixed in."""
        out = {}
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if f.name == "flow":
                out[f.name] = b if share >= 0.5 else a
            elif f.name == "brushes":
                out[f.name] = tuple(x + share * (y - x) for x, y in zip(a, b, strict=True))
            else:
                out[f.name] = a + share * (b - a)
        return Feeling(**out)

    def to_json(self) -> dict:
        return asdict(self)


NEUTRAL = Feeling()
FEELINGS = {
    "awe": Feeling(1.12, 0.012, -0.02, 1.28, 0.55, 0.6, 0.55, 0.72, 1.35, 1.15, 0.25, 0.10, 0.62, 0.08, 0.06, 0.0,
                   0.45, (1.4, 0.6, 1.2, 0.6, 0.9, 0.5), "toward"),
    "serenity": Feeling(0.92, -0.006, 0.04, 0.85, 0.15, 0.2, 0.25, 0.30, 1.30, 1.20, 0.12, 0.06, 0.55, 0.0, -0.04,
                        0.0, 0.35, (0.8, 1.0, 1.2, 0.2, 0.1, 1.6), "horizontal"),
    "joy": Feeling(1.30, 0.020, 0.05, 1.05, 0.30, 0.5, 0.75, 0.45, 0.70, 0.95, 0.50, 0.35, 0.45, 0.05, 0.0, 1.5,
                   0.20, (1.0, 1.4, 1.2, 0.6, 0.3, 0.3), "swirl"),
    "tenderness": Feeling(0.85, 0.015, 0.07, 0.80, 0.35, 0.3, 0.20, 0.25, 0.90, 1.10, 0.40, 0.12, 0.60, 0.06, 0.0,
                          0.0, 0.30, (0.5, 1.5, 1.3, 0.2, 0.0, 1.4), "follow"),
    "melancholy": Feeling(0.60, -0.020, -0.05, 0.82, 0.0, -0.2, 0.22, 0.40, 1.25, 1.00, 0.20, 0.10, 0.70, 0.0, 0.03,
                          0.0, 0.40, (0.9, 0.8, 1.0, 1.0, 0.2, 1.3), "vertical"),
    "fear": Feeling(0.75, -0.012, -0.10, 1.35, 0.0, 0.45, 0.80, 0.60, 0.85, 0.80, 0.15, 0.50, 0.68, 0.10, 0.05, 3.0,
                    0.50, (1.2, 0.6, 0.8, 1.4, 0.8, 0.3), "follow"),
    "anger": Feeling(1.25, 0.035, -0.06, 1.40, 0.10, 0.3, 0.95, 0.90, 1.10, 1.05, 0.10, 0.45, 0.50, 0.10, 0.0, 4.0,
                     0.45, (1.3, 0.4, 0.7, 1.3, 1.5, 0.1), "follow"),
    "mystery": Feeling(0.80, -0.010, -0.12, 1.10, 0.25, 0.35, 0.35, 0.45, 1.10, 1.25, 0.35, 0.20, 0.78, 0.05, 0.0,
                       0.0, 0.55, (0.7, 0.9, 1.0, 0.5, 0.2, 1.8), "swirl"),
}
# pleasure, arousal, contrast, warmth of each emotion's typical picture
_PROTOTYPES = {"awe": (0.50, 0.25, 1.00, 0.30), "serenity": (0.62, 0.02, 0.45, -0.10), "joy": (0.72, 0.38, 0.70, 0.40),
               "tenderness": (0.70, 0.10, 0.40, 0.25), "melancholy": (0.38, -0.02, 0.50, -0.35),
               "fear": (0.25, 0.15, 1.00, -0.20), "anger": (0.33, 0.40, 1.05, 0.60),
               "mystery": (0.28, 0.02, 0.80, -0.15)}


def emotion_name(name: str) -> str:
    key = str(name).strip().lower()
    key = ALIASES.get(key, key)
    if key not in EMOTIONS and key != "neutral":
        raise ValueError(f"no emotion {name!r}; there are {', '.join(EMOTIONS)} (and neutral)")
    return key


def feeling_of(emotion, strength: float = 1.0) -> Feeling:
    """A :class:`Feeling` from a name, a ``{name: share}`` reading (:func:`read_emotion`), a Feeling, or None
    (neutral). ``strength`` 0 gives the neutral feeling, 1 the full one."""
    if emotion is None:
        return NEUTRAL
    if isinstance(emotion, Feeling):
        return NEUTRAL.blend(emotion, strength)
    if isinstance(emotion, str):
        name = emotion_name(emotion)
        return NEUTRAL if name == "neutral" else NEUTRAL.blend(FEELINGS[name], strength)
    shares = {emotion_name(k): float(v) for k, v in emotion.items() if emotion_name(k) != "neutral"}
    total = sum(shares.values()) or 1.0
    mixed, used = NEUTRAL, 0.0
    for name, share in sorted(shares.items(), key=lambda kv: -kv[1]):
        used += share / total
        mixed = mixed.blend(FEELINGS[name], (share / total) / used)
    return NEUTRAL.blend(mixed, strength)


def colour_feeling(image: torch.Tensor) -> dict:
    """Pleasure, arousal and dominance from a picture's colours (Valdez and Mehrabian, 1994), with its contrast
    and warmth. ``image`` is sRGB ``(3, H, W)``."""
    lab = image_to_oklab(image.float())
    brightness = float(lab[0].mean())
    saturation = min(1.0, float(torch.sqrt(lab[1] ** 2 + lab[2] ** 2).mean()) / 0.2)
    return {"pleasure": 0.69 * brightness + 0.22 * saturation, "arousal": -0.31 * brightness + 0.60 * saturation,
            "dominance": -0.76 * brightness + 0.32 * saturation, "contrast": min(1.5, float(lab[0].std()) / 0.25),
            "warmth": max(-1.0, min(1.0, 6.0 * float(lab[2].mean()) + 3.0 * float(lab[1].mean()))),
            "brightness": brightness, "saturation": saturation}


def read_emotion(image: torch.Tensor, energy: float = 0.0) -> dict:
    """``{emotion: share}`` (shares add to 1) for a picture; ``energy`` (0..1) is how busy its brushwork is
    (:func:`helix_lm.paint.style.read_style` measures it) and raises arousal."""
    f = colour_feeling(image)
    point = torch.tensor([f["pleasure"], f["arousal"] + 0.25 * energy, f["contrast"], f["warmth"]])
    scale = torch.tensor([0.18, 0.14, 0.35, 0.45])
    names = list(_PROTOTYPES)
    d2 = (((point - torch.tensor([_PROTOTYPES[n] for n in names])) / scale) ** 2).sum(-1)
    shares = torch.softmax(-d2 / 1.5, 0)
    return {n: float(s) for n, s in zip(names, shares, strict=True)}


def strongest(reading: dict) -> str:
    return max(reading, key=reading.get)


__all__ = ["ALIASES", "EMOTIONS", "FEELINGS", "FLOWS", "NEUTRAL", "Feeling", "colour_feeling", "emotion_name",
           "feeling_of", "read_emotion", "strongest"]
