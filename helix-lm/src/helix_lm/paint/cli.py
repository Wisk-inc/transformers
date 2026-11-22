# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""``helix paint make|read|change|sizes|molab``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _hand(args):
    """The hand and the eye for a command: the trained model with the eye it was trained with, or the untrained
    hand, alone or with a pretrained eye named by ``--eye``."""
    if args.model:
        from .model import load_hand

        return load_hand(args.model, args.size, polish=args.polish)
    if args.eye:
        from .eye import DEFAULT, load_eye

        return None, load_eye(DEFAULT if args.eye == "siglip" else args.eye)
    return None, None


def main(argv: list[str] | None = None) -> int:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["molab"]:
        from ..scripts.pinxit_molab import main as molab

        return molab(argv[1:])
    parser = argparse.ArgumentParser(prog="helix paint", description="Pinxit: paint a photo stroke by stroke")
    sub = parser.add_subparsers(dest="cmd", required=True)
    make = sub.add_parser("make", help="paint a photo")
    make.add_argument("photo")
    make.add_argument("--style", default=None, help="a painting whose style to take: a picture of it, or the name "
                                                    "of a famous one ('mona lisa'; see: helix paint styles)")
    make.add_argument("--emotion", default=None, help="awe (majesty), serenity, joy, tenderness, melancholy, fear, "
                                                    "anger, mystery")
    make.add_argument("--out", default="painting.png", help=".png/.jpg picture, .pinxit (to change later), .svg, .gif")
    make.add_argument("--keep", default=None, help="also save the painting with its memory here (.pinxit)")
    make.add_argument("--width", type=int, default=1600)
    make.add_argument("--work", type=int, default=512)
    make.add_argument("--restraint", type=float, default=None)
    make.add_argument("--compose", type=float, default=0.6)
    make.add_argument("--exaggerate", type=float, default=0.6)
    make.add_argument("--seed", type=int, default=0)
    make.add_argument("--sheet", default=None, help="a picture of the photo, the plan and every pass")
    change = sub.add_parser("change", help="change a saved painting (.pinxit)")
    change.add_argument("painting")
    change.add_argument("--add", default=None, help="a picture to paint in")
    change.add_argument("--words", default=None, help="words to paint in (needs --model or --eye siglip)")
    change.add_argument("--touch", action="store_true", help="take the box further")
    change.add_argument("--rework", default=None, help="paint again with this emotion")
    change.add_argument("--undo", action="store_true")
    change.add_argument("--box", default="0.3,0.3,0.7,0.7", help="left,top,right,bottom as shares of the canvas")
    change.add_argument("--out", default=None)
    change.add_argument("--width", type=int, default=1600)
    read = sub.add_parser("read", help="measure a painting's style and feeling")
    read.add_argument("picture")
    read.add_argument("--out", default=None)
    sub.add_parser("sizes", help="the model sizes")
    sub.add_parser("styles", help="the famous paintings whose style can be asked for by name")
    for p in (make, change):
        p.add_argument("--model", default=None, help="a trained checkpoint or a Hub repository; without it the "
                                                     "untrained hand paints")
        p.add_argument("--size", default="px-base")
        p.add_argument("--polish", type=float, default=0.25)
        p.add_argument("--eye", default=None, help="without --model: a pretrained eye that says what the photo "
                                                   "is about and reads --words ('siglip' for the usual one)")
    args, rest = parser.parse_known_args(argv)
    if args.cmd == "styles":
        from .style import style_names

        for name, what in style_names().items():
            print(f"{name:20s} {what}")
        return 0
    if args.cmd == "sizes":
        from .model import Pinxit, PinxitConfig

        for size in ("px-tiny", "px-small", "px-base", "px-large"):
            print(f"{size}: {Pinxit(PinxitConfig.named(size)).num_parameters() / 1e6:.1f}M parameters")
        return 0
    if args.cmd == "read":
        from .emotion import strongest
        from .style import read_style

        card = read_style(args.picture, name=Path(args.picture).stem)
        print(f"feeling: {strongest(card.emotion)}; strokes flow {card.flow}; length {card.length:.2f}, width "
              f"{card.width:.2f}, texture {card.texture:.2f}, edges {card.hardness:.2f}; {len(card.palette)} pigments")
        if args.out:
            card.save(args.out)
        return 0
    hand, eye = _hand(args)
    if args.cmd == "make":
        from .painting import paint

        painting = paint(args.photo, style=args.style, emotion=args.emotion, hand=hand, eye=eye, log=print,
                         work=args.work, restraint=args.restraint, compose=args.compose, exaggerate=args.exaggerate,
                         seed=args.seed)
        painting.save(args.out, args.width)
        if args.keep:
            painting.save(args.keep)
        if args.sheet:
            painting.sheet(args.sheet)
        print(json.dumps(painting.summary(), indent=1))
        return 0
    from .painting import Painting

    painting = Painting.load(args.painting)
    box = tuple(float(v) for v in args.box.split(","))
    if args.undo:
        painting.undo()
    if args.add:
        painting.add(image=args.add, box=box, hand=hand, log=print)
    if args.words:
        painting.add(text=args.words, box=box, eye=eye, log=print)
    if args.touch:
        painting.touch(box, hand=hand, log=print)
    if args.rework:
        painting.rework(args.rework, hand=hand, log=print)
    painting.save(args.painting)
    if args.out:
        painting.save(args.out, args.width)
    print(json.dumps(painting.summary(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
