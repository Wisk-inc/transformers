#!/usr/bin/env python3
# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Thin shim: the program lives in the package so ``pip install helix-lm`` ships it (see helix_lm/scripts)."""

from helix_lm.scripts.tristream_molab import main

if __name__ == "__main__":
    raise SystemExit(main())
