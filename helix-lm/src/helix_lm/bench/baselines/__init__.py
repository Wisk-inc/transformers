# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Systems TriStream is measured against: the input itself (the ceiling for content, the floor for voice
similarity), Griffin-Lim (a phase-free floor for resynthesis) and a harmonic-plus-noise vocoder with an envelope
transplant (the classic parametric way to resynthesise, convert and sing)."""

from .griffin_lim import GriffinLim
from .harmonic_noise import HarmonicNoise
from .identity import Identity

BASELINES = {"identity": Identity, "griffin-lim": GriffinLim, "harmonic-noise": HarmonicNoise}

__all__ = ["BASELINES", "GriffinLim", "HarmonicNoise", "Identity"]
