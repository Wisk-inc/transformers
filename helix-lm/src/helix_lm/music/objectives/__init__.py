# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Training objectives aimed at what the benchmarks measure: pitch in cents (a blurred target over the pitch bins
and a sub-bin decoder), voicing with its boundaries weighted, spectra at several resolutions, and the spectral
envelope that carries the vowels and the brightness."""

from .formant_loss import envelope_loss, log_envelope
from .pitch_loss import BIN_CENTS, blurred_targets, decode_pitch, pitch_distribution_loss
from .stft_loss import mrstft_loss
from .vuv_loss import boundary_weights, voicing_loss

__all__ = ["BIN_CENTS", "blurred_targets", "boundary_weights", "decode_pitch", "envelope_loss", "log_envelope",
           "mrstft_loss", "pitch_distribution_loss", "voicing_loss"]
