# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Every metric the TriStream benchmarks report, with its unit and which way is better. The functions live in the
submodules; this table is what reports, composites and regression checks read.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricInfo:
    name: str
    unit: str
    better: str            # "lower" or "higher"
    what: str


_ROWS = [
    ("f0_rmse_cents", "cents", "lower", "pitch error on frames voiced in both"),
    ("f0_rmse_hz", "Hz", "lower", "pitch error on frames voiced in both"),
    ("log_f0_corr", "", "higher", "correlation of log pitch contours"),
    ("vuv_error", "share", "lower", "frames whose voicing differs"),
    ("gpe", "share", "lower", "gross pitch errors (over 20%) among voiced frames"),
    ("fpe_cents", "cents", "lower", "spread of the pitch error without gross errors"),
    ("ffe", "share", "lower", "frames with a voicing or a gross pitch error"),
    ("mcd_db", "dB", "lower", "mel-cepstral distortion after DTW alignment"),
    ("mrstft", "", "lower", "multi-resolution STFT distance"),
    ("lsd_db", "dB", "lower", "log-spectral distance"),
    ("note_accuracy", "share", "higher", "voiced frames within 50 cents of the written note"),
    ("note_error_cents", "cents", "lower", "mean absolute error to the written note"),
    ("vibrato_rate_error_hz", "Hz", "lower", "vibrato rate difference"),
    ("vibrato_extent_error_cents", "cents", "lower", "vibrato depth difference"),
    ("onset_error_ms", "ms", "lower", "note onset timing error"),
    ("loudness_corr", "", "higher", "correlation of loudness contours"),
    ("rms_error_db", "dB", "lower", "frame loudness error"),
    ("hnr_error_db", "dB", "lower", "harmonics-to-noise ratio difference"),
    ("jitter_error", "share", "lower", "period-to-period pitch jitter difference"),
    ("shimmer_error", "share", "lower", "period-to-period amplitude shimmer difference"),
    ("formant_error_hz", "Hz", "lower", "F1 and F2 error"),
    ("tilt_error_db", "dB/oct", "lower", "spectral tilt difference"),
    ("speaker_similarity", "", "higher", "cosine similarity of voice embeddings"),
    ("fad", "", "lower", "Frechet distance between embedding distributions"),
    ("quality_proxy", "", "higher", "signal-based quality proxy (not a listening test)"),
    ("glitches_per_minute", "1/min", "lower", "clicks and dropouts"),
    ("buzz_db", "dB", "lower", "codec frame-rate buzz"),
    ("consonant_clarity", "", "higher", "consonant onsets kept, against the reference"),
    ("rtf", "", "lower", "seconds of compute per second of audio"),
    ("peak_memory_gb", "GB", "lower", "peak accelerator memory"),
    ("shift_error_cents", "cents", "lower", "asked pitch shift against the one sung"),
    ("drift_cents_per_min", "cents/min", "lower", "pitch drift over a long take"),
    ("timbre_drift", "", "lower", "voice embedding drift over a long take"),
    ("register_pops_per_minute", "1/min", "lower", "register breaks heard as pops"),
    ("si_snr_db", "dB", "higher", "scale-invariant signal-to-noise ratio"),
]
INFO = {name: MetricInfo(name, unit, better, what) for name, unit, better, what in _ROWS}


def better(name: str) -> str:
    """``"lower"`` or ``"higher"``; a metric missing from the table counts as lower-is-better."""
    return INFO[name].better if name in INFO else "lower"


def is_improvement(name: str, new: float, old: float) -> bool:
    return new > old if better(name) == "higher" else new < old


__all__ = ["INFO", "MetricInfo", "better", "is_improvement"]
