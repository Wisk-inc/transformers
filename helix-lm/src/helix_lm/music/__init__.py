# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Music: what a song is doing, and what a voice should do inside it.

* :mod:`analysis` -- tempo, beats and downbeats, key and scale, chords and tension, loudness, brightness,
  structure (intro / verse / build / chorus / drop), reverb, groove and a "vibe" vector, all from the waveform
  with no external audio library;
* :mod:`frisson` -- the critic: a timeline of how likely a moment is to give a listener chills, from the
  features the chills literature found predictive (dynamic surges, new voice entries, register expansion,
  harmonic surprise, tension released, groove), with the weights as documented priors to be learned;
* :mod:`conductor` -- the actor: turns the analysis and the critic into a vocal plan -- where the voice
  enters (bar-aligned, before the chorus, on the resolution), which notes (snapped to the key, inside the
  singer's range), which harmonies, and how (breathiness, power, vibrato, warmth, echo) -- for TriStream to sing;
* :mod:`harmony` -- scales, chords, voice leading and harmony lines;
* :mod:`separation` -- a band-split separator with bleed control and target-singer extraction for duets;
* :mod:`effects` -- reverb, echo, muffle, retune to scale, formant shift, doubling, de-essing, ducking,
  limiting, breath insertion, and glitch detection;
* :mod:`conversion` -- the voice-swap pipeline: separate, analyse, convert to the cloned voice, remix;
* :mod:`tristream` -- TriStream-2: the 765M source-filter singing model on the Anansi trunk;
* :mod:`regularize` -- the data rules: accent balance and tagging, pitch-range buckets, effects
  augmentation, duet mixing, dry-vocal checks;
* :mod:`data` -- the music and singing dataset catalogue with licences, and multi-file ingestion.

Everything is pure PyTorch and runs on a CPU at toy scale (``tests/test_music.py``). The published evidence
behind the design is in ``docs/TRISTREAM.md``.
"""

from .analysis import AnalysisResult, analyze, estimate_key, estimate_tempo, snap_midi_to_scale, track_beats

__all__ = ["AnalysisResult", "analyze", "estimate_key", "estimate_tempo", "snap_midi_to_scale", "track_beats"]
