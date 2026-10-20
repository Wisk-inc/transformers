# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Singing voice synthesis conditioning: the score as streams at the codec's frame rate.

A singing model is a speech model that is told the pitch. Following DiffSinger/VISinger (explicit F0) and
Prompt-Singer (pitch as tokens in a decoder-only transformer), the score becomes four per-frame streams, each an
embedding, summed into the audio frame embeddings of :class:`~helix_lm.audio.omni.OmniForCausalLM`:

* **pitch**: log-F0 quantized to 256 bins over 50-1100 Hz (12 semitones an octave, ~4.5 bins a semitone) plus
  an "unvoiced" bin; given from a MIDI note or an F0 curve;
* **note**: the MIDI note number (0-127) plus "rest", for when the exact curve is not given and the model has to
  invent vibrato and portamento;
* **phoneme**: the lyric's phoneme id, stretched over the note's duration;
* **technique**: a tag per frame (plain, breathy, falsetto, vibrato, glissando, mixed, belting; GTSinger's set).

During training the F0 stream is dropped at random (``pitch_dropout``) so the model works from notes alone, and
the technique tags default to "plain". ``frames_from_notes`` lays a score out in frames: duration in seconds
times the frame rate, rounded with carried remainders so a bar keeps its length.
"""

from __future__ import annotations

import math

import torch
from torch import nn

TECHNIQUES = ("plain", "breathy", "falsetto", "vibrato", "glissando", "mixed", "belting", "pharyngeal")
F0_MIN, F0_MAX, PITCH_BINS = 50.0, 1100.0, 256


def hz_to_bins(f0: torch.Tensor) -> torch.Tensor:
    """F0 in Hz (0 or negative = unvoiced) -> bin ids in ``[0, PITCH_BINS]``; the last id is unvoiced."""
    voiced = f0 > 0
    log_f0 = torch.log(f0.clamp(min=F0_MIN).clamp(max=F0_MAX))
    scaled = (log_f0 - math.log(F0_MIN)) / (math.log(F0_MAX) - math.log(F0_MIN))
    bins = (scaled * (PITCH_BINS - 1)).round().long()
    return torch.where(voiced, bins, torch.full_like(bins, PITCH_BINS))


def bins_to_hz(bins: torch.Tensor) -> torch.Tensor:
    scaled = bins.clamp(max=PITCH_BINS - 1).float() / (PITCH_BINS - 1)
    hz = torch.exp(math.log(F0_MIN) + scaled * (math.log(F0_MAX) - math.log(F0_MIN)))
    return torch.where(bins >= PITCH_BINS, torch.zeros_like(hz), hz)


def midi_to_hz(note: torch.Tensor) -> torch.Tensor:
    return 440.0 * torch.pow(2.0, (note.float() - 69.0) / 12.0)


def frames_from_notes(notes: list[tuple[int, float, int]], frame_rate: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ``[(midi_note or -1 for rest, duration_seconds, phoneme_id), ...]`` -> ``(note_ids (T,), phoneme_ids (T,))``
    at ``frame_rate``; rounding remainders carry over so the total length is right.
    """
    note_ids, phoneme_ids, carry = [], [], 0.0
    for midi, seconds, phoneme in notes:
        exact = seconds * frame_rate + carry
        count = int(round(exact))
        carry = exact - count
        note_ids += [128 if midi < 0 else int(midi)] * count
        phoneme_ids += [int(phoneme)] * count
    return torch.tensor(note_ids, dtype=torch.long), torch.tensor(phoneme_ids, dtype=torch.long)


class SingingConditioner(nn.Module):
    """Score streams -> ``(B, T, hidden)`` conditioning to add to audio frame embeddings."""

    def __init__(self, hidden_size: int, num_phonemes: int = 128, pitch_dropout: float = 0.3) -> None:
        super().__init__()
        self.pitch = nn.Embedding(PITCH_BINS + 1, hidden_size)
        self.note = nn.Embedding(129, hidden_size)  # 128 MIDI notes + rest
        self.phoneme = nn.Embedding(num_phonemes, hidden_size)
        self.technique = nn.Embedding(len(TECHNIQUES), hidden_size)
        self.pitch_dropout = pitch_dropout
        for table in (self.pitch, self.note, self.phoneme, self.technique):
            nn.init.normal_(table.weight, std=0.02)

    def forward(
        self,
        note_ids: torch.Tensor,
        phoneme_ids: torch.Tensor,
        f0_hz: torch.Tensor | None = None,
        technique_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond = self.note(note_ids) + self.phoneme(phoneme_ids)
        if technique_ids is None:
            technique_ids = torch.zeros_like(note_ids)
        cond = cond + self.technique(technique_ids)
        if f0_hz is None:
            f0_hz = torch.where(note_ids < 128, midi_to_hz(note_ids), torch.zeros_like(note_ids, dtype=torch.float))
        pitch = self.pitch(hz_to_bins(f0_hz))
        if self.training and self.pitch_dropout > 0:
            keep = (torch.rand(note_ids.shape[0], 1, 1, device=note_ids.device) >= self.pitch_dropout).to(pitch.dtype)
            pitch = pitch * keep
        return cond + pitch


def estimate_f0(
    wave: torch.Tensor, sample_rate: int, frame_rate: float, f_min: float = F0_MIN, f_max: float = F0_MAX
) -> torch.Tensor:
    """
    A plain autocorrelation pitch tracker for tests and data preparation: ``(B, T)`` waveform -> ``(B, frames)``
    F0 in Hz, 0 where unvoiced. Not a replacement for a real tracker on noisy recordings.
    """
    hop = int(round(sample_rate / frame_rate))
    window = 2 * int(sample_rate / f_min)
    lag_min, lag_max = int(sample_rate / f_max), int(sample_rate / f_min)
    padded = torch.nn.functional.pad(wave, (0, window))
    frames = padded.unfold(-1, window, hop)  # (B, frames, window)
    frames = frames - frames.mean(dim=-1, keepdim=True)
    energy = frames.pow(2).mean(dim=-1)
    spectrum = torch.fft.rfft(frames, n=2 * window, dim=-1)
    autocorr = torch.fft.irfft(spectrum * spectrum.conj(), n=2 * window, dim=-1)[..., :window]
    autocorr = autocorr / autocorr[..., :1].clamp(min=1e-8)
    candidates = autocorr[..., lag_min:lag_max]
    best = candidates.argmax(dim=-1) + lag_min
    strength = candidates.max(dim=-1).values
    f0 = sample_rate / best.float()
    voiced = (strength > 0.5) & (energy > 1e-5)
    return torch.where(voiced, f0, torch.zeros_like(f0))
