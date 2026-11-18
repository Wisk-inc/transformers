# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Voice similarity.

With TriStream's codec and speaker encoder (:func:`encoder_from`, or any ``embed``), it is the cosine of two
embeddings, the measure the model was trained on. Without one the fallback compares the long-term spectral
envelope: the mean of the first 12 mel cepstra over each signal's frames within 20 dB of its loudest (each frame's
bands held within 50 dB of its loudest band), turned into a distance in dB (the scale of mel cepstral distortion)
and then into ``exp(-distance / 4 dB)``. On the synthetic voices one singer on two notes is 1.3 to 1.7 dB apart
(about 0.7), two singers on one vowel 3.6 to 4.0 dB (about 0.4). It also moves with the vowels sung, so it ranks
systems on one set and catches a conversion that kept the source's voice; it is not a speaker verification score.

Until 2.4.0 the fallback was the cosine of the cepstra's mean and spread. That cosine scored a harmonic-noise copy
of a take at -0.6 against the take itself and two different singers at 0.96, so it could not tell a converted
voice from the source; it was replaced before any results were published.
"""

from __future__ import annotations

import math

import torch

from .mfcc import mel_cepstrum

#: dB of cepstral distance at which the fallback similarity falls to 1/e
SCALE_DB = 4.0
_DB = 10.0 / math.log(10.0) * math.sqrt(2.0)          # the mel cepstral distortion constant


def cepstral_print(wave: torch.Tensor, sample_rate: int, n_coeffs: int = 13, within_db: float = 20.0,
                   floor_db: float = 50.0):
    """The mean of cepstra ``1..n_coeffs-1`` over the frames within ``within_db`` of the loudest one, with each
    frame's bands held within ``floor_db`` of its loudest band."""
    c = mel_cepstrum(wave, sample_rate, n_coeffs=n_coeffs, floor_db=floor_db)
    if c.shape[0] == 0:
        return torch.zeros(n_coeffs - 1)
    level_db = c[:, 0] * math.sqrt(2.0) * 20.0 / math.log(10.0)      # c0 is the mean log amplitude / sqrt(2)
    keep = level_db > level_db.max() - within_db
    return c[keep][:, 1:].mean(0)


def envelope_distance_db(a: torch.Tensor, b: torch.Tensor, sample_rate: int) -> float:
    return _DB * float((cepstral_print(a, sample_rate) - cepstral_print(b, sample_rate)).norm())


def encoder_from(codec, speaker_encoder):
    """``embed(wave, sample_rate)`` with TriStream's own codec frames and speaker encoder."""

    def embed(wave: torch.Tensor, sample_rate: int) -> torch.Tensor:
        device = next(codec.parameters()).device
        with torch.no_grad():
            frames = codec.listen(wave.flatten()[None].to(device))
            e = speaker_encoder(frames)[0].float().cpu()
        return e / e.norm().clamp(min=1e-9)

    return embed


def speaker_similarity(a: torch.Tensor, b: torch.Tensor, sample_rate: int, embed=None) -> float:
    if embed is None:
        return math.exp(-envelope_distance_db(a, b, sample_rate) / SCALE_DB)
    ea, eb = embed(a, sample_rate), embed(b, sample_rate)
    return float((ea * eb).sum() / (ea.norm() * eb.norm()).clamp(min=1e-9))


def similarity_metrics(pred: torch.Tensor, target_voice: torch.Tensor, sample_rate: int, source=None, embed=None):
    """Similarity to the voice asked for and, given the source performance, to the voice that was replaced (a
    conversion should be nearer the first than the second)."""
    out = {"speaker_similarity": speaker_similarity(pred, target_voice, sample_rate, embed)}
    if source is not None:
        out["source_similarity"] = speaker_similarity(pred, source, sample_rate, embed)
        out["identity_margin"] = out["speaker_similarity"] - out["source_similarity"]
    return out


__all__ = ["SCALE_DB", "cepstral_print", "encoder_from", "envelope_distance_db", "similarity_metrics",
           "speaker_similarity"]
