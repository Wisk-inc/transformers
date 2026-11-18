# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Voice similarity: the cosine of two voice embeddings.

With TriStream's codec and speaker encoder (:func:`encoder_from`), the embedding is the one the model was trained
to match. Without them the fallback is the mean and spread of 20 mel cepstra over the loud frames: a weak voice
print, enough to rank systems on one set and to catch a conversion that kept the source's voice, not a speaker
verification score.
"""

from __future__ import annotations

import torch

from .mfcc import mel_cepstrum


def cepstral_print(wave: torch.Tensor, sample_rate: int) -> torch.Tensor:
    c = mel_cepstrum(wave, sample_rate, n_coeffs=21)
    level = c[:, 0]
    loud = c[level > level.max() - 2.0][:, 1:] if c.shape[0] else c[:, 1:]
    if loud.shape[0] < 2:
        loud = c[:, 1:]
    v = torch.cat([loud.mean(0), loud.std(0, unbiased=False)])
    return v / v.norm().clamp(min=1e-9)


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
    embed = embed or cepstral_print
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


__all__ = ["cepstral_print", "encoder_from", "similarity_metrics", "speaker_similarity"]
