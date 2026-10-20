# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
A speaker encoder: codec frames in, one unit vector per utterance out.

ECAPA-TDNN's shape (Desplanques et al., 2020) at a tenth of the size: dilated 1D convolutions with squeeze-and-
excitation over the frame sequence, attentive statistics pooling (a learned attention decides which frames count,
then mean and standard deviation are concatenated), a linear map to the embedding, L2 normalization. Trained with
an additive-angular-margin softmax over speaker identities (:func:`aam_softmax_loss`), it places utterances of one
voice close together: the number used to score voice cloning (SIM-o uses WavLM-TDNN embeddings the same way).

In Anansi the embedding is a *second* conditioning path for cloning. The first and stronger one is in-context:
the reference audio's codec tokens and transcript are placed in the prompt (what CosyVoice, VALL-E and Llasa do).
The embedding adds a compact, always-available voice vector, useful when the reference is long or the prompt
budget is short, and it is what a product calls "a voice".
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, bottleneck: int = 32) -> None:
        super().__init__()
        self.down = nn.Linear(channels, bottleneck)
        self.up = nn.Linear(bottleneck, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        s = torch.sigmoid(self.up(F.relu(self.down(x.mean(dim=-1)))))
        return x * s[..., None]


class TDNNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 1)
        self.conv2 = nn.Conv1d(channels, channels, 3, dilation=dilation, padding=dilation)
        self.conv3 = nn.Conv1d(channels, channels, 1)
        self.norm = nn.BatchNorm1d(channels)
        self.se = SqueezeExcite(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = self.norm(self.conv3(y))
        return x + self.se(y)


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.attention = nn.Sequential(nn.Conv1d(channels, hidden, 1), nn.Tanh(), nn.Conv1d(hidden, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T) -> (B, 2C)
        weights = torch.softmax(self.attention(x), dim=-1)
        mean = (weights * x).sum(dim=-1)
        var = (weights * x.pow(2)).sum(dim=-1) - mean.pow(2)
        return torch.cat([mean, var.clamp(min=1e-6).sqrt()], dim=-1)


class SpeakerEncoder(nn.Module):
    """``(B, frames, frame_dim)`` codec frames -> ``(B, embedding_dim)`` unit-norm speaker embeddings."""

    def __init__(
        self, frame_dim: int = 256, channels: int = 128, embedding_dim: int = 192, blocks: tuple[int, ...] = (2, 3, 4)
    ) -> None:
        super().__init__()
        self.input = nn.Conv1d(frame_dim, channels, 5, padding=2)
        self.blocks = nn.ModuleList(TDNNBlock(channels, d) for d in blocks)
        self.merge = nn.Conv1d(channels * len(blocks), channels, 1)
        self.pool = AttentiveStatisticsPooling(channels)
        self.out = nn.Linear(2 * channels, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input(frames.transpose(1, 2)))
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        x = F.relu(self.merge(torch.cat(outs, dim=1)))
        return F.normalize(self.out(self.pool(x)), dim=-1)


def speaker_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between embeddings (SIM-o style; both inputs already unit-norm)."""
    return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1)


class AAMSoftmax(nn.Module):
    """Additive angular margin softmax (ArcFace) over ``num_speakers`` classes for training the encoder."""

    def __init__(self, embedding_dim: int, num_speakers: int, margin: float = 0.2, scale: float = 30.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_speakers, embedding_dim) * 0.01)
        self.margin, self.scale = margin, scale

    def forward(self, embeddings: torch.Tensor, speakers: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings, dim=-1), F.normalize(self.weight, dim=-1)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        target = torch.cos(theta + self.margin)
        one_hot = F.one_hot(speakers, cosine.shape[-1]).bool()
        logits = torch.where(one_hot, target, cosine) * self.scale
        return F.cross_entropy(logits, speakers)


def aam_softmax_loss(head: AAMSoftmax, embeddings: torch.Tensor, speakers: torch.Tensor) -> torch.Tensor:
    return head(embeddings, speakers)


__all__ = ["AAMSoftmax", "SpeakerEncoder", "aam_softmax_loss", "speaker_similarity", "math"]
