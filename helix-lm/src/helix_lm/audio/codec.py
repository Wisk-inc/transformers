# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The codec: encoder frames -> split residual vector quantization -> decoder, and the losses that train it.

The quantizer follows Mimi's split (Moshi, 2024): the first codebook is the **semantic** one, trained to agree
with a pretrained speech model's features when a teacher is given (``semantic_target``), so it carries *what was
said*; the other seven are a plain residual VQ stack that carries *how it sounded*. Each codebook has 2048
entries of 8 factorized dimensions (DAC's trick: codes are looked up in a low-dimensional, L2-normalized space
and projected up, which keeps every entry in use). Codebooks learn by exponential moving average with dead-code
restarts; the encoder learns through the straight-through estimator and a commitment loss.

Bit rate: 8 codebooks x 11 bits x 25 Hz = 2.2 kbit/s, Higgs-Audio-class. ``quantizer_dropout`` trains the decoder
to work from any prefix of codebooks, so a language model can emit the semantic token first and the acoustic
ones later (the delay pattern in :mod:`omni`).

Losses (:func:`codec_losses`), with the weights DAC/BigCodec converged on: multi-scale log-mel L1 (x15), a
time-domain L1 (x1, early training only), commitment (x0.25) and codebook (x1); the adversarial and
feature-matching terms come from :class:`MultiPeriodDiscriminator` when one is passed. At toy scale the test
overfits a few seconds of a sine-and-noise signal and checks the spectral loss drops and every code is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .frontend import WaveDecoder, WaveEncoder


@dataclass
class CodecConfig:
    sample_rate: int = 24000
    dim: int = 256
    channels: int = 32
    strides: tuple[int, ...] = (4, 5, 6, 8)
    num_codebooks: int = 8
    codebook_size: int = 2048
    code_dim: int = 8
    quantizer_dropout: float = 0.5
    ema_decay: float = 0.99
    dead_code_threshold: float = 1.0  # restart a code whose EMA usage falls below this many hits
    semantic_dim: int = 0  # width of the teacher features for the semantic codebook; 0 = no distillation
    continuous_passthrough: float = (
        0.25  # train the decoder on unquantized frames this often (for flow-matching inputs)
    )
    mel_scales: tuple[int, ...] = field(default_factory=lambda: (32, 64, 128, 256, 512, 1024, 2048))

    @property
    def frame_rate(self) -> float:
        hop = 1
        for s in self.strides:
            hop *= s
        return self.sample_rate / hop


class VectorQuantizer(nn.Module):
    """One codebook in a factorized, normalized code space with EMA updates and dead-code restarts."""

    def __init__(self, dim: int, code_dim: int, size: int, decay: float, dead_threshold: float) -> None:
        super().__init__()
        self.down = nn.Linear(dim, code_dim, bias=False)
        self.up = nn.Linear(code_dim, dim, bias=False)
        self.register_buffer("codebook", F.normalize(torch.randn(size, code_dim), dim=-1))
        self.register_buffer("ema_size", torch.ones(size))
        self.register_buffer("ema_sum", self.codebook.clone())
        self.decay, self.dead_threshold = decay, dead_threshold

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x (B, T, dim)`` -> ``(quantized (B, T, dim), codes (B, T), commitment loss)``."""
        z = F.normalize(self.down(x), dim=-1)
        flat = z.reshape(-1, z.shape[-1])
        distances = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.codebook.T + self.codebook.pow(2).sum(1)
        codes = distances.argmin(dim=-1)
        picked = self.codebook[codes].reshape(z.shape)
        if self.training:
            self._update(flat, codes)
        commitment = F.mse_loss(z, picked.detach())
        quantized = z + (picked - z).detach()  # straight-through
        return self.up(quantized), codes.reshape(z.shape[:-1]), commitment

    @torch.no_grad()
    def _update(self, flat: torch.Tensor, codes: torch.Tensor) -> None:
        one_hot = F.one_hot(codes, self.codebook.shape[0]).to(flat.dtype)
        counts, sums = one_hot.sum(0), one_hot.T @ flat
        self.ema_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.ema_sum.mul_(self.decay).add_(sums, alpha=1 - self.decay)
        self.codebook.copy_(F.normalize(self.ema_sum / self.ema_size.clamp(min=1e-5)[:, None], dim=-1))
        dead = self.ema_size < self.dead_threshold * (1 - self.decay) * 10
        if dead.any():  # restart dead codes on random current inputs
            replacement = flat[torch.randint(0, flat.shape[0], (int(dead.sum()),), device=flat.device)]
            self.codebook[dead] = F.normalize(replacement, dim=-1)
            self.ema_sum[dead] = self.codebook[dead]
            self.ema_size[dead] = 1.0

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.up(self.codebook[codes])

    def usage(self) -> float:
        """Fraction of codes with meaningful EMA usage."""
        return float((self.ema_size > self.dead_threshold * (1 - self.decay) * 10).float().mean())


class ResidualVQ(nn.Module):
    """Codebook 0 (semantic) in parallel with a residual stack of ``num_codebooks - 1`` acoustic codebooks."""

    def __init__(self, config: CodecConfig) -> None:
        super().__init__()
        c = config
        make = lambda: VectorQuantizer(c.dim, c.code_dim, c.codebook_size, c.ema_decay, c.dead_code_threshold)  # noqa: E731
        self.semantic = make()
        self.acoustic = nn.ModuleList(make() for _ in range(c.num_codebooks - 1))
        self.semantic_head = nn.Linear(c.dim, c.semantic_dim) if c.semantic_dim else None
        self.config = config

    def forward(self, frames: torch.Tensor, num_codebooks: int | None = None):
        """
        ``frames (B, dim, T)`` -> ``(quantized (B, dim, T), codes (B, Q, T), commitment, semantic_features)``.
        ``num_codebooks`` limits the acoustic stack (quantizer dropout during training, bandwidth at inference).
        """
        x = frames.transpose(1, 2)
        q_sem, code_sem, commit = self.semantic(x)
        residual = x - q_sem.detach() * 0  # the semantic branch is parallel, not residual (Mimi's split)
        total = q_sem
        codes = [code_sem]
        limit = len(self.acoustic) if num_codebooks is None else max(0, num_codebooks - 1)
        residual = x
        for i, vq in enumerate(self.acoustic):
            if i >= limit:
                break
            q, code, c = vq(residual)
            residual = residual - q
            total = total + q
            commit = commit + c
            codes.append(code)
        semantic_features = self.semantic_head(q_sem) if self.semantic_head is not None else None
        return total.transpose(1, 2), torch.stack(codes, dim=1), commit, semantic_features

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """``codes (B, Q, T)`` -> ``frames (B, dim, T)``; any prefix of codebooks works."""
        total = self.semantic.decode(codes[:, 0])
        for i in range(1, codes.shape[1]):
            total = total + self.acoustic[i - 1].decode(codes[:, i])
        return total.transpose(1, 2)


class AudioCodec(nn.Module):
    """Encoder + split RVQ + decoder."""

    def __init__(self, config: CodecConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or CodecConfig()
        self.encoder = WaveEncoder(config.dim, config.channels, config.strides)
        self.quantizer = ResidualVQ(config)
        self.decoder = WaveDecoder(config.dim, config.channels, config.strides)

    @property
    def hop(self) -> int:
        return self.encoder.hop

    def forward(self, wave: torch.Tensor, num_codebooks: int | None = None) -> dict:
        frames = self.encoder(wave)
        dropout = self.training and num_codebooks is None and self.config.quantizer_dropout > 0
        if dropout and torch.rand(()) < self.config.quantizer_dropout:
            num_codebooks = int(torch.randint(1, self.config.num_codebooks + 1, ()))
        quantized, codes, commitment, semantic = self.quantizer(frames, num_codebooks)
        if (
            self.training
            and self.config.continuous_passthrough > 0
            and torch.rand(()) < self.config.continuous_passthrough
        ):
            quantized = frames  # the decoder also learns the continuous manifold TriStream's flow decoder lands on
        recon = self.decoder(quantized)[..., : wave.shape[-1]]
        return {"recon": recon, "codes": codes, "commitment": commitment, "frames": frames, "semantic": semantic}

    @torch.no_grad()
    def encode(self, wave: torch.Tensor, num_codebooks: int | None = None) -> torch.Tensor:
        """``(B, T)`` or ``(B, 1, T)`` waveform -> ``(B, Q, frames)`` integer codes."""
        return self.quantizer(self.encoder(wave), num_codebooks)[1]

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.quantizer.decode(codes))

    @torch.no_grad()
    def decode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Continuous frames ``(B, frames, dim)`` (a flow decoder's output) -> waveform ``(B, 1, samples)``."""
        return self.decoder(frames.transpose(1, 2))

    @torch.no_grad()
    def listen(self, wave: torch.Tensor) -> torch.Tensor:
        """Continuous frames ``(B, frames, dim)`` for a language model's input projector (no quantization)."""
        return self.encoder(wave).transpose(1, 2)


# ------------------------------------------------------------------------------------------- losses --
def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, device) -> torch.Tensor:
    """A triangular mel filterbank ``(n_mels, n_fft // 2 + 1)`` without external dependencies."""
    f_max = sample_rate / 2
    mel = lambda f: 2595.0 * torch.log10(1 + f / 700.0)  # noqa: E731
    inv = lambda m: 700.0 * (10 ** (m / 2595.0) - 1)  # noqa: E731
    points = inv(torch.linspace(0, mel(torch.tensor(f_max)), n_mels + 2, device=device))
    bins = torch.floor((n_fft + 1) * points / sample_rate).long()
    bank = torch.zeros(n_mels, n_fft // 2 + 1, device=device)
    for m in range(1, n_mels + 1):
        lo, mid, hi = bins[m - 1], bins[m], bins[m + 1]
        if mid > lo:
            bank[m - 1, lo:mid] = torch.arange(lo, mid, device=device).float().sub(lo) / (mid - lo)
        if hi > mid:
            bank[m - 1, mid:hi] = (hi - torch.arange(mid, hi, device=device).float()) / (hi - mid)
    return bank


def _magnitude(x: torch.Tensor, n_fft: int, window: torch.Tensor) -> torch.Tensor:
    return torch.stft(x, n_fft, hop_length=n_fft // 4, window=window, return_complex=True).abs()


def multiscale_mel_loss(pred: torch.Tensor, target: torch.Tensor, sample_rate: int, scales) -> torch.Tensor:
    """L1 between log-mel spectrograms at several window sizes: short ones catch transients, long ones pitch."""
    pred, target = pred.reshape(-1, pred.shape[-1]), target.reshape(-1, target.shape[-1])
    total = pred.new_zeros(())
    for n_fft in scales:
        if n_fft > pred.shape[-1]:
            continue
        window = torch.hann_window(n_fft, device=pred.device)
        n_mels = max(5, min(80, n_fft // 8))
        bank = mel_filterbank(sample_rate, n_fft, n_mels, pred.device)
        mp = (bank @ _magnitude(pred, n_fft, window)).clamp(min=1e-5).log()
        mt = (bank @ _magnitude(target, n_fft, window)).clamp(min=1e-5).log()
        total = total + F.l1_loss(mp, mt)
    return total / max(1, len(scales))


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period
        channels = (1, 32, 64, 128, 128)
        self.convs = nn.ModuleList(
            nn.Conv2d(channels[i], channels[i + 1], (5, 1), (3, 1), padding=(2, 0)) for i in range(len(channels) - 1)
        )
        self.out = nn.Conv2d(channels[-1], 1, (3, 1), padding=(1, 0))

    def forward(self, wave: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        b, c, t = wave.shape
        pad = (-t) % self.period
        x = F.pad(wave, (0, pad)).reshape(b, c, -1, self.period)
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        return self.out(x).flatten(1), features


class MultiPeriodDiscriminator(nn.Module):
    """HiFi-GAN's multi-period discriminator: the waveform folded at prime periods sees periodic structure."""

    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11)) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(PeriodDiscriminator(p) for p in periods)

    def forward(self, wave: torch.Tensor):
        return [d(wave) for d in self.discriminators]


def codec_losses(
    output: dict,
    wave: torch.Tensor,
    config: CodecConfig,
    discriminator: MultiPeriodDiscriminator | None = None,
    semantic_target: torch.Tensor | None = None,
    time_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """The generator's losses, weighted; ``total`` is what to backpropagate. The discriminator's own loss is
    :func:`discriminator_loss`."""
    recon = output["recon"]
    wave = wave.reshape(recon.shape)
    terms = {
        "mel": 15.0 * multiscale_mel_loss(recon, wave, config.sample_rate, config.mel_scales),
        "time": time_weight * F.l1_loss(recon, wave),
        "commitment": 0.25 * output["commitment"],
    }
    if discriminator is not None:
        fake, real = discriminator(recon), discriminator(wave)
        adv = sum(((1 - f) ** 2).mean() for f, _ in fake) / len(fake)
        fm = sum(
            F.l1_loss(ff, rf.detach())
            for (_, ffs), (_, rfs) in zip(fake, real, strict=True)
            for ff, rf in zip(ffs, rfs, strict=True)
        )
        terms["adversarial"], terms["feature_matching"] = 1.0 * adv, 2.0 * fm / len(fake)
    if semantic_target is not None and output.get("semantic") is not None:
        sem = output["semantic"]
        target = F.interpolate(semantic_target.transpose(1, 2), size=sem.shape[1], mode="linear").transpose(1, 2)
        terms["semantic"] = 1.0 * (1 - F.cosine_similarity(sem, target, dim=-1)).mean()
    terms["total"] = sum(terms.values())
    return terms


def discriminator_loss(
    discriminator: MultiPeriodDiscriminator, recon: torch.Tensor, wave: torch.Tensor
) -> torch.Tensor:
    fake, real = discriminator(recon.detach()), discriminator(wave.reshape(recon.shape))
    return sum(((1 - r) ** 2).mean() + (f**2).mean() for (f, _), (r, _) in zip(fake, real, strict=True)) / len(fake)
