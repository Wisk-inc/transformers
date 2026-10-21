# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
TriStream-2: a singing voice as the body makes it, on the 2.0 stack.

TriStream-1 (CorxLabs, 300M) built the source-filter theory of voice into the wiring: three encoder streams that
cannot exchange information until a late fusion trunk -- the **glottal source** (pitch and nothing else), the
**vocal-tract filter** (who is singing and which vowel) and **turbulent noise** (breath and fricatives) -- a lyric
encoder, a fusion trunk, and a rectified-flow decoder to a mel spectrogram, with a speaker encoder for zero-shot
cloning and an adversarial singer head against memorisation. TriStream-2 keeps every one of those decisions and
changes what they are made of:

* the streams read richer signals: the source stream gets log-F0, voicing, energy, breath and vibrato; the
  filter stream gets the codec's continuous frames (or its semantic codes for conversion without lyrics), the
  speaker embedding, the control tokens (emotion, technique, register, accent, language) and the conductor's
  sliders, plus cross-attention to the lyrics; the noise stream gets the high-band residual of the frames;
* the fusion trunk is an **Anansi** body (hyper-connected streams, a looped core), so the depth spent fusing is a
  dial;
* the decoder is a rectified-flow transformer over the **codec's continuous 25 Hz frames**, decoded to 24 kHz by
  the codec itself (no separate vocoder, no mel), with a source-filter **sketch** (harmonics at the F0 under the
  filter stream's envelope plus the noise stream's band energies) fed to the decoder as a second conditioning
  path, so pitch is exact and breath is placed where the source says;
* two adversarial heads: singer (as before) and **accent** on the content, so an accent is a chosen tag, never an
  inherited one;
* editing is the training task (masked spans reconstructed from context, lyrics and controls); generation is
  the fully-masked case; conversion is editing with the identity swapped; duets are two passes with two speaker
  embeddings on the extracted stems.

Named size ``ts-765m`` is 765M parameters at width 1280 (see :func:`describe_tristream_sizes`); ``ts-tiny`` runs
the tests on a CPU. Nothing here has been trained at scale; ``docs/TRISTREAM.md`` has the plan and the evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..anansi import AnansiConfig, AnansiModel
from ..audio.singing import PITCH_BINS, TECHNIQUES, hz_to_bins
from .regularize import ACCENTS, EMOTIONS, AccentAdversary

REGISTERS = ("chest", "mixed", "head", "falsetto", "whistle", "fry")
CONTROL_SLIDERS = (
    "breathiness",
    "whisper",
    "power",
    "tension",
    "vibrato",
    "warmth",
    "echo",
    "room",
    "volume",
    "brightness",
)


@dataclass
class TriStreamConfig:
    hidden: int = 1280
    heads: int = 10
    mlp: int = 3816
    source_layers: int = 4
    filter_layers: int = 8
    noise_layers: int = 3
    lyric_layers: int = 4
    trunk_prelude: int = 2
    trunk_core: int = 8
    trunk_coda: int = 2
    trunk_loops: float = 1.5
    trunk_streams: int = 2
    decoder_layers: int = 6
    frame_dim: int = 256
    frame_rate: float = 25.0
    sample_rate: int = 24000
    num_phonemes: int = 512
    num_singers: int = 4096
    semantic_codebook: int = 2048
    speaker_dim: int = 192
    flow_steps: int = 16
    mask_span: tuple[float, float] = (0.1, 0.5)
    adversary_scale: float = 0.1
    dropout: float = 0.0
    # hardening (docs/TRISTREAM.md section 12)
    subframes: int = 4  # F0 read 4 times inside each 40 ms frame (100 Hz), so runs and vibrato are not averaged away
    lookahead_frames: int = 8  # the trunk sees the control streams one block ahead: the index is early, not stale
    event_dim: int = 4  # onset strength, beat phase, downbeat, section change (from the song analysis)
    monotonic_sigma: float = 4.0  # frames; width of the alignment prior on lyric attention (0 = off)
    cfg_dropout: float = 0.1  # classifier-free guidance: drop the speaker and the sliders this often in training
    formant_invariance_weight: float = 0.05  # the envelope must not move when the pitch is shifted (0 = off)
    sustain_sigma_gain: float = 3.0  # the alignment prior widens up to (1 + gain)x on sustained notes (melismas)
    formant_smoothness_weight: float = 0.02  # the envelope may not jump between frames of a held pitch
    transition_lookahead_blocks: int = 2  # extra look-ahead, gated by upcoming downbeats and section changes
    name: str = "tristream"

    @property
    def source_dim(self) -> int:
        return 7 + self.subframes

    @classmethod
    def named(cls, size: str, **overrides) -> TriStreamConfig:
        rows = {
            "ts-tiny": {
                "hidden": 64,
                "heads": 4,
                "mlp": 160,
                "source_layers": 1,
                "filter_layers": 2,
                "noise_layers": 1,
                "lyric_layers": 1,
                "trunk_prelude": 1,
                "trunk_core": 1,
                "trunk_coda": 1,
                "trunk_loops": 1.0,
                "trunk_streams": 1,
                "decoder_layers": 2,
                "frame_dim": 32,
                "num_phonemes": 64,
                "num_singers": 16,
                "semantic_codebook": 64,
                "speaker_dim": 16,
            },
            "ts-300m": {
                "hidden": 768,
                "heads": 12,
                "mlp": 2048,
                "source_layers": 3,
                "filter_layers": 5,
                "noise_layers": 2,
                "lyric_layers": 4,
                "trunk_prelude": 2,
                "trunk_core": 6,
                "trunk_coda": 2,
                "decoder_layers": 5,
            },
            "ts-765m": {},
        }
        if size not in rows:
            raise ValueError(f"unknown TriStream size {size!r}; choose from {', '.join(rows)}")
        return cls(**{**rows[size], "name": size, **overrides})

    def anansi(self) -> AnansiConfig:
        """The fusion trunk's configuration (no embeddings of its own: it is fed vectors)."""
        return AnansiConfig(
            vocab_size=8, hidden_size=self.hidden, intermediate_size=self.mlp, num_attention_heads=self.heads,
            num_key_value_heads=max(1, self.heads // 2), head_dim=self.hidden // self.heads,
            num_recurrent_heads=max(1, self.heads // 2), prelude_layers=self.trunk_prelude,
            core_layers=self.trunk_core, coda_layers=self.trunk_coda, mean_loops=self.trunk_loops,
            min_loops=1, max_loops=max(2, math.ceil(self.trunk_loops) + 1), num_streams=self.trunk_streams,
            ngram_memory_size=0, mtp_depth=0, loss_chunk_tokens=0, block_size=8, local_blocks=4,
            index_branching=4, index_topk=4, landmark_dim=32, attention_backend="sdpa",
        )  # fmt: skip


# ------------------------------------------------------------------------------------------- blocks --
class Encoder(nn.Module):
    """A small bidirectional transformer (pre-norm, GELU) for an analysis stream."""

    def __init__(self, dim: int, heads: int, mlp: int, layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            dim, heads, mlp, dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.layers = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.position = nn.Parameter(torch.zeros(1, 4096, dim))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.position[:, : x.shape[1]]
        return self.norm(self.layers(x, src_key_padding_mask=pad_mask))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        key_pad: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = None
        if bias is not None:  # (B, T, L) additive -> (B * heads, T, L); the padding mask joins it as -inf
            mask = bias
            if key_pad is not None:
                mask = mask.masked_fill(key_pad[:, None, :], float("-inf"))
                key_pad = None
            mask = mask.repeat_interleave(self.attn.num_heads, dim=0).to(queries.dtype)
        out, _ = self.attn(
            self.norm_q(queries), self.norm_kv(keys), self.norm_kv(keys), key_padding_mask=key_pad, attn_mask=mask
        )
        return queries + out


def alignment_bias(durations: torch.Tensor, frames: int, sigma: float | torch.Tensor) -> torch.Tensor:
    """
    The monotonic prior on lyric attention: phoneme ``j`` is expected around the centre of its span given the
    durations (frames per phoneme, ``(B, L)``); the bias is a Gaussian in frame distance, ``(B, T, L)``. With
    ground-truth durations in training and predicted ones at inference, a frame can only attend near the
    phoneme that is sung there, so the lyric stream cannot slip against the pitch stream during a run.
    ``sigma`` may be a per-frame ``(B, T)`` tensor: wide where a note is held (a melisma stretches one vowel
    over seconds and the prior must not hurry it), narrow where syllables move.
    """
    ends = durations.float().clamp(min=0).cumsum(-1)
    centres = ends - durations.float().clamp(min=0) / 2  # (B, L)
    pos = torch.arange(frames, device=durations.device).float()[None, :, None]
    if torch.is_tensor(sigma):
        sigma = sigma.float()[:, :, None]
    return -((pos - centres[:, None, :]) ** 2) / (2 * sigma**2)


def sustain_profile(source: torch.Tensor, threshold: float = 0.015, horizon: int = 25) -> torch.Tensor:
    """
    How long the current note has been held, per frame, in [0, 1]: voiced frames whose log-F0 moved less than
    about half a semitone since the previous frame extend a run; the run length over ``horizon`` frames (one
    second at 25 Hz) is the profile. Channel 0 of the source stream is (log f0 - 5) / 2, channel 1 voicing.
    """
    log_f0, voiced = source[..., 0], source[..., 1] > 0.5
    steady = voiced.clone()
    steady[:, 1:] &= voiced[:, :-1] & ((log_f0[:, 1:] - log_f0[:, :-1]).abs() < threshold)
    run = torch.zeros_like(log_f0)
    for t in range(1, source.shape[1]):
        run[:, t] = (run[:, t - 1] + 1) * steady[:, t].float()
    return (run / horizon).clamp(max=1.0)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None


def shift_ahead(x: torch.Tensor, frames: int) -> torch.Tensor:
    """``x[:, t]`` becomes ``x[:, t + frames]`` (the last frame is held at the end): the future, for a model
    that generates offline and has no reason to pretend it cannot see it."""
    t = x.shape[1]
    if frames <= 0 or t == 0:
        return x
    tail = x[:, -1:].expand(-1, min(frames, t), *x.shape[2:])
    return torch.cat([x[:, frames:], tail], dim=1)[:, :t]


def sway_schedule(steps: int, strength: float = -1.0, device=None) -> torch.Tensor:
    """F5-TTS's sway sampling: ``t + s (cos(pi t / 2) - 1 + t)``; ``s < 0`` spends more steps near ``t = 0``, where
    the flow decides the coarse structure. Returns ``steps + 1`` times from 0 to 1."""
    t = torch.linspace(0, 1, steps + 1, device=device)
    return t + strength * (torch.cos(math.pi / 2 * t) - 1 + t)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


# ----------------------------------------------------------------------------------------- sketch --
def source_filter_sketch(
    f0_hz: torch.Tensor, envelope: torch.Tensor, noise_bands: torch.Tensor, voicing: torch.Tensor
) -> torch.Tensor:
    """
    The physical sketch the decoder is conditioned on: for each frame, the harmonic comb at the F0 (band energies of
    the first harmonics placed into ``envelope``'s bands) scaled by the filter stream's envelope, plus the noise
    stream's band energies where unvoiced or breathy. All tensors ``(B, T, bands)`` except ``f0_hz``/``voicing``
    ``(B, T)``. Returns ``(B, T, bands)`` log-energies.
    """
    b, t, bands = envelope.shape
    band_centres = torch.logspace(math.log10(60.0), math.log10(8000.0), bands, device=f0_hz.device)
    harmonics = torch.arange(1, 41, device=f0_hz.device).float()
    freqs = f0_hz.clamp(min=1.0)[..., None] * harmonics  # (B, T, 40)
    distance = (torch.log2(freqs[..., None] / band_centres) * 12).abs()  # semitones from each band centre
    comb = torch.exp(-0.5 * (distance / 1.5) ** 2).sum(-2) / harmonics.numel()  # (B, T, bands)
    comb = comb / (comb.amax(-1, keepdim=True) + 1e-6)
    harmonic_part = voicing[..., None] * comb * envelope.exp()
    noisy_part = (1 - voicing[..., None] * 0.7) * noise_bands.exp()
    return (harmonic_part + noisy_part + 1e-6).log()


# ------------------------------------------------------------------------------------------ model --
class TriStreamForSinging(nn.Module):
    def __init__(self, config: TriStreamConfig | None = None) -> None:
        super().__init__()
        self.config = c = config or TriStreamConfig()
        d = c.hidden
        bands = 32
        self.bands = bands
        # --- source stream: pitch and nothing else (no path from identity to here)
        # log-f0, voiced, energy, breath, vibrato depth, vibrato rate, f0 delta, + sub-frame f0 offsets
        self.source_in = nn.Linear(c.source_dim, d)
        self.source = Encoder(d, c.heads, c.mlp, c.source_layers, c.dropout)
        # --- filter stream: content, identity, controls
        self.frame_in = nn.Linear(c.frame_dim, d)
        self.semantic_embed = nn.Embedding(c.semantic_codebook + 1, d)  # + "none"
        self.phoneme_frame_embed = nn.Embedding(c.num_phonemes + 1, d)
        self.speaker_in = nn.Linear(c.speaker_dim, d)
        self.emotion = nn.Embedding(len(EMOTIONS), d)
        self.technique = nn.Embedding(len(TECHNIQUES), d)
        self.register = nn.Embedding(len(REGISTERS), d)
        self.accent = nn.Embedding(len(ACCENTS), d)
        self.slider_in = nn.Linear(len(CONTROL_SLIDERS), d)
        self.quality_in = nn.Linear(1, d)  # the intake's 0..1 input quality: "this input is dirty" as a condition
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.filter = Encoder(d, c.heads, c.mlp, c.filter_layers, c.dropout)
        self.lyric_cross = CrossAttention(d, c.heads)
        # --- noise stream
        self.noise_in = nn.Linear(bands + 2, d)  # high-band residual energies, breath, flatness
        self.noise = Encoder(d, c.heads, c.mlp, c.noise_layers, c.dropout)
        # --- lyric encoder
        self.phoneme_embed = nn.Embedding(c.num_phonemes + 1, d)
        self.lyrics = Encoder(d, c.heads, c.mlp, c.lyric_layers, c.dropout)
        # --- fusion trunk (Anansi) and its cross-attention to the lyrics
        self.fuse = nn.Linear(3 * d, d)
        self.ahead = nn.Linear(2 * d, d)  # the source and noise streams one block ahead
        self.ahead2 = nn.Linear(2 * d, d)  # two blocks ahead, switched on by an upcoming downbeat or section change
        self.event_in = nn.Linear(c.event_dim, d)
        self.null_speaker = nn.Parameter(torch.zeros(c.speaker_dim))  # the "no voice given" condition for guidance
        self.trunk = AnansiModel(c.anansi())
        self.fusion_cross = CrossAttention(d, c.heads)
        # --- predictors for generation: pitch per frame, duration per phoneme
        self.pitch_head = nn.Linear(d, PITCH_BINS + 1 + 3)  # bins + unvoiced, fine offset, vibrato depth, rate
        self.duration_head = nn.Linear(d, 1)
        self.envelope_head = nn.Linear(d, bands)
        self.noise_band_head = nn.Linear(d, bands)
        # --- rectified-flow decoder over codec frames
        self.flow_in = nn.Linear(c.frame_dim + d + bands, d)
        self.time_in = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.decoder = Encoder(d, c.heads, c.mlp, c.decoder_layers, c.dropout)
        self.flow_out = nn.Linear(d, c.frame_dim)
        # --- adversaries
        self.singer_head = nn.Linear(d, c.num_singers)
        self.accent_adversary = AccentAdversary(d, len(ACCENTS), c.adversary_scale)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    # ---------------------------------------------------------------------------------- streams --
    def encode_source(self, source: torch.Tensor) -> torch.Tensor:
        return self.source(self.source_in(source))

    def encode_filter(
        self, batch: dict, lyric_states: torch.Tensor, lyric_pad: torch.Tensor | None, mask: torch.Tensor | None
    ) -> torch.Tensor:
        c = self.config
        frames = batch.get("frames")
        b, t = batch["source"].shape[:2]
        x = torch.zeros(b, t, c.hidden, device=batch["source"].device)
        if frames is not None:
            fx = self.frame_in(frames)
            if mask is not None:
                fx = torch.where(mask[..., None], self.mask_token.expand_as(fx), fx)
            x = x + fx
        if batch.get("semantic") is not None:
            x = x + self.semantic_embed(batch["semantic"].clamp(min=0))
        if batch.get("phoneme_frames") is not None:
            x = x + self.phoneme_frame_embed(batch["phoneme_frames"].clamp(min=0))
        speaker = batch["speaker"]
        drop = batch.get("_drop_condition")
        if drop is not None:
            speaker = torch.where(drop[:, None], self.null_speaker[None].expand_as(speaker), speaker)
        x = x + self.speaker_in(speaker)[:, None]
        for name, table in (
            ("emotion", self.emotion),
            ("technique", self.technique),
            ("register", self.register),
            ("accent", self.accent),
        ):
            ids = batch.get(name)
            if ids is not None:
                x = x + (table(ids)[:, None] if ids.dim() == 1 else table(ids))
        if batch.get("sliders") is not None:
            sliders = batch["sliders"]
            if drop is not None:
                sliders = sliders * (~drop)[:, None, None]
            x = x + self.slider_in(sliders)
        if batch.get("quality") is not None:
            x = x + self.quality_in(batch["quality"].float().reshape(-1, 1))[:, None]
        x = self.lyric_cross(x, lyric_states, lyric_pad, batch.get("_alignment_bias"))
        return self.filter(x)

    def lyric_alignment(self, batch: dict, lyric_states: torch.Tensor, lyric_pad: torch.Tensor) -> torch.Tensor | None:
        """The alignment bias from the batch's durations, or from the duration head when none are given."""
        c = self.config
        if c.monotonic_sigma <= 0 or lyric_states.shape[1] < 2:
            return None
        t = batch["source"].shape[1]
        durations = batch.get("durations")
        if durations is None:
            durations = torch.expm1(self.duration_head(lyric_states)[..., 0].detach()).clamp(min=0) * (~lyric_pad)
            durations = durations / durations.sum(-1, keepdim=True).clamp(min=1) * t  # fit the predicted total to T
        sigma = c.monotonic_sigma * (1.0 + c.sustain_sigma_gain * sustain_profile(batch["source"]))
        return alignment_bias(durations, t, sigma)

    def encode_noise(self, noise: torch.Tensor) -> torch.Tensor:
        return self.noise(self.noise_in(noise))

    def encode_lyrics(self, phonemes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad = phonemes < 0
        ids = phonemes.clamp(min=0)
        empty = pad.all(-1)  # no lyrics at all (conversion without a transcript): one "none" token, never all-pad
        if empty.any():
            pad = pad.clone()
            ids = ids.clone()
            pad[empty, 0] = False
            ids[empty, 0] = self.config.num_phonemes
        states = self.lyrics(self.phoneme_embed(ids), pad)
        return states, pad

    def fuse_streams(
        self,
        source: torch.Tensor,
        filt: torch.Tensor,
        noise: torch.Tensor,
        lyric_states: torch.Tensor,
        lyric_pad: torch.Tensor | None,
        num_loops: int | None = None,
        events: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        fused = self.fuse(torch.cat([source, filt, noise], dim=-1))
        if c.lookahead_frames > 0:
            # what the pitch and the breath will do one block from now, so the index strand's choice at this
            # block boundary already knows about the drop, the pivot, the held top note
            fused = fused + self.ahead(shift_ahead(torch.cat([source, noise], dim=-1), c.lookahead_frames))
        if events is not None:
            ev = self.event_in(events)
            fused = fused + ev + (shift_ahead(ev, c.lookahead_frames) if c.lookahead_frames > 0 else 0)
            if c.transition_lookahead_blocks > 0 and c.lookahead_frames > 0:
                # a downbeat or a section change inside the next blocks switches on a longer look-ahead, so the
                # index decision before a drop is made with the drop itself in hand, not the intro's context
                span = c.transition_lookahead_blocks * c.lookahead_frames
                flag = events[..., 1:3].amax(-1)  # (B, T): downbeat, section change
                upcoming = F.max_pool1d(F.pad(flag[:, None], (0, span)), span + 1, stride=1)[:, 0, : flag.shape[1]]
                far = shift_ahead(torch.cat([source, noise], dim=-1), span)
                fused = fused + upcoming[..., None] * self.ahead2(far)
        hidden, _, _ = self.trunk(inputs_embeds=fused, num_loops=num_loops)
        return self.fusion_cross(hidden, lyric_states, lyric_pad, bias)

    # --------------------------------------------------------------------------------- the flow --
    def velocity(
        self, noisy: torch.Tensor, t: torch.Tensor, hidden: torch.Tensor, sketch: torch.Tensor
    ) -> torch.Tensor:
        x = (
            self.flow_in(torch.cat([noisy, hidden, sketch], dim=-1))
            + self.time_in(timestep_embedding(t, self.config.hidden))[:, None]
        )
        return self.flow_out(self.decoder(x))

    def sketch(
        self, hidden: torch.Tensor, source: torch.Tensor, tract_ratio: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        envelope = self.envelope_head(hidden)
        if tract_ratio is not None:  # the register plan's bounded formant move (music.register)
            from .register import warp_envelope

            envelope = warp_envelope(envelope, tract_ratio)
        noise_bands = self.noise_band_head(hidden)
        f0 = torch.exp(source[..., 0] * 2.0 + 5.0)  # the source stream's first channel is (log f0 - 5) / 2
        voicing = source[..., 1]
        return source_filter_sketch(f0, envelope, noise_bands, voicing), envelope, noise_bands

    # ----------------------------------------------------------------------------------- losses --
    def forward(self, batch: dict, num_loops: int | None = None) -> dict:
        """
        Training step on a batch of aligned 25 Hz records:
        ``source (B, T, 7)``, ``noise (B, T, bands + 2)``, ``frames (B, T, frame_dim)`` (target and input),
        ``phonemes (B, L)`` (-1 padded), ``speaker (B, speaker_dim)``, ``singer (B,)``, ``accent (B,)``, optional
        ``emotion``/``technique``/``register`` ids, ``sliders (B, T, 10)``, ``phoneme_frames (B, T)``,
        ``durations (B, L)`` (frames per phoneme), ``f0_hz (B, T)``. A random span is masked (editing); with
        ``batch["mask_all"]`` the whole target is (generation).
        """
        c = self.config
        frames = batch["frames"]
        target = batch.get("target_frames", frames)  # the input may be a roomy, noisy copy; the target stays dry
        b, t, _ = frames.shape
        if batch.get("mask_all"):
            mask = torch.ones(b, t, dtype=torch.bool, device=frames.device)
        else:
            mask = torch.zeros(b, t, dtype=torch.bool, device=frames.device)
            for i in range(b):
                span = int(t * (c.mask_span[0] + (c.mask_span[1] - c.mask_span[0]) * torch.rand(()).item()))
                start = int(torch.randint(0, max(1, t - span), ()).item())
                mask[i, start : start + span] = True
        batch = dict(batch)
        if self.training and c.cfg_dropout > 0:
            batch["_drop_condition"] = torch.rand(b, device=frames.device) < c.cfg_dropout
        lyric_states, lyric_pad = self.encode_lyrics(batch["phonemes"])
        bias = self.lyric_alignment(batch, lyric_states, lyric_pad)
        batch["_alignment_bias"] = bias
        source = self.encode_source(batch["source"])
        filt = self.encode_filter(batch, lyric_states, lyric_pad, mask)
        noise = self.encode_noise(batch["noise"])
        events = batch.get("events")
        hidden = self.fuse_streams(source, filt, noise, lyric_states, lyric_pad, num_loops, events, bias)
        sketch, envelope, noise_bands = self.sketch(hidden, batch["source"], batch.get("tract_ratio"))

        # rectified flow on the masked frames
        z = torch.randn_like(target)
        tt = torch.rand(b, device=frames.device)
        noisy = (1 - tt[:, None, None]) * z + tt[:, None, None] * target
        v = self.velocity(noisy, tt, hidden, sketch)
        flow = ((v - (target - z)) ** 2).mean(-1)
        # per-sample loss weights: a sung clip may count for more than a spoken one (domain balance, data.py)
        sample_weight = batch.get("weights")
        sample_weight = (
            torch.ones(b, device=frames.device) if sample_weight is None else sample_weight.to(frames.device).float()
        )
        weighted_mask = mask.float() * sample_weight[:, None]
        losses = {"flow": (flow * weighted_mask).sum() / weighted_mask.sum().clamp(min=1e-6)}

        # predictors (pitch per frame from the fused state, durations per phoneme from the lyric states)
        if batch.get("f0_hz") is not None:
            pitch = self.pitch_head(hidden)
            bins = hz_to_bins(batch["f0_hz"])
            per_frame = F.cross_entropy(
                pitch[..., : PITCH_BINS + 1].reshape(-1, PITCH_BINS + 1), bins.reshape(-1), reduction="none"
            ).reshape(b, t)
            losses["pitch"] = (per_frame * sample_weight[:, None]).sum() / (sample_weight.sum() * t).clamp(min=1e-6)
        if batch.get("durations") is not None:
            pred = self.duration_head(lyric_states)[..., 0]
            target = torch.log1p(batch["durations"].float())
            keep = ~lyric_pad
            losses["duration"] = ((pred - target) ** 2 * keep).sum() / keep.sum().clamp(min=1)
        # adversaries: the trunk must not name the singer; the content must not name the accent
        if batch.get("singer") is not None:
            pooled = GradientReversal.apply(hidden.mean(1), c.adversary_scale)
            losses["singer_adv"] = F.cross_entropy(self.singer_head(pooled), batch["singer"])
        if batch.get("accent") is not None:
            losses["accent_adv"] = self.accent_adversary(filt, batch["accent"])
        # formant / pitch decoupling: shift the source stream by up to seven semitones on half the batch; the
        # envelope the filter path produces must not move (a chipmunk or a robot is an envelope that followed F0)
        if c.formant_invariance_weight > 0 and b >= 2:
            half = b // 2
            semitones = torch.rand(half, device=frames.device) * 14 - 7
            shifted = batch["source"][:half].clone()
            shifted[..., 0] = shifted[..., 0] + semitones[:, None] * math.log(2) / 12 / 2 * shifted[..., 1]
            source_s = self.encode_source(shifted)
            hidden_s = self.fuse_streams(
                source_s, filt[:half], noise[:half], lyric_states[:half], lyric_pad[:half], num_loops,
                None if events is None else events[:half], None if bias is None else bias[:half],
            )  # fmt: skip
            losses["formant_invariance"] = (self.envelope_head(hidden_s) - envelope[:half].detach()).abs().mean()
        # formant smoothness: on a held pitch the envelope must not jump between frames (the "formant wall"
        # click at a register break is exactly such a jump)
        if c.formant_smoothness_weight > 0 and t > 1:
            src = batch["source"]
            held = (src[:, 1:, 1] > 0.5) & (src[:, :-1, 1] > 0.5) & ((src[:, 1:, 0] - src[:, :-1, 0]).abs() < 0.03)
            jumps = (envelope[:, 1:] - envelope[:, :-1]).abs().mean(-1)
            losses["formant_smoothness"] = (jumps * held.float()).sum() / held.float().sum().clamp(min=1)
        losses["total"] = (
            losses["flow"]
            + 0.3 * losses.get("pitch", 0)
            + 0.1 * losses.get("duration", 0)
            + 0.1 * losses.get("singer_adv", 0)
            + 0.1 * losses.get("accent_adv", 0)
            + c.formant_invariance_weight * losses.get("formant_invariance", 0)
            + c.formant_smoothness_weight * losses.get("formant_smoothness", 0)
        )
        losses["mask_fraction"] = mask.float().mean()
        return losses

    # -------------------------------------------------------------------------------- sampling --
    @torch.no_grad()
    def sample(
        self,
        batch: dict,
        steps: int | None = None,
        num_loops: int | None = None,
        mask: torch.Tensor | None = None,
        guidance: float = 1.0,
        sway: float = -1.0,
        adaptive: bool = False,
        tolerance: float = 0.02,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Frames for the masked positions (all positions when ``mask`` is None) by Euler integration of the flow
        from noise to data; unmasked positions keep the input frames. ``guidance > 1`` applies classifier-free
        guidance against the no-voice / no-sliders condition (sharper identity and control), ``sway`` bends the
        time grid towards the start (F5-TTS), ``adaptive`` stops early once the velocity field has settled and
        records the count in ``self.last_steps``. Returns ``(B, T, frame_dim)``.
        """
        self.eval()
        c = self.config
        steps = steps or c.flow_steps
        b, t = batch["source"].shape[:2]
        device = batch["source"].device
        frames_in = batch.get("frames")
        if mask is None:
            mask = torch.ones(b, t, dtype=torch.bool, device=device)
        batch = dict(batch)
        lyric_states, lyric_pad = self.encode_lyrics(batch["phonemes"])
        bias = self.lyric_alignment(batch, lyric_states, lyric_pad)
        batch["_alignment_bias"] = bias
        source = self.encode_source(batch["source"])
        noise = self.encode_noise(batch["noise"])
        events = batch.get("events")
        filt = self.encode_filter(batch, lyric_states, lyric_pad, mask)
        hidden = self.fuse_streams(source, filt, noise, lyric_states, lyric_pad, num_loops, events, bias)
        sketch, _, _ = self.sketch(hidden, batch["source"], batch.get("tract_ratio"))
        if guidance != 1.0:
            uncond = dict(batch, _drop_condition=torch.ones(b, dtype=torch.bool, device=device))
            filt_u = self.encode_filter(uncond, lyric_states, lyric_pad, mask)
            hidden_u = self.fuse_streams(source, filt_u, noise, lyric_states, lyric_pad, num_loops, events, bias)
            sketch_u, _, _ = self.sketch(hidden_u, batch["source"], batch.get("tract_ratio"))
        times = sway_schedule(steps, sway, device)
        x = torch.randn(b, t, c.frame_dim, device=device, generator=generator)
        previous, settled, used = None, 0, steps
        for i in range(steps):
            tt = times[i].expand(b)
            dt = float(times[i + 1] - times[i])
            v = self.velocity(x, tt, hidden, sketch)
            if guidance != 1.0:
                v = v + (guidance - 1.0) * (v - self.velocity(x, tt, hidden_u, sketch_u))
            if adaptive and previous is not None and i >= steps // 2:
                change = float((v - previous).norm() / (v.norm() + 1e-6))
                settled = settled + 1 if change < tolerance else 0
                if settled >= 2:  # the field has stopped changing: finish in one step
                    x = x + v * float(1.0 - times[i])
                    used = i + 1
                    break
            previous = v
            x = x + v * dt
        self.last_steps = used
        if frames_in is not None:
            x = torch.where(mask[..., None], x, frames_in)
        return x

    @torch.no_grad()
    def predict_pitch(self, batch: dict, num_loops: int | None = None) -> torch.Tensor:
        """F0 in Hz per frame from the controls alone (notes, lyrics, style): the melody when no curve is given."""
        from ..audio.singing import bins_to_hz

        lyric_states, lyric_pad = self.encode_lyrics(batch["phonemes"])
        source = self.encode_source(batch["source"])
        filt = self.encode_filter(
            batch,
            lyric_states,
            lyric_pad,
            torch.ones(batch["source"].shape[:2], dtype=torch.bool, device=batch["source"].device),
        )
        noise = self.encode_noise(batch["noise"])
        hidden = self.fuse_streams(source, filt, noise, lyric_states, lyric_pad, num_loops, batch.get("events"))
        bins = self.pitch_head(hidden)[..., : PITCH_BINS + 1].argmax(-1)
        return bins_to_hz(bins)

    @torch.no_grad()
    def sketch_bands(self, batch: dict, num_loops: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """The envelope and noise band log-energies ``(B, T, bands)`` the model would sketch for ``batch``: what the
        physical fallback renders when a span fails its gates."""
        batch = dict(batch)
        lyric_states, lyric_pad = self.encode_lyrics(batch["phonemes"])
        bias = self.lyric_alignment(batch, lyric_states, lyric_pad)
        batch["_alignment_bias"] = bias
        b, t = batch["source"].shape[:2]
        mask = torch.ones(b, t, dtype=torch.bool, device=batch["source"].device)
        source = self.encode_source(batch["source"])
        filt = self.encode_filter(batch, lyric_states, lyric_pad, mask)
        noise = self.encode_noise(batch["noise"])
        hidden = self.fuse_streams(source, filt, noise, lyric_states, lyric_pad, num_loops, batch.get("events"), bias)
        _, envelope, noise_bands = self.sketch(hidden, batch["source"], batch.get("tract_ratio"))
        return envelope, noise_bands

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------------------------- features --
def source_features(
    f0_hz: torch.Tensor,
    energy: torch.Tensor,
    breath: torch.Tensor | None = None,
    vibrato: tuple[torch.Tensor, torch.Tensor] | None = None,
    f0_fine: torch.Tensor | None = None,
    subframes: int = 4,
) -> torch.Tensor:
    """
    ``(B, T)`` curves -> the source stream's ``(B, T, 7 + subframes)`` input: (log f0 - 5)/2, voicing, log energy,
    breath, vibrato depth, vibrato rate, f0 delta, then the F0 at ``subframes`` points inside the frame as
    offsets from the frame's F0 (from ``f0_fine (B, T * subframes)`` when given, zeros otherwise). The offsets
    are what lets a 100 Hz run or a 6 Hz vibrato survive a 25 Hz frame.
    """
    voiced = (f0_hz > 0).float()
    log_f0 = (torch.log(f0_hz.clamp(min=1.0)) - 5.0) / 2.0 * voiced
    delta = F.pad(log_f0[:, 1:] - log_f0[:, :-1], (1, 0))
    breath = torch.zeros_like(f0_hz) if breath is None else breath.float()
    depth, rate = (torch.zeros_like(f0_hz), torch.zeros_like(f0_hz)) if vibrato is None else vibrato
    base = [log_f0, voiced, torch.log(energy.clamp(min=1e-6)) / 10 + 1, breath, depth, rate / 10, delta]
    b, t = f0_hz.shape
    if f0_fine is None or subframes <= 0:
        fine = torch.zeros(b, t, max(subframes, 0), device=f0_hz.device, dtype=f0_hz.dtype)
    else:
        ff = F.pad(f0_fine.float(), (0, max(0, t * subframes - f0_fine.shape[-1])))[:, : t * subframes]
        ff = ff.reshape(b, t, subframes)
        fine = (torch.log(ff.clamp(min=1.0)) - torch.log(f0_hz.clamp(min=1.0))[..., None]) * 2.0
        fine = fine * (ff > 0).float() * voiced[..., None]
    return torch.cat([torch.stack(base, dim=-1), fine], dim=-1)


def event_features(analysis, frames: int, frame_rate: float = 25.0) -> torch.Tensor:
    """
    ``(1, frames, 4)`` per-frame event track from a song analysis: beat phase (0 at the beat, rising to 1),
    downbeat flag, section-change flag, and a loudness surge (positive RMS step over the last bar). The trunk
    receives it for the current frame and one block ahead, so nothing in the arrangement arrives unannounced.
    """
    out = torch.zeros(1, frames, 4)
    beat_times = analysis.beat_times()
    if beat_times.numel() == 0:
        return out
    times = torch.arange(frames) / frame_rate
    idx = torch.searchsorted(beat_times, times, right=True).clamp(max=beat_times.numel() - 1) - 1
    idx = idx.clamp(min=0)
    nxt = (idx + 1).clamp(max=beat_times.numel() - 1)
    span = (beat_times[nxt] - beat_times[idx]).clamp(min=1e-3)
    out[0, :, 0] = ((times - beat_times[idx]) / span).clamp(0, 1)
    downbeats = analysis.downbeats
    if downbeats.numel() and int(downbeats.max()) >= beat_times.numel():  # frame indices, not beat indices
        downbeats = torch.searchsorted(analysis.beats.contiguous(), downbeats.contiguous())
    down = {int(d) for d in downbeats.tolist()}
    for i in range(frames):
        if int(idx[i]) in down and out[0, i, 0] < 1.0 / max(1, span[i] * frame_rate) + 1e-6:
            out[0, i, 1] = 1.0
    starts = {int(s["start_beat"]) for s in analysis.segments}
    for i in range(frames):
        if int(idx[i]) in starts and out[0, i, 0] < 1.0 / max(1, span[i] * frame_rate) + 1e-6:
            out[0, i, 2] = 1.0
    rms = analysis.rms
    if rms.numel() > 1:
        per_frame = F.interpolate(rms[None, None].float(), size=frames, mode="linear", align_corners=False)[0, 0]
        bar = max(1, int(4 * 60.0 / max(analysis.tempo, 1) * frame_rate))
        prev = F.pad(per_frame, (bar, 0))[:frames]
        out[0, :, 3] = F.relu(torch.log(per_frame + 1e-6) - torch.log(prev + 1e-6)).clamp(max=3) / 3
    return out


def noise_features(frames: torch.Tensor, breath: torch.Tensor | None = None, bands: int = 32) -> torch.Tensor:
    """
    The noise stream's input from codec frames: a fixed random projection of the frame vector into ``bands``
    (a stand-in for high-band energies until the codec's encoder exposes them), plus breath and a flatness proxy.
    """
    b, t, d = frames.shape
    g = torch.Generator(device="cpu").manual_seed(7)
    proj = torch.randn(d, bands, generator=g).to(frames.device) / math.sqrt(d)
    energies = torch.log1p((frames @ proj).abs())
    flatness = (energies.exp().log().mean(-1) - energies.exp().mean(-1).clamp(min=1e-6).log())[..., None]
    breath = torch.zeros(b, t, 1, device=frames.device) if breath is None else breath.float()[..., None]
    return torch.cat([energies, breath, flatness], dim=-1)


def describe_tristream_sizes() -> str:
    rows = [f"{'size':8} {'params':>9} {'hidden':>7} {'trunk':>12}"]
    for name in ("ts-tiny", "ts-300m", "ts-765m"):
        config = TriStreamConfig.named(name)
        with torch.device("meta"):
            model = TriStreamForSinging(config)
        trunk = f"{config.trunk_prelude}+{config.trunk_core}x{config.trunk_loops:g}+{config.trunk_coda}"
        rows.append(f"{name:8} {model.num_parameters() / 1e6:8.0f}M {config.hidden:>7} {trunk:>12}")
    return "\n".join(rows)
