# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
One sequence, two kinds of slot: the multimodal wrapper around an Anansi (or HELIX) language model.

Every position of the sequence is either a **text token** or an **audio frame** (one codec frame: ``Q`` codes
at 25 Hz). The two share the trunk; only the periphery differs, which is the design the omni models converged
on (Moshi 2024, Qwen3-Omni 2025; Meta's Mixture-of-Transformers: share the sequence mixer, separate the
embeddings and heads):

* **in**: a text slot is its token embedding; an audio slot is the sum of its ``Q`` code embeddings, plus the
  projected continuous frame when the audio is being *listened to* (understanding keeps the detail that
  quantization drops), plus any score conditioning (:mod:`singing`), plus a modality embedding;
* **out, text**: the language model's own head on text slots;
* **out, audio**: codebook 0 (the semantic one) from the trunk through an audio head; codebooks ``1..Q-1`` from a
  small **depth transformer** that reads the trunk state and the lower codebooks (CSM / Moshi's depth
  transformer), so a frame's acoustic detail is produced in ``Q-1`` cheap steps rather than ``Q-1`` trunk steps;
* **acoustic delay**: in the sequence, codebooks ``1..Q-1`` are shifted ``delay`` frames later than codebook 0
  (:func:`apply_delay`), so when the trunk emits a frame's semantics the acoustics of the *previous* frame are
  what the depth transformer renders -- Moshi's trick for streaming generation.

Zero-shot voice cloning is in-context: :meth:`OmniForCausalLM.clone_prompt` lays out ``[reference transcript]
[reference audio frames] [target text]`` and the model continues with audio in that voice; the loss is masked on
the prompt in training. A speaker embedding (:mod:`speaker`) can be added as one more conditioning vector.

Text replay matters: Moshi lost 4.6 MMLU points when audio was added without 50% text-only batches; the audio
data plan in ``docs/ANANSI.md`` keeps text at half of every batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from ..anansi import AnansiConfig, AnansiForCausalLM
from ..model import HelixRMSNorm

AUDIO_SLOT = -1  # value of text_ids at audio positions


@dataclass
class OmniConfig:
    backbone: AnansiConfig = field(default_factory=lambda: AnansiConfig.named("a-20m"))
    num_codebooks: int = 8
    codebook_size: int = 2048
    frame_dim: int = 256
    delay: int = 1
    depth_layers: int = 2
    depth_heads: int = 4
    speaker_dim: int = 0
    audio_loss_weight: float = 1.0
    depth_loss_weight: float = 1.0


# ---------------------------------------------------------------------------------- delay pattern --
def apply_delay(codes: torch.Tensor, delay: int, pad: int) -> torch.Tensor:
    """``(B, Q, T)`` -> ``(B, Q, T + delay)``: codebook 0 stays, codebooks 1.. shift right by ``delay``; ``pad``
    fills."""
    b, q, t = codes.shape
    out = codes.new_full((b, q, t + delay), pad)
    out[:, 0, :t] = codes[:, 0]
    out[:, 1:, delay:] = codes[:, 1:]
    return out


def undo_delay(delayed: torch.Tensor, delay: int) -> torch.Tensor:
    """The inverse of :func:`apply_delay` (drops the padding)."""
    t = delayed.shape[-1] - delay
    out = delayed[:, :, :t].clone()
    out[:, 1:] = delayed[:, 1:, delay:]
    return out


# ------------------------------------------------------------------------------- depth transformer --
class DepthTransformer(nn.Module):
    """A tiny causal transformer over the codebook axis: ``[trunk state, code_0, ..., code_{Q-2}]`` ->
    ``code_1..code_{Q-1}``."""

    def __init__(self, config: OmniConfig) -> None:
        super().__init__()
        h = config.backbone.hidden_size
        d = max(128, h // 4)
        self.in_proj = nn.Linear(h, d)
        self.code_embed = nn.Embedding(config.num_codebooks * config.codebook_size, d)
        self.position = nn.Parameter(torch.zeros(config.num_codebooks, d))
        layer = nn.TransformerEncoderLayer(
            d, config.depth_heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True
        )
        self.layers = nn.TransformerEncoder(layer, config.depth_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.heads = nn.ModuleList(nn.Linear(d, config.codebook_size) for _ in range(config.num_codebooks - 1))
        self.config = config
        nn.init.normal_(self.position, std=0.02)

    def forward(self, trunk: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """``trunk (N, h)``, ``codes (N, Q)`` ground truth -> logits ``(N, Q-1, codebook_size)`` for codebooks
        1..Q-1."""
        n, q = codes.shape
        offsets = torch.arange(q - 1, device=codes.device) * self.config.codebook_size
        lower = self.code_embed(codes[:, :-1] + offsets)  # embeddings of codebooks 0..Q-2
        x = torch.cat([self.in_proj(trunk)[:, None], lower], dim=1) + self.position[:q]
        mask = torch.triu(torch.full((q, q), float("-inf"), device=codes.device), diagonal=1)
        y = self.norm(self.layers(x, mask=mask))
        return torch.stack([head(y[:, i + 1]) for i, head in enumerate(self.heads[: q - 1])], dim=1)

    @torch.no_grad()
    def generate(self, trunk: torch.Tensor, first_code: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
        """Sample codebooks 1..Q-1 for one frame given the trunk state and codebook 0: ``(N, Q)`` codes."""
        codes = first_code[:, None]
        for _ in range(self.config.num_codebooks - 1):
            padded = torch.cat([codes, codes.new_zeros(codes.shape[0], 1)], dim=1)
            logits = self.forward(trunk, padded)[:, codes.shape[1] - 1]
            nxt = (
                logits.argmax(-1)
                if temperature <= 0
                else torch.multinomial((logits / temperature).softmax(-1), 1)[:, 0]
            )
            codes = torch.cat([codes, nxt[:, None]], dim=1)
        return codes


# ----------------------------------------------------------------------------------------- the model --
class OmniForCausalLM(nn.Module):
    def __init__(self, config: OmniConfig) -> None:
        super().__init__()
        self.config = config
        self.lm = AnansiForCausalLM(config.backbone)
        h = config.backbone.hidden_size
        self.audio_embed = nn.Embedding(config.num_codebooks * config.codebook_size + 1, h)  # + "empty" code
        self.frame_proj = nn.Linear(config.frame_dim, h, bias=False)
        self.modality = nn.Embedding(2, h)
        self.audio_norm = HelixRMSNorm(h)
        self.audio_head = nn.Linear(h, config.codebook_size + 1, bias=False)  # + "end of audio"
        self.depth = DepthTransformer(config)
        self.speaker_proj = nn.Linear(config.speaker_dim, h, bias=False) if config.speaker_dim else None
        for table in (self.audio_embed, self.modality):
            nn.init.normal_(table.weight, std=0.02)
        nn.init.normal_(self.audio_head.weight, std=0.02)
        nn.init.zeros_(self.frame_proj.weight)

    @property
    def empty_code(self) -> int:
        return self.config.num_codebooks * self.config.codebook_size

    @property
    def end_of_audio(self) -> int:
        return self.config.codebook_size

    # ---------------------------------------------------------------------------------- embedding --
    def embed(
        self,
        text_ids: torch.Tensor,
        audio_codes: torch.Tensor | None = None,
        audio_frames: torch.Tensor | None = None,
        conditioning: torch.Tensor | None = None,
        speaker: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        ``text_ids (B, T)`` with ``AUDIO_SLOT`` at audio positions; ``audio_codes (B, Q, T)`` (ignored at text
        positions); ``audio_frames (B, T, frame_dim)`` continuous frames for listening; ``conditioning (B, T, h)``
        from the singing conditioner; ``speaker (B, speaker_dim)``.
        """
        is_audio = text_ids == AUDIO_SLOT
        embeds = self.lm.model.embed_tokens(text_ids.clamp(min=0)) * (~is_audio)[..., None]
        if audio_codes is not None:
            q = audio_codes.shape[1]
            offsets = (torch.arange(q, device=text_ids.device) * self.config.codebook_size)[None, :, None]
            shifted = torch.where(
                audio_codes >= 0, audio_codes + offsets, torch.full_like(audio_codes, self.empty_code)
            )
            audio = self.audio_embed(shifted).sum(dim=1) * is_audio[..., None]
            embeds = embeds + audio
        if audio_frames is not None:
            embeds = embeds + self.frame_proj(audio_frames) * is_audio[..., None]
        if conditioning is not None:
            embeds = embeds + conditioning * is_audio[..., None]
        if speaker is not None and self.speaker_proj is not None:
            embeds = embeds + self.speaker_proj(speaker)[:, None]
        return embeds + self.modality(is_audio.long())

    # ---------------------------------------------------------------------------------- training --
    def forward(
        self,
        text_ids: torch.Tensor,
        audio_codes: torch.Tensor | None = None,
        audio_frames: torch.Tensor | None = None,
        conditioning: torch.Tensor | None = None,
        speaker: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        num_loops: int | None = None,
    ) -> dict:
        """
        Teacher-forced losses on a mixed sequence. ``audio_codes`` must already carry the acoustic delay
        (:func:`apply_delay`). ``loss_mask (B, T)`` False where no loss is taken (the cloning prompt).
        Returns text, audio (codebook 0) and depth (codebooks 1..) losses and their weighted ``loss``.
        """
        is_audio = text_ids == AUDIO_SLOT
        embeds = self.embed(text_ids, audio_codes, audio_frames, conditioning, speaker)
        hidden, _, loops = self.lm.model(inputs_embeds=embeds, num_loops=num_loops)
        hidden = hidden * self.config.backbone.output_multiplier
        mask = torch.ones_like(is_audio) if loss_mask is None else loss_mask
        # Predictions at position t are for position t+1.
        h, targets_text, next_is_audio, keep = hidden[:, :-1], text_ids[:, 1:], is_audio[:, 1:], mask[:, 1:]
        out = {"loops": loops}
        text_keep = keep & ~next_is_audio
        text_loss = hidden.new_zeros(())
        if text_keep.any():
            logits = self.lm.lm_head(h[text_keep]).float()
            text_loss = F.cross_entropy(logits, targets_text[text_keep])
        out["text_loss"] = text_loss
        audio_loss = depth_loss = hidden.new_zeros(())
        audio_keep = keep & next_is_audio
        if audio_codes is not None and audio_keep.any():
            states = h[audio_keep]
            codes = audio_codes[:, :, 1:].permute(0, 2, 1)[audio_keep]  # (N, Q)
            logits0 = self.audio_head(self.audio_norm(states)).float()
            first = torch.where(codes[:, 0] >= 0, codes[:, 0], torch.full_like(codes[:, 0], self.end_of_audio))
            audio_loss = F.cross_entropy(logits0, first)
            valid = (codes >= 0).all(dim=1)
            if valid.any():
                depth_logits = self.depth(states[valid], codes[valid]).float()  # (N, Q-1, V)
                depth_loss = F.cross_entropy(
                    depth_logits.reshape(-1, depth_logits.shape[-1]), codes[valid][:, 1:].reshape(-1)
                )
        out["audio_loss"], out["depth_loss"] = audio_loss, depth_loss
        out["loss"] = (
            text_loss + self.config.audio_loss_weight * audio_loss + self.config.depth_loss_weight * depth_loss
        )
        return out

    # --------------------------------------------------------------------------------- prompting --
    def clone_prompt(
        self, reference_text: torch.Tensor, reference_codes: torch.Tensor, target_text: torch.Tensor
    ) -> dict:
        """
        The in-context cloning layout for one example: ``[reference transcript][reference audio][target text]``,
        with the acoustic delay applied to the reference audio and the loss masked over the whole prompt.
        Returns ``text_ids (1, T)``, ``audio_codes (1, Q, T)`` and ``loss_mask (1, T)``; append the target audio
        (delayed) to train, or generate from here.
        """
        delayed = apply_delay(reference_codes[None], self.config.delay, pad=-1)[0]  # (Q, Tr + delay)
        n_ref_audio = delayed.shape[1]
        text = torch.cat([reference_text, torch.full((n_ref_audio,), AUDIO_SLOT, dtype=torch.long), target_text])
        codes = torch.full((self.config.num_codebooks, text.shape[0]), -1, dtype=torch.long)
        start = reference_text.shape[0]
        codes[:, start : start + n_ref_audio] = delayed
        mask = torch.zeros(text.shape[0], dtype=torch.bool)
        return {"text_ids": text[None], "audio_codes": codes[None], "loss_mask": mask[None]}

    @torch.no_grad()
    def generate_audio(
        self,
        text_ids: torch.Tensor,
        audio_codes: torch.Tensor | None,
        max_frames: int,
        temperature: float = 0.0,
        num_loops: int | None = None,
    ) -> torch.Tensor:
        """
        Continue a prompt with up to ``max_frames`` audio frames (greedy or sampled), frame by frame: codebook 0
        from the trunk, the rest from the depth transformer; stops at end-of-audio. Returns ``(B, Q, frames)``
        codes with the delay removed. Recomputes the prompt each step (simple, not fast).
        """
        self.eval()
        b = text_ids.shape[0]
        q = self.config.num_codebooks
        text = text_ids
        codes = audio_codes if audio_codes is not None else torch.full((b, q, text.shape[1]), -1, dtype=torch.long)
        produced = []
        for _ in range(max_frames):
            embeds = self.embed(text, codes)
            hidden, _, _ = self.lm.model(inputs_embeds=embeds, num_loops=num_loops)
            state = hidden[:, -1] * self.config.backbone.output_multiplier
            logits0 = self.audio_head(self.audio_norm(state))
            first = (
                logits0.argmax(-1)
                if temperature <= 0
                else torch.multinomial((logits0 / temperature).softmax(-1), 1)[:, 0]
            )
            if bool((first == self.end_of_audio).all()):
                break
            first = first.clamp(max=self.config.codebook_size - 1)
            frame = self.depth.generate(state, first, temperature)  # (B, Q) undelayed frame
            produced.append(frame)
            # In the sequence the frame's acoustics land `delay` slots later: write codebook 0 now and the
            # rest into the slot `delay` ahead (created as empty until then).
            new_slot = torch.full((b, q, 1), -1, dtype=torch.long)
            new_slot[:, 0, 0] = frame[:, 0]
            codes = torch.cat([codes, new_slot], dim=2)
            text = torch.cat([text, torch.full((b, 1), AUDIO_SLOT, dtype=torch.long)], dim=1)
            back = len(produced) - 1 - self.config.delay
            if back >= 0:
                codes[:, 1:, -1] = produced[back][:, 1:]
        if not produced:
            return torch.zeros(b, q, 0, dtype=torch.long)
        return torch.stack(produced, dim=2)
