# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The audio stack at toy scale: shapes, causality, quantizer identities, a codec overfit, score conditioning,
the omni wrapper's losses, delay pattern and generation, and the file readers."""

import torch

from helix_lm.anansi import AnansiConfig
from helix_lm.audio import (
    AudioCodec,
    CodecConfig,
    OmniConfig,
    OmniForCausalLM,
    SingingConditioner,
    SpeakerEncoder,
    WaveDecoder,
    WaveEncoder,
    apply_delay,
    codec_losses,
    hz_to_bins,
    undo_delay,
)
from helix_lm.audio.codec import MultiPeriodDiscriminator, discriminator_loss
from helix_lm.audio.data import (
    describe_audio_sources,
    encode_files,
    read_codes,
    read_wave,
    synthetic_voice,
    write_wave,
)
from helix_lm.audio.frontend import receptive_field_is_causal
from helix_lm.audio.omni import AUDIO_SLOT
from helix_lm.audio.singing import bins_to_hz, estimate_f0, frames_from_notes
from helix_lm.audio.speaker import AAMSoftmax, speaker_similarity

SR = 8000
TINY = CodecConfig(
    sample_rate=SR, dim=32, channels=8, strides=(2, 4, 5, 8), num_codebooks=4, codebook_size=64, code_dim=4
)


def test_frontend_shapes_and_causality():
    encoder, decoder = (
        WaveEncoder(TINY.dim, TINY.channels, TINY.strides),
        WaveDecoder(TINY.dim, TINY.channels, TINY.strides),
    )
    wave = torch.randn(2, 1, 3200)
    frames = encoder(wave)
    assert frames.shape == (2, TINY.dim, 3200 // encoder.hop) and encoder.hop == 320
    out = decoder(frames)
    assert out.shape == (2, 1, 3200) and out.abs().max() <= 1.0
    assert receptive_field_is_causal(encoder, length=3200)


def test_codec_roundtrip_and_quantizer_identities():
    torch.manual_seed(0)
    codec = AudioCodec(TINY).eval()
    wave = synthetic_voice(0.4, SR)[None]
    codes = codec.encode(wave)
    assert codes.shape == (1, TINY.num_codebooks, wave.shape[-1] // codec.hop)
    assert codes.min() >= 0 and codes.max() < TINY.codebook_size
    recon = codec.decode(codes)
    assert recon.shape == (1, 1, codes.shape[-1] * codec.hop)
    frames = codec.encoder(wave)
    quantized, codes2, _, _ = codec.quantizer(frames)
    assert torch.equal(codes, codes2) and torch.allclose(codec.quantizer.decode(codes), quantized, atol=1e-5)
    fewer = codec.encode(wave, num_codebooks=2)
    assert fewer.shape[1] == 2 and torch.equal(fewer[:, 0], codes[:, 0])


def test_codec_overfits_a_tone_and_uses_its_codebook():
    torch.manual_seed(0)
    codec = AudioCodec(TINY)
    disc = MultiPeriodDiscriminator((2, 3))
    wave = synthetic_voice(0.3, SR)[None]
    opt = torch.optim.Adam(codec.parameters(), lr=3e-3)
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3)
    first = last = None
    for step in range(40):
        out = codec(wave)
        terms = codec_losses(out, wave, TINY, discriminator=disc if step > 20 else None)
        opt.zero_grad()
        terms["total"].backward()
        opt.step()
        if step > 20:
            d_loss = discriminator_loss(disc, out["recon"], wave)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()
        mel = terms["mel"].item()
        first = mel if first is None else first
        last = mel
    assert last < 0.8 * first, (first, last)
    assert codec.quantizer.semantic.usage() > 0.05
    assert all(torch.isfinite(v) for v in terms.values())


def test_speaker_encoder_embeds_and_trains():
    torch.manual_seed(0)
    encoder = SpeakerEncoder(frame_dim=16, channels=16, embedding_dim=8)
    head = AAMSoftmax(8, 2)
    frames = torch.randn(4, 20, 16)
    speakers = torch.tensor([0, 0, 1, 1])
    emb = encoder(frames)
    assert emb.shape == (4, 8) and torch.allclose(emb.norm(dim=-1), torch.ones(4), atol=1e-4)
    assert speaker_similarity(emb[0], emb[0]).item() > 0.99
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-2)
    losses = []
    for _ in range(15):
        loss = head(encoder(frames), speakers)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_singing_streams_and_pitch_tracking():
    f0 = torch.tensor([0.0, 55.0, 220.0, 440.0, 1000.0])
    bins = hz_to_bins(f0)
    assert bins[0] == 256 and bins[1] < bins[2] < bins[3] < bins[4]
    back = bins_to_hz(bins)
    assert back[0] == 0 and torch.allclose(back[1:], f0[1:], rtol=0.02)
    notes, phonemes = frames_from_notes([(60, 0.5, 3), (-1, 0.25, 0), (62, 0.3, 4)], frame_rate=25)
    assert notes.shape == (26,) and notes[0] == 60 and notes[13] == 128 and phonemes[-1] == 4
    cond = SingingConditioner(hidden_size=16).eval()
    a = cond(notes[None], phonemes[None])
    b = cond(notes[None], phonemes[None], f0_hz=torch.full((1, 26), 330.0))
    assert a.shape == (1, 26, 16) and not torch.allclose(a, b), "the pitch stream changes the conditioning"
    tone = synthetic_voice(0.5, SR, f0=220.0, vibrato_hz=0.0)[None]
    tracked = estimate_f0(tone, SR, frame_rate=25)
    voiced = tracked[tracked > 0]
    assert voiced.numel() > 5 and (voiced.median() - 220).abs() < 11


def test_delay_pattern_is_invertible():
    codes = torch.randint(0, 10, (2, 4, 7))
    delayed = apply_delay(codes, delay=2, pad=-1)
    assert (
        delayed.shape == (2, 4, 9) and torch.equal(delayed[:, 0, :7], codes[:, 0]) and (delayed[:, 1:, :2] == -1).all()
    )
    assert torch.equal(undo_delay(delayed, 2), codes)


def _omni():
    backbone = AnansiConfig.named("a-20m", vocab_size=64, ngram_memory_size=0, mtp_depth=0, loss_chunk_tokens=0)
    return OmniForCausalLM(
        OmniConfig(backbone=backbone, num_codebooks=4, codebook_size=16, frame_dim=8, delay=1, depth_layers=1)
    )


def test_omni_losses_prompt_layout_and_generation():
    torch.manual_seed(0)
    model = _omni()
    text = torch.tensor([[1, 2, 3, AUDIO_SLOT, AUDIO_SLOT, AUDIO_SLOT, AUDIO_SLOT, 4, 5]])
    raw = torch.randint(0, 16, (1, 4, 3))
    codes = torch.full((1, 4, 9), -1)
    codes[:, :, 3:7] = apply_delay(raw, 1, pad=-1)
    frames = torch.randn(1, 9, 8)
    model.train()
    out = model(text, codes, audio_frames=frames)
    assert all(torch.isfinite(out[k]) for k in ("text_loss", "audio_loss", "depth_loss", "loss")) and out["loss"] > 0
    out["loss"].backward()
    prompt = model.clone_prompt(torch.tensor([7, 8]), raw[0], torch.tensor([9, 10]))
    assert prompt["text_ids"].shape == (1, 2 + 4 + 2) and not prompt["loss_mask"].any()
    assert (prompt["text_ids"][0, 2:6] == AUDIO_SLOT).all() and torch.equal(
        prompt["audio_codes"][0, 0, 2:5], raw[0, 0]
    )
    model.eval()
    generated = model.generate_audio(prompt["text_ids"], prompt["audio_codes"], max_frames=3)
    assert generated.shape[0] == 1 and generated.shape[1] == 4 and generated.shape[2] <= 3
    if generated.shape[2]:
        assert generated.min() >= 0 and generated.max() < 16


def test_wave_files_and_code_shards(tmp_path):
    wave = synthetic_voice(0.2, SR)
    path = write_wave(tmp_path / "tone.wav", wave, SR)
    back = read_wave(path, SR)
    assert back.shape == wave.shape and (back - wave).abs().max() < 1e-3
    assert read_wave(path, 16000).shape[0] == 2 * wave.shape[0]
    codec = AudioCodec(TINY).eval()
    manifest = encode_files(codec, [path], tmp_path / "codes", SR)
    assert manifest["files"][0]["frames"] == wave.shape[0] // codec.hop
    codes = read_codes(tmp_path / "codes" / "tone.codes")
    assert codes.shape == (TINY.num_codebooks, manifest["files"][0]["frames"])
    table = describe_audio_sources()
    assert "emilia-yodas" in table and "gtsinger" in table and "no" in describe_audio_sources(commercial_only=False)
    assert "gtsinger" not in describe_audio_sources(commercial_only=True)
