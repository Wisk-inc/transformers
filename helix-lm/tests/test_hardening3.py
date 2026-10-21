# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Round three: melismas against the alignment prior, codebook starvation on the high register, the noise-floor
shortcut, register-break pops, and index staleness at a drop."""

import torch

from helix_lm.audio import AudioCodec, CodecConfig
from helix_lm.audio.codec import codebook_usage_by_register
from helix_lm.audio.data import synthetic_voice
from helix_lm.audio.refiner import harmonic_oscillator
from helix_lm.music.analysis import analyze, synthetic_song
from helix_lm.music.longform import plan_chunks, transition_frames
from helix_lm.music.register import register_break_report, tract_ratio_curve, warp_envelope
from helix_lm.music.regularize import augment, random_eq, synthetic_room
from helix_lm.music.tristream import (
    TriStreamConfig,
    TriStreamForSinging,
    alignment_bias,
    noise_features,
    source_features,
    sustain_profile,
)

SR = 16000


def test_alignment_prior_widens_on_held_notes():
    b, t = 2, 60
    f0 = torch.full((b, t), 220.0)
    f0[1] = 220.0 * (1 + 0.06 * torch.randn(t))  # a jittery, moving line against a held note
    source = source_features(f0, torch.rand(b, t))
    profile = sustain_profile(source)
    assert profile[0].max() > 0.9 and profile[1].max() < 0.3, "held notes read as sustained; motion does not"
    sigma = 4.0 * (1 + 3.0 * profile)
    durations = torch.full((b, 3), 20)
    bias = alignment_bias(durations, t, sigma)
    far = 25  # a frame 15 frames from phoneme 0's centre (frame 10)
    assert bias[0, far, 0] > bias[1, far, 0], "the held note's prior is wider: a melisma may span its vowel"
    assert torch.allclose(alignment_bias(durations, t, 4.0), alignment_bias(durations, t, torch.full((b, t), 4.0)))
    cfg = TriStreamConfig.named("ts-tiny")
    model = TriStreamForSinging(cfg).eval()
    frames = torch.randn(b, t, cfg.frame_dim)
    batch = {
        "source": source, "noise": noise_features(frames), "frames": frames,
        "speaker": torch.randn(b, cfg.speaker_dim), "phonemes": torch.randint(0, cfg.num_phonemes, (b, 3)),
        "durations": durations,
    }  # fmt: skip
    assert torch.isfinite(model.sample(batch, steps=2)).all()


def test_codebook_audit_sees_starvation():
    torch.manual_seed(0)
    codec = AudioCodec(
        CodecConfig(
            sample_rate=SR, dim=32, channels=8, strides=(2, 4, 5, 8), num_codebooks=4, codebook_size=64, code_dim=4
        )
    ).eval()
    high = torch.stack([synthetic_voice(0.4, SR, f0=f) for f in (520.0, 600.0, 700.0, 800.0)])
    low = torch.stack([synthetic_voice(0.4, SR, f0=f) for f in (110.0, 150.0, 200.0, 260.0)])
    audit = codebook_usage_by_register(codec, high, low)
    assert len(audit["high"]) == 4 and len(audit["ratio"]) == 4 and isinstance(audit["starved"], bool)
    assert all(0 <= u <= 1 for u in audit["high"] + audit["low"])


def test_augmentation_breaks_the_clean_equals_sung_shortcut():
    voice = synthetic_voice(1.0, SR)
    g = torch.Generator().manual_seed(3)
    eq, info = random_eq(voice, SR, g)
    assert eq.shape == voice.shape and torch.isfinite(eq).all() and -6 <= info["tilt_db_per_oct"] <= 6
    roomed, room = synthetic_room(voice, SR, g)
    assert roomed.shape == voice.shape and 0.2 <= room["t60"] <= 2.0
    seen = set()
    for seed in range(16):
        out, applied = augment(voice, SR, seed=seed)
        assert out.shape == voice.shape and torch.isfinite(out).all()
        seen |= set(applied)
    assert {"room_t60", "eq_tilt_db_per_oct"} <= seen, seen
    # the model trains with a roomy input and a dry target
    cfg = TriStreamConfig.named("ts-tiny")
    model = TriStreamForSinging(cfg)
    b, t = 2, 16
    f0 = torch.full((b, t), 220.0)
    clean = torch.randn(b, t, cfg.frame_dim)
    batch = {
        "source": source_features(f0, torch.rand(b, t)), "noise": noise_features(clean + 0.3), "frames": clean + 0.3,
        "target_frames": clean, "phonemes": torch.full((b, 1), -1, dtype=torch.long),
        "speaker": torch.randn(b, cfg.speaker_dim), "f0_hz": f0,
    }  # fmt: skip
    losses = model(batch)
    assert "formant_smoothness" in losses and all(torch.isfinite(v) for v in losses.values())


def test_register_moves_are_smooth_and_pops_are_caught():
    f0 = torch.cat([torch.full((50,), 220.0), torch.zeros(5), torch.linspace(220, 700, 60), torch.full((40,), 700.0)])
    curve = tract_ratio_curve(f0, (110.0, 330.0), 25.0)
    assert float(curve.min()) >= 1.0 and float(curve.max()) <= 1.2 + 1e-6
    assert float((curve[1:] - curve[:-1]).abs().max()) < 0.02, "no step in the formant ratio: no wall"
    assert float(curve[:50].max()) < 1.001, "nothing moves inside the voice's own range"
    env = torch.randn(1, f0.numel(), 32)
    warped = warp_envelope(env, curve[None])
    assert warped.shape == env.shape and torch.allclose(warped[0, :40], env[0, :40], atol=1e-5)
    wave = harmonic_oscillator(
        f0[None].repeat_interleave(4, 1), torch.ones(1, f0.numel() * 4, 4) * 0.2, SR, SR // 100
    )[0]
    assert register_break_report(wave, f0, SR, 330.0)["ok"]
    crossing = int(((f0 > 330).int()[1:] - (f0 > 330).int()[:-1]).nonzero()[0]) + 1
    popped = wave.clone()
    popped[int(crossing / 25 * SR)] += 0.9
    report = register_break_report(popped, f0, SR, 330.0)
    assert report["pops"] >= 1 and not report["ok"]


def test_transitions_get_look_ahead_and_chunk_edges():
    song = analyze(synthetic_song(16.0, SR, bpm=120, key=7), SR)
    anchors = transition_frames(song, 25.0)
    assert anchors.numel() >= 1 and (anchors > 0).all()
    chunks = plan_chunks(400, 150, 20, anchors=anchors, snap_frames=30)
    edges = {end for _, end in chunks[:-1]}
    assert any(int(a) in edges for a in anchors.tolist()), (anchors, chunks)
    # the two-block look-ahead switches on only where a transition is coming
    cfg = TriStreamConfig.named("ts-tiny")
    model = TriStreamForSinging(cfg).eval()
    b, t = 1, 40
    f0 = torch.full((b, t), 220.0)
    frames = torch.randn(b, t, cfg.frame_dim)
    source = source_features(f0, torch.rand(b, t))
    noise = noise_features(frames)
    quiet = torch.zeros(b, t, 4)
    drop = quiet.clone()
    drop[0, 30, 2] = 1.0
    with torch.no_grad():
        lyric, pad = model.encode_lyrics(torch.full((b, 1), -1, dtype=torch.long))
        s_enc, n_enc = model.encode_source(source), model.encode_noise(noise)
        filt = model.encode_filter(
            {"source": source, "frames": frames, "speaker": torch.randn(b, cfg.speaker_dim)}, lyric, pad, None
        )
        h_quiet = model.fuse_streams(s_enc, filt, n_enc, lyric, pad, events=quiet)
        h_drop = model.fuse_streams(s_enc, filt, n_enc, lyric, pad, events=drop)
    diff = (h_quiet - h_drop).abs().mean(-1)[0]
    assert diff[:10].max() < diff[22:31].max(), "the frames just before the drop are the ones that change"
