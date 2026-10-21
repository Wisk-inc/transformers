# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""The hardening of TriStream-2: the refiner, the ear, the humanizer, the sync and routing fixes, the fail-safe
loop and the room model, all at toy scale on a CPU."""

import math

import torch

from helix_lm.audio import AudioCodec, CodecConfig, SpeakerEncoder
from helix_lm.audio.data import synthetic_voice
from helix_lm.audio.refiner import (
    HarmonicNoiseRefiner,
    RefinerConfig,
    harmonic_oscillator,
    refiner_losses,
    render_sketch,
    transient_transplant,
)
from helix_lm.audio.singing import estimate_f0
from helix_lm.music.analysis import analyze, synthetic_song
from helix_lm.music.artifacts import (
    artifact_report,
    consonant_sharpness,
    frame_rate_buzz,
    robotic_score,
    summarize_report,
    sync_lag,
)
from helix_lm.music.conductor import plan_vocals, render_controls
from helix_lm.music.conversion import Converter
from helix_lm.music.effects import apply_room, build_rir, estimate_room, lowpass, multiband_reverb_time, reverb
from helix_lm.music.failsafe import Gates, synthesize_with_gates
from helix_lm.music.humanize import HumanStyle, decimate, humanize
from helix_lm.music.tristream import (
    TriStreamConfig,
    TriStreamForSinging,
    alignment_bias,
    event_features,
    noise_features,
    shift_ahead,
    source_features,
    sway_schedule,
)

SR = 16000
NOTES = [(60, 0.0, 0.5), (62, 0.5, 1.0), (64, 1.0, 1.8), (67, 2.1, 3.0), (65, 3.0, 3.6), (64, 3.6, 4.5)]


def machine_curve(notes, fine_rate=100.0):
    end = int(notes[-1][2] * fine_rate) + 50
    f0 = torch.zeros(end)
    for midi, a, z in notes:
        seg = slice(int(a * fine_rate), int(z * fine_rate))
        tt = torch.arange(int(z * fine_rate) - int(a * fine_rate)) / fine_rate
        f0[seg] = 440 * 2 ** ((midi - 69) / 12) * 2 ** (0.4 / 12 * torch.sin(2 * math.pi * 6 * tt))
    return f0


def test_refiner_is_phase_true_and_trains():
    torch.manual_seed(0)
    cfg = RefinerConfig(
        sample_rate=SR, frame_dim=32, frame_rate=25, fine_rate=100, harmonics=16, noise_bands=12, dim=16,
        residual_dim=8, residual_dilations=(1, 3),
    )  # fmt: skip
    refiner = HarmonicNoiseRefiner(cfg)
    voice = synthetic_voice(0.64, SR)[None]
    tf = voice.shape[-1] // cfg.hop
    voice = voice[:, : tf * cfg.hop]
    frames = torch.randn(1, tf // cfg.upsample, 32)
    f0 = estimate_f0(voice, SR, frame_rate=100)[0][None][:, :tf]
    out = refiner(voice[:, None], frames, f0)
    assert out["wave"].shape == (1, 1, voice.shape[-1]) and torch.isfinite(out["wave"]).all()
    first = refiner_losses(out, voice, SR)["total"]
    opt = torch.optim.Adam(refiner.parameters(), lr=3e-3)
    for _ in range(6):
        losses = refiner_losses(refiner(voice[:, None], frames, f0), voice, SR)
        opt.zero_grad()
        losses["total"].backward()
        opt.step()
    assert losses["total"] < first * 1.05 and all(torch.isfinite(v) for v in losses.values())
    # the oscillator puts the pitch exactly where the curve says, with no aliasing above Nyquist
    curve = torch.full((1, 100), 220.0)
    amps = torch.zeros(1, 100, 4)
    amps[..., 0] = 0.3
    tone = harmonic_oscillator(curve, amps, SR, cfg.hop)
    est = estimate_f0(tone, SR, frame_rate=25)[0]
    assert abs(float(est[est > 0].median()) - 220) < 4
    high = harmonic_oscillator(torch.full((1, 50), 6000.0), torch.ones(1, 50, 4) * 0.2, SR, cfg.hop)
    assert high.abs().max() <= 0.21, "harmonics above Nyquist are silenced"
    # the physical fallback renders something finite, bounded and voiced
    sketch = render_sketch(f0, torch.randn(1, frames.shape[1], 32) * 0.3, torch.randn(1, frames.shape[1], 32) - 2, SR)
    assert sketch.shape == (1, voice.shape[-1]) and float(sketch.abs().max()) <= 0.5
    est = estimate_f0(sketch, SR, frame_rate=25)[0]
    assert (est > 0).float().mean() > 0.5
    # the transplant keeps the source's unvoiced high band and leaves voiced spans alone
    f025 = estimate_f0(voice, SR, 25)[0]
    same = transient_transplant(voice[0], voice[0] * 0.5, f025, SR)
    assert same.shape == voice[0].shape and torch.isfinite(same).all()


def test_the_ear_hears_buzz_smear_lag_and_robots():
    voice = synthetic_voice(0.64, SR)
    f025 = estimate_f0(voice[None], SR, 25)[0]
    clean = frame_rate_buzz(voice, SR)["buzz_db"]
    t = torch.arange(voice.numel()) / SR
    buzzed = frame_rate_buzz(voice * (1 + 0.5 * torch.sin(2 * math.pi * 25 * t)), SR)
    assert buzzed["buzz_db"] > clean + 20 and not buzzed["ok"]
    smeared = consonant_sharpness(lowpass(voice, 1500, SR), voice, SR)
    assert smeared["ratio"] < consonant_sharpness(voice, voice, SR)["ratio"] and not smeared["ok"]
    human = humanize(NOTES, HumanStyle(), 100.0, seed=1)["f0_hz"]
    amps = torch.zeros(1, human.numel(), 4)
    amps[..., 0], amps[..., 1] = 0.3, 0.1
    wave = harmonic_oscillator(human[None], amps, SR, SR // 100)[0]
    delayed = torch.cat([torch.zeros(int(0.08 * SR)), wave])[: wave.numel()]
    assert abs(sync_lag(human[::4], wave, SR, 25)["lag_ms"]) <= 20
    lag = sync_lag(human[::4], delayed, SR, 25)
    assert 60 <= lag["lag_ms"] <= 100 and not lag["ok"]
    machine, sung = robotic_score(machine_curve(NOTES), 100), robotic_score(human, 100)
    assert machine["score"] > 0.5 > sung["score"], (machine, sung)
    assert machine["parts"]["jumps"] > 0.5 and sung["parts"]["jumps"] == 0.0
    assert robotic_score(machine_curve(NOTES)[::4], 25)["score"] > robotic_score(human[::4], 25)["score"]
    report = artifact_report(voice, SR, ref=voice, f0_hz=f025, frame_rate=25)
    assert report["ok"], summarize_report(report)


def test_humanizer_glides_delays_vibrato_and_breathes():
    out = humanize(NOTES, HumanStyle.named("soul"), 100.0, seed=2)
    f0, energy, breath = out["f0_hz"], out["energy"], out["breath"]
    assert torch.isfinite(f0).all() and torch.isfinite(breath).all() and 0 <= energy.min() <= energy.max() <= 1
    cents = torch.where(f0 > 0, 1200 * torch.log2(f0.clamp(min=1) / 440), torch.zeros_like(f0))
    step = (cents[1:] - cents[:-1]).abs() * ((f0[1:] > 0) & (f0[:-1] > 0)).float()
    assert step.max() < 100, "no instantaneous jump larger than a semitone: notes glide"
    # vibrato is absent before its onset on a long note and present later (scoop and wander switched off)
    held = humanize([(67, 0.0, 1.5)], HumanStyle(scoop_cents=0, wander_cents=0, jitter_cents=0), 100.0, seed=1)
    held_cents = 1200 * torch.log2(held["f0_hz"][:150].clamp(min=1) / 440)
    assert held_cents[:25].std() < 1.0 < held_cents[50:100].std()
    assert breath.max() > 0.5 and breath[: int(0.05 * 100)].max() >= 0
    same = humanize(NOTES, HumanStyle.named("soul"), 100.0, seed=2)["f0_hz"]
    assert torch.equal(same, f0), "the same seed is the same singer"
    assert not torch.equal(humanize(NOTES, HumanStyle.named("pop"), 100.0, seed=3)["f0_hz"], f0)
    assert decimate(f0, 100.0, 25.0).numel() == f0.numel() // 4
    song = analyze(synthetic_song(10.0, SR, bpm=120, key=7), SR)
    controls = render_controls(plan_vocals(song, singer_range=(55, 79)), song)
    assert {"f0_fine", "energy", "breath"} <= set(controls)
    assert robotic_score(controls["f0_fine"], 100)["score"] < 0.5


def test_tristream_sync_and_routing_mechanisms():
    torch.manual_seed(0)
    cfg = TriStreamConfig.named("ts-tiny")
    model = TriStreamForSinging(cfg)
    b, t, length = 2, 32, 7
    f0 = torch.where(torch.rand(b, t) > 0.2, 220.0 * (torch.rand(b, t) + 0.5), torch.zeros(b, t))
    f0_fine = f0.repeat_interleave(4, dim=1) * (1 + 0.01 * torch.randn(b, t * 4))
    frames = torch.randn(b, t, cfg.frame_dim)
    source = source_features(f0, torch.rand(b, t), f0_fine=f0_fine)
    assert source.shape == (b, t, cfg.source_dim) and source[..., 7:].abs().sum() > 0
    batch = {
        "source": source, "noise": noise_features(frames), "frames": frames,
        "phonemes": torch.randint(0, cfg.num_phonemes, (b, length)), "speaker": torch.randn(b, cfg.speaker_dim),
        "singer": torch.randint(0, cfg.num_singers, (b,)), "accent": torch.randint(0, 5, (b,)),
        "sliders": torch.rand(b, t, 10), "durations": torch.randint(1, 6, (b, length)), "f0_hz": f0,
        "events": torch.rand(b, t, 4),
    }  # fmt: skip
    losses = model(batch)
    assert "formant_invariance" in losses and all(torch.isfinite(v) for v in losses.values())
    losses["total"].backward()
    assert model.ahead.weight.grad.abs().sum() > 0 and model.event_in.weight.grad.abs().sum() > 0
    model.eval()
    mask = torch.zeros(b, t, dtype=torch.bool)
    mask[:, 8:16] = True
    g = torch.Generator().manual_seed(1)
    out = model.sample(batch, steps=6, mask=mask, guidance=2.0, adaptive=True, generator=g)
    assert out.shape == frames.shape and torch.equal(out[:, :8], frames[:, :8]) and 1 <= model.last_steps <= 6
    g = torch.Generator().manual_seed(1)
    again = model.sample(batch, steps=6, mask=mask, guidance=2.0, adaptive=True, generator=g)
    assert torch.allclose(out, again), "a generator makes sampling reproducible"
    assert model.sample({k: v for k, v in batch.items() if k != "durations"}, steps=2).shape == frames.shape
    envelope, noise_bands = model.sketch_bands(batch)
    assert envelope.shape == (b, t, 32) and noise_bands.shape == (b, t, 32)
    # the pieces themselves
    bias = alignment_bias(torch.tensor([[2, 3, 1]]), 6, 1.0)
    assert bias.shape == (1, 6, 3) and bias[0, 1, 0] > bias[0, 5, 0], "frames near a phoneme's span score higher"
    x = torch.arange(10.0)[None, :, None]
    assert shift_ahead(x, 3)[0, :, 0].tolist() == [3, 4, 5, 6, 7, 8, 9, 9, 9, 9]
    times = sway_schedule(8, -1.0)
    assert times[0] == 0 and abs(float(times[-1]) - 1) < 1e-6 and (times[1:] > times[:-1]).all()
    assert float(times[4]) < 0.5, "sway spends more steps near t = 0"
    song = analyze(synthetic_song(8.0, SR, bpm=120, key=7), SR)
    events = event_features(song, 200, 25.0)
    assert events.shape == (1, 200, 4) and events[0, :, 1].sum() >= 1 and events[0, :, 2].sum() >= 1


def test_failsafe_loop_escalates_and_falls_back():
    torch.manual_seed(0)
    sr = 8000
    ccfg = CodecConfig(
        sample_rate=sr, dim=32, channels=8, strides=(2, 4, 5, 8), num_codebooks=4, codebook_size=64, code_dim=4
    )
    codec = AudioCodec(ccfg).eval()
    cfg = TriStreamConfig.named("ts-tiny", frame_dim=32, speaker_dim=16, sample_rate=sr)
    model = TriStreamForSinging(cfg).eval()
    speaker = SpeakerEncoder(frame_dim=32, embedding_dim=16).eval()
    voice, reference = synthetic_voice(0.8, sr), synthetic_voice(0.6, sr, f0=196.0)
    frames = codec.listen(voice[None])
    t = frames.shape[1]
    f0 = estimate_f0(voice[None], sr, frame_rate=25)[0][:t]
    f0 = torch.nn.functional.pad(f0, (0, t - f0.numel()))
    batch = {
        "source": source_features(f0[None], frames.pow(2).mean(-1)), "noise": noise_features(frames), "frames": frames,
        "phonemes": torch.full((1, 1), -1, dtype=torch.long), "speaker": speaker(frames),
    }  # fmt: skip
    gates = Gates(max_rounds=2, steps=2, min_similarity=None, max_f0_rmse_cents=1e9)
    wave, report = synthesize_with_gates(model, codec, batch, sr, f0, source_wave=voice, gates=gates)
    assert wave.shape == (voice.numel(),) and torch.isfinite(wave).all() and wave.abs().max() <= 1.0
    assert 1 <= len(report.rounds) <= 2 and report.rounds[-1]["steps"] >= 2 and "round" in report.summary()
    # gates that cannot pass drive every fallback, and the result is still finite and bounded
    rcfg = RefinerConfig(
        sample_rate=sr,
        frame_dim=32,
        frame_rate=25,
        fine_rate=100,
        harmonics=8,
        noise_bands=8,
        dim=8,
        residual_dim=4,
        residual_dilations=(1,),
    )
    refiner = HarmonicNoiseRefiner(rcfg).eval()
    strict = Gates(max_rounds=2, steps=2, min_similarity=1.01, max_f0_rmse_cents=0.0)
    wave2, report2 = synthesize_with_gates(
        model, codec, batch, sr, f0, refiner=refiner, speaker_encoder=speaker, reference_wave=reference,
        source_wave=voice, gates=strict,
    )  # fmt: skip
    assert torch.isfinite(wave2).all() and report2.rounds[1]["guidance"] > report2.rounds[0]["guidance"]
    assert any("sketch" in a or "transplant" in a for a in report2.applied), report2.applied
    assert any("source vocal" in a for a in report2.applied), "nothing that fails its gates ships as it is"
    # through the converter
    converter = Converter(model, codec, speaker, refiner=refiner, gates=gates)
    out, how = converter.convert(voice, f0, speaker(frames), sr, reference=reference)
    assert out.shape[-1] == voice.numel() and "fail-safe" in how and converter.last_report is not None


def test_room_model_matches_bands():
    dry = torch.cat([synthetic_voice(0.5, SR), torch.zeros(int(0.8 * SR))])  # a stop, so a decay can be measured
    wet = reverb(dry, SR, decay_seconds=1.2, mix=0.4)
    assert sum(multiband_reverb_time(wet, SR)) > sum(multiband_reverb_time(dry, SR))
    room = estimate_room(wet, SR)
    assert len(room["decays"]) == 4 and 0.01 <= room["pre_delay"] <= 0.04 and 0 <= room["wet"] <= 0.5
    ir = build_rir(room["decays"], SR, room["pre_delay"])
    assert ir[0] == 1.0 and ir.numel() > int(0.05 * SR)
    roomed = apply_room(dry, room, SR)
    assert roomed.shape == dry.shape and (roomed - dry).abs().mean() > 0
    assert CodecConfig.at_frame_rate(50).frame_rate == 50.0 and CodecConfig.at_frame_rate(75).frame_rate == 75.0
